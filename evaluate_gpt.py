import json
import argparse
import random
import time
import json
import os
import requests
import multiprocessing
from functools import partial

from build_dataset import build_dataset, calculate_metrics
from build_prompt_gpt import create_prompt_gpt, parse_score_gpt

from build_icl import build_icl


def build_params_gpt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-2024-04-09", 
                 "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"),
        default=None,
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=("vanilla", "cot", "icl"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "mt-bench",
                 "halu-eval-summary", "halu-eval-qa", "halu-eval-dialogue", "salad-bench", "toxic-chat",
                 "llmbar-neighbor", "llmbar-natural", "llmbar-gptinst", "llmbar-gptout", "llmbar-manual"),
        default=None,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=None,
        help="The maximum number of new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--logit-file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--pool-number",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--multi-process",
        type=str,
        default="False",
    )
    args = parser.parse_args()
    return args


def request_gpt(prompt, model, temperature, max_new_tokens):

    url = "https://api.ai-gaochao.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-agEcX3Su78Bu09c2F49978C6Ba424977B936C8710fAb42E0",
    }
    max_tries = 5
    res = ''
    response = None
    sys_info = {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."}
    for i in range(max_tries):
        try:
            messages = [sys_info, {"role": "user", "content": prompt}]
            messages = [{"role": "user", "content": prompt}]
            data = {"model": model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_new_tokens}
            response = requests.post(
                url, headers=headers, data=json.dumps(data))
            response = response.json()
            res = response['choices'][0]['message']['content'].strip()
            break
        except Exception as e:
            print("Exception! The response is " + str(response))
            time.sleep(5)
            continue
    return res


def gpt_scoring(prompt, model, temperature, max_new_tokens):

    prediction = request_gpt(
        prompt, model, temperature=temperature, max_new_tokens=max_new_tokens)

    counter.value += 1
    print(f"gpt_scoring {counter.value} finished.")

    return prediction


def init(c):
    global counter
    counter = c


if __name__ == "__main__":
    args = build_params_gpt()
    random.seed(42)

    if "prometheus" in args.data_type:
        args.prompt_type = "cot"
    
    # 根据是否使用COT自动设置最大长度，避免浪费API和提高速度
    if args.max_new_token is None:
        if args.prompt_type == "cot":
            args.max_new_token = 1024
        else:
            args.max_new_token = 16

    dataset = build_dataset(args.data_type, args.data_path)

    instruction = create_prompt_gpt(args.data_type, args.prompt_type)

    prompts = []
    answers = []
    if args.prompt_type == "icl":
        dataset = build_icl(args.data_type, args.data_path, args.model_type, dataset)

    for example in dataset:
        if "prometheus" in args.data_type:
            prompt = instruction.format(question_body=example["question_body"],
                                        rubric=example["rubric"],
                                        answer_body=example["answer_body"])
            prompts.append(prompt)
        else:
            example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
            if args.prompt_type == "icl":
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            demonstrations=example["demonstrations"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])
            else:
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])                
            prompts.append(prompt)

        answers.append(example["score"])

    if args.logit_file is None:
        args.logit_file = f"{args.data_type}-{args.model_type}-{args.prompt_type}.jsonl"

    # 如果logit_file已经存在，就直接读取内容，仅仅对其进行重新后处理抽取分数
    if args.logit_file is not None and os.path.exists(args.logit_file):
        with open(args.logit_file, "r", encoding="utf-8") as fin:
            lines = [json.loads(line) for line in fin.readlines()]

        predictions = [line["prediction"] for line in lines]
    
    else:
        manager = multiprocessing.Manager()
        counter = manager.Value("counter", 0)
        pool = multiprocessing.Pool(
            processes=args.pool_number, initializer=init, initargs=(counter,))

        if args.multi_process == "False":
            predictions = [gpt_scoring(sample, model=args.model_type, temperature=args.temperature, max_new_tokens=args.max_new_token)
                        for sample in prompts]
        else:
            pool_fn = partial(gpt_scoring, model=args.model_type, temperature=args.temperature, max_new_tokens=args.max_new_token)
            predictions = pool.map(pool_fn, prompts)

    is_pair = "prometheus" not in args.data_type
    is_cot = args.prompt_type == "cot"
    pred_scores = [parse_score_gpt(p, is_pair=is_pair, is_cot=is_cot) for p in predictions]

    # 存储prediction和score到文件中，便于后续确认是否后处理存在问题
    if args.logit_file is not None:
        with open(args.logit_file, "w", encoding="utf-8") as fout:
            for prediction, score in zip(predictions, pred_scores):
                json_line = {"score": score, "prediction": prediction}
                fout.write(json.dumps(json_line)+"\n")

    print(args)
    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print(f"model: {args.model_type}, data: {args.data_type}")
    print(metrics_dicts)