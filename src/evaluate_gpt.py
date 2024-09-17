import json
import argparse
import random
import time
import json
import openai
import os
import re
import requests
import multiprocessing
from functools import partial

from evaluate_judge import build_dataset, calculate_metrics
from build_prompt_gpt import parse_score_gpt, create_prompt_gpt


def build_params_gpt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        # choices=("gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-2024-04-09", 
        #          "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"),
        default=None,
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=("vanilla", "cot"),
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
    parser.add_argument(
        "--rewrite-output",
        type=str,
        default="False",
    )
    args = parser.parse_args()
    return args


# def request_gpt(prompt, model, temperature, max_new_tokens):

#     # url = "https://api.ai-gaochao.cn/v1/chat/completions"
#     url = "https://idealab.alibaba-inc.com/api/openai/v1"
#     # headers = {
#     #     "Content-Type": "application/json",
#     #     "Authorization": "Bearer sk-agEcX3Su78Bu09c2F49978C6Ba424977B936C8710fAb42E0",
#     # }
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer sk-f84283ab79d26d15be359b6d6979308a",
#     }
#     model = "gpt-4o-0513"
#     max_tries = 5
#     res = ''
#     response = None
#     sys_info = {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."}
#     for i in range(max_tries):
#         try:
#             messages = [sys_info, {"role": "user", "content": prompt}]
#             messages = [{"role": "user", "content": prompt}]
#             data = {"model": model, "messages": messages,
#                     "temperature": temperature, "max_tokens": max_new_tokens}
#             response = requests.post(
#                 url, headers=headers, data=json.dumps(data))
#             response = response.json()
#             res = response['choices'][0]['message']['content'].strip()
#             break
#         except Exception as e:
#             print("Exception! The response is " + str(response))
#             time.sleep(5)
#             continue
#     return res

def request_gpt(prompt, model, temperature, max_new_tokens):
    model = "gpt-4o-0513"
    api_key = "f84283ab79d26d15be359b6d6979308a"
    client = openai.OpenAI(api_key=api_key, base_url="https://idealab.alibaba-inc.com/api/openai/v1")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    }
    max_tries = 20
    res = ''
    for i in range(max_tries):
        try:
            chat_completion = client.chat.completions.create(model=payload['model'], temperature=temperature, messages=payload['messages'])
            res = chat_completion.choices[0].message.content
            break
        except Exception as e:
            if i == max_tries-1:
                raise Exception("MAX_RETRY exceeded! Please check your codes! ")
            print("Exception! The exception is "+str(e))
            time.sleep(5)
            continue
    return res

def gpt_scoring(prompt, model, temperature, max_new_tokens):

    prediction = request_gpt(prompt, model, temperature=temperature, max_new_tokens=max_new_tokens)

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
    for example in dataset:
        if args.data_type in ["prometheus-ind", "prometheus-ood", "halu-eval-summary", "halu-eval-dialogue", "halu-eval-qa", "toxic-chat"]:
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

    print("Prompt built finished! Sampled prompt:")
    print(prompts[random.randint(0, len(prompts)-1)]+"\n")

    if args.logit_file is None:
        args.logit_file = f"./outputs/{args.data_type}-{args.model_name}-{args.prompt_type}.jsonl"

    if os.path.exists(args.logit_file):
        if args.rewrite_output == "True":
            os.remove(args.logit_file)        
        else:
            # 如果logit_file已经存在，就直接读取内容，仅仅对其进行重新后处理抽取分数
            with open(args.logit_file, "r", encoding="utf-8") as fin:
                lines = [json.loads(line) for line in fin.readlines()]

            predictions = [line["prediction"] for line in lines]
    
    if not os.path.exists(args.logit_file) or args.rewrite_output == "True":
        manager = multiprocessing.Manager()
        counter = manager.Value("counter", 0)
        pool = multiprocessing.Pool(processes=args.pool_number, initializer=init, initargs=(counter,))

        if args.multi_process == "False":
            predictions = [gpt_scoring(sample, model=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_token)
                           for sample in prompts]
        else:
            pool_fn = partial(gpt_scoring, model=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_token)
            predictions = pool.map(pool_fn, prompts)

    # is_pair = "prometheus" not in args.data_type and args.data_type not in ['halu-eval-summary', 'halu-eval-qa', 'halu-eval-dialogue', 'toxic-chat']
    # is_cot = args.prompt_type == "cot"
    pred_scores = [parse_score_gpt(p, data_type=args.data_type, prompt_type=args.prompt_type) for p in predictions]

    # 存储prediction和score到文件中，便于后续确认是否后处理存在问题
    with open(args.logit_file, "w", encoding="utf-8") as fout:
        for prediction, score in zip(predictions, pred_scores):
            json_line = {"score": score, "prediction": prediction}
            fout.write(json.dumps(json_line)+"\n")

    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print("**********************************************")
    print(f"Model: {args.model_name}, Data: {args.data_type}")
    print(metrics_dicts)
    print("**********************************************")