import os
import json
import argparse
import random
import re
import copy
import tqdm
import time
import json
import requests
import multiprocessing

from evaluate_judge import build_dataset, parse_predictions, calculate_metrics

def build_params_gpt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt4",
        choices=("gpt-3.5", "gpt-4")
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus", "llama", "deberta"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "llmbar-neighbor", "llmbar-natural", "llmbar-gptinst", "llmbar-gptout", "llmbar-manual"),
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
        default=2048,
        help="The maximum number of new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
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
        default=8,        
    )
    parser.add_argument(
        "--multi-process",
        type=str,
        default="True",        
    )
    args = parser.parse_args()
    return args

def parse_score_gpt(review):
    try:
        score = re.search(r"\[\[[A|B|C]\]\]", review).group()
        if score == "[[A]]":
            return [1, 0]
        elif score == "[[B]]":
            return [0, 1]
        else:
            return [1, 1]
    except Exception as e:
        # print(f'{e}\nContent: {review}\n'
        #              'You must manually fix the score pair.')
        return [-1, -1]

def request_gpt4(prompt, temperature, max_new_token):
    url = "https://api.chatgpt-3.vip/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-YbA0PBLo6X76clo88aAb29Fc0852428c8850390375AbA32d", # 请确保替换'$sk'为您的实际token
    }
    max_tries = 2
    res = ''
    response = None
    for i in range(max_tries):
        try:
            messages = [{"role": "user", "content": prompt}]
            data = {"model": "gpt-4-1106-preview", "messages": messages, "temperature": temperature, "max_tokens": max_new_token}
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response = response.json()
            print(response)
            res = response['choices'][0]['message']['content'].strip()

            break
        except Exception as e:
            print(response)
            time.sleep(2)
            continue
    return res

def create_prompt_gpt(data_type):
    if "prometheus" not in data_type:
        instruction = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Always choose a better answer and do not output a tie. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better..

    # [User Question]\n{question_body}\n\n[The Start of Assistant A's Answer]\n{answer1_body}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer2_body}\n[The End of Assistant B's Answer]

    # Your Evaluation:"""
    else:
        pass
    return instruction

def gpt4_scoring(prompt):

    prediction = request_gpt4(prompt, temperature=0.0, max_new_token=1024)
    prediction = parse_score_gpt(prediction)

    counter.value += 1    
    print(f"gpt4_scoring {counter.value} finished.")

    return prediction

def init(c):
    global counter
    counter = c

if __name__ == "__main__":
    args = build_params_gpt()
    random.seed(42)

    dataset = build_dataset(args.data_type, args.data_path)

    instruction = create_prompt_gpt(args.data_type)
    prompts = []
    answers = []
    for example in dataset:
        if "prometheus" in args.data_type:
            prompt = instruction.format(question_body=example["question_body"],
                                        rubric=example["rubric"],
                                        answer_body=example["answer_body"])
            prompts.append(prompt)
        else:
            example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
            prompt = instruction.format(question_body=example["question_body"],
                                        rubric=example["rubric"],
                                        answer1_body=example["answer1_body"],
                                        answer2_body=example["answer2_body"])
            prompts.append(prompt)

        answers.append(example["score"])

    manager = multiprocessing.Manager()
    counter = manager.Value("counter", 0)
    pool = multiprocessing.Pool(processes=args.pool_number, initializer=init, initargs=(counter,))

    if args.multi_process == "False":
        pred_scores = [gpt4_scoring(sample) for sample in dataset]
    else:
        pred_scores = pool.map(gpt4_scoring, dataset)

    if args.logit_file is not None:
        with open(args.logit_file, "w", encoding="utf-8") as fout:
            for pred in pred_scores:
                fout.write(json.dumps(pred)+"\n")

    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print(f"model: {args.model_type}, data: {args.data_type}")
    print(metrics_dicts)