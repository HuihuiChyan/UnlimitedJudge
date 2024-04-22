import os
import json
import argparse
import random
import torch
import datasets
import re
import copy
import tqdm
import time
import json
import openai
import requests
import multiprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, StoppingCriteria, GPT2Tokenizer

def build_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--class-type",
        type=str,
        choices=("generation", "regression", "classification"),
        default=None,
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
        "--eval-batch-size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
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

def gpt4_scoring(sample):
    instruction = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Always choose a better answer and do not output a tie. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better..

# [User Question]\n{question_body}\n\n[The Start of Assistant A's Answer]\n{answer1_body}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer2_body}\n[The End of Assistant B's Answer]

# Your Evaluation:"""


    prompt = instruction.format(**sample)

    prediction = request_gpt4(prompt, temperature=0.0, max_new_token=1024)
    prediction = parse_score_gpt(prediction)

    counter.value += 1    
    print(f"gpt4_scoring {counter.value} finished.")

    return prediction

def init(c):
    global counter
    counter = c

if __name__ == "__main__":
    args = build_params()
    random.seed(42)

    data_path_question = "question.jsonl"
    data_path_answer = "llama2-7b-chat-logprobs-entropy-auto-j.jsonl"

    with open(data_path_question, "r") as fin:
        lines_qes = [line.strip() for line in fin.readlines()]
        lines_qes = [json.loads(line) for line in lines_qes]
        dataset_qes = [line['turns'][0] for line in lines_qes]

    with open(data_path_answer, "r") as fin:
        lines_ans = [line.strip() for line in fin.readlines()]
        lines_ans = [json.loads(line) for line in lines_ans]
        dataset_ans = [[line['choices'][0]['turns'][0], line['choices'][1]['turns'][0]] for line in lines_ans]

    dataset = []
    for qes, ans in zip(dataset_qes, dataset_ans):
        example = {"rubric": "Please rate the helpfulness, relevance, accuracy, level of details of their responses."}
        example["question_body"] = qes
        example["answer1_body"] = ans[0]
        example["answer2_body"] = ans[1]
        dataset.append(example)

    import pdb;pdb.set_trace()
    answers = [example["score"] for example in dataset]

    manager = multiprocessing.Manager()
    counter = manager.Value("counter", 0)
    pool = multiprocessing.Pool(processes=args.pool_number, initializer=init, initargs=(counter,))

    # if args.cot_augmentation == "self-cot":
    #     if args.multi_process == "False":
    #         samples = [gpt4_planning(sample) for sample in dataset]
    #         samples = [gpt4_aspecting(sample) for sample in samples]
    #         predictions = [gpt4_scoring_cot_with_score(sample) for sample in samples]
    #     else:
    #         samples = pool.map(gpt4_planning, dataset)
    #         counter.value = 0
    #         samples = pool.map(gpt4_aspecting, samples)
    #         counter.value = 0
    #         predictions = pool.map(gpt4_scoring_cot_with_score, samples)

    if args.multi_process == "False":
        predictions = [gpt4_scoring(sample) for sample in dataset]
    else:
        predictions = pool.map(gpt4_scoring, dataset)

    with open(data_path_answer.rstrip(".jsonl")+"-gpt4.jsonl", "w") as fout:
        for line, score in zip(lines_ans, predictions):
            line["judge_score"] = score
            fout.write(json.dumps(line)+"\n")