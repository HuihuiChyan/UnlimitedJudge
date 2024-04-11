import os
import json
import argparse
import random
import torch
import datasets
import re
import ray
import copy
import vllm
import time
from tqdm import tqdm
import multiprocessing
import requests

def parse_score_single(score_output):
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("Rating: [["):pos2].strip())
    else:
        return 0.0

def request_gpt4(prompt, temperature, max_new_token):
    # url = "http://internal.ai-chat.host/v1/chat/completions"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": "Bearer sk-QKZWGHrLphnxiyH8F7F1Fd62E18345409cA3049c7f9191E4", # 请确保替换'$sk'为您的实际token
    # }
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
            messages = [{"role": "system", "content": prompt}]
            data = {"model": "gpt-4-1106-preview", "messages": messages, "temperature": temperature, "max_tokens": max_new_token}
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response = response.json()
            # if response['choices'][0]["finish_reason"] != "stop":
            #     raise openai.error.APIError("Completion stopped before completion.")
            res = response['choices'][0]['message']['content'].strip()
            print(prompt)
            print(res)
            print("——————————————————————————————————————————————————————")
            break
        except Exception as e:
            print(response)
            time.sleep(2)
            continue
    return res

def gpt4_scoring(prompt):

    prediction = request_gpt4(prompt, temperature=0.0, max_new_token=1024)
    prediction = parse_score_single(prediction)

    counter.value += 1    
    print(f"gpt4_scoring {counter.value} finished.")

    return prediction

def calculate_metrics(y_true_list, y_pred_list, data_type):
    if "prometheus" not in data_type:
        y_true = translate_score_to_win_list(y_true_list)
        y_pred = translate_score_to_win_list(y_pred_list)
    else:
        y_true = y_true_list
        y_pred = y_pred_list

    if data_type == "auto-j":
        y_true_frd = y_true[:len(y_true)//2]
        y_pred_frd = y_pred[:len(y_pred)//2]
        y_pred_rev = y_pred[len(y_pred)//2:]
        y_pred_rev = [0-y for y in y_pred_rev]

        acc_cnt = 0
        con_cnt = 0
        for i in range(len(y_true_frd)):
            if y_pred_frd[i] == y_pred_rev[i]:
                con_cnt += 1
                if y_true_frd[i] == y_pred_frd[i]:
                    acc_cnt += 1

        # add metrics to dict
        metrics_dict = {
            'accuracy': acc_cnt/len(y_true_frd),
            'consistency': con_cnt/len(y_true_frd),
        }
    
    elif args.data_type == "pandalm":
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # add metrics to dict
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    elif "prometheus" in args.data_type:
        from scipy.stats import pearsonr, spearmanr, kendalltau
        
        pearson = pearsonr(y_true, y_pred)[0]
        kendalltau = kendalltau(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]

        # add metrics to dict
        metrics_dict = {
            'pearson': pearson,
            'kendalltau': kendalltau,
            'spearman': spearman,
        }

    elif args.data_type in ["judgelm", "mt-bench"]:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # add metrics to dict
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    return metrics_dict

def init(c):
    global counter
    counter = c

def main(args):
    random.seed(42)
    with open(os.path.join(args.data_path, "./llama2-7b-chat/llama2-7b-chat_test.jsonl"), "r") as fin:
        lines = [line.strip() for line in fin.readlines()]
        dataset = [json.loads(line) for line in lines]

    instruction1 = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"""
    instruction2 = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>"""

    prompts = []
    answers = []
    for example in dataset:
        prompt = instruction1.format(question=example["question1_body"],
                                     answer=example["answer1_body"])
        prompts.append(prompt)
        prompt = instruction2.format(question_1=example["question1_body"],
                                     answer_1=example["answer1_body"],
                                     question_2=example["question2_body"],
                                     answer_2=example["answer2_body"])                    
        prompts.append(prompt)

    manager = multiprocessing.Manager()
    counter = manager.Value("counter", 0)
    pool = multiprocessing.Pool(processes=8, initializer=init, initargs=(counter,))

    if args.multi_process == "False":
        predictions = [gpt4_scoring(sample) for sample in prompts]
    else:
        predictions = pool.map(gpt4_scoring, prompts)

    predictions_a = [pred for pred in predictions[0::2]]
    predictions_b = [pred for pred in predictions[1::2]]
    pred_scores = [[pred[0], pred[1]] for pred in zip(predictions_a, predictions_b)]

    with open(os.path.join(args.data_path, "./llama2-7b-chat/llama2-7b-chat-selfeval.jsonl"), "w") as fout:
        for i,line in enumerate(pred_scores):
            fout.write(json.dumps({"question_id": i+80, "evaluations": line})+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
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
        choices=("judgelm", "pandalm", "auto-j", "prometheus", "llama", "deberta", "longformer"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "mt-bench"),
        default=None,
    )
    parser.add_argument(
        "--eval-batch-size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
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
        "--add-reference",
        type=str,
        choices=("True", "False"),
        default="True"
    )
    parser.add_argument(
        "--multi-process",
        type=str,
        choices=("True", "False"),
        default="True"
    )
    args = parser.parse_args()

    main(args)