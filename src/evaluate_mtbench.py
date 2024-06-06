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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, StoppingCriteria

from build_prompt_mtbench import create_prompt_predefined
from evaluate import batched_generation

def parse_score_single(score_output):
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("Rating: [["):pos2].strip())
    else:
        return 0.0

def calculate_metrics(y_true_list, y_pred_list, data_type):
    assert data_type == "mt-bench"
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

if __name__ == "__main__":
    random.seed(42)

    instruction1 = """<s>[INST] Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".
<|The Start of Assistant's Conversation with User|>\n\n### User:\n{question}\n\n### Assistant:\n{answer}\n\n<|The End of Assistant's Conversation with User|>[/INST]"""
    instruction2 = """<s>[INST] Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".
<|The Start of Assistant's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant:\n{answer_2}\n\n<|The End of Assistant's Conversation with User|>[/INST]"""
    with open(os.path.join(args.data_path, "./llama2-7b-chat/llama2-7b-chat_test.jsonl"), "r") as fin:
        lines = [line.strip() for line in fin.readlines()]
        dataset = [json.loads(line) for line in lines]

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

    predictions = batched_generation(args.model_name_or_path, prompts, 
                                     max_new_token=args.max_new_token, 
                                     temperature=args.temperature,
                                     top_p=args.top_p)

    predictions = [parse_score_single(pred) for pred in predictions]

    predictions_a = [pred for pred in predictions[0::2]]
    predictions_b = [pred for pred in predictions[1::2]]
    pred_scores = [[pred[0], pred[1]] for pred in zip(predictions_a, predictions_b)]

    calculate_metrics(pred_scores, args.data_type)