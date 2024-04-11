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
import sklearn
import scipy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from build_prompt import create_prompt

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
    args = parser.parse_args()
    return args

def translate_score_to_win_list(score_list, T=0.0):
    win_list = []
    for i in range(len(score_list)):
        if score_list[i][0] - score_list[i][1] > T:
            win_list.append(1)
        elif score_list[i][1] - score_list[i][0] > T:
            win_list.append(-1)
        else:
            win_list.append(0)
    return win_list

@torch.inference_mode()
def batched_generation(
    model_path,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    print("start load model")
    model = vllm.LLM(model=model_path, tensor_parallel_size=1)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
    )
    print("model loaded")

    pred_list = model.generate(prompts, sampling_params)
    pred_list = [it.outputs[0].text for it in pred_list]

    return pred_list

def load_dataset(data_type, data_path, add_reference=False):
    if data_type == "judgelm":
        with open(os.path.join(data_path, "judgelm/judgelm_val_5k.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

        with open(os.path.join(data_path, "judgelm/judgelm_val_5k_gpt4.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset_score = [json.loads(line) for line in lines]

        for example, example_score in zip(dataset, dataset_score):
            example["score"] = example_score["score"]
    
    elif data_type == "pandalm":
        with open(os.path.join(data_path, "pandalm/testset-v1.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif data_type == "auto-j":
        with open(os.path.join(data_path, "auto-j/testdata_pairwise_conv.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

        reve_dataset = []
        for example in dataset:
            rev_example = copy.deepcopy(example)
            temp_body = rev_example["answer1_body"]
            rev_example["answer1_body"] = rev_example["answer2_body"]
            rev_example["answer2_body"] = temp_body
            reve_dataset.append(rev_example)

        dataset.extend(reve_dataset)

    elif data_type == "prometheus-ind":
        with open(os.path.join(data_path, "prometheus/feedback_collection_test.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif data_type == "prometheus-ood":
        with open(os.path.join(data_path, "prometheus/feedback_collection_ood_test.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif data_type == "llmbar-neighbor":
        with open(os.path.join(data_path, "llmbar/Neighbor/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]    

    elif data_type == "llmbar-natural":
        with open(os.path.join(data_path, "llmbar/Natural/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif data_type == "llmbar-manual":
        with open(os.path.join(data_path, "llmbar/Manual/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif data_type == "llmbar-gptinst":
        with open(os.path.join(data_path, "llmbar/GPTInst/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif data_type == "llmbar-gptout":
        with open(os.path.join(data_path, "llmbar/GPTOut/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
    
    return dataset

def parse_predictions(model_type, data_type):
    def parse_score_judgelm_pair(review):
        try:
            score_pair = review.split('\n')[0]
            score_pair = score_pair.replace(',', ' ')
            sp = score_pair.split(' ')
            return [float(sp[0]), float(sp[1])]
        except Exception as e:
            return [1.0, 1.0]

    def parse_score_judgelm_single(review):
        try:
            score = review.split('\n')[0].strip()
            return float(score)
        except Exception as e:
            # print(f'{e}\nContent: {review}\n'
            #              'You must manually fix the score pair.')
            return 1.0

    def parse_score_pandalm_pair(review):
        score = review.split('\n')[0].strip()
        if score == "1":
            return [1, 0]
        elif score == "2":
            return [0, 1]
        elif score == "Tie":
            return [1, 1]
        else:
            return [1, 1]

    def parse_score_pandalm_single(review):
        score = review.split('\n')[0].strip()
        if score in ['1', '2', '3', '4', '5']:
            return int(score)
        else:
            return 5

    def parse_score_autoj_pair(raw_output):
        raw_output = raw_output.strip()
        pos = raw_output.rfind('final decision is ')
        pred_label = -1
        if pos != -1:
            pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
            if pred_rest.startswith('response 1'):
                return [1, 0]
            elif pred_rest.startswith('response 2'):
                return [0, 1]
            elif pred_rest.startswith('tie'):
                return [1, 1]
            else:
                return [-1, -1]
        else:
            return [-1, -1]

    def parse_score_autoj_single(score_output):
        if "Rating: [[" in score_output:
            pos = score_output.rfind("Rating: [[")
            pos2 = score_output.find("]]", pos)
            assert pos != -1 and pos2 != -1
            return float(score_output[pos + len("Rating: [["):pos2].strip())
        else:
            return 0.0

    def parse_score_prometheus(review, is_pair=True):
        if is_pair:
            try:
                score = review.split('[RESULT]')[1].strip()
                score_pair = score.replace(',', ' ').replace('\n', ' ').replace('.', ' ')
                if '  ' in score_pair:
                    score_pair = score_pair.replace('  ', ' ')
                sp = score_pair.split(' ')
                return [float(sp[0]), float(sp[1])]
            except:
                return [1.0, 1.0]
        else:
            try:
                score = review.split('[RESULT]')[1].strip()
                if score in ["1", "2", "3", "4", "5"]:
                    return int(score)
                else:
                    return 1
            except:
                return 1

    if args.model_type == "judgelm":
        is_pair = "prometheus" not in args.data_type
        pred_scores = [parse_score_judgelm(pred, is_pair=is_pair) for pred in predictions]
    elif args.model_type == "pandalm":
        is_pair = "prometheus" not in args.data_type
        pred_scores = [parse_score_pandalm(pred, is_pair=is_pair) for pred in predictions]
    elif args.model_type == "auto-j":
        is_pair = "prometheus" not in args.data_type
        pred_scores = [parse_score_autoj(pred, is_pair=is_pair) for pred in predictions]
    elif args.model_type == "prometheus":
        is_pair = "prometheus" not in args.data_type
        pred_scores = [parse_score_prometheus(pred, is_pair=False) for pred in predictions]
        if "prometheus" not in args.data_type:
            predictions_a = [pred for pred in pred_scores[0::2]]
            predictions_b = [pred for pred in pred_scores[1::2]]
            pred_scores = [[pred[0], pred[1]] for pred in zip(predictions_a, predictions_b)]

    return pred_scores

def calculate_metrics(y_true_list, y_pred_list, data_type):
    if "prometheus" not in data_type:
        y_true = translate_score_to_win_list(y_true_list)
        y_pred = translate_score_to_win_list(y_pred_list)
    else:
        y_true = y_true_list
        y_pred = y_pred_list

    if data_type in ["judgelm", "pandalm"] or "llmbar" in data_type:
        
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
        recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
        f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')

        # add metrics to dict
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    elif data_type == "auto-j":
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

    elif "prometheus" in args.data_type:
        pearson = scipy.stats.pearsonr(y_true, y_pred)[0]
        kendalltau = scipy.stats.kendalltau(y_true, y_pred)[0]
        spearman = scipy.stats.spearmanr(y_true, y_pred)[0]

        # add metrics to dict
        metrics_dict = {
            'pearson': pearson,
            'kendalltau': kendalltau,
            'spearman': spearman,
        }
    return metrics_dict

if __name__ == "__main__":

    args = build_params()
    random.seed(42)

    dataset = load_dataset(args.data_type, args.data_path)
    
    instruction = instructions_pre[args.prompt_name]
    prompts = []
    answers = []
    for example in dataset:
        if args.model_type in ["judgelm", "pandalm", "auto-j"]:
            if "prometheus" in args.data_type:
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer_body=example["answer_body"])
                prompts.append(prompt)
            else:
                prompt = instruction.format(question_body=example["question_body"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])
                prompts.append(prompt)

        elif args.model_type == "prometheus":
            if "prometheus" in args.data_type:
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer_body=example["answer_body"])
                prompts.append(prompt)
            else:
                example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
                prompt_a = instruction.format(question_body=example["question_body"],
                                              rubric=example["rubric"],
                                              answer_body=example["answer1_body"])
                prompt_b = instruction.format(question_body=example["question_body"],
                                              rubric=example["rubric"],
                                              answer_body=example["answer2_body"])
                prompts.append(prompt_a)
                prompts.append(prompt_b)
        
        answers.append(example["score"])

    pred_scores = batched_generation(args.model_name_or_path, prompts, 
                                        max_new_token=args.max_new_token, 
                                        temperature=args.temperature,
                                        top_p=args.top_p)

    parsed_scores = parse_predictions(model_type, data_type)
    
    if args.logit_file is not None:
        with open(args.logit_file, "w", encoding="utf-8") as fout:
            for pred in pred_scores:
                fout.write(json.dumps(pred)+"\n")

    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print(f"model: {args.model_type}, data: {args.data_type}")
    print(metrics_dicts)