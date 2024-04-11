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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from template import instructions_pre
from train import create_prompt, format_instruction

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
    # try:
    #     score = re.sub(r"[1-9]\.|[0-9]{2,}", "", score_output)
    #     score = re.sub(r"[1-9]+\-year-old", "", score)
    #     score = score.replace("1 and 5", "")
    #     score = re.search(r"[0-9]", score).group()
    #     return float(score)
    # except:
    #     return 1.0


def parse_score_prometheus_pair(review):
    try:
        score = review.split('[RESULT]')[1].strip()
        score_pair = score.replace(',', ' ').replace('\n', ' ').replace('.', ' ')
        if '  ' in score_pair:
            score_pair = score_pair.replace('  ', ' ')
        sp = score_pair.split(' ')
        return [float(sp[0]), float(sp[1])]
    except:
        return [1.0, 1.0]

def parse_score_prometheus_single(review):
    try:
        score = review.split('[RESULT]')[1].strip()
        if score in ["1", "2", "3", "4", "5"]:
            return int(score)
        else:
            return 1
    except:
        return 1

@torch.inference_mode()
def batched_classification(
    model_path,
    prompts,
    eval_batch_size=16,
    is_regression=False,
):
    print("start load model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, low_cpu_mem_usage=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print("model loaded")
    
    # lens = [len(tokenizer.tokenize(prompt)) for prompt in prompts]
    # avg_lens = sum(lens) / len(lens)
    # longCnt = sum([int(len>512) for len in lens])
    # print(f"average length is {avg_lens}")
    # print(f"{longCnt} sents exceed 512")

    pred_list = []
    pbar = tqdm(total=len(prompts))
    for i in range(0, len(prompts), eval_batch_size):
        batched_prompt = prompts[i:i+eval_batch_size]

        input_ids = tokenizer(
            batched_prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).input_ids
        pbar.update(len(batched_prompt))

        logits = model(input_ids=input_ids.cuda(), 
                       attention_mask=input_ids.ne(tokenizer.pad_token_id).cuda())
        if is_regression:
            for logit in logits.logits.squeeze(-1).tolist():
                pred_list.append(logit)
        else:
            for logit in logits.logits.argmax(-1).tolist():
                if logit == 1:
                    pred_list.append([1, 0])           
                elif logit == 2:
                    pred_list.append([0, 1])
                elif logit == 0:
                    pred_list.append([1, 1])
    return pred_list

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

def calculate_metrics(y_true_list, y_pred_list, data_type):
    if "prometheus" not in data_type:
        y_true = translate_score_to_win_list(y_true_list)
        y_pred = translate_score_to_win_list(y_pred_list)
    else:
        y_true = y_true_list
        y_pred = y_pred_list

    if args.data_type in ["judgelm", "faireval", "llmeval2", "neighbor", "natural", "gptinst", "gptout", "manual"]:
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
    return metrics_dict

def main(args):
    random.seed(42)
    if args.data_type == "judgelm":
        with open(os.path.join(args.data_path, "judgelm/judgelm_val_5k.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        with open(os.path.join(args.data_path, "judgelm/judgelm_val_5k_references.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset_ref = [json.loads(line) for line in lines]

        for example, example_ref in zip(dataset, dataset_ref):
            example["reference"] = example_ref["reference"]

        if args.add_reference == "True":
            with open(os.path.join(args.data_path, "judgelm/judgelm_val_5k_gpt4_with_reference.jsonl"), "r") as fin:
                lines = [line.strip() for line in fin.readlines()]
                dataset_score = [json.loads(line) for line in lines]
        else:
            with open(os.path.join(args.data_path, "judgelm/judgelm_val_5k_gpt4.jsonl"), "r") as fin:
                lines = [line.strip() for line in fin.readlines()]
                dataset_score = [json.loads(line) for line in lines]            
        for example, example_score in zip(dataset, dataset_score):
            example["score"] = example_score["score"]
    
    elif args.data_type == "pandalm":
        with open(os.path.join(args.data_path, "pandalm/testset-v1.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "auto-j":
        with open(os.path.join(args.data_path, "auto-j/testdata_pairwise_conv.jsonl"), "r") as fin:
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
    elif args.data_type == "prometheus-ind":
        with open(os.path.join(args.data_path, "prometheus/feedback_collection_test.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "prometheus-ood":
        with open(os.path.join(args.data_path, "prometheus/feedback_collection_ood_test.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "faireval":
        with open(os.path.join(args.data_path, "faireval/faireval.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "llmeval2":
        with open(os.path.join(args.data_path, "llmeval2/llmeval2.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "neighbor":
        with open(os.path.join(args.data_path, "Neighbor/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]    

    elif args.data_type == "natural":
        with open(os.path.join(args.data_path, "Natural/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "manual":
        with open(os.path.join(args.data_path, "Manual/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "gptinst":
        with open(os.path.join(args.data_path, "GPTInst/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "gptout":
        with open(os.path.join(args.data_path, "GPTOut/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]  
    
    if args.model_type in ["llama", "deberta"]:
        is_prometheus = ("prometheus" in args.data_type)
        instruction = create_prompt(args.class_type, args.model_type, is_prometheus=is_prometheus)
        instruction = instruction["noref"]
        prompts = []
        answers = []
        for example in dataset:
            if "prometheus" in args.data_type:
                example["rubric"] = example["rubric"].split("]\nScore 1:")[0][1:]
            prompt = format_instruction(instruction, example, args.class_type)
            prompts.append(prompt)
            answers.append(example["score"])

    else:
        instruction = instructions_pre[args.prompt_name]
        prompts = []
        answers = []
        for example in dataset:
            if args.model_type in ["judgelm", "pandalm", "auto-j"]:
                if "reference" not in example.keys():
                    example["reference"] = {"text": None}
                if "prometheus" in args.data_type:
                    prompt = instruction.format(question_body=example["question_body"],
                                                rubric=example["rubric"],
                                                reference=example["reference"]['text'],
                                                answer_body=example["answer_body"])
                    prompts.append(prompt)
                else:
                    prompt = instruction.format(question_body=example["question_body"],
                                                answer1_body=example["answer1_body"],
                                                reference=example["reference"]['text'],
                                                answer2_body=example["answer2_body"])
                    prompts.append(prompt)

            elif args.model_type == "prometheus":
                if "reference" not in example.keys():
                    example["reference"] = {"text": None}
                if "prometheus" in args.data_type:
                    prompt = instruction.format(question_body=example["question_body"],
                                                rubric=example["rubric"],
                                                reference=example["reference"]['text'],
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

    import pdb;pdb.set_trace()

    if args.class_type == "generation":
        predictions = batched_generation(args.model_name_or_path, prompts, 
                                         max_new_token=args.max_new_token, 
                                         temperature=args.temperature,
                                         top_p=args.top_p)
    elif args.class_type == "classification":
        pred_scores = batched_classification(args.model_name_or_path, prompts, is_regression=False)
    elif args.class_type == "regression":
        pred_scores = batched_classification(args.model_name_or_path, prompts, is_regression=True)

    if args.model_type == "judgelm" and "prometheus" not in args.data_type:
        pred_scores = [parse_score_judgelm_pair(pred) for pred in predictions]
    elif args.model_type == "judgelm" and "prometheus" in args.data_type:
        pred_scores = [parse_score_judgelm_single(pred) for pred in predictions]
    elif args.model_type == "auto-j" and "prometheus" not in args.data_type:
        pred_scores = [parse_score_autoj_pair(pred) for pred in predictions]
    elif args.model_type == "auto-j" and "prometheus" in args.data_type:
        pred_scores = [parse_score_autoj_single(pred) for pred in predictions]
    elif args.model_type == "pandalm" and "prometheus" not in args.data_type:
        pred_scores = [parse_score_pandalm_pair(pred) for pred in predictions]
    elif args.model_type == "pandalm" and "prometheus" in args.data_type:
        pred_scores = [parse_score_pandalm_single(pred) for pred in predictions]
    elif args.model_type == "prometheus":
        pred_scores = [parse_score_prometheus_single(pred) for pred in predictions]
        if "prometheus" not in args.data_type:
            predictions_a = [pred for pred in pred_scores[0::2]]
            predictions_b = [pred for pred in pred_scores[1::2]]
            pred_scores = [[pred[0], pred[1]] for pred in zip(predictions_a, predictions_b)]
    
    if args.logit_file is not None:
        with open(args.logit_file, "w", encoding="utf-8") as fout:
            for pred in pred_scores:
                fout.write(json.dumps(pred)+"\n")

    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print(f"model: {args.model_type}, data: {args.data_type}")
    print(metrics_dicts)

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
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "faireval", "llmeval2", "neighbor", "natural", "gptinst", "gptout", "manual"),
        default=None,
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
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
        "--logit-file",
        type=str,
        default=None
    )
    args = parser.parse_args()

    main(args)