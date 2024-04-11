import os
import json
import argparse
import random
import torch
import datasets
import re
import copy
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def batched_logprobs(
    model,
    tokenizer,
    prompts,
    prompts_prefix,
    estimation_mode = "logprobs-suffix"
):
    logprobs = []
    for i in tqdm(range(len(prompts))):
        input_ids = tokenizer([prompts_prefix[i]], max_length=1024, truncation=True).input_ids
        output_ids = tokenizer([prompts[i]], max_length=1024, truncation=True).input_ids

        prefix_len = len(input_ids[0])
        target_len = len(output_ids[0]) - prefix_len

        output_ids = torch.as_tensor(output_ids).cuda()
        input_ids = copy.deepcopy(output_ids).cuda()

        second_outputs = model(
            input_ids=input_ids,
            labels=output_ids,
            output_hidden_states=True,
            output_attentions=True,
        )

        if estimation_mode == "logprobs-full":
            shifted_input_ids = torch.roll(input_ids, shifts=-1)
            log_probs = torch.nn.functional.log_softmax(second_outputs["logits"], dim=-1)
            # output_ids[0][:prefix_len] = -100
            # log_probs[output_ids==-100] = 0
            output = torch.gather(log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1).sum(-1).tolist()[0] / (target_len + prefix_len)

        elif estimation_mode == "logprobs-suffix":
            shifted_input_ids = torch.roll(input_ids, shifts=-1)
            log_probs = torch.nn.functional.log_softmax(second_outputs["logits"], dim=-1)
            output_ids[0][:prefix_len] = -100
            log_probs[output_ids==-100] = 0
            output = torch.gather(log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1).sum(-1).tolist()[0] / (target_len + 1e-6)

        elif estimation_mode == "entropy-full":
            log_probs = torch.nn.functional.log_softmax(second_outputs["logits"], dim=-1)
            # output_ids[0][:prefix_len] = -100
            # log_probs[output_ids!=-100] = 0
            log_probs = log_probs * second_outputs["logits"]
            output = -(log_probs.sum(-1) / (target_len + prefix_len)).sum(-1).tolist()[0] / 32000
        
        elif estimation_mode == "variance-full":
            log_probs = torch.nn.functional.log_softmax(second_outputs["logits"], dim=-1)
            log_probs = torch.var(log_probs, dim=-1)
            # output_ids[0][:prefix_len] = -100                    
            # log_probs[output_ids!=-100] = 0
            output = -log_probs.sum(-1).tolist()[0] / (target_len + prefix_len)

        logprobs.append(output)
    return logprobs

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
    
    instruction = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {question_body} ASSISTANT: {answer_body}"
    instruction_pre = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {question_body} ASSISTANT: "

    prompts1 = []
    prompts2 = []
    answers = []
    prompts_pre = []
    for example in dataset:
        prompt1 = instruction.format(question_body=example["question_body"],
                                     answer_body=example["answer1_body"],)
        prompt2 = instruction.format(question_body=example["question_body"],
                                     answer_body=example["answer2_body"],)
        prompt_pre = instruction_pre.format(question_body=example["question_body"])
        prompts1.append(prompt1)
        prompts2.append(prompt2)
        prompts_pre.append(prompt_pre)
        answers.append(example["score"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    tokenizer.pad_token = tokenizer.eos_token
    predictions1 = batched_logprobs(model, tokenizer, prompts1, prompts_pre, args.class_type)
    predictions2 = batched_logprobs(model, tokenizer, prompts2, prompts_pre, args.class_type)
    pred_scores = []
    for pre in zip(predictions1, predictions2):
        pred_scores.append([pre[0], pre[1]])

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
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    main(args)