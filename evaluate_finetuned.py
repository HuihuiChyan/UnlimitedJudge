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

from train import create_prompt, format_instruction

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

if __name__ == "__main__":

    args = build_params()
    random.seed(42)

    dataset = load_dataset(args.data_type, args.data_path, args.add_reference)
    
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

    if args.class_type == "generation":
        pred_scores = batched_generation(args.model_name_or_path, prompts, 
                                         max_new_token=args.max_new_token, 
                                         temperature=args.temperature,
                                         top_p=args.top_p)
    elif args.class_type == "classification":
        pred_scores = batched_classification(args.model_name_or_path, prompts, is_regression=False)
    elif args.class_type == "regression":
        pred_scores = batched_classification(args.model_name_or_path, prompts, is_regression=True)
    
    parsed_scores = parse_predictions(model_type, data_type)
    
    if args.logit_file is not None:
        with open(args.logit_file, "w", encoding="utf-8") as fout:
            for pred in pred_scores:
                fout.write(json.dumps(pred)+"\n")

    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print(f"model: {args.model_type}, data: {args.data_type}")
    print(metrics_dicts)