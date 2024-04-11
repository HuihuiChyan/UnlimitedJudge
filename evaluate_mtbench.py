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

def parse_score_single(score_output):
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("Rating: [["):pos2].strip())
    else:
        return 0.0

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
    max_new_token=512,
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

def main(args):
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
    
    import pdb;pdb.set_trace()

    predictions = [parse_score_single(pred) for pred in predictions]

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
    args = parser.parse_args()

    main(args)