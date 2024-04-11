import os
import json
import argparse
import random
import torch
import datasets
import re
import copy
import vllm
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        score_pair = review.strip().split('\n')[0]
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

def parse_score_autoj_pair(review):
    if re.search(r"output \(a\)", review.lower()):
        return [1, 0]
    elif re.search(r"output \(b\)", review.lower()):
        return [0, 1]
    else:
        return [1, 1]

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

def batched_generation(
    model_path,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    print("start load model")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("model loaded")
    pbar = tqdm(total=len(prompts))
    pred_list = []
    batch_size = 8
    for i in range(0, len(prompts), batch_size):
        tokenized = tokenizer(prompts[i: i+batch_size], return_tensors="pt", padding="longest")
        input_ids = tokenized.input_ids.to("cuda")
        attention_mask = tokenized.attention_mask.to("cuda")
        outputs = model.generate(input_ids, attention_mask=attention_mask,\
                                 temperature=0.1, do_sample=True, top_p=0.1,
                                 max_new_tokens=256, output_attentions=True)
        pred_list.extend(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:]))
        pbar.update(batch_size)

    return pred_list

def batched_generation_vllm(
    model_path,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    print("start load model")
    model = vllm.LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.8)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=16,
        top_p=top_p,
    )
    print("model loaded")
    pred_list = model.generate(prompts, sampling_params)
    pred_list = [it.outputs[0].text for it in pred_list]
    return pred_list

def batched_logprobs_vllm(
    model,
    tokenizer,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    input_ids = []
    attn_masks = []
    for prompt in prompts:
        sample = tokenizer(prompt, return_tensors=None)
        input_ids.append(sample['input_ids'])
        attn_masks.append(sample['attention_mask'])
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=1,
        prompt_logprobs=0,
    )
    outputs = model.generate(sampling_params=sampling_params, prompt_token_ids=input_ids)
    logits = [output.prompt_logprobs for output in outputs]
    for logit in logits:
        for i in range(len(logit)):
            if i == 0:
                logit[0] = 0.0
            else:
                logit[i] = list(logit[i].values())[0]

    predicts = []
    logprobs = []
    for logit, attn_mask in zip(logits, attn_masks):
        logprob = np.multiply(attn_mask, logit).sum()/(sum(attn_mask)+1)
        logprobs.append(logprob)
    return logprobs

def batched_logprobs(
    model,
    tokenizer,
    prompts,
    prompts_prefix,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    logprobs = []
    for i in tqdm(range(len(prompts))):
        # input_ids = tokenizer.encode(prompts_prefix[i])
        # tgt_ids = tokenizer.encode(prompts[i])[1:]
        # output_ids = [-100] * len(input_ids)
        # output_ids[len(input_ids) - len(tgt_ids):] = tgt_ids
        prefix_ids = tokenizer.encode(prompts_prefix[i])
        input_ids = tokenizer.encode(prompts[i])
        output_ids = copy.deepcopy(input_ids)
        target_len = len(input_ids) - len(prefix_ids)
        output_ids[:len(prefix_ids)-1] = [-100] * (len(prefix_ids)-1)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
        output_ids = torch.LongTensor(output_ids).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                labels=output_ids,
                output_hidden_states=True,
                output_attentions=True,
            )

        loss, logits, hidden_states = outputs[0], outputs[1], outputs.hidden_states[0]
        shifted_input_ids = torch.roll(input_ids, shifts=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs[output_ids== -100] = 0
        
        attentions = torch.cat(outputs['attentions'][-8:], dim=0).sum(dim=0).sum(dim=0)
        attentions = attentions.masked_fill(output_ids!=-100, 0.0)
        # attentions[:, 0] = 0.0
        attentions = attentions.sum(dim=-1)/1024
        attentions = torch.softmax(attentions, dim=-1)
        log_probs = log_probs * attentions.unsqueeze(0).unsqueeze(-1)

        scores = torch.gather(log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1).sum(-1) / (output_ids!= -100).long().sum(-1)

        # loss = loss.item()
        # score = -loss
        logprobs.append(len(prefix_ids))
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

    elif args.data_type == "llama2-7b-chat":
        with open(os.path.join(args.data_path, "./llama2-7b-chat/llama2-7b-chat_test.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
            
    if args.class_type == "generation":
        from build_prompt import instructions
        instruction = instructions["generation"]
        prompts = []
        answers = []
        for example in dataset:
            if "reference" not in example.keys():
                example["reference"] = {"text": None}
            prompt = instruction.format(question_body=example["question_body"],
                                        reference=example["reference"]['text'],
                                        answer1_body=example["answer1_body"],
                                        answer2_body=example["answer2_body"])
            prompts.append(prompt)
            answers.append(example["score"])

        # model = vllm.LLM(model=args.model_name_or_path, tensor_parallel_size=1)
        predictions = batched_generation(args.model_name_or_path, prompts, 
                                         max_new_token=args.max_new_token, 
                                         temperature=args.temperature,
                                         top_p=args.top_p)

        # pred_scores = [parse_score_judgelm_pair(pred) for pred in predictions]
        pred_scores = [parse_score_autoj_pair(pred) for pred in predictions]

    elif args.class_type == "logprobs":
        from build_prompt import instructions
        instruction = instructions["logprobs"]
        instruction_pre = instructions["logprobs_pre"]
        prompts1 = []
        prompts2 = []
        answers = []
        prompts_pre = []
        for example in dataset:
            if "reference" not in example.keys():
                example["reference"] = {"text": None}
            prompt1 = instruction.format(question_body=example["question_body"],
                                         reference=example["reference"]['text'],
                                         answer_body=example["answer1_body"],)
            prompt2 = instruction.format(question_body=example["question_body"],
                                         reference=example["reference"]['text'],
                                         answer_body=example["answer2_body"],)
            prompt_pre = instruction_pre.format(question_body=example["question_body"],
                                                reference=example["reference"]['text'],)
            prompts1.append(prompt1)
            prompts2.append(prompt2)
            prompts_pre.append(prompt_pre)
            answers.append(example["score"])

        if args.use_vllm:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            model = vllm.LLM(model=args.model_name_or_path, tensor_parallel_size=1)
            tokenizer.pad_token = tokenizer.eos_token
            predictions1 = batched_logprobs(model, tokenizer, prompts1, 
                                            max_new_token=args.max_new_token, 
                                            temperature=args.temperature,
                                            top_p=args.top_p)
            predictions2 = batched_logprobs(model, tokenizer, prompts2, 
                                            max_new_token=args.max_new_token, 
                                            temperature=args.temperature,
                                            top_p=args.top_p)
            predictions_pre = batched_logprobs(model, tokenizer, prompts_pre, 
                                            max_new_token=args.max_new_token, 
                                            temperature=args.temperature,
                                            top_p=args.top_p)
            pred_scores = []
            for pre in zip(predictions1, predictions2, predictions_pre):
                pred_scores.append([pre[0]-pre[2], pre[1]-pre[2]])
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
            tokenizer.pad_token = tokenizer.eos_token
            predictions1 = batched_logprobs(model, tokenizer, prompts1, prompts_pre,
                                            max_new_token=args.max_new_token, 
                                            temperature=args.temperature,
                                            top_p=args.top_p)
            predictions2 = batched_logprobs(model, tokenizer, prompts2, prompts_pre,
                                            max_new_token=args.max_new_token, 
                                            temperature=args.temperature,
                                            top_p=args.top_p)
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
        choices=("generation", "logprobs"),
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
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "faireval", "llmeval2", "neighbor", "natural", "gptinst", "gptout", "manual", "llama2-7b-chat"),
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