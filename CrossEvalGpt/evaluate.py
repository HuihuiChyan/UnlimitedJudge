import os
import json
import argparse
import random
import torch
import datasets
import re
import ray
import copy
import tqdm
import time
import openai
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, StoppingCriteria, GPT2Tokenizer
from build_prompt import create_prompt, create_prompt_aspect

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

def parse_score_judgelm(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            return float(sp[0])
    except Exception as e:
        # print(f'{e}\nContent: {review}\n'
        #              'You must manually fix the score pair.')
        return [-1, -1]

def parse_score_mtbench(review):
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

def parse_score_prometheus(review):
    score = 1
    try:
        score = review.split('[RESULT]')[1].strip()
        if score in ["1", "2", "3", "4", "5"]:
            return int(score)
        else:
            return 1
    except:
        return 1

def calculate_metrics(y_true_list, y_pred_list, data_type):
    if "prometheus" not in data_type:
        y_true = translate_score_to_win_list(y_true_list)
        y_pred = translate_score_to_win_list(y_pred_list)
    else:
        y_true = y_true_list
        y_pred = y_pred_list
    if args.data_type in ["judgelm", "mt-bench", "manual", "neighbor", "gptinst", "gptout", "natural"]:
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

def request_gpt4(prompt, temperature, max_new_token):
    url = "http://internal.ai-chat.host/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-QKZWGHrLphnxiyH8F7F1Fd62E18345409cA3049c7f9191E4", # 请确保替换'$sk'为您的实际token
    }
    # url = "https://api.chatgpt-3.vip/v1/chat/completions"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": "Bearer sk-YbA0PBLo6X76clo88aAb29Fc0852428c8850390375AbA32d", # 请确保替换'$sk'为您的实际token
    # }
    max_tries = 2
    res = ''
    for i in range(max_tries):
        try:
            messages = [{"role": "user", "content": prompt}]
            data = {"model": "gpt-4-1106-preview", "messages": messages, "temperature": temperature, "max_tokens": max_new_token}
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response = response.json()
            # if response['choices'][0]["finish_reason"] != "stop":
            #     raise openai.error.APIError("Completion stopped before completion.")
            res = response['choices'][0]['message']['content'].strip()
            # print(prompt)
            print(res)
            # print("——————————————————————————————————————————————————————")
            return res
            break
        except Exception as e:
            print(response)
            time.sleep(2)
            continue
    return res

def batched_request(
    examples,
    instruction,
    data_type,
    temperature,
    max_new_token,
):
    pred_list = []
    for example in tqdm.tqdm(examples):
        if data_type == "mt-bench":
            if example["turn"] == 1:
                prompt = instruction["single"].format(question=example["question"],
                                                    answer_a=example["answer_a"],
                                                    answer_b=example["answer_b"])
            else:
                prompt = instruction["multi"].format(question_1=example["question_1"],
                                                    answer_a_1=example["answer_a_1"],
                                                    answer_b_1=example["answer_b_1"],
                                                    question_2=example["question_2"],
                                                    answer_a_2=example["answer_a_2"],
                                                    answer_b_2=example["answer_b_2"])                           
        elif "prometheus" in data_type:           
            prompt = instruction.format(question_body=example["question_body"],
                                        rubric=example["rubric"],
                                        reference=example["reference"]['text'],
                                        answer_body=example["answer_body"])
        else:     
            prompt = instruction.format(question_body=example["question_body"],
                                        answer1_body=example["answer1_body"],
                                        answer2_body=example["answer2_body"])
        pred = request_gpt4(prompt, temperature, max_new_token)
        pred_list.append(pred)

    return pred_list

def batched_request_cot(
    examples,
    instruction,
    temperature,
    max_new_token,
):
    aspects = {"Helpfulness", "Relevence", "Accuracy", "Depth", "Creativity", "Detailedness"}
    aspect_prompts = {"Helpfulness": "Helpfulness: How useful was the response in providing clear and practical advice or solutions?",
                      "Relevence": "Relevence: Did the response directly answer the question and stay on topic?",
                      "Accuracy": "Accuracy: Was the information provided accurate and well-researched?",
                      "Depth": "Depth: How thorough and detailed was the response in terms of analysis and explanation?",
                      "Creativity": "Creativity: Did the response offer novel or unique perspectives or solutions?",
                      "Detailedness": "Detailedness: Was the response detailed enough to provide necessary context and background information?"}
    pred_list = []
    scores_list_a = []
    scores_list_b = []
    for example in tqdm.tqdm(examples):
        scores_dict_a = {}
        scores_dict_b = {}
        for aspect in aspects:
            rubric = aspect_prompts[aspect]
            if example["turn"] == 1:
                prompt = instruction["aspect_single"].format(question=example["question"],
                                                             answer=example["answer_a"],
                                                             rubric=rubric,)
            else:
                prompt = instruction["aspect_multi"].format(question_1=example["question_1"],
                                                            answer_1=example["answer_a_1"],
                                                            question_2=example["question_2"],
                                                            answer_2=example["answer_a_2"],
                                                            rubric=rubric,)
            pred = request_gpt4(prompt, temperature, max_new_token)
            scores_dict_a[aspect] = parse_score_prometheus(pred)

        for aspect in aspects:
            rubric = aspect_prompts[aspect]
            if example["turn"] == 1:
                prompt = instruction["aspect_single"].format(question=example["question"],
                                                             answer=example["answer_b"],
                                                             rubric=rubric,)
            else:
                prompt = instruction["aspect_multi"].format(question_1=example["question_1"],
                                                            answer_1=example["answer_b_1"],
                                                            question_2=example["question_2"],
                                                            answer_2=example["answer_b_2"],
                                                            rubric=rubric,)
            pred = request_gpt4(prompt, temperature, max_new_token)
            scores_dict_b[aspect] = parse_score_prometheus(pred)
        
        example["scores_a"] = instruction["scores"].format(**scores_dict_a)
        example["scores_b"] = instruction["scores"].format(**scores_dict_b)

        if example["turn"] == 1:
            prompt = instruction["single"].format(**example)
        else:
            prompt = instruction["multi"].format(**example)
    
        pred_list.append([request_gpt4(prompt, temperature, max_new_token), scores_dict_a, scores_dict_b])

    return pred_list

def batched_request_cot_with_score(
    examples,
    instruction,
    score_list_a,
    score_list_b,
    temperature,
    max_new_token,
):
    pred_list = []
    scores_list_a = []
    scores_list_b = []
    pbar = tqdm.tqdm(total=len(examples))
    for i in range(len(examples)):
        example = examples[i]
        scores_dict_a = score_list_a[i]
        scores_dict_b = score_list_b[i]
        
        example["scores_a"] = instruction["scores"].format(**scores_dict_a)
        example["scores_b"] = instruction["scores"].format(**scores_dict_b)

        if example["turn"] == 1:
            prompt = instruction["single"].format(**example)
        else:
            prompt = instruction["multi"].format(**example)
    
        pred_list.append([request_gpt4(prompt, temperature, max_new_token), scores_dict_a, scores_dict_b])
        pbar.update(1)

    return pred_list

def main(args):
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
        
        dataset = dataset
    
    elif args.data_type == "pandalm":
        with open(os.path.join(args.data_path, "pandalm/testset-v1.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        
        if args.demo_augmentation == "True":
            with open(os.path.join(args.data_path, "pandalm/pandalm_train.jsonl"), "r") as fin:
                lines = [line.strip() for line in fin.readlines()]
                demo_dataset = [json.loads(line) for line in lines]

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

        if args.demo_augmentation == "True":
            with open(os.path.join(args.data_path, "auto-j/pairwise_traindata_conv.jsonl"), "r") as fin:
                lines = [line.strip() for line in fin.readlines()]
                demo_dataset = [json.loads(line) for line in lines]

    elif args.data_type == "prometheus-ind":
        with open(os.path.join(args.data_path, "prometheus/feedback_collection_test.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        
        if args.demo_augmentation=="True":
            with open(os.path.join(args.data_path, "prometheus/new_feedback_collection.jsonl"), "r") as fin:
                lines = [line.strip() for line in fin.readlines()]
                demo_dataset = [json.loads(line) for line in lines]

    elif args.data_type == "prometheus-ood":
        with open(os.path.join(args.data_path, "prometheus/feedback_collection_ood_test.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "mt-bench":
        dataset = datasets.load_dataset("parquet", data_files={"train": "data/mt-bench/human-00000-of-00001-25f4910818759289.parquet"})

        samples = []
        for sample in dataset["train"]:
            if sample["turn"] == 1:
                example = {"question": sample["conversation_a"][0]["content"],
                           "answer_a": sample["conversation_a"][1]["content"],
                           "answer_b": sample["conversation_b"][1]["content"],
                           "turn": 1}
            else:
                example = {"question_1": sample["conversation_a"][0]["content"],
                           "answer_a_1": sample["conversation_a"][1]["content"],
                           "answer_b_1": sample["conversation_b"][1]["content"],
                           "question_2": sample["conversation_a"][2]["content"],
                           "answer_a_2": sample["conversation_a"][3]["content"],
                           "answer_b_2": sample["conversation_b"][3]["content"],
                           "turn": 2}
            if sample["winner"] == "model_a":
                example["score"] = [1, 0]
            elif sample["winner"] == "model_b":
                example["score"] = [0, 1]
            elif sample["winner"] == "tie":
                example["score"] = [1, 1]
            
            samples.append(example)
        
        random.seed(42)
        random.shuffle(samples)
        dataset = samples[:200]

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

    if args.cot_augmentation == "True":
        instruction = create_prompt_aspect()
    else:
        instruction = create_prompt(args.model_type, args.data_type, args.demo_augmentation=="True", args.cot_augmentation=="True")
    
    answers = [example["score"] for example in dataset]
    if args.load_scores:
        with open("scores_list_a.jsonl", "r") as fin:
            scores_list_a = [json.loads(line.strip()) for line in fin.readlines()]
        with open("scores_list_b.jsonl", "r") as fin:
            scores_list_b = [json.loads(line.strip()) for line in fin.readlines()]

        assert len(scores_list_a) == len(scores_list_b) == len(dataset)
        if args.use_ray == "True":
            batched_request_func = ray.remote(
                batched_request_cot_with_score
            ).remote
            thread_num = 8
            chunk_size = len(dataset) // thread_num
            ans_handles = []
            for i in range(0, len(dataset), chunk_size):
                ans_handles.append(batched_request_func(dataset[i : i + chunk_size], 
                                                        instruction,
                                                        scores_list_a[i : i + chunk_size],
                                                        scores_list_b[i : i + chunk_size],
                                                        args.temperature,
                                                        args.max_new_token))
            preds = ray.get(ans_handles)
            predictions = []
            scores_list_a = []
            scores_list_b = []
            for pred in preds:
                for p in pred:
                    predictions.append(p[0])
                    scores_list_a.append(p[1])
                    scores_list_b.append(p[2])
        else:
            preds = batched_request_cot_with_score(dataset,
                                                  instruction,
                                                  scores_list_a,
                                                  scores_list_b,
                                                  args.temperature,
                                                  args.max_new_token,)
            predictions = []
            scores_list_a = []
            scores_list_b = []
            for p in preds:
                predictions.append(p[0])
                scores_list_a.append(p[1])
                scores_list_b.append(p[2])    
    else:
        if args.data_type == "mt-bench":
            batched_request_func = batched_request_cot
            if args.use_ray == "True":
                batched_request_func_ray = ray.remote(
                    batched_request_func
                ).remote
                thread_num = 8
                chunk_size = len(dataset) // thread_num
                ans_handles = []
                for i in range(0, len(dataset), chunk_size):
                    ans_handles.append(batched_request_func_ray(dataset[i : i + chunk_size], instruction, args.temperature, args.max_new_token))
                preds = ray.get(ans_handles)
                predictions = []
                scores_list_a = []
                scores_list_b = []
                for pred in preds:
                    for p in pred:
                        predictions.append(p[0])
                        scores_list_a.append(p[1])
                        scores_list_b.append(p[2])
            else:
                predictions = batched_request_func(dataset,
                                                instruction,
                                                args.temperature,
                                                args.max_new_token,)
        
            with open("scores_list_a.jsonl", "w") as fout:
                for line in scores_list_a:
                    fout.write(json.dumps(line)+"\n")

            with open("scores_list_b.jsonl", "w") as fout:
                for line in scores_list_b:
                    fout.write(json.dumps(line)+"\n")
        else:
            batched_request_func = batched_request
            if args.use_ray == "True":
                batched_request_func_ray = ray.remote(
                    batched_request_func
                ).remote
                thread_num = 8
                chunk_size = len(dataset) // thread_num
                ans_handles = []
                for i in range(0, len(dataset), chunk_size):
                    ans_handles.append(batched_request_func_ray(dataset[i : i + chunk_size], instruction, args.data_type, args.temperature, args.max_new_token))
                preds = ray.get(ans_handles)
                predictions = []
                for pred in preds:
                    predictions.extend(pred)
            else:
                predictions = batched_request_func(dataset,
                                                   instruction,
                                                   args.data_type,
                                                   args.temperature,
                                                   args.max_new_token,)

    if args.data_type == "mt-bench":
        predictions = [parse_score_mtbench(pred) for pred in predictions]
    elif args.data_type in ["judgelm", "pandalm", "neighbor", "natural", "gptinst", "gptout", "manual"]:
        predictions = [parse_score_judgelm(pred) for pred in predictions]
    elif args.data_type == "auto-j":
        if "prometheus" not in args.data_type:
            predictions = [parse_score_autoj_pair(pred) for pred in predictions]
        elif "prometheus" in args.data_type:
            predictions = [parse_score_autoj_single(pred) for pred in predictions]
    elif "prometheus" in args.data_type:
        predictions = [parse_score_prometheus(pred) for pred in predictions]
    
    # with open("prometheus-ind-gpt4.json", "w") as fout:
    #     json.dump(predictions, fout)
    if args.logit_file is not None:
        with open(args.logit_file, "w") as fout:
            for p in predictions:
                fout.write(json.dumps(p) + "\n")

    metrics_dicts = calculate_metrics(answers, predictions, args.data_type)
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
        "--model-type",
        type=str,
        choices=("gpt4"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "mt-bench", "natural", "neighbor", "manual", "gptinst", "gptout"),
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
        "--eval-with-logits",
        type=str,
        choices=("True", "False"),
        default="True",
    )
    parser.add_argument(
        "--logit-file",
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
        "--use-ray",
        type=str,
        choices=("True", "False"),
        default="True"
    )
    parser.add_argument(
        "--add-reference",
        type=str,
        choices=("True", "False"),
        default="True"
    )
    parser.add_argument(
        "--demo-augmentation",
        type=str,
        choices=("True", "False"),
        default="False"
    )
    parser.add_argument(
        "--cot-augmentation",
        type=str,
        choices=("True", "False"),
        default="False"
    )
    parser.add_argument(
        "--load-scores",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    main(args)