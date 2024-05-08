import os
import re
import json
import torch
import argparse
from sklearn.neighbors import NearestNeighbors
import random
import copy
import scipy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from build_prompt import create_prompt


def build_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=("vanilla", "cot", "no_cot", "icl"),
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
        choices=("judgelm", "pandalm", "auto-j",
                 "prometheus", "llama", "deberta"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood",
                 "llmbar-neighbor", "llmbar-natural", "llmbar-gptinst", "llmbar-gptout", "llmbar-manual"),
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


@torch.inference_mode()
def batched_generation(
    model_path,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    print("start load model")
    import vllm
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


def build_dataset(data_type, data_path):
    if data_type == "judgelm":
        with open(os.path.join(data_path, "judgelm/judgelm_val_5k.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

        with open(os.path.join(data_path, "judgelm/judgelm_val_5k_gpt4.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset_score = [json.loads(line) for line in lines]

        new_dataset = []
        for example, example_score in zip(dataset, dataset_score):
            example["score"] = example_score["score"]

            if example["score"] != [-1, -1]:
                new_dataset.append(example)
        
        dataset = new_dataset

    elif data_type == "pandalm":
        with open(os.path.join(data_path, "pandalm/testset-v1.json"), "r") as fin:
            lines = json.load(fin)

        dataset = []
        for line in lines:
            example = {}
            if line["input"].strip() == "":
                example["question_body"] = line["instruction"]
            else:
                example["question_body"] = line["input"] + \
                    "\n" + line["instruction"]
            example["answer1_body"] = line["response1"]
            example["answer2_body"] = line["response2"]
            if line["annotator1"] == line["annotator2"] or line["annotator1"] == line["annotator2"]:
                example["score"] = line["annotator1"]
            elif line["annotator2"] == line["annotator3"]:
                example["score"] = line["annotator2"]
            else:
                example["score"] = random.choice(
                    [line["annotator1"], line["annotator2"], line["annotator3"]])
            # unify the score to judgelm format
            score_mapping = {"0": [1, 1], "1": [1, 0], "2": [0, 1]}
            example["score"] = score_mapping[str(example["score"])]
            dataset.append(example)

    elif data_type == "auto-j":
        with open(os.path.join(data_path, "auto-j/testdata_pairwise.jsonl"), "r") as fin:
            lines = [json.loads(line.strip()) for line in fin.readlines()]

            dataset = []
            for line in lines:
                example = {"question_body": line["prompt"],
                           "answer1_body": line["response 1"],
                           "answer2_body": line["response 2"],
                           "score": line["label"]}
                # unify the score to judgelm format
                score_mapping = {"0": [1, 0], "1": [0, 1], "2": [1, 1]}
                example["score"] = score_mapping[str(example["score"])]
                dataset.append(example)

        reve_dataset = []
        for example in dataset:
            rev_example = copy.deepcopy(example)
            temp_body = rev_example["answer1_body"]
            rev_example["answer1_body"] = rev_example["answer2_body"]
            rev_example["answer2_body"] = temp_body
            reve_dataset.append(rev_example)

        dataset.extend(reve_dataset)

    elif data_type == "prometheus-ind":
        with open(os.path.join(data_path, "prometheus/feedback_collection_test.json"), "r") as fin:
            lines = json.load(fin)

            dataset = []
            for line in lines:
                example = {}
                example["question_body"] = re.search(
                    r"###The instruction to evaluate:\n[\s\S]+\n\n###Response to evaluate", line["instruction"]).group()[32:-25]
                example["answer_body"] = re.search(
                    r"###Response to evaluate:\n[\s\S]+\n\n###Reference Answer", line["instruction"]).group()[25:-21]
                example["rubric"] = re.search(
                    r"###Score Rubrics:\n\[[\s\S]+\]\nScore 1", line["instruction"]).group()[19:-9]
                example["score"] = line["gpt4_score"]
                dataset.append(example)

    elif data_type == "prometheus-ood":
        with open(os.path.join(data_path, "prometheus/feedback_collection_ood_test.json"), "r") as fin:
            lines = json.load(fin)

            dataset = []
            for line in lines:
                example = {}
                example["question_body"] = re.search(
                    r"###The instruction to evaluate:\n[\s\S]+\n\n###Response to evaluate", line["instruction"]).group()[32:-25]
                example["answer_body"] = re.search(
                    r"###Response to evaluate:\n[\s\S]+\n\n###Reference Answer", line["instruction"]).group()[25:-21]
                example["rubric"] = re.search(
                    r"###Score Rubrics:\n\[[\s\S]+\]\nScore 1", line["instruction"]).group()[19:-9]
                example["score"] = line["gpt4_score"]
                dataset.append(example)

    elif "llmbar" in data_type:
        llmbar_category = data_type.replace("llmbar-", "")
        with open(os.path.join(data_path, "llmbar/"+llmbar_category+"/dataset.json"), "r") as fin:
            lines = json.load(fin)

            dataset = []
            for line in lines:
                example = {}
                example["question_body"] = line["input"]
                example["answer1_body"] = line["output_1"]
                example["answer2_body"] = line["output_2"]
                # unify the score to judgelm format
                score_mapping = {"0": [1, 1], "1": [1, 0], "2": [0, 1]}
                example["score"] = score_mapping[str(line["label"])]
                dataset.append(example)

    return dataset


def parse_predictions(predictions, model_type, data_type, prompt_type):
    def parse_score_judgelm(review, is_pair=True):
        if is_pair:
            try:
                if prompt_type == "cot":
                    score_pair = review.split('\n')[-1]
                else:
                    score_pair = review.split('\n')[0]

                score_pair = score_pair.replace(',', ' ')
                sp = score_pair.split(' ')
                return [float(sp[0]), float(sp[1])]
            except Exception as e:
                return [1.0, 1.0]  # default is Tie
        else:
            try:
                score = review.split('\n')[0].strip()
                return float(score)
            except Exception as e:
                return 5.0  # default is middle score

    def parse_score_pandalm(review, is_pair=True):
        if is_pair:
            if prompt_type == "cot":
                score = review.split('\n')[-1].strip()
            else:
                score = review.split('\n')[0].strip()

            if score == "1":
                return [1, 0]
            elif score == "2":
                return [0, 1]
            elif score == "Tie":
                return [1, 1]
            else:
                return [1, 1]  # default is Tie
        else:
            score = review.split('\n')[0].strip()
            if score in ['1', '2', '3', '4', '5']:
                return int(score)
            else:
                return 3  # default is middle score

    def parse_score_autoj(review, is_pair=True):
        if is_pair:
            review = review.strip()
            pos = review.rfind('final decision is ')
            pred_label = -1
            if pos != -1:
                pred_rest = review[pos +
                                   len('final decision is '):].strip().lower()
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

        else:
            if "Rating: [[" in review:
                pos = review.rfind("Rating: [[")
                pos2 = review.find("]]", pos)
                assert pos != -1 and pos2 != -1
                return float(review[pos + len("Rating: [["):pos2].strip())
            else:
                return 0.0

    def parse_score_prometheus(review, is_pair=False):
        if is_pair:
            try:
                score = review.split('[RESULT]')[1].strip()
                score_pair = score.replace(',', ' ').replace(
                    '\n', ' ').replace('.', ' ')
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

    if model_type == "judgelm":
        is_pair = "prometheus" not in data_type
        pred_scores = [parse_score_judgelm(
            pred, is_pair=is_pair) for pred in predictions]
    elif model_type == "pandalm":
        is_pair = "prometheus" not in data_type
        pred_scores = [parse_score_pandalm(
            pred, is_pair=is_pair) for pred in predictions]
    elif model_type == "auto-j":
        is_pair = "prometheus" not in data_type
        pred_scores = [parse_score_autoj(
            pred, is_pair=is_pair) for pred in predictions]
    elif model_type == "prometheus":
        is_pair = "prometheus" not in data_type
        pred_scores = [parse_score_prometheus(
            pred, is_pair=False) for pred in predictions]
        if "prometheus" not in data_type:
            predictions_a = [pred for pred in pred_scores[0::2]]
            predictions_b = [pred for pred in pred_scores[1::2]]
            pred_scores = [[pred[0], pred[1]]
                           for pred in zip(predictions_a, predictions_b)]

    return pred_scores


def calculate_metrics(y_true_list, y_pred_list, data_type):

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

    if "prometheus" not in data_type:
        y_true = translate_score_to_win_list(y_true_list)
        y_pred = translate_score_to_win_list(y_pred_list)
    else:
        y_true = y_true_list
        y_pred = y_pred_list

    if data_type in ["judgelm", "pandalm"] or "llmbar" in data_type:

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
            'agreement': acc_cnt/len(y_true_frd),
            'consistency': con_cnt/len(y_true_frd),
        }

    elif "prometheus" in data_type:
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


def get_nearest_neighbor(data_type, data_path, dataset, prompt):
    embeddings = torch.load(os.path.join(
        args.data_path, args.data_type, "embedding.bin"))

    neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1, metric="cosine")

    neigh.fit(embeddings)

    neighbors_dist, neighbors_indices = neigh.kneighbors(embeddings)


if __name__ == "__main__":

    args = build_params()
    random.seed(42)

    dataset = build_dataset(args.data_type, args.data_path)

    instruction = create_prompt(
        args.model_type, args.data_type, args.prompt_type)

    if args.prompt_type == "icl":
        with open(os.path.join(args.data_path, args.data_type, "nearest_neighbors_data.json"), "r") as infile:
            nearest_neighbors_data = json.load(infile)

    prompts = []
    answers = []
    for index, example in enumerate(dataset):
        if args.model_type in ["judgelm", "pandalm", "auto-j"]:
            if "prometheus" in args.data_type:
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer_body=example["answer_body"])
                prompts.append(prompt)
            else:

                
                if args.prompt_type == "icl":
                    neighbor_example = nearest_neighbors_data[index]
                    neighbor_example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
                    neighbor_prompt = instruction.format(question_body=neighbor_example["question_body"],
                                                         rubric=neighbor_example["rubric"],
                                                         answer1_body=neighbor_example["answer1_body"],
                                                         answer2_body=neighbor_example["answer2_body"])
                example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])
                if args.prompt_type == "icl":
                    prompt = neighbor_prompt + \
                        neighbor_example["review"] + prompt
                prompts.append(prompt)
                print(prompt)
                print("--------------------------------")
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

    predictions = batched_generation(args.model_name_or_path, prompts,
                                     max_new_token=args.max_new_token,
                                     temperature=args.temperature,
                                     top_p=args.top_p)

    pred_scores = parse_predictions(
        predictions, args.model_type, args.data_type, args.prompt_type)

    if args.logit_file is not None:
        with open(args.logit_file, "w", encoding="utf-8") as fout:
            for pred in pred_scores:
                fout.write(json.dumps(pred)+"\n")

    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print(f"model: {args.model_type}, data: {args.data_type}")
    print(metrics_dicts)
