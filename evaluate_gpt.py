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
import multiprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, StoppingCriteria, GPT2Tokenizer

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

def calculate_metrics(y_true_list, y_pred_list, data_type):
    if "prometheus" not in data_type:
        y_true = translate_score_to_win_list(y_true_list)
        y_pred = translate_score_to_win_list(y_pred_list)
    else:
        y_true = y_true_list
        y_pred = y_pred_list

    if data_type in ["judgelm", "pandalm", "mt-bench", "manual", "neighbor", "gptinst", "gptout", "natural"]:
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
    elif "prometheus" in data_type:
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
            messages = [{"role": "user", "content": prompt}]
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

def gpt4_scoring(sample):
    instruction1 = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Directly output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

Your Evaluation:"""
#     instruction1 = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

# # [User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

# # Your Evaluation:"""

    instruction2 = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. You should focus on who provides a better answer to the second user question. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Directly output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>

Your Evaluation:"""
#     instruction2 = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. You should focus on who provides a better answer to the second user question. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

# <|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>

# Your Evaluation:"""

    if sample["turn"] == 1:
        prompt = instruction1.format(**sample)
    else:
        prompt = instruction2.format(**sample)

    prediction = request_gpt4(prompt, temperature=0.0, max_new_token=1024)
    prediction = parse_score_mtbench(prediction)

    counter.value += 1    
    print(f"gpt4_scoring {counter.value} finished.")

    return prediction

def gpt4_scoring_cot_with_score(sample):

    instruction1 = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. The following are some scores derived for different aspects. Please refer to the scores to compare the assistants' performance. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

Scores for Assistant A:
{scores_a}

Scores for Assistant B:
{scores_b}

Your Evaluation:
"""
#     instruction1 = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. The following are some scores derived for different aspects. Please refer to the scores to compare the assistants' performance. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

# [User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]

# Scores for Assistant A:
# {scores_a}

# Scores for Assistant B:
# {scores_b}

# Your Evaluation:
# """
    instruction2 = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. The following are some scores derived for different aspects. Please refer to the scores to compare the assistants' performance. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>

Scores for Assistant A:
{scores_a}

Scores for Assistant B:
{scores_b}

Your Evaluation:
"""
#     instruction2 = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. The following are some scores derived for different aspects. Please refer to the scores to compare the assistants' performance. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.

# <|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>

# Scores for Assistant A:
# {scores_a}

# Scores for Assistant B:
# {scores_b}

# Your Evaluation:
# """
    def format_score_dict(rubrics, score_dict):
        score_instruction = ""
        for i,rubric in enumerate(rubrics):
            score_instruction = score_instruction + str(i+1) +". " + rubric + ": " + str(score_dict[rubric]) + "\n"
        
        return score_instruction.strip()
    
    sample["scores_a"] = format_score_dict(sample["rubrics"], sample["score_dict_a"])
    sample["scores_b"] = format_score_dict(sample["rubrics"], sample["score_dict_b"])

    if sample["turn"] == 1:
        prompt = instruction1.format(**sample)
    else:
        prompt = instruction2.format(**sample)

    prediction = request_gpt4(prompt, temperature=0.0, max_new_token=1024)
    prediction = parse_score_mtbench(prediction)

    # scores_a = 0
    # for item in list(sample["score_dict_a"].items()):
    #     scores_a += item[1]

    # scores_b = 0
    # for item in list(sample["score_dict_b"].items()):
    #     scores_b += item[1]

    # prediction = [scores_a, scores_b]

    counter.value += 1    
    print(f"gpt4_scoring_cot_with_score {counter.value} finished.")

    return prediction

def gpt4_planning(sample):

    instruction1 = "We want to evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your task is to propose five most important criteria to compare the two responses with respect to the question. Write an evaluation criterion in each line.\nExample Criterion: Does the model recognize the scalability needs of a business or system? \nAnother Example Criterion: Does the model's response account for various abilities, backgrounds, and experiences? \nUser Question: {question}\nEvaluation Criteria:"

    instruction2 = "We want to evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your task is to propose five most important criteria to compare the two responses with respect to the question. Write an evaluation criterion in each line.\nExample Criterion: Does the model recognize the scalability needs of a business or system? \nAnother Example Criterion: Does the model's response account for various abilities, backgrounds, and experiences? \nUser Question Turn1: {question_1}\nUser Question Turn2: {question_2}\nEvaluation Criteria:"

    def filter_number(rubric):
        if re.match(r"[0-9]+\.", rubric):
            rubric = rubric[2:].strip()
        return rubric

    try:
        if sample["turn"] == 1:
            prompt = instruction1.format(question=sample["question"])
        else:
            prompt = instruction2.format(question_1=sample["question_1"], question_2=sample["question_2"])

        rubrics = request_gpt4(prompt, temperature=0.3, max_new_token=1024)
        rubrics = rubrics.strip().replace("\n\n", "\n").split("\n")
        rubrics = [filter_number(r) for r in rubrics]

        assert len(rubrics) == 5
        sample["rubrics"] = rubrics

    except Exception as e:
        print("Exception! " + str(e))
        sample["rubrics"] = ["How useful was the response in providing clear and practical advice or solutions?",
                             "Did the response directly answer the question and stay on topic?",
                             "Was the information provided accurate and well-researched?",
                             "How thorough and detailed was the response in terms of analysis and explanation?",
                             "Did the response offer novel or unique perspectives or solutions?",
                             "Was the response detailed enough to provide necessary context and background information?"]

    counter.value += 1    
    print(f"gpt4_planning {counter.value} finished.")

    return sample

def gpt4_aspecting(sample):

    instruction1 = """You are a fair and accurate evaluator.

###Task Description:
A conversation between a user and an AI assistant, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the assistant response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction and response to evaluate:
[User Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]

###Score Rubrics:
{rubric}

###Feedback: """
    instruction2 = """You are a fair and accurate evaluator.

###Task Description:
A conversation between a user and an AI assistant, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the assistant response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction and response to evaluate:
[User Question]\nUser: {question_1} Assistant: {answer_1} User: {question_2}\n\n[The Start of Assistant's Answer]\n{answer_2}\n[The End of Assistant's Answer]

###Score Rubrics:
{rubric}

###Feedback: """

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

    if "rubrics" not in sample.keys():
        sample["rubrics"] = ["Helpfulness: How useful was the response in providing clear and practical advice or solutions?",
                             "Relevence: Did the response directly answer the question and stay on topic?",
                             "Accuracy: Was the information provided accurate and well-researched?",
                             "Depth: How thorough and detailed was the response in terms of analysis and explanation?",
                             "Creativity: Did the response offer novel or unique perspectives or solutions?",
                             "Detailedness: Was the response detailed enough to provide necessary context and background information?"]

    sample["score_dict_a"] = {}
    sample["score_dict_b"] = {}

    for rubric in sample["rubrics"]:
        if sample["turn"] == 1:
            prompt = instruction1.format(question=sample["question"],
                                         answer=sample["answer_a"],
                                         rubric=rubric,)
        else:
            prompt = instruction2.format(question_1=sample["question_1"],
                                         answer_1=sample["answer_a_1"],
                                         question_2=sample["question_2"],
                                         answer_2=sample["answer_a_2"],
                                         rubric=rubric,)
        prediction = request_gpt4(prompt, temperature=0.0, max_new_token=1024)
        sample["score_dict_a"][rubric] = parse_score_prometheus(prediction)

    for rubric in sample["rubrics"]:
        if sample["turn"] == 1:
            prompt = instruction1.format(question=sample["question"],
                                         answer=sample["answer_b"],
                                         rubric=rubric,)
        else:
            prompt = instruction2.format(question_1=sample["question_1"],
                                         answer_1=sample["answer_b_1"],
                                         question_2=sample["question_2"],
                                         answer_2=sample["answer_b_2"],
                                         rubric=rubric,)
        prediction = request_gpt4(prompt, temperature=0.0, max_new_token=1024)
        sample["score_dict_b"][rubric] = parse_score_prometheus(prediction)

    counter.value += 1    
    print(f"gpt4_aspecting {counter.value} finished.")

    return sample

@torch.inference_mode()
def prometheus_scoring(
    model_path,
    samples,
    eval_batch_size=8,
):

    # instruction1 = "You are a helpful and precise assistant for checking the quality of the answer. We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[The Start of Response]\n{answer}\n\n[The End of Response]\n\n[Feedback]"
    # instruction2 = "You are a helpful and precise assistant for checking the quality of the answer. We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\nUser: {question_1} Assistant: {answer_1} User: {question_2}\n\n[The Start of Response]\n{answer_2}\n\n[The End of Response]\n\n[Feedback]"
    instruction1 = "<s>[INST]You are a helpful and precise assistant for checking the quality of the answer. We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[Response]\n{answer}\n\n[/INST]"
    instruction2 = "<s>[INST]You are a helpful and precise assistant for checking the quality of the answer. We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question_1}\n\n[Response]\n{answer_1}\n\n[Question]\n{question_2}\n\n[Response]\n{answer_2}\n\n[/INST]"

    prompts = []
    for sample in samples:
        if "rubrics" not in sample.keys():
            sample["rubrics"] = ["Helpfulness: How useful was the response in providing clear and practical advice or solutions?",
                                 "Relevence: Did the response directly answer the question and stay on topic?",
                                 "Accuracy: Was the information provided accurate and well-researched?",
                                 "Depth: How thorough and detailed was the response in terms of analysis and explanation?",
                                 "Creativity: Did the response offer novel or unique perspectives or solutions?"]
        for rubric in sample["rubrics"]:
            if ":" in rubric:
                rubric = rubric.split(":")[1].strip()
            if sample["turn"] == 1:
                prompt = instruction1.format(question=sample["question"],
                                             answer=sample["answer_a"],
                                             rubric=rubric,)
            else:
                prompt = instruction2.format(question_1=sample["question_1"],
                                             answer_1=sample["answer_a_1"],
                                             question_2=sample["question_2"],
                                             answer_2=sample["answer_a_2"],
                                             rubric=rubric,)
            prompts.append(prompt)

    for sample in samples:
        for rubric in sample["rubrics"]:
            if ":" in rubric:
                rubric = rubric.split(":")[1].strip()
            if sample["turn"] == 1:
                prompt = instruction1.format(question=sample["question"],
                                             answer=sample["answer_b"],
                                             rubric=rubric,)
            else:
                prompt = instruction2.format(question_1=sample["question_1"],
                                             answer_1=sample["answer_b_1"],
                                             question_2=sample["question_2"],
                                             answer_2=sample["answer_b_2"],
                                             rubric=rubric,)
            prompts.append(prompt)
                                              
    print("start load prometheus model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print("prometheus model loaded")

    pred_list = []
    pbar = tqdm.tqdm(total=len(prompts))
    for i in range(0, len(prompts), eval_batch_size):
        batched_prompt = prompts[i:i+eval_batch_size]

        input_ids = tokenizer(
            batched_prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).input_ids
        logits = model(input_ids=input_ids.cuda(), 
                       attention_mask=input_ids.ne(tokenizer.pad_token_id).cuda())
        
        pbar.update(len(batched_prompt))

        for logit in logits.logits.squeeze(-1).tolist():
            pred_list.append(round(logit, 2))

    pred_scores_a = pred_list[:len(pred_list)//2]
    pred_scores_b = pred_list[len(pred_list)//2:]

    rubric_num = len(sample["rubrics"])
    
    score_dict = {}
    for i in range(len(pred_scores_a)):
        sample_id = i//rubric_num
        rubric = samples[sample_id]["rubrics"][i%rubric_num]
        score_dict[rubric] = pred_scores_a[i]
        if (i+1)%rubric_num == 0:
            samples[sample_id]["score_dict_a"] = score_dict
            score_dict = {}

    score_dict = {}
    for i in range(len(pred_scores_b)):
        sample_id = i//rubric_num
        rubric = samples[sample_id]["rubrics"][i%rubric_num]
        score_dict[rubric] = pred_scores_b[i]
        if (i+1)%rubric_num == 0:
            samples[sample_id]["score_dict_b"] = score_dict
            score_dict = {}

    return samples

def init(c):
    global counter
    counter = c

def main(args):
    if args.data_type == "judgelm":
        with open(os.path.join(args.data_path, "judgelm/judgelm_val_5k.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

        with open(os.path.join(args.data_path, "judgelm/judgelm_val_5k_gpt4.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset_score = [json.loads(line) for line in lines]

        for example, example_score in zip(dataset, dataset_score):
            example["score"] = example_score["score"]
            example["answer_a"] = example["answer1_body"]
            example["answer_b"] = example["answer2_body"]
            example["question"] = example["question_body"]
            example["turn"] = 1

    elif args.data_type == "pandalm":
        with open(os.path.join(args.data_path, "pandalm/testset-v1.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

    elif args.data_type == "auto-j":
        with open(os.path.join(args.data_path, "auto-j/testdata_pairwise_conv.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

        random.seed(42)
        random.shuffle(dataset)
        
        reve_dataset = []
        for example in dataset:
            rev_example = copy.deepcopy(example)
            temp_body = rev_example["answer_a"]
            rev_example["answer_a"] = rev_example["answer_b"]
            rev_example["answer_b"] = temp_body
            reve_dataset.append(rev_example)

        dataset.extend(reve_dataset)

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
        dataset = samples

    elif args.data_type == "neighbor":
        with open(os.path.join(args.data_path, "Neighbor/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

            for example in dataset:
                example["answer_a"] = example["answer1_body"]
                example["answer_b"] = example["answer2_body"]
                example["question"] = example["question_body"]
                example["turn"] = 1

    elif args.data_type == "natural":
        with open(os.path.join(args.data_path, "Natural/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

            for example in dataset:
                example["answer_a"] = example["answer1_body"]
                example["answer_b"] = example["answer2_body"]
                example["question"] = example["question_body"]
                example["turn"] = 1

    elif args.data_type == "manual":
        with open(os.path.join(args.data_path, "Manual/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

            for example in dataset:
                example["answer_a"] = example["answer1_body"]
                example["answer_b"] = example["answer2_body"]
                example["question"] = example["question_body"]
                example["turn"] = 1

    elif args.data_type == "gptinst":
        with open(os.path.join(args.data_path, "GPTInst/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

            for example in dataset:
                example["answer_a"] = example["answer1_body"]
                example["answer_b"] = example["answer2_body"]
                example["question"] = example["question_body"]
                example["turn"] = 1

    elif args.data_type == "gptout":
        with open(os.path.join(args.data_path, "GPTOut/dataset.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

            for example in dataset:
                example["answer_a"] = example["answer1_body"]
                example["answer_b"] = example["answer2_body"]
                example["question"] = example["question_body"]
                example["turn"] = 1
    
    answers = [example["score"] for example in dataset]

    manager = multiprocessing.Manager()
    counter = manager.Value("counter", 0)
    pool = multiprocessing.Pool(processes=args.pool_number, initializer=init, initargs=(counter,))

    if args.cot_augmentation == "self-cot":
        if args.multi_process == "False":
            samples = [gpt4_planning(sample) for sample in dataset]
            samples = [gpt4_aspecting(sample) for sample in samples]
            predictions = [gpt4_scoring_cot_with_score(sample) for sample in samples]
        else:
            samples = pool.map(gpt4_planning, dataset)
            counter.value = 0
            samples = pool.map(gpt4_aspecting, samples)
            counter.value = 0
            predictions = pool.map(gpt4_scoring_cot_with_score, samples)
    elif args.cot_augmentation == "prometheus":
        if args.multi_process == "False":
            samples = [gpt4_planning(sample) for sample in dataset]
            samples = prometheus_scoring(args.model_name_or_path, samples)
            predictions = [gpt4_scoring_cot_with_score(sample) for sample in samples]
        else:
            samples = pool.map(gpt4_planning, dataset)
            samples = prometheus_scoring(args.model_name_or_path, samples)
            counter.value = 0
            predictions = pool.map(gpt4_scoring_cot_with_score, samples)
    elif args.cot_augmentation == "False":
        if args.multi_process == "False":
            predictions = [gpt4_scoring(sample) for sample in dataset]
        else:
            predictions = pool.map(gpt4_scoring, dataset)
    elif args.cot_augmentation == "prometheus-noplan":
        if args.multi_process == "False":
            samples = prometheus_scoring(args.model_name_or_path, dataset)
            predictions = [gpt4_scoring_cot_with_score(sample) for sample in samples]
        else:
            samples = prometheus_scoring(args.model_name_or_path, dataset)
            predictions = pool.map(gpt4_scoring_cot_with_score, samples)
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
        "--data-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--multi-process",
        type=str,
        choices=("True", "False"),
        default="False"
    )
    parser.add_argument(
        "--cot-augmentation",
        type=str,
        choices=("prometheus", "self-cot", "False", "prometheus-noplan"),
        default="False"
    )
    parser.add_argument(
        "--pool-number",
        type=int,
        default=8      
    )
    args = parser.parse_args()

    main(args)