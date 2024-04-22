import os
import json
import argparse
import random
import re
import copy
import tqdm
import time
import json
import requests
import multiprocessing

from evaluate_judge import build_dataset, parse_predictions, calculate_metrics

def build_params_gpt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("gpt-4", "gpt-3.5"),
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
    # parser.add_argument(
    #     "--top-p",
    #     type=float,
    #     default=1.0,
    #     help="The temperature for sampling.",
    # )
    parser.add_argument(
        "--logit-file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--pool-number",
        type=int,
        default=8,        
    )
    parser.add_argument(
        "--multi-process",
        type=str,
        default="False",        
    )
    args = parser.parse_args()
    return args

def request_gpt4(prompt, temperature, max_new_token):
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
            res = response['choices'][0]['message']['content'].strip()
            break
        except Exception as e:
            print("Exception! The response is "+str(response))
            time.sleep(2)
            continue
    return res

def parse_score_gpt(review, is_pair=True):
    if is_pair:
        try:
            score_pair = review.split('\n')[0]
            score_pair = score_pair.replace(',', ' ')
            sp = score_pair.split(' ')
            return [float(sp[0]), float(sp[1])]
        except Exception as e:
            return [1.0, 1.0] # default is Tie 
    else:
        try:
            score = review.split('\n')[0].strip()
            return float(score)
        except Exception as e:
            return 5.0 # default is middle score

def create_prompt_gpt(data_type):
    if "prometheus" not in data_type:
        # We use JudgeLM prompt directly.
        instruction = """You are a helpful and precise assistant for checking the quality of the answer.
[Question]
{question_body}

[The Start of Assistant 1's Answer]
{answer1_body}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer2_body}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
{rubric} Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:"""
    else:
        # We use Prometheus prompt directly.
        instruction = """You are a fair evaluator language model.

###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{question_body}

###Response to evaluate:
{answer_body}

###Score Rubrics:
{rubric}

###Feedback: [/INST]"""

    return instruction

def gpt4_scoring(prompt):

    prediction = request_gpt4(prompt, temperature=0.0, max_new_token=1024)

    counter.value += 1    
    print(f"gpt4_scoring {counter.value} finished.")

    return prediction

def init(c):
    global counter
    counter = c

if __name__ == "__main__":
    args = build_params_gpt()
    random.seed(42)

    dataset = build_dataset(args.data_type, args.data_path)

    instruction = create_prompt_gpt(args.data_type)
    prompts = []
    answers = []
    for example in dataset:
        if "prometheus" in args.data_type:
            prompt = instruction.format(question_body=example["question_body"],
                                        rubric=example["rubric"],
                                        answer_body=example["answer_body"])
            prompts.append(prompt)
        else:
            example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
            prompt = instruction.format(question_body=example["question_body"],
                                        rubric=example["rubric"],
                                        answer1_body=example["answer1_body"],
                                        answer2_body=example["answer2_body"])
            prompts.append(prompt)

        answers.append(example["score"])

    manager = multiprocessing.Manager()
    counter = manager.Value("counter", 0)
    pool = multiprocessing.Pool(processes=args.pool_number, initializer=init, initargs=(counter,))

    if args.multi_process == "False":
        predictions = [gpt4_scoring(sample) for sample in prompts]
    else:
        predictions = pool.map(gpt4_scoring, prompts)
    
    pred_scores = [parse_score_gpt(p) for p in predictions]

    if args.logit_file is not None:
        with open(args.logit_file, "w", encoding="utf-8") as fout:
            for pred in pred_scores:
                fout.write(json.dumps(pred)+"\n")

    import pdb;pdb.set_trace()
    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print(f"model: {args.model_type}, data: {args.data_type}")
    print(metrics_dicts)