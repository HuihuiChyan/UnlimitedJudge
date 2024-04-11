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
from build_prompt import create_prompt, create_prompt_aspect

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
            # print(res)
            # print("——————————————————————————————————————————————————————")
            return res
            break
        except Exception as e:
            print(response)
            time.sleep(2)
            continue
    return res

def request_cot(example):
    aspects = {"Helpfulness", "Relevence", "Accuracy", "Depth", "Creativity", "Detailedness"}
    aspect_prompts = {"Helpfulness": "Helpfulness: How useful was the response in providing clear and practical advice or solutions?",
                      "Relevence": "Relevence: Did the response directly answer the question and stay on topic?",
                      "Accuracy": "Accuracy: Was the information provided accurate and well-researched?",
                      "Depth": "Depth: How thorough and detailed was the response in terms of analysis and explanation?",
                      "Creativity": "Creativity: Did the response offer novel or unique perspectives or solutions?",
                      "Detailedness": "Detailedness: Was the response detailed enough to provide necessary context and background information?"}
    instruction = """You are a fair and accurate evaluator.

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
    scores_dict = {"question": example["question"],
                    "answer": example["answer"]}
    for aspect in aspects:
        rubric = aspect_prompts[aspect]
        prompt = instruction.format(question=example["question"],
                                    answer=example["answer"],
                                    rubric=rubric,)
        pred = request_gpt4(prompt, temperature=0.0, max_new_token=512)
        scores_dict[aspect] = parse_score_prometheus(pred)
    counter.value += 1    
    print(f"{counter.value} finished.")
    return scores_dict

def init(c):
    global counter
    counter = c

def main(args):
    assert args.data_type == "chatbot-arena"
    dataset = datasets.load_dataset("parquet", data_files={"train": "data/chatbot-arena/train-00000-of-00001-cced8514c7ed782a.parquet"})

    samples = []
    for sample in dataset["train"]:
        if sample["turn"] == 1:
            example = {"question": sample["conversation_a"][0]["content"],
                        "answer": sample["conversation_a"][1]["content"]}
            samples.append(example)
            example = {"question": sample["conversation_b"][0]["content"],
                        "answer": sample["conversation_b"][1]["content"]}
            samples.append(example)

    random.seed(42)
    random.shuffle(samples)
    samples = samples[:1000]

    pool_number = 16
    manager = multiprocessing.Manager()
    counter = manager.Value("counter", 0)
    pool = multiprocessing.Pool(pool_number, initializer=init, initargs=(counter,))

    output_file = "scores_list.jsonl"
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as fout:
            counter.value = len([line.strip() for line in fout.readlines()])
            samples = samples[counter.value:]
            print("Continue crawling from line "+str(counter.value))

    if args.use_ray == "True":
        batch_size = 50
        with open(output_file, "a", encoding="utf-8") as fout:
            pbar = tqdm.tqdm(total=len(samples))
            for i in range(0, len(samples), batch_size):
                batched_samples = samples[i: i + batch_size]
                results = pool.map(request_cot, batched_samples, pool_number)
                for result in results:
                    fout.write(json.dumps(result)+"\n")
                pbar.update(batch_size)
    else:
        predictions = batched_request_cot(samples,
                                          instruction,
                                          args.temperature,
                                          args.max_new_token,)

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
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "mt-bench", "chatbot-arena"),
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
        "--output-logit-file",
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