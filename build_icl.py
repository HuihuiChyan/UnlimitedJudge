import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm


def build_trainset(data_type, data_path):
    trainset = []
    if data_type == "judgelm":
        with open(os.path.join(data_path, "judgelm/judgelm_train_100k.jsonl"), "r") as fin:
            lines = [json.loads(line.strip()) for line in fin.readlines()]
            for line in lines:
                example = {
                    "question_body": line["question_body"],
                    "answer1_body": line["answer1_body"],
                    "answer2_body": line["answer2_body"],
                    "evaluation": line["text"],
                }
                trainset.append(example)

    elif data_type == "pandalm":
        with open(os.path.join(data_path, "pandalm/train.json"), "r") as fin:
            lines = json.load(fin)

        for line in lines:
            parts = line["input_sequence"].split("###")
            output_parts = line["output_sequence"].split("\n\n### ")

            example = {
                "question_body": "",
                "answer1_body": "",
                "answer2_body": "",
                "winner": "",
                "score": "",
                "reason": "",
                "reference": ""
            }

            # Extract winner from the first line of output_sequence
            winner = output_parts[0].strip()
            example["winner"] = winner
            # Based on winner assign score
            if winner == "1":
                example["score"] = "5 0"
            elif winner == "2":
                example["score"] = "0 5"
            elif winner == "Tie":
                example["score"] = "5 5"
            else:
                # In case of invalid or missing winner
                example["score"] = "0 0"

            # Extract reason and reference from output_parts
            for part in output_parts[1:]:
                part = part.strip()
                if part.startswith("Reason:"):
                    example["reason"] = part[len("Reason:"):].strip()
                if part.startswith("Reference:"):
                    example["reference"] = part[len("Reference:"):].strip()

            for part in parts:
                section = part.strip()
                if section.startswith("Instruction:"):
                    example["question_body"] = section[len(
                        "Instruction:"):].strip()
                elif section.startswith("Response 1:"):
                    example["answer1_body"] = section[len(
                        "Response 1:"):].strip()
                elif section.startswith("Response 2:"):
                    example["answer2_body"] = section[len(
                        "Response 2:"):].strip()

            trainset.append(example)

    return trainset


def concat_sentence(example):
    question_body = str(example["question_body"])

    answer1_body = str(example["answer1_body"])
    answer2_body = str(example["answer2_body"])
    sentence = question_body + answer1_body + answer2_body

    return sentence

def build_demo_instruction(model_type):
    if "gpt" in model_type:
        instruction = """The following is an in-context illustration for the evaluation task:
[Question]
{demo1_question_body}

[The Start of Assistant 1's Answer]
{demo1_answer1_body}
[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{demo1_answer2_body}
[The End of Assistant 2's Answer]

[Your Evaluation]
{demo1_evaluation}

The following is another in-context illustration for the evaluation task:
[Question]
{demo2_question_body}

[The Start of Assistant 1's Answer]
{demo2_answer1_body}
[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{demo2_answer2_body}
[The End of Assistant 2's Answer]

[Your Evaluation]
{demo2_evaluation}
"""
    return instruction

def build_icl(data_type, data_path, model_type, test_samples):

    dataset = []
    trainset = build_trainset(data_type, data_path)
    instruction = build_demo_instruction(model_type)
    for sample in test_samples:
        demo_samples = np.random.choice(trainset, 2, replace=False)
        demonstrations = instruction.format(demo1_question_body=demo_samples[0]["demo1_question_body"],
                                           demo1_answer1_body=demo_samples[0]["demo1_answer1_body"], 
                                           demo1_answer2_body=demo_samples[0]["demo1_answer2_body"], 
                                           demo1_evaluation=demo_samples[0]["demo1_evaluation"], 
                                           demo2_question_body=demo_samples[1]["demo2_question_body"],
                                           demo2_answer1_body=demo_samples[1]["demo2_answer1_body"], 
                                           demo2_answer2_body=demo_samples[1]["demo2_answer2_body"], 
                                           demo2_evaluation=demo_samples[1]["demo2_evaluation"])
        sample["demonstrations"] = demonstrations
        dataset.append(sample)
    return dataset