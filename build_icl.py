import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm


def build_trainset(data_type, data_path):
    trainset = {"win": [], "lose": [], "tie": []}
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
                if line['score'][0] > line['score'][1]:
                    trainset['win'].append(example)
                elif line['score'][0] < line['score'][1]:
                    trainset['lose'].append(example)
                else:
                    trainset['tie'].append(example)

    elif data_type == "pandalm":
        with open(os.path.join(data_path, "pandalm/train.json"), "r") as fin:
            lines = json.load(fin)

        for line in lines:
            input_parts = line["input_sequence"].split("###")
            output_parts = line["output_sequence"].split("\n\n### ")

            example = {}

            # Extract winner from the first line of output_sequence
            winner = output_parts[0].strip()
                
            # # Based on winner assign score
            # if winner == "1":
            #     example["evaluation"] = "Response 1 is better."
            # elif winner == "2":
            #     example["evaluation"] = "Response 2 is better."
            # elif winner == "Tie":
            #     example["evaluation"] = "There is a tie."
            # else:
            #     # In case of invalid or missing winner
            #     example["evaluation"] = "There is a tie."

            # # Based on winner assign score
            # if winner == "1":
            #     example["evaluation"] = "So, the final decision is Response 1."
            # elif winner == "2":
            #     example["evaluation"] = "So, the final decision is Response 2."
            # elif winner == "Tie":
            #     example["evaluation"] = "So, the final decision is Tie."
            # else:
            #     # In case of invalid or missing winner
            #     example["evaluation"] = "So, the final decision is Tie."

            # Based on winner assign score
            if winner == "1":
                example["evaluation"] = "1 0"
            elif winner == "2":
                example["evaluation"] = "0 1"
            elif winner == "Tie":
                example["evaluation"] = "1 1"
            else:
                # In case of invalid or missing winner
                example["evaluation"] = "1 1"

            for part in input_parts:
                section = part.strip()
                if section.startswith("Instruction:"):
                    example["question_body"] = section[len("Instruction:"):].strip()
                elif section.startswith("Response 1:"):
                    example["answer1_body"] = section[len("Response 1:"):].strip()
                elif section.startswith("Response 2:"):
                    example["answer2_body"] = section[len("Response 2:"):].strip()

            # Extract reason and reference from output_parts
            for part in output_parts[1:]:
                part = part.strip()
                if part.startswith("Reason:"):
                    reason = part[len("Reason:"):].strip()
                # if part.startswith("Reference:"):
                #     reference = part[len("Reference:"):].strip()

            example["evaluation"] = example["evaluation"] + "\n" + reason
            # example["evaluation"] = reason + "\n" + example["evaluation"]
            
            if winner == "1":
                trainset["win"].append(example)
            elif winner == "2":
                trainset["lose"].append(example)
            else:
                trainset["tie"].append(example)
    return trainset


def build_demo_instruction(model_type, data_type):
    if "gpt" in model_type:
        if data_type == "judgelm":
            instruction = "## Please evaluate the given responses according to the question.\n[Question]\n{question_body}\n\n[The Start of Assistant 1's Answer]\n{answer1_body}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer2_body}\n[The End of Assistant 2's Answer]\n\n[Your Evaluation]\n{evaluation}"
        elif data_type == "pandalm":
            instruction = "[Question]\n{question_body}\n\n[Response 1]\n{answer1_body}\n\n[Response 2]\n{answer2_body}\n\n[Your Evaluation]\n{evaluation}"
    else:
        from build_prompt_judge import create_prompt
        instruction = create_prompt(model_type, data_type)+"{evaluation}"
    return instruction

def build_icl(data_type, data_path, model_type, test_samples, pos_num=2, neg_num=2, tie_num=1):
    dataset = []
    trainset = build_trainset(data_type, data_path)
    instruction = build_demo_instruction(model_type, data_type)
    for sample in test_samples:
        demo_sample_win = np.random.choice(trainset['win'], pos_num, replace=False)
        demo_sample_lose = np.random.choice(trainset['lose'], neg_num, replace=False)
        demo_sample_tie = np.random.choice(trainset['tie'], tie_num, replace=False)
        demo_samples = list(demo_sample_win) + list(demo_sample_lose) + list(demo_sample_tie)
        random.shuffle(demo_samples)
        # demo_samples = [demo_samples[0]]
        demonstrations = []
        for demo_sample in demo_samples:
            demonstrations.append(instruction.format(question_body=demo_sample["question_body"],
                                                     answer1_body=demo_sample["answer1_body"], 
                                                     answer2_body=demo_sample["answer2_body"], 
                                                     evaluation=demo_sample["evaluation"]))
        sample["demonstrations"] = "\n\n".join(demonstrations)
        dataset.append(sample)
    return dataset