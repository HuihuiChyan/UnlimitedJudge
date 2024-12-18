import json
import torch
import argparse
import random
import vllm

from build_dataset import build_dataset, calculate_metrics
from build_prompt_judge import create_prompt, create_prompt_cot, parse_predictions
from build_icl import build_icl


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
        choices=("vanilla", "cot", "icl"),
        default="vanilla",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus", "llama", "deberta",),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "mt-bench",
                 "halu-eval-summary", "halu-eval-qa", "halu-eval-dialogue", "salad-bench", "toxic-chat",
                 "llmbar-neighbor", "llmbar-natural", "llmbar-gptinst", "llmbar-gptout", "llmbar-manual"),
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
    return parser


@torch.inference_mode()
def batched_generation(
    model_path,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    print("Start load VLLM model!")
    model = vllm.LLM(model=model_path, tensor_parallel_size=1, dtype="bfloat16")
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
    )
    print("VLLM model loaded!")

    pred_list = model.generate(prompts, sampling_params)
    pred_list = [it.outputs[0].text for it in pred_list]

    return pred_list

if __name__ == "__main__":

    random.seed(42)
    parser = build_params()
    args = parser.parse_args()
    
    dataset = build_dataset(args.data_type, args.data_path)

    if args.prompt_type in ["vanilla", "icl"]:
        instruction = create_prompt(args.model_type, args.data_type)
    else:
        instruction = create_prompt_cot(args.model_type, args.data_type)
    
    if args.prompt_type == "icl":
        dataset = build_icl(args.data_type, args.data_path, args.model_type, dataset, pos_num=1, neg_num=1, tie_num=0)

    prompts = []
    answers = []
    for index, example in enumerate(dataset):
        if args.model_type in ["judgelm", "pandalm", "auto-j", "llama-3"]:
            if args.data_type in ["prometheus-ind", "prometheus-ood", "toxic-chat", "halu-eval-summary", "halu-eval-dialogue", "halu-eval-qa"]:
                prompt = instruction.format(question_body=example["question_body"],
                                            answer_body=example["answer_body"])
                prompts.append(prompt)
            else:
                example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])
                prompts.append(prompt)

        elif args.model_type == "prometheus":
            if args.data_type in ["prometheus-ind", "prometheus-ood", "toxic-chat", "halu-eval-summary", "halu-eval-dialogue", "halu-eval-qa"]:
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

    print("Prompt built finished! Sampled prompt:")
    print(prompts[random.randint(0, len(prompts)-1)]+"\n")

    predictions = batched_generation(args.model_name_or_path, prompts,
                                     max_new_token=args.max_new_token,
                                     temperature=args.temperature,
                                     top_p=args.top_p)

    pred_scores = parse_predictions(predictions, args.model_type, args.data_type, args.prompt_type)

    if args.logit_file is not None:
        with open(args.logit_file, "w", encoding="utf-8") as fout:
            for pred in pred_scores:
                fout.write(json.dumps(pred)+"\n")

    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print("**********************************************")
    print(f"Model: {args.model_type}, Data: {args.data_type}, Prompt: {args.prompt_type}")
    print(metrics_dicts)
    print("**********************************************")