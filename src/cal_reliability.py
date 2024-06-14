from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from tqdm import tqdm
import os
import gc
import json
import vllm
import copy
from build_prompt_judge import create_prompt, parse_predictions
from build_dataset import build_dataset, calculate_metrics
from evaluate_judge import build_params


@torch.inference_mode()
def get_multi_answer(
    model_path,
    prompts,
    max_new_token=2048,
    temperature=0.1,
    top_p=1.0,
):
    print("Start load VLLM model!")
    model = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), dtype="bfloat16", gpu_memory_utilization=0.8)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
    )
    print("VLLM model loaded!")

    tokenizer = model.get_tokenizer()
    MAX_LEN = model.llm_engine.model_config.max_model_len - 512
    prompt_ids = [tokenizer.encode(prompt)[-MAX_LEN:] for prompt in prompts]

    pred_list = model.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)

    prompt_token_ids = [it.prompt_token_ids for it in pred_list]
    output_token_ids = [it.outputs[0].token_ids for it in pred_list]

    prefix_lens = [len(prompt_ids) for prompt_ids in prompt_token_ids]
    target_lens = [len(output_ids) for output_ids in output_token_ids]

    output_tokens = [it.outputs[0].text for it in pred_list]

    output_ids = [ids[0]+ids[1] for ids in zip(prompt_token_ids, output_token_ids)]

    return output_tokens, prefix_lens, target_lens, output_ids

@torch.inference_mode()
def get_single_evaluation(
    model,
    output_ids_ori,
    prefix_len,
    target_len,
):
    # output_ids_ori: The predicted ids consist of both instruction and response, shape is [1, sequence_len]
    # prefix_len: The length of the instruction part
    # target_len: The length of the response part

    assert output_ids_ori.size()[0] == 1
    output_ids_ori = output_ids_ori.to(model.device)

    input_ids = copy.deepcopy(output_ids_ori)
    output_ids = output_ids_ori.clone()
    output_ids[0][:prefix_len] = -100  # instruction masking
    outputs = model(
        input_ids=torch.as_tensor(input_ids),
        labels=output_ids,
        output_hidden_states=True,
        output_attentions=True,
    )
    # the predict ids should be shifted left
    shifted_input_ids = torch.roll(input_ids, shifts=-1)
    logprobs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)

    logprobs_variance = torch.var(logprobs, dim=-1)
    logprobs_variance[output_ids == -100] = 0  # instruction masking
    # averaged on target length
    evaluation_var = logprobs_variance.sum(-1)[0] / target_len

    logprobs[output_ids == -100] = 0  # instruction masking
    # The original entropy has a minus sign, but we remove it to keep the positive correlation
    logprobs_entropy = torch.mean(logprobs * outputs["logits"], dim=-1)
    # averaged on target length
    evaluation_ent = logprobs_entropy.sum(-1)[0] / target_len

    evaluation_logit = torch.gather(logprobs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    evaluation_logit = evaluation_logit.sum(-1)[0] / target_len

    return {"logit": evaluation_logit, "entropy": evaluation_ent, "variance": evaluation_var}

if __name__ == "__main__":
    parser = build_params()
    parser.add_argument("--cali-model-name-or-path", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()
    random.seed(42)

    estimation_mode = "logprobs-both"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = build_dataset(args.data_type)
    print(f"Loaded dataset from {args.data_path}")
    print(f"The length is {len(dataset)}")

    instruction = create_prompt(args.model_type, args.data_type)

    prompts = []
    answers = []
    for index, example in enumerate(dataset):
        if args.model_type in ["judgelm", "pandalm", "auto-j"]:
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
    
    predictions, prefix_lens, target_lens, output_ids = get_multi_answer(args.model_name_or_path, prompts, args.max_new_token)

    pred_scores = parse_predictions(predictions, args.model_type, args.data_type, args.prompt_type)

    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print("**********************************************")
    print(f"Model: {args.model_type}, Data: {args.data_type}, Prompt: {args.prompt_type}")
    print(metrics_dicts)
    print("**********************************************")

    with open(args.logit_file, "w", encoding="utf-8") as fout:
        for pred in pred_scores:
            fout.write(json.dumps(pred)+"\n")

    # 初始化结果字典
    results = {"Entropy": [], "Variance": []}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path).half().to(device)
    model.eval()

    for i in tqdm(range(len(predictions)), desc="Calculating reliability score"):
        evaluation = get_single_evaluation(
            model,
            torch.as_tensor([output_ids[i]]),
            prefix_lens[i],
            target_lens[i],
        )
        entropy = evaluation["entropy"]
        variance = evaluation["variance"]
        # 将结果添加到字典中
        results["Entropy"].append(entropy.item() if isinstance(
            entropy, torch.Tensor) else entropy)
        results["Variance"].append(variance.item() if isinstance(
            variance, torch.Tensor) else variance)

    if args.cali_model_name_or_path is not None:
        results["entropy_cali"] = []
        results["variance_cali"] = []
        model = AutoModelForCausalLM.from_pretrained(
            args.cali_model_name_or_path).half().to(device)
        model.eval()

        for i in tqdm(range(len(predictions)), desc="Calculating calibration reliability score"):
            evaluation = get_single_evaluation(
                model,
                torch.as_tensor([output_ids[i]]),
                prefix_lens[i],
                target_lens[i],
            )
            entropy = evaluation["entropy"]
            variance = evaluation["variance"]
            # 将结果添加到字典中
            results["entropy_cali"].append(entropy.item() if isinstance(
                entropy, torch.Tensor) else entropy)
            results["variance_cali"].append(variance.item() if isinstance(
                variance, torch.Tensor) else variance)

    # 将所有结果写入 JSON 文件
    with open(args.output_file, "w") as file_out:
        json.dump(results, file_out, indent=4)

    print(f"All reliability scores have been saved to {args.output_file}.")
