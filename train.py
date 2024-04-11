import json
import copy
import torch
import pathlib
import numpy as np
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List

import transformers
from transformers import Trainer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_type: str = field(default="deberta")
    class_type: str = field(default="regression")
    complex_prompt: str = field(default="False")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    swap_aug_ratio: float = 0.0
    ref_drop_ratio: float = 1.0

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def swap_first_two_integers(s):
    # find the first space
    first_space_index = s.find(' ')
    if first_space_index != -1:
        # find the second space
        second_space_index = s.find('\n', first_space_index + 1)
        if second_space_index != -1:
            # swap the first two integers
            new_s = s[first_space_index + 1:second_space_index] + ' ' + s[:first_space_index] + '\n' + s[second_space_index + 1:]
            return new_s
    return s

def create_prompt(class_type, model_type, is_prometheus=False):
    instruction = {}
    if class_type  == "classification":
        if model_type == "deberta":
            instruction["wtref"] = "{rubric}[SEP]Question: {question}[SEP]Reference Answer: {reference}[SEP]Response 1: {answer1}[SEP]Response 2: {answer2}" 
            instruction["noref"] = "{rubric}[SEP]Question: {question}[SEP]Response 1: {answer1}[SEP]Response 2: {answer2}"
        else:
            instruction["wtref"] = "[INST]We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Response 1]\n{answer1}\n\n[The End of Response 1]\n\n[The Start of Response 2]\n{answer2}\n\n[The End of Response 2]\n\n[/INST]"
            instruction["noref"] = "[INST]We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[The Start of Response 1]\n{answer1}\n\n[The End of Response 1]\n\n[The Start of Response 2]\n{answer2}\n\n[The End of Response 2]\n\n[/INST]"

    elif class_type == "regression":
        if model_type == "deberta":
            instruction["wtref"] = "{rubric}[SEP]Question: {question}[SEP]Reference Answer: {reference}[SEP]Response: {answer}" 
            instruction["noref"] = "{rubric}[SEP]Question: {question}[SEP]Response: {answer}"
        else:
            instruction["wtref"] = "[INST]We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Response]\n{answer}\n\n[The End of Response]\n\n[/INST]"
            instruction["noref"] = "[INST]We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[The Start of Response]\n{answer}\n\n[The End of Response]\n\n[/INST]"

    elif class_type == "generation":
        if model_type == "deberta":
            raise Exception("Do not use deberta in generation mode!")
        if is_prometheus:
            instruction["wtref"] = "[INST]We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Response]\n{answer}\n\n[The End of Response]\n\n[Feedback]\n[/INST]"
            instruction["noref"] = "[INST]We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[The Start of Response]\n{answer}\n\n[The End of Response]\n\n[Feedback]\n[/INST]"                
        else:
            instruction["wtref"] = "[INST]We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n[The Start of Response 1]\n{answer1}\n\n[The End of Response 1]\n\n[The Start of Response 2]\n{answer2}\n\n[The End of Response 2]\n\n[Feedback]\n"
            instruction["noref"] = "[INST]We would like to request your feedback on the response to the user question displayed below.\n{rubric}\n[Question]\n{question}\n\n[The Start of Response 1]\n{answer1}\n\n[The End of Response 1]\n\n[The Start of Response 2]\n{answer2}\n\n[The End of Response 2]\n\n[Feedback]\n"

    return instruction

def format_instruction(instruction, example, class_type):
    if "rubric" not in example.keys():
        example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of the response."
    if "reference" not in example.keys():
        example["reference"] = {"text": None}
    if class_type == "classification":
        prompt = instruction.format(question_body=example["question_body"],
                                    rubric=example["rubric"],
                                    reference=example["reference"]['text'],
                                    answer1_body=example["answer1_body"],
                                    answer2_body=example["answer2_body"])
        return prompt
    elif class_type in ["regression", "generation"]:
        if "answer2_body" in example.keys():
            prompt_a = instruction.format(question_body=example["question_body"],
                                          rubric=example["rubric"],
                                          reference=example["reference"]['text'],
                                          answer_body=example["answer1_body"])
            prompt_b = instruction.format(question_body=example["question_body"],
                                          rubric=example["rubric"],
                                          reference=example["reference"]['text'],
                                          answer_body=example["answer2_body"])
            return [prompt_a, prompt_b]
        else:
            prompt = instruction.format(question_body=example["question_body"],
                                        rubric=example["rubric"],
                                        reference=example["reference"]['text'],
                                        answer_body=example["answer_body"])
            return prompt

def preprocess(
    sources,
    tokenizer,
    class_type,
    instruction,
    swap_aug_ratio,
    ref_drop_ratio,
) -> Dict:

    # Apply prompt templates
    conversations = []
    labels = []
    for i, source in enumerate(sources):
        if ref_drop_ratio > 0 and np.random.rand() < ref_drop_ratio:
            instruction = instruction["noref"]
        else:
            instruction = instruction["wtref"]
            source["score"] = source["score_w_reference"]
            source["text"] = source["text_w_reference"]

        if swap_aug_ratio > 0 and np.random.rand() < swap_aug_ratio:
            if "answer2_body" in source.keys():
                source["answer2_body"], source["answer1_body"] = source["answer1_body"], source["answer2_body"]
                source['score'] = [source['score'][1], source['score'][0]]
                source['text'] = swap_first_two_integers(source['text'])
                source['text'] = source['text'].replace('Assistant 1', 'Assistant X')
                source['text'] = source['text'].replace('Assistant 2', 'Assistant 1')
                source['text'] = source['text'].replace('Assistant X', 'Assistant 2')

        prompt = format_instruction(instruction, source, class_type)
        if class_type == "classification":
            conversations.append(prompt)
            if source['score'][0] > source['score'][1]:
                labels.append(1)
            elif source['score'][1] > source['score'][0]:
                labels.append(2)
            else:
                labels.append(0)
        elif class_type == "regression" and "answer2_body" in source.keys():
            conversations.append(prompt[0])
            labels.append(source['score'][0])
            conversations.append(prompt[1])
            labels.append(source['score'][1])
        elif class_type == "regression" and "answer2_body" not in source.keys():
            conversations.append(prompt)
            labels.append(source['score'])
        elif class_type == "generation":
            conversations.append(prompt)
            labels.append(source['text'])

    return conversations, labels

class LazySupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, instruction, class_type, ref_drop_ratio, swap_aug_ratio):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.instruction = instruction
        self.class_type = class_type
        self.ref_drop_ratio = ref_drop_ratio
        self.swap_aug_ratio = swap_aug_ratio

class MultiLazySupervisedDataset(LazySupervisedDataset):
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # if i in self.cached_data_dict:
        #     return self.cached_data_dict[i]
        
        conversations, labels = preprocess(
            [self.raw_data[i]],
            self.tokenizer,
            self.class_type,
            self.instruction,
            self.swap_aug_ratio,
            self.ref_drop_ratio,
        )
        # Tokenize conversations
        tokenized = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        ret = dict(
            input_ids=tokenized["input_ids"][0],
            labels=labels[0],
            attention_mask=tokenized["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret
        return ret

class SingleLazySupervisedDataset(LazySupervisedDataset):

    def __len__(self):
        return len(self.raw_data) * 2

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # if i in self.cached_data_dict:
        #     return self.cached_data_dict[i]

        conversations, labels = preprocess(
            [self.raw_data[i//2]],
            self.tokenizer,
            self.class_type,
            self.instruction,
            self.swap_aug_ratio,
            self.ref_drop_ratio,
        )
        # Tokenize conversations
        tokenized = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        if i % 2 == 0:
            ret = dict(
                input_ids=tokenized["input_ids"][0],
                labels=labels[0],
                attention_mask=tokenized["attention_mask"][0],
            )
        else:
            ret = dict(
                input_ids=tokenized["input_ids"][1],
                labels=labels[1],
                attention_mask=tokenized["attention_mask"][1],
            )

        self.cached_data_dict[i] = ret

        return ret

class GenerationLazySupervisedDataset(LazySupervisedDataset):
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # if i in self.cached_data_dict:
        #     return self.cached_data_dict[i]

        conversations, labels = preprocess(
            [self.raw_data[i]],
            self.tokenizer,
            self.class_type,
            self.instruction,
            self.swap_aug_ratio,
            self.ref_drop_ratio,
        )
        conv_labels = [conversations[0] + labels[0]]
        # Tokenize conversations
        tokenized = self.tokenizer(
            conv_labels,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        labels = copy.deepcopy(tokenized.input_ids)
        source_tokenized = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_len = source_tokenized.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        labels[0][:source_len] = IGNORE_TOKEN_ID

        ret = dict(
            input_ids=tokenized.input_ids[0],
            labels=labels[0],
            attention_mask=tokenized.attention_mask[0],
        )
        self.cached_data_dict[i] = ret

        return ret

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, class_type, instruction, ref_drop_ratio, swap_aug_ratio,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if class_type == "classification":
        dataset_cls = MultiLazySupervisedDataset
    elif class_type == "regression":
        if "prometheus" in data_args.data_path:
            dataset_cls = MultiLazySupervisedDataset
        else:
            dataset_cls = SingleLazySupervisedDataset
    elif class_type == "generation":
        dataset_cls = GenerationLazySupervisedDataset

    rank0_print("Loading data...")

    with open(data_args.data_path, "r") as fin:
        train_data = [json.loads(line) for line in fin.readlines()]

    train_dataset = dataset_cls(train_data,
        tokenizer=tokenizer,
        instruction=instruction,
        class_type=class_type,
        ref_drop_ratio=ref_drop_ratio,
        swap_aug_ratio=swap_aug_ratio,
    )

    rank0_print("Loading data finished")

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.use_cache = False

    if model_args.class_type == "regression":
        config.problem_type = "regression"
        config.num_labels = 1
        MODEL_CLS = AutoModelForSequenceClassification
    elif model_args.class_type == "classification":
        config.problem_type = "single_label_classification"
        config.num_labels = 3
        MODEL_CLS = AutoModelForSequenceClassification
    elif model_args.class_type == "generation":
        MODEL_CLS = AutoModelForCausalLM

    model = MODEL_CLS.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    model = model.to(torch.bfloat16) if training_args.bf16 else model.to(torch.float16)
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = model.config.eos_token_id
    if "prometheus" in data_args.data_path:
        is_prometheus = True
    else:
        is_prometheus = False

    instruction = create_prompt(model_args.class_type, model_args.model_type, is_prometheus)

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        class_type=model_args.class_type,
        instruction=instruction,
        ref_drop_ratio=data_args.ref_drop_ratio,
        swap_aug_ratio=data_args.swap_aug_ratio,
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            trainer.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            trainer.save_model()


if __name__ == "__main__":
    train()