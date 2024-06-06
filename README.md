# SelfEval

This is the official repository for paper **An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Models are Task-specific Classifiers**.

If you have any quesions, you can contact me with Wechat huanghui20200708.

## ⚡️ Usage
### Preparation
Please refer to the following command to prepare your environment.

```shell
pip install -r requirements.txt
```
Please download pre-trained LLMs and put them under ``models``. Specifically, our study are based on the following four finetuned models:

* [JudgeLM-7B](https://huggingface.co/BAAI/JudgeLM-7B-v1.0)

* [PandaLM-7B](https://huggingface.co/WeOpenML/PandaLM-7B-v1)

* [Prometheus-7b-v1.0](https://huggingface.co/kaist-ai/prometheus-7b-v1.0)

* [Auto-J-13b](https://huggingface.co/GAIR/autoj-13b)

Our study are based on the following data, and we have downloaded the respective testsets and put them under ``data``. 

* [JudgeLM-test](https://huggingface.co/datasets/BAAI/JudgeLM-100K/)

* [PandaLM-test](https://github.com/WeOpenML/PandaLM/blob/main/data/testset-v1.json)

* [Auto-J-test](https://github.com/GAIR-NLP/auto-j/blob/main/data/test/testdata_pairwise.jsonl)

* [Prometheus-test](https://github.com/kaistAI/prometheus/blob/main/evaluation/benchmark/data)

* [MT-bench-test](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)

* [LLMBar-test](https://github.com/princeton-nlp/LLMBar/tree/main/Dataset/LLMBar)

* [Halu-Eval](https://github.com/RUCAIBox/HaluEval/tree/main/data)

* [Toxic-Chat](https://huggingface.co/datasets/lmsys/toxic-chat)

* [SALAD-Bench](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data)

## Evaluate judges on different benchmarks

Run the following script to evaluate the open-sourced judge model on different testsets.

```shell
MODEL_PATH=./models/JudgeLM-7B
MODEL_TYPE=judgelm
PROMPT_TYPE=vanilla
DATA_TYPE=judgelm
python3 -u src/evaluate_judge.py \
    --model-name-or-path $MODEL_PATH \
    --prompt-type $PROMPT_TYPE \
    --model-type $MODEL_TYPE \
    --data-type $DATA_TYPE \
    --max-new-token 1024
```

Run the following script to evaluate GPT-3.5/4 on different testsets.

```shell
MODEL_NAME=gpt-3.5-turbo-0613
PROMPT_TYPE=vanilla
DATA_TYPE=judgelm
python3 -u src/evaluate_gpt.py \
    --model-name $MODEL_NAME \
    --prompt-type $PROMPT_TYPE \
    --data-type $DATA_TYPE \
    --multi-process True \
    --max-new-token 1024 \
    --rewrite-output True
```

## Fine-tune your own judge model
You can train your own judge based on open-source judge data and foundation models.

We support different architecutes: LLaMA, DeBERTa

We also support different architectures: Generation, Regression, Classification

```shell
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATH=./models/llama2-7b-chat
MODEL_TYPE=llama
CLASS_TYPE=generation
DATA_PATH=./data/prometheus/new_feedback_collection.jsonl
OUTPUT_DIR=./output/llama2-generation-prometheus
torchrun --nproc_per_node=4 --master_port=20001 src/finetune.py \
    --model_name_or_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --class_type $CLASS_TYPE \
    --data_path $DATA_PATH \
    --bf16 True \
    --swap_aug_ratio 0.0 \
    --ref_drop_ratio 1.0 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
```

After that, run the following script to evaluate the finetuned judges on different testsets.

```shell
MODEL_PATH=./models/llama2-generation-prometheus
MODEL_TYPE=llama
DATA_TYPE=judgelm
CLASS_TYPE=generation
python -u evaluate_finetuned.py \
    --model-name-or-path $MODEL_NAME \
    --model-type $MODEL_TYPE \
    --data-type $DATA_TYPE \
    --class-type $CLASS_TYPE
```

# 💬 Citation
If you find our work is helpful, please cite as:

```
@misc{huang2024empirical,
      title={An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Models are Task-specific Classifiers}, 
      author={Hui Huang and Yingqi Qu and Jing Liu and Muyun Yang and Tiejun Zhao},
      year={2024},
      eprint={2403.02839},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```