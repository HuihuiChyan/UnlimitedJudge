# Unlimited Judge

This is the official repository for paper **On the Limitations of Fine-tuned Judge Models for LLM Evaluation**.

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

To obtain the calibrated reliability scores, or to finetune your own judge model for comparison, you also need to download the following base models:

* [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.3)

* [Llama-7B](https://huggingface.co/huggyllama/llama-7b)

* [Llama2-chat-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

* [Llama2-chat-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)

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
python -u src/evaluate_finetuned.py \
    --model-name-or-path $MODEL_NAME \
    --model-type $MODEL_TYPE \
    --data-type $DATA_TYPE \
    --class-type $CLASS_TYPE
```


## Obtain the reliability score
Run the following script to obtain the confidence scores.

```shell
MODEL_PATH=./models/JudgeLM-7B
BASE_MODEL_PATH=./models/Vicuna-7B
MODEL_TYPE=judgelm
DATA_TYPE=salad-bench

python3 -u src/cal_reliability.py \
    --model-name-or-path $MODEL_PATH \
    --cali-model-name-or-path $BASE_MODEL_PATH \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --max-new-token 1024 \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"

```

After that, you can run the following script to perform *CascadedEval*, by allocating the less confident samples to GPT-4 for re-evaluation.

```shell
MODEL_TYPE="judgelm"
DATA_TYPE="salad-bench"
python3 -u src/cascaded_eval.py \
    --data-type $DATA_TYPE \
    --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json" \
    --logit-file-gpt "outputs/${DATA_TYPE}-gpt-4-turbo-128k-vanilla.jsonl"
```

You can also run the following script to evaluate the effectiveness of the scores, by bucketing the testset according to the score:

```shell
MODEL_TYPE=judgelm
DATA_TYPE=salad-bench
python3 -u src/evaluate_reliability.py \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"
```
