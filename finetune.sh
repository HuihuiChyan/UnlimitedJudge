#!/bin/bash
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20001 src/finetune.py \
    --model_name_or_path ./models/llama2-7b-chat \
    --model_type "llama" \
    --class_type "generation" \
    --data_path ./data/prometheus/new_feedback_collection.jsonl \
    --bf16 True \
    --swap_aug_ratio 0.0 \
    --ref_drop_ratio 1.0 \
    --output_dir ./output/llama2-generation-prometheus \
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