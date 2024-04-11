#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -u evaluate_mtbench.py \
    --model-name-or-path "./models/llama2-7b-chat" \
    --model-type "judgelm" \
    --class-type "generation" \
    --data-type "mt-bench" \
    --eval-batch-size 16 \
    --data-path ./data \
    --max-new-token 1024 