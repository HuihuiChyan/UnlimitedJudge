#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/JudgeLM-7B" \
    --model-type "auto-j" \
    --data-type "prometheus-ind" \
    --eval-batch-size 16 \
    --max-new-token 1024