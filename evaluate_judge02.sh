#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/JudgeLM-7B" \
    --model-type "auto-j" \
    --data-type "prometheus-ood" \
    --eval-batch-size 16 \
    --max-new-token 1024