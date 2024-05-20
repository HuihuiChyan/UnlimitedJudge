#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/Prometheus-7B" \
    --prompt-type "vanilla" \
    --model-type "prometheus" \
    --data-type "toxic-chat" \
    --eval-batch-size 16 \
    --max-new-token 1024