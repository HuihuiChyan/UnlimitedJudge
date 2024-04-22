#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/Auto-J-13B" \
    --model-type "auto-j" \
    --data-type "prometheus-ood" \
    --eval-batch-size 16 \
    --max-new-token 1024