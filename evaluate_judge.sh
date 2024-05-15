#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/Auto-J-13B" \
    --prompt-type "vanilla" \
    --model-type "auto-j" \
    --data-type "halu-eval-summary" \
    --eval-batch-size 16 \
    --max-new-token 1024