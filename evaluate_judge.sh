#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/Auto-J-13B" \
    --prompt-type "vanilla" \
    --model-type "auto-j" \
    --data-type "halu-eval-dialogue" \
    --max-new-token 1024