#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python3 -u src/evaluate_judge.py \
    --model-name-or-path "./models/Auto-J-13b" \
    --prompt-type "vanilla" \
    --model-type "auto-j" \
    --data-type "toxic-chat" \
    --max-new-token 1024