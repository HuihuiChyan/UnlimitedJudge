#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/Auto-J" \
    --prompt-type "icl" \
    --model-type "auto-j" \
    --data-type "judgelm" \
    --max-new-token 1024