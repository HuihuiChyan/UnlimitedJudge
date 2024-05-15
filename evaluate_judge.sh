#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/JudgeLM-7B" \
    --prompt-type "vanilla" \
    --model-type "judgelm" \
    --data-type "salad-bench" \
    --eval-batch-size 16 \
    --max-new-token 1024