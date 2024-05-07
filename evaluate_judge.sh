#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,2
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/JudgeLM-7B" \
    --prompt-type "icl" \
    --model-type "judgelm" \
    --data-type "pandalm" \
    --eval-batch-size 16 \
    --max-new-token 1024