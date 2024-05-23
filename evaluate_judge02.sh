#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/PandaLM-7B" \
    --prompt-type "vanilla" \
    --model-type "pandalm" \
    --data-type "judgelm" \
    --max-new-token 1024