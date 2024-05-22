#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
python3 -u evaluate_judge.py \
    --model-name-or-path "./models/PandaLM-7B" \
    --prompt-type "icl" \
    --model-type "pandalm" \
    --data-type "pandalm" \
    --max-new-token 1024