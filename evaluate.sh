#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u evaluate.py \
    --model-name-or-path "./models/judgelm" \
    --model-type "judgelm" \
    --data-type "judgelm" \
    --eval-batch-size 16 \
    --max-new-token 32