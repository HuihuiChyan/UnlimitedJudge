#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python3 -u evaluate.py \
    --model-name-or-path "./models/judgelm" \
    --model-type "judgelm" \
    --data-type "prometheus-ind" \
    --eval-batch-size 16 \
    --max-new-token 32