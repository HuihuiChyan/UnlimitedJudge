#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 -u evaluate.py \
    --model-name-or-path "./models/Auto-J-13B" \
    --model-type "judgelm" \
    --data-type "auto-j" \
    --eval-batch-size 16 \
    --max-new-token 32