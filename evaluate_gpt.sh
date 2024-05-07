#!/bin/bash
python3 -u evaluate_gpt.py \
    --model-type "gpt-4" \
    --prompt-type "cot" \
    --data-type "judgelm" \
    --data-path ./data \
    --multi-process True