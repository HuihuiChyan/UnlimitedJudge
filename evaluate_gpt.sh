#!/bin/bash
python3 -u evaluate_gpt.py \
    --model-type "gpt-4-turbo-1106-preview" \
    --prompt-type "vanilla" \
    --data-type "pandalm" \
    --data-path ./data \
    --multi-process True