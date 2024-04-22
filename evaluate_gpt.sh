#!/bin/bash
python -u evaluate_gpt.py \
    --model-type "gpt-4" \
    --data-type "pandalm" \
    --data-path ./data \
    --multi-process True