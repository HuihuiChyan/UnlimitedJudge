#!/bin/bash
python -u evaluate_gpt.py \
    --model-type "gpt-3.5-turbo-0613" \
    --prompt-type "icl" \
    --data-type "judgelm" \
    --data-path ./data \
    --multi-process True \
