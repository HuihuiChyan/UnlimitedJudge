#!/bin/bash
python -u evaluate_gpt.py \
    --model-type "gpt-4-1106-preview" \
    --prompt-type "icl" \
    --data-type "pandalm" \
    --data-path ./data \
    --max-new-token 32 \
    --multi-process True \
    --save-logit True