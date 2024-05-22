#!/bin/bash
python -u evaluate_gpt.py \
    --model-type "gpt-3.5-turbo-0125" \
    --prompt-type "icl" \
    --data-type "pandalm" \
    --data-path ./data \
    --max-new-token 32 \
    --multi-process True \
    --save-logit False