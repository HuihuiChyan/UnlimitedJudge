#!/bin/bash
python -u evaluate_gpt.py \
    --model-type "gpt-3.5-turbo-0125" \
    --prompt-type "cot" \
    --data-type "judgelm" \
    --data-path ./data \
    --multi-process True \
    --save-logit True