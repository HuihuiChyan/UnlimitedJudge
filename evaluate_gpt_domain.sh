#!/bin/bash
python -u evaluate_gpt_domain.py \
    --model-type "gpt-3.5-turbo-0613" \
    --prompt-type "vanilla" \
    --data-type "halu-eval-dialogue" \
    --data-path ./data \
    --multi-process True \
    --max-new-token 1024 \
    --rewrite-output False