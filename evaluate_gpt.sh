#!/bin/bash
python3 -u src/evaluate_gpt.py \
    --model-name "gpt-3.5-turbo-0613" \
    --prompt-type "vanilla" \
    --data-type "halu-eval-dialogue" \
    --multi-process True \
    --max-new-token 1024 \
    --rewrite-output True