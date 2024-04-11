#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python -u evaluate_gpt.py \
    --model-name-or-path "./output/mistral-7B-regression-prometheus" \
    --model-type "gpt4" \
    --data-type "judgelm" \
    --data-path ./data \
    --multi-process True \
    --cot-augmentation prometheus-noplan