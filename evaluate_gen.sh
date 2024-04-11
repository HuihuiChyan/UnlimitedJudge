export CUDA_VISIBLE_DEVICES=0
python -u evaluate_gen.py \
    --model-name-or-path "./models/mistral-7B" \
    --model-type "judgelm" \
    --data-type "llama2-7b-chat" \
    --class-type "generation" \
    --data-path ./data/