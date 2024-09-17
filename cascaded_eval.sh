export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="auto-j"
DATA_TYPE="auto-j"

python3 -u src/cascaded_eval.py \
    --data-type $DATA_TYPE \
    --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json" \
    --logit-file-gpt "outputs/${DATA_TYPE}-gpt-4-turbo-128k-vanilla.jsonl"