export CUDA_VISIBLE_DEVICES=6

MODEL_TYPE="pandalm"
DATA_TYPE="salad-bench"

python3 -u src/cal_reliability.py \
    --model-name-or-path "./models/PandaLM-7B" \
    --cali-model-name-or-path "./models/llama-7b/" \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --max-new-token 512 \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"

python3 -u src/evaluate_reliability.py \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"

MODEL_TYPE="judgelm"

python3 -u src/cal_reliability.py \
    --model-name-or-path "./models/JudgeLM-7B" \
    --cali-model-name-or-path "./models/vicuna-7b/" \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --max-new-token 1024 \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"

python3 -u src/evaluate_reliability.py \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"