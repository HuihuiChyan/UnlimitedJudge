export CUDA_VISIBLE_DEVICES=6

DATA_TYPE="salad-bench"
MODEL_TYPE="judgelm"

python3 -u src/cal_reliability.py \
    --model-name-or-path "./models/JudgeLM-7B" \
    --cali-model-name-or-path "./models/llama2-7b-chat-hf/" \
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

DATA_TYPE="salad-bench"

# python3 -u src/cal_reliability.py \
#     --model-name-or-path "/home/disk/huanghui/UnlimitedJudge/models/PandaLM-7B" \
#     --cali-model-name-or-path "/home/disk/huanghui/CrossEval-corr/models/llama-7b/" \
#     --model-type "pandalm" \
#     --data-type $DATA_TYPE \
#     --max-new-token 1024 \
#     --logit-file "relia_scores/pandalm/${DATA_TYPE}-logit.jsonl" \
#     --output-file "relia_scores/pandalm/${DATA_TYPE}-relia.json"

# python3 -u src/evaluate_reliability.py \
#     --model-type "pandalm" \
#     --data-type $DATA_TYPE \
#     --logit-file "relia_scores/pandalm/${DATA_TYPE}-logit.jsonl" \
#     --output-file "relia_scores/pandalm/${DATA_TYPE}-relia.json"

# DATA_TYPE="salad-bench"

# python3 -u src/cal_reliability.py \
#     --model-name-or-path "/home/disk/huanghui/UnlimitedJudge/models/Auto-J-13B" \
#     --cali-model-name-or-path "/home/disk/huanghui/CrossEval-corr/models/llama2-chat-13b/" \
#     --model-type "auto-j" \
#     --data-type $DATA_TYPE \
#     --max-new-token 1024 \
#     --logit-file "relia_scores/auto-j/${DATA_TYPE}-logit.jsonl" \
#     --output-file "relia_scores/auto-j/${DATA_TYPE}-relia.json"

# python3 -u src/evaluate_reliability.py \
#     --model-type "auto-j" \
#     --data-type $DATA_TYPE \
#     --logit-file "relia_scores/auto-j/${DATA_TYPE}-logit.jsonl" \
#     --output-file "relia_scores/auto-j/${DATA_TYPE}-relia.json"