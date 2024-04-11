#!/bin/bash
#SBATCH -N 1 # 指定node的数量
#SBATCH --gres=gpu:1 # 需要使用多少GPU
#SBATCH -o evaluate.log # 把输出结果STDOUT保存在哪一个文件
#SBATCH -w wxhd11
# srun -N 1 -G 1 -w wxhd11 \
export CUDA_VISIBLE_DEVICES=0
python -u evaluate.py \
    --model-name-or-path "./models/autoj-13b" \
    --model-type "auto-j" \
    --class-type "generation" \
    --data-type "auto-j" \
    --prompt-name "autoj" \
    --eval-batch-size 16 \
    --data-path ./data \
    --max-new-token 1024