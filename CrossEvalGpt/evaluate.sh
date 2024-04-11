#!/bin/bash
#SBATCH -N 1 # 指定node的数量
#SBATCH --gres=gpu:1 # 需要使用多少GPU
#SBATCH -o evaluate.log # 把输出结果STDOUT保存在哪一个文件
#SBATCH -w wxhd11
# srun -N 1 -G 1 -w wxhd11 \
python -u evaluate.py \
    --data-type "prometheus-ind" \
    --eval-batch-size 16 \
    --data-path ./data \
    --max-new-token 256 \
    --use-ray True \
    --cot-augmentation False