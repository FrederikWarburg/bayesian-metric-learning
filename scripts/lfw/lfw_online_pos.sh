#!/bin/bash
#BSUB -q gpua100
#BSUB -J online-pos-lfw
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o ../bsub_logs/gpu_%J.out
#BSUB -e ../bsub_logs/gpu_%J.err

module load python3/3.9.11 cuda/11.4 cudnn/v8.2.2.26-prod-cuda-11.4
source ../metric_learning/bin/activate
export CUDA_VISIBLE_DEVICES=0

for seed in {1..5}
do
    python run.py --config "../configs/lfw/laplace_online_pos.yaml" --seed $seed
    rm -rf ../lightning_logs/lfw/laplace_online/$seed/contrastive/pos/checkpoints
done
