#!/bin/bash
#BSUB -q gpua100
#BSUB -J posthoc-lfw
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o ../bsub_logs/gpu_%J.out
#BSUB -e ../bsub_logs/gpu_%J.err

module load python3/3.9.11 cuda/11.4 cudnn/v8.2.2.26-prod-cuda-11.4
source ../metric_learning/bin/activate
export CUDA_VISIBLE_DEVICES=1

CONFIG_PATHS=(
    "laplace_posthoc_fix" \
    "laplace_posthoc_full" \
    "laplace_posthoc_pos" \
    "laplace_posthoc_arccos_full" \
    "laplace_posthoc_arccos_pos" \
    "pfe" \
)

for f in ${CONFIG_PATHS[@]}
do
    for seed in {1..5}
    do
        python run.py --config "../configs/lfw/$f.yaml" --seed $seed
    done
done

python run.py --config "../configs/lfw/deepensemble.yaml"
