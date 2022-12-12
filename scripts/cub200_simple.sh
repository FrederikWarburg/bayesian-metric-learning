#!/bin/bash

CONFIG_PATHS=(#"deterministic" \
             #"pfe" \
            #"laplace_posthoc_fix" \
            #"laplace_posthoc_full" \ 
            #"laplace_posthoc_pos" \
            "laplace_online_fix" \
            "laplace_online_full" \ 
            "laplace_online_pos" \
            #"laplace_posthoc_arccos_full" \
            #"laplace_posthoc_arccos_pos" \
            "laplace_online_arccos_full" \
            "laplace_online_arccos_pos" \
            )
CONFIG_PATHS=("mcdrop")

for f in ${CONFIG_PATHS[@]}
do
    for seed in {1..5}
    do
        CUDA_VISIBLE_DEVICES=6 python run.py --config "../configs/cub200/$f.yaml" --seed $seed
    done
done

