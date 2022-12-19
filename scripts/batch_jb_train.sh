#!/bin/bash

# for num in 3
# do
#     for cfg in ft_first_${num}_qa ord_first_${num}_qa essel_${num}_qa
#     do
#         sbatch --job-name=$cfg scripts/jb_train.sh $cfg
#         sleep 4 # so that wandb runs don't get assigned the same number
#     done
# done

for cfg in essel_2
do
    sbatch --job-name=$cfg scripts/jb_train.sh $cfg
    sleep 4 # so that wandb runs don't get assigned the same number
done