#!/bin/bash

for cfg in essel_2 ord_first_2 ft_first_2 essel ord_first ft_first
do
    sbatch --job-name=$cfg scripts/jb_train_qa.sh $cfg
    sleep 8 # so that wandb runs don't get assigned the same number
done

# for cfg in essel_test
# do
#     sbatch --job-name=$cfg scripts/jb_train_qa.sh $cfg
#     sleep 8 # so that wandb runs don't get assigned the same number
# done