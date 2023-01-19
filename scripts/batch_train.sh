#!/bin/bash

# NCC
# for cfg in essel_7 ord_first_7 ft_first_7
# do
#     sbatch --job-name=$cfg scripts/train.sh $cfg
#     sleep 4 # so that wandb runs don't get assigned the same number
# done

# OFFICE
for cfg in text_4 text_5 text_6
do
    sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
    sleep 4 # so that wandb runs don't get assigned the same number
done