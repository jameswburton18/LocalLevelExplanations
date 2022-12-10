#!/bin/bash

for cfg in essel_2 ord_first_2 ft_first_2
do
    sbatch --job-name=$cfg scripts/jb_train.sh $cfg
    sleep 8 # so that wandb runs don't get assigned the same number
done