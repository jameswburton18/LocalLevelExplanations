#!/bin/bash

for num in 3 4
do
    for cfg in ft_first_$num ord_first_$num essel_$num
    do
        sbatch --job-name=$cfg scripts/jb_train.sh $cfg
        sleep 4 # so that wandb runs don't get assigned the same number
    done
done