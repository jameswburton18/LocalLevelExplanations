#!/bin/bash

for cfg in essel ord_first ft_first
do
    sbatch --job-name=$cfg scripts/jb_train.sh $cfg
    sleep 8 # so that wandb runs don't get assigned the same number
done