#!/bin/bash

for cfg in ft_first_2_qa #essel_qa #ord_first_2_qa essel_2_qa ft_first_2_qa
do
    sbatch --job-name=$cfg scripts/jb_train.sh $cfg
    sleep 4 # so that wandb runs don't get assigned the same number
done