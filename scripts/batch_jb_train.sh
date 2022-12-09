#!/bin/bash

for cfg in essel ord_first ft_first
do
    sbatch --job-name=cfg scripts/jb_train.sh $cfg
done