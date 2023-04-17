#!/bin/bash

# NCC
# for cfg in ft_first #ord_first essel essel_2 ord_first_2 ft_first_2  essel ord_first ft_first
# do
#     sbatch --job-name=$cfg scripts/train_qa.sh $cfg
#     sleep 4 # so that wandb runs don't get assigned the same number
# done

# OFFICE
for cfg in text_5 text_6
do
    sbatch --job-name=${cfg}_qa_us scripts/train_qa_unseen_office_pc.sh $cfg
    sleep 4 # so that wandb runs don't get assigned the same number
done
