#!/bin/bash

for num in 3 4
do
    for cfg in ft_first_${num}_qa ord_first_${num}_qa essel_${num}_qa
    do
        sbatch --job-name=$cfg scripts/jb_train.sh $cfg
        sleep 4 # so that wandb runs don't get assigned the same number
    done
done

# for cfg in ft_first_2_qa ord_first_2_qa #essel_qa essel_2_qa #ft_first_qa ord_first_qa 
# do
#     sbatch --job-name=$cfg scripts/jb_train.sh $cfg
#     sleep 4 # so that wandb runs don't get assigned the same number
# done