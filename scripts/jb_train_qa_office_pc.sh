#!/bin/bash

# Job name appears in the squeue output, output is the output filename 
#SBATCH -o ncc-logs/%x-%A.out
#SBATCH --ntasks 20 # set this to 20 such that only one job runs at a time
# #SBATCH --gres gpu

source ./env/bin/activate

# Commands to be run:
python --version
nvidia-smi
hostname
echo "Node id: $SLURM_NODEID"

mem=`nvidia-smi --query-gpu=memory.total --format=csv | tail -n 1 | awk '{print $1}'`
echo "$mem Mb available"

date '+%c'
# python combine_LIME.py --dataset fraud --model joint
python src/jb_train_qa.py --config $1