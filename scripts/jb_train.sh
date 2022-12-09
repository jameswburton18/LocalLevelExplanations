#!/bin/bash
# Instructing SLURM to locate and assign X number of nodes with Y number of
# cores in each node. X,Y are integers.
#SBATCH -N 1
#SBATCH -c 4

# Governs the run time limit and resource limit for the job. See QOS tables for combinations
# Available QOS: [debug, short, long-high-prio, long-low-prio, long-cpu]
#SBATCH -p res-gpu-small
#SBATCH --qos long-high-prio
#SBATCH -t 04-00:00
# -x shows which ones to ignore
# #SBATCH -x gpu[0-8]

# Job name appears in the squeue output, output is the output filename 
#SBATCH -o ncc-logs/%x-%A.out

# Pick how much memory to allocate per CPU core.
#SBATCH --mem 8G
#SBATCH --gres gpu

# Make python available:
module load python/3.9.9
source ./env/bin/activate

# Commands to be run:

date '+%c'
# python combine_LIME.py --dataset fraud --model joint
python jb_train.py --config $1
