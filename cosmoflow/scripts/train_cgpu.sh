#!/bin/bash
#SBATCH -C gpu -c 10
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -J train-cgpu
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

. scripts/setup_cgpu.sh
#export HOROVOD_TIMELINE=./timeline.json

set -x
srun -l -u python train.py -d --rank-gpu $@
