#!/bin/bash
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -J train-cori
#SBATCH --image docker:sfarrell/cosmoflow-cpu-mpich:latest
#SBATCH -o logs/%x-%j.out

export OMP_NUM_THREADS=32
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
export HDF5_USE_FILE_LOCKING=FALSE

set -x
srun -l -u shifter python train.py -d $@
