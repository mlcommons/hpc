#!/bin/bash
#SBATCH -C gpu
#SBATCH -J ocp-pm
#SBATCH -A nstaff_g
#SBATCH -q early_science
#SBATCH --image=sfarrell/mlperf-ocp:latest
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --time 4:00:00
#SBATCH -o logs/slurm-%x-%j.out

args=$@

# Default settings
: "${OCP_CONFIG:=configs/mlperf_hpc_pm.yml}"

# Distributed config
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29504
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn

set -x
srun -l -u shifter scripts/run_training.sh --config-yml $OCP_CONFIG $args
