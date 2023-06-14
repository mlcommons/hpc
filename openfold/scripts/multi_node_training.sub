#!/bin/bash
#
# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage: sbatch scripts/multi_node_training.sub

#SBATCH --job-name mlperf-hpc:openfold-reference
#SBATCH --time 02:15:00
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive

# Print current datetime:
echo "START" $(date +"%Y-%m-%d %H:%M:%S")

# Print node list:
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NODELIST=$SLURM_NODELIST"

# Note: the following srun commands assume that pyxis plugin is installed on a SLURM cluster.
# https://github.com/NVIDIA/pyxis

# Download container and give it a name:
srun \
--mpi=none \
--container-image=openfold_pyt \
--container-name=$SLURM_JOB_ID \
bash -c 'echo "srun SLURM_JOB_ID=$SLURM_JOB_ID SLURMD_NODENAME=$SLURMD_NODENAME"'

# Print current datetime again:
echo "READY" $(date +"%Y-%m-%d %H:%M:%S")

# Set number of threads to use for parallel regions:
export OMP_NUM_THREADS=1

# Set MLPerf variables:
export DATESTAMP=$(date +"%y%m%d%H%M%S%N")
export EXP_ID=1

# Run the command:
# Note: MASTER_ADDR and MASTER_PORT variables are set automatically by pyxis.
srun \
--mpi=none \
--container-name=$SLURM_JOB_ID \
--container-mounts=/path/to/data:/data:ro,/path/to/training_rundir:/training_rundir \
bash -c \
'echo "srun SLURMD_NODENAME=$SLURMD_NODENAME MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"; \
torchrun \
--nnodes=$SLURM_JOB_NUM_NODES \
--nproc_per_node=8 \
--rdzv_id=$SLURM_JOB_ID \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
train.py \
--training_dirpath /training_rundir \
--pdb_mmcif_chains_filepath /data/pdb_mmcif/processed/chains.csv \
--pdb_mmcif_dicts_dirpath /data/pdb_mmcif/processed/dicts \
--pdb_obsolete_filepath /data/pdb_mmcif/processed/obsolete.dat \
--pdb_alignments_dirpath /data/open_protein_set/processed/pdb_alignments \
--initialize_parameters_from /data/mlperf_hpc_openfold_resumable_checkpoint.pt \
--seed 1234567890 \
--num_train_iters 2000 \
--val_every_iters 40 \
--local_batch_size 1 \
--base_lr 1e-3 \
--warmup_lr_init 1e-5 \
--warmup_lr_iters 0 \
--num_train_dataloader_workers 14 \
--num_val_dataloader_workers 2 \
--distributed'
