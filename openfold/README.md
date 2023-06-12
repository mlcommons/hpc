# OpenFold PyTorch benchmark implementation

This repository defines the reference implementation for the MLPerf HPC OpenFold benchmark.

## Installation

There are 2 options:

1. Build a docker image based on provided Dockerfile (recommended).
2. Build a local virtual environment using provided scripts.

### Option 1. Build a docker image.

```bash
docker build -t openfold_pyt -f Dockerfile .
```

### Option 2. Build a local virtual environment.

```bash
# build:
bash scripts/build_local_openfold_venv.sh /path/to/openfold-venv

# activate:
source scripts/activate_local_openfold_venv.sh /path/to/openfold-venv

# deactivate:
source scripts/deactivate_local_openfold_venv.sh
```

## Dataset

### Download original dataset

**You should ignore this and [download already processed dataset and checkpoint for parameter initialization](#download-processed-dataset-and-checkpoint)!**

Note: downloading scripts require `aria2c`, `rsync`, and `AWS CLI`.

To download PDB and OpenProteinSet use the following commands:

```bash
bash scripts/download_pdb_mmcif.sh /data/pdb_mmcif/original
bash scripts/download_open_protein_set.sh /data/open_protein_set/original
```

After downloading completes, check if you have the following directory tree:

```bash
data
├── open_protein_set
│   └── original  # 606 GiB (524k files, 394k dirs)
│       ├── LICENSE
│       ├── duplicate_pdb_chains.txt
│       └── pdb/
└── pdb_mmcif
    └── original  # +55 GiB (+200k files)
        ├── clusters-by-entity-40.txt
        ├── obsolete.dat
        └── raw/
```

### Preprocess original dataset

**You should ignore this and [download already processed dataset and checkpoint for parameter initialization](#download-processed-dataset-and-checkpoint)!**

```bash
python scripts/preprocess_pdb_mmcif.py \
--pdb_mmcif_dirpath /data/pdb_mmcif/original \
--output_dirpath /data/pdb_mmcif/processed \
--pdb_clusters_by_entity_filepath /data/pdb_mmcif/original/clusters-by-entity-40.txt \
--pdb_obsolete_filepath /data/pdb_mmcif/original/obsolete.dat

python scripts/preprocess_open_protein_set.py \
--open_protein_set_dirpath /data/open_protein_set/original/ \
--output_dirpath /data/open_protein_set/processed \
--num_shards 10
```

The above will produce 616 GiB (~1k files), in the following directory tree:

```bash
data
├── open_protein_set
│   └── processed  # 601 GiB (11 files, 1 dir)
│       └── pdb_alignments/
└── pdb_mmcif
    └── processed  # 15 GiB (+1k files)
        ├── chains.csv
        ├── obsolete.dat
        └── dicts/
```

### Download processed dataset and checkpoint

Use one of:
- **Globus (recommended)**: https://app.globus.org/file-manager?origin_id=6b83c9b6-c9dc-11ed-9622-4b6fcc022e5a&path=%2F
- HTTPS: https://portal.nersc.gov/cfs/m4291/openfold/

```bash
data
├── mlperf_hpc_openfold_resumable_checkpoint.pt  # 1.1 GiB
├── open_protein_set
│   └── processed  # 601 GiB (11 files, 1 dir)
│       └── pdb_alignments/
└── pdb_mmcif
    └── processed  # 15 GiB (+1k files)
        ├── chains.csv
        ├── obsolete.dat
        └── dicts/
```

## Checkpoint

Verify checkpoint:

```bash
sha256sum mlperf_hpc_openfold_resumable_checkpoint.pt
```

The expected output:

```bash
b518be4677048f2c0f94889c91e2da73655a73b825a8aa7f8b6f5e580d8ffbed  mlperf_hpc_openfold_resumable_checkpoint.pt
```

## Multi-node training command

To launch multi-node benchmark training execute the following command once per each node on your system:

```bash
torchrun \
--nnodes=16 \
--nproc_per_node=8 \
--rdzv_id=$SLURM_JOB_ID \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
train.py \
--training_dirpath /path/to/training_rundir \
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
--distributed
```

SLURM sbatch example: [scripts/multi_node_training.sub](scripts/multi_node_training.sub).

## Single GPU training example

Use the following command to launch single GPU sanity check training and validation on two samples:

```bash
python train.py \
--training_dirpath /path/to/training_rundir \
--pdb_mmcif_chains_filepath /data/pdb_mmcif/processed/chains.csv \
--pdb_mmcif_dicts_dirpath /data/pdb_mmcif/processed/dicts \
--pdb_obsolete_filepath /data/pdb_mmcif/processed/obsolete.dat \
--pdb_alignments_dirpath /data/open_protein_set/processed/pdb_alignments \
--initialize_parameters_from /data/mlperf_hpc_openfold_resumable_checkpoint.pt \
--train_max_pdb_release_date 2021-12-11 \
--target_avg_lddt_ca_value 0.9 \
--seed 1234567890 \
--num_train_iters 80 \
--log_every_iters 4 \
--val_every_iters 8 \
--local_batch_size 1 \
--base_lr 1e-3 \
--warmup_lr_init 1e-5 \
--warmup_lr_iters 0 \
--num_train_dataloader_workers 2 \
--num_val_dataloader_workers 1 \
--use_only_pdb_chain_ids 7ny6_A 7e6g_A
```
