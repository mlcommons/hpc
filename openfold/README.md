# OpenFold PyTorch benchmark implementation

This repository defines the reference implementation for the MLPerf HPC OpenFold benchmark.

## Download datasets

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

## Installation

There are 2 options:

1. Build a docker image based on provided Dockerfile (recommended).
2. Build a local virtual environment based on provided scripts.

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

## Dataset preprocessing

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

The above will produce 616 GiB (~1k files).

## Training command

Single GPU training command example:

```bash
python train.py \
--training_dirpath /path/to/training_rundir/ \
--pdb_mmcif_chains_filepath /data/pdb_mmcif/processed/chains.csv \
--pdb_mmcif_dicts_dirpath /data/pdb_mmcif/processed/dicts \
--pdb_obsolete_filepath /data/pdb_mmcif/processed/obsolete.dat \
--pdb_alignments_dirpath /data/open_protein_set/processed/pdb_alignments \
--initialize_parameters_from /path/to/mlperf_hpc_openfold_resumable_checkpoint.pt \
--seed 1234567890 \
--num_train_iters 2000 \
--log_every_iters 10 \
--checkpoint_every_iters 50 \
--keep_last_checkpoints 1 \
--val_every_iters 100 \
--keep_best_checkpoints 1 \
--keep_val_checkpoints \
--device_batch_size 1 \
--init_lr 1e-3 \
--final_lr 5e-5 \
--warmup_lr_length 0 \
--init_lr_length 2000 \
--gradient_clipping \
--clip_grad_max_norm 0.1 \
--num_train_dataloader_workers 14 \
--num_val_dataloader_workers 2 \
--save_process_logs
```

When launching distributed training use also `--distributed` flag.

For single node training replace `python train.py` with `torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py`.

For multi-node training use full syntax: `torchrun --nnodes=16 --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py` that is executed once per each node. See the [scripts/multi_node_training.sub](scripts/multi_node_training.sub) example.
