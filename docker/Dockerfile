FROM nvcr.io/nvidia/pytorch:21.08-py3

# PyG
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0 8.0" pip install --no-cache-dir \
    torch-scatter torch-sparse torch-geometric==1.7.2

# MLPerf logging
RUN pip install --no-cache-dir git+https://github.com/mlcommons/logging.git

# Other packages
RUN pip install --no-cache-dir ray submitit demjson wandb ase pymatgen
