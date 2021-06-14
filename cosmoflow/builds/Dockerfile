FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

RUN python -m pip install --no-cache-dir -U pip

RUN pip install --no-cache-dir pandas wandb

# Install MLPerf-logging
RUN pip install --no-cache-dir "git+https://github.com/mlperf-hpc/logging.git@hpc-0.5.0"
