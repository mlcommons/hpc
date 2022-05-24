FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3

RUN python -m pip install --no-cache-dir -U pip

RUN pip install --no-cache-dir pandas wandb

# Install MLPerf-logging
RUN pip install --no-cache-dir "git+https://github.com/mlcommons/logging.git"
