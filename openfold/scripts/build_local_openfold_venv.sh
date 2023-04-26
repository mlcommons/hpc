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
# Usage: bash scripts/build_local_openfold_venv.sh /path/to/openfold-venv

set -e  # immediately exit on first error

# Setup text effects:
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

# Read input argument:
PREFIX_PATH=$1
if [ -z $PREFIX_PATH ]; then
    echo "${BOLD}${RED}Input error:${NORMAL} missing path!"
    echo "Please, specify venv location!"
    exit 1
fi

# Check if prefix path already exists:
if [ -f $PREFIX_PATH ] || [ -d $PREFIX_PATH ] ; then
    echo "${BOLD}${RED}Build error:${NORMAL} ${BOLD}$PREFIX_PATH${NORMAL} already exists!"
    echo "Remove ${BOLD}$PREFIX_PATH${NORMAL} manually or set different location."
    exit 1
fi

echo "Building ${GREEN}${BOLD}$PREFIX_PATH${NORMAL}..."

# Install conda to specified prefix path:
wget -4 https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $PREFIX_PATH/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh

# Create conda environment:
$PREFIX_PATH/conda/bin/conda create --name=openfold-venv -y python==3.8.*

# Activate conda environment:
source scripts/activate_local_openfold_venv.sh $PREFIX_PATH

# Install requirements:
echo "Installing requirements..."
conda install -y \
    pytorch::pytorch==2.0.* \
    conda-forge::numpy==1.22.2 \
    conda-forge::pandas==1.5.2 \
    conda-forge::scipy==1.10.1 \
    conda-forge::tqdm==4.65.0 \
    conda-forge::psutil==5.9.4 \
    conda-forge::biopython==1.79 \
    conda-forge::Pympler==1.0.1 \
    bioconda::kalign3==3.3.*

pip install dacite==1.8.0 \
    "git+https://github.com/mlcommons/logging.git@2.1.0" \
    "git+https://github.com/NVIDIA/mlperf-common.git"

# Install OpenFold source code package in editable mode:
pip install -e .

echo "${GREEN}${BOLD}$0 finished successfully!${NORMAL}"
