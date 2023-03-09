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
# Usage: source scripts/activate_local_openfold_venv.sh /path/to/openfold-venv
#
# Exit: conda deactivate

# Setup text effects:
GREEN=$(tput setaf 2)
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

# Read input argument:
PREFIX_PATH=$1

# Activate conda environment:
source $PREFIX_PATH/conda/etc/profile.d/conda.sh && \
conda activate openfold-venv && \
echo -e "${GREEN}${BOLD}openfold-venv activated!${NORMAL}"
