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
# Usage: source scripts/deactivate_local_openfold_venv.sh

# Setup text effects:
CYAN=$(tput setaf 6)
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

# Deactivate conda environment:
conda deactivate && \
echo -e "${CYAN}${BOLD}openfold-venv deactivated!${NORMAL}"
