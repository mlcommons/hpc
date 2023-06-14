#!/bin/bash
#
# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 OpenFold Consortium
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
# Usage: bash download_open_protein_set.sh /path/to/data/open_protein_set/original

set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v aws &> /dev/null ; then
    echo "Error: AWS CLI could not be found. Check https://aws.amazon.com/cli/ and install AWS CLI."
    exit 1
fi

DOWNLOAD_DIR="${1}/"
mkdir -p "${DOWNLOAD_DIR}"

# download root files:
aws s3 cp --no-sign-request s3://openfold/LICENSE "${DOWNLOAD_DIR}"
aws s3 cp --no-sign-request s3://openfold/duplicate_pdb_chains.txt "${DOWNLOAD_DIR}"

# download pdb directory:
mkdir -p "${DOWNLOAD_DIR}/pdb"
aws s3 cp --no-sign-request s3://openfold/pdb "${DOWNLOAD_DIR}/pdb" --recursive
