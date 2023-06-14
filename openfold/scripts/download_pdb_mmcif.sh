#!/bin/bash
#
# Copyright 2021 DeepMind Technologies Limited
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
# Usage: bash download_pdb_mmcif.sh /path/to/data/pdb_mmcif/original

set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v aria2c &> /dev/null ; then
    echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
    exit 1
fi

if ! command -v rsync &> /dev/null ; then
    echo "Error: rsync could not be found. Please install rsync."
    exit 1
fi

DOWNLOAD_DIR="$1"
DOWNLOAD_RAW_DIR="${DOWNLOAD_DIR}/raw"

echo "Running rsync to fetch all mmCIF files (note that the rsync progress estimate might be inaccurate)..."
echo "If the download speed is too slow, try changing the mirror to:"
echo "  * rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/ (Europe)"
echo "  * ftp.pdbj.org::ftp_data/structures/divided/mmCIF/ (Asia)"
echo "or see https://www.wwpdb.org/ftp/pdb-ftp-sites for more download options."
mkdir -p "${DOWNLOAD_RAW_DIR}"
rsync --recursive --links --perms --times --compress --info=progress2 --delete --port=33444 \
  rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ \
  "${DOWNLOAD_RAW_DIR}"

aria2c "ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat" --dir="${DOWNLOAD_DIR}"

aria2c "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt" --dir="${DOWNLOAD_DIR}"
