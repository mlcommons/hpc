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

import json
from pathlib import Path
from typing import Dict


def load_alignments_super_index(
    alignments_super_index_filepath: Path,
    verbose: bool = False,
    pprefix: str = "",
) -> Dict[str, dict]:
    if verbose:
        print(f"{pprefix}Loading {repr(alignments_super_index_filepath)}...")
    with open(alignments_super_index_filepath) as f:
        alignments_super_index = json.load(f)
    if verbose:
        print(
            f"{pprefix}alignments_super_index ({len(alignments_super_index)})"
            f" loaded from {repr(alignments_super_index_filepath)} successfully!"
        )
    return alignments_super_index


def load_alignments(
    alignments_super_index: Dict[str, dict],
    alignments_dirpath: Path,
    key: str,
) -> dict:
    alignments_index = alignments_super_index[key]
    alignments_db_path = alignments_dirpath / alignments_index["db"]
    alignments = {}
    with open(alignments_db_path, "rb") as f:
        for file_index in alignments_index["files"]:
            filename, start, size = file_index
            f.seek(start)
            content = f.read(size).decode("utf-8")
            alignments[filename] = content
    return alignments
