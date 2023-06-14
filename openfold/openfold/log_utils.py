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
from typing import List

import pandas as pd


def save_logs(logs: List[dict], outpath: Path, append: bool) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for log in logs:
        line = json.dumps(log)
        lines.append(line)
    outstr = "\n".join(lines) + "\n"
    mode = "a" if append else "w"
    with open(outpath, mode) as f:
        f.write(outstr)


def read_logs(
    filepath: Path,
    drop_overridden_iterations: bool = True,
) -> pd.DataFrame:
    with open(filepath) as f:
        logs = f.read().strip().split("\n")
    logs = [json.loads(log) for log in logs]
    logs_df = pd.DataFrame(logs)
    if drop_overridden_iterations:
        logs_df = logs_df.drop_duplicates("iteration", keep="last")
        logs_df = logs_df.reset_index(drop=True).copy()
    return logs_df
