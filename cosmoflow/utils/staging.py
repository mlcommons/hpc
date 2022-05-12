# 'Regression of 3D Sky Map to Cosmological Parameters (CosmoFlow)'
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
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
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
#
# NOTICE.  This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government consequently retains certain rights. As such,
# the U.S. Government has been granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
# to reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit other to do so.

"""Utility code for staging data files into local storage"""

# System imports
import os
import shutil
import logging


def stage_files(input_dir, output_dir, n_files, rank=0, size=1):
    """Stage specified number of files to directory.

    This function works in a distributed fashion. Each rank will only stage
    its chunk of the file list.
    """
    if rank == 0:
        logging.info(f'Staging {n_files} files to {output_dir}')

    # Find all the files in the input directory
    files = sorted(os.listdir(input_dir))

    # Make sure there are at least enough files available
    if len(files) < n_files:
        raise ValueError(f'Cannot stage {n_files} files; only {len(files)} available')

    # Take the specified number of files
    files = files[:n_files]

    # Copy my chunk into the output directory
    os.makedirs(output_dir, exist_ok=True)
    for f in files[rank::size]:
        logging.debug(f'Staging file {f}')
        shutil.copyfile(os.path.join(input_dir, f),
                        os.path.join(output_dir, f))
    logging.debug('Data staging completed')
