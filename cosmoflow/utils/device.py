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

"""
Hardware/device configuration
"""

# System
import os
import logging
import socket

# Externals
import tensorflow as tf

# Locals
import utils.distributed as dist


def configure_session(gpu=None, intra_threads=None, inter_threads=None,
                      kmp_blocktime=None, kmp_affinity=None, omp_num_threads=None):
    """Sets the thread knobs in the TF backend"""
    if kmp_blocktime is not None:
        os.environ['KMP_BLOCKTIME'] = str(kmp_blocktime)
    if kmp_affinity is not None:
        os.environ['KMP_AFFINITY'] = kmp_affinity
    if omp_num_threads is not None:
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    if dist.rank() == 0:
        logging.info('KMP_BLOCKTIME %s', os.environ.get('KMP_BLOCKTIME', ''))
        logging.info('KMP_AFFINITY %s', os.environ.get('KMP_AFFINITY', ''))
        logging.info('OMP_NUM_THREADS %s', os.environ.get('OMP_NUM_THREADS', ''))
        logging.info('INTRA_THREADS %i', intra_threads)
        logging.info('INTER_THREADS %i', inter_threads)

    if gpu is not None:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) < gpu + 1:
            raise RuntimeError(f'{socket.gethostname()}: ' +
                               f'GPU {gpu} unavailable, ' +
                               f'{len(gpu_devices)} visible')
        tf.config.set_visible_devices(gpu_devices[gpu], 'GPU')

    if intra_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    if inter_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
