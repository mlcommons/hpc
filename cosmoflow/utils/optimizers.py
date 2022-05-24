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
Utilty code for constructing optimizers and scheduling learning rates.
"""

# System
import math
from functools import partial

# Externals
from tensorflow import keras
import horovod.tensorflow.keras as hvd
try:
    from mlperf_logging import mllog
    have_mlperf_logging = True
except ImportError:
    have_mlperf_logging = False

# Locals
import utils.distributed


def _lr_schedule(epoch, base_lr, peak_lr, n_warmup_epochs, decay_schedule={}):
    """Learning rate schedule function.

    Gives the learning rate as a function of epoch according to
    additional settings:
        base_lr: baseline unscaled learning rate at beginning of training.
        peak_lr: scaled learning at end of warmup period
        n_warmup_epochs: number of linear warmup epochs
        decay_schedule: a dict of epoch number -> decay factor
    """
    # Linear LR warmup
    if epoch < n_warmup_epochs:
        return epoch * (peak_lr - base_lr) / n_warmup_epochs + base_lr
    else:
        # Find the most recent decay factor
        decay_factor = 1.
        decay_epoch = 0
        for e, d in decay_schedule.items():
            if e >= decay_epoch and e < epoch:
                decay_epoch, decay_factor = e, d
        return peak_lr * decay_factor


def get_lr_schedule(base_lr, global_batch_size, base_batch_size=None,
                    scaling=None, n_warmup_epochs=0, decay_schedule={}):
    """Get the learning rate schedule function"""
    if scaling == 'linear':
        scale_factor = global_batch_size / base_batch_size
    elif scaling == 'sqrt':
        scale_factor = math.sqrt(global_batch_size / base_batch_size)
    else:
        scale_factor = 1.;
    peak_lr = base_lr * scale_factor

    # MLPerf logging
    # NOTE: there is currently a confusing mismatch between the parameter
    # naming convention in this implementation and MLPerf's hyperparameter
    # conventions. Here we define base LR to be the LR at a baseline batch
    # size and the "peak" LR to be the value scaled according to current batch
    # size. We will leave things as-is for now.
    if utils.distributed.rank() == 0 and have_mlperf_logging:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.OPT_BASE_LR, value=peak_lr)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_EPOCHS, value=n_warmup_epochs)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_FACTOR, value=scale_factor)
        mllogger.event(key=mllog.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
                       value=sorted(decay_schedule.keys()))
        mllogger.event(key=mllog.constants.OPT_LR_DECAY_FACTOR,
                       value=max(decay_schedule.values()) if len(decay_schedule)>0 else 1)
    return partial(_lr_schedule, base_lr=base_lr, peak_lr=peak_lr,
                   n_warmup_epochs=n_warmup_epochs,
                   decay_schedule=decay_schedule)


def get_optimizer(name, distributed=False, **opt_args):
    """Configure the optimizer"""

    # MLPerf logging
    if utils.distributed.rank() == 0 and have_mlperf_logging:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.OPT_NAME, value=name)

    # Construct the optimizer
    OptType = getattr(keras.optimizers, name)
    opt = OptType(**opt_args)

    # Distributed optimizer wrapper
    if distributed:
        opt = hvd.DistributedOptimizer(opt)

    return opt
