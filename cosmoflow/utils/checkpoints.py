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

"""Utility code for handling checkpoint loading"""

# System imports
import os
import logging

# External imports
import tensorflow as tf
import horovod.tensorflow.keras as hvd

def reload_last_checkpoint(checkpoint_format, n_epochs, distributed):
    """Finds and loads the last checkpoint matching the provided pattern"""
    # Count down from n_epochs to 0 to find the last epoch.
    # Note that keras names checkpoint files with epoch number starting from 1.
    # So the matched number corresponds to the new initial epoch.
    for epoch in range(n_epochs, 0, -1):
        checkpoint = checkpoint_format.format(epoch=epoch)
        if os.path.exists(checkpoint):
            logging.info('Found last checkpoint at %s', checkpoint)
            # Use special reload to prepare the DistributedOptimizer
            if distributed:
                model = hvd.load_model(checkpoint)
            else:
                model = tf.keras.models.load_model(checkpoint)
            return epoch, model
    raise Exception('Unable to find a checkpoint file at %s' % checkpoint_format)
