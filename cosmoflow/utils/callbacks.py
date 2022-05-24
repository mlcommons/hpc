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
This module contains some utility callbacks for Keras training.
"""

# System
from time import time
import logging

# Externals
import tensorflow as tf
try:
    from mlperf_logging import mllog
    have_mlperf_logging = True
except ImportError:
    have_mlperf_logging = False


class MLPerfLoggingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback for logging MLPerf results"""
    def __init__(self, metric='val_mean_absolute_error', log_key='eval_error'):
        self.mllogger = mllog.get_mllogger()
        self.metric = metric
        self.log_key = log_key

    def on_epoch_begin(self, epoch, logs={}):
        self.mllogger.start(key=mllog.constants.EPOCH_START,
                            metadata={'epoch_num': epoch + 1})
        self._epoch = epoch

    def on_test_begin(self, logs):
        self.mllogger.start(key=mllog.constants.EVAL_START,
                            metadata={'epoch_num': self._epoch + 1})

    def on_test_end(self, logs):
        self.mllogger.end(key=mllog.constants.EVAL_STOP,
                          metadata={'epoch_num': self._epoch + 1})

    def on_epoch_end(self, epoch, logs={}):
        self.mllogger.end(key=mllog.constants.EPOCH_STOP,
                          metadata={'epoch_num': epoch + 1})
        eval_metric = logs[self.metric]
        self.mllogger.event(key=self.log_key, value=eval_metric,
                            metadata={'epoch_num': epoch + 1})


class StopAtTargetCallback(tf.keras.callbacks.Callback):
    """A Keras callback for stopping training at specified target quality"""

    def __init__(self, metric='val_mean_absolute_error', target_max=None):
        self.metric = metric
        self.target_max = target_max

    def on_epoch_end(self, epoch, logs={}):
        eval_metric = logs[self.metric]
        if self.target_max is not None and eval_metric <= self.target_max:
            self.model.stop_training = True
            logging.info('Target reached; stopping training')


class TimingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback which records the time of each epoch"""
    def __init__(self):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.starttime
        self.times.append(epoch_time)
        logs['time'] = epoch_time
