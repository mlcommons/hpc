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
Random dummy dataset specification.
"""

# Externals
import tensorflow as tf


def construct_dataset(sample_shape, target_shape,
                       batch_size=1, n_samples=32):
    x = tf.random.uniform([n_samples]+sample_shape)
    y = tf.random.uniform([n_samples]+target_shape)
    data = tf.data.Dataset.from_tensor_slices((x, y))
    return data.repeat().batch(batch_size).prefetch(4)


def get_datasets(sample_shape, target_shape, batch_size,
                 n_train, n_valid, dist, n_epochs=None, shard=False):
    train_dataset = construct_dataset(sample_shape, target_shape, batch_size=batch_size)
    valid_dataset = None
    if n_valid > 0:
        valid_dataset = construct_dataset(sample_shape, target_shape, batch_size=batch_size)
    n_train_steps = n_train  // batch_size
    n_valid_steps = n_valid  // batch_size
    if shard:
        n_train_steps = n_train_steps // dist.size
        n_valid_steps = n_valid_steps // dist.size

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train=n_train, n_valid=n_valid, n_train_steps=n_train_steps,
                n_valid_steps=n_valid_steps)
