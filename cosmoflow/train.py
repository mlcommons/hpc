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
Main training script for the CosmoFlow Keras benchmark
"""

# System imports
import os
import argparse
import logging
import pickle
from types import SimpleNamespace

# External imports
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(logging.ERROR)
import horovod.tensorflow.keras as hvd
from mlperf_logging import mllog

# Local imports
from data import get_datasets
from models import get_model
# Fix for loading Lambda layer checkpoints
from models.layers import *
from utils.optimizers import get_optimizer, get_lr_schedule
from utils.callbacks import (TimingCallback, MLPerfLoggingCallback,
                             StopAtTargetCallback)
from utils.device import configure_session
from utils.argparse import ReadYaml
from utils.checkpoints import reload_last_checkpoint
from utils.mlperf_logging import configure_mllogger, log_submission_info

# Stupid workaround until absl logging fix, see:
# https://github.com/tensorflow/tensorflow/issues/26691
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/cosmo.yaml')
    add_arg('--output-dir', help='Override output directory')

    # Override data settings
    add_arg('--data-dir', help='Override the path to input files')
    add_arg('--n-train', type=int, help='Override number of training samples')
    add_arg('--n-valid', type=int, help='Override number of validation samples')
    add_arg('--batch-size', type=int, help='Override the batch size')
    add_arg('--n-epochs', type=int, help='Override number of epochs')
    add_arg('--apply-log', type=int, choices=[0, 1], help='Apply log transform to data')
    add_arg('--stage-dir', help='Local directory to stage data to before training')
    add_arg('--n-parallel-reads', type=int, help='Override num parallel read calls')
    add_arg('--prefetch', type=int, help='Override data prefetch number')

    # Hyperparameter settings
    add_arg('--conv-size', type=int, help='CNN size parameter')
    add_arg('--fc1-size', type=int, help='Fully-connected size parameter 1')
    add_arg('--fc2-size', type=int, help='Fully-connected size parameter 2')
    add_arg('--hidden-activation', help='Override hidden activation function')
    add_arg('--dropout', type=float, help='Override dropout')
    add_arg('--optimizer', help='Override optimizer type')
    add_arg('--lr', type=float, help='Override learning rate')

    # Runtime / device settings
    add_arg('-d', '--distributed', action='store_true')
    add_arg('--rank-gpu', action='store_true',
            help='Use GPU based on local rank')
    add_arg('--resume', action='store_true',
            help='Resume from last checkpoint')
    add_arg('--intra-threads', type=int, default=32,
            help='TF intra-parallel threads')
    add_arg('--inter-threads', type=int, default=2,
            help='TF inter-parallel threads')
    add_arg('--kmp-blocktime', help='Set KMP_BLOCKTIME')
    add_arg('--kmp-affinity', help='Set KMP_AFFINITY')
    add_arg('--omp-num-threads', help='Set OMP_NUM_THREADS')

    # Other settings
    add_arg('--tensorboard', action='store_true',
            help='Enable TB logger')
    add_arg('--print-fom', action='store_true',
            help='Print parsable figure of merit')
    add_arg('-v', '--verbose', action='store_true')
    return parser.parse_args()

def init_workers(distributed=False):
    if distributed:
        hvd.init()
        return SimpleNamespace(rank=hvd.rank(), size=hvd.size(),
                               local_rank=hvd.local_rank(),
                               local_size=hvd.local_size())
    else:
        return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1)

def config_logging(verbose):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

def load_config(args):
    """Reads the YAML config file and returns a config dictionary"""
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Expand paths
    output_dir = config['output_dir'] if args.output_dir is None else args.output_dir
    config['output_dir'] = os.path.expandvars(output_dir)

    # Override data config from command line
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.n_train is not None:
        config['data']['n_train'] = args.n_train
    if args.n_valid is not None:
        config['data']['n_valid'] = args.n_valid
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.n_epochs is not None:
        config['data']['n_epochs'] = args.n_epochs
    if args.apply_log is not None:
        config['data']['apply_log'] = bool(args.apply_log)
    if args.stage_dir is not None:
        config['data']['stage_dir'] = args.stage_dir
    if args.n_parallel_reads is not None:
        config['data']['n_parallel_reads'] = args.n_parallel_reads
    if args.prefetch is not None:
        config['data']['prefetch'] = args.prefetch

    # Hyperparameters
    if args.conv_size is not None:
        config['model']['conv_size'] = args.conv_size
    if args.fc1_size is not None:
        config['model']['fc1_size'] = args.fc1_size
    if args.fc2_size is not None:
        config['model']['fc2_size'] = args.fc2_size
    if args.hidden_activation is not None:
        config['model']['hidden_activation'] = args.hidden_activation
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
    if args.optimizer is not None:
        config['optimizer']['name'] = args.optimizer
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr

    return config

def save_config(config):
    output_dir = config['output_dir']
    config_file = os.path.join(output_dir, 'config.pkl')
    logging.info('Writing config via pickle to %s', config_file)
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)

def load_history(output_dir):
    return pd.read_csv(os.path.join(output_dir, 'history.csv'))

def print_training_summary(output_dir, print_fom):
    history = load_history(output_dir)
    if 'val_loss' in history.keys():
        best = history.val_loss.idxmin()
        logging.info('Best result:')
        for key in history.keys():
            logging.info('  %s: %g', key, history[key].loc[best])
        # Figure of merit printing for HPO parsing
        if print_fom:
            print('FoM:', history['val_loss'].loc[best])
    logging.info('Total epoch time: %.3f', history.time.sum())
    logging.info('Mean epoch time: %.3f', history.time.mean())

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    dist = init_workers(args.distributed)
    config = load_config(args)
    os.makedirs(config['output_dir'], exist_ok=True)
    config_logging(verbose=args.verbose)
    logging.info('Initialized rank %i size %i local_rank %i local_size %i',
                 dist.rank, dist.size, dist.local_rank, dist.local_size)
    if dist.rank == 0:
        logging.info('Configuration: %s', config)

    # Setup MLPerf logging
    mllogger = configure_mllogger(config['output_dir'])
    if dist.rank == 0:
        mllogger.event(key=mllog.constants.CACHE_CLEAR)
        mllogger.start(key=mllog.constants.INIT_START)

    # Device and session configuration
    gpu = dist.local_rank if args.rank_gpu else None
    if gpu is not None:
        logging.info('Taking gpu %i', gpu)
    configure_session(gpu=gpu,
                      intra_threads=args.intra_threads,
                      inter_threads=args.inter_threads,
                      kmp_blocktime=args.kmp_blocktime,
                      kmp_affinity=args.kmp_affinity,
                      omp_num_threads=args.omp_num_threads)

    # Start MLPerf logging
    if dist.rank == 0:
        log_submission_info(**config.get('mlperf', {}))
        mllogger.end(key=mllog.constants.INIT_STOP)
        mllogger.start(key=mllog.constants.RUN_START)

    # Load the data
    data_config = config['data']
    if dist.rank == 0:
        logging.info('Loading data')
    datasets = get_datasets(dist=dist, **data_config)
    logging.debug('Datasets: %s', datasets)

    # Construct or reload the model
    if dist.rank == 0:
        logging.info('Building the model')
    train_config = config['train']
    initial_epoch = 0
    checkpoint_format = os.path.join(config['output_dir'], 'checkpoint-{epoch:03d}.h5')
    if args.resume and os.path.exists(checkpoint_format.format(epoch=1)):
        # Reload model from last checkpoint
        initial_epoch, model = reload_last_checkpoint(
            checkpoint_format, data_config['n_epochs'],
            distributed=args.distributed)
    else:
        # Build a new model
        model = get_model(**config['model'])
        # Configure the optimizer
        opt = get_optimizer(distributed=args.distributed,
                            **config['optimizer'])
        # Compile the model
        model.compile(optimizer=opt, loss=train_config['loss'],
                      metrics=train_config['metrics'])

    if dist.rank == 0:
        model.summary()

    # Save configuration to output directory
    if dist.rank == 0:
        config['n_ranks'] = dist.size
        save_config(config)

    # Prepare the callbacks
    if dist.rank == 0:
        logging.info('Preparing callbacks')
    callbacks = []
    if args.distributed:

        # Broadcast initial variable states from rank 0 to all processes.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

        # Average metrics across workers
        callbacks.append(hvd.callbacks.MetricAverageCallback())

    # Learning rate decay schedule
    if 'lr_schedule' in config:
        global_batch_size = data_config['batch_size'] * dist.size
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(
            get_lr_schedule(global_batch_size=global_batch_size,
                            **config['lr_schedule'])))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Checkpointing and logging from rank 0 only
    if dist.rank == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))
        callbacks.append(tf.keras.callbacks.CSVLogger(
            os.path.join(config['output_dir'], 'history.csv'), append=args.resume))
        if args.tensorboard:
            callbacks.append(tf.keras.callbacks.TensorBoard(
                os.path.join(config['output_dir'], 'tensorboard')))
        callbacks.append(MLPerfLoggingCallback())

    # Early stopping
    patience = train_config.get('early_stopping_patience', None)
    if patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=1e-5, patience=patience, verbose=1))

    # Stopping at specified target
    target_mae = train_config.get('target_mae', None)
    callbacks.append(StopAtTargetCallback(target_max=target_mae))

    if dist.rank == 0:
        logging.debug('Callbacks: %s', callbacks)

    # Train the model
    if dist.rank == 0:
        logging.info('Beginning training')
    fit_verbose = 1 if (args.verbose and dist.rank==0) else 2
    model.fit(datasets['train_dataset'],
              steps_per_epoch=datasets['n_train_steps'],
              epochs=data_config['n_epochs'],
              validation_data=datasets['valid_dataset'],
              validation_steps=datasets['n_valid_steps'],
              callbacks=callbacks,
              initial_epoch=initial_epoch,
              verbose=fit_verbose)

    # Stop MLPerf timer
    if dist.rank == 0:
        mllogger.end(key=mllog.constants.RUN_STOP, metadata={'status': 'success'})

    # Print training summary
    if dist.rank == 0:
        print_training_summary(config['output_dir'], args.print_fom)

    # Finalize
    if dist.rank == 0:
        logging.info('All done!')

if __name__ == '__main__':
    main()
