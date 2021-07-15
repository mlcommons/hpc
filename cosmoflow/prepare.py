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
Data preparation script which reads HDF5 files and produces either split HDF5
files or TFRecord files.
"""

# System
import os
import argparse
import logging
import multiprocessing as mp
from functools import partial

# Externals
import h5py
import numpy as np
import tensorflow as tf

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir',
        default='/project/projectdirs/m3363/www/cosmoUniverse_2019_05_4parE')
    parser.add_argument('-o', '--output-dir',
        default='/global/cscratch1/sd/sfarrell/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--sample-size', type=int, default=128)
    parser.add_argument('--write-tfrecord', action='store_true',
                        help='Enable writing tfrecord files, otherwise split hdf5s')
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--n-tasks', type=int, default=1)
    parser.add_argument('--gzip', action='store_true')
    return parser.parse_args()

def find_files(input_dir, max_files=None):
    files = []
    for subdir, _, subfiles in os.walk(input_dir):
        for f in subfiles:
            if f.endswith('hdf5'):
                files.append(os.path.join(subdir, f))
    files = sorted(files)
    if max_files is not None:
        files = files[:max_files]
    return files

def read_hdf5(file_path):
    with h5py.File(file_path, mode='r') as f:
        x = f['full'][:]
        y = f['unitPar'][:]
    return x, y

def split_universe(x, size):
    """Generator function for iterating over the sub-universes"""
    n = x.shape[0] // size
    # Loop over each split
    for xi in np.split(x, n, axis=0):
        for xij in np.split(xi, n, axis=1):
            for xijk in np.split(xij, n, axis=2):
                yield xijk

def write_record(output_file, example, compression=None):
    with tf.io.TFRecordWriter(output_file, options=compression) as writer:
        writer.write(example.SerializeToString())

def write_hdf5(output_file, x, y, compression=None):
    with h5py.File(output_file, mode='w') as f:
        f.create_dataset('x', data=x, compression=compression)
        f.create_dataset('y', data=y)

def process_file(input_file, output_dir, sample_size, write_tfrecord,
                 compression=False):
    logging.info('Reading %s', input_file)

    # Load the data
    x, y = read_hdf5(input_file)

    # Loop over sub-volumes
    for i, xi in enumerate(split_universe(x, sample_size)):

        # Output file name pattern. To avoid name collisions,
        # we prepend the subdirectory name to the output file name.
        # We also append the subvolume index
        subdir = os.path.basename(os.path.dirname(input_file))
        output_file_prefix = os.path.join(
            output_dir,
            subdir + '_' + os.path.basename(input_file).replace('.hdf5', '_%03i' % i)
        )

        if write_tfrecord:

            # Convert to TF example
            feature_dict = dict(
                x=tf.train.Feature(bytes_list=tf.train.BytesList(value=[xi.tostring()])),
                #x=tf.train.Feature(float_list=tf.train.FloatList(value=xi.flatten())),
                y=tf.train.Feature(float_list=tf.train.FloatList(value=y)))
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

            # Determine output file name
            output_file = output_file_prefix + '.tfrecord'

            # Write the output file
            logging.info('Writing %s', output_file)
            compression_type = 'GZIP' if compression else None
            write_record(output_file, tf_example, compression=compression_type)

        else:

            # Just write a new HDF5 file
            output_file = output_file_prefix + '.hdf5'
            logging.info('Writing %s', output_file)
            compression_type = 'gzip' if compression else None
            write_hdf5(output_file, xi, y, compression=compression_type)

def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Select my subset of input files
    input_files = find_files(args.input_dir, max_files=args.max_files)
    input_files = np.array_split(input_files, args.n_tasks)[args.task]

    # Process input files with a worker pool
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_file, output_dir=args.output_dir,
                               sample_size=args.sample_size,
                               write_tfrecord=args.write_tfrecord,
                               compression=args.gzip)
        pool.map(process_func, input_files)

    logging.info('All done!')

if __name__ == '__main__':
    main()
