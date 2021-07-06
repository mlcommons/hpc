# The MIT License (MIT)
#
# Copyright (c) 2018 Pyjcsx
# Modifications Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import glob
import h5py as h5
import numpy as np
import math
from time import sleep

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def peek_shapes_hdf5(data_dir):
    files = glob.iglob(os.path.join(data_dir, "*.h5"))
    with h5.File(next(files), "r") as fin:
        data_shape = fin["climate"]["data"].shape
        label_shape = fin["climate"]["labels_0"].shape
        
    return data_shape, label_shape


#dataset class
class CamDataset(Dataset):
  
    def init_reader(self):
        #shuffle
        if self.shuffle:
            self.rng.shuffle(self.all_files)
            
        #shard the dataset
        self.global_size = len(self.all_files)
        if self.allow_uneven_distribution:
            # this setting covers the data set completely

            # deal with bulk
            num_files_local = self.global_size // self.comm_size
            start_idx = self.comm_rank * num_files_local
            end_idx = start_idx + num_files_local
            self.files = self.all_files[start_idx:end_idx]

            # deal with remainder
            for idx in range(self.comm_size * num_files_local, self.global_size):
                if idx % self.comm_size == self.comm_rank:
                    self.files.append(self.all_files[idx])
        else:
            # here, every worker gets the same number of samples, 
            # potentially under-sampling the data
            num_files_local = self.global_size // self.comm_size
            start_idx = self.comm_rank * num_files_local
            end_idx = start_idx + num_files_local
            self.files = self.all_files[start_idx:end_idx]
            self.global_size = self.comm_size * len(self.files)
            
        #my own files
        self.local_size = len(self.files)

        #print sizes
        #print("Rank {} local size {} (global {})".format(self.comm_rank, self.local_size, self.global_size))

  
    def __init__(self, source, statsfile, channels,
                 allow_uneven_distribution = False,
                 shuffle = False,
                 preprocess = True,
                 transpose = True,
                 comm_size = 1, comm_rank = 0, seed = 12345):
        
        self.source = source
        self.statsfile = statsfile
        self.channels = channels
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.transpose = transpose
        self.all_files = sorted( [ os.path.join(self.source,x) for x in os.listdir(self.source) if x.endswith('.h5') ] )
        self.comm_size = comm_size
        self.comm_rank = comm_rank
        self.allow_uneven_distribution = allow_uneven_distribution
        
        #split list of files
        self.rng = np.random.RandomState(seed)
        
        #init reader
        self.init_reader()

        #get shapes
        filename = os.path.join(self.source, self.files[0])
        with h5.File(filename, "r") as fin:
            self.data_shape = fin['climate']['data'].shape
            self.label_shape = fin['climate']['labels_0'].shape
        
        #get statsfile for normalization
        #open statsfile
        with h5.File(self.statsfile, "r") as f:
            data_shift = f["climate"]["minval"][self.channels]
            data_scale = 1. / ( f["climate"]["maxval"][self.channels] - data_shift )

        #reshape into broadcastable shape
        self.data_shift = np.reshape( data_shift, (1, 1, data_shift.shape[0]) ).astype(np.float32)
        self.data_scale = np.reshape( data_scale, (1, 1, data_scale.shape[0]) ).astype(np.float32)

        if comm_rank == 0:
            print("Initialized dataset with ", self.global_size, " samples.")


    def __len__(self):
        return self.local_size


    @property
    def shapes(self):
        return self.data_shape, self.label_shape


    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])

        #load data and project
        with h5.File(filename, "r") as f:
            data = f["climate/data"][..., self.channels]
            label = f["climate/labels_0"][...].astype(np.int64)
        
        #preprocess
        data = self.data_scale * (data - self.data_shift)

        if self.transpose:
            #transpose to NCHW
            data = np.transpose(data, (2,0,1))
        
        return data, label, filename
