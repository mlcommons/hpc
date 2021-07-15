# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

import collections
import ctypes
import logging.config
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.init as init
import torch.utils.collect_env
from mlperf_logging.mllog import constants
from mlperf_logging import mllog

#comm wrapper
from utils import comm

class mlperf_logger(object):

    def __init__(self, filename, benchmark, organization):
        self.mllogger = mllog.get_mllogger()
        self.comm_rank = comm.get_rank()
        self.comm_size = comm.get_size()
        self.constants = constants

        # create logging dir if it does not exist
        logdir = os.path.dirname(filename)
        if self.comm_rank == 0:
            if not os.path.isdir(logdir):
                os.makedirs(logdir)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # create config
        mllog.config(filename = filename)
        self.mllogger.logger.propagate = False
        self.log_event(key = constants.SUBMISSION_BENCHMARK,
                       value = benchmark)

        self.log_event(key = constants.SUBMISSION_ORG,
                       value = organization)
        
        self.log_event(key = constants.SUBMISSION_DIVISION,
                       value = 'closed')

        self.log_event(key = constants.SUBMISSION_STATUS,
                       value = 'onprem')

        self.log_event(key = constants.SUBMISSION_PLATFORM,
                       value = f'{self.comm_size}xSUBMISSION_PLATFORM_PLACEHOLDER')
        

    def log_start(self, *args, **kwargs):
        self._log_print(self.mllogger.start, *args, **kwargs)
        
    def log_end(self, *args, **kwargs):
        self._log_print(self.mllogger.end, *args, **kwargs)
        
    def log_event(self, *args, **kwargs):
        self._log_print(self.mllogger.event, *args, **kwargs)

    def _log_print(self, logger, *args, **kwargs):
        """
        Wrapper for MLPerf compliance logging calls.
        All arguments but 'sync' and 'log_all_ranks' are passed to
        mlperf_logging.mllog.
        If 'sync' is set to True then the wrapper will synchronize all distributed
        workers. 'sync' should be set to True for all compliance tags that require
        accurate timing (RUN_START, RUN_STOP etc.)
        If 'log_all_ranks' is set to True then all distributed workers will print
        logging message, if set to False then only worker with rank=0 will print
        the message.
        """
        if kwargs.pop('sync', False):
            self.barrier()
        if 'stack_offset' not in kwargs:
            kwargs['stack_offset'] = 3
        if 'value' not in kwargs:
            kwargs['value'] = None

        if kwargs.pop('log_all_ranks', False):
            log = True
        else:
            log = (self.comm_rank == 0)

        if log:
            logger(*args, **kwargs)

    def barrier(self):
        """
        Works as a temporary distributed barrier, currently pytorch
        doesn't implement barrier for NCCL backend.
        Calls all_reduce on dummy tensor and synchronizes with GPU.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
                    

