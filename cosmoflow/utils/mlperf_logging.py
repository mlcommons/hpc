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
Utilities for MLPerf logging.
Depends on the mlperf_logging package at
https://github.com/mlperf/logging
"""

# System
import os

# Externals
try:
    from mlperf_logging import mllog
    have_mlperf_logging = True
except ImportError:
    have_mlperf_logging = False


def configure_mllogger(log_dir):
    """Setup the MLPerf logger"""
    if not have_mlperf_logging:
        raise RuntimeError('mlperf_logging package unavailable')
    mllog.config(filename=os.path.join(log_dir, 'mlperf.log'))
    return mllog.get_mllogger()


def log_submission_info(benchmark='cosmoflow',
                        org='UNDEFINED',
                        division='UNDEFINED',
                        status='UNDEFINED',
                        platform='UNDEFINED'):
    """Log general MLPerf submission details from config"""
    mllogger = mllog.get_mllogger()
    mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value=benchmark)
    mllogger.event(key=mllog.constants.SUBMISSION_ORG, value=org)
    mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value=division)
    mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value=status)
    mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value=platform)
