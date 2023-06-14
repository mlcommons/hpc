# Copyright 2023 NVIDIA CORPORATION
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.04-py3

FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive

# Install pip requirements:
RUN pip install \
    biopython==1.79 \
    Pympler==1.0.1 \
    dacite==1.8.0 \
    "git+https://github.com/mlcommons/logging.git@2.1.0" \
    "git+https://github.com/NVIDIA/mlperf-common.git"

# Build and install Kalign from source:
RUN wget -q -P /workspace/downloads https://github.com/TimoLassmann/kalign/archive/refs/tags/v3.3.5.tar.gz \
    && tar -xzf /workspace/downloads/v3.3.5.tar.gz --directory /workspace \
    && rm -r /workspace/downloads \
    && ls /workspace \
    && cd /workspace/kalign-3.3.5 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j \
    && make install \
    && rm -r /workspace/kalign-3.3.5

# Copy OpenFold source code into the docker image:
COPY . /workspace/openfold
WORKDIR /workspace/openfold

# Install OpenFold source code package in editable mode:
RUN pip install -e .
