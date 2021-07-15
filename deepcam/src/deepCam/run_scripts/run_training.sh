#!/bin/bash

# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

# parameters
data_dir=""
output_dir=""
run_tag="test_run"
local_batch_size=2

python ./train.py \
       --wireup_method "dummy" \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir} \
       --output_dir ${output_dir} \
       --model_prefix "segmentation" \
       --optimizer "LAMB" \
       --adam_eps 1e-6 \
       --start_lr 0.0055 \
       --lr_schedule type="multistep",milestones="800",decay_rate="0.1" \
       --lr_warmup_steps 400 \
       --lr_warmup_factor 1. \
       --weight_decay 1e-2 \
       --logging_frequency 10 \
       --save_frequency 0 \
       --max_epochs 200 \
       --max_inter_threads 4 \
       --seed $(date +%s) \
       --batchnorm_group_size 1 \
       --local_batch_size ${local_batch_size}
