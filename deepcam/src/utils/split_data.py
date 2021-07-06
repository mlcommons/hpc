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

import os
import numpy as np

#filedir="/gpfs/fs1/tkurth/cam5_dataset/All-Hist/all"
inputdir="/global/cscratch1/sd/tkurth/data/cam5_data/All-Hist/all"
outputdir="/global/cscratch1/sd/tkurth/data/cam5_data/All-Hist"
#inputdir="/data/all"
#outputdir="/data"
train_fraction=0.8
validation_fraction=0.1
test_fraction=0.1
seed=12345

#set rng
np.random.seed(seed)

#get files
files = sorted([ x for x in os.listdir(inputdir) if x.startswith("data") and x.endswith(".h5") ])

#shuffle files
np.random.shuffle(files)

#print splits
num_all = len(files)
num_train = int(len(files) * train_fraction)
num_validation = int(len(files) * validation_fraction)
num_test = num_all - num_train - num_validation

print("Following split will be used: ")
print("Total files: {}".format(num_all))
print("Train files: {}".format(num_train))
print("Validation files: {}".format(num_validation))
print("Test files: {}".format(num_test))

#get train chunk
start = 0
end = num_train
train_files = files[start:end]

#get validation chunk
start = num_train
end = num_train + num_validation
validation_files = files[start:end]

#get rest for test
start = num_train+ num_validation
test_files = files[start:]

#create symbolic links for all of these
#training
traindir=os.path.join(outputdir, "train")
if not os.path.isdir(traindir):
    os.makedirs(traindir)
for f in train_files:
    src = os.path.join(inputdir, f)
    dst = os.path.join(traindir, f)
    os.symlink(src, dst)

#validation
validationdir=os.path.join(outputdir, "validation")
if not os.path.isdir(validationdir):
    os.makedirs(validationdir)
for f in validation_files:
    src = os.path.join(inputdir, f)
    dst = os.path.join(validationdir, f)
    os.symlink(src, dst)

#test
testdir=os.path.join(outputdir, "test")
if not os.path.isdir(testdir):
    os.makedirs(testdir)
for f in test_files:
    src = os.path.join(inputdir, f)
    dst = os.path.join(testdir, f)
    os.symlink(src, dst)
                        
        
