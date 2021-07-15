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

import torch
from torch import Tensor


def compute_score(prediction: Tensor, gt: Tensor, num_classes: int) -> Tensor:
    # flatten input
    batch_size = gt.shape[0]
    tpt = torch.zeros((batch_size, num_classes), dtype=torch.long, device=prediction.device)
    fpt = torch.zeros((batch_size, num_classes), dtype=torch.long, device=prediction.device)
    fnt = torch.zeros((batch_size, num_classes), dtype=torch.long, device=prediction.device)
    
    # create views:
    pv = prediction.view(batch_size, -1)
    gtv = gt.view(batch_size, -1)
    
    # compute per class accuracy
    for j in range(0, num_classes):
        # compute helper tensors
        pv_eq_j = (pv == j)
        pv_ne_j = (pv != j)
        gtv_eq_j = (gtv == j)
        gtv_ne_j = (gtv != j)
        
        #true positve: prediction and gt agree and gt is of class j: (p == j) & (g == j)
        tpt[:, j] = torch.sum(torch.logical_and(pv_eq_j, gtv_eq_j), dim=1)
        
        #false positive: prediction is of class j and gt not of class j: (p == j) & (g != j)
        fpt[:, j] = torch.sum(torch.logical_and(pv_eq_j, gtv_ne_j), dim=1)

        #false negative: prediction is not of class j and gt is of class j: (p != j) & (g == j)
        fnt[:, j] = torch.sum(torch.logical_and(pv_ne_j, gtv_eq_j), dim=1)
        
    # compute IoU per batch
    uniont = (tpt + fpt + fnt) * num_classes
    iout = torch.sum(torch.nan_to_num(tpt.float() / uniont.float(), nan=1./float(num_classes)), dim=1)
        
    # average over batch dim
    iout = torch.mean(iout)
    
    return iout
