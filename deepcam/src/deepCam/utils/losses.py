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

import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

class FPLoss(nn.Module):

    def __init__(self, weight, fpw_1=0, fpw_2=0):

        # init superclass
        super(FPLoss, self).__init__()
        
        # collect constants
        self.fpw_1 = fpw_1
        self.fpw_2 = fpw_2

        # instantiate loss
        self.criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float(), reduction='none')


    def forward(self, logit, target):

        # squeeze target
        target = target.squeeze(1)
        
        # get losses and predictions
        losses = self.criterion(logit, target.long())
        preds = torch.max(logit, 1)[1]

        #is fp 1
        is_fp_one = (torch.eq(preds, 1) & torch.ne(target, 1)).float()
        fp_matrix_one = (is_fp_one * self.fpw_1) + 1
        losses = torch.mul(fp_matrix_one, losses)

        #is fp 2
        is_fp_two = (torch.eq(preds, 2) & torch.ne(target, 2)).float()
        fp_matrix_two = (is_fp_two * self.fpw_2) + 1
        losses = torch.mul(fp_matrix_two, losses)

        # average
        loss = torch.mean(losses)

        return loss
        

class CELoss(nn.Module):

    def __init__(self, weight):

        # init superclass
        super(CELoss, self).__init__()

        # instantiate loss
        self.criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float(), reduction='none')


    def forward(self, logit, target):

        # squeeze target
        target = target.squeeze(1)
        
        # get losses and predictions
        losses = self.criterion(logit, target)

        # average
        loss = torch.mean(losses)

        return loss
