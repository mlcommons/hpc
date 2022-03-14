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

# base stuff
import os

# torch
import torch
import torch.distributed as dist

# custom stuff
from utils import metric


def train_epoch(pargs, comm_rank, comm_size,
                device, step, epoch, 
                net, criterion, 
                optimizer, scheduler,
                train_loader,
                logger):

    # make sure net is set to train
    net.train()

    # get LR
    if pargs.lr_schedule:
        current_lr = scheduler.get_last_lr()[0]
    else:
        current_lr = pargs.start_lr
    
    # do the training loop
    for inputs, label, filename in train_loader:
    
        # send to device
        inputs = inputs.to(device)
        label = label.to(device)
    
        # forward pass
        outputs = net.forward(inputs)
        loss = criterion(outputs, label) / float(pargs.gradient_accumulation_frequency)

        # backward pass
        loss.backward()
    
        # optimizer step if requested
        if (step + 1) % pargs.gradient_accumulation_frequency == 0:
            optimizer.step()
            optimizer.zero_grad()

            # do a scheduler step if relevant
            if pargs.lr_schedule:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
    
        # increase step counter
        step += 1
        
        #log if requested
        if (pargs.logging_frequency > 0) and (step % pargs.logging_frequency == 0):
    
            # allreduce for loss
            loss_avg = loss.detach()
            if dist.is_initialized():
                dist.reduce(loss_avg, dst=0, op=dist.ReduceOp.SUM)
            loss_avg_train = loss_avg.item() / float(comm_size)
    
            # Compute score
            predictions = torch.argmax(torch.softmax(outputs, 1), 1)
            iou = metric.compute_score(predictions, label, num_classes=3)
            iou_avg = iou.detach()
            if dist.is_initialized():
                dist.reduce(iou_avg, dst=0, op=dist.ReduceOp.SUM)
            iou_avg_train = iou_avg.item() / float(comm_size)

            # log values
            logger.log_event(key = "learning_rate", value = current_lr, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_accuracy", value = iou_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})
            logger.log_event(key = "train_loss", value = loss_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})

    # end of epoch logging
    # allreduce for loss
    loss_avg = loss.detach()
    if dist.is_initialized():
        dist.reduce(loss_avg, dst=0, op=dist.ReduceOp.SUM)
    loss_avg_train = loss_avg.item() / float(comm_size)

    # Compute score
    predictions = torch.argmax(torch.softmax(outputs, 1), 1)
    iou = metric.compute_score(predictions, label, num_classes=3)
    iou_avg = iou.detach()
    if dist.is_initialized():
        dist.reduce(iou_avg, dst=0, op=dist.ReduceOp.SUM)
    iou_avg_train = iou_avg.item() / float(comm_size)
        
    # log values
    logger.log_event(key = "learning_rate", value = current_lr, metadata = {'epoch_num': epoch+1, 'step_num': step})
    logger.log_event(key = "train_accuracy", value = iou_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})
    logger.log_event(key = "train_loss", value = loss_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step}) 
    
    return step
