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

# Basics
import os
import numpy as np
import datetime as dt
import subprocess as sp

# logging
import utils.mlperf_log_utils as mll

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable

# Custom
from driver import train_epoch, validate
from utils import parser
from utils import losses
from utils import optimizer_helpers as oh
from utils import bnstats as bns
from data import get_dataloaders, get_datashapes
from architecture import deeplab_xception

# DDP
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

#comm wrapper
from utils import comm

#main function
def main(pargs):

    #init distributed training
    comm_local_group = comm.init(pargs.wireup_method, pargs.batchnorm_group_size)
    comm_rank = comm.get_rank()
    comm_local_rank = comm.get_local_rank()
    comm_size = comm.get_size()
    comm_local_size = comm.get_local_size()
    
    # set up logging
    pargs.logging_frequency = max([pargs.logging_frequency, 0])
    log_file = os.path.normpath(os.path.join(pargs.output_dir, "logs", pargs.run_tag + ".log"))
    logger = mll.mlperf_logger(log_file, "deepcam", "Umbrella Corp.")
    logger.log_start(key = "init_start", sync = True)        
    logger.log_event(key = "cache_clear")
    
    #set seed
    seed = pargs.seed
    logger.log_event(key = "seed", value = seed)
    
    # Some setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda", comm_local_rank)
        torch.cuda.manual_seed(seed)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        
    #set up directories
    root_dir = os.path.join(pargs.data_dir_prefix)
    output_dir = pargs.output_dir
    plot_dir = os.path.join(output_dir, "plots")
    if comm_rank == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    # logging of rank information
    logger.log_event(key = "number_of_ranks", value = comm_size)
    logger.log_event(key = "number_of_nodes", value = (comm_size // comm_local_size))
    logger.log_event(key = "accelerators_per_node", value = comm_local_size)
    
    # Logging hyperparameters
    logger.log_event(key = "global_batch_size", value = (pargs.local_batch_size * comm_size))
    logger.log_event(key = "batchnorm_group_size", value = pargs.batchnorm_group_size)
    logger.log_event(key = "gradient_accumulation_frequency", value = pargs.gradient_accumulation_frequency)
    logger.log_event(key = "checkpoint", value = pargs.checkpoint)

    # Define architecture
    n_input_channels = len(pargs.channels)
    n_output_channels = 3
    net = deeplab_xception.DeepLabv3_plus(n_input = n_input_channels, 
                                          n_classes = n_output_channels, 
                                          os=16, pretrained=False, 
                                          rank = comm_rank,
                                          process_group = comm_local_group)
    net.to(device)
    
    #select loss
    #some magic numbers
    loss_pow = -0.125
    class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
    # extract loss
    criterion = losses.CELoss(class_weights).to(device)
    criterion = torch.jit.script(criterion)

    #select optimizer
    optimizer = oh.get_optimizer(pargs, net, logger)

    #restart from checkpoint if desired
    if pargs.checkpoint is not None:
        checkpoint = torch.load(pargs.checkpoint, map_location = device)
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['model'])
    else:
        start_step = 0
        start_epoch = 0   

    #broadcast model and optimizer state
    steptens = torch.tensor(np.array([start_step, start_epoch]), requires_grad=False).to(device)
    if dist.is_initialized():
        dist.broadcast(steptens, src = 0)
        
    #unpack the bcasted tensor
    start_step = int(steptens.cpu().numpy()[0])
    start_epoch = int(steptens.cpu().numpy()[1])  
                                    
    #select scheduler
    scheduler = None
    if pargs.lr_schedule:
        pargs.lr_schedule["lr_warmup_steps"] = pargs.lr_warmup_steps
        pargs.lr_schedule["lr_warmup_factor"] = pargs.lr_warmup_factor
        scheduler = oh.get_lr_schedule(pargs.start_lr, pargs.lr_schedule, optimizer, logger, last_step = start_step)
    
    # print parameters
    if comm_rank == 0:
        print(net)
        print("Total number of elements:", sum(p.numel() for p in net.parameters() if p.requires_grad))
        
    # get input shapes for the upcoming model preprocessing
    # input_shape:
    tshape, _ = get_datashapes(pargs, root_dir)
    input_shape = tuple([tshape[2], tshape[0], tshape[1]])
    
    #distributed model parameters
    bucket_cap_mb = 25
    if pargs.batchnorm_group_size > 1:
        bucket_cap_mb = 220
    
    # get stream, relevant for graph capture
    ddp_net = DDP(net, device_ids=[device.index],
                  output_device=device.index,
                  find_unused_parameters=False,
                  broadcast_buffers=False,
                  bucket_cap_mb=bucket_cap_mb,
                  gradient_as_bucket_view=False)

    # get stats handler here
    bnstats_handler = bns.BatchNormStatsSynchronize(ddp_net, reduction = "mean", inplace = True)
    
    # create handles
    net_validate = ddp_net
    net_train = ddp_net
        
    # Set up the data feeder
    train_loader, train_size, validation_loader, validation_size = get_dataloaders(pargs, root_dir, device, seed, comm_size, comm_rank)
    
    # log size of datasets
    logger.log_event(key = "train_samples", value = train_size)
    val_size = validation_size
    logger.log_event(key = "eval_samples", value = val_size)

    # get start steps
    step = start_step
    epoch = start_epoch
    current_lr = pargs.start_lr if not pargs.lr_schedule else scheduler.get_last_lr()[0]
    stop_training = False
    net_train.train()

    # start trining
    logger.log_end(key = "init_stop", sync = True)
    logger.log_start(key = "run_start", sync = True)

    # training loop
    while True:

        # start epoch
        logger.log_start(key = "epoch_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=True)

        train_loader.sampler.set_epoch(epoch)

        # training
        step = train_epoch(pargs, comm_rank, comm_size,
                           device, step, epoch, 
                           net_train, criterion, 
                           optimizer, scheduler,
                           train_loader,
                           logger)

        # average BN stats
        bnstats_handler.synchronize()
        
        # validation
        stop_training = validate(pargs, comm_rank, comm_size,
                                 device, step, epoch, 
                                 net_validate, criterion, validation_loader, 
                                 logger)

        # log the epoch
        logger.log_end(key = "epoch_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
        epoch += 1
        
        #save model if desired
        if (pargs.save_frequency > 0) and (epoch % pargs.save_frequency == 0):
            logger.log_start(key = "save_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
            if comm_rank == 0:
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'model': net_train.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(output_dir, pargs.model_prefix + "_step_" + str(step) + ".cpt") )
                logger.log_end(key = "save_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
                    
        # are we done?
        if (epoch >= pargs.max_epochs) or stop_training:
            break

    # run done
    logger.log_end(key = "run_stop", sync = True, metadata = {'status' : 'success'})


if __name__ == "__main__":

    #arguments
    pargs = parser.parse_arguments()
    
    #run the stuff
    main(pargs)
