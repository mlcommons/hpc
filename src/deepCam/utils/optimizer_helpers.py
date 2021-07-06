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

import re
import numpy as np
import torch
import torch.optim as optim

try:
    import apex.optimizers as aoptim
    have_apex = True
except ImportError:
    print("NVIDIA APEX not found")
    have_apex = False

#warmup scheduler
have_warmup_scheduler = False
try:
    from warmup_scheduler import GradualWarmupScheduler
    have_warmup_scheduler = True
except ImportError:
    pass 
    
    
def get_lr_schedule(start_lr, scheduler_arg, optimizer, logger, last_step = -1):
    #add the initial_lr to the optimizer
    for pgroup in optimizer.param_groups:
        pgroup["initial_lr"] = start_lr

    # after-scheduler
    scheduler_after = None
    
    #now check
    if scheduler_arg["type"] == "multistep":
        # preprocess and set the parameters
        milestones = [ int(x) for x in scheduler_arg["milestones"].split() ]
        gamma = float(scheduler_arg["decay_rate"])
        
        # create scheduler
        scheduler_after = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma = gamma, last_epoch = last_step)

        # save back the parameters for better logging
        scheduler_arg["milestones"] = milestones
        scheduler_arg["decay_rate"] = gamma
    
    elif scheduler_arg["type"] == "cosine_annealing":
        # set parameters
        t_max = int(scheduler_arg["t_max"])
        eta_min = 0. if "eta_min" not in scheduler_arg else float(scheduler_arg["eta_min"])

        # create scheduler
        scheduler_after =  optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = t_max, eta_min = eta_min)

        # save back the parameters for better logging
        scheduler_arg["t_max"] = t_max
        scheduler_arg["eta_min"] = eta_min
    
    else:
        raise ValueError("Error, scheduler type {} not supported.".format(scheduler_arg["type"]))

    # LR warmup
    if scheduler_arg["lr_warmup_steps"] > 0:
        if have_warmup_scheduler:
            scheduler = GradualWarmupScheduler(optimizer, multiplier=scheduler_arg["lr_warmup_factor"],
                                               total_epoch=scheduler_arg["lr_warmup_steps"],
                                               after_scheduler=scheduler_after)
            
        # Throw an error if the package is not found
        else:
            raise Exception(f'Requested {pargs.lr_warmup_steps} LR warmup steps '
                            'but warmup scheduler not found. Install it from '
                            'https://github.com/ildoonet/pytorch-gradual-warmup-lr')
    else:
        scheduler = scheduler_after

    # log scheduler data
    for key in scheduler_arg:
        logger.log_event(key = "scheduler_" + key, value = scheduler_arg[key])

    return scheduler


                     
def get_optimizer(pargs, net, logger):
    # these should be constant
    defaults = {"adam_eps": 1e-6}
    
    optimizer = None
    if pargs.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(),
                               lr = pargs.start_lr,
                               betas = pargs.optimizer_betas,
                               eps = defaults["adam_eps"],
                               weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(),
                                lr = pargs.start_lr,
                                betas = pargs.optimizer_betas,
                                eps = defaults["adam_eps"],
                                weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "LAMB":
        if have_apex:
            optimizer = aoptim.FusedLAMB(net.parameters(),
                                         lr = pargs.start_lr,
                                         betas = pargs.optimizer_betas,
                                         eps = defaults["adam_eps"],
                                         weight_decay = pargs.weight_decay,
                                         set_grad_none = True)
        else:
            raise NotImplementedError("Error, optimizer LAMB requires APEX")
    else:
        raise NotImplementedError("Error, optimizer {} not supported".format(pargs.optimizer))

    # log the optimizer parameters
    logger.log_event(key = "opt_name", value = pargs.optimizer)
    paramgroup = optimizer.param_groups[0]
    for key in [x for x in paramgroup if x != "params"]:
        logger.log_event(key = "opt_" + key, value = paramgroup[key])
    
    return optimizer
