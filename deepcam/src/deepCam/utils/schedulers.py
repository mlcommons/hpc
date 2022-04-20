import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter

class MultiStepLRWarmup(_LRScheduler):

    @torch.jit.ignore
    def __init__(self, optimizer, warmup_steps, warmup_factor,
                 milestones, gamma=0.1, last_epoch=-1, verbose=False):
        
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.warmup_slope = 1./float(self.warmup_steps) if self.warmup_steps > 0 else 1.
        self.milestones = Counter([x + self.warmup_steps + 1 for x in milestones])
        self.gamma = gamma
        super(MultiStepLRWarmup, self).__init__(optimizer, last_epoch, verbose)

    @torch.jit.export
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # compute LR
        if self.last_epoch >= self.warmup_steps:
            # decay phase
            if self.last_epoch not in self.milestones:
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                    for group in self.optimizer.param_groups] 

        else:
            # linear warmup phase
            if self.warmup_factor == 1.0:
                return [base_lr * (float(self.last_epoch) * self.warmup_slope)
                        for base_lr in self.base_lrs]
            else:
                return [base_lr * ((self.warmup_factor - 1.) * float(self.last_epoch) * self.warmup_slope + 1.)
                        for base_lr in self.base_lrs]



class CosineAnnealingLRWarmup(_LRScheduler):

    @torch.jit.ignore
    def __init__(self, optimizer, warmup_steps, warmup_factor,
                 T_max, eta_min=0, last_epoch=-1, verbose=False):
        
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.warmup_slope = 1./float(self.warmup_steps) if self.warmup_steps > 0 else 1.
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLRWarmup, self).__init__(optimizer, last_epoch, verbose)

    @torch.jit.export     
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # compute LR
        if self.last_epoch >= self.warmup_steps:
            # cosine phase
            last_epoch_eff = self.last_epoch - self.warmup_steps
            if last_epoch_eff == 0:
                return [group['lr'] for group in self.optimizer.param_groups]
            elif (last_epoch_eff - 1 - self.T_max) % (2 * self.T_max) == 0:
                return [group['lr'] + (base_lr - self.eta_min) *
                        (1 - math.cos(math.pi / self.T_max)) / 2
                        for base_lr, group in
                        zip(self.base_lrs, self.optimizer.param_groups)]
            return [(1 + math.cos(math.pi * last_epoch_eff / self.T_max)) /
                    (1 + math.cos(math.pi * (last_epoch_eff - 1) / self.T_max)) *
                    (group['lr'] - self.eta_min) + self.eta_min
                    for group in self.optimizer.param_groups]

        else:
            # linear warmup phase
            if self.warmup_factor == 1.0:
                return [base_lr * (float(self.last_epoch) * self.warmup_slope)
                        for base_lr in self.base_lrs]
            else:
                return [base_lr * ((self.warmup_factor - 1.) * float(self.last_epoch) * self.warmup_slope + 1.)
                        for base_lr in self.base_lrs]
