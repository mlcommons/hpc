import torch
import torch.nn as nn
import torch.distributed as dist

class BatchNormStatsSynchronize(nn.Module):

    def __init__(self, model, reduction = "mean", inplace = False, normalize_before_add = False):
        super(BatchNormStatsSynchronize, self).__init__()
        
        # store args
        self.reduction = reduction
        self.inplace = inplace
        self.normalize_before_add = normalize_before_add
        
        # get world size
        self.comm_size = 1
        if dist.is_initialized():
            self.comm_size = dist.get_world_size()
        
        # create tensor list
        self.paramlist = []
        self.paramcount = 0
        for m in model.modules():
            if hasattr(m, "running_mean"):
                self.paramlist.append(m.running_mean)
                self.paramcount += m.running_mean.numel()
            if hasattr(m, "running_var"):
                self.paramlist.append(m.running_var)
                self.paramcount += m.running_var.numel()
        
        assert self.paramcount > 0
        
        self.device = self.paramlist[0].device
        
        # create big tensor which holds all stats
        self.buffer = torch.zeros((self.paramcount), 
                                   dtype = self.paramlist[0].dtype, 
                                   device = self.device,
                                   requires_grad = False)
        
        # create views to parameter buffers
        self.paramviews = []
        offset = 0
        for param in self.paramlist:
            numel = param.numel()
            self.paramviews.append(self.buffer[offset:offset+numel].view(numel))
            offset += numel
        
        # replace the oeriginal parameters with views to the new buffer
        if self.inplace:
            idx = 0
            for m in model.modules():
                if hasattr(m, "running_mean"):
                    #m.running_mean = self.paramviews[idx]
                    del m.running_mean
                    m.register_buffer("running_mean", self.paramviews[idx], persistent=True)
                    self.paramviews[idx].fill_(0.)
                    idx += 1
                if hasattr(m, "running_var"):
                    #m.running_var = self.paramviews[idx]
                    del m.running_var
                    m.register_buffer("running_var", self.paramviews[idx], persistent=True)
                    self.paramviews[idx].fill_(1.)
                    idx += 1


    def _copy_params_to_buffer(self):
        for idx, param in enumerate(self.paramlist):
            self.paramviews[idx].copy_(param)


    def _copy_params_from_buffer(self):
        for idx, param in enumerate(self.paramlist):
            param.copy_(self.paramviews[idx])


    def synchronize(self):
        if dist.is_initialized():
            # sync the device before
            torch.cuda.synchronize()
            
            with torch.no_grad():
                if not self.inplace:
                    self._copy_params_to_buffer()
        
                if self.reduction == "mean":
                    # normalize before?
                    if self.normalize_before_add:
                        self.buffer /= float(self.comm_size)

                    # sum
                    dist.all_reduce(self.buffer, op=dist.ReduceOp.SUM)

                    # normalize after?
                    if not self.normalize_before_add:
                        self.buffer /= float(self.comm_size)
                else:
                    raise NotImplementedError(f"Error, reduction {self.reduction} not supported.")
            
                if not self.inplace:
                    self._copy_params_from_buffer()

