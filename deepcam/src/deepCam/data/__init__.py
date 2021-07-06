import os
from glob import glob
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from .cam_hdf5_dataset import CamDataset, peek_shapes_hdf5

# helper function for determining the data shapes
def get_datashapes(pargs, root_dir):
    
    return peek_shapes_hdf5(os.path.join(root_dir, "train"))
    

# helper function to de-clutter the main training script
def get_dataloaders(pargs, root_dir, device, seed, comm_size, comm_rank):
    
    # import only what we need
    train_dir = os.path.join(root_dir, "train")
    train_set = CamDataset(train_dir, 
                           statsfile = os.path.join(root_dir, 'stats.h5'),
                           channels = pargs.channels,
                           allow_uneven_distribution = False,
                           shuffle = True, 
                           preprocess = True,
                           comm_size = 1,
                           comm_rank = 0)
    
    distributed_train_sampler = DistributedSampler(train_set,
                                                   num_replicas = comm_size,
                                                   rank = comm_rank,
                                                   shuffle = True,
                                                   drop_last = True)
    
    train_loader = DataLoader(train_set,
                              pargs.local_batch_size,
                              num_workers = min([pargs.max_inter_threads, pargs.local_batch_size]),
                              sampler = distributed_train_sampler,
                              pin_memory = True,
                              drop_last = True)

    train_size = train_set.global_size

    validation_dir = os.path.join(root_dir, "validation")
    validation_set = CamDataset(validation_dir, 
                                statsfile = os.path.join(root_dir, 'stats.h5'),
                                channels = pargs.channels,
                                allow_uneven_distribution = True,
                                shuffle = False,
                                preprocess = True,
                                comm_size = comm_size,
                                comm_rank = comm_rank)
    
    # use batch size = 1 here to make sure that we do not drop a sample
    validation_loader = DataLoader(validation_set,
                                   1,
                                   num_workers = min([pargs.max_inter_threads, pargs.local_batch_size]),
                                   pin_memory = True,
                                   drop_last = False)
    
    validation_size = validation_set.global_size    
        
    return train_loader, train_size, validation_loader, validation_size
