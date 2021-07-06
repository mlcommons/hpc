# Deep Learning Climate Segmentation Benchmark

PyTorch implementation for the climate segmentation benchmark, based on the
Exascale Deep Learning for Climate Analytics codebase here:
https://github.com/azrael417/ClimDeepLearn, and the paper:
https://arxiv.org/abs/1810.01993

## Dataset

The dataset for this benchmark comes from CAM5 [1] simulations and is hosted at
NERSC. The samples are stored in HDF5 files with input images of shape
(768, 1152, 16) and pixel-level labels of shape (768, 1152). The labels have
three target classes (background, atmospheric river, tropical cycline) and were
produced with TECA [2].

The current recommended way to get the data is to use GLOBUS and the following
globus endpoint:

https://app.globus.org/file-manager?origin_id=0b226e2c-4de0-11ea-971a-021304b0cca7&origin_path=%2F

The dataset folder contains a README with some technical description of the
dataset and an All-Hist folder containing all of the data files.

### Preprocessing
The dataset is split into train/val/test and ships with the `stats.h5` file containing summary statistics.

## Before you run

Make sure you have a working python environment with `pytorch` and `h5py` setup. 
If you want to use learning rate warmup, you must also install the warmup-scheduler package
available at https://github.com/ildoonet/pytorch-gradual-warmup-lr.

## How to run the benchmark

Submission example scripts are in `src/deepCam/run_scripts`.

## Hyperparameters

The table below contains the modifiable hyperparameters. Unless otherwise stated, parameters not
listed in the table below are fixed and changing those could lead to an invalid submission.

|Parameter Name |Default | Constraints | Description|
--- | --- | --- | ---
`--optimizer` | `"Adam"` | Optimizer of Adam or LAMB* type. This benchmark implements `"Adam"` and `"AdamW"` from PyTorch as well as `"FusedLAMB"` from NVIDIA APEX. Algorithmic equivalent implementations to those listed before are allowed. | The optimizer to choose
`--start_lr` | 1e-3 | >= 0. | Start learning rate (or base learning rate if warmup is used)
`--optimizer_betas` | `[0.9, 0.999]` | N/A | Momentum terms for Adam-type optimizers
`--weight_decay` | 1e-6 | >= 0. | L2 weight regularization term
`--lr_warmup_steps` | 0 | >= 0 | Number of steps for learning rate warmup
`--lr_warmup_factor` | 1. | >= 1. | When warmup is used, the target learning_rate will be lr_warmup_factor * start_lr
`--lr_schedule` | - | `type="multistep",milestones="<milestone_list>",decay_rate="<value>"` or `type="cosine_annealing",t_max="<value>",eta_min="<value>"` | Specifies the learning rate schedule. Multistep decays the current learning rate by `decay_rate` at every milestone in the list. Note that the milestones are in unit of steps, not epochs. Number and value of milestones and the `decay_rate` can be chosen arbitrarily. For a milestone list, please specify it as whitespace separated values, for example `milestones="5000 10000"`. For cosine annealing, the minimal lr is given by the value of `eta_min` and the period length in number of steps by `T_max`
`--batchnorm_group_size` | 1 | >= 1 | Determines how many ranks participate in the batchnorm. Specifying a value > 1 will replace nn.BatchNorm2d with nn.SyncBatchNorm everywhere in the model. Currently, nn.SyncBatchNorm only supports node-local batch normalization, but using an Implementation of that same functionality which span arbitrary number of workers is allowed
`--gradient_accumulation_frequency` | 1 | >= 1 | Specifies the number of gradient accumulation steps before a weight update is performed
`--seed` | 333 | > 0 | Random number generator seed. Multiple submissions which employ the same seed are **forbidden**. Please specify a seed depending on system clock or similar.

*LAMB optimizer has additional hyperparameters such as the global grad clipping norm value. For the purpose of this benchmark, consider all those parameters which are LAMB specific and fixed. The defaults are specified in the [NVIDIA APEX documentation for FusedLAMB](https://nvidia.github.io/apex/_modules/apex/optimizers/fused_lamb.html).

Note that the command line arguments do not directly correspond to logging entries. For compliance checking of oiutput logs, use the table below:

|Key| Constraints | Required |
--- | --- | ---
`seed` | `x > 0` | True
`global_batch_size` | `x > 0` | `True`
`num_workers` | `x > 0` | `True`
`batchnorm_group_size` | `x > 1` | `False`
`gradient_accumulation_frequency` | `x >= 1` | `True`
`opt_name` | `x in ["Adam", "AdamW", "LAMB"]` | `True`
`opt_lr` | `x >= 0.` | `True`
`opt_betas` | unconstrained | `True`
`opt_eps` | `x == 1e-6` | `True`
`opt_weight_decay` | `x >= 0.` | `True`
`opt_bias_correction` | `x == True` | `True if opt_name == "LAMB" else False`
`opt_grad_averaging` | `x == True` | `True if opt_name == "LAMB" else False`
`opt_max_grad_norm` | `x == 1.0` | `True if opt_name == "LAMB" else False`
`scheduler_type` | `x in ["multistep", "cosine_annealing"]` | `True`
`scheduler_milestones` | unconstrained | `True if scheduler_type == "multistep" else False`
`scheduler_decay_rate` | `x >= 1.` | `True if scheduler_type == "multistep" else False`
`scheduler_t_max` | `x >= 0` | `True if scheduler_type == "cosine_annealing" else False`
`scheduler_eta_min` | `x >= 0.` | `True if scheduler_type == "cosine_annealing" else False`
`scheduler_lr_warmup_steps` | `x >= 0` | `False`
`scheduler_lr_warmup_factor` | `x >= 1.` | `True if scheduler_lr_warmup_steps > 0 else False`

The first column lists the keys as they would appear in the logfile. The second column lists the parameters constraints as an equation for parameter variable x. Those can be used to generate lambda expressions in Python. The third one if the corresponding entry has to be in the log file or not. Since there are multiple optimizers and learning rate schedules to choose from, not all parameters need to be logged for a given run. This is expressed by conditional expressions in that column.
**Please note that besides the benchmark specific rules above, standard MLPerf HPC logging rules apply.**

### Using Docker

The implementation comes with a Dockerfile optimized for NVIDIA workstations but usable on 
other NVIDIA multi-gpu systems. Use the Dockerfile 
`docker/Dockerfile.train` to build the container and the script `src/deepCam/run_scripts/run_training.sh`
for training. The data_dir variable should point to the full path of the `All-Hist` directory containing the downloaded dataset.

## References

1. Wehner, M. F., Reed, K. A., Li, F., Bacmeister, J., Chen, C.-T., Paciorek, C., Gleckler, P. J., Sperber, K. R., Collins, W. D., Gettelman, A., et al.: The effect of horizontal resolution on simulation quality in the Community Atmospheric Model, CAM5. 1, Journal of Advances in Modeling Earth Systems, 6, 980-997, 2014.
2. Prabhat, Byna, S., Vishwanath, V., Dart, E., Wehner, M., Collins, W. D., et al.: TECA: Petascale pattern recognition for climate science, in: International Conference on Computer Analysis of Images and Patterns, pp. 426-436, Springer, 2015b.
