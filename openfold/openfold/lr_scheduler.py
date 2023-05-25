# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if len(optimizer.param_groups) == 1:
        return optimizer.param_groups[0]["lr"]
    elif len(optimizer.param_groups) > 1:
        lr_values_list = [pg["lr"] for pg in optimizer.param_groups]
        lr_values_set = set(lr_values_list)
        if len(lr_values_set) == 1:
            return lr_values_list[0]
        else:
            raise NotImplementedError("multiple different lr values")
    else:
        raise RuntimeError("no param groups")


def set_learning_rate(optimizer: torch.optim.Optimizer, lr_value: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_value


class AlphaFoldLRScheduler:
    """AlphaFold learning rate schedule."""

    def __init__(
        self,
        init_lr: float,
        final_lr: float,
        warmup_lr_length: int,
        init_lr_length: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.warmup_lr_length = warmup_lr_length
        self.init_lr_length = init_lr_length
        self.optimizer = optimizer
        # create LR values for the warm-up:
        assert warmup_lr_length >= 0
        self._warmup_linspace = torch.linspace(
            start=init_lr / max(warmup_lr_length, 1),
            end=init_lr,
            steps=warmup_lr_length,
            dtype=torch.float64,
        )
        self._prev_lr_value = None

    def __call__(self, iteration: int) -> None:
        # Determine lr_value for given iteration:
        if iteration <= self.warmup_lr_length:
            lr_value = self._warmup_linspace[iteration - 1].item()
            lr_value = round(lr_value, 10)
        elif iteration <= self.init_lr_length:
            lr_value = self.init_lr
        else:
            lr_value = self.final_lr
        # Set only if differs from the previous call:
        if lr_value != self._prev_lr_value:
            set_learning_rate(
                optimizer=self.optimizer,
                lr_value=lr_value,
            )
            self._prev_lr_value = lr_value


class OpenFoldBenchmarkLRScheduler:
    """Learning rate schedule for MLPerf HPC OpenFold partial training benchmark."""

    def __init__(
        self,
        base_lr: float,
        warmup_lr_init: float,
        warmup_lr_iters: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.base_lr = base_lr
        self.warmup_lr_init = warmup_lr_init
        self.warmup_lr_iters = warmup_lr_iters
        self.optimizer = optimizer
        # create LR values for the warm-up:
        assert warmup_lr_iters >= 0
        self._warmup_linspace = torch.linspace(
            start=warmup_lr_init,
            end=base_lr,
            steps=warmup_lr_iters,
            dtype=torch.float64,
        )
        self._prev_lr_value = None

    def __call__(self, iteration: int) -> None:
        # Determine lr_value for given iteration:
        if iteration <= self.warmup_lr_iters:
            lr_value = self._warmup_linspace[iteration - 1].item()
            lr_value = round(lr_value, 10)
        else:
            lr_value = self.base_lr
        # Set only if differs from the previous call:
        if lr_value != self._prev_lr_value:
            set_learning_rate(
                optimizer=self.optimizer,
                lr_value=lr_value,
            )
            self._prev_lr_value = lr_value
