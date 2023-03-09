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


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    lr_value: float,
    verbose: bool,
) -> None:
    if verbose:
        print(f"set_learning_rate: lr_value={lr_value}")
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
        iteration: int,
        verbose: bool = False,
    ) -> None:
        self.init_lr = init_lr
        self.final_lr = final_lr
        assert warmup_lr_length >= 0
        self.warmup_lr_length = warmup_lr_length
        self.init_lr_length = init_lr_length
        self.optimizer = optimizer
        assert iteration > 0
        self.iteration = iteration
        self.verbose = verbose
        self.warmup_linspace = torch.linspace(
            start=init_lr / max(warmup_lr_length, 1),
            end=init_lr,
            steps=warmup_lr_length,
            dtype=torch.float64,
        )
        self.lr_value = None
        self._prev_lr_value = None
        self._set_lr_value()
        set_learning_rate(
            optimizer=self.optimizer,
            lr_value=self.lr_value,
            verbose=self.verbose,
        )

    def step(self) -> None:
        self.iteration += 1
        self._prev_lr_value = self.lr_value
        self._set_lr_value()
        if self.lr_value != self._prev_lr_value:
            set_learning_rate(
                optimizer=self.optimizer,
                lr_value=self.lr_value,
                verbose=self.verbose,
            )

    def _set_lr_value(self) -> None:
        if self.iteration <= self.warmup_lr_length:
            self.lr_value = self.warmup_linspace[self.iteration - 1].item()
            self.lr_value = round(self.lr_value, 10)
        elif self.iteration <= self.init_lr_length:
            self.lr_value = self.init_lr
        else:
            self.lr_value = self.final_lr
