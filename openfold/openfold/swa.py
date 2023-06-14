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
import torch.nn as nn

from openfold.model.alphafold import AlphaFold


class AlphaFoldSWA(nn.Module):
    """AlphaFold SWA (Stochastic Weight Averaging) module wrapper."""

    def __init__(self, alphafold: AlphaFold, enabled: bool, decay_rate: float) -> None:
        super(AlphaFoldSWA, self).__init__()
        if enabled:
            self.averaged_model = torch.optim.swa_utils.AveragedModel(
                model=alphafold,
                avg_fn=swa_avg_fn(decay_rate=decay_rate),
            )
            self.enabled = True
        else:
            self.averaged_model = None
            self.enabled = False

    def update(self, alphafold: AlphaFold) -> None:
        if self.enabled:
            self.averaged_model.update_parameters(model=alphafold)

    def forward(self, batch):
        if not self.enabled:
            raise RuntimeError("AlphaFoldSWA is not enabled")
        return self.averaged_model(batch)


class swa_avg_fn:
    """Averaging function for EMA with configurable decay rate
    (Supplementary '1.11.7 Evaluator setup')."""

    def __init__(self, decay_rate: float) -> None:
        self._decay_rate = decay_rate

    def __call__(
        self,
        averaged_model_parameter: torch.Tensor,
        model_parameter: torch.Tensor,
        num_averaged: torch.Tensor,
    ) -> torch.Tensor:
        # for decay_rate = 0.999:
        # return averaged_model_parameter * 0.999 + model_parameter * 0.001
        # avg * 0.999 + m * 0.001
        # 999*avg/1000 + m/1000
        # (999*avg + avg - avg)/1000 + m/1000
        # (1000*avg - avg)/1000 + m/1000
        # 1000*avg/1000 - avg/1000 + m/1000
        # avg + (m - avg)/1000
        # avg + (m - avg)*0.001
        return averaged_model_parameter + (
            model_parameter - averaged_model_parameter
        ) * (1.0 - self._decay_rate)
