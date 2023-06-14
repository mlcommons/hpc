# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
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

from openfold.model.linear import Linear


class BackboneUpdate(nn.Module):
    """Backbone Update module.

    Supplementary '1.8.3 Backbone update': Algorithm 23.

    Args:
        c_s: Single representation dimension (channels).

    """

    def __init__(self, c_s: int) -> None:
        super(BackboneUpdate, self).__init__()
        self.linear = Linear(c_s, 6, bias=True, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)
