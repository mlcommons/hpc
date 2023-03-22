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

from openfold.model.layer_norm import LayerNorm
from openfold.model.linear import Linear


class PairTransition(nn.Module):
    """Pair Transition module.

    Supplementary '1.6.7 Transition in the pair stack': Algorithm 15.

    Args:
        c_z: Pair or template representation dimension (channels).
        n: `c_z` multiplier to obtain hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        n: int,
    ) -> None:
        super(PairTransition, self).__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear_1 = Linear(c_z, n * c_z, bias=True, init="relu")
        self.linear_2 = Linear(n * c_z, c_z, bias=True, init="final")

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pair Transition forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            z_update: [batch, N_res, N_res, c_z] pair representation update

        """
        # DeepMind forgets to apply the MSA mask here.
        z = self.layer_norm(z)
        z = self.linear_1(z)
        z = torch.relu(z)
        z = self.linear_2(z)
        return z
