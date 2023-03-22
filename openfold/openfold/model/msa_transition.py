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


class MSATransition(nn.Module):
    """MSA Transition module.

    Supplementary '1.6.3 MSA transition': Algorithm 9.

    Args:
        c_m: MSA (or Extra MSA) representation dimension (channels).
        n: `c_m` multiplier to obtain hidden dimension (channels).

    """

    def __init__(
        self,
        c_m: int,
        n: int,
    ) -> None:
        super(MSATransition, self).__init__()
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, n * c_m, bias=True, init="relu")
        self.linear_2 = Linear(n * c_m, c_m, bias=True, init="final")

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSA Transition forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask

        Returns:
            m_update: [batch, N_seq, N_res, c_m] MSA representation update

        """
        # DeepMind forgets to apply the MSA mask here.
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = torch.relu(m)
        m = self.linear_2(m)
        return m
