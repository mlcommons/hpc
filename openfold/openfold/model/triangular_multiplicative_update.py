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
from openfold.model.layer_norm import LayerNorm
from openfold.torch_utils import is_autocast_fp16_enabled


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle Multiplicative Update module.

    Supplementary '1.6.5 Triangular multiplicative update': Algorithms 11 and 12.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        tmu_type: "outgoing" or "incoming"

    """
    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        tmu_type: str,
    ) -> None:
        super(TriangleMultiplicativeUpdate, self).__init__()
        # configuration:
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._is_outgoing = {"outgoing": True, "incoming": False}[tmu_type]
        # submodules:
        self.linear_a_p = Linear(c_z, c_hidden, bias=True, init="default")
        self.linear_a_g = Linear(c_z, c_hidden, bias=True, init="gating")
        self.linear_b_p = Linear(c_z, c_hidden, bias=True, init="default")
        self.linear_b_g = Linear(c_z, c_hidden, bias=True, init="gating")
        self.linear_g = Linear(c_z, c_z, bias=True, init="gating")
        self.linear_z = Linear(c_hidden, c_z, bias=True, init="final")
        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)

    def forward(
        self,
        z: torch.Tensor,     # [batch, N_res, N_res, c_z] pair representation
        mask: torch.Tensor,  # [batch, N_res, N_res] pair mask
    ) -> torch.Tensor:       # [batch, N_res, N_res, c_z] pair representation update
        z = self.layer_norm_in(z)
        # z: [batch, N_res, N_res, c_z]

        mask = mask.unsqueeze(-1)
        # mask: [batch, N_res, N_res, 1]

        a = torch.sigmoid(self.linear_a_g(z)) * mask
        a = a * self.linear_a_p(z)
        # a: [batch, N_res, N_res, c_hidden]

        b = torch.sigmoid(self.linear_b_g(z)) * mask
        b = b * self.linear_b_p(z)
        # b: [batch, N_res, N_res, c_hidden]

        if is_autocast_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)
        # x: [batch, N_res, N_res, c_hidden]

        del a, b

        x = self.layer_norm_out(x)
        # x: [batch, N_res, N_res, c_hidden]

        x = self.linear_z(x)
        # x: [batch, N_res, N_res, c_z]

        g = torch.sigmoid(self.linear_g(z))
        # g: [batch, N_res, N_res, c_z]

        x = x * g
        # x: [batch, N_res, N_res, c_z]

        return x

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_outgoing:
            a = a.movedim(-1, -3)
            b = b.swapdims(-1, -3)
        else:
            a = a.swapdims(-1, -3)
            b = b.movedim(-1, -3)

        p = torch.matmul(a, b)

        return p.movedim(-3, -1)


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """Triangle Multiplication Outgoing module.

    Supplementary '1.6.5 Triangular multiplicative update':
    Algorithm 11 Triangular multiplicative update using "outgoing" edges.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).

    """
    def __init__(
        self,
        c_z: int,
        c_hidden: int,
    ) -> None:
        super(TriangleMultiplicationOutgoing, self).__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            tmu_type="outgoing",
        )


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """Triangle Multiplication Incoming module.

    Supplementary '1.6.5 Triangular multiplicative update':
    Algorithm 12 Triangular multiplicative update using "incoming" edges.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).

    """
    def __init__(
        self,
        c_z: int,
        c_hidden: int,
    ) -> None:
        super(TriangleMultiplicationIncoming, self).__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            tmu_type="incoming",
        )
