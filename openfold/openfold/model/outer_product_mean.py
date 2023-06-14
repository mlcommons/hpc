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

from typing import Optional

import torch
import torch.nn as nn

from openfold.helpers import slice_generator
from openfold.model.layer_norm import LayerNorm
from openfold.model.linear import Linear
from openfold.torch_utils import is_autocast_fp16_enabled


class OuterProductMean(nn.Module):
    """Outer Product Mean module.

    Supplementary '1.6.4 Outer product mean': Algorithm 10.

    Args:
        c_m: MSA (or Extra MSA) representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        eps: Epsilon to prevent division by zero.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        eps: float,
        chunk_size: Optional[int],
    ) -> None:
        super(OuterProductMean, self).__init__()
        assert eps == 1e-3
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps
        self.chunk_size = chunk_size
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=True, init="default")
        self.linear_2 = Linear(c_m, c_hidden, bias=True, init="default")
        self.linear_out = Linear(c_hidden**2, c_z, bias=True, init="final")

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Outer Product Mean forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask

        Returns:
            outer: [batch, N_res, N_res, c_z] pair representation update

        """
        if is_autocast_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(m=m.float(), mask=mask)
        else:
            return self._forward(m=m, mask=mask)

    def _forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        m = self.layer_norm(m)
        # m: [batch, N_seq, N_res, c_m]

        mask = mask.unsqueeze(-1)
        # mask: [batch, N_seq, N_res, 1]

        a = self.linear_1(m) * mask
        # a: [batch, N_seq, N_res, c_hidden]

        b = self.linear_2(m) * mask
        # b: [batch, N_seq, N_res, c_hidden]

        a = a.transpose(-2, -3)
        # a: [batch, N_res, N_seq, c_hidden]

        b = b.transpose(-2, -3)
        # b: [batch, N_res, N_seq, c_hidden]

        outer = self._outer_forward(a=a, b=b)
        # outer: [batch, N_res, N_res, c_z]

        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        # norm: [batch, N_res, N_res, 1]

        outer = outer / (norm + self.eps)
        # outer: [batch, N_res, N_res, c_z]

        return outer

    def _outer_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is None:
            return self._outer(a=a, b=b)
        else:
            return self._outer_chunked(a=a, b=b, chunk_size=self.chunk_size)

    def _outer(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        # outer: [batch, a_N_res, b_N_res, c_hidden, c_hidden]

        outer = outer.reshape(outer.shape[:-2] + (self.c_hidden * self.c_hidden,))
        # outer: [batch, a_N_res, b_N_res, c_hidden * c_hidden]

        outer = self.linear_out(outer)
        # outer: [batch, a_N_res, b_N_res, c_z]
        return outer

    def _outer_chunked(
        self, a: torch.Tensor, b: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        outer_chunks = []
        subbatch_size = a.size(1)
        for left, right in slice_generator(0, subbatch_size, chunk_size):
            a_chunk = a[:, left:right]
            outer_chunk = self._outer(a=a_chunk, b=b)
            outer_chunks.append(outer_chunk)
        return torch.cat(outer_chunks, dim=1)
