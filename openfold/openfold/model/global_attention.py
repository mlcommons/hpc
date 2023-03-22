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

import math
from typing import Optional

import torch
import torch.nn as nn

from openfold.helpers import slice_generator
from openfold.model.linear import Linear


class GlobalAttention(nn.Module):
    """Global Attention module.

    Args:
        c_e: Extra MSA representation dimension (channels).
        c_hidden: Per-head hidden dimension (channels).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        eps: Epsilon to prevent division by zero.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_e: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        eps: float,
        chunk_size: Optional[int],
    ) -> None:
        super(GlobalAttention, self).__init__()
        self.c_e = c_e
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf
        self.eps = eps
        self.chunk_size = chunk_size
        self.linear_q = Linear(c_e, c_hidden * num_heads, bias=False, init="glorot")
        self.linear_k = Linear(c_e, c_hidden, bias=False, init="glorot")
        self.linear_v = Linear(c_e, c_hidden, bias=False, init="glorot")
        self.linear_g = Linear(c_e, c_hidden * num_heads, bias=True, init="gating")
        self.linear_o = Linear(c_hidden * num_heads, c_e, bias=True, init="final")

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Global Attention forward pass.

        Args:
            m: [batch, N_res, N_extra_seq, c_e] transposed extra MSA representation
            mask: [batch, N_res, N_extra_seq] transposed extra MSA mask

        Returns:
            m_update: [batch, N_res, N_extra_seq, c_e]
                transposed extra MSA representation update

        """
        if self.chunk_size is None:
            return self._forward(m=m, mask=mask)
        else:
            return self._forward_chunked(m=m, mask=mask, chunk_size=self.chunk_size)

    def _forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        q_num = torch.sum(m * mask.unsqueeze(-1), dim=-2)
        q_den = torch.sum(mask, dim=-1).add(self.eps).unsqueeze(-1)
        q = q_num / q_den
        # q: [batch, N_res, c_e]
        del q_num, q_den

        q = self.linear_q(q)
        # q: [batch, N_res, num_heads * c_hidden]

        q = q * math.sqrt(1 / self.c_hidden)
        # q: [batch, N_res, num_heads * c_hidden]

        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        # q: [batch, N_res, num_heads, c_hidden]

        k = self.linear_k(m)
        # k: [batch, N_res, N_extra_seq, c_hidden]

        v = self.linear_v(m)
        # v: [batch, N_res, N_extra_seq, c_hidden]

        bias = ((mask - 1.0) * self.inf).unsqueeze(-2)
        # v: [batch, N_res, 1, N_extra_seq]

        a = torch.matmul(q, k.transpose(-1, -2))
        # a: [batch, N_res, num_heads, N_extra_seq]

        a += bias
        # a: [batch, N_res, num_heads, N_extra_seq]

        a = torch.softmax(a, dim=-1)
        # a: [batch, N_res, num_heads, N_extra_seq]

        o = torch.matmul(a, v)
        # o: [batch, N_res, num_heads, c_hidden]

        g = torch.sigmoid(self.linear_g(m))
        # g: [batch, N_res, N_extra_seq, num_heads * c_hidden]

        g = g.view(g.shape[:-1] + (self.num_heads, self.c_hidden))
        # g: [batch, N_res, N_extra_seq, num_heads, c_hidden]

        o = o.unsqueeze(-3) * g
        # o: [batch, N_res, N_extra_seq, num_heads, c_hidden]

        o = o.reshape(o.shape[:-2] + (self.num_heads * self.c_hidden,))
        # o: [batch, N_res, N_extra_seq, num_heads * c_hidden]

        m = self.linear_o(o)
        # m: [batch, N_res, N_extra_seq, c_e]

        return m

    def _forward_chunked(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        output_chunks = []
        subbatch_size = m.size(1)
        for left, right in slice_generator(0, subbatch_size, chunk_size):
            m_chunk = m[:, left:right]
            mask_chunk = mask[:, left:right]
            output_chunk = self._forward(
                m=m_chunk,
                mask=mask_chunk,
            )
            output_chunks.append(output_chunk)
        return torch.cat(output_chunks, dim=1)
