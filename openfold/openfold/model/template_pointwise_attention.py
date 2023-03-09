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

from openfold.model.attention import Attention


class TemplatePointwiseAttention(nn.Module):
    """Template Pointwise Attention module.

    Supplementary '1.7.1 Template stack': Algorithm 17.

    Args:
        c_t: Template representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden: Hidden dimension (per-head).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.

    """
    def __init__(
        self,
        c_t: int,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super(TemplatePointwiseAttention, self).__init__()
        self.mha = Attention(
            c_q=c_z,
            c_k=c_t,
            c_v=c_t,
            c_hidden=c_hidden,
            num_heads=num_heads,
            gating=False,
            inf=inf,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        t: torch.Tensor,              # [batch, N_templ, N_res, N_res, c_t] template representation
        z: torch.Tensor,              # [batch, N_res, N_res, c_z] pair representation
        template_mask: torch.Tensor,  # [batch, N_templ] template mask
    ) -> torch.Tensor:                # [batch, N_res, N_res, c_z] pair representation update
        t = t.movedim(-4, -2)
        # t: [batch, N_res, N_res, N_templ, c_t]

        z = z.unsqueeze(-2)
        # z: [batch, N_res, N_res, 1, c_z]

        template_mask = template_mask.unsqueeze(-2).unsqueeze(-3).unsqueeze(-4).unsqueeze(-5)
        # template_mask: [batch, 1, 1, 1, 1, N_templ]

        z = self.mha(
            input_q=z,
            input_k=t,
            input_v=t,
            mask=template_mask,
            bias=None,
        )
        # z: [batch, N_res, N_res, 1, c_z]

        z = z.squeeze(-2)
        # z: [batch, N_res, N_res, c_z]

        return z
