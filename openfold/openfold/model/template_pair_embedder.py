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


class TemplatePairEmbedder(nn.Module):
    """Template Pair Embedder module.

    Embeds the "template_pair_feat" feature.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2, line 9.

    Args:
        tp_dim: Input `template_pair_feat` dimension (channels).
        c_t: Output template representation dimension (channels).

    """

    def __init__(
        self,
        tp_dim: int,
        c_t: int,
    ) -> None:
        super(TemplatePairEmbedder, self).__init__()
        self.tp_dim = tp_dim
        self.c_t = c_t
        self.linear = Linear(tp_dim, c_t, bias=True, init="relu")

    def forward(
        self,
        template_pair_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Template Pair Embedder forward pass.

        Args:
            template_pair_feat: [batch, N_res, N_res, tp_dim]

        Returns:
            template_pair_embedding: [batch, N_res, N_res, c_t]

        """
        return self.linear(template_pair_feat)
