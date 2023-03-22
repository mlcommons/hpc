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


class ExtraMSAEmbedder(nn.Module):
    """Extra MSA Embedder module.

    Embeds the "extra_msa_feat" feature.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2, line 15.

    Args:
        emsa_dim: Input `extra_msa_feat` dimension (channels).
        c_e: Output extra MSA representation dimension (channels).

    """

    def __init__(
        self,
        emsa_dim: int,
        c_e: int,
    ) -> None:
        super(ExtraMSAEmbedder, self).__init__()
        self.linear = Linear(emsa_dim, c_e, bias=True, init="default")

    def forward(
        self,
        extra_msa_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Extra MSA Embedder forward pass.

        Args:
            extra_msa_feat: [batch, N_extra_seq, N_res, emsa_dim]

        Returns:
            extra_msa_embedding: [batch, N_extra_seq, N_res, c_e]

        """
        return self.linear(extra_msa_feat)
