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

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.model.linear import Linear


class InputEmbedder(nn.Module):
    """Input Embedder module.

    Supplementary '1.5 Input embeddings'.

    Args:
        tf_dim: Input `target_feat` dimension (channels).
        msa_dim: Input `msa_feat` dimension (channels).
        c_z: Output pair representation dimension (channels).
        c_m: Output MSA representation dimension (channels).
        relpos_k: Relative position clip distance.

    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
    ) -> None:
        super(InputEmbedder, self).__init__()
        self.tf_dim = tf_dim
        self.msa_dim = msa_dim
        self.c_z = c_z
        self.c_m = c_m
        self.relpos_k = relpos_k
        self.num_bins = 2 * relpos_k + 1
        self.linear_tf_z_i = Linear(tf_dim, c_z, bias=True, init="default")
        self.linear_tf_z_j = Linear(tf_dim, c_z, bias=True, init="default")
        self.linear_tf_m = Linear(tf_dim, c_m, bias=True, init="default")
        self.linear_msa_m = Linear(msa_dim, c_m, bias=True, init="default")
        self.linear_relpos = Linear(self.num_bins, c_z, bias=True, init="default")

    def forward(
        self,
        target_feat: torch.Tensor,
        residue_index: torch.Tensor,
        msa_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input Embedder forward pass.

        Supplementary '1.5 Input embeddings': Algorithm 3.

        Args:
            target_feat: [batch, N_res, tf_dim]
            residue_index: [batch, N_res]
            msa_feat: [batch, N_clust, N_res, msa_dim]

        Returns:
            msa_emb: [batch, N_clust, N_res, c_m]
            pair_emb: [batch, N_res, N_res, c_z]

        """
        tf_emb_i = self.linear_tf_z_i(target_feat)  # a_i
        # tf_emb_i: [batch, N_res, c_z]

        tf_emb_j = self.linear_tf_z_j(target_feat)  # b_j
        # tf_emb_j: [batch, N_res, c_z]

        pair_emb = self.relpos(residue_index.to(dtype=tf_emb_j.dtype))
        pair_emb = pair_emb + tf_emb_i.unsqueeze(-2)
        pair_emb = pair_emb + tf_emb_j.unsqueeze(-3)
        # pair_emb: [batch, N_res, N_res, c_z]

        msa_emb = self.linear_msa_m(msa_feat)
        msa_emb = msa_emb + self.linear_tf_m(target_feat).unsqueeze(-3)
        # msa_emb: [batch, N_clust, N_res, c_m]

        return msa_emb, pair_emb

    def relpos(self, residue_index: torch.Tensor) -> torch.Tensor:
        """Relative position encoding.

        Supplementary '1.5 Input embeddings': Algorithm 4.

        """
        bins = torch.arange(
            start=-self.relpos_k,
            end=self.relpos_k + 1,
            step=1,
            dtype=residue_index.dtype,
            device=residue_index.device,
        )
        relative_distances = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        return self.linear_relpos(_one_hot_relpos(relative_distances, bins))


def _one_hot_relpos(
    relative_distances: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """One-hot encoding with nearest bin.

    Supplementary '1.5 Input embeddings': Algorithm 5.

    """
    indices = (relative_distances.unsqueeze(-1) - bins).abs().argmin(dim=-1)
    return F.one_hot(indices, num_classes=len(bins)).to(dtype=relative_distances.dtype)
