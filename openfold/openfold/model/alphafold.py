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

from dataclasses import asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import openfold.data.residue_constants as rc
from openfold.config import AlphaFoldConfig
from openfold.model.auxiliary_heads import AuxiliaryHeads
from openfold.model.evoformer_stack import EvoformerStack
from openfold.model.extra_msa_embedder import ExtraMSAEmbedder
from openfold.model.extra_msa_stack import ExtraMSAStack
from openfold.model.input_embedder import InputEmbedder
from openfold.model.recycling_embedder import RecyclingEmbedder
from openfold.model.structure_module import StructureModule
from openfold.model.template_angle_embedder import TemplateAngleEmbedder
from openfold.model.template_pair_embedder import TemplatePairEmbedder
from openfold.model.template_pair_stack import TemplatePairStack
from openfold.model.template_pointwise_attention import TemplatePointwiseAttention
from openfold.rigid_utils import Rigid
from openfold.torch_utils import map_tensor_tree


class AlphaFold(nn.Module):
    """AlphaFold2 module.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2.

    """

    def __init__(self, config: AlphaFoldConfig) -> None:
        super(AlphaFold, self).__init__()
        self.input_embedder = InputEmbedder(
            **asdict(config.input_embedder_config),
        )
        self.recycling_embedder = RecyclingEmbedder(
            **asdict(config.recycling_embedder_config),
        )
        if config.templates_enabled:
            self.template_angle_embedder = TemplateAngleEmbedder(
                **asdict(config.template_angle_embedder_config),
            )
            self.template_pair_embedder = TemplatePairEmbedder(
                **asdict(config.template_pair_embedder_config),
            )
            self.template_pair_stack = TemplatePairStack(
                **asdict(config.template_pair_stack_config),
            )
            self.template_pointwise_attention = TemplatePointwiseAttention(
                **asdict(config.template_pointwise_attention_config),
            )
        self.extra_msa_embedder = ExtraMSAEmbedder(
            **asdict(config.extra_msa_embedder_config),
        )
        self.extra_msa_stack = ExtraMSAStack(
            **asdict(config.extra_msa_stack_config),
        )
        self.evoformer_stack = EvoformerStack(
            **asdict(config.evoformer_stack_config),
        )
        self.structure_module = StructureModule(
            **asdict(config.structure_module_config),
        )
        self.auxiliary_heads = AuxiliaryHeads(config.auxiliary_heads_config)
        self.config = config

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Initialize previous recycling embeddings:
        prevs = {}

        # Forward iterations with autograd disabled:
        num_recycling_iters = batch["aatype"].shape[-1] - 1
        for j in range(num_recycling_iters):
            feats = map_tensor_tree(fn=lambda t: t[..., j].contiguous(), tree=batch)
            with torch.no_grad():
                outputs, prevs = self._forward_iteration(
                    feats=feats,
                    prevs=prevs,
                    gradient_checkpointing=False,
                )
                del outputs

        # https://github.com/pytorch/pytorch/issues/65766
        if torch.is_autocast_enabled():
            torch.clear_autocast_cache()

        # Final iteration with autograd enabled:
        feats = map_tensor_tree(fn=lambda t: t[..., -1].contiguous(), tree=batch)
        outputs, prevs = self._forward_iteration(
            feats=feats,
            prevs=prevs,
            gradient_checkpointing=self.training,
        )
        del prevs

        # Run auxiliary heads:
        aux_outputs = self.auxiliary_heads(outputs)
        outputs.update(aux_outputs)

        return outputs

    def _forward_iteration(
        self,
        feats: Dict[str, torch.Tensor],
        prevs: Dict[str, torch.Tensor],
        gradient_checkpointing: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        outputs = {}

        batch_size = feats["aatype"].shape[0]
        N_res = feats["aatype"].shape[1]
        N_clust = feats["msa_feat"].shape[1]

        seq_mask = feats["seq_mask"]
        # seq_mask: [batch, N_res]

        pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)  # outer product
        # pair_mask: [batch, N_res, N_res]

        msa_mask = feats["msa_mask"]
        # msa_mask: [batch, N_clust, N_res]

        # Initialize MSA and pair representations:
        m, z = self.input_embedder(
            target_feat=feats["target_feat"],
            residue_index=feats["residue_index"],
            msa_feat=feats["msa_feat"],
        )
        # m: [batch, N_clust, N_res, c_m]
        # z: [batch, N_res, N_res, c_z]

        # Extract recycled representations:
        m0_prev = prevs.pop("m0_prev", None)
        z_prev = prevs.pop("z_prev", None)
        x_prev = prevs.pop("x_prev", None)

        # Initialize recycling representations if needed:
        if m0_prev is None:
            m0_prev = torch.zeros_like(m[:, 0], requires_grad=False)
        if z_prev is None:
            z_prev = torch.zeros_like(z, requires_grad=False)
        if x_prev is None:
            x_prev = z.new_zeros(
                size=(batch_size, N_res, rc.ATOM_TYPE_NUM, 3),
                requires_grad=False,
            )

        x_prev = _pseudo_beta_fn(
            aatype=feats["aatype"],
            all_atom_positions=x_prev,
        ).to(dtype=z.dtype)

        m0_prev_emb, z_prev_emb = self.recycling_embedder(
            m0_prev=m0_prev,
            z_prev=z_prev,
            x_prev=x_prev,
        )
        # m0_prev_emb: [batch, N_res, c_m]
        # z_prev_emb: [batch, N_res, N_res, c_z]

        m[:, 0] += m0_prev_emb
        z = z + z_prev_emb

        del m0_prev, z_prev, x_prev, m0_prev_emb, z_prev_emb

        # Embed templates and merge with MSA/pair representation:
        if self.config.templates_enabled:
            template_feats = {
                k: t for k, t in feats.items() if k.startswith("template_")
            }
            template_embeds = self._embed_templates(
                feats=template_feats,
                z=z,
                pair_mask=pair_mask,
                gradient_checkpointing=gradient_checkpointing,
            )

            z = z + template_embeds["template_pair_embedding"]
            # z: [batch, N_res, N_res, c_z]

            if self.config.embed_template_torsion_angles:
                m = torch.cat([m, template_embeds["template_angle_embedding"]], dim=1)
                # m: [batch, N_seq, N_res, c_m]

                msa_mask = torch.cat(
                    [
                        feats["msa_mask"],
                        feats["template_torsion_angles_mask"][..., 2],
                    ],
                    dim=-2,
                )
                # msa_mask: [batch, N_seq, N_res]

            del template_feats, template_embeds

        # N_seq = m.shape[1]

        # Embed extra MSA features and merge with pairwise embeddings:
        # N_extra_seq = feats["extra_msa"].shape[1]
        a = self.extra_msa_embedder(_build_extra_msa_feat(feats))
        # a: [batch, N_extra_seq, N_res, c_e]
        z = self.extra_msa_stack(
            m=a,
            z=z,
            msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=m.dtype),
            gradient_checkpointing=gradient_checkpointing,
        )
        # z: [batch, N_res, N_res, c_z]
        del a

        # Evoformer forward pass:
        m, z, s = self.evoformer_stack(
            m=m,
            z=z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            gradient_checkpointing=gradient_checkpointing,
        )
        # m: [batch, N_seq, N_res, c_m]
        # z: [batch, N_res, N_res, c_z]
        # s: [batch, N_res, c_s]
        outputs["msa"] = m[:, :N_clust]
        outputs["pair"] = z
        outputs["single"] = s

        # Predict 3D structure:
        sm_outputs = self.structure_module(
            s=outputs["single"],
            z=outputs["pair"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            aatype=feats["aatype"],
        )
        outputs.update(sm_outputs)
        outputs["final_atom_positions"] = _atom14_to_atom37(
            atom14_positions=outputs["sm_positions"][:, -1],
            residx_atom37_to_atom14=feats["residx_atom37_to_atom14"],
            atom37_atom_exists=feats["atom37_atom_exists"],
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm_frames"][:, -1]

        # Save embeddings for next recycling iteration:
        prevs = {}
        prevs["m0_prev"] = m[:, 0]
        prevs["z_prev"] = outputs["pair"]
        prevs["x_prev"] = outputs["final_atom_positions"]

        return outputs, prevs

    def _embed_templates(
        self,
        feats: Dict[str, torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        gradient_checkpointing: bool,
    ) -> Dict[str, torch.Tensor]:
        # Embed the templates one at a time:
        pair_embeds = []
        N_templ = feats["template_aatype"].shape[1]
        for i in range(N_templ):
            single_template_feats = map_tensor_tree(fn=lambda t: t[:, i], tree=feats)
            t = _build_template_pair_feat(
                feats=single_template_feats,
                min_bin=self.config.template_pair_feat_distogram_min_bin,
                max_bin=self.config.template_pair_feat_distogram_max_bin,
                num_bins=self.config.template_pair_feat_distogram_num_bins,
                use_unit_vector=self.config.template_pair_feat_use_unit_vector,
                inf=self.config.template_pair_feat_inf,
                eps=self.config.template_pair_feat_eps,
            ).to(dtype=z.dtype)
            t = self.template_pair_embedder(t)
            # t: [batch, N_res, N_res, c_t]
            pair_embeds.append(t)
            del t

        t = torch.stack(pair_embeds, dim=1)
        # t: [batch, N_templ, N_res, N_res, c_t]
        del pair_embeds

        t = self.template_pair_stack(
            t=t,
            mask=pair_mask.to(dtype=z.dtype),
            gradient_checkpointing=gradient_checkpointing,
        )
        # t: [batch, N_templ, N_res, N_res, c_t]

        t = self.template_pointwise_attention(
            t=t,
            z=z,
            template_mask=feats["template_mask"].to(dtype=z.dtype),
        )
        # t: [batch, N_res, N_res, c_z]

        t_mask = (torch.sum(feats["template_mask"], dim=1) > 0).to(dtype=t.dtype)
        t_mask = t_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t = t * t_mask
        # t: [batch, N_res, N_res, c_z]

        template_embeds = {}
        template_embeds["template_pair_embedding"] = t

        if self.config.embed_template_torsion_angles:
            template_angle_feat = _build_template_angle_feat(feats)
            a = self.template_angle_embedder(template_angle_feat)
            # a: [batch, N_templ, N_res, c_m]
            template_embeds["template_angle_embedding"] = a

        return template_embeds


def _pseudo_beta_fn(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
) -> torch.Tensor:
    is_gly = torch.eq(aatype, rc.RESTYPE_ORDER["G"])
    ca_idx = rc.ATOM_ORDER["CA"]
    cb_idx = rc.ATOM_ORDER["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly.unsqueeze(-1), [1] * is_gly.ndim + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )
    return pseudo_beta


def _build_extra_msa_feat(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    msa_1hot = F.one_hot(input=feats["extra_msa"], num_classes=23)
    msa_feat = [
        msa_1hot,
        feats["extra_has_deletion"].unsqueeze(-1),
        feats["extra_deletion_value"].unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)


def _build_template_pair_feat(
    feats: Dict[str, torch.Tensor],
    min_bin: int,
    max_bin: int,
    num_bins: int,
    use_unit_vector: bool,
    inf: float,
    eps: float,
) -> torch.Tensor:
    template_mask = feats["template_pseudo_beta_mask"]
    template_mask_2d = template_mask.unsqueeze(-1) * template_mask.unsqueeze(-2)

    # Compute distogram (this seems to differ slightly from Alg. 5)
    tpb = feats["template_pseudo_beta"]
    dgram = torch.sum(
        input=(tpb.unsqueeze(-2) - tpb.unsqueeze(-3)) ** 2,
        dim=-1,
        keepdim=True,
    )
    lower = (
        torch.linspace(
            start=min_bin,
            end=max_bin,
            steps=num_bins,
            device=tpb.device,
        )
        ** 2
    )
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).to(dtype=dgram.dtype)

    to_concat = [dgram, template_mask_2d.unsqueeze(-1)]

    aatype_one_hot = F.one_hot(
        input=feats["template_aatype"],
        num_classes=(rc.RESTYPE_NUM + 2),
    )

    N_res = feats["template_aatype"].shape[-1]

    to_concat.append(
        aatype_one_hot.unsqueeze(-3).expand(*aatype_one_hot.shape[:-2], N_res, -1, -1)
    )
    to_concat.append(
        aatype_one_hot.unsqueeze(-2).expand(*aatype_one_hot.shape[:-2], -1, N_res, -1)
    )

    n, ca, c = [rc.ATOM_ORDER[a] for a in ["N", "CA", "C"]]
    rigids = Rigid.make_transform_from_reference(
        n_xyz=feats["template_all_atom_positions"][..., n, :],
        ca_xyz=feats["template_all_atom_positions"][..., ca, :],
        c_xyz=feats["template_all_atom_positions"][..., c, :],
        eps=eps,
    )
    points = rigids.get_trans().unsqueeze(-3)
    rigid_vec = rigids.unsqueeze(-1).invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))

    t_aa_masks = feats["template_all_atom_mask"]
    template_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    template_mask_2d = template_mask.unsqueeze(-1) * template_mask.unsqueeze(-2)

    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar.unsqueeze(-1)

    if not use_unit_vector:
        unit_vector = unit_vector * 0.0

    to_concat.extend(torch.unbind(unit_vector.unsqueeze(-2), dim=-1))
    to_concat.append(template_mask_2d.unsqueeze(-1))

    t = torch.cat(to_concat, dim=-1)
    t = t * template_mask_2d.unsqueeze(-1)

    return t


def _build_template_angle_feat(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    template_aatype = feats["template_aatype"]
    torsion_angles_sin_cos = feats["template_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = feats["template_alt_torsion_angles_sin_cos"]
    torsion_angles_mask = feats["template_torsion_angles_mask"]
    template_angle_feat = torch.cat(
        [
            F.one_hot(input=template_aatype, num_classes=22),
            torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
            alt_torsion_angles_sin_cos.reshape(
                *alt_torsion_angles_sin_cos.shape[:-2], 14
            ),
            torsion_angles_mask,
        ],
        dim=-1,
    )
    return template_angle_feat


def _atom14_to_atom37(
    atom14_positions: torch.Tensor,
    residx_atom37_to_atom14: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
) -> torch.Tensor:
    # atom14_positions: [batch, N_res, 14, 3]
    # residx_atom37_to_atom14: [batch, N_res, 37]
    # atom37_atom_exists: [batch, N_res, 37]

    indices = residx_atom37_to_atom14.unsqueeze(-1)
    # indices: [batch, N_res, 37, 1]
    indices = indices.expand(-1, -1, -1, 3)
    # indices: [batch, N_res, 37, 3]

    atom37_positions = torch.gather(atom14_positions, 2, indices)
    # atom37_positions: [batch, N_res, 37, 3]

    atom37_mask = atom37_atom_exists.unsqueeze(-1)
    # atom37_mask: [batch, N_res, 37, 1]

    atom37_positions = atom37_positions * atom37_mask
    # atom37_positions: [batch, N_res, 37, 3]

    return atom37_positions
