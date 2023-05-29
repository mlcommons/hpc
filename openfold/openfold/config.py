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

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import dacite


@dataclass
class InputEmbedderConfig:
    tf_dim: int = 22
    msa_dim: int = 49
    c_z: int = 128
    c_m: int = 256
    relpos_k: int = 32


@dataclass
class RecyclingEmbedderConfig:
    c_m: int = 256
    c_z: int = 128
    min_bin: float = 3.25
    max_bin: float = 20.75
    num_bins: int = 15
    inf: float = 1e8


@dataclass
class TemplateAngleEmbedderConfig:
    ta_dim: int = 57
    c_m: int = 256


@dataclass
class TemplatePairEmbedderConfig:
    tp_dim: int = 88
    c_t: int = 64


@dataclass
class TemplatePairStackConfig:
    c_t: int = 64
    c_hidden_tri_att: int = 16
    c_hidden_tri_mul: int = 64
    num_blocks: int = 2
    num_heads_tri: int = 4
    pair_transition_n: int = 2
    dropout_rate: float = 0.25
    inf: float = 1e9
    chunk_size_tri_att: Optional[int] = None


@dataclass
class TemplatePointwiseAttentionConfig:
    c_t: int = 64
    c_z: int = 128
    c_hidden: int = 16
    num_heads: int = 4
    inf: float = 1e5
    chunk_size: Optional[int] = None


@dataclass
class ExtraMSAEmbedderConfig:
    emsa_dim: int = 25
    c_e: int = 64


@dataclass
class ExtraMSAStackConfig:
    c_e: int = 64
    c_z: int = 128
    c_hidden_msa_att: int = 8
    c_hidden_opm: int = 32
    c_hidden_tri_mul: int = 128
    c_hidden_tri_att: int = 32
    num_heads_msa: int = 8
    num_heads_tri: int = 4
    num_blocks: int = 4
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    inf: float = 1e9
    eps: float = 1e-8
    eps_opm: float = 1e-3
    chunk_size_msa_att: Optional[int] = None
    chunk_size_opm: Optional[int] = None
    chunk_size_tri_att: Optional[int] = None


@dataclass
class EvoformerStackConfig:
    c_m: int = 256
    c_z: int = 128
    c_hidden_msa_att: int = 32
    c_hidden_opm: int = 32
    c_hidden_tri_mul: int = 128
    c_hidden_tri_att: int = 32
    c_s: int = 384
    num_heads_msa: int = 8
    num_heads_tri: int = 4
    num_blocks: int = 48
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    inf: float = 1e9
    eps_opm: float = 1e-3
    chunk_size_msa_att: Optional[int] = None
    chunk_size_opm: Optional[int] = None
    chunk_size_tri_att: Optional[int] = None


@dataclass
class StructureModuleConfig:
    c_s: int = 384
    c_z: int = 128
    c_hidden_ipa: int = 16
    c_hidden_ang_res: int = 128
    num_heads_ipa: int = 12
    num_qk_points: int = 4
    num_v_points: int = 8
    dropout_rate: float = 0.1
    num_blocks: int = 8
    num_ang_res_blocks: int = 2
    num_angles: int = 7
    scale_factor: float = 10.0
    inf: float = 1e5
    eps: float = 1e-8


@dataclass
class PerResidueLDDTCaPredictorConfig:
    c_s: int = 384
    c_hidden: int = 128
    num_bins: int = 50


@dataclass
class DistogramHeadConfig:
    c_z: int = 128
    num_bins: int = 64


@dataclass
class MaskedMSAHeadConfig:
    c_m: int = 256
    c_out: int = 23


@dataclass
class ExperimentallyResolvedHeadConfig:
    c_s: int = 384
    c_out: int = 37


@dataclass
class TMScoreHeadConfig:
    c_z: int = 128
    num_bins: int = 64
    max_bin: int = 31


@dataclass
class AuxiliaryHeadsConfig:
    per_residue_lddt_ca_predictor_config: PerResidueLDDTCaPredictorConfig = field(
        default=PerResidueLDDTCaPredictorConfig(),
    )
    distogram_head_config: DistogramHeadConfig = field(
        default=DistogramHeadConfig(),
    )
    masked_msa_head_config: MaskedMSAHeadConfig = field(
        default=MaskedMSAHeadConfig(),
    )
    experimentally_resolved_head_config: ExperimentallyResolvedHeadConfig = field(
        default=ExperimentallyResolvedHeadConfig(),
    )
    tm_score_head_config: TMScoreHeadConfig = field(
        default=TMScoreHeadConfig(),
    )
    tm_score_head_enabled: bool = False


@dataclass
class FAPELossConfig:
    weight: float = 1.0
    backbone_clamp_distance: float = 10.0
    backbone_loss_unit_distance: float = 10.0
    backbone_weight: float = 0.5
    sidechain_clamp_distance: float = 10.0
    sidechain_length_scale: float = 10.0
    sidechain_weight: float = 0.5
    eps: float = 1e-4


@dataclass
class SupervisedChiLossConfig:
    weight: float = 1.0
    chi_weight: float = 0.5
    angle_norm_weight: float = 0.01
    eps: float = 1e-8


@dataclass
class DistogramLossConfig:
    weight: float = 0.3
    min_bin: float = 2.3125
    max_bin: float = 21.6875
    num_bins: int = 64
    eps: float = 1e-8


@dataclass
class MaskedMSALossConfig:
    weight: float = 2.0
    eps: float = 1e-8


@dataclass
class PLDDTLossConfig:
    weight: float = 0.01
    cutoff: float = 15.0
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    num_bins: int = 50
    eps: float = 1e-8


@dataclass
class ExperimentallyResolvedLossConfig:
    weight: float = 0.0
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    eps: float = 1e-8


@dataclass
class ViolationLossConfig:
    weight: float = 0.0
    violation_tolerance_factor: float = 12.0
    clash_overlap_tolerance: float = 1.5
    eps: float = 1e-8


@dataclass
class TMLossConfig:
    enabled: bool = False
    weight: float = 0.0
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    num_bins: int = 64
    max_bin: int = 31
    eps: float = 1e-8


@dataclass
class LossConfig:
    fape_loss_config: FAPELossConfig = field(
        default=FAPELossConfig(),
    )
    supervised_chi_loss_config: SupervisedChiLossConfig = field(
        default=SupervisedChiLossConfig(),
    )
    distogram_loss_config: DistogramLossConfig = field(
        default=DistogramLossConfig(),
    )
    masked_msa_loss_config: MaskedMSALossConfig = field(
        default=MaskedMSALossConfig(),
    )
    plddt_loss_config: PLDDTLossConfig = field(
        default=PLDDTLossConfig(),
    )
    experimentally_resolved_loss_config: ExperimentallyResolvedLossConfig = field(
        default=ExperimentallyResolvedLossConfig(),
    )
    violation_loss_config: ViolationLossConfig = field(
        default=ViolationLossConfig(),
    )
    tm_loss_config: TMLossConfig = field(
        default=TMLossConfig(),
    )


@dataclass
class AlphaFoldConfig:
    preset: str = "default"

    # AlphaFold modules configuration:
    input_embedder_config: InputEmbedderConfig = field(
        default=InputEmbedderConfig(),
    )
    recycling_embedder_config: RecyclingEmbedderConfig = field(
        default=RecyclingEmbedderConfig(),
    )
    template_angle_embedder_config: TemplateAngleEmbedderConfig = field(
        default=TemplateAngleEmbedderConfig(),
    )
    template_pair_embedder_config: TemplatePairEmbedderConfig = field(
        default=TemplatePairEmbedderConfig(),
    )
    template_pair_stack_config: TemplatePairStackConfig = field(
        default=TemplatePairStackConfig(),
    )
    template_pointwise_attention_config: TemplatePointwiseAttentionConfig = field(
        default=TemplatePointwiseAttentionConfig(),
    )
    extra_msa_embedder_config: ExtraMSAEmbedderConfig = field(
        default=ExtraMSAEmbedderConfig(),
    )
    extra_msa_stack_config: ExtraMSAStackConfig = field(
        default=ExtraMSAStackConfig(),
    )
    evoformer_stack_config: EvoformerStackConfig = field(
        default=EvoformerStackConfig(),
    )
    structure_module_config: StructureModuleConfig = field(
        default=StructureModuleConfig(),
    )
    auxiliary_heads_config: AuxiliaryHeadsConfig = field(
        default=AuxiliaryHeadsConfig(),
    )

    # Training loss configuration:
    loss_config: LossConfig = field(default=LossConfig())
    use_clamped_fape_probability: float = 0.9
    self_distillation_plddt_threshold: float = 50.0

    # Adam optimizer constants:
    optimizer_adam_beta_1 = 0.9
    optimizer_adam_beta_2 = 0.999
    optimizer_adam_eps = 1e-6
    optimizer_adam_weight_decay = 0.0
    optimizer_adam_amsgrad = False

    # Whether to enable gradient clipping by the max norm value:
    gradient_clipping: bool = True
    clip_grad_max_norm: float = 0.1

    # Whether to enable Stochastic Weight Averaging (SWA):
    swa_enabled: bool = True
    swa_decay_rate: float = 0.9

    # Sequence crop & pad size (for "train" mode only):
    train_sequence_crop_size: int = 256  # N_res

    # Recycling (last dimension in the batch dict):
    num_recycling_iters: int = 3

    # Primary sequence and MSA related features names:
    primary_raw_feature_names: List[str] = field(
        default_factory=lambda: [
            "aatype",
            "residue_index",
            "msa",
            "num_alignments",
            "seq_length",
            "between_segment_residues",
            "deletion_matrix",
        ]
    )

    # MSA features configuration:
    max_msa_clusters: int = 124  # Number of clustered MSA sequences (N_clust)
    max_extra_msa: int = 1024  # Number of unclustered extra sequences (N_extra_seq)
    max_distillation_msa_clusters: int = 1000
    masked_msa_enabled: bool = True
    masked_msa_profile_prob: float = 0.1
    masked_msa_same_prob: float = 0.1
    masked_msa_uniform_prob: float = 0.1
    masked_msa_replace_fraction: float = 0.15
    msa_cluster_features: bool = True

    # Template features configuration:
    templates_enabled: bool = True
    max_templates: int = 4  # Number of templates (N_templ)
    shuffle_top_k_prefiltered: int = 20
    embed_template_torsion_angles: bool = True
    template_pair_feat_distogram_min_bin: float = 3.25
    template_pair_feat_distogram_max_bin: float = 50.75
    template_pair_feat_distogram_num_bins: int = 39
    template_pair_feat_use_unit_vector: bool = False
    template_pair_feat_inf: float = 1e5
    template_pair_feat_eps: float = 1e-6
    template_raw_feature_names: List[str] = field(
        default_factory=lambda: [
            "template_all_atom_positions",
            "template_sum_probs",
            "template_aatype",
            "template_all_atom_mask",
        ]
    )

    # Target and related to supervised training feature names:
    supervised_raw_features_names: List[str] = field(
        default_factory=lambda: [
            "all_atom_mask",
            "all_atom_positions",
            "resolution",
            "is_distillation",
        ]
    )

    @classmethod
    def from_preset(
        cls,
        stage: str,
        precision: str = "tf32",
        inference_chunk_size: int = 128,
    ) -> AlphaFoldConfig:
        cfg = {"preset": f"{stage}-{precision}"}

        if stage == "initial_training":
            _update(cfg, _initial_training_stage())
        elif stage == "finetuning":
            _update(cfg, _finetuning_stage())
        elif stage == "finetuning_ptm":
            _update(cfg, _finetuning_stage())
            _update(cfg, _ptm_preset())
        elif stage == "inference":
            _update(cfg, _inference_stage(chunk_size=inference_chunk_size))
        elif stage == "inference_ptm":
            _update(cfg, _inference_stage(chunk_size=inference_chunk_size))
            _update(cfg, _ptm_preset())
        else:
            raise ValueError(f"unknown stage={repr(stage)}")

        if precision in {"fp32", "tf32", "bf16"}:
            pass
        elif precision in {"amp", "fp16"}:
            _update(cfg, _half_precision_settings())
        else:
            raise ValueError(f"unknown precision={repr(precision)}")

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict) -> AlphaFoldConfig:
        return dacite.from_dict(
            data_class=AlphaFoldConfig,
            data=cfg,
            config=dacite.Config(
                check_types=True,
                strict=True,
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _initial_training_stage() -> dict:
    return {
        "train_sequence_crop_size": 256,  # N_res
        "max_msa_clusters": 124,  # N_clust
        "max_extra_msa": 1024,  # N_extra_seq
    }


def _finetuning_stage() -> dict:
    return {
        "train_sequence_crop_size": 384,  # N_res
        "max_msa_clusters": 508,  # N_clust
        "max_extra_msa": 5120,  # N_extra_seq
        "loss_config": {
            "experimentally_resolved_loss_config": {
                "weight": 0.01,
            },
            "violation_loss_config": {
                "weight": 1.0,
            },
        },
        "extra_msa_stack_config": {
            "chunk_size_msa_att": 1024,
            "chunk_size_opm": 128,
        },
    }


def _inference_stage(chunk_size: int) -> dict:
    return {
        "max_msa_clusters": 508,  # N_clust
        "max_extra_msa": 5120,  # N_extra_seq
        "template_pair_stack_config": {
            "chunk_size_tri_att": chunk_size,
        },
        "template_pointwise_attention_config": {
            "chunk_size": chunk_size,
        },
        "extra_msa_stack_config": {
            "chunk_size_msa_att": chunk_size,
            "chunk_size_opm": chunk_size,
            "chunk_size_tri_att": chunk_size,
        },
        "evoformer_stack_config": {
            "chunk_size_msa_att": chunk_size,
            "chunk_size_opm": chunk_size,
            "chunk_size_tri_att": chunk_size,
        },
    }


def _ptm_preset() -> dict:
    return {
        "auxiliary_heads_config": {
            "tm_score_head_enabled": True,
        },
        "loss_config": {
            "tm_loss_config": {
                "enabled": True,
                "weight": 0.1,
            },
        },
    }


def _half_precision_settings() -> dict:
    return {
        "recycling_embedder_config": {"inf": 1e4},
        "template_pair_stack_config": {"inf": 1e4},
        "template_pointwise_attention_config": {"inf": 1e4},
        "extra_msa_stack_config": {"inf": 1e4},
        "evoformer_stack_config": {"inf": 1e4},
        "structure_module_config": {"inf": 1e4},
        "template_pair_feat_inf": 1e4,
    }


def _update(left: dict, right: dict) -> dict:
    assert isinstance(left, dict)
    assert isinstance(right, dict)
    for k, v in right.items():
        if isinstance(v, dict):
            left[k] = _update(left.get(k, {}), v)
        else:
            left[k] = v
    return left


FEATURE_SHAPES = {
    "aatype": ("N_res",),
    "all_atom_mask": ("N_res", 37),
    "all_atom_positions": ("N_res", 37, 3),
    "atom14_alt_gt_exists": ("N_res", 14),
    "atom14_alt_gt_positions": ("N_res", 14, 3),
    "atom14_atom_exists": ("N_res", 14),
    "atom14_atom_is_ambiguous": ("N_res", 14),
    "atom14_gt_exists": ("N_res", 14),
    "atom14_gt_positions": ("N_res", 14, 3),
    "atom37_atom_exists": ("N_res", 37),
    "backbone_rigid_mask": ("N_res",),
    "backbone_rigid_tensor": ("N_res", 4, 4),
    "bert_mask": ("N_clust", "N_res"),
    "chi_angles_sin_cos": ("N_res", 4, 2),
    "chi_mask": ("N_res", 4),
    "extra_deletion_value": ("N_extra_seq", "N_res"),
    "extra_has_deletion": ("N_extra_seq", "N_res"),
    "extra_msa": ("N_extra_seq", "N_res"),
    "extra_msa_mask": ("N_extra_seq", "N_res"),
    "extra_msa_row_mask": ("N_extra_seq",),
    "is_distillation": (),
    "msa_feat": ("N_clust", "N_res", 49),
    "msa_mask": ("N_clust", "N_res"),
    "msa_row_mask": ("N_clust",),
    "pseudo_beta": ("N_res", 3),
    "pseudo_beta_mask": ("N_res",),
    "residue_index": ("N_res",),
    "residx_atom14_to_atom37": ("N_res", 14),
    "residx_atom37_to_atom14": ("N_res", 37),
    "resolution": (),
    "rigidgroups_alt_gt_frames": ("N_res", 8, 4, 4),
    "rigidgroups_group_exists": ("N_res", 8),
    "rigidgroups_group_is_ambiguous": ("N_res", 8),
    "rigidgroups_gt_exists": ("N_res", 8),
    "rigidgroups_gt_frames": ("N_res", 8, 4, 4),
    "seq_length": (),
    "seq_mask": ("N_res",),
    "target_feat": ("N_res", 22),
    "template_aatype": ("N_templ", "N_res"),
    "template_all_atom_mask": ("N_templ", "N_res", 37),
    "template_all_atom_positions": ("N_templ", "N_res", 37, 3),
    "template_alt_torsion_angles_sin_cos": ("N_templ", "N_res", 7, 2),
    "template_mask": ("N_templ",),
    "template_pseudo_beta": ("N_templ", "N_res", 3),
    "template_pseudo_beta_mask": ("N_templ", "N_res"),
    "template_sum_probs": ("N_templ", 1),
    "template_torsion_angles_mask": ("N_templ", "N_res", 7),
    "template_torsion_angles_sin_cos": ("N_templ", "N_res", 7, 2),
    "true_msa": ("N_clust", "N_res"),
}
