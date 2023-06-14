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

import datetime
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from openfold.config import AlphaFoldConfig
from openfold.data.alignments import load_alignments, load_alignments_super_index
from openfold.data.cameo_targets import CAMEO_TARGETS
from openfold.data.features import (
    create_mmcif_features,
    create_msa_features,
    create_template_features,
    process_features,
)
from openfold.data.mmcif import load_mmcif_chains_df, load_mmcif_dict
from openfold.data.templates import TemplateHitFeaturizer
from openfold.helpers import datetime_from_string


class PDBDataset(Dataset):
    """Dataset containing Protein Data Bank (PDB) structures."""

    def __init__(
        self,
        mmcif_chains_df: pd.DataFrame,
        alignments_super_index: Dict[str, dict],
        pdb_mmcif_dicts_dirpath: Path,
        pdb_alignments_dirpath: Path,
        template_hit_featurizer: TemplateHitFeaturizer,
        alphafold_config: AlphaFoldConfig,
        mode: str,  # "train" or "eval"
        verbose: bool = False,
        name: str = "InitialDataset",
    ) -> None:
        assert mode in {"train", "eval"}
        self.mmcif_chains = mmcif_chains_df.to_dict("records")
        self.alignments_super_index = alignments_super_index
        self.pdb_mmcif_dicts_dirpath = pdb_mmcif_dicts_dirpath
        self.pdb_alignments_dirpath = pdb_alignments_dirpath
        self.template_hit_featurizer = template_hit_featurizer
        self.alphafold_config = alphafold_config
        self.mode = mode
        self.verbose = verbose
        self.name = name
        if verbose:
            print(f"{name}: initialized successfully!")

    def __getitem__(self, index_seed_pair: Tuple[int, int]) -> dict:
        if not isinstance(index_seed_pair, tuple):
            raise TypeError(
                f"__getitem__ expects {tuple} in format (index, seed),"
                f" but provided {type(index_seed_pair)} argument"
                f" contains value {repr(index_seed_pair)}"
            )
        index, seed = index_seed_pair
        assert isinstance(index, int)
        assert isinstance(seed, int)

        # Get sample metadata:
        sample = self.mmcif_chains[index]
        sequence = sample["sequence"]
        pdb_id = sample["pdb_id"]
        pdb_chain_id = sample["pdb_chain_id"]
        author_chain_id = sample["author_chain_id"]
        release_date = sample["release_date"]
        seqlen = len(sequence)

        # Load sample data (mmcif):
        mmcif_dict = load_mmcif_dict(
            mmcif_dicts_dirpath=self.pdb_mmcif_dicts_dirpath,
            pdb_id=pdb_id,
        )

        # Load sample data (alignments):
        alignments = load_alignments(
            alignments_super_index=self.alignments_super_index,
            alignments_dirpath=self.pdb_alignments_dirpath,
            key=pdb_chain_id,
        )

        # Create mmCIF features:
        mmcif_features = create_mmcif_features(
            mmcif_dict=mmcif_dict,
            author_chain_id=author_chain_id,
        )

        # Create template features:
        template_features = create_template_features(
            sequence=sequence,
            hhr_string=alignments.get("pdb70_hits.hhr", ""),
            template_hit_featurizer=self.template_hit_featurizer,
            release_date=release_date,
            pdb_id=pdb_id,
            days_before_release=60,
            shuffling_seed=seed,
        )

        # Create MSA features:
        msa_features = create_msa_features(
            sequence=sequence,
            a3m_strings=[
                alignments.get("uniref90_hits.a3m", ""),
                alignments.get("bfd_uniclust_hits.a3m", ""),
                alignments.get("mgnify_hits.a3m", ""),
            ],
        )

        # Process features:
        raw_features = {**mmcif_features, **template_features, **msa_features}
        feats = process_features(
            raw_features=raw_features,
            alphafold_config=self.alphafold_config,
            mode=self.mode,
            seed=seed,
        )

        # Add id tuple:
        feats["id"] = (self.name, index, seed, pdb_chain_id, seqlen)

        return feats

    def __len__(self) -> int:
        return len(self.mmcif_chains)


class InitialTrainingDataset(PDBDataset):
    """Dataset for the initial training stage."""

    def __init__(
        self,
        pdb_mmcif_chains_filepath: Path,
        pdb_mmcif_dicts_dirpath: Path,
        pdb_obsolete_filepath: Path,
        pdb_alignments_dirpath: Path,
        max_pdb_release_date: str,
        alphafold_config: AlphaFoldConfig,
        filter_by_alignments: bool = False,
        use_only_pdb_chain_ids: Optional[List[str]] = None,
        verbose: bool = False,
        name: str = "InitialTrainingDataset",
    ) -> None:
        if verbose:
            print(f"{name}: initialization...")

        # Load pdb mmcif chains metadata:
        pdb_mmcif_chains_df = load_mmcif_chains_df(
            mmcif_chains_filepath=pdb_mmcif_chains_filepath,
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        # Load alignments metadata:
        alignments_super_index = load_alignments_super_index(
            alignments_super_index_filepath=(pdb_alignments_dirpath / "super.index"),
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        # Select pdb chains used as training samples:
        mmcif_chains_df = _filter_pdb_chains_for_training(
            mmcif_chains_df=pdb_mmcif_chains_df,
            min_release_date="1900-01-01",
            max_release_date=max_pdb_release_date,
        )

        if use_only_pdb_chain_ids is not None:
            assert isinstance(use_only_pdb_chain_ids, list)
            selector = mmcif_chains_df["pdb_chain_id"].isin(set(use_only_pdb_chain_ids))
            mmcif_chains_df = mmcif_chains_df[selector].copy()

        if filter_by_alignments:
            mmcif_chains_df = _filter_by_alignments(
                mmcif_chains_df=mmcif_chains_df,
                alignments_super_index=alignments_super_index,
            )
        else:
            _assert_pdb_alignments(
                mmcif_chains_df=mmcif_chains_df,
                alignments_super_index=alignments_super_index,
            )

        if verbose:
            print(f"{name}: mmcif_chains_df.shape={mmcif_chains_df.shape}")

        # Compute pdb cluster size:
        mmcif_chains_df = _compute_pdb_cluster_size(mmcif_chains_df)

        if verbose:
            print(f"{name}: mmcif_chains_df.shape={mmcif_chains_df.shape}")

        # Create template hit featurizer:
        template_hit_featurizer = TemplateHitFeaturizer(
            max_template_hits=alphafold_config.max_templates,
            pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
            template_pdb_chain_ids=set(pdb_mmcif_chains_df["pdb_chain_id"]),
            pdb_release_dates=_get_pdb_release_dates(pdb_mmcif_chains_df),
            pdb_obsolete_filepath=pdb_obsolete_filepath,
            shuffle_top_k_prefiltered=alphafold_config.shuffle_top_k_prefiltered,
            verbose=False,
        )

        super().__init__(
            mmcif_chains_df=mmcif_chains_df,
            alignments_super_index=alignments_super_index,
            pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
            pdb_alignments_dirpath=pdb_alignments_dirpath,
            template_hit_featurizer=template_hit_featurizer,
            alphafold_config=alphafold_config,
            mode="train",
            verbose=verbose,
            name=name,
        )

    def get_sampler_weights(self) -> torch.Tensor:
        """Get weights for training sampler (Supplementary '1.2.5 Filtering')."""
        return torch.tensor(
            data=[_get_weight(sample) for sample in self.mmcif_chains],
            dtype=torch.float64,
        )


class ValidationDataset(PDBDataset):
    """Validation dataset."""

    def __init__(
        self,
        pdb_mmcif_chains_filepath: Path,
        pdb_mmcif_dicts_dirpath: Path,
        pdb_obsolete_filepath: Path,
        pdb_alignments_dirpath: Path,
        min_cameo_submission_date: str,
        max_cameo_submission_date: str,
        max_sequence_length: int,
        alphafold_config: AlphaFoldConfig,
        filter_by_alignments: bool = False,
        use_only_pdb_chain_ids: Optional[List[str]] = None,
        verbose: bool = False,
        name: str = "ValidationDataset",
    ) -> None:
        if verbose:
            print(f"{name}: initialization...")

        # Load pdb mmcif chains metadata:
        pdb_mmcif_chains_df = load_mmcif_chains_df(
            mmcif_chains_filepath=pdb_mmcif_chains_filepath,
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        # Load alignments metadata:
        alignments_super_index = load_alignments_super_index(
            alignments_super_index_filepath=(pdb_alignments_dirpath / "super.index"),
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        # Select pdb chains used as validation samples:
        mmcif_chains_df = _select_cameo_targets_for_validation(
            mmcif_chains_df=pdb_mmcif_chains_df,
            min_submission_date=min_cameo_submission_date,
            max_submission_date=max_cameo_submission_date,
            max_sequence_length=max_sequence_length,
        )

        if use_only_pdb_chain_ids is not None:
            assert isinstance(use_only_pdb_chain_ids, list)
            selector = mmcif_chains_df["pdb_chain_id"].isin(set(use_only_pdb_chain_ids))
            mmcif_chains_df = mmcif_chains_df[selector].copy()

        if filter_by_alignments:
            mmcif_chains_df = _filter_by_alignments(
                mmcif_chains_df=mmcif_chains_df,
                alignments_super_index=alignments_super_index,
            )
        else:
            _assert_pdb_alignments(
                mmcif_chains_df=mmcif_chains_df,
                alignments_super_index=alignments_super_index,
            )

        if verbose:
            print(f"{name}: mmcif_chains_df.shape={mmcif_chains_df.shape}")

        # Create template hit featurizer:
        template_hit_featurizer = TemplateHitFeaturizer(
            max_template_hits=alphafold_config.max_templates,
            pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
            template_pdb_chain_ids=set(pdb_mmcif_chains_df["pdb_chain_id"]),
            pdb_release_dates=_get_pdb_release_dates(pdb_mmcif_chains_df),
            pdb_obsolete_filepath=pdb_obsolete_filepath,
            shuffle_top_k_prefiltered=None,
            verbose=False,
        )

        super().__init__(
            mmcif_chains_df=mmcif_chains_df,
            alignments_super_index=alignments_super_index,
            pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
            pdb_alignments_dirpath=pdb_alignments_dirpath,
            template_hit_featurizer=template_hit_featurizer,
            alphafold_config=alphafold_config,
            mode="eval",
            verbose=verbose,
            name=name,
        )

    @property
    def pdb_chain_ids(self) -> List[str]:
        return [mmcif_chain["pdb_chain_id"] for mmcif_chain in self.mmcif_chains]


def _filter_pdb_chains_for_training(
    mmcif_chains_df: pd.DataFrame,
    min_release_date: str = "1900-01-01",
    max_release_date: str = "2999-12-31",
    max_resolution: float = 9.0,
    max_aa_frequency: float = 0.8,
) -> pd.DataFrame:
    # Supplementary '1.2.5 Filtering'
    is_release_date_between = mmcif_chains_df["release_date"].between(
        left=min_release_date,
        right=max_release_date,
        inclusive="both",
    )
    is_resolution_below_max = mmcif_chains_df["resolution"] < max_resolution
    is_resolution_nan = mmcif_chains_df["resolution"].isnull()
    is_resolution_acceptable = is_resolution_below_max | is_resolution_nan
    is_aa_frequency_acceptable = pd.Series(
        [
            _is_aa_frequency_acceptable(sequence, max_aa_frequency)
            for sequence in mmcif_chains_df["sequence"].tolist()
        ]
    )
    is_in_pdb_clusters = ~mmcif_chains_df["pdb_cluster_id"].eq(-1)
    selector = (
        is_release_date_between
        & is_resolution_acceptable
        & is_aa_frequency_acceptable
        & is_in_pdb_clusters
    )
    mmcif_chains_df = mmcif_chains_df[selector].copy()
    return mmcif_chains_df


def _is_aa_frequency_acceptable(sequence: str, max_aa_frequency: float) -> bool:
    if len(sequence) == 0:
        return False
    cnt = Counter(sequence)
    top = cnt.most_common(1)[0]
    top_aa_count = top[1]
    top_aa_freq = top_aa_count / len(sequence)
    return top_aa_freq <= max_aa_frequency


def _get_weight(sample: dict) -> float:
    # Supplementary '1.2.5 Filtering'
    sequence_length = sample["sequence_length"]
    pdb_cluster_size = sample["pdb_cluster_size"]

    length_probability = max(min(sequence_length, 512), 256) / 512

    cluster_probability = 1.0
    if pdb_cluster_size > 0:
        cluster_probability = 1 / math.sqrt(pdb_cluster_size)

    weight = length_probability * cluster_probability
    return weight


def _select_cameo_targets_for_validation(
    mmcif_chains_df: pd.DataFrame,
    min_submission_date: str,
    max_submission_date: str,
    max_sequence_length: int = 700,
) -> pd.DataFrame:
    # Supplementary '1.11.7 Evaluator setup'
    selected_cameo_targets = {}
    pdb_chain_ids_mapping = _get_pdb_chain_ids_mapping(mmcif_chains_df)
    cameo_submission_dates = list(CAMEO_TARGETS.keys())
    for cameo_submission_date in cameo_submission_dates:
        if not (min_submission_date <= cameo_submission_date <= max_submission_date):
            # ignore dates out of query range
            continue
        for cameo_target_id in CAMEO_TARGETS[cameo_submission_date]:
            cameo_pdb_mmcif_chain_id = _parse_cameo_target(cameo_target_id)
            if cameo_pdb_mmcif_chain_id not in pdb_chain_ids_mapping:
                # ignore CAMEO targets not found in mmcif chains (PDB)
                continue
            cameo_pdb_author_chain_id = pdb_chain_ids_mapping[cameo_pdb_mmcif_chain_id]
            if cameo_pdb_author_chain_id in selected_cameo_targets:
                # ignore CAMEO targets that map to already selected pdb author chain id
                continue
            selected_cameo_targets[cameo_pdb_author_chain_id] = cameo_target_id
    is_cameo_target = mmcif_chains_df["pdb_chain_id"].isin(selected_cameo_targets)
    mmcif_chains_df = mmcif_chains_df[is_cameo_target].copy()
    # Filter by max sequence length:
    is_sequence_length_acceptable = (
        mmcif_chains_df["sequence_length"] <= max_sequence_length
    )
    mmcif_chains_df = mmcif_chains_df[is_sequence_length_acceptable].copy()
    # Add cameo target column:
    selected_cameo_targets_df = pd.DataFrame(
        data={
            "pdb_chain_id": list(selected_cameo_targets.keys()),
            "cameo_target_id": list(selected_cameo_targets.values()),
        }
    )
    mmcif_chains_df = mmcif_chains_df.merge(
        right=selected_cameo_targets_df,
        how="left",
        on="pdb_chain_id",
    )
    return mmcif_chains_df


def _get_pdb_chain_ids_mapping(mmcif_chains_df: pd.DataFrame) -> Dict[str, str]:
    """Get mapping from `{pdb_id}_{mmcif_chain_id}` to `{pdb_id}_{author_chain_id}`."""
    pdb_chain_ids_mapping = {}
    pdb_author_chain_ids = set()
    for pdb_id, author_chain_id, mmcif_chain_ids in zip(
        mmcif_chains_df["pdb_id"].values,
        mmcif_chains_df["author_chain_id"].values,
        mmcif_chains_df["mmcif_chain_ids"].values,
    ):
        mmcif_chain_ids = mmcif_chain_ids.split(";")
        pdb_author_chain_id = f"{pdb_id}_{author_chain_id}"
        for mmcif_chain_id in mmcif_chain_ids:
            pdb_mmcif_chain_id = f"{pdb_id}_{mmcif_chain_id}"
            assert pdb_mmcif_chain_id not in pdb_chain_ids_mapping
            pdb_chain_ids_mapping[pdb_mmcif_chain_id] = pdb_author_chain_id
        pdb_author_chain_ids.add(pdb_author_chain_id)
    # Sometimes CAMEO targets are in format: "pdb_id [author_chain_id]"
    # The following loop will allow them to be included in the validation set.
    for pdb_author_chain_id in list(pdb_author_chain_ids):
        if pdb_author_chain_id not in pdb_chain_ids_mapping:
            pdb_chain_ids_mapping[pdb_author_chain_id] = pdb_author_chain_id
    return pdb_chain_ids_mapping


def _parse_cameo_target(cameo_target_id: str) -> str:
    pdb_id, mmcif_chain_id = cameo_target_id.split()
    pdb_id = pdb_id.lower()
    mmcif_chain_id = mmcif_chain_id.strip("[]")
    pdb_mmcif_chain_id = f"{pdb_id}_{mmcif_chain_id}"
    return pdb_mmcif_chain_id


def _filter_by_alignments(
    mmcif_chains_df: pd.DataFrame,
    alignments_super_index: Dict[str, dict],
) -> pd.DataFrame:
    selector = mmcif_chains_df["pdb_chain_id"].isin(alignments_super_index)
    mmcif_chains_df = mmcif_chains_df[selector].copy()
    return mmcif_chains_df


def _assert_pdb_alignments(
    mmcif_chains_df: pd.DataFrame,
    alignments_super_index: Dict[str, dict],
) -> None:
    pdb_chain_ids = set(mmcif_chains_df["pdb_chain_id"])
    alignments_super_index_ids = set(alignments_super_index.keys())
    if not pdb_chain_ids.issubset(alignments_super_index_ids):
        diff = pdb_chain_ids - alignments_super_index_ids
        raise RuntimeError(
            f"`mmcif_chains_df` has {len(diff)} ids"
            " not present in `alignments_super_index`."
            " To filter them out, set `filter_by_alignments` flag."
        )


def _get_pdb_release_dates(
    pdb_mmcif_chains_df: pd.DataFrame,
) -> Dict[str, datetime.datetime]:
    return {
        pdb_id: datetime_from_string(release_date, "%Y-%m-%d")
        for pdb_id, release_date in zip(
            pdb_mmcif_chains_df["pdb_id"].values,
            pdb_mmcif_chains_df["release_date"].values,
        )
    }


def _compute_pdb_cluster_size(mmcif_chains_df: pd.DataFrame) -> pd.DataFrame:
    vc = mmcif_chains_df["pdb_cluster_id"].value_counts()
    if -1 in vc:
        # -1 means 'unassigned to any cluster'
        vc[-1] = 0
    vc = vc.rename("pdb_cluster_size")
    vc = vc.reset_index()
    vc = vc.rename(columns={"index": "pdb_cluster_id"})
    mmcif_chains_df = mmcif_chains_df.merge(vc, on="pdb_cluster_id", how="left")
    return mmcif_chains_df
