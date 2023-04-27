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

import shutil
import time
from pathlib import Path
from typing import List, Optional

import torch

from openfold.model.alphafold import AlphaFold
from openfold.swa import AlphaFoldSWA

RESUMABLE_CHECKPOINT_FILENAME = "resumable_checkpoint.pt"
SWA_CHECKPOINT_FILENAME = "swa_checkpoint.pt"


def resume_checkpoint(
    alphafold: AlphaFold,
    optimizer: Optional[torch.optim.Optimizer],
    swa_alphafold: Optional[AlphaFoldSWA],
    checkpoint_dirpath: Path,
    device: torch.device,
) -> int:
    # Load the resumable checkpoint:
    resumable_checkpoint_filepath = checkpoint_dirpath / RESUMABLE_CHECKPOINT_FILENAME
    resumable_checkpoint = torch.load(
        resumable_checkpoint_filepath, map_location=device
    )
    iteration = resumable_checkpoint["iteration"]
    alphafold_state_dict = resumable_checkpoint["alphafold_state_dict"]
    alphafold.load_state_dict(alphafold_state_dict, strict=True)
    if optimizer is not None:
        optimizer_state_dict = resumable_checkpoint["optimizer_state_dict"]
        optimizer.load_state_dict(optimizer_state_dict)
    # Load SWA state dict:
    if swa_alphafold is not None and swa_alphafold.enabled:
        swa_checkpoint_filepath = checkpoint_dirpath / SWA_CHECKPOINT_FILENAME
        swa_state_dict = torch.load(swa_checkpoint_filepath, map_location=device)
        swa_alphafold.load_state_dict(swa_state_dict, strict=True)
    return iteration


def resume_from_latest_checkpoint(
    alphafold: AlphaFold,
    optimizer: Optional[torch.optim.Optimizer],
    swa_alphafold: Optional[AlphaFoldSWA],
    training_dirpath: Path,
    device: torch.device,
    verbose: bool,
) -> int:
    checkpoints_dirpath = training_dirpath / "checkpoints"
    last_checkpoints_dirpath = checkpoints_dirpath / "last"
    last_checkpoint_dirpaths = _get_sorted_last_checkpoint_dirpaths(
        last_checkpoints_dirpath=last_checkpoints_dirpath,
    )
    if len(last_checkpoint_dirpaths) == 0:
        return 0
    last_checkpoint_dirpath = last_checkpoint_dirpaths[0]
    if verbose:
        print(f"Resuming checkpoint from {repr(last_checkpoint_dirpath)}...")
    iteration = resume_checkpoint(
        alphafold=alphafold,
        optimizer=optimizer,
        swa_alphafold=swa_alphafold,
        checkpoint_dirpath=last_checkpoint_dirpath,
        device=device,
    )
    if verbose:
        print(f"Checkpoint resumed from {repr(last_checkpoint_dirpath)} successfully!")
    return iteration


def save_checkpoint(
    alphafold: AlphaFold,
    optimizer: torch.optim.Optimizer,
    swa_alphafold: AlphaFoldSWA,
    iteration: int,
    checkpoint_dirpath: Path,
) -> None:
    checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    # Save SWA state dict:
    if swa_alphafold.enabled:
        swa_state_dict = swa_alphafold.state_dict()
        swa_checkpoint_filepath = checkpoint_dirpath / SWA_CHECKPOINT_FILENAME
        torch.save(swa_state_dict, swa_checkpoint_filepath)
    # Save the resumable checkpoint:
    if hasattr(alphafold, "module"):
        alphafold_state_dict = alphafold.module.state_dict()
    else:
        alphafold_state_dict = alphafold.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    resumable_checkpoint = {
        "iteration": iteration,
        "alphafold_state_dict": alphafold_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }
    resumable_checkpoint_filepath = checkpoint_dirpath / RESUMABLE_CHECKPOINT_FILENAME
    torch.save(resumable_checkpoint, resumable_checkpoint_filepath)


def save_checkpoint_from_training(
    alphafold: AlphaFold,
    optimizer: torch.optim.Optimizer,
    swa_alphafold: AlphaFoldSWA,
    iteration: int,
    training_dirpath: Path,
    keep_last_checkpoints: int,
    keep_best_checkpoints: int,
    keep_val_checkpoints: bool,
    is_validation: bool,
    val_avg_lddt_ca: Optional[float],
) -> None:
    not_save_checkpoint = (
        keep_last_checkpoints == 0
        and keep_best_checkpoints == 0
        and not keep_val_checkpoints
    )
    if not_save_checkpoint:
        return
    print("Saving checkpoint...")
    perf = -time.perf_counter()
    checkpoints_dirpath = training_dirpath / "checkpoints"
    iteration_str = f"{iteration:06}"
    # Save tmp checkpoint:
    tmp_checkpoint_dirpath = checkpoints_dirpath / ".tmp"
    save_checkpoint(
        alphafold=alphafold,
        optimizer=optimizer,
        swa_alphafold=swa_alphafold,
        iteration=iteration,
        checkpoint_dirpath=tmp_checkpoint_dirpath,
    )
    if is_validation:
        assert val_avg_lddt_ca is not None
        assert 0.0 <= val_avg_lddt_ca <= 1.0
        val_avg_lddt_ca_str = f"{val_avg_lddt_ca:.6f}".replace(".", "")
    # Copy if validation checkpoint:
    if is_validation and keep_val_checkpoints:
        val_checkpoint_dirpath = checkpoints_dirpath / "val" / f"{iteration_str}"
        _copy_checkpoint_dirpath(
            source_dirpath=tmp_checkpoint_dirpath,
            target_dirpath=val_checkpoint_dirpath,
            force=True,
        )
        print(f"Saved to {repr(val_checkpoint_dirpath)} successfully!")
    # Copy if best checkpoint:
    if is_validation and keep_best_checkpoints > 0:
        best_checkpoint_dirpath = (
            checkpoints_dirpath / "best" / f"{val_avg_lddt_ca_str}_{iteration_str}"
        )
        if _is_best_checkpoint_to_save(
            best_checkpoint_dirpath=best_checkpoint_dirpath,
            keep_best_checkpoints=keep_best_checkpoints,
        ):
            _copy_checkpoint_dirpath(
                source_dirpath=tmp_checkpoint_dirpath,
                target_dirpath=best_checkpoint_dirpath,
                force=True,
            )
            print(f"Saved to {repr(best_checkpoint_dirpath)} successfully!")
    # Move tmp to last checkpoints:
    last_checkpoint_dirpath = checkpoints_dirpath / "last" / iteration_str
    _move_checkpoint_dirpath(
        source_dirpath=tmp_checkpoint_dirpath,
        target_dirpath=last_checkpoint_dirpath,
        force=True,
    )
    print(f"Saved to {repr(last_checkpoint_dirpath)} successfully!")
    # Delete expendable checkpoints:
    _delete_best_checkpoints(
        best_checkpoints_dirpath=(checkpoints_dirpath / "best"),
        keep_best_checkpoints=keep_best_checkpoints,
    )
    _delete_last_checkpoints(
        last_checkpoints_dirpath=(checkpoints_dirpath / "last"),
        keep_last_checkpoints=keep_last_checkpoints,
    )
    perf += time.perf_counter()
    print(f"Checkpoint saved successfully! ({perf:.3f}s)")


def _copy_checkpoint_dirpath(
    source_dirpath: Path,
    target_dirpath: Path,
    force: bool,
) -> None:
    assert source_dirpath != target_dirpath
    if target_dirpath.exists() and force:
        shutil.rmtree(target_dirpath)
    assert not target_dirpath.exists()
    shutil.copytree(src=source_dirpath, dst=target_dirpath)


def _move_checkpoint_dirpath(
    source_dirpath: Path,
    target_dirpath: Path,
    force: bool,
) -> None:
    assert source_dirpath != target_dirpath
    if target_dirpath.exists() and force:
        shutil.rmtree(target_dirpath)
    assert not target_dirpath.exists()
    shutil.move(src=source_dirpath, dst=target_dirpath)


def _get_sorted_best_checkpoint_dirpaths(best_checkpoints_dirpath: Path) -> List[Path]:
    assert best_checkpoints_dirpath.name == "best"
    best_checkpoint_dirpaths = list(best_checkpoints_dirpath.glob("[0-9_]*"))
    return sorted(best_checkpoint_dirpaths, reverse=True)


def _get_sorted_last_checkpoint_dirpaths(last_checkpoints_dirpath: Path) -> List[Path]:
    assert last_checkpoints_dirpath.name == "last"
    last_checkpoint_dirpaths = list(last_checkpoints_dirpath.glob("[0-9]*"))
    return sorted(last_checkpoint_dirpaths, reverse=True)


def _delete_best_checkpoints(
    best_checkpoints_dirpath: Path,
    keep_best_checkpoints: int,
) -> None:
    sorted_best_checkpoints = _get_sorted_best_checkpoint_dirpaths(
        best_checkpoints_dirpath=best_checkpoints_dirpath,
    )
    surplus_best_checkpoints = sorted_best_checkpoints[keep_best_checkpoints:]
    for surplus_best_checkpoint in surplus_best_checkpoints:
        shutil.rmtree(surplus_best_checkpoint)


def _delete_last_checkpoints(
    last_checkpoints_dirpath: Path,
    keep_last_checkpoints: int,
) -> None:
    sorted_last_checkpoints = _get_sorted_last_checkpoint_dirpaths(
        last_checkpoints_dirpath=last_checkpoints_dirpath,
    )
    surplus_last_checkpoints = sorted_last_checkpoints[keep_last_checkpoints:]
    for surplus_last_checkpoint in surplus_last_checkpoints:
        shutil.rmtree(surplus_last_checkpoint)


def _is_best_checkpoint_to_save(
    best_checkpoint_dirpath: Path,
    keep_best_checkpoints: int,
) -> bool:
    if keep_best_checkpoints == 0:
        return False
    sorted_best_checkpoints = _get_sorted_best_checkpoint_dirpaths(
        best_checkpoints_dirpath=best_checkpoint_dirpath.parent,
    )
    if keep_best_checkpoints > len(sorted_best_checkpoints):
        return True
    for checkpoint_dirpath in sorted_best_checkpoints[:keep_best_checkpoints]:
        if best_checkpoint_dirpath >= checkpoint_dirpath:
            return True
    return False
