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

from typing import Dict, List, Optional

import torch


def dist_reduce_losses_avg(
    losses: Dict[str, torch.Tensor],
    is_main_process: bool,
    main_rank: int,
    device: torch.device,
    synchronize: bool,
) -> Optional[Dict[str, torch.Tensor]]:
    """Reduces to average losses from all processes.

    Should be used only when distributed training is on.

    All input `losses` must have the same order of keys.

    Returns averaged values in the main process, otherwise `None`.

    """
    reduce_tensor = torch.stack(list(losses.values()))
    assert reduce_tensor.ndim == 1
    reduce_tensor = reduce_tensor.to(device=device, dtype=torch.float32)
    torch.distributed.reduce(
        tensor=reduce_tensor,
        dst=main_rank,
        op=torch.distributed.ReduceOp.AVG,
    )
    if synchronize:
        torch.distributed.barrier()
    if not is_main_process:
        return None
    losses_avg = {}
    i = 0
    for key in losses.keys():
        losses_avg[key] = reduce_tensor[i]
        i += 1
    return losses_avg


def dist_gather_val_metrics(
    val_metrics_list: List[dict],
    val_pdb_chain_ids: List[str],
    is_main_process: bool,
    main_rank: int,
    world_size: int,
    device: torch.device,
    synchronize: bool,
) -> Optional[List[dict]]:
    """Gathers validation metrics to main process.

    Should be used only when distributed training is on.

    All dictionaries inside `val_metrics_list` must have the same order of keys.

    `val_pdb_chain_ids` must come from validation dataset
    and correspond to the order of validation samples.

    Returns gathered values in the main process, otherwise `None`.

    """
    keys = list(val_metrics_list[0].keys())
    keys.remove("pdb_chain_id")
    values_list = [
        [val_metrics[key] for key in keys] for val_metrics in val_metrics_list
    ]
    gather_tensor = torch.tensor(values_list, device=device)
    if is_main_process:
        gather_list = [torch.zeros_like(gather_tensor) for _ in range(world_size)]
    else:
        gather_list = None
    torch.distributed.gather(
        tensor=gather_tensor,
        gather_list=gather_list,
        dst=main_rank,
    )
    if synchronize:
        torch.distributed.barrier()
    if not is_main_process:
        return None
    gather_list = [tensor.cpu() for tensor in gather_list]
    gather_val_metrics_list = []
    for val_index in range(len(val_pdb_chain_ids)):
        i = val_index % world_size
        j = val_index // world_size
        gather_val_metrics = {}
        for k, key in enumerate(keys):
            gather_value = gather_list[i][j][k].item()
            if key == "val_index":
                gather_value = int(gather_value)
                assert gather_value == val_index
                gather_val_metrics["val_index"] = val_index
                gather_val_metrics["pdb_chain_id"] = val_pdb_chain_ids[val_index]
            else:
                gather_val_metrics[key] = gather_value
        gather_val_metrics_list.append(gather_val_metrics)
    return gather_val_metrics_list
