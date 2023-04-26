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
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import Sampler

from openfold.datasets import InitialTrainingDataset, ValidationDataset
from openfold.helpers import get_seed_from_string


class InitialTrainingSampler(Sampler[Tuple[int, int]]):
    """Sampler for initial training dataset."""

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        local_batch_size: int,
        global_batch_size: int,
        num_train_iters: int,
        seed: int,
        is_distributed: bool,
        rank: Optional[int],
        world_size: Optional[int],
        num_prev_iters: int,
    ) -> None:
        assert num_prev_iters <= num_train_iters
        if is_distributed:
            assert rank is not None
            assert world_size is not None
            assert global_batch_size % world_size == 0
        weights = dataset.get_sampler_weights()
        num_samples_in_device_epoch = num_train_iters * local_batch_size
        num_samples_in_global_epoch = num_train_iters * global_batch_size
        # Sample indices:
        index_generator = torch.Generator()
        index_generator.manual_seed(seed)
        random_indices = torch.multinomial(
            input=weights,
            num_samples=num_samples_in_global_epoch,
            replacement=True,
            generator=index_generator,
        )
        # Sample seeds:
        seed_generator = torch.Generator()
        seed_generator.manual_seed(seed)
        random_seeds = torch.randint(
            low=0,
            high=2**63 - 1,
            size=[num_samples_in_global_epoch],
            generator=seed_generator,
        )
        # Create (index, seed) pairs:
        assert random_indices.size() == random_seeds.size()
        indices = random_indices.tolist()
        seeds = random_seeds.tolist()
        assert len(indices) == len(seeds)
        index_seed_pairs = list(zip(indices, seeds))
        if is_distributed:
            index_seed_pairs = index_seed_pairs[rank::world_size]
        assert len(index_seed_pairs) == num_samples_in_device_epoch
        # Move forward by skipping previous iterations:
        offset = num_prev_iters * local_batch_size
        assert offset <= len(index_seed_pairs)
        self.index_seed_pairs = index_seed_pairs[offset:]

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        assert hasattr(self, "index_seed_pairs")
        yield from self.index_seed_pairs
        del self.index_seed_pairs

    def __len__(self) -> int:
        assert hasattr(self, "index_seed_pairs")
        return len(self.index_seed_pairs)


class ValidationSampler(Sampler[Tuple[int, int]]):
    """Sampler for validation dataset."""

    def __init__(
        self,
        dataset: ValidationDataset,
        is_distributed: bool,
        rank: Optional[int],
        world_size: Optional[int],
    ) -> None:
        dataset_length = len(dataset)
        if is_distributed:
            assert rank is not None
            assert world_size is not None
            epoch_length = math.ceil(dataset_length / world_size)
        else:
            epoch_length = dataset_length
        seeds = [
            get_seed_from_string(pdb_chain_id) for pdb_chain_id in dataset.pdb_chain_ids
        ]
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self._dataset_length = dataset_length
        self._epoch_length = epoch_length
        self._seeds = seeds

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        indices = list(range(self._dataset_length))
        if self.is_distributed:
            padding_size = self._epoch_length * self.world_size - self._dataset_length
            assert padding_size < self.world_size
            if padding_size <= len(indices):
                padding = indices[:padding_size]
            else:
                nrepeat = math.ceil(padding_size / len(indices))
                padding = (indices * nrepeat)[:padding_size]
            indices = indices + padding
            indices = indices[self.rank :: self.world_size]
        assert len(indices) == self._epoch_length
        seeds = [self._seeds[index] for index in indices]
        assert len(seeds) == self._epoch_length
        assert len(indices) == len(seeds)
        yield from zip(indices, seeds)

    def __len__(self) -> int:
        return self._epoch_length
