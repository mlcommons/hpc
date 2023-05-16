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

import queue
import random
import threading
from typing import Iterator, List

import torch
from torch.utils.data import DataLoader

from openfold.datasets import InitialTrainingDataset, ValidationDataset
from openfold.samplers import InitialTrainingSampler, ValidationSampler
from openfold.torch_utils import collate, map_tensor_tree


class InitialTrainingDataloaderPT(DataLoader):
    """Dataloader for the initial training stage."""

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        sampler: InitialTrainingSampler,
        local_batch_size: int,
        num_workers: int,
        seed: int,
        uniform_recycling_iters: List[int],
        gradient_accumulation_iters: int,
        num_prev_iters: int,
    ) -> None:
        super(InitialTrainingDataloaderPT, self).__init__(
            dataset=dataset,
            collate_fn=collate,
            sampler=sampler,
            batch_size=local_batch_size,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=(4 if num_workers > 0 else None),
            persistent_workers=bool(num_workers > 0),
        )
        self._set_train_batch_properties_fn = TrainBatchProperties(
            seed=seed,
            uniform_recycling_iters=uniform_recycling_iters,
            gradient_accumulation_iters=gradient_accumulation_iters,
            num_prev_iters=num_prev_iters,
        )

    def __iter__(self) -> Iterator[dict]:
        iterator = super().__iter__()
        for batch in iterator:
            yield self._set_train_batch_properties_fn(batch)


class InitialTrainingDataloaderPQ:
    """Dataloader for the initial training stage with non-blocking priority queue."""

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        sampler: InitialTrainingSampler,
        local_batch_size: int,
        num_workers: int,
        seed: int,
        uniform_recycling_iters: List[int],
        gradient_accumulation_iters: int,
        num_prev_iters: int,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = local_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = 4
        self._set_train_batch_properties_fn = TrainBatchProperties(
            seed=seed,
            uniform_recycling_iters=uniform_recycling_iters,
            gradient_accumulation_iters=gradient_accumulation_iters,
            num_prev_iters=num_prev_iters,
        )

    def _start_workers(self) -> None:
        # create queues:
        queue_maxsize = self.num_workers * self.prefetch_factor
        self._index_seed_pair_queue = torch.multiprocessing.Queue(maxsize=queue_maxsize)
        self._sample_queue = torch.multiprocessing.Queue(maxsize=queue_maxsize)
        self._batch_queue = queue.PriorityQueue(maxsize=queue_maxsize)

        # create sampler process:
        self._sampler_process = torch.multiprocessing.Process(
            target=_initial_sampler_worker,
            args=(
                self.sampler,
                self._index_seed_pair_queue,
            ),
        )
        self._sampler_process.daemon = True
        self._sampler_process.start()

        # create worker processes:
        self._worker_processes = []
        for _ in range(self.num_workers):
            worker_process = torch.multiprocessing.Process(
                target=_initial_training_worker,
                args=(
                    self.dataset,
                    self._index_seed_pair_queue,
                    self._sample_queue,
                ),
            )
            worker_process.daemon = True
            worker_process.start()
            self._worker_processes.append(worker_process)

        # create batcher thread:
        self._batcher_thread = threading.Thread(
            target=_initial_batcher_thread,
            args=(
                self._sample_queue,
                self._batch_queue,
                self.batch_size,
            ),
        )
        self._batcher_thread.daemon = True
        self._batcher_thread.start()

    def _close_workers(self) -> None:
        if hasattr(self, "_sampler_process"):
            self._sampler_process.terminate()
            del self._sampler_process

        if hasattr(self, "_worker_processes"):
            for worker_process in self._worker_processes:
                worker_process.terminate()
            del self._worker_processes

        if hasattr(self, "_batcher_thread"):
            del self._batcher_thread

        if hasattr(self, "_index_seed_pair_queue"):
            del self._index_seed_pair_queue

        if hasattr(self, "_sample_queue"):
            del self._sample_queue

        if hasattr(self, "_batch_queue"):
            del self._batch_queue

    def _multiprocessing_iter(self) -> Iterator[dict]:
        self._start_workers()
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.batch_size
        for _ in range(num_dataloader_iters):
            _, batch = self._batch_queue.get()
            yield batch
        self._close_workers()

    def _synchronous_iter(self) -> Iterator[dict]:
        sampler_iterator = iter(self.sampler)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.batch_size
        for _ in range(num_dataloader_iters):
            samples = []
            for _ in range(self.batch_size):
                index_seed_pair = next(sampler_iterator)
                sample = self.dataset[index_seed_pair]
                samples.append(sample)
            batch = collate(samples)
            yield batch

    def __iter__(self) -> Iterator[dict]:
        if self.num_workers > 0:
            iterator = self._multiprocessing_iter()
        elif self.num_workers == 0:
            iterator = self._synchronous_iter()
        for batch in iterator:
            yield self._set_train_batch_properties_fn(batch)

    def __del__(self) -> None:
        self._close_workers()


class ValidationDataloader(DataLoader):
    """Validation dataloader."""

    def __init__(
        self,
        dataset: ValidationDataset,
        sampler: ValidationSampler,
        num_workers: int,
    ) -> None:
        super(ValidationDataloader, self).__init__(
            dataset=dataset,
            collate_fn=collate,
            sampler=sampler,
            batch_size=1,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=(4 if num_workers > 0 else None),
            persistent_workers=bool(num_workers > 0),
        )


class TrainBatchProperties:
    """Assigns randomized global train batch properties."""

    def __init__(
        self,
        seed: int,
        uniform_recycling_iters: List[int],
        gradient_accumulation_iters: int,
        num_prev_iters: int,
    ) -> None:
        self._random_num_recycling_iters_iterator = (
            _random_num_recycling_iters_generator(
                uniform_recycling_iters=uniform_recycling_iters,
                seed=seed,
            )
        )
        assert gradient_accumulation_iters >= 1
        self._gradient_accumulation_iters = gradient_accumulation_iters
        assert num_prev_iters >= 0
        self._iteration = num_prev_iters
        self._num_recycling_iters = None
        # restore rng state by iterating through previous iterations:
        assert num_prev_iters % gradient_accumulation_iters == 0
        for _ in range(num_prev_iters // gradient_accumulation_iters):
            next(self._random_num_recycling_iters_iterator)

    def __call__(self, batch: dict) -> dict:
        self._iteration += 1
        if (self._iteration - 1) % self._gradient_accumulation_iters == 0:
            self._num_recycling_iters = next(self._random_num_recycling_iters_iterator)
        assert self._num_recycling_iters is not None
        batch = map_tensor_tree(
            fn=lambda t: t[..., : self._num_recycling_iters + 1],
            tree=batch,
        )
        return batch


def _random_num_recycling_iters_generator(
    uniform_recycling_iters: List[int],
    seed: int,
) -> Iterator[int]:
    assert isinstance(uniform_recycling_iters, list)
    assert len(uniform_recycling_iters) > 0
    rng = random.Random(seed)
    while True:
        num_recycling_iters_values = uniform_recycling_iters.copy()
        rng.shuffle(num_recycling_iters_values)
        for num_recycling_iters in num_recycling_iters_values:
            yield num_recycling_iters


def _initial_sampler_worker(
    sampler: InitialTrainingSampler,
    index_seed_pair_queue: torch.multiprocessing.Queue,
) -> None:
    for priority, index_seed_pair in enumerate(sampler):
        index_seed_pair_queue.put((priority, index_seed_pair))


def _initial_training_worker(
    dataset: InitialTrainingDataset,
    index_seed_pair_queue: torch.multiprocessing.Queue,
    sample_queue: torch.multiprocessing.Queue,
) -> None:
    while True:
        priority, index_seed_pair = index_seed_pair_queue.get()
        sample = dataset[index_seed_pair]
        sample_queue.put((priority, sample))


def _initial_batcher_thread(
    sample_queue: torch.multiprocessing.Queue,
    batch_queue: queue.PriorityQueue,
    batch_size: int,
) -> None:
    while True:
        samples = []
        priorities = []
        for _ in range(batch_size):
            priority, sample = sample_queue.get()
            samples.append(sample)
            priorities.append(priority)
        batch = collate(samples)
        priority = min(priorities)
        batch_queue.put((priority, batch))
