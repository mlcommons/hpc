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

import argparse
import json
import multiprocessing
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import psutil
from tqdm import tqdm

from openfold.helpers import hash_string_into_number

NUM_PHYSICAL_CPU_CORES = psutil.cpu_count(logical=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open_protein_set_dirpath",
        type=Path,
        required=True,
        help="""Path to OpenProteinSet data directory.
        Download it via `scripts/download_open_protein_set.sh`.""",
    )
    parser.add_argument(
        "--output_dirpath",
        type=Path,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=10,
        help="Num shards in output database.",
    )
    parser.add_argument(
        "--num_parallel_processes",
        type=int,
        default=NUM_PHYSICAL_CPU_CORES,
        help="""Num parallel processes used during preprocessing.
        Default is equal to num physical CPU cores.""",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to override existing output files.",
    )
    args = parser.parse_args()
    return args


def load_pdb_chain_clusters(pdb_chain_clusters_filepath: Path) -> List[List[str]]:
    with open(pdb_chain_clusters_filepath) as f:
        pdb_chain_clusters = f.read().strip().split("\n")
    pdb_chain_clusters = [
        pdb_chain_cluster.split() for pdb_chain_cluster in pdb_chain_clusters
    ]
    return pdb_chain_clusters


def write_pdb_alignments_shard(
    shard_id: int,
    shard: List[Tuple[Path, List[str]]],
    output_pdb_alignments_dirpath: Path,
    force: bool,
) -> Dict[str, dict]:
    shard_index = {}
    shard_filename = f"{shard_id}.db"
    shard_filepath = output_pdb_alignments_dirpath / shard_filename
    if not force:
        assert not shard_filepath.exists()
    with open(shard_filepath, "wb") as f_out:
        start = 0
        for pdb_subdirpath_cluster_pair in shard:
            open_protein_set_pdb_subdirpath = pdb_subdirpath_cluster_pair[0]
            pdb_chain_cluster = pdb_subdirpath_cluster_pair[1]
            alignments_index = {
                "db": shard_filename,
                "files": [],
            }
            filepaths = sorted(list(open_protein_set_pdb_subdirpath.glob("*/*")))
            for filepath in filepaths:
                filename = filepath.name
                with open(filepath, "rb") as f_in:
                    filebytes = f_in.read()
                f_out.write(filebytes)
                size = len(filebytes)
                file_index = [filename, start, size]
                alignments_index["files"].append(file_index)
                start += size
            for pdb_chain_id in pdb_chain_cluster:
                assert pdb_chain_id not in shard_index
                shard_index[pdb_chain_id] = alignments_index
    return shard_index


def apply_func_parallel(
    func: Callable,
    args_list: List[tuple],
    num_parallel_processes: int,
) -> list:
    if not isinstance(args_list, list):
        raise TypeError(
            f"args_list is of type {type(args_list)}, but it should be of type {list}."
        )
    for args in args_list:
        if not isinstance(args, tuple):
            raise TypeError(
                f"args is of type {type(args)}, but it should be of type {tuple}."
            )

    if num_parallel_processes > 0:
        async_results = []
        pool = multiprocessing.Pool(num_parallel_processes)
        for args in args_list:
            ar = pool.apply_async(func, args)
            async_results.append(ar)
        results = [ar.get() for ar in tqdm(async_results)]
        pool.close()
        pool.join()

    else:
        results = []
        for args in tqdm(args_list):
            r = func(*args)
            results.append(r)

    return results


def preprocess_open_protein_set_pdb_alignments(
    open_protein_set_dirpath: Path,
    output_pdb_alignments_dirpath: Path,
    num_shards: int,
    num_parallel_processes: int,
    force: bool,
) -> None:
    print("preprocess_open_protein_set_pdb_alignments has started...")

    open_protein_set_pdb_dirpath = open_protein_set_dirpath / "pdb"
    if not open_protein_set_pdb_dirpath.exists():
        raise FileNotFoundError(f"{repr(open_protein_set_pdb_dirpath)} does not exist!")

    open_protein_set_pdb_subdirpaths = sorted(
        list(open_protein_set_pdb_dirpath.glob("*"))
    )
    print(
        f"Found {len(open_protein_set_pdb_subdirpaths)} subdirectories"
        f" inside {repr(open_protein_set_pdb_dirpath)}."
    )

    pdb_chain_clusters_filepath = open_protein_set_dirpath / "duplicate_pdb_chains.txt"
    pdb_chain_clusters = load_pdb_chain_clusters(
        pdb_chain_clusters_filepath=pdb_chain_clusters_filepath,
    )
    pdb_chain_id_to_cluster_index = {}
    for cluster_index, pdb_chain_cluster in enumerate(pdb_chain_clusters):
        for pdb_chain_id in pdb_chain_cluster:
            pdb_chain_id_to_cluster_index[pdb_chain_id] = cluster_index

    shards = {shard_id: [] for shard_id in range(num_shards)}
    assigned_clusters_indexes = set()
    print("sharding pdb alignments...")
    for open_protein_set_pdb_subdirpath in tqdm(open_protein_set_pdb_subdirpaths):
        pdb_chain_id = open_protein_set_pdb_subdirpath.name
        shard_id = hash_string_into_number(pdb_chain_id) % num_shards
        assert pdb_chain_id in pdb_chain_id_to_cluster_index
        cluster_index = pdb_chain_id_to_cluster_index[pdb_chain_id]
        assert cluster_index not in assigned_clusters_indexes
        pdb_chain_cluster = pdb_chain_clusters[cluster_index]
        pdb_subdirpath_cluster_pair = (
            open_protein_set_pdb_subdirpath,
            pdb_chain_cluster,
        )
        shards[shard_id].append(pdb_subdirpath_cluster_pair)
        assigned_clusters_indexes.add(cluster_index)
    print("pdb alignments shards:")
    for shard_id, shard in shards.items():
        print(f"shard_id={shard_id} len(shard)={len(shard)}")

    output_pdb_alignments_dirpath.mkdir(exist_ok=force)
    print(
        "output pdb alignments shards will be saved to "
        f"{repr(output_pdb_alignments_dirpath)}"
    )

    print("writing pdb alignments shards...")
    shard_index_list = apply_func_parallel(
        func=write_pdb_alignments_shard,
        args_list=[
            tuple([shard_id, shard, output_pdb_alignments_dirpath, force])
            for shard_id, shard in shards.items()
        ],
        num_parallel_processes=num_parallel_processes,
    )
    pdb_alignments_super_index = {}
    for shard_index in shard_index_list:
        for pdb_chain_id, alignments_index in shard_index.items():
            assert pdb_chain_id not in pdb_alignments_super_index
            pdb_alignments_super_index[pdb_chain_id] = alignments_index
    print(f"len(pdb_alignments_super_index)={len(pdb_alignments_super_index)}")

    pdb_alignments_super_index_filepath = output_pdb_alignments_dirpath / "super.index"
    if not force:
        assert not pdb_alignments_super_index_filepath.exists()
    with open(pdb_alignments_super_index_filepath, "w") as f:
        json.dump(pdb_alignments_super_index, f)
    print(
        "pdb alignments super index saved to "
        f"{repr(pdb_alignments_super_index_filepath)} "
        "successfully!"
    )

    print("preprocess_open_protein_set_pdb_alignments finished successfully!")


def preprocess_open_protein_set(
    open_protein_set_dirpath: Path,
    output_dirpath: Path,
    num_shards: int,
    num_parallel_processes: int,
    force: bool,
) -> None:
    print("preprocess_open_protein_set has started...")

    print(f"open_protein_set_dirpath={repr(open_protein_set_dirpath)}")
    print(f"output_dirpath={repr(output_dirpath)}")
    print(f"num_shards={num_shards}")
    print(f"num_parallel_processes={num_parallel_processes}")
    print(f"force={force}")

    output_dirpath.mkdir(exist_ok=force, parents=True)

    preprocess_open_protein_set_pdb_alignments(
        open_protein_set_dirpath=open_protein_set_dirpath,
        output_pdb_alignments_dirpath=(output_dirpath / "pdb_alignments"),
        num_shards=num_shards,
        num_parallel_processes=num_parallel_processes,
        force=force,
    )

    print("preprocess_open_protein_set finished successfully!")


def main() -> None:
    args = parse_args()
    preprocess_open_protein_set(
        open_protein_set_dirpath=args.open_protein_set_dirpath,
        output_dirpath=args.output_dirpath,
        num_shards=args.num_shards,
        num_parallel_processes=args.num_parallel_processes,
        force=args.force,
    )


if __name__ == "__main__":
    main()
