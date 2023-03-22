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
import multiprocessing
import pickle
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import psutil
from pympler.asizeof import asizeof
from tqdm import tqdm

from openfold.data.mmcif import (
    compress_mmcif_dict_atoms,
    load_mmcif_gz_file,
    parse_mmcif_string,
)

NUM_PHYSICAL_CPU_CORES = psutil.cpu_count(logical=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_mmcif_dirpath",
        type=Path,
        required=True,
        help="""Path to 'pdb_mmcif' data directory.
        Download it via `scripts/download_pdb_mmcif.sh`.""",
    )
    parser.add_argument(
        "--output_dirpath",
        type=Path,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--pdb_clusters_by_entity_filepath",
        type=Path,
        required=True,
        help="""Path to a cluster file (e.g. PDB40
        https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt
        ).""",
    )
    parser.add_argument(
        "--pdb_obsolete_filepath",
        type=Path,
        required=True,
        help="""Path to `obsolete.dat` file.
        Download it via `scripts/download_pdb_mmcif.sh`.""",
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


def create_mmcif_dict(
    mmcif_gz_filepath: Path,
    compress_atoms: bool,
) -> Tuple[Optional[dict], dict]:
    """Creates `mmcif_dict` from a `.cif.gz` file.

    Returns tuple: (`mmcif_dict`, `preprocessing_log`).

    `mmcif_dict` will be `None` if it cannot be created.

    `preprocessing_log` contains maximum information always.

    """
    pdb_id = mmcif_gz_filepath.name.split(".")[0]
    subdirname = mmcif_gz_filepath.parent.name

    start_time = time.perf_counter()
    mmcif_string = load_mmcif_gz_file(mmcif_gz_filepath)
    end_time = time.perf_counter()
    loading_time = end_time - start_time

    preprocessing_log = {
        "pdb_id": pdb_id,
        "subdirname": subdirname,
        "mmcif_gz_path": str(
            mmcif_gz_filepath.relative_to(mmcif_gz_filepath.parents[2])
        ),
        "mmcif_gz_size": mmcif_gz_filepath.stat().st_size,
        "loading_time": f"{loading_time:.6f}",
        "mmcif_string_len": len(mmcif_string),
    }

    try:
        start_time = time.perf_counter()
        mmcif_dict = parse_mmcif_string(mmcif_string)
        end_time = time.perf_counter()
        parsing_time = end_time - start_time
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            print("exit(1) due to KeyboardInterrupt")
            exit(1)
        else:
            preprocessing_log["mmcif_dict_size"] = 0
            preprocessing_log["parsing_time"] = float("nan")
            preprocessing_log["error"] = repr(e)
            return None, preprocessing_log

    if compress_atoms:
        mmcif_dict["atoms"] = compress_mmcif_dict_atoms(mmcif_dict["atoms"])

    preprocessing_log["mmcif_dict_size"] = asizeof(mmcif_dict)
    preprocessing_log["parsing_time"] = f"{parsing_time:.6f}"
    preprocessing_log["error"] = None

    return mmcif_dict, preprocessing_log


def create_mmcif_dicts(
    pdb_mmcif_raw_subdirpath: Path,
    pdb_mmcif_dicts_dirpath: Path,
    force: bool,
) -> pd.DataFrame:
    mmcif_dicts = {}
    preprocessing_logs = []

    subdirname = pdb_mmcif_raw_subdirpath.name
    mmcif_gz_filepaths = sorted(list(pdb_mmcif_raw_subdirpath.glob("*.cif.gz")))
    for mmcif_gz_filepath in mmcif_gz_filepaths:
        pdb_id = mmcif_gz_filepath.name.split(".")[0]

        mmcif_dict, preprocessing_log = create_mmcif_dict(
            mmcif_gz_filepath=mmcif_gz_filepath,
            compress_atoms=True,
        )

        if mmcif_dict is not None:
            assert pdb_id == mmcif_dict["pdb_id"]
            assert pdb_id not in mmcif_dicts
            mmcif_dicts[pdb_id] = mmcif_dict

        assert pdb_id == preprocessing_log["pdb_id"]
        assert subdirname == preprocessing_log["subdirname"]
        preprocessing_logs.append(preprocessing_log)

    output_fielpath = pdb_mmcif_dicts_dirpath / subdirname
    if not force:
        assert not output_fielpath.exists()
    with open(output_fielpath, "wb") as f:
        pickle.dump(mmcif_dicts, f)

    preprocessing_logs_df = pd.DataFrame(preprocessing_logs)
    return preprocessing_logs_df


def load_pdb_cluster_ids(clusters_by_entity_filepath: Path) -> Dict[str, int]:
    cluster_ids = {}
    with open(clusters_by_entity_filepath) as f:
        clusters = f.read().strip().split("\n")
    for cluster_id, cluster in enumerate(clusters):
        pdb_entity_ids = cluster.split()
        for pdb_entity_id in pdb_entity_ids:
            cluster_ids[pdb_entity_id.upper()] = cluster_id
    return cluster_ids


def create_mmcif_chains(
    pdb_mmcif_dicts_filepath: Path,
    pdb_cluster_ids: Dict[str, int],
) -> pd.DataFrame:
    with open(pdb_mmcif_dicts_filepath, "rb") as f:
        mmcif_dicts = pickle.load(f)

    mmcif_chains = []
    for pdb_id, mmcif_dict in mmcif_dicts.items():
        assert pdb_id == mmcif_dict["pdb_id"]

        author_chain_id_to_mmcif_chain_ids = defaultdict(list)
        author_chain_id_to_entity_ids = defaultdict(set)

        mmcif_to_author_mapping = mmcif_dict["mmcif_chain_id_to_author_chain_id"]
        entity_to_mmcifs_mapping = mmcif_dict["entity_id_to_mmcif_chain_ids"]

        for mmcif_chain_id, author_chain_id in mmcif_to_author_mapping.items():
            author_chain_id_to_mmcif_chain_ids[author_chain_id].append(mmcif_chain_id)
            for entity_id, mmcif_chain_ids in entity_to_mmcifs_mapping.items():
                if mmcif_chain_id in mmcif_chain_ids:
                    author_chain_id_to_entity_ids[author_chain_id].add(entity_id)

        for author_chain_id in mmcif_dict["author_chain_ids"]:
            pdb_chain_id = pdb_id + "_" + author_chain_id
            mmcif_chain_ids = author_chain_id_to_mmcif_chain_ids[author_chain_id]
            mmcif_chain_ids = ";".join(mmcif_chain_ids)

            chain_cluster_ids = []
            for entity_id in list(author_chain_id_to_entity_ids[author_chain_id]):
                pdb_entity_id = f"{pdb_id}_{entity_id}".upper()
                if pdb_entity_id in pdb_cluster_ids:
                    chain_cluster_ids.append(pdb_cluster_ids[pdb_entity_id])

            if len(chain_cluster_ids) == 1:
                pdb_cluster_id = chain_cluster_ids[0]
            elif len(chain_cluster_ids) == 0:
                pdb_cluster_id = -1
            else:
                # should never happen,
                # but when it does,
                # count and take the most common id
                pdb_cluster_id = Counter(chain_cluster_ids).most_common()[0][0]

            mmcif_chain = {
                "pdb_chain_id": pdb_chain_id,  # format: `{pdb_id}_{author_chain_id}`
                "pdb_id": pdb_id,
                "author_chain_id": author_chain_id,
                "mmcif_chain_ids": mmcif_chain_ids,
                "release_date": mmcif_dict["release_date"],
                "resolution": mmcif_dict["resolution"],
                "pdb_cluster_id": pdb_cluster_id,
                "sequence_length": len(mmcif_dict["sequences"][author_chain_id]),
                "sequence": mmcif_dict["sequences"][author_chain_id],
            }
            mmcif_chains.append(mmcif_chain)

    mmcif_chains_df = pd.DataFrame(mmcif_chains)
    return mmcif_chains_df


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


def preprocess_pdb_mmcif(
    pdb_mmcif_dirpath: Path,
    output_dirpath: Path,
    pdb_clusters_by_entity_filepath: Path,
    pdb_obsolete_filepath: Path,
    num_parallel_processes: int,
    force: bool,
) -> None:
    print("preprocess_pdb_mmcif has started...")

    print(f"pdb_mmcif_dirpath={repr(pdb_mmcif_dirpath)}")
    print(f"output_dirpath={repr(output_dirpath)}")
    print(f"pdb_clusters_by_entity_filepath={repr(pdb_clusters_by_entity_filepath)}")
    print(f"pdb_obsolete_filepath={repr(pdb_obsolete_filepath)}")
    print(f"num_parallel_processes={num_parallel_processes}")
    print(f"force={force}")

    if not pdb_mmcif_dirpath.exists():
        raise FileNotFoundError(f"{repr(pdb_mmcif_dirpath)} does not exist!")
    if not pdb_clusters_by_entity_filepath.exists():
        raise FileNotFoundError(
            f"{repr(pdb_clusters_by_entity_filepath)} does not exist!"
        )
    if not pdb_obsolete_filepath.exists():
        raise FileNotFoundError(f"{repr(pdb_obsolete_filepath)} does not exist!")

    pdb_mmcif_raw_dirpath = pdb_mmcif_dirpath / "raw"
    if not pdb_mmcif_raw_dirpath.exists():
        raise FileNotFoundError(f"{repr(pdb_mmcif_raw_dirpath)} does not exist!")

    pdb_mmcif_raw_subdirpaths = sorted(list(pdb_mmcif_raw_dirpath.glob("*")))
    pdb_mmcif_raw_subdirpaths = [
        subdirpath for subdirpath in pdb_mmcif_raw_subdirpaths if subdirpath.is_dir()
    ]
    num_mmcif_gz_files = sum(
        len(list(subdirpath.glob("*.cif.gz")))
        for subdirpath in pdb_mmcif_raw_subdirpaths
    )
    print(
        f"Found {len(pdb_mmcif_raw_subdirpaths)} subdirectories"
        f" inside {repr(pdb_mmcif_raw_dirpath)}"
        f" containing {num_mmcif_gz_files} `.cif.gz` files in total."
    )

    output_dirpath.mkdir(exist_ok=force, parents=True)

    pdb_mmcif_dicts_dirpath = output_dirpath / "dicts"
    pdb_mmcif_dicts_dirpath.mkdir(exist_ok=force)
    print(f"mmcif dicts will be saved to {repr(pdb_mmcif_dicts_dirpath)}")

    print("Preprocessing (creating mmcif dicts)...")
    preprocessing_logs_dfs = apply_func_parallel(
        func=create_mmcif_dicts,
        args_list=[
            tuple([pdb_mmcif_raw_subdirpath, pdb_mmcif_dicts_dirpath, force])
            for pdb_mmcif_raw_subdirpath in pdb_mmcif_raw_subdirpaths
        ],
        num_parallel_processes=num_parallel_processes,
    )

    preprocessing_logs_df = pd.concat(preprocessing_logs_dfs)
    summary = (
        preprocessing_logs_df["error"].fillna("SUCCESS").value_counts(dropna=False)
    )
    header = pd.Series(index=["__num_mmcif_gz_files__"], data=[num_mmcif_gz_files])
    preprocessing_logs_filepath = output_dirpath / "dicts_preprocessing_logs.csv"
    if not force:
        assert not preprocessing_logs_filepath.exists()
    preprocessing_logs_df.to_csv(preprocessing_logs_filepath, index=False)
    print("preprocessing_logs_df.shape", preprocessing_logs_df.shape)
    print("Preprocessing summary:")
    print(pd.concat([header, summary]).to_string())
    print(
        f"Preprocessing logs saved to {repr(preprocessing_logs_filepath)} successfully!"
    )

    pdb_cluster_ids = load_pdb_cluster_ids(pdb_clusters_by_entity_filepath)
    print("Generating mmcif chains...")
    pdb_mmcif_dicts_filepaths = sorted(list(pdb_mmcif_dicts_dirpath.glob("*")))
    mmcif_chains_dfs = apply_func_parallel(
        func=create_mmcif_chains,
        args_list=[
            tuple([pdb_mmcif_dicts_filepath, pdb_cluster_ids])
            for pdb_mmcif_dicts_filepath in pdb_mmcif_dicts_filepaths
        ],
        num_parallel_processes=num_parallel_processes,
    )
    mmcif_chains_df = pd.concat(mmcif_chains_dfs)
    pdb_mmcif_chains_filepath = output_dirpath / "chains.csv"
    if not force:
        assert not pdb_mmcif_chains_filepath.exists()
    mmcif_chains_df.to_csv(pdb_mmcif_chains_filepath, index=False)
    print(f"mmcif chains saved to {repr(pdb_mmcif_chains_filepath)} successfully!")

    print("copying pdb obsolete file...")
    src_pdb_obsolete_filepath = pdb_obsolete_filepath
    dst_pdb_obsolete_filepath = output_dirpath / pdb_obsolete_filepath.name
    if not force:
        assert not dst_pdb_obsolete_filepath.exists()
    shutil.copyfile(src=src_pdb_obsolete_filepath, dst=dst_pdb_obsolete_filepath)
    print(
        f"pdb obsolete file copied to {repr(dst_pdb_obsolete_filepath)} successfully!"
    )

    print("preprocess_pdb_mmcif finished successfully!")


def main() -> None:
    args = parse_args()
    preprocess_pdb_mmcif(
        pdb_mmcif_dirpath=args.pdb_mmcif_dirpath,
        output_dirpath=args.output_dirpath,
        pdb_clusters_by_entity_filepath=args.pdb_clusters_by_entity_filepath,
        pdb_obsolete_filepath=args.pdb_obsolete_filepath,
        num_parallel_processes=args.num_parallel_processes,
        force=args.force,
    )


if __name__ == "__main__":
    main()
