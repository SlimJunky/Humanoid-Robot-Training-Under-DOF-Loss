


from __future__ import annotations

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

''' Small script to edit existing HDF5 bc demonstrations data set and add expected ratio of train and valid masks for robo-mimic to use'''

import argparse
from pathlib import Path

import h5py
import numpy as np


def find_project_root() -> Path:

    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "source").exists():
            return parent
        if (parent / ".git").exists():
            return parent

    # Expected fallback if script is in /scripts_robomimic/script_name.py
    return current.parents[1]


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return project_root / path


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_mask_dataset(mask_group: h5py.Group, key: str, names: list[str], overwrite: bool) -> None:
    if key in mask_group:
        if not overwrite:
            raise ValueError(f"mask/{key} already exists. Use --overwrite to replace it.")
        del mask_group[key]

    data = np.array([name.encode("utf-8") for name in names], dtype="S") # look in hdf5 numpy array and assign mask dataset based on mask_group
    mask_group.create_dataset(key, data=data)


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Add robomimic train/valid mask splits to an existing HDF5 demo dataset"
    )
    parser.add_argument("dataset", help="path to the to robomimic-style HDF5 dataset")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=1, help="shuffle seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing train/valid masks")
    args = parser.parse_args()

    project_root = find_project_root()
    dataset_path = resolve_project_path(args.dataset.strip(), project_root)

    total_ratio = args.train_ratio + args.valid_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"train-ratio + valid-ratio must sum to 1.0: {total_ratio}")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with h5py.File(dataset_path, "r+") as f:
        if "data" not in f:
            raise KeyError("Expected top-level group data in dataset")

        data_group = f["data"]
        if not isinstance(data_group, h5py.Group):
            raise TypeError("Expected /data to be h5py Group")

        demo_names = sorted(list(data_group.keys()))
        if not demo_names:
            raise ValueError("No demos found under /data")

        rng = np.random.default_rng(args.seed)
        demo_names = list(rng.permutation(demo_names))

        n = len(demo_names)
        n_train = int(round(n * args.train_ratio))

        # Keep split valid after rounding
        n_train = min(max(n_train, 1), n - 1) if n > 1 else n
        train_names = demo_names[:n_train]
        valid_names = demo_names[n_train:]

        if len(train_names) == 0 or len(valid_names) == 0:
            raise ValueError(
                f"split produced empty subset: train={len(train_names)} valid={len(valid_names)}. "
                f"Adjust ratios or dataset size."
            )

        mask_group = f.require_group("mask")
        write_mask_dataset(mask_group, "train", train_names, args.overwrite)
        write_mask_dataset(mask_group, "valid", valid_names, args.overwrite)

    print(f"Updated dataset in place: {dataset_path}")
    print(f"train={len(train_names)} valid={len(valid_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())