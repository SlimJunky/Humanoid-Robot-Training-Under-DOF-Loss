
# Inspect amass reads the.npz files and tries to make a guess based on rough motion type data and filename keywords to create a JSON manifest for clips I require.
# Those clips will then be used as the required motion capture data and a skeleton will be mapped onto unitree G1.


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Key words to inspect AMASS .npz files.
LABEL_KEYWORDS = {
    "walk": ["walk", "walking"],
    "run": ["run", "running", "jog", "jogging"],
    "jump": ["jump", "jumping", "hop", "hopping"],
    "wave": ["wave", "waving", "reach", "reaching", "stretch", "stretching", "hand"],
}


def guess_label(name: str) -> str | None:
    n = name.lower()
    for label, keywords in LABEL_KEYWORDS.items():
        if any(k in n for k in keywords):
            return label
    return None


def to_shape(value: Any) -> list[int] | None:
    try:
        return list(np.asarray(value).shape)
    except Exception:
        return None

# Inspecting npz file of AMASS SMPL here for specific conditions trying to build a manifest of likely clips for walking, running, jumping and reach / wave.

def inspect_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)

    keys = list(data.keys())
    row: dict[str, Any] = {
        "file": str(path),
        "relative_file": str(path.name),
        "clip_name": path.stem,
        "label_guess": guess_label(path.stem),
        "keys": keys,
    }

    for key in ["poses", "trans", "betas", "dmpls"]:
        if key in data:
            row[f"{key}_shape"] = to_shape(data[key])

    if "mocap_framerate" in data:
        try:
            row["fps"] = float(np.asarray(data["mocap_framerate"]).item())
        except Exception:
            row["fps"] = None
    else:
        row["fps"] = None

    if "gender" in data:
        try:
            gender_val = data["gender"]
            if isinstance(gender_val, np.ndarray) and gender_val.shape == ():
                gender_val = gender_val.item()
            row["gender"] = str(gender_val)
        except Exception:
            row["gender"] = None

    # frame count
    if "trans" in data:
        row["num_frames"] = int(np.asarray(data["trans"]).shape[0])
    elif "poses" in data:
        row["num_frames"] = int(np.asarray(data["poses"]).shape[0])
    else:
        row["num_frames"] = None

    # rough duration
    if row["num_frames"] is not None and row["fps"]:
        row["duration_s"] = row["num_frames"] / row["fps"]
    else:
        row["duration_s"] = None

    # AMASS pose split hints
    if "poses" in data:
        pose_shape = np.asarray(data["poses"]).shape
        if len(pose_shape) == 2 and pose_shape[1] >= 3:
            row["root_orient_dims"] = 3
            row["body_pose_dims"] = int(pose_shape[1] - 3)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root folder containing AMASS/CMU .npz files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/manifests/cmu_manifest.json",
        help="Where to write the manifest JSON",
    )
    parser.add_argument(
        "--include_all",
        action="store_true",
        help="If set, keep clips even when no label_guess is found",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for npz_path in sorted(input_dir.rglob("*.npz")):
        if npz_path.name.lower() == "shape.npz":
            continue

        try:
            row = inspect_npz(npz_path)
            if args.include_all or row["label_guess"] is not None:
                rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "file": str(npz_path),
                    "clip_name": npz_path.stem,
                    "error": str(exc),
                }
            )

    summary = {
        "input_dir": str(input_dir),
        "num_entries": len(rows),
        "entries": rows,
    }

    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote manifest to: {output}")


if __name__ == "__main__":
    main()