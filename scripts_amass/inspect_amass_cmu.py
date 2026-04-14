
# Inspect amass cmu mocap SMPL+H download files. Reads the.npz files and tries to make a guess based on rough motion type data and filename keywords 
# To create a JSON and CSV with information about the different poses that i can match up with the dataset website.
# Those clips will then be used as the required motion capture data and a skeleton will be mapped onto Unitree G1. 


from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


LABEL_KEYWORDS = {
    "walk": ["walk", "walking"],
    "run": ["run", "running", "jog", "jogging", "sprint"],
    "jump": ["jump", "jumping", "hop", "hopping", "leap"],
    "wave": ["wave", "waving", "reach", "reaching", "stretch", "stretching", "hand"],
}


def guess_label(text: str) -> str | None:
    t = text.lower()
    for label, keywords in LABEL_KEYWORDS.items():
        if any(keyword in t for keyword in keywords):
            return label
    return None


def to_shape(value: Any) -> list[int] | None:
    try:
        return list(np.asarray(value).shape)
    except Exception:
        return None


def safe_scalar(value: Any) -> Any:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        return value
    except Exception:
        return value


def inspect_npz(path: Path, input_dir: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    rel_path = path.relative_to(input_dir)

    keys = list(data.keys())

    row: dict[str, Any] = {
        "file": str(path),
        "relative_file": rel_path.as_posix(),
        "clip_name": rel_path.stem,
        "file_stem": path.stem,
        "parent_folder": path.parent.name,
        "subject_folder": rel_path.parts[0] if len(rel_path.parts) > 0 else None,
        # Guess from full relative path, not just "poses"
        "label_guess": guess_label(rel_path.as_posix()),
        "keys": keys,
    }

    for key in ["poses", "trans", "betas", "dmpls"]:
        if key in data:
            row[f"{key}_shape"] = to_shape(data[key])

    if "mocap_framerate" in data:
        try:
            row["fps"] = float(safe_scalar(data["mocap_framerate"]))
        except Exception:
            row["fps"] = None
    else:
        row["fps"] = None

    if "gender" in data:
        try:
            row["gender"] = str(safe_scalar(data["gender"]))
        except Exception:
            row["gender"] = None
    else:
        row["gender"] = None

    if "trans" in data:
        row["num_frames"] = int(np.asarray(data["trans"]).shape[0])
    elif "poses" in data:
        row["num_frames"] = int(np.asarray(data["poses"]).shape[0])
    else:
        row["num_frames"] = None

    if row["num_frames"] is not None and row["fps"]:
        row["duration_s"] = row["num_frames"] / row["fps"]
    else:
        row["duration_s"] = None

    if "poses" in data:
        pose_shape = np.asarray(data["poses"]).shape
        if len(pose_shape) == 2 and pose_shape[1] >= 3:
            row["root_orient_dims"] = 3
            row["body_pose_dims"] = int(pose_shape[1] - 3)

    return row


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject_folder",
        "parent_folder",
        "relative_file",
        "clip_name",
        "label_guess",
        "fps",
        "num_frames",
        "duration_s",
        "gender",
        "poses_shape",
        "trans_shape",
        "betas_shape",
        "dmpls_shape",
        "keys",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "subject_folder": row.get("subject_folder"),
                    "parent_folder": row.get("parent_folder"),
                    "relative_file": row.get("relative_file"),
                    "clip_name": row.get("clip_name"),
                    "label_guess": row.get("label_guess"),
                    "fps": row.get("fps"),
                    "num_frames": row.get("num_frames"),
                    "duration_s": row.get("duration_s"),
                    "gender": row.get("gender"),
                    "poses_shape": row.get("poses_shape"),
                    "trans_shape": row.get("trans_shape"),
                    "betas_shape": row.get("betas_shape"),
                    "dmpls_shape": row.get("dmpls_shape"),
                    "keys": ";".join(row.get("keys", [])) if isinstance(row.get("keys"), list) else row.get("keys"),
                    "error": row.get("error"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect AMASS/CMU .npz files and build a manifest."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root folder containing AMASS/CMU .npz files",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/manifests/cmu_manifest.json",
        help="Where to write the manifest JSON",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/manifests/cmu_manifest.csv",
        help="Where to write the manifest CSV",
    )
    parser.add_argument(
        "--include_all",
        action="store_true",
        help="If set, keep clips even when no label_guess is found",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for npz_path in sorted(input_dir.rglob("*.npz")):
        if npz_path.name.lower() == "shape.npz":
            continue

        try:
            row = inspect_npz(npz_path, input_dir)
            if args.include_all or row["label_guess"] is not None:
                rows.append(row)
        except Exception as exc:
            rel_path = npz_path.relative_to(input_dir)
            rows.append(
                {
                    "file": str(npz_path),
                    "relative_file": rel_path.as_posix(),
                    "clip_name": rel_path.stem,
                    "parent_folder": npz_path.parent.name,
                    "subject_folder": rel_path.parts[0] if len(rel_path.parts) > 0 else None,
                    "error": str(exc),
                }
            )

    label_counts = Counter(row.get("label_guess") or "unlabeled" for row in rows)

    summary = {
        "input_dir": str(input_dir),
        "num_entries": len(rows),
        "label_counts": dict(label_counts),
        "entries": rows,
    }

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(rows, output_csv)

    print(f"Wrote JSON manifest to: {output_json}")
    print(f"Wrote CSV manifest to:  {output_csv}")
    print("Label counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()