from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


#Limits for data JSON default
DEFAULT_MAX_INLINE_ELEMENTS = 200
DEFAULT_PREVIEW_ROWS = 3
DEFAULT_PREVIEW_COLS = 12


# Look through selected motions .npz files and select specific attributes. Save into the data\manifests folder of the project for data inspection before converting


@dataclass
class FieldSummary:
    name: str
    kind: str
    dtype: str
    shape: list[int]
    ndim: int
    size: int
    bytes: int
    is_numeric: bool
    is_scalar: bool
    scalar_value: Any | None
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    std_value: float | None
    finite_fraction: float | None
    nan_count: int | None
    inf_count: int | None
    preview: Any
    full_values_included: bool


#UTC timestamp for data
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# Output directory
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Handle potential files in .npz that that I may not expect from AMASS github repo which suggested a set of key values expected
def maybe_decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


# Converts values into JSON format
def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, bytes):
        return maybe_decode_scalar(value)
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    return repr(value)


def build_array_preview(arr: np.ndarray, preview_rows: int, preview_cols: int) -> Any:
    if arr.ndim == 0:
        return sanitize_for_json(arr.item())

    if arr.ndim == 1:
        head = arr[:preview_cols]
        tail = arr[-preview_cols:] if arr.size > preview_cols else None
        return {
            "head": sanitize_for_json(head.tolist()),
            "tail": sanitize_for_json(tail.tolist()) if tail is not None and arr.size > preview_cols else None,
        }

    rows = min(arr.shape[0], preview_rows)
    if arr.ndim == 2:
        cols = min(arr.shape[1], preview_cols)
        return {
            "top_left": sanitize_for_json(arr[:rows, :cols].tolist()),
            "bottom_left": sanitize_for_json(arr[-rows:, :cols].tolist()) if arr.shape[0] > rows else None,
        }

    slice_spec = [slice(0, min(s, preview_rows if i == 0 else preview_cols)) for i, s in enumerate(arr.shape)]
    return {
        "leading_block": sanitize_for_json(arr[tuple(slice_spec)].tolist())
    }


# Computes any numeric values found to obtain other numeric key values. This helps identify key points in data manifest.
def numeric_stats(arr: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "min_value": None,
        "max_value": None,
        "mean_value": None,
        "std_value": None,
        "finite_fraction": None,
        "nan_count": None,
        "inf_count": None,
    }

    if arr.size == 0:
        return stats

    arr64 = np.asarray(arr, dtype=np.float64)
    finite_mask = np.isfinite(arr64)
    finite_count = int(finite_mask.sum())
    stats["finite_fraction"] = float(finite_count / arr64.size)
    stats["nan_count"] = int(np.isnan(arr64).sum())
    stats["inf_count"] = int(np.isinf(arr64).sum())

    if finite_count == 0:
        return stats

    finite_vals = arr64[finite_mask]
    stats["min_value"] = float(np.min(finite_vals))
    stats["max_value"] = float(np.max(finite_vals))
    stats["mean_value"] = float(np.mean(finite_vals))
    stats["std_value"] = float(np.std(finite_vals))
    return stats


# Takes one field from .npz and turns it into a FieldSummary for NumPy arrays.
def summarize_value(
    name: str,
    value: Any,
    *,
    max_inline_elements: int,
    preview_rows: int,
    preview_cols: int,
) -> FieldSummary:
    if isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.array(value, dtype=object)

    is_scalar = arr.ndim == 0
    scalar_value = None
    is_numeric = np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)

    if is_scalar:
        scalar_value = sanitize_for_json(maybe_decode_scalar(arr.item()))

    preview = None
    full_values_included = False

    if is_scalar:
        preview = scalar_value
    elif arr.size <= max_inline_elements:
        preview = sanitize_for_json(arr.tolist())
        full_values_included = True
    else:
        preview = build_array_preview(arr, preview_rows, preview_cols)

    stats = numeric_stats(arr) if is_numeric else {
        "min_value": None,
        "max_value": None,
        "mean_value": None,
        "std_value": None,
        "finite_fraction": None,
        "nan_count": None,
        "inf_count": None,
    }

    return FieldSummary(
        name=name,
        kind=type(value).__name__,
        dtype=str(arr.dtype),
        shape=list(arr.shape),
        ndim=int(arr.ndim),
        size=int(arr.size),
        bytes=int(arr.nbytes),
        is_numeric=bool(is_numeric),
        is_scalar=bool(is_scalar),
        scalar_value=scalar_value,
        min_value=stats["min_value"],
        max_value=stats["max_value"],
        mean_value=stats["mean_value"],
        std_value=stats["std_value"],
        finite_fraction=stats["finite_fraction"],
        nan_count=stats["nan_count"],
        inf_count=stats["inf_count"],
        preview=preview,
        full_values_included=full_values_included,
    )


# Looks for AMASS SMPL+H key values from selected .npz
def derive_amass_summary(data: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "gender": None,
        "mocap_framerate": None,
        "num_frames": None,
        "duration_seconds": None,
        "poses_shape": None,
        "trans_shape": None,
        "betas_shape": None,
        "dmpls_shape": None,
        "root_orient_shape": None,
        "pose_body_shape": None,
        "pose_hand_shape": None,
    }

    if "gender" in data:
        gender = data["gender"]
        if isinstance(gender, np.ndarray) and gender.ndim == 0:
            gender = gender.item()
        summary["gender"] = sanitize_for_json(maybe_decode_scalar(gender))

    if "mocap_framerate" in data:
        fr = data["mocap_framerate"]
        if isinstance(fr, np.ndarray) and fr.ndim == 0:
            fr = fr.item()
        try:
            summary["mocap_framerate"] = float(fr)
        except Exception:
            summary["mocap_framerate"] = sanitize_for_json(fr)

    if "poses" in data and isinstance(data["poses"], np.ndarray):
        poses = data["poses"]
        summary["poses_shape"] = list(poses.shape)
        if poses.ndim >= 2 and poses.shape[1] >= 3:
            summary["root_orient_shape"] = [poses.shape[0], 3]
        if poses.ndim >= 2 and poses.shape[1] >= 66:
            summary["pose_body_shape"] = [poses.shape[0], 63]
            summary["pose_hand_shape"] = [poses.shape[0], poses.shape[1] - 66]
        elif poses.ndim >= 2 and poses.shape[1] > 3:
            summary["pose_body_shape"] = [poses.shape[0], poses.shape[1] - 3]

    if "trans" in data and isinstance(data["trans"], np.ndarray):
        trans = data["trans"]
        summary["trans_shape"] = list(trans.shape)
        if trans.ndim >= 1:
            summary["num_frames"] = int(trans.shape[0])

    if summary["num_frames"] is None and "poses" in data and isinstance(data["poses"], np.ndarray):
        poses = data["poses"]
        if poses.ndim >= 1:
            summary["num_frames"] = int(poses.shape[0])

    if summary["num_frames"] is not None and isinstance(summary["mocap_framerate"], float) and summary["mocap_framerate"] > 0:
        summary["duration_seconds"] = float(summary["num_frames"] / summary["mocap_framerate"])

    if "betas" in data and isinstance(data["betas"], np.ndarray):
        summary["betas_shape"] = list(data["betas"].shape)

    if "dmpls" in data and isinstance(data["dmpls"], np.ndarray):
        summary["dmpls_shape"] = list(data["dmpls"].shape)

    return summary


# Main function and decides final output folder and destination. Loads .npz file, uses other functions to generate CSV and JSON
def make_manifest(
    npz_path: Path,
    out_dir: Path,
    *,
    max_inline_elements: int,
    preview_rows: int,
    preview_cols: int,
) -> tuple[Path, Path]:
    ensure_dir(out_dir)

    with np.load(npz_path, allow_pickle=True) as raw:
        data = {key: raw[key] for key in raw.files}

    # Summarize every field into Numpy Arrays
    field_summaries: list[FieldSummary] = []
    for key, value in data.items():
        field_summaries.append(
            summarize_value(
                key,
                value,
                max_inline_elements=max_inline_elements,
                preview_rows=preview_rows,
                preview_cols=preview_cols,
            )
        )

    manifest = {
        "manifest_version": 1,
        "generated_at_utc": utc_now_iso(),
        "source": {
            "file_name": npz_path.name,
            "file_stem": npz_path.stem,
            "absolute_path": str(npz_path.resolve()),
            "file_size_bytes": npz_path.stat().st_size,
        },
        "amass_summary": derive_amass_summary(data),
        "field_count": len(field_summaries),
        "fields": [
            {
                "name": f.name,
                "kind": f.kind,
                "dtype": f.dtype,
                "shape": f.shape,
                "ndim": f.ndim,
                "size": f.size,
                "bytes": f.bytes,
                "is_numeric": f.is_numeric,
                "is_scalar": f.is_scalar,
                "scalar_value": sanitize_for_json(f.scalar_value),
                "min_value": sanitize_for_json(f.min_value),
                "max_value": sanitize_for_json(f.max_value),
                "mean_value": sanitize_for_json(f.mean_value),
                "std_value": sanitize_for_json(f.std_value),
                "finite_fraction": sanitize_for_json(f.finite_fraction),
                "nan_count": sanitize_for_json(f.nan_count),
                "inf_count": sanitize_for_json(f.inf_count),
                "preview": sanitize_for_json(f.preview),
                "full_values_included": f.full_values_included,
            }
            for f in field_summaries
        ],
    }

    json_path = out_dir / f"{npz_path.stem}.json"
    csv_path = out_dir / f"{npz_path.stem}.csv"


    with json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "kind",
                "dtype",
                "shape",
                "ndim",
                "size",
                "bytes",
                "is_numeric",
                "is_scalar",
                "scalar_value",
                "min_value",
                "max_value",
                "mean_value",
                "std_value",
                "finite_fraction",
                "nan_count",
                "inf_count",
                "full_values_included",
                "preview",
            ],
        )
        writer.writeheader()
        for fsum in field_summaries:
            writer.writerow(
                {
                    "name": fsum.name,
                    "kind": fsum.kind,
                    "dtype": fsum.dtype,
                    "shape": json.dumps(fsum.shape),
                    "ndim": fsum.ndim,
                    "size": fsum.size,
                    "bytes": fsum.bytes,
                    "is_numeric": fsum.is_numeric,
                    "is_scalar": fsum.is_scalar,
                    "scalar_value": json.dumps(sanitize_for_json(fsum.scalar_value), ensure_ascii=False),
                    "min_value": json.dumps(sanitize_for_json(fsum.min_value), ensure_ascii=False),
                    "max_value": json.dumps(sanitize_for_json(fsum.max_value), ensure_ascii=False),
                    "mean_value": json.dumps(sanitize_for_json(fsum.mean_value), ensure_ascii=False),
                    "std_value": json.dumps(sanitize_for_json(fsum.std_value), ensure_ascii=False),
                    "finite_fraction": json.dumps(sanitize_for_json(fsum.finite_fraction), ensure_ascii=False),
                    "nan_count": json.dumps(sanitize_for_json(fsum.nan_count), ensure_ascii=False),
                    "inf_count": json.dumps(sanitize_for_json(fsum.inf_count), ensure_ascii=False),
                    "full_values_included": fsum.full_values_included,
                    "preview": json.dumps(sanitize_for_json(fsum.preview), ensure_ascii=False),
                }
            )

    return json_path, csv_path

# Script file arguments for future reference to change behaviour if I need to re-run this script. 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an AMASS .npz motion file and write JSON/CSV manifests."
    )
    parser.add_argument("npz_path", help="Path to the AMASS .npz file to inspect.")
    parser.add_argument(
        "--out-dir",
        default="data/manifests",
        help="Output folder for manifest files. Defaults to data/manifests",
    )
    parser.add_argument(
        "--max-inline-elements",
        type=int,
        default=DEFAULT_MAX_INLINE_ELEMENTS,
        help="Arrays with this many elements or fewer are written inline in full in the JSON preview.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=DEFAULT_PREVIEW_ROWS,
        help="Rows to include in array previews for larger arrays.",
    )
    parser.add_argument(
        "--preview-cols",
        type=int,
        default=DEFAULT_PREVIEW_COLS,
        help="Columns/elements to include in array previews for larger arrays.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    npz_path = Path(args.npz_path)
    out_dir = Path(args.out_dir)

    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")
    if npz_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz file, got: {npz_path.name}")

    json_path, csv_path = make_manifest(
        npz_path=npz_path,
        out_dir=out_dir,
        max_inline_elements=args.max_inline_elements,
        preview_rows=args.preview_rows,
        preview_cols=args.preview_cols,
    )

    print(f"Manifest JSON: {json_path}")
    print(f"Manifest CSV : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
