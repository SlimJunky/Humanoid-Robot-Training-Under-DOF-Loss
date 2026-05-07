
from __future__ import annotations

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

'''Based on G1 dump data and retarget ready json run this to created first_pass .npz and first_pass .json to load retarget-ready walk file
load the dumped G1 joint order and soft limit and start every frame from G1 default joint pose. Clips mapped joints to expected minimal G1 soft limits
Saves the resulting G1 joint targets. Generates a first pass .npz file where one row per frame and one column per G1 joint.
Also generates a JSON file that can be sanity checked for num_frames, num_joints rows and columns and mapped_g1_joints and clip_count.
Ultimately prepares target file before creating a replay playback ready for demonstrations that need to be provided for Behavioral Cloning'''

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


# SMPL+H body joint order corresponding to pose_body[:, 0:63].
SMPLH_BODY_JOINTS = [
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


# Mapping spec trying to match up to the minimal Unitree G1 asset. 
DEFAULT_MAPPING_SPEC: dict[str, dict[str, Any]] = {
    # torso / spine
    "torso_joint": {
        "source_joint": "spine3",
        "source_axis": 0, # Which axis
        "scale": 0.30, # How large
        
        "sign": 1.0, # Whether to flip mapping
        "offset_mode": "add_to_default",
        "notes": "First-pass torso bend from spine3 axis 0.",
    },
    # hips
    "left_hip_pitch_joint": {
        "source_joint": "left_hip",
        "source_axis": 0,
        "scale": 0.50,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: hip axis 0 -> hip pitch.",
    },
    "right_hip_pitch_joint": {
        "source_joint": "right_hip",
        "source_axis": 0,
        "scale": 0.50,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: hip axis 0 -> hip pitch.",
    },
    "left_hip_roll_joint": {
        "source_joint": "left_hip",
        "source_axis": 1,
        "scale": 0.50,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: hip axis 1 -> hip roll.",
    },
    "right_hip_roll_joint": {
        "source_joint": "right_hip",
        "source_axis": 1,
        "scale": 0.50,
        "sign": -1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: hip axis 1 -> hip roll.",
    },
    "left_hip_yaw_joint": {
        "source_joint": "left_hip",
        "source_axis": 2,
        "scale": 0.50,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: hip axis 2 -> hip yaw.",
    },
    "right_hip_yaw_joint": {
        "source_joint": "right_hip",
        "source_axis": 2,
        "scale": 0.50,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: hip axis 2 -> hip yaw.",
    },
    # knees
    "left_knee_joint": {
        "source_joint": "left_knee",
        "source_axis": 0,
        "scale": 0.60,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: knee axis 0 -> knee flexion.",
    },
    "right_knee_joint": {
        "source_joint": "right_knee",
        "source_axis": 0,
        "scale": 0.60,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: knee axis 0 -> knee flexion.",
    },
    # ankles
    "left_ankle_pitch_joint": {
        "source_joint": "left_ankle",
        "source_axis": 0,
        "scale": 0.30,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: ankle axis 0 -> ankle pitch.",
    },
    "right_ankle_pitch_joint": {
        "source_joint": "right_ankle",
        "source_axis": 0,
        "scale": 0.30,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: ankle axis 0 -> ankle pitch.",
    },
    "left_ankle_roll_joint": {
        "source_joint": "left_ankle",
        "source_axis": 1,
        "scale": 0.25,
        "sign": 1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: ankle axis 1 -> ankle roll.",
    },
    "right_ankle_roll_joint": {
        "source_joint": "right_ankle",
        "source_axis": 1,
        "scale": 0.25,
        "sign": -1.0,
        "offset_mode": "add_to_default",
        "notes": "First-pass guess: ankle axis 1 -> ankle roll.",
    },
}

def find_project_root() -> Path:
    '''Find the repository root from this script location.'''
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "source").exists():
            return parent
        if (parent / ".git").exists():
            return parent

    # Expected fallback if script is in /scripts_amass/script_name.py
    return current.parents[1]


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    '''Resolve absolute paths directly, and relative paths from the project root'''
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return project_root / path


def project_relative(path: Path, project_root: Path) -> str:
    '''Store paths relative to the project root where possible'''
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

@dataclass
class MappingRecord:
    g1_joint: str
    g1_index: int
    default_pos: float
    lower_soft_limit: float
    upper_soft_limit: float
    source_joint: str
    source_joint_index: int
    source_axis: int
    scale: float
    sign: float
    offset_mode: str
    notes: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Same as other scripts sanitize values. NumPy arrays into JSON-safe values.
def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def build_smplh_index() -> dict[str, int]:
    return {name: i for i, name in enumerate(SMPLH_BODY_JOINTS)}


# Reads G1 dump JSON and extracts appropriate arrays for default positions, soft lower limitis and soft upper limits. 
def get_joint_table_arrays(joint_table: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    default_joint_pos = np.array([row["default_pos"] for row in joint_table], dtype=np.float64)
    soft_lo = np.array([row["soft_limit_lower"] for row in joint_table], dtype=np.float64)
    soft_hi = np.array([row["soft_limit_upper"] for row in joint_table], dtype=np.float64)
    return default_joint_pos, soft_lo, soft_hi


def body_joint_block(pose_body: np.ndarray, joint_name: str, smpl_index: dict[str, int]) -> np.ndarray:
    if joint_name not in smpl_index:
        raise KeyError(f"Unknown SMPLH body joint: {joint_name}")
    j = smpl_index[joint_name]
    start = j * 3
    end = start + 3
    return pose_body[:, start:end]


def load_mapping_spec(path: Path | None, project_root: Path) -> dict[str, dict[str, Any]]:
    if path is None:
        return DEFAULT_MAPPING_SPEC

    path = resolve_project_path(path, project_root)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Use the npz re-target file and g1 Dump to get a first pass re-target. Slow walk should have [256, 63] frame to pose body.
def map_motion_to_g1(
    retarget_npz_path: Path,
    g1_dump_json_path: Path,
    out_dir: Path,
    mapping_spec_path: Path | None = None,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    project_root = project_root or find_project_root()

    retarget_npz_path = resolve_project_path(retarget_npz_path, project_root)
    g1_dump_json_path = resolve_project_path(g1_dump_json_path, project_root)
    out_dir = resolve_project_path(out_dir, project_root)

    if not retarget_npz_path.exists():
        raise FileNotFoundError(f"Retarget-ready motion not found: {retarget_npz_path}")
    if not g1_dump_json_path.exists():
        raise FileNotFoundError(f"G1 dump JSON not found: {g1_dump_json_path}")

    mapping_spec = load_mapping_spec(mapping_spec_path, project_root)

    with np.load(retarget_npz_path, allow_pickle=True) as data:
        if "pose_body" not in data:
            raise KeyError(f"'pose_body' not found in: {retarget_npz_path}")
        pose_body = np.asarray(data["pose_body"], dtype=np.float64)
        time_s = np.asarray(data["time_s"], dtype=np.float64) if "time_s" in data else None
        source_fps = float(np.asarray(data["target_fps"]).item()) if "target_fps" in data else None
        category = str(np.asarray(data["category"]).item()) if "category" in data else retarget_npz_path.parent.name
        source_motion_name = str(np.asarray(data["source_file_name"]).item()) if "source_file_name" in data else retarget_npz_path.name

    if pose_body.ndim != 2 or pose_body.shape[1] != 63:
        raise ValueError(f"Expected pose_body shape [T, 63], got: {pose_body.shape}")

    with g1_dump_json_path.open("r", encoding="utf-8") as f:
        g1_dump = json.load(f)

    joint_names = g1_dump["joint_names"]
    joint_table = g1_dump["joint_table"]

    if len(joint_names) != len(joint_table):
        raise ValueError("joint_names and joint_table length mismatch in G1 dump JSON.")

    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    default_joint_pos, soft_lo, soft_hi = get_joint_table_arrays(joint_table)

    # Make unmapped/default joints safe once before creating per-frame targets to not get weird clipping.
    default_joint_pos_safe = np.clip(default_joint_pos, soft_lo, soft_hi)

    num_frames = pose_body.shape[0]
    num_joints = len(joint_names)

    joint_targets = np.tile(default_joint_pos_safe[None, :], (num_frames, 1)) # Use the joint_pos_safe
    smpl_index = build_smplh_index()

    applied_records: list[MappingRecord] = []
    missing_targets: list[str] = []

    for g1_joint, spec in mapping_spec.items():
        if g1_joint not in name_to_idx:
            missing_targets.append(g1_joint)
            continue

        source_joint = spec["source_joint"]
        source_axis = int(spec["source_axis"])
        scale = float(spec.get("scale", 1.0))
        sign = float(spec.get("sign", 1.0))
        offset_mode = str(spec.get("offset_mode", "add_to_default"))
        notes = str(spec.get("notes", ""))

        if source_axis not in (0, 1, 2):
            raise ValueError(f"source_axis must be 0,1,2 for {g1_joint}, got {source_axis}")

        source_block = body_joint_block(pose_body, source_joint, smpl_index)
        source_channel = source_block[:, source_axis]

        g1_idx = name_to_idx[g1_joint]
        g1_default = default_joint_pos[g1_idx]

        if offset_mode == "add_to_default":
            joint_targets[:, g1_idx] = g1_default + sign * scale * source_channel
        elif offset_mode == "replace":
            joint_targets[:, g1_idx] = sign * scale * source_channel
        else:
            raise ValueError(f"Unsupported offset_mode '{offset_mode}' for {g1_joint}")

        applied_records.append(
            MappingRecord(
                g1_joint=g1_joint,
                g1_index=g1_idx,
                default_pos=float(g1_default),
                lower_soft_limit=float(soft_lo[g1_idx]),
                upper_soft_limit=float(soft_hi[g1_idx]),
                source_joint=source_joint,
                source_joint_index=smpl_index[source_joint],
                source_axis=source_axis,
                scale=scale,
                sign=sign,
                offset_mode=offset_mode,
                notes=notes,
            )
        )

    #Clipping check
    joint_targets_before_clip = joint_targets.copy()
    joint_targets = np.clip(joint_targets, soft_lo[None, :], soft_hi[None, :])

    clip_count = int(np.count_nonzero(np.abs(joint_targets - joint_targets_before_clip) > 1e-12))

     # Count how many values were clipped for each joint good for debugging in JSON and improving fine tuning of mapping making sure its correct
     # Expecting clipping for the fingers but this can be ignored and identified in JSON.
    per_joint_clip_counts = {
        joint_names[j]: int(
            np.count_nonzero(np.abs(joint_targets[:, j] - joint_targets_before_clip[:, j]) > 1e-12)
        )
        for j in range(num_joints)
    }

    # Same idea, but only for the joints you actually mapped
    mapped_joint_set = {r.g1_joint for r in applied_records}
    mapped_per_joint_clip_counts = {
        name: count
        for name, count in per_joint_clip_counts.items()
        if name in mapped_joint_set
    }

    # Sorted version is easier to inspect in JSON
    mapped_per_joint_clip_counts_sorted = dict(
        sorted(mapped_per_joint_clip_counts.items(), key=lambda kv: kv[1], reverse=True)
    )

    final_out_dir = out_dir / category
    ensure_dir(final_out_dir)

    #Default output naming convention.
    out_stem = f"{retarget_npz_path.stem}_g1_first_pass"
    out_npz = final_out_dir / f"{out_stem}.npz"
    out_json = final_out_dir / f"{out_stem}.json"

    np.savez_compressed(
        out_npz,
        joint_targets=joint_targets,
        joint_names=np.array(joint_names, dtype=object),
        default_joint_pos=default_joint_pos_safe, # prevents weird noisy hand clipping if unused.
        raw_default_joint_pos=default_joint_pos,
        soft_joint_lower=soft_lo,
        soft_joint_upper=soft_hi,
        time_s=time_s if time_s is not None else np.arange(num_frames, dtype=np.float64),
        source_fps=np.array(source_fps if source_fps is not None else -1.0, dtype=np.float64),
        source_motion=np.array(project_relative(retarget_npz_path, project_root), dtype=object), #needs re-target ready npz
        g1_dump_json=np.array(project_relative(g1_dump_json_path, project_root), dtype=object), #needs g1 dump information
        category=np.array(category, dtype=object),
        mapped_g1_joint_names=np.array([r.g1_joint for r in applied_records], dtype=object),
    )

    summary = {
        "source": {
            "retarget_ready_npz": project_relative(retarget_npz_path, project_root),
            "g1_dump_json": project_relative(g1_dump_json_path, project_root),
            "source_motion_name": source_motion_name,
            "category": category,
        },
        "output": {
            "mapped_npz": project_relative(out_npz, project_root),
            "mapped_json": project_relative(out_json, project_root),
            "num_frames": num_frames,
            "num_g1_joints": num_joints,
        },
        "mapping_summary": {
            "total_g1_joints_in_dump": len(joint_names),
            "num_joints_mapped": len(applied_records),
            "mapped_g1_joints": [r.g1_joint for r in applied_records],
            "missing_g1_joints_from_dump": missing_targets,
            # These few will all give on top of generated JSON file plausible reasons for high clipping per frame and which joints are causing it
            "clip_count": clip_count, 
            "per_joint_clip_counts": per_joint_clip_counts,
            "mapped_per_joint_clip_counts": mapped_per_joint_clip_counts,
            "mapped_per_joint_clip_counts_sorted": mapped_per_joint_clip_counts_sorted,
            "notes": [
                "This is a first-pass offline mapping.",
                "Unmapped joints remain at G1 default joint positions.",
                "Mapped joints are clipped to G1 soft joint limits.",
                "Axis choices and signs are first guesses and should be tuned by playback.",
            ],
        },
        "applied_mapping": [asdict(r) for r in applied_records],
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(sanitize(summary), f, indent=2, ensure_ascii=False)

    return out_npz, out_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a first-pass offline AMASS slow-walk from minimal Unitree G1 mapping using retarget-ready motion and a G1 asset dump."
    )
    parser.add_argument(
        "retarget_npz",
        help="Path to the retarget-ready walk .npz file (e.g. ..._retarget_ready.npz).",
    )
    parser.add_argument(
        "--g1-dump-json",
        default="outputs/g1_asset_dump.json",
        help="Path to g1_asset_dump.json. Defaults to outputs/g1_asset_dump.json",
    )
    parser.add_argument(
        "--out-dir",
        default="data/mapped",
        help="Base output directory. Category subfolder will be created inside this folder.",
    )
    parser.add_argument(
        "--mapping-spec-json",
        default=None,
        help="Optional JSON file overriding the default first-pass mapping spec.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = find_project_root()

    retarget_npz = resolve_project_path(args.retarget_npz.strip(), project_root)
    g1_dump_json = resolve_project_path(args.g1_dump_json.strip(), project_root)
    out_dir = resolve_project_path(args.out_dir.strip(), project_root)
    mapping_spec_json = resolve_project_path(args.mapping_spec_json.strip(), project_root) if args.mapping_spec_json else None

    out_npz, out_json = map_motion_to_g1(
        retarget_npz_path=retarget_npz,
        g1_dump_json_path=g1_dump_json,
        out_dir=out_dir,
        mapping_spec_path=mapping_spec_json,
        project_root=project_root,
    )

    print(f"Mapped NPZ : {project_relative(out_npz, project_root)}")
    print(f"Mapped JSON: {project_relative(out_json, project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())