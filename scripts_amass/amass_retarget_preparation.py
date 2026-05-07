from __future__ import annotations

# Copyright (c) 2026, Mikolaj Wyrzykowski
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
'''prepare data into JSON format and correct .npz file for retargeting at 60 FPS'''

# Target FPS
DEFAULT_TARGET_FPS = 60.0

# AMASS / SMPL+H body layout used for the 63-dim body block (21 joints * 3 axis-angle values)
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

SMPLH_LEFT_HAND_JOINTS = [
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
]

SMPLH_RIGHT_HAND_JOINTS = [
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]

# Common G1-style joint names seen in Isaac Lab / Isaac Sim examples for the 37-DoF articulated version. Need to actually check G1 asset properly
G1_TEMPLATE_JOINTS_37 = [
    "torso_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_elbow_roll_joint", "right_elbow_roll_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_five_joint", "left_three_joint", "left_zero_joint",
    "right_five_joint", "right_three_joint", "right_zero_joint",
    "left_six_joint", "left_four_joint", "left_one_joint",
    "right_six_joint", "right_four_joint", "right_one_joint",
    "left_two_joint", "right_two_joint",
]

def find_project_root() -> Path:
    """Find the repository root from this script location"""
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "source").exists():
            return parent
        if (parent / ".git").exists():
            return parent

    # Expected fallback if script is in: repo/scripts_amass/script_name.py
    return current.parents[1]


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    """Resolve absolute paths directly, and relative paths from the project root"""
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return project_root / path


def project_relative(path: Path, project_root: Path) -> str:
    """Store paths relative to the project root where possible"""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()



@dataclass
class RetargetPrep:
    source_path: Path
    output_npz: Path
    output_json: Path
    data: dict[str, Any]
    metadata: dict[str, Any]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# Correct data format for JSON and altered .npz file
def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def maybe_scalar(arr_or_value: Any) -> Any:
    if isinstance(arr_or_value, np.ndarray) and arr_or_value.ndim == 0:
        return arr_or_value.item()
    if isinstance(arr_or_value, np.generic):
        return arr_or_value.item()
    return arr_or_value


def derive_category(npz_path: Path) -> str:
    return npz_path.parent.name

# Isaac sim & Isaac lab prefer quaternions just like other robotic simulation packages and environments.
def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    '''Convert Nx3 axis-angle to Nx4 quaternion (w, x, y, z).'''
    aa = np.asarray(axis_angle, dtype=np.float64)
    if aa.ndim != 2 or aa.shape[1] != 3:
        raise ValueError(f"Expected shape [N, 3] for axis-angle, got {aa.shape}")

    angles = np.linalg.norm(aa, axis=1, keepdims=True)
    quat = np.zeros((aa.shape[0], 4), dtype=np.float64)

    small = angles[:, 0] < 1e-12
    large = ~small

    quat[small, 0] = 1.0

    if np.any(large):
        half = 0.5 * angles[large]
        axes = aa[large] / angles[large]
        quat[large, 0] = np.cos(half[:, 0])
        quat[large, 1:] = axes * np.sin(half)

    return quat

#Extract yaw angle in radians from Nx4 quaternion (w, x, y, z).
def quat_to_yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)



def linear_resample_array(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.copy()
    if arr.shape[0] <= 1 or src_fps <= 0 or dst_fps <= 0 or np.isclose(src_fps, dst_fps):
        return arr.copy()

    src_frames = arr.shape[0]
    duration = (src_frames - 1) / src_fps
    dst_frames = int(round(duration * dst_fps)) + 1

    src_t = np.arange(src_frames, dtype=np.float64) / src_fps
    dst_t = np.arange(dst_frames, dtype=np.float64) / dst_fps
    flat = arr.reshape(src_frames, -1).astype(np.float64)
    out = np.empty((dst_frames, flat.shape[1]), dtype=np.float64)

    for i in range(flat.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, flat[:, i])

    return out.reshape((dst_frames,) + arr.shape[1:]).astype(arr.dtype, copy=False)


# Trim for FPS target
def trim_time_series(arr: np.ndarray, start_frame: int, end_frame_exclusive: int) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.ndim == 0:
        return arr
    if arr.shape[0] < end_frame_exclusive:
        raise ValueError(
            f"Cannot trim array of shape {arr.shape} with end frame {end_frame_exclusive}"
        )
    return arr[start_frame:end_frame_exclusive]

# Find joint pose names
def pose_block_names(num_body_dims: int, num_hand_dims: int) -> dict[str, Any]:
    body_joint_count = num_body_dims // 3
    hand_joint_count = num_hand_dims // 3
    return {
        "body_joint_names": SMPLH_BODY_JOINTS[:body_joint_count],
        "left_hand_joint_names": SMPLH_LEFT_HAND_JOINTS[: min(15, hand_joint_count // 2)],
        "right_hand_joint_names": SMPLH_RIGHT_HAND_JOINTS[: min(15, hand_joint_count - min(15, hand_joint_count // 2))],
    }


def prepare_retarget_ready(
    npz_path: Path,
    out_dir: Path,
    *,
    target_fps: float,
    trim_start_s: float,
    trim_end_s: float | None,
    mirror_category: bool,
    normalize_root_xy: bool,
    keep_vertical_root: bool,
    project_root: Path | None = None,
) -> RetargetPrep:
    project_root = project_root or find_project_root()
    npz_path = resolve_project_path(npz_path, project_root)
    out_dir = resolve_project_path(out_dir, project_root)
    with np.load(npz_path, allow_pickle=True) as raw:
        source = {key: raw[key] for key in raw.files}

    poses = np.asarray(source["poses"], dtype=np.float64)
    trans = np.asarray(source["trans"], dtype=np.float64)
    betas = np.asarray(source["betas"], dtype=np.float64)
    dmpls = np.asarray(source["dmpls"], dtype=np.float64) if "dmpls" in source else None
    gender = sanitize(maybe_scalar(source.get("gender")))
    mocap_framerate = float(maybe_scalar(source.get("mocap_framerate", 120.0)))

    if poses.ndim != 2 or poses.shape[1] < 66:
        raise ValueError(f"Expected poses shape [N, >=66], got {poses.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Expected trans shape [N, 3], got {trans.shape}")
    if poses.shape[0] != trans.shape[0]:
        raise ValueError(f"poses and trans frame counts do not match: {poses.shape[0]} vs {trans.shape[0]}")
    if dmpls is not None and dmpls.shape[0] != poses.shape[0]:
        raise ValueError(f"dmpls and poses frame counts do not match: {dmpls.shape[0]} vs {poses.shape[0]}")

    src_frames = poses.shape[0]
    # Short hand if else to not accidentally get one more or one less frame duration. Generated JSON Metadata should be correct this way.
    duration_s = (src_frames - 1) / mocap_framerate if src_frames > 1 else 0.0

    start_frame = max(0, int(round(trim_start_s * mocap_framerate)))
    end_frame = src_frames if trim_end_s is None else min(src_frames, int(round(trim_end_s * mocap_framerate)))
    if end_frame <= start_frame:
        raise ValueError(f"Invalid trim window: start={trim_start_s}s end={trim_end_s}s")

    poses = trim_time_series(poses, start_frame, end_frame)
    trans = trim_time_series(trans, start_frame, end_frame)
    if dmpls is not None:
        dmpls = trim_time_series(dmpls, start_frame, end_frame)

    root_orient = poses[:, :3]
    pose_body = poses[:, 3:66]
    pose_hand = poses[:, 66:]

    root_quat = axis_angle_to_quat(root_orient)
    root_yaw = quat_to_yaw(root_quat)

    '''Want to keep X and Y plane starting at zero with an option to flatten and do the same for Z however I wont be using any complex motions that involve vertical
    Jumping or anything that will move Z value too off starting position regardless.''' 
    if normalize_root_xy:
        trans = trans.copy()
        trans[:, 0] -= trans[0, 0]
        trans[:, 1] -= trans[0, 1]
        if not keep_vertical_root:
            trans[:, 2] -= trans[0, 2]

    # If target FPS differs then all time-series in data are resampled otherwise keep original FPS.
    if target_fps > 0 and not np.isclose(target_fps, mocap_framerate):
        root_orient = linear_resample_array(root_orient, mocap_framerate, target_fps)
        pose_body = linear_resample_array(pose_body, mocap_framerate, target_fps)
        pose_hand = linear_resample_array(pose_hand, mocap_framerate, target_fps)
        trans = linear_resample_array(trans, mocap_framerate, target_fps)
        root_quat = linear_resample_array(root_quat, mocap_framerate, target_fps)
        root_yaw = linear_resample_array(root_yaw[:, None], mocap_framerate, target_fps)[:, 0]
        if dmpls is not None:
            dmpls = linear_resample_array(dmpls, mocap_framerate, target_fps)
        out_fps = float(target_fps)
    else:
        out_fps = mocap_framerate

    num_frames = int(root_orient.shape[0])
    time_s = np.arange(num_frames, dtype=np.float64) / out_fps

    # Builds joint-name meta data for body and hands. Likely wont use hands.
    body_names = pose_block_names(pose_body.shape[1], pose_hand.shape[1])

    mapping_template = {
        "notes": [
            "This is a TEMPLATE only. Final joint mapping must match exact Unitree G1 asset and DOF order and be done manually after setup.",
            "AMASS root/body/hand blocks are in SMPL+H axis-angle format.",
            "Do not send these values directly to G1 joints without offsets, axis conversion, limits, and pose alignment.",
        ],
        "g1_template_joint_names_37": G1_TEMPLATE_JOINTS_37,
        "suggested_body_mapping": {
            "torso_joint": "spine3_or_chest_fused",
            "left_hip_pitch_joint": "left_hip",
            "right_hip_pitch_joint": "right_hip",
            "left_knee_joint": "left_knee",
            "right_knee_joint": "right_knee",
            "left_ankle_pitch_joint": "left_ankle",
            "right_ankle_pitch_joint": "right_ankle",
            "left_shoulder_pitch_joint": "left_shoulder",
            "right_shoulder_pitch_joint": "right_shoulder",
            "left_elbow_pitch_joint": "left_elbow",
            "right_elbow_pitch_joint": "right_elbow",
        },
    }

    category = derive_category(npz_path)
    final_out_dir = out_dir / category if mirror_category else out_dir
    ensure_dir(final_out_dir)

    out_npz = final_out_dir / f"{npz_path.stem}_retarget_ready.npz"
    out_json = final_out_dir / f"{npz_path.stem}_retarget_ready.json"

    packaged = {
        "source_file_stem": np.array(npz_path.stem),
        "source_file_name": np.array(npz_path.name),
        "category": np.array(category),
        "gender": np.array(gender),
        "source_fps": np.array(mocap_framerate, dtype=np.float64),
        "target_fps": np.array(out_fps, dtype=np.float64),
        "time_s": time_s,
        "root_pos": trans.astype(np.float64),
        "root_orient_axis_angle": root_orient.astype(np.float64),
        "root_quat_wxyz": root_quat.astype(np.float64),
        "root_yaw": root_yaw.astype(np.float64),
        "pose_body": pose_body.astype(np.float64),
        "pose_hand": pose_hand.astype(np.float64),
        "betas": betas.astype(np.float64),
        "body_joint_names": np.array(body_names["body_joint_names"], dtype=object),
        "left_hand_joint_names": np.array(body_names["left_hand_joint_names"], dtype=object),
        "right_hand_joint_names": np.array(body_names["right_hand_joint_names"], dtype=object),
    }
    if dmpls is not None:
        packaged["dmpls"] = dmpls.astype(np.float64)

    # once out npz is ready then this function defined previously will save it into more Isaac Lab friendly ready format with minimal changes.
    np.savez_compressed(out_npz, **packaged)

    metadata = {
        "source": {
            "project_path": project_relative(npz_path, project_root),
            "file_name": npz_path.name,
            "category": category,
        },
        "trim": {
            "trim_start_s": trim_start_s,
            "trim_end_s": trim_end_s,
            "source_num_frames": src_frames,
            "source_duration_s": duration_s,
        },
        "output": {
            "npz_path": project_relative(out_npz, project_root),
            "json_path": project_relative(out_json, project_root),
            "target_fps": out_fps,
            "num_frames": num_frames,
            "duration_s": float((num_frames - 1) / out_fps) if num_frames > 1 else 0.0,
        },
        "amass_layout": {
            "poses_shape_after_trim_before_resample": [int(end_frame - start_frame), int(source["poses"].shape[1])],
            "root_orient_shape": list(root_orient.shape),
            "pose_body_shape": list(pose_body.shape),
            "pose_hand_shape": list(pose_hand.shape),
            "root_pos_shape": list(trans.shape),
            "betas_shape": list(betas.shape),
            "dmpls_shape": list(dmpls.shape) if dmpls is not None else None,
        },
        "normalization": {
            "normalize_root_xy": normalize_root_xy,
            "keep_vertical_root": keep_vertical_root,
        },
        "joint_names": {
            "smplh_body_joints": body_names["body_joint_names"],
            "smplh_left_hand_joints": body_names["left_hand_joint_names"],
            "smplh_right_hand_joints": body_names["right_hand_joint_names"],
        },
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(sanitize(metadata), f, indent=2, ensure_ascii=False)

    return RetargetPrep(
        source_path=npz_path,
        output_npz=out_npz,
        output_json=out_json,
        data=packaged,
        metadata=metadata,
    )

# Script arguments to parse and prepare data into JSON format and correct .npz file for retargeting at 60 FPS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an AMASS .npz motion file into a simpler retarget-ready intermediate package."
    )
    parser.add_argument("npz_path", help="Path to the AMASS .npz file.")
    parser.add_argument(
        "--out-dir",
        default="data/retarget_ready",
        help="Base output directory. Defaults to data/retarget_ready",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="Target output FPS for resampling. Defaults to 60.",
    )
    # For trimming the motion clips or returning to pick different selected motions if the ones chosen do no train well.
    parser.add_argument(
        "--trim-start-s",
        type=float,
        default=0.0,
        help="Trim away this many seconds from the start before resampling.",
    )
    parser.add_argument(
        "--trim-end-s",
        type=float,
        default=None,
        help="Optional end time in seconds. If omitted, keep until the end.",
    )
    # Do not normalize X/Y
    parser.add_argument(
        "--keep-root-global",
        action="store_true",
        help="Keep the root translation in the original global frame instead of normalizing X/Y to start at zero.",
    )
    #Subtract starting Z value 
    parser.add_argument(
        "--flatten-root-z",
        action="store_true",
        help="Also normalize the vertical root position so Z starts at zero.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = find_project_root()
    npz_path = resolve_project_path(args.npz_path.strip(), project_root)
    out_dir = resolve_project_path(args.out_dir.strip(), project_root)

    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")
    if npz_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz file, got: {npz_path.name}")

    result = prepare_retarget_ready(
        npz_path=npz_path,
        out_dir=out_dir,
        target_fps=args.target_fps,
        trim_start_s=args.trim_start_s,
        trim_end_s=args.trim_end_s,
        mirror_category=True,
        normalize_root_xy=not args.keep_root_global,
        keep_vertical_root=not args.flatten_root_z,
        project_root=project_root,
    )

    print(f"Retarget-ready NPZ : {project_relative(result.output_npz, project_root)}")
    print(f"Retarget-ready JSON: {project_relative(result.output_json, project_root)}")
    print(f"Frames             : {result.metadata['output']['num_frames']}")
    print(f"Output FPS         : {result.metadata['output']['target_fps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
