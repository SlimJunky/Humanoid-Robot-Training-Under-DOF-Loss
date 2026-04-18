import numpy as np

path = r"data\mapped\Walk\37_01_poses_slow_walk_retarget_ready_g1_first_pass.npz" # Quickly change the path when needed lazy debug check

with np.load(path, allow_pickle=True) as data:
    q = data["joint_targets"]
    joint_names = data["joint_names"]
    q_default = data["default_joint_pos"]
    q_lo = data["soft_joint_lower"]
    q_hi = data["soft_joint_upper"]

print("joint_targets shape:", q.shape)
print("num joints:", len(joint_names))
print("has NaN:", np.isnan(q).any())
print("has Inf:", np.isinf(q).any())
print("within lower limits:", np.all(q >= q_lo[None, :] - 1e-9))
print("within upper limits:", np.all(q <= q_hi[None, :] + 1e-9))

# How much each joint moves away from default
motion_range = np.max(np.abs(q - q_default[None, :]), axis=0)

for name, r in zip(joint_names, motion_range):
    print(f"{name:30s} range_from_default={r:.4f}")