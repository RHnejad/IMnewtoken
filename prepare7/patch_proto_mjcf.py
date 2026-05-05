#!/usr/bin/env python3
"""Patch protomotions/data/assets/mjcf/smpl_humanoid.xml so its joint range
values exactly match the limits stored in the frozen inference config
(resolved_configs_inference.pt).

During ProtoMotion training, extract_kinematic_info() parsed the training MJCF
with dm_control and stored the joint limits (in radians) in
robot_config.kinematic_info.  IsaacGym reads the MJCF at inference time and
also reports limits in radians; _verify_joint_limits() checks the two match.

PHC's smpl_humanoid.xml (which we borrowed) encodes joint ranges in *degrees*
but uses different values from the training MJCF — e.g. Hip joints are ±90° in
PHC but ±180° in ProtoMotions.  This script patches every joint's range to the
values expected by the frozen config.

Run from inside the ProtoMotion container, working dir
/workspace/repo/prepare7/ProtoMotions:

    python /workspace/repo/prepare7/patch_proto_mjcf.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch

PROTO_ROOT = Path(__file__).resolve().parent / "ProtoMotions"
sys.path.insert(0, str(PROTO_ROOT))

CONFIGS_PT = PROTO_ROOT / "data/pretrained_models/motion_tracker/smpl/resolved_configs_inference.pt"
MJCF_PATH  = PROTO_ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml"

# ── Load frozen inference config ──────────────────────────────────────────────
print(f"Loading config from {CONFIGS_PT}")
configs     = torch.load(str(CONFIGS_PT), map_location="cpu", weights_only=False)
robot_cfg   = configs["robot"]
kin         = robot_cfg.kinematic_info

dof_names   = list(kin.dof_names)          # ['L_Hip_x', 'L_Hip_y', ...]
lower_rad   = kin.dof_limits_lower.numpy() # radians
upper_rad   = kin.dof_limits_upper.numpy()

print(f"\nJoint limits extracted from training config ({len(dof_names)} DOFs):")
print(f"  {'Joint':<20}  {'Lower (deg)':>12}  {'Upper (deg)':>12}")
print(f"  {'-'*20}  {'-'*12}  {'-'*12}")
for i, name in enumerate(dof_names):
    print(f"  {name:<20}  {np.degrees(lower_rad[i]):>12.4f}  {np.degrees(upper_rad[i]):>12.4f}")

# ── Read MJCF ─────────────────────────────────────────────────────────────────
print(f"\nPatching {MJCF_PATH}")
content = MJCF_PATH.read_text()

misses = []
for i, name in enumerate(dof_names):
    lo_deg = np.degrees(lower_rad[i])
    hi_deg = np.degrees(upper_rad[i])
    new_range = f"{lo_deg:.4f} {hi_deg:.4f}"

    # Match: <joint name="L_Hip_x" ... range="..." .../>
    # Replaces only the range="..." attribute for this specific joint
    pattern     = rf'(<joint\s+name="{re.escape(name)}"[^>]*\s)range="[^"]*"'
    replacement = rf'\1range="{new_range}"'
    patched, n  = re.subn(pattern, replacement, content)

    if n == 0:
        misses.append(name)
        print(f"  WARNING: joint '{name}' not found in MJCF — skipping")
    else:
        content = patched

if misses:
    print(f"\n{len(misses)} joint(s) not patched: {misses}")
    print("These joints may not exist in the MJCF (e.g. hand joints) — that is OK.")

MJCF_PATH.write_text(content)
print(f"\nDone. Patched {len(dof_names) - len(misses)}/{len(dof_names)} joints.")
print(f"Saved to {MJCF_PATH}")
