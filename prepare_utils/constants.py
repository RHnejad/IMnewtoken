"""
constants.py — Single source of truth for SMPL↔Newton mappings and dimensions.

Previously duplicated in:
  - prepare2/pd_utils.py (DOFS_PER_PERSON, COORDS_PER_PERSON, BODIES_PER_PERSON)
  - prepare2/retarget.py (SMPL_TO_NEWTON, N_SMPL_JOINTS)
  - prepare2/gen_smpl_xml.py (R_ROT)
  - prepare3/xml_builder.py (R_ROT, BODY_TO_SMPLX)
  - prepare4/retarget.py (SMPL_TO_NEWTON, N_SMPL_JOINTS, N_JOINT_Q)
  - prepare4/gen_xml.py (R_ROT, BODY_TO_SMPLX, SMPL_22_PARENTS)
  - prepare5/phc_config.py (COORDS_PER_PERSON, DOFS_PER_PERSON, BODIES_PER_PERSON)

Import from here:
    from prepare_utils.constants import SMPL_TO_NEWTON, R_ROT, COORDS_PER_PERSON
"""
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# Per-person DOF / coordinate counts (Newton freejoint + 23 hinges)
# ═══════════════════════════════════════════════════════════════
COORDS_PER_PERSON = 76      # 7 (freejoint q: 3 pos + 4 quat xyzw) + 69 (23×3 hinges)
DOFS_PER_PERSON = 75        # 6 (freejoint qd: 3 lin + 3 ang) + 69 (23×3 hinges)
BODIES_PER_PERSON = 24      # Pelvis + 23 joint bodies
N_SMPL_JOINTS = 22          # SMPL-X joints used (0–21)
N_NEWTON_BODIES = 24        # Newton bodies per person
N_JOINT_Q = 76              # = COORDS_PER_PERSON

# ═══════════════════════════════════════════════════════════════
# SMPL-X joint index → Newton body index
# ═══════════════════════════════════════════════════════════════
# Verified via discover_ik.py against smpl.xml template.
# Both SMPL-X and Newton have 22 usable joints; Newton has 2 extra
# terminal bodies (L_Hand, R_Hand) with no SMPL-X correspondence.
SMPL_TO_NEWTON = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3,
    8: 7, 9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19,
    15: 13, 16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22,
}

# Inverse: Newton body index → SMPL-X joint index
NEWTON_TO_SMPL = {v: k for k, v in SMPL_TO_NEWTON.items()}

# ═══════════════════════════════════════════════════════════════
# XML body name → SMPL-X joint index (for gen_xml)
# ═══════════════════════════════════════════════════════════════
BODY_TO_SMPLX = {
    "Pelvis":     0,
    "L_Hip":      1,  "L_Knee":     4,  "L_Ankle":    7,  "L_Toe":     10,
    "R_Hip":      2,  "R_Knee":     5,  "R_Ankle":    8,  "R_Toe":     11,
    "Torso":      3,  "Spine":      6,  "Chest":      9,
    "Neck":      12,  "Head":      15,
    "L_Thorax":  13,  "L_Shoulder": 16, "L_Elbow":   18, "L_Wrist":   20, "L_Hand": None,
    "R_Thorax":  14,  "R_Shoulder": 17, "R_Elbow":   19, "R_Wrist":   21, "R_Hand": None,
}

# Ordered body names: index 0 = root (Pelvis), 1–23 = joint bodies
BODY_NAMES = [
    "Pelvis",
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

# DOF names: 0–5 = root (3 trans + 3 rot), 6–74 = 23×3 hinge DOFs
DOF_NAMES = (
    ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"]
    + [f"{BODY_NAMES[b+1]}_{axis}" for b in range(23) for axis in ("rx", "ry", "rz")]
)

# ═══════════════════════════════════════════════════════════════
# Coordinate transform: SMPL-X body-local → Newton body-local
# ═══════════════════════════════════════════════════════════════
# Newton output = R_ROT @ SMPL-X input
#   SMPL-X Z (forward) → Newton X
#   SMPL-X X (left)    → Newton Y
#   SMPL-X Y (up)      → Newton Z (world "up", Z-up convention)
R_ROT = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════
# SMPL-X 22-joint kinematic tree (parent indices)
# ═══════════════════════════════════════════════════════════════
# parent[i] = parent joint of joint i; -1 = root.
# Matches kintree_table from SMPLX_NEUTRAL.npz.
# Joints 12 (neck), 13 (L_collar), 14 (R_collar) are all children of 9 (Chest).
SMPL_22_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]

# ═══════════════════════════════════════════════════════════════
# Simulation defaults
# ═══════════════════════════════════════════════════════════════
DEFAULT_FPS = 30
DEFAULT_DT = 1.0 / DEFAULT_FPS
DEFAULT_SIM_FREQ = 480
DEFAULT_SIM_SUBSTEPS = DEFAULT_SIM_FREQ // DEFAULT_FPS  # 16
DEFAULT_BODY_MASS_KG = 75.0
DEFAULT_TORQUE_LIMIT = 1000.0  # Nm clamp

# Armature values (regularization for thin SMPL-X limbs)
ARMATURE_HINGE = 0.5
ARMATURE_ROOT = 5.0

# ═══════════════════════════════════════════════════════════════
# Template XML path
# ═══════════════════════════════════════════════════════════════
TEMPLATE_XML = os.path.join(PROJECT_ROOT, "prepare", "assets", "smpl.xml")

# Default XML cache directory
XML_CACHE_DIR = os.path.join(PROJECT_ROOT, "prepare_utils", "xml_cache")
