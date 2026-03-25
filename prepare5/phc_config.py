"""
phc_config.py — PHC-matched configuration for Newton-based motion tracking.

Contains PD gains, reward weights, simulation parameters, and body group
definitions that match PHC (Perpetual Humanoid Controller) as closely as
possible within the Newton simulator.

References:
  - PHC: phc/assets/smpl_config.py (IsaacLab actuator gains)
  - PHC: phc/env/tasks/humanoid_im.py (reward function, termination)
  - PHC: phc/data/cfg/env/env_im.yaml (simulation parameters)
  - PP-Motion: uses PHC as physics filter for plausibility scoring
"""
import numpy as np

# ═══════════════════════════════════════════════════════════════
# Simulation parameters
# ═══════════════════════════════════════════════════════════════

FPS = 30                    # Motion data framerate
DT = 1.0 / FPS             # Per-frame timestep
SIM_FREQ = 480              # Physics simulation Hz
SIM_SUBSTEPS = SIM_FREQ // FPS  # 16 substeps per frame

# Per-person DOF / coordinate counts (Newton free-joint + 23 hinges)
COORDS_PER_PERSON = 76      # 7 (free-joint q) + 69 (23 × 3 hinges)
DOFS_PER_PERSON = 75        # 6 (free-joint qd) + 69 (23 × 3 hinges)
BODIES_PER_PERSON = 24      # Pelvis + 23 joints

# Body mass
DEFAULT_BODY_MASS_KG = 75.0

# ═══════════════════════════════════════════════════════════════
# PD gains — PHC-matched
# ═══════════════════════════════════════════════════════════════
#
# PHC (smpl_config.py) uses higher gains than our old prepare2/ setup.
# These are adapted for Newton's MuJoCo-Warp solver at 480 Hz.
#
# PHC IsaacLab gains:
#   Legs:      kp=800, kd=80
#   Torso:     kp=1000, kd=100
#   Arms:      kp=300-500, kd=30-50
#
# Our previous gains (prepare2/pd_utils.py):
#   Legs:      kp=200-300, kd=20-30
#   Torso:     kp=500, kd=50
#   Arms:      kp=50-200, kd=5-20
#
# We use PHC-level gains since 480 Hz explicit PD should be stable.

# Per-body PD gains for Newton's built-in PD (joint_target_ke/kd).
# Applied to hinge DOFs only (indices 6:75 per person).
# Root DOFs use separate custom PD via control.joint_f.
# Default gain set: matched to PHC (smpl_config.py IsaacLab).
# The old prepare2/pd_utils.py used lower gains (Hip=300, Torso=500).
# These are 2-3x higher, giving stiffer tracking.
PHC_BODY_GAINS = {
    # Legs — PHC uses 800/80
    "L_Hip":      (800, 80),
    "L_Knee":     (800, 80),
    "L_Ankle":    (800, 80),
    "L_Toe":      (500, 50),
    "R_Hip":      (800, 80),
    "R_Knee":     (800, 80),
    "R_Ankle":    (800, 80),
    "R_Toe":      (500, 50),
    # Torso — PHC uses 1000/100
    "Torso":      (1000, 100),
    "Spine":      (1000, 100),
    "Chest":      (1000, 100),
    "Neck":       (500, 50),
    "Head":       (500, 50),
    # Arms — PHC uses 300-500/30-50
    "L_Thorax":   (500, 50),
    "L_Shoulder": (500, 50),
    "L_Elbow":    (300, 30),
    "L_Wrist":    (300, 30),
    "L_Hand":     (300, 30),
    "R_Thorax":   (500, 50),
    "R_Shoulder": (500, 50),
    "R_Elbow":    (300, 30),
    "R_Wrist":    (300, 30),
    "R_Hand":     (300, 30),
}

# Old gain set (from prepare2/pd_utils.py) for comparison
OLD_BODY_GAINS = {
    "L_Hip": (300, 30),    "L_Knee": (300, 30),    "L_Ankle": (200, 20),
    "L_Toe": (100, 10),    "R_Hip": (300, 30),      "R_Knee": (300, 30),
    "R_Ankle": (200, 20),  "R_Toe": (100, 10),
    "Torso": (500, 50),    "Spine": (500, 50),      "Chest": (500, 50),
    "Neck": (200, 20),     "Head": (100, 10),
    "L_Thorax": (200, 20), "L_Shoulder": (200, 20), "L_Elbow": (150, 15),
    "L_Wrist": (100, 10),  "L_Hand": (50, 5),
    "R_Thorax": (200, 20), "R_Shoulder": (200, 20), "R_Elbow": (150, 15),
    "R_Wrist": (100, 10),  "R_Hand": (50, 5),
}

# Root PD gains (applied via control.joint_f since FREE joints
# skip Newton's built-in PD).
ROOT_POS_KP = 5000.0    # N/m  (higher than old 2000 for stiffer root tracking)
ROOT_POS_KD = 500.0
ROOT_ROT_KP = 2000.0    # Nm/rad (higher than old 1000)
ROOT_ROT_KD = 200.0

OLD_ROOT_POS_KP = 2000.0
OLD_ROOT_POS_KD = 400.0
OLD_ROOT_ROT_KP = 1000.0
OLD_ROOT_ROT_KD = 200.0

# Torque limit (Nm) — must match prepare4/dynamics.py DEFAULT_TORQUE_LIMIT
TORQUE_LIMIT = 1000.0

# Armature for numerical stability
ARMATURE_HINGE = 0.5
ARMATURE_ROOT = 5.0

# ═══════════════════════════════════════════════════════════════
# PHC reward function parameters
# ═══════════════════════════════════════════════════════════════
#
# From PHC humanoid_im.py lines 1524-1554:
#   reward = w_pos * exp(-k_pos * pos_err²)
#          + w_rot * exp(-k_rot * rot_err²)
#          + w_vel * exp(-k_vel * vel_err²)
#          + w_ang_vel * exp(-k_ang_vel * ang_vel_err²)

REWARD_WEIGHTS = {
    'w_pos':     0.5,    # body position weight
    'w_rot':     0.3,    # body rotation weight
    'w_vel':     0.1,    # body linear velocity weight
    'w_ang_vel': 0.1,    # body angular velocity weight
}

REWARD_COEFFICIENTS = {
    'k_pos':     100.0,  # position sensitivity
    'k_rot':     10.0,   # rotation sensitivity
    'k_vel':     0.1,    # velocity sensitivity
    'k_ang_vel': 0.1,    # angular velocity sensitivity
}

# ═══════════════════════════════════════════════════════════════
# Termination conditions (from PHC humanoid_im.py)
# ═══════════════════════════════════════════════════════════════

# Max distance between sim and ref root before termination
TERMINATION_DISTANCE = 0.25  # meters

# Min height before termination (character has fallen)
MIN_HEIGHT = 0.3  # meters

# ═══════════════════════════════════════════════════════════════
# Body groups for analysis
# ═══════════════════════════════════════════════════════════════

BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

BODY_GROUPS = {
    "L_Leg":  ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"],
    "R_Leg":  ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
    "Spine":  ["Torso", "Spine", "Chest", "Neck", "Head"],
    "L_Arm":  ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
    "R_Arm":  ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
}

# SMPL joint → Newton body index mapping (from prepare2/retarget.py)
SMPL_TO_NEWTON = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3,
    8: 7, 9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19,
    15: 13, 16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22,
}
N_SMPL_JOINTS = 22

# Settle frames — hold initial pose to establish contacts
SETTLE_FRAMES = 15
