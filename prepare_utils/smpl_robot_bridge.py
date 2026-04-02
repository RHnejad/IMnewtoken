"""
smpl_robot_bridge.py — Bridge to PHC's SMPL_Robot for XML generation.

Uses the original SMPL_Robot class from smpl_sim (cloned in
prepare_utils/smpl_sim_repo/) to generate MJCF XMLs that are
identical to what PHC produces.

This gives us mesh-derived capsule sizing, volume-preserving geom
radii, and proper body-segment density computation — all the things
that our heuristic _apply_phc_style() approximates.

Requirements (available in conda intermask or mimickit):
    - torch, numpy, scipy, lxml, mujoco
    - SMPL model files at data/body_model/smpl/ (pkl format)

Usage:
    from prepare_utils.smpl_robot_bridge import generate_phc_xml, get_or_create_phc_xml

    # Generate PHC-style XML from betas
    xml_path = generate_phc_xml(betas, output_path="out.xml")

    # Cached version
    xml_path = get_or_create_phc_xml(betas)
"""
import os
import sys
import hashlib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add smpl_sim_repo to path so we can import SMPL_Robot
_SMPL_SIM_REPO = os.path.join(os.path.dirname(__file__), "smpl_sim_repo")
if _SMPL_SIM_REPO not in sys.path:
    sys.path.insert(0, _SMPL_SIM_REPO)

# Shim: if lxml is not installed, register our stdlib-based compatibility
# module so smpl_sim's `from lxml.etree import ...` works without lxml.
try:
    import lxml  # noqa: F401
except ImportError:
    import importlib
    _compat = importlib.import_module("lxml_compat")
    sys.modules["lxml"] = _compat
    sys.modules["lxml.etree"] = _compat.etree

# Cache directory for PHC-generated XMLs
_PHC_ROBOT_CACHE_DIR = os.path.join(PROJECT_ROOT, "prepare_utils", "xml_cache_phc_robot")

# SMPL model directory (pkl files for neutral/male/female)
# We use SMPL-X since that's what InterMask has available.
# PHC supports both via smpl_model config.
_SMPL_MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "body_model", "smpl")
# Fallback to SMPL-X if SMPL not available
_SMPL_MODEL_DIR_FALLBACK = os.path.join(PROJECT_ROOT, "data", "body_model", "smplx")

# Default robot config matching PHC's smpl_humanoid.yaml
PHC_ROBOT_CFG = {
    "model": "smpl",  # SMPL (24 bodies), NOT SMPL-X (52 bodies with fingers)
    "mesh": False,              # capsule-based, not mesh geoms
    "replace_feet": True,
    "rel_joint_lm": False,      # PHC default: no relative joint limits
    "upright_start": False,     # Our retargeting handles coords via R_ROT;
                                # upright_start rotates rest pose which breaks
                                # body offsets relative to our retargeting.
    "remove_toe": False,
    "freeze_hand": False,       # We use hand joints in our 24-body skeleton
    "real_weight": True,
    "real_weight_porpotion_capsules": True,
    "real_weight_porpotion_boxes": True,
    "masterfoot": False,
    "master_range": 30,
    "big_ankle": False,
    "box_body": False,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
    "sim": "mujoco",            # NOT isaacgym — we don't need it
    "ball_joint": False,
    "create_vel_sensors": False,
}

# Lazy singleton — creating SMPL_Robot is expensive (loads SMPL model 3x)
_robot_instance = None


def _get_robot(smpl_model_dir=None):
    """Get or create the SMPL_Robot singleton.

    First call loads SMPL model files and is slow (~5s).
    Subsequent calls reuse the cached instance.
    """
    global _robot_instance
    if _robot_instance is not None:
        return _robot_instance

    if smpl_model_dir is None:
        # Prefer SMPL (24 bodies, matches PHC default).
        # Fall back to SMPL-X only if SMPL not available.
        if os.path.isdir(_SMPL_MODEL_DIR):
            smpl_model_dir = _SMPL_MODEL_DIR
        elif os.path.isdir(_SMPL_MODEL_DIR_FALLBACK):
            smpl_model_dir = _SMPL_MODEL_DIR_FALLBACK
            PHC_ROBOT_CFG["model"] = "smplx"  # switch to SMPL-X mode

    if smpl_model_dir is None or not os.path.isdir(smpl_model_dir):
        raise FileNotFoundError(
            f"SMPL model directory not found.\n"
            f"Checked: {_SMPL_MODEL_DIR}\n"
            f"         {_SMPL_MODEL_DIR_FALLBACK}\n"
            f"Place SMPL or SMPL-X pkl files in one of these directories."
        )

    from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

    print(f"[smpl_robot_bridge] Loading SMPL_Robot from {smpl_model_dir}...")
    _robot_instance = SMPL_Robot(PHC_ROBOT_CFG, data_dir=smpl_model_dir)
    print(f"[smpl_robot_bridge] SMPL_Robot ready.")
    return _robot_instance


def generate_phc_xml(betas, output_path, gender=0, smpl_model_dir=None):
    """Generate a PHC-identical MJCF XML from SMPL betas.

    Uses SMPL_Robot from smpl_sim to produce XMLs with:
    - Mesh-derived capsule fromto and radii
    - Volume-preserving geom sizing
    - PHC-style joint ranges, stiffness, damping
    - PHC motor gears and density values

    Args:
        betas: (10,) or (16,) numpy array of SMPL shape parameters
        output_path: where to write the XML
        gender: 0=neutral, 1=male, 2=female
        smpl_model_dir: path to SMPL pkl files (default: data/body_model/smpl/)

    Returns:
        output_path
    """
    import torch

    robot = _get_robot(smpl_model_dir)

    betas_t = torch.tensor(betas, dtype=torch.float32).reshape(1, -1)
    # Pad to 10 if shorter
    if betas_t.shape[1] < 10:
        betas_t = torch.cat([betas_t, torch.zeros(1, 10 - betas_t.shape[1])], dim=1)

    robot.load_from_skeleton(
        betas=betas_t,
        gender=[gender],
        objs_info=None,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    robot.write_xml(output_path)

    # Post-process: adapt SMPL_Robot output to our 24-body Newton skeleton.
    #
    # SMPL_Robot with model="smplx" generates 52 bodies (24 body + 30 finger
    # bones: 5 fingers x 3 phalanges x 2 hands). Our skeleton expects exactly
    # 24 bodies. We strip all finger sub-bodies, keeping only L_Hand/R_Hand
    # as terminal nodes under L_Wrist/R_Wrist.
    #
    # Also: SMPL_Robot outputs body positions in SMPL-X native coords (X,Y,Z),
    # but our Newton skeleton uses R_ROT-transformed coords where
    # Newton = R_ROT @ SMPL-X, i.e. (X_n, Y_n, Z_n) = (Z_sx, X_sx, Y_sx).
    # Also zero Pelvis pos and remove scene elements that conflict with Newton.
    import xml.etree.ElementTree as ET
    from prepare_utils.constants import R_ROT, BODY_NAMES
    tree = ET.parse(output_path)
    root = tree.getroot()

    wb = root.find("worldbody")
    pelvis = wb.find("body[@name='Pelvis']")

    # ── Strip SMPL-X finger bodies ──────────────────────────
    # Keep only the 24 bodies that match our skeleton.
    # SMPL-X adds L_Index1/2/3, L_Middle1/2/3, etc. under L_Wrist.
    # We replace those with a single L_Hand / R_Hand body.
    allowed_bodies = set(BODY_NAMES)  # 24 bodies

    def _strip_extra_bodies(parent):
        """Remove child bodies not in our 24-body skeleton, recursively."""
        to_remove = []
        for child in parent.findall("body"):
            cname = child.get("name")
            if cname not in allowed_bodies:
                to_remove.append(child)
            else:
                _strip_extra_bodies(child)
        for child in to_remove:
            parent.remove(child)

    # Before stripping, save finger root positions to compute hand center.
    # SMPL-X has 5 fingers per hand (Index, Middle, Pinky, Ring, Thumb),
    # each with 3 phalanges. The average of the first phalanx positions
    # gives the hand center.
    hand_positions = {}
    for side, wrist_name, hand_name in [("L", "L_Wrist", "L_Hand"),
                                         ("R", "R_Wrist", "R_Hand")]:
        wrist = pelvis.find(f".//body[@name='{wrist_name}']")
        if wrist is None:
            continue
        finger_positions = []
        for finger in ["Index1", "Middle1", "Pinky1", "Ring1", "Thumb1"]:
            fname = f"{side}_{finger}"
            fbody = wrist.find(f"body[@name='{fname}']")
            if fbody is not None:
                pos_str = fbody.get("pos", "0 0 0")
                finger_positions.append(np.array([float(v) for v in pos_str.split()]))
        if finger_positions:
            hand_positions[hand_name] = np.mean(finger_positions, axis=0)

    _strip_extra_bodies(pelvis)

    # Create L_Hand/R_Hand as terminal bodies under L_Wrist/R_Wrist,
    # positioned at the average of the finger root positions.
    for side, wrist_name, hand_name in [("L", "L_Wrist", "L_Hand"),
                                         ("R", "R_Wrist", "R_Hand")]:
        wrist = pelvis.find(f".//body[@name='{wrist_name}']")
        if wrist is None:
            continue
        hand = wrist.find(f"body[@name='{hand_name}']")
        if hand is None:
            hand_pos = hand_positions.get(hand_name,
                                          np.array([0.0, 0.08 if side == "L" else -0.08, 0.0]))
            hand_elem = ET.SubElement(wrist, "body")
            hand_elem.set("name", hand_name)
            hand_elem.set("pos", f"{hand_pos[0]:.4f} {hand_pos[1]:.4f} {hand_pos[2]:.4f}")
            for axis_name, axis_vec in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]:
                j = ET.SubElement(hand_elem, "joint")
                j.set("name", f"{hand_name}_{axis_name}")
                j.set("type", "hinge")
                j.set("pos", "0 0 0")
                j.set("axis", axis_vec)
                j.set("stiffness", "300")
                j.set("damping", "30")
                j.set("armature", "0.02")
                j.set("range", "-180.0000 180.0000")
            # Small sphere — just enough for contact, not visually prominent
            g = ET.SubElement(hand_elem, "geom")
            g.set("type", "sphere")
            g.set("contype", "1")
            g.set("conaffinity", "1")
            g.set("density", "1000")
            g.set("size", "0.025")
            g.set("pos", "0.0000 0.0000 0.0000")

    # Strip actuator motors for removed bodies
    actuator = root.find("actuator")
    if actuator is not None:
        valid_joints = set()
        for j in pelvis.iter("joint"):
            jname = j.get("name")
            if jname:
                valid_joints.add(jname)
        # freejoint has name but no motor
        to_remove_motors = []
        for motor in actuator.findall("motor"):
            if motor.get("joint") not in valid_joints:
                to_remove_motors.append(motor)
        for m in to_remove_motors:
            actuator.remove(m)
        # Add motors for hand joints if they're new
        for hand_name in ["L_Hand", "R_Hand"]:
            for axis in ["x", "y", "z"]:
                jname = f"{hand_name}_{axis}"
                if jname in valid_joints:
                    existing = actuator.find(f"motor[@joint='{jname}']")
                    if existing is None:
                        m = ET.SubElement(actuator, "motor")
                        m.set("name", jname)
                        m.set("joint", jname)
                        m.set("gear", "500")

    # Apply R_ROT to all body positions (recursive)
    def _rotate_body_positions(body_elem):
        pos_str = body_elem.get("pos")
        if pos_str:
            pos = np.array([float(v) for v in pos_str.split()])
            rotated = R_ROT @ pos
            body_elem.set("pos", f"{rotated[0]:.4f} {rotated[1]:.4f} {rotated[2]:.4f}")

        # Also rotate capsule fromto and box pos for geoms
        for geom in body_elem.findall("geom"):
            fromto = geom.get("fromto")
            if fromto:
                vals = [float(v) for v in fromto.split()]
                p1 = R_ROT @ np.array(vals[:3])
                p2 = R_ROT @ np.array(vals[3:])
                geom.set("fromto", f"{p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f} "
                                   f"{p2[0]:.4f} {p2[1]:.4f} {p2[2]:.4f}")
            gpos = geom.get("pos")
            if gpos and geom.get("type") in ("box", "sphere"):
                gp = np.array([float(v) for v in gpos.split()])
                rp = R_ROT @ gp
                geom.set("pos", f"{rp[0]:.4f} {rp[1]:.4f} {rp[2]:.4f}")
            # Rotate box sizes (half-extents) for box geoms
            gsize = geom.get("size")
            if gsize and geom.get("type") == "box":
                sz = np.array([float(v) for v in gsize.split()])
                if len(sz) == 3:
                    rsz = np.abs(R_ROT @ sz)  # abs because half-extents are positive
                    geom.set("size", f"{rsz[0]:.4f} {rsz[1]:.4f} {rsz[2]:.4f}")

        for child in body_elem.findall("body"):
            _rotate_body_positions(child)

    # Rotate all child bodies (not Pelvis itself — we zero that)
    if pelvis is not None:
        for child in pelvis.findall("body"):
            _rotate_body_positions(child)
        # Rotate Pelvis geom positions too
        for geom in pelvis.findall("geom"):
            gpos = geom.get("pos")
            if gpos:
                gp = np.array([float(v) for v in gpos.split()])
                rp = R_ROT @ gp
                geom.set("pos", f"{rp[0]:.4f} {rp[1]:.4f} {rp[2]:.4f}")
        pelvis.set("pos", "0 0 0")

    # Remove floor geom (Newton adds its own ground plane)
    for geom in wb.findall("geom"):
        if geom.get("type") == "plane" or geom.get("name") == "floor":
            wb.remove(geom)

    # Remove lights (Newton has its own)
    for light in wb.findall("light"):
        wb.remove(light)

    # Remove asset section (textures/materials — Newton doesn't use them)
    asset = root.find("asset")
    if asset is not None:
        root.remove(asset)

    # Remove statistic/option if present (Newton has its own)
    for tag in ["statistic", "option"]:
        elem = root.find(tag)
        if elem is not None:
            root.remove(elem)

    tree.write(output_path, xml_declaration=False)
    return output_path


def _cache_key_phc(betas, gender=0):
    """Cache key for PHC robot XMLs."""
    parts = [
        b"phc_robot",
        np.asarray(betas, dtype=np.float64).tobytes(),
        str(gender).encode(),
    ]
    return hashlib.sha256(b"|".join(parts)).hexdigest()[:16]


def get_or_create_phc_xml(betas, gender=0, cache_dir=None, smpl_model_dir=None):
    """Get or generate+cache a PHC-robot XML.

    Args:
        betas: (10,) numpy array of SMPL shape parameters
        gender: 0=neutral, 1=male, 2=female
        cache_dir: where to cache (default: prepare_utils/xml_cache_phc_robot/)
        smpl_model_dir: path to SMPL pkl files

    Returns:
        xml_path: path to cached or newly generated XML
    """
    if cache_dir is None:
        cache_dir = _PHC_ROBOT_CACHE_DIR

    os.makedirs(cache_dir, exist_ok=True)
    h = _cache_key_phc(betas, gender)
    xml_path = os.path.join(cache_dir, f"smpl_phc_robot_{h}.xml")

    if not os.path.exists(xml_path):
        generate_phc_xml(betas, xml_path, gender=gender,
                         smpl_model_dir=smpl_model_dir)

    return xml_path


def is_available():
    """Check if SMPL_Robot dependencies are available."""
    try:
        import torch
        import lxml
        from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
        return os.path.isdir(_SMPL_MODEL_DIR) or os.path.isdir(_SMPL_MODEL_DIR_FALLBACK)
    except ImportError:
        return False
