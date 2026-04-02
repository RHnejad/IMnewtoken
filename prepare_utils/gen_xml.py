"""
gen_xml.py — Unified per-subject SMPL MJCF XML generator for Newton.

Generates MJCF XML with body offsets matching given SMPL-X betas or
NPY-derived bone lengths, supporting multiple foot geometry variants:
  - "box"     : box ankle + box toe (default, from prepare2)
  - "sphere"  : 8-sphere cluster per foot (4 ankle: heel + 2 balls + arch;
                                        4 toe: inner/outer back + inner/outer front)
  - "capsule" : capsule per foot segment (heel-to-ball + ball-to-tip)

Bone source options:
  - "smplx"   : compute rest-pose joints via SMPL-X BodyModel FK (requires betas)
  - "npy"     : extract bone LENGTHS from InterHuman NPY positions and scale
                 template T-pose offsets accordingly (requires npy_path)

Canonical location: prepare_utils/gen_xml.py
(Moved from prepare4/gen_xml.py; prepare4/gen_xml.py re-exports for backward compat)

Usage:
    from prepare_utils.gen_xml import generate_xml, get_or_create_xml

    # From SMPL-X betas (default)
    xml_path = generate_xml(betas=betas, foot_geom="sphere", output_path="out.xml")

    # From NPY positions (per-subject bone lengths)
    xml_path = generate_xml(npy_path="data/.../person1/0042.npy",
                            bone_source="npy", output_path="out.xml")

    # Cached version
    xml_path = get_or_create_xml(betas=betas, foot_geom="box")
    xml_path = get_or_create_xml(npy_path="path.npy", bone_source="npy")

    # Batch generation from dataset
    generate_xmls_from_dataset(data_root="data/InterHuman",
                               output_dir="data/xml_npy")
"""
import os
import sys
import copy
import hashlib
import tempfile
import numpy as np
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# Constants — imported from prepare_utils (single source of truth)
# ═══════════════════════════════════════════════════════════════
from prepare_utils.constants import R_ROT, BODY_TO_SMPLX, SMPL_22_PARENTS

VALID_FOOT_GEOMS = ("box", "sphere", "capsule")
VALID_BONE_SOURCES = ("smplx", "npy")
VALID_JOINT_STYLES = ("default", "phc")

# Template XML path
_TEMPLATE_XML = os.path.join(PROJECT_ROOT, "prepare", "assets", "smpl.xml")

# Cache directories (shared across all prepare folders)
_DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "prepare_utils", "xml_cache")
_DEFAULT_PHC_CACHE_DIR = os.path.join(PROJECT_ROOT, "prepare_utils", "xml_cache_phc")
_DEFAULT_NPY_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "xml_npy")

# ═══════════════════════════════════════════════════════════════
# PHC-style joint and actuator properties
# ═══════════════════════════════════════════════════════════════
# Extracted from PHC_orig/PHC/phc/data/assets/mjcf/smpl_0_humanoid.xml
# (the SMPL_Robot-generated neutral-gender reference skeleton).
#
# PHC uses ±180° for most joints, ±720° for arms (shoulder/elbow),
# uniform gear=500, and body-group-specific stiffness/damping.
# Torso geoms are capsules instead of spheres.

PHC_JOINT_PROPS = {
    # body_name: (stiffness, damping, range_deg)
    # range_deg applies to all 3 hinge axes of that body
    "L_Hip":      (800, 80, 180),
    "R_Hip":      (800, 80, 180),
    "L_Knee":     (800, 80, 180),
    "R_Knee":     (800, 80, 180),
    "L_Ankle":    (800, 80, 180),
    "R_Ankle":    (800, 80, 180),
    "L_Toe":      (500, 50, 180),
    "R_Toe":      (500, 50, 180),
    "Torso":      (1000, 100, 180),
    "Spine":      (1000, 100, 180),
    "Chest":      (1000, 100, 180),
    "Neck":       (500, 50, 180),
    "Head":       (500, 50, 180),
    "L_Thorax":   (500, 50, 180),
    "R_Thorax":   (500, 50, 180),
    "L_Shoulder":  (500, 50, 720),
    "R_Shoulder":  (500, 50, 720),
    "L_Elbow":    (500, 50, 720),
    "R_Elbow":    (500, 50, 720),
    "L_Wrist":    (300, 30, 180),
    "R_Wrist":    (300, 30, 180),
    "L_Hand":     (300, 30, 180),
    "R_Hand":     (300, 30, 180),
}

PHC_MOTOR_GEAR = 500  # uniform for all joints

# PHC body-group densities (from smpl_0_humanoid.xml)
PHC_DENSITIES = {
    "Pelvis": 4630,
    "L_Hip": 2041, "R_Hip": 2041,
    "L_Knee": 1235, "R_Knee": 1235,
    "L_Ankle": 1000, "R_Ankle": 1000,
    "L_Toe": 1000, "R_Toe": 1000,
    "Torso": 2041, "Spine": 2041, "Chest": 2041,
    "Neck": 1000, "Head": 1000,
    "L_Thorax": 1000, "R_Thorax": 1000,
    "L_Shoulder": 1000, "R_Shoulder": 1000,
    "L_Elbow": 1000, "R_Elbow": 1000,
    "L_Wrist": 1000, "R_Wrist": 1000,
    "L_Hand": 1000, "R_Hand": 1000,
}

# PHC uses capsules for torso/spine/chest/neck/thorax (our template uses spheres)
PHC_CAPSULE_BODIES = {"Torso", "Spine", "Chest", "Neck", "L_Thorax", "R_Thorax"}


# ═══════════════════════════════════════════════════════════════
# SMPL-X joint computation
# ═══════════════════════════════════════════════════════════════

def _compute_smplx_joints(betas, smplx_model_path=None):
    """Compute SMPL-X rest-pose joint positions for given betas.

    Returns: (55, 3) rest-pose joint positions in SMPL-X coords.
    """
    import torch

    if smplx_model_path is None:
        smplx_model_path = os.path.join(
            PROJECT_ROOT, "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"
        )

    body_model_dir = os.path.join(PROJECT_ROOT, "data", "body_model")
    if body_model_dir not in sys.path:
        sys.path.insert(0, body_model_dir)

    from body_model import BodyModel
    bm = BodyModel(smplx_model_path, num_betas=len(betas))

    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = bm(betas=betas_t)

    return out.Jtr[0].numpy().astype(np.float64)


def get_smplx_body_offset(betas):
    """Get rest-pose pelvis position for given betas.

    This offset must be added to SMPL-X trans to get the correct
    world position of the pelvis in Newton.
    """
    joints = _compute_smplx_joints(betas)
    return joints[0]


# ═══════════════════════════════════════════════════════════════
# NPY bone-length extraction
# ═══════════════════════════════════════════════════════════════

def _extract_npy_bone_lengths(npy_path, frame=0):
    """Extract bone lengths from NPY joint positions.

    NPY files contain per-subject skeleton geometry (from SMPL-X FK
    with original betas). Bone lengths are computed as Euclidean
    distances between parent-child joint pairs, which are invariant
    to coordinate system and pose.

    Args:
        npy_path: path to InterHuman NPY file (T, 492)
        frame: which frame to use (default: 0; lengths are constant
               across frames since the skeleton is rigid)

    Returns:
        bone_lengths: dict mapping (child_smplx_idx, parent_smplx_idx) → length
    """
    npy_data = np.load(npy_path).astype(np.float64)
    if npy_data.ndim != 2 or npy_data.shape[1] < 22 * 3:
        raise ValueError(
            f"NPY shape {npy_data.shape} too small; expected (T, >=66)"
        )

    positions = npy_data[frame, :22 * 3].reshape(22, 3)

    bone_lengths = {}
    for child_idx in range(22):
        parent_idx = SMPL_22_PARENTS[child_idx]
        if parent_idx < 0:
            continue
        length = float(np.linalg.norm(positions[child_idx] - positions[parent_idx]))
        bone_lengths[(child_idx, parent_idx)] = length

    return bone_lengths


def _extract_npy_joint_positions(npy_path, frame=0):
    """Extract 22-joint positions from NPY file.

    Args:
        npy_path: path to InterHuman NPY file (T, 492)
        frame: which frame to use (default: 0; bone lengths are
               constant across frames)

    Returns:
        positions: (22, 3) joint positions (in NPY coordinate system)
    """
    npy_data = np.load(npy_path).astype(np.float64)
    if npy_data.ndim != 2 or npy_data.shape[1] < 22 * 3:
        raise ValueError(
            f"NPY shape {npy_data.shape} too small; expected (T, >=66)"
        )
    return npy_data[frame, :22 * 3].reshape(22, 3)


def _update_body_pos_from_bone_lengths(body_elem, bone_lengths, parent_smplx_idx):
    """Recursively update body positions by scaling template offsets to match
    NPY-derived bone lengths (keyed by SMPL_22_PARENTS).

    Preserves the template T-pose offset DIRECTION for each bone but
    scales the MAGNITUDE to match the NPY bone length.

    Note: this function uses SMPL_22_PARENTS to key bone lengths,
    which matches the SMPL-X kintree_table and the XML hierarchy.
    Prefer _update_body_pos_from_npy_positions() for direct use since
    it avoids the intermediate bone_lengths dict.

    Args:
        body_elem: XML body Element (already has template positions)
        bone_lengths: dict from _extract_npy_bone_lengths()
        parent_smplx_idx: SMPL-X joint index of the parent body
    """
    body_name = body_elem.get("name")
    smplx_idx = BODY_TO_SMPLX.get(body_name)

    if smplx_idx is not None and parent_smplx_idx is not None:
        key = (smplx_idx, parent_smplx_idx)
        if key in bone_lengths:
            current_pos = np.array([float(x) for x in body_elem.get("pos").split()])
            current_len = np.linalg.norm(current_pos)

            if current_len > 1e-6:
                npy_len = bone_lengths[key]
                scale = npy_len / current_len
                new_pos = current_pos * scale
                body_elem.set(
                    "pos",
                    f"{new_pos[0]:.6f} {new_pos[1]:.6f} {new_pos[2]:.6f}",
                )

    current_idx = smplx_idx if smplx_idx is not None else parent_smplx_idx

    for child in body_elem:
        if child.tag == "body":
            _update_body_pos_from_bone_lengths(child, bone_lengths, current_idx)


def _update_body_pos_from_npy_positions(body_elem, npy_positions, parent_smplx_idx):
    """Recursively update body positions by scaling template offsets using
    NPY joint-pair distances.

    Computes Euclidean distance between each joint and its parent (as
    determined by the XML tree walk, which matches the SMPL-X kintree_table),
    then scales the template T-pose offset to match that distance.

    Args:
        body_elem: XML body Element (already has template positions)
        npy_positions: (22, 3) NPY joint positions
        parent_smplx_idx: SMPL-X joint index of the XML parent body
    """
    body_name = body_elem.get("name")
    smplx_idx = BODY_TO_SMPLX.get(body_name)

    if smplx_idx is not None and parent_smplx_idx is not None:
        # Distance from this joint to its XML-hierarchy parent joint
        npy_dist = float(np.linalg.norm(
            npy_positions[smplx_idx] - npy_positions[parent_smplx_idx]
        ))

        current_pos = np.array([float(x) for x in body_elem.get("pos").split()])
        current_len = np.linalg.norm(current_pos)

        if current_len > 1e-6 and npy_dist > 1e-6:
            scale = npy_dist / current_len
            new_pos = current_pos * scale
            body_elem.set(
                "pos",
                f"{new_pos[0]:.6f} {new_pos[1]:.6f} {new_pos[2]:.6f}",
            )

    current_idx = smplx_idx if smplx_idx is not None else parent_smplx_idx

    for child in body_elem:
        if child.tag == "body":
            _update_body_pos_from_npy_positions(child, npy_positions, current_idx)


# ═══════════════════════════════════════════════════════════════
# XML body position update
# ═══════════════════════════════════════════════════════════════

def _update_body_pos(body_elem, joints, parent_joint_idx):
    """Recursively update body `pos` attributes based on SMPL-X joints."""
    body_name = body_elem.get("name")
    smplx_idx = BODY_TO_SMPLX.get(body_name)

    if smplx_idx is not None and parent_joint_idx is not None:
        offset_smplx = joints[smplx_idx] - joints[parent_joint_idx]
        offset_xml = R_ROT @ offset_smplx
        body_elem.set("pos", f"{offset_xml[0]:.6f} {offset_xml[1]:.6f} {offset_xml[2]:.6f}")

    current_idx = smplx_idx if smplx_idx is not None else parent_joint_idx

    for child in body_elem:
        if child.tag == "body":
            _update_body_pos(child, joints, current_idx)


# ═══════════════════════════════════════════════════════════════
# Geom scaling utilities
# ═══════════════════════════════════════════════════════════════

def _rotation_matrix_from_to(a, b):
    """Rotation matrix R such that R @ a_norm == b_norm (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-6:
        return np.eye(3)
    if dot < -1.0 + 1e-6:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    kmat = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1.0 - dot) / s ** 2)


def _scale_geoms(updated_body, template_body):
    """Scale and realign collision geoms to match per-subject proportions."""
    body_name = updated_body.get("name")
    template_children = {c.get("name"): c for c in template_body if c.tag == "body"}

    for child_el in updated_body:
        if child_el.tag != "body":
            continue
        child_name = child_el.get("name")
        tpl_child = template_children.get(child_name)
        if tpl_child is None:
            continue

        new_cpos = np.array([float(x) for x in child_el.get("pos").split()])
        tpl_cpos = np.array([float(x) for x in tpl_child.get("pos").split()])
        tpl_dist = np.linalg.norm(tpl_cpos)
        if tpl_dist < 1e-6:
            _scale_geoms(child_el, tpl_child)
            continue
        new_dist = np.linalg.norm(new_cpos)
        scale = new_dist / tpl_dist

        R = _rotation_matrix_from_to(tpl_cpos, new_cpos)

        for geom in updated_body.findall("geom"):
            gtype = geom.get("type", "sphere")

            if gtype == "capsule" and geom.get("fromto"):
                ft = np.array([float(x) for x in geom.get("fromto").split()])
                p1 = scale * (R @ ft[:3])
                p2 = scale * (R @ ft[3:])
                geom.set("fromto", " ".join(f"{x:.6f}" for x in np.concatenate([p1, p2])))

            elif gtype == "box" and body_name in ("L_Ankle", "R_Ankle"):
                tpl_geom = template_body.find("geom[@type='box']")
                tpl_gpos = np.array([float(x) for x in tpl_geom.get("pos").split()])
                tpl_gsize = np.array([float(x) for x in tpl_geom.get("size").split()])

                heel_x = tpl_gpos[0] - tpl_gsize[0]
                toe_x = new_cpos[0]

                g_pos = np.array([float(x) for x in geom.get("pos").split()])
                g_size = np.array([float(x) for x in geom.get("size").split()])
                g_pos[0] = (heel_x + toe_x) / 2.0
                g_size[0] = max((toe_x - heel_x) / 2.0, 0.01)

                geom.set("pos", f"{g_pos[0]:.4f} {g_pos[1]:.4f} {g_pos[2]:.4f}")
                geom.set("size", f"{g_size[0]:.4f} {g_size[1]:.4f} {g_size[2]:.4f}")

        # Adjust toe geom position for visual continuity with foot box
        if child_name in ("L_Toe", "R_Toe"):
            for toe_geom in child_el.findall("geom"):
                tpl_toe_geom = tpl_child.find(
                    f"geom[@type='{toe_geom.get('type', 'box')}']"
                )
                if tpl_toe_geom is None:
                    continue
                tpl_tg_pos = np.array(
                    [float(x) for x in tpl_toe_geom.get("pos").split()]
                )
                tg_pos = np.array(
                    [float(x) for x in toe_geom.get("pos").split()]
                )
                tg_pos[1] = tpl_tg_pos[1] + (tpl_cpos[1] - new_cpos[1])
                tg_pos[2] = tpl_tg_pos[2] + (tpl_cpos[2] - new_cpos[2])
                toe_geom.set(
                    "pos", f"{tg_pos[0]:.4f} {tg_pos[1]:.4f} {tg_pos[2]:.4f}"
                )

        _scale_geoms(child_el, tpl_child)


# ═══════════════════════════════════════════════════════════════
# Foot geometry replacement: sphere cluster
# ═══════════════════════════════════════════════════════════════

def _parse_vec3(raw):
    return np.asarray([float(x) for x in raw.split()], dtype=np.float64)


def _fmt_vec3(v):
    return f"{float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}"


def _copy_geom_attrs(src, dst):
    """Copy contact/material attributes from source geom to destination."""
    for k in ("contype", "conaffinity", "density", "friction",
              "solimp", "solref", "margin", "condim", "material",
              "rgba", "group", "class"):
        if k in src.attrib:
            dst.set(k, src.get(k))


def _make_sphere_geom(name, center, radius, src):
    g = ET.Element("geom")
    g.set("name", name)
    g.set("type", "sphere")
    g.set("pos", _fmt_vec3(center))
    g.set("size", f"{float(radius):.6f}")
    _copy_geom_attrs(src, g)
    return g


def _make_capsule_geom(name, p1, p2, radius, src):
    g = ET.Element("geom")
    g.set("name", name)
    g.set("type", "capsule")
    g.set("fromto", f"{_fmt_vec3(p1)} {_fmt_vec3(p2)}")
    g.set("size", f"{float(radius):.6f}")
    _copy_geom_attrs(src, g)
    return g


def _first_box_geom(body):
    for geom in body.findall("geom"):
        if geom.get("type", "sphere") == "box":
            return geom
    return None


def _remove_box_geoms(body):
    for geom in list(body.findall("geom")):
        if geom.get("type", "sphere") == "box":
            body.remove(geom)


def _replace_one_foot_spheres(root, side):
    """Replace box foot geoms with 8-sphere cluster for one side (4 ankle + 4 toe).

    Ankle section (4 spheres):
        - heel: rear-center of foot
        - ball_inner: inner ball of foot (medial)
        - ball_outer: outer ball of foot (lateral)
        - mid_arch: mid-sole between heel and balls

    Toe section (4 spheres):
        - toe_back_inner: inner-back of toe box
        - toe_back_outer: outer-back of toe box
        - toe_front_inner: inner-front (tip, medial)
        - toe_front_outer: outer-front (tip, lateral)
    """
    ankle = root.find(f".//body[@name='{side}_Ankle']")
    if ankle is None:
        return False
    toe = ankle.find(f"body[@name='{side}_Toe']")
    if toe is None:
        return False

    foot_box = _first_box_geom(ankle)
    toe_box = _first_box_geom(toe)
    if foot_box is None or toe_box is None:
        return False

    foot_pos = _parse_vec3(foot_box.get("pos", "0 0 0"))
    foot_size = _parse_vec3(foot_box.get("size", "0.05 0.03 0.02"))
    toe_pos = _parse_vec3(toe_box.get("pos", "0 0 0"))
    toe_size = _parse_vec3(toe_box.get("size", "0.02 0.03 0.015"))

    sx, sy, sz = [max(1e-4, float(v)) for v in foot_size]
    tsx, tsy, tsz = [max(1e-4, float(v)) for v in toe_size]

    y_sign = 1.0 if side == "L" else -1.0

    # --- Ankle section: 4 spheres ---
    heel = np.array([
        foot_pos[0] - 0.58 * sx,
        foot_pos[1],
        foot_pos[2] - 0.18 * sz,
    ])
    ball_in = np.array([
        foot_pos[0] + 0.24 * sx,
        foot_pos[1] + y_sign * 0.32 * sy,
        foot_pos[2] - 0.15 * sz,
    ])
    ball_out = np.array([
        foot_pos[0] + 0.24 * sx,
        foot_pos[1] - y_sign * 0.32 * sy,
        foot_pos[2] - 0.15 * sz,
    ])
    mid_arch = np.array([
        foot_pos[0] - 0.15 * sx,
        foot_pos[1],
        foot_pos[2] - 0.16 * sz,
    ])

    r_heel = max(0.006, min(0.90 * sz, 0.45 * sy))
    r_ball = max(0.006, min(1.00 * sz, 0.40 * sy))
    r_arch = max(0.006, min(0.80 * sz, 0.35 * sy))

    # --- Toe section: 4 spheres ---
    toe_back_inner = np.array([
        toe_pos[0] - 0.30 * tsx,
        toe_pos[1] + y_sign * 0.30 * tsy,
        toe_pos[2] - 0.18 * tsz,
    ])
    toe_back_outer = np.array([
        toe_pos[0] - 0.30 * tsx,
        toe_pos[1] - y_sign * 0.30 * tsy,
        toe_pos[2] - 0.18 * tsz,
    ])
    toe_front_inner = np.array([
        toe_pos[0] + 0.40 * tsx,
        toe_pos[1] + y_sign * 0.30 * tsy,
        toe_pos[2] - 0.18 * tsz,
    ])
    toe_front_outer = np.array([
        toe_pos[0] + 0.40 * tsx,
        toe_pos[1] - y_sign * 0.30 * tsy,
        toe_pos[2] - 0.18 * tsz,
    ])

    r_toe = max(0.006, min(1.05 * tsz, 0.45 * tsy))

    # Remove old box geoms
    _remove_box_geoms(ankle)
    _remove_box_geoms(toe)

    # Ankle: 4 spheres
    ankle.append(_make_sphere_geom(f"{side}_foot_heel_sphere", heel, r_heel, foot_box))
    ankle.append(_make_sphere_geom(f"{side}_foot_ball_inner_sphere", ball_in, r_ball, foot_box))
    ankle.append(_make_sphere_geom(f"{side}_foot_ball_outer_sphere", ball_out, r_ball, foot_box))
    ankle.append(_make_sphere_geom(f"{side}_foot_mid_arch_sphere", mid_arch, r_arch, foot_box))

    # Toe: 4 spheres
    toe.append(_make_sphere_geom(f"{side}_toe_back_inner_sphere", toe_back_inner, r_toe, toe_box))
    toe.append(_make_sphere_geom(f"{side}_toe_back_outer_sphere", toe_back_outer, r_toe, toe_box))
    toe.append(_make_sphere_geom(f"{side}_toe_front_inner_sphere", toe_front_inner, r_toe, toe_box))
    toe.append(_make_sphere_geom(f"{side}_toe_front_outer_sphere", toe_front_outer, r_toe, toe_box))
    return True


def _replace_one_foot_capsules(root, side):
    """Replace box foot geoms with capsules for one side."""
    ankle = root.find(f".//body[@name='{side}_Ankle']")
    if ankle is None:
        return False
    toe = ankle.find(f"body[@name='{side}_Toe']")
    if toe is None:
        return False

    foot_box = _first_box_geom(ankle)
    toe_box = _first_box_geom(toe)
    if foot_box is None or toe_box is None:
        return False

    foot_pos = _parse_vec3(foot_box.get("pos", "0 0 0"))
    foot_size = _parse_vec3(foot_box.get("size", "0.05 0.03 0.02"))
    toe_pos = _parse_vec3(toe_box.get("pos", "0 0 0"))
    toe_size = _parse_vec3(toe_box.get("size", "0.02 0.03 0.015"))

    sx, sy, sz = [max(1e-4, float(v)) for v in foot_size]
    tsx, tsy, tsz = [max(1e-4, float(v)) for v in toe_size]

    # Foot capsule: heel → ball-of-foot, radius from smaller of Y/Z half-sizes
    r_foot = min(sy, sz)
    foot_p1 = np.array([foot_pos[0] - sx, foot_pos[1], foot_pos[2] - sz * 0.3])
    foot_p2 = np.array([foot_pos[0] + sx, foot_pos[1], foot_pos[2] - sz * 0.3])

    # Toe capsule: ball → tip
    r_toe = min(tsy, tsz)
    toe_p1 = np.array([toe_pos[0] - tsx, toe_pos[1], toe_pos[2] - tsz * 0.3])
    toe_p2 = np.array([toe_pos[0] + tsx, toe_pos[1], toe_pos[2] - tsz * 0.3])

    _remove_box_geoms(ankle)
    _remove_box_geoms(toe)

    ankle.append(_make_capsule_geom(f"{side}_foot_capsule", foot_p1, foot_p2, r_foot, foot_box))
    toe.append(_make_capsule_geom(f"{side}_toe_capsule", toe_p1, toe_p2, r_toe, toe_box))
    return True


def _apply_foot_geom(root, foot_geom):
    """Apply the chosen foot geometry to the XML tree."""
    if foot_geom == "box":
        return  # default, no changes needed
    elif foot_geom == "sphere":
        for side in ("L", "R"):
            if not _replace_one_foot_spheres(root, side):
                raise RuntimeError(f"Failed to replace {side} foot with spheres")
    elif foot_geom == "capsule":
        for side in ("L", "R"):
            if not _replace_one_foot_capsules(root, side):
                raise RuntimeError(f"Failed to replace {side} foot with capsules")
    else:
        raise ValueError(f"Unknown foot_geom={foot_geom!r}, expected one of {VALID_FOOT_GEOMS}")


# ═══════════════════════════════════════════════════════════════
# Main generation function
# ═══════════════════════════════════════════════════════════════

def _apply_phc_style(root):
    """Apply PHC-style joint, actuator, and geom properties to an MJCF tree.

    Modifies the XML in-place to match PHC's smpl_0_humanoid.xml conventions:
      - All joint ranges set to ±180° (±720° for shoulders/elbows)
      - PHC-tier stiffness/damping per body group
      - Uniform motor gear=500
      - PHC densities per body group
      - Torso/spine/chest/neck geoms → capsule (if currently sphere)
      - Adds <contact/> and <size njmax="700" nconmax="700"/>
    """
    worldbody = root.find("worldbody")

    # Walk all bodies and update joints + geoms
    for body in worldbody.iter("body"):
        bname = body.get("name")

        # Update joints (Pelvis has freejoint, no hinge joints to update)
        if bname in PHC_JOINT_PROPS:
            stiff, damp, range_deg = PHC_JOINT_PROPS[bname]
            for joint in body.findall("joint"):
                joint.set("stiffness", str(stiff))
                joint.set("damping", str(damp))
                joint.set("range", f"-{range_deg:.4f} {range_deg:.4f}")
                joint.set("armature", "0.02")

        # Update geom density (includes Pelvis)
        if bname in PHC_DENSITIES:
            for geom in body.findall("geom"):
                geom.set("density", str(PHC_DENSITIES[bname]))

        # Convert sphere geoms to capsule for PHC_CAPSULE_BODIES
        if bname in PHC_CAPSULE_BODIES:
            for geom in body.findall("geom"):
                if geom.get("type") == "sphere":
                    # Convert sphere to short capsule (fromto = small segment at center)
                    size = float(geom.get("size", "0.07"))
                    pos = geom.get("pos", "0 0 0")
                    px, py, pz = [float(v) for v in pos.split()]
                    # Short capsule segment along Z (vertical in Newton body-local)
                    half_len = size * 0.5
                    geom.set("type", "capsule")
                    geom.set("fromto", f"{px:.4f} {py:.4f} {pz - half_len:.4f} "
                                       f"{px:.4f} {py:.4f} {pz + half_len:.4f}")
                    geom.set("size", f"{size:.4f}")
                    if "pos" in geom.attrib:
                        del geom.attrib["pos"]

    # Update all actuators to uniform gear
    actuator = root.find("actuator")
    if actuator is not None:
        for motor in actuator.findall("motor"):
            motor.set("gear", str(PHC_MOTOR_GEAR))

    # Add <contact/> and <size> if not present (PHC includes these)
    if root.find("contact") is None:
        ET.SubElement(root, "contact")
    if root.find("size") is None:
        size_elem = ET.SubElement(root, "size")
        size_elem.set("njmax", "700")
        size_elem.set("nconmax", "700")


def generate_xml(betas=None, foot_geom="box", output_path=None,
                 template_path=None, bone_source="smplx", npy_path=None,
                 joint_style="default"):
    """Generate MJCF XML with per-subject body offsets and chosen foot geometry.

    Args:
        betas: (10,) SMPL-X shape parameters (required if bone_source="smplx")
        foot_geom: "box" | "sphere" | "capsule"
        output_path: where to write (None → return XML string)
        template_path: path to template smpl.xml
        bone_source: "smplx" (default) | "npy"
            - "smplx": compute rest-pose joints from BodyModel FK with betas
            - "npy": extract bone lengths from NPY positions and scale template
        npy_path: path to InterHuman NPY file (required if bone_source="npy")
        joint_style: "default" | "phc"
            - "default": use template joint limits, stiffness, gears
            - "phc": apply PHC-style properties (wide limits, high gains,
                     uniform gear=500, capsule torso geoms)

    Returns:
        output_path if writing to file, or XML string if output_path is None
    """
    if foot_geom not in VALID_FOOT_GEOMS:
        raise ValueError(f"foot_geom={foot_geom!r}, expected one of {VALID_FOOT_GEOMS}")
    if bone_source not in VALID_BONE_SOURCES:
        raise ValueError(f"bone_source={bone_source!r}, expected one of {VALID_BONE_SOURCES}")
    if joint_style not in VALID_JOINT_STYLES:
        raise ValueError(f"joint_style={joint_style!r}, expected one of {VALID_JOINT_STYLES}")

    if bone_source == "smplx" and betas is None:
        raise ValueError("betas required when bone_source='smplx'")
    if bone_source == "npy" and npy_path is None:
        raise ValueError("npy_path required when bone_source='npy'")

    if template_path is None:
        template_path = _TEMPLATE_XML

    tree = ET.parse(template_path)
    root = tree.getroot()
    template_root = copy.deepcopy(root)

    worldbody = root.find("worldbody")
    pelvis = worldbody.find("body")
    assert pelvis.get("name") == "Pelvis", f"Expected Pelvis, got {pelvis.get('name')}"
    pelvis.set("pos", "0 0 0")

    if bone_source == "smplx":
        # Existing path: compute SMPL-X rest-pose joints and set offsets directly
        joints = _compute_smplx_joints(betas)
        pelvis_idx = BODY_TO_SMPLX["Pelvis"]
        for child in pelvis:
            if child.tag == "body":
                _update_body_pos(child, joints, pelvis_idx)

    elif bone_source == "npy":
        # NPY path: extract joint positions, scale template offsets by
        # actual joint-pair distances (matching the XML body hierarchy)
        npy_positions = _extract_npy_joint_positions(npy_path)
        pelvis_idx = BODY_TO_SMPLX["Pelvis"]
        for child in pelvis:
            if child.tag == "body":
                _update_body_pos_from_npy_positions(child, npy_positions, pelvis_idx)

    template_pelvis = template_root.find("worldbody/body")
    _scale_geoms(pelvis, template_pelvis)

    # Apply foot geometry variant (must come after _scale_geoms so box sizes are correct)
    _apply_foot_geom(root, foot_geom)

    # Apply PHC-style joint/actuator/geom properties if requested
    if joint_style == "phc":
        _apply_phc_style(root)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        tree.write(output_path, xml_declaration=False)
        return output_path
    else:
        return ET.tostring(root, encoding="unicode")


# ═══════════════════════════════════════════════════════════════
# Caching
# ═══════════════════════════════════════════════════════════════

def _cache_key(betas=None, foot_geom="box", bone_source="smplx",
               npy_path=None, joint_style="default"):
    """SHA-256 hash for cache key, covering both bone source modes."""
    parts = [bone_source.encode(), foot_geom.encode(), joint_style.encode()]
    if bone_source == "smplx" and betas is not None:
        parts.append(np.asarray(betas, dtype=np.float64).tobytes())
    elif bone_source == "npy" and npy_path is not None:
        # Use absolute path so the same NPY always maps to the same key
        parts.append(os.path.abspath(npy_path).encode())
    return hashlib.sha256(b"|".join(parts)).hexdigest()[:16]


def get_or_create_xml(betas=None, foot_geom="box", cache_dir=None,
                      bone_source="smplx", npy_path=None, joint_style="default"):
    """Get or generate+cache a per-subject XML.

    When joint_style="phc", tries to use SMPL_Robot from smpl_sim for
    exact PHC-identical output (mesh-derived capsule sizing). Falls back
    to our heuristic _apply_phc_style() if smpl_sim or SMPL pkl files
    are not available.

    Args:
        betas: (10,) numpy array (required if bone_source="smplx")
        foot_geom: "box" | "sphere" | "capsule"
        cache_dir: directory to cache XMLs (default depends on bone_source/joint_style)
        bone_source: "smplx" | "npy"
        npy_path: path to NPY file (required if bone_source="npy")
        joint_style: "default" | "phc"

    Returns:
        xml_path: path to the XML file
    """
    # For PHC style with betas, try SMPL_Robot first (exact PHC output)
    if joint_style == "phc" and bone_source == "smplx" and betas is not None:
        try:
            from prepare_utils.smpl_robot_bridge import is_available, get_or_create_phc_xml
            if is_available():
                return get_or_create_phc_xml(betas, gender=0, cache_dir=cache_dir)
        except ImportError:
            pass
        # Fall through to heuristic if SMPL_Robot not available

    if cache_dir is None:
        if bone_source == "npy":
            cache_dir = _DEFAULT_NPY_CACHE_DIR
        elif joint_style == "phc":
            cache_dir = _DEFAULT_PHC_CACHE_DIR
        else:
            cache_dir = _DEFAULT_CACHE_DIR

    os.makedirs(cache_dir, exist_ok=True)
    h = _cache_key(betas=betas, foot_geom=foot_geom,
                   bone_source=bone_source, npy_path=npy_path,
                   joint_style=joint_style)
    prefix = "npy" if bone_source == "npy" else "smpl"
    style_tag = f"_{joint_style}" if joint_style != "default" else ""
    xml_path = os.path.join(cache_dir, f"{prefix}_{foot_geom}{style_tag}_{h}.xml")

    if not os.path.exists(xml_path):
        generate_xml(betas=betas, foot_geom=foot_geom, output_path=xml_path,
                     bone_source=bone_source, npy_path=npy_path,
                     joint_style=joint_style)

    return xml_path


# ═══════════════════════════════════════════════════════════════
# Batch generation from dataset
# ═══════════════════════════════════════════════════════════════

def generate_xmls_from_dataset(data_root, output_dir=None, foot_geom="box",
                               split=None):
    """Generate per-subject XMLs for all sequences in an InterHuman dataset.

    Creates one XML per person per sequence, saved as:
        <output_dir>/<seq_id>_person1.xml
        <output_dir>/<seq_id>_person2.xml

    Args:
        data_root: InterHuman data root (contains motions_processed/person{1,2}/)
        output_dir: where to write XMLs (default: data/xml_npy/)
        foot_geom: "box" | "sphere" | "capsule"
        split: if given, only process sequences listed in
               <data_root>/split/<split>.txt (e.g. "train", "val", "test")

    Returns:
        list of generated XML paths
    """
    if output_dir is None:
        output_dir = _DEFAULT_NPY_CACHE_DIR

    person1_dir = os.path.join(data_root, "motions_processed", "person1")
    person2_dir = os.path.join(data_root, "motions_processed", "person2")

    if not os.path.isdir(person1_dir):
        raise FileNotFoundError(f"Person1 dir not found: {person1_dir}")

    # Collect sequence IDs
    seq_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(person1_dir)
        if f.endswith(".npy")
    )

    # Filter by split if requested
    if split is not None:
        split_file = os.path.join(data_root, "split", f"{split}.txt")
        if os.path.isfile(split_file):
            valid_ids = set(
                line.strip() for line in open(split_file) if line.strip()
            )
            seq_ids = [s for s in seq_ids if s in valid_ids]
        else:
            print(f"Warning: split file {split_file} not found, processing all")

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for seq_id in seq_ids:
        for person_idx, person_dir in [(1, person1_dir), (2, person2_dir)]:
            npy_path = os.path.join(person_dir, f"{seq_id}.npy")
            if not os.path.isfile(npy_path):
                continue

            out_path = os.path.join(output_dir, f"{seq_id}_person{person_idx}.xml")
            if os.path.exists(out_path):
                generated.append(out_path)
                continue

            try:
                generate_xml(
                    bone_source="npy",
                    npy_path=npy_path,
                    foot_geom=foot_geom,
                    output_path=out_path,
                )
                generated.append(out_path)
            except Exception as e:
                print(f"Warning: failed {seq_id} person{person_idx}: {e}")

    return generated


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate per-subject SMPL XML")
    parser.add_argument("--bone-source", choices=VALID_BONE_SOURCES, default="smplx",
                        help="Source for body dimensions: 'smplx' (betas FK) or 'npy' (NPY positions)")
    parser.add_argument("--betas", nargs=10, type=float, default=None,
                        help="10 SMPL-X beta values (required for bone_source=smplx)")
    parser.add_argument("--npy-path", default=None,
                        help="Path to InterHuman NPY file (required for bone_source=npy)")
    parser.add_argument("--foot-geom", choices=VALID_FOOT_GEOMS, default="box",
                        help="Foot geometry type")
    parser.add_argument("--output", default=None, help="Output XML path")
    parser.add_argument("--template", default=None, help="Template XML path")

    # Batch mode
    parser.add_argument("--batch", action="store_true",
                        help="Batch generate XMLs from dataset (bone_source=npy)")
    parser.add_argument("--data-root", default=None,
                        help="InterHuman data root for batch mode")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for batch mode")
    parser.add_argument("--split", default=None,
                        help="Dataset split to process (train/val/test)")

    args = parser.parse_args()

    if args.batch:
        if args.data_root is None:
            args.data_root = os.path.join(PROJECT_ROOT, "data", "InterHuman")
        paths = generate_xmls_from_dataset(
            data_root=args.data_root,
            output_dir=args.output_dir,
            foot_geom=args.foot_geom,
            split=args.split,
        )
        print(f"Generated {len(paths)} XMLs in {args.output_dir or _DEFAULT_NPY_CACHE_DIR}")
    else:
        betas = np.array(args.betas, dtype=np.float64) if args.betas else None

        if args.output:
            out = generate_xml(
                betas=betas,
                foot_geom=args.foot_geom,
                output_path=args.output,
                template_path=args.template,
                bone_source=args.bone_source,
                npy_path=args.npy_path,
            )
        else:
            out = get_or_create_xml(
                betas=betas,
                foot_geom=args.foot_geom,
                bone_source=args.bone_source,
                npy_path=args.npy_path,
            )
        print(f"Generated: {out}")
