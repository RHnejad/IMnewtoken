"""
xml_builder.py — Generate per-subject SMPL MJCF XML for Newton with betas hashing cache.

Ported from prepare2/gen_smpl_xml.py with an added caching mechanism:
- Hashes the betas array (SHA-256) to create a unique filename.
- Stores generated XML files in prepare3/xml_cache/.
- Skips MJCF regeneration if a cached XML already exists for those betas.

This is critical for RL environments where env.reset() is called thousands
of times — regenerating the MJCF every reset would be prohibitively slow.

Usage:
    from prepare3.xml_builder import get_or_create_xml

    xml_path = get_or_create_xml(betas)  # returns cached path if exists
"""
import os
import sys
import copy
import hashlib
import numpy as np
import xml.etree.ElementTree as ET

# ─── Project paths ───────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xml_cache")

# ─── Coordinate transform ───────────────────────────────────
# R_ROT maps SMPL-X body-local coordinates → Newton XML body-local frame.
#   Newton output = R_ROT @ SMPL-X input  →  (SMPL-Z, SMPL-X, SMPL-Y)
R_ROT = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


# ─── XML body name → SMPL-X joint index ─────────────────────
BODY_TO_SMPLX = {
    "Pelvis":     0,
    "L_Hip":      1,
    "L_Knee":     4,
    "L_Ankle":    7,
    "L_Toe":     10,
    "R_Hip":      2,
    "R_Knee":     5,
    "R_Ankle":    8,
    "R_Toe":     11,
    "Torso":      3,   # Spine1
    "Spine":      6,   # Spine2
    "Chest":      9,   # Spine3
    "Neck":      12,
    "Head":      15,
    "L_Thorax":  13,   # L_Collar
    "L_Shoulder":16,
    "L_Elbow":   18,
    "L_Wrist":   20,
    "L_Hand":    None,
    "R_Thorax":  14,   # R_Collar
    "R_Shoulder":17,
    "R_Elbow":   19,
    "R_Wrist":   21,
    "R_Hand":    None,
}

# Expected counts
N_NEWTON_BODIES = 24       # 24 bodies in the MJCF skeleton
N_SMPLX_JOINTS = 22        # 22 corresponding SMPL-X joints (excl. L/R_Hand)

# Template XML path
_TEMPLATE_XML = os.path.join(PROJECT_ROOT, "prepare", "assets", "smpl.xml")


# ═══════════════════════════════════════════════════════════════
# Betas Hashing & Caching
# ═══════════════════════════════════════════════════════════════
def _hash_betas(betas: np.ndarray) -> str:
    """SHA-256 hash of betas array, truncated to 16 chars."""
    betas_bytes = betas.astype(np.float64).tobytes()
    return hashlib.sha256(betas_bytes).hexdigest()[:16]


def get_cached_xml_path(betas: np.ndarray) -> str:
    """Return the cache file path for given betas (may not exist yet)."""
    h = _hash_betas(betas)
    return os.path.join(CACHE_DIR, f"smpl_{h}.xml")


def get_or_create_xml(betas: np.ndarray, template_path: str = None) -> str:
    """Get or create a cached MJCF XML for the given SMPL-X betas.

    If a cached XML exists for this betas hash, returns its path immediately.
    Otherwise generates it and returns the path.

    Args:
        betas: (10,) numpy array of SMPL-X shape parameters.
        template_path: optional override for the template XML file.

    Returns:
        xml_path: absolute path to the generated/cached XML file.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    xml_path = get_cached_xml_path(betas)
    if os.path.exists(xml_path):
        return xml_path
    return generate_smpl_xml(betas, output_path=xml_path,
                             template_path=template_path)


# ═══════════════════════════════════════════════════════════════
# SMPL-X Joint Computation
# ═══════════════════════════════════════════════════════════════
def _compute_smplx_joints(betas, smplx_model_path=None):
    """Compute SMPL-X rest-pose joint positions for given betas.

    Args:
        betas: (10,) numpy array of shape parameters
        smplx_model_path: path to SMPLX_NEUTRAL.npz

    Returns:
        joints: (55, 3) rest-pose joint positions in SMPL-X coords
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


# ═══════════════════════════════════════════════════════════════
# XML Modification Utilities
# ═══════════════════════════════════════════════════════════════
def _update_body_pos(body_elem, joints, parent_joint_idx):
    """Recursively update body `pos` attributes from SMPL-X joint positions."""
    body_name = body_elem.get("name")
    smplx_idx = BODY_TO_SMPLX.get(body_name)

    if smplx_idx is not None and parent_joint_idx is not None:
        offset_smplx = joints[smplx_idx] - joints[parent_joint_idx]
        offset_xml = R_ROT @ offset_smplx
        body_elem.set("pos",
                       f"{offset_xml[0]:.6f} {offset_xml[1]:.6f} {offset_xml[2]:.6f}")

    current_idx = smplx_idx if smplx_idx is not None else parent_joint_idx

    for child in body_elem:
        if child.tag == "body":
            _update_body_pos(child, joints, current_idx)


def _rotation_matrix_from_to(a, b):
    """Rotation matrix R such that R @ a_norm == b_norm (Rodrigues formula)."""
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
    """Scale and realign collision geoms to match per-subject body proportions."""
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
                geom.set("fromto",
                         " ".join(f"{x:.6f}" for x in np.concatenate([p1, p2])))

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

        # Adjust toe geom position for visual continuity
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


def _find_body(root_elem, name):
    """Find a body element by name, searching recursively."""
    if root_elem.get("name") == name:
        return root_elem
    for child in root_elem:
        if child.tag == "body":
            result = _find_body(child, name)
            if result is not None:
                return result
    return None


# ═══════════════════════════════════════════════════════════════
# Main Generation Function
# ═══════════════════════════════════════════════════════════════
def generate_smpl_xml(betas, output_path=None, template_path=None):
    """Generate an MJCF XML with body offsets matching the given SMPL-X betas.

    Produces an MJCF with exactly 24 Newton bodies and 22 SMPL-X joints
    (matching the prepare2 skeleton topology).

    Args:
        betas: (10,) numpy array of shape parameters
        output_path: where to write the XML (if None, returns XML string)
        template_path: path to template smpl.xml

    Returns:
        output_path if writing to file, or XML string if output_path is None
    """
    if template_path is None:
        template_path = _TEMPLATE_XML

    tree = ET.parse(template_path)
    root = tree.getroot()
    template_root = copy.deepcopy(root)

    joints = _compute_smplx_joints(betas)

    worldbody = root.find("worldbody")
    pelvis = worldbody.find("body")
    assert pelvis.get("name") == "Pelvis", \
        f"Expected Pelvis, got {pelvis.get('name')}"

    pelvis.set("pos", "0 0 0")

    pelvis_idx = BODY_TO_SMPLX["Pelvis"]
    for child in pelvis:
        if child.tag == "body":
            _update_body_pos(child, joints, pelvis_idx)

    template_pelvis = template_root.find("worldbody/body")
    _scale_geoms(pelvis, template_pelvis)

    # Verify body count (iter("body") on pelvis includes pelvis itself)
    body_count = sum(1 for _ in pelvis.iter("body"))
    assert body_count == N_NEWTON_BODIES, \
        f"Expected {N_NEWTON_BODIES} bodies, got {body_count}"

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        tree.write(output_path, xml_declaration=False)
        return output_path
    else:
        return ET.tostring(root, encoding="unicode")


def get_smplx_body_offset(betas):
    """Get the rest-pose pelvis position for given betas.

    Args:
        betas: (10,) numpy array

    Returns:
        offset: (3,) pelvis position in SMPL-X coords
    """
    joints = _compute_smplx_joints(betas)
    return joints[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate per-subject SMPL XML")
    parser.add_argument("--betas", nargs=10, type=float, required=True,
                        help="10 SMPL-X beta values")
    parser.add_argument("--output", required=True, help="Output XML path")
    parser.add_argument("--template", default=None, help="Template XML path")
    args = parser.parse_args()

    betas = np.array(args.betas, dtype=np.float64)
    out = generate_smpl_xml(betas, output_path=args.output,
                            template_path=args.template)
    print(f"Generated: {out}")
