"""
gen_smpl_xml.py — Generate per-subject SMPL MJCF XML for Newton.

Given SMPL-X betas, computes rest-pose joint positions via BodyModel FK,
then generates an MJCF XML with body offsets matching that subject's
skeleton proportions.

This ensures Newton's FK exactly matches SMPL-X FK (modulo pose blend shapes),
so direct rotation transfer yields near-zero position error.

Usage:
    from prepare2.gen_smpl_xml import generate_smpl_xml

    xml_path = generate_smpl_xml(
        betas=np.array([...]),       # (10,) SMPL-X shape params
        output_path="assets/smpl_subject_001.xml",
    )
"""
import os
import sys
import copy
import numpy as np
import xml.etree.ElementTree as ET

# ─── Coordinate transform ───────────────────────────────────────
# R_ROT maps SMPL-X body-local coordinates → Newton XML body-local frame.
#
#   Newton output = R_ROT @ SMPL-X input  →  (SMPL-Z, SMPL-X, SMPL-Y)
#
#   Key mapping:  SMPL-X Y (local "up")  →  Newton Z  (world "up", Z-up)
#                 SMPL-X X               →  Newton Y
#                 SMPL-X Z (forward)     →  Newton X
#
# This preserves the Z-up world convention:
#   - Joints going upward (Torso, Spine, Head…) have  positive Z offsets in the XML.
#   - Joints going downward (Hips, Knees, Ankles…) have negative Z offsets.
#   - Joints going forward  (Toes…)                have  positive X offsets.
#
# Retarget pipeline also uses up_axis=Z everywhere; joint_q root Z ≈ 0.35 m
# is the standing pelvis height. Correctness is verified empirically by
# near-zero MPJPE (0.00 cm) in the rotation-transfer retargeting test.
R_ROT = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


# ─── XML body name → SMPL-X joint index ─────────────────────────
# Maps each body in the XML tree to the corresponding SMPL-X joint.
# Parent info comes from the XML nesting (not explicit here).
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
    "L_Hand":    None,  # No SMPL-X joint (SMPL has it at 22)
    "R_Thorax":  14,   # R_Collar
    "R_Shoulder":17,
    "R_Elbow":   19,
    "R_Wrist":   21,
    "R_Hand":    None,  # No SMPL-X joint (SMPL has it at 23)
}

# Template XML path (relative to InterMask root)
_TEMPLATE_XML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prepare", "assets", "smpl.xml"
)


def _compute_smplx_joints(betas, smplx_model_path=None):
    """
    Compute SMPL-X rest-pose joint positions for given betas.

    Args:
        betas: (10,) numpy array of shape parameters
        smplx_model_path: path to SMPLX_NEUTRAL.npz

    Returns:
        joints: (55, 3) rest-pose joint positions in SMPL-X coords
    """
    import torch

    if smplx_model_path is None:
        smplx_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"
        )

    # Add body_model to path
    body_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "body_model"
    )
    if body_model_dir not in sys.path:
        sys.path.insert(0, body_model_dir)

    from body_model import BodyModel
    bm = BodyModel(smplx_model_path, num_betas=len(betas))

    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)  # (1, 10)
    with torch.no_grad():
        out = bm(betas=betas_t)

    return out.Jtr[0].numpy().astype(np.float64)  # (55, 3)


def _update_body_pos(body_elem, joints, parent_joint_idx):
    """
    Recursively update body `pos` attributes based on SMPL-X joint positions.

    Args:
        body_elem: XML Element for this body
        joints: (55, 3) SMPL-X joint positions
        parent_joint_idx: SMPL-X joint index of the parent body
    """
    body_name = body_elem.get("name")
    smplx_idx = BODY_TO_SMPLX.get(body_name)

    if smplx_idx is not None and parent_joint_idx is not None:
        # Compute local offset: child - parent, in SMPL-X coords
        offset_smplx = joints[smplx_idx] - joints[parent_joint_idx]
        # Transform to Newton XML frame
        offset_xml = R_ROT @ offset_smplx
        body_elem.set("pos", f"{offset_xml[0]:.6f} {offset_xml[1]:.6f} {offset_xml[2]:.6f}")

    # Determine this body's joint index for children
    current_idx = smplx_idx if smplx_idx is not None else parent_joint_idx

    # Recurse into child bodies
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
    """
    Scale and realign collision geoms to match per-subject body proportions.

    For capsule geoms (limb segments): rotates fromto endpoints to match the
    new bone direction, then scales by new_length / template_length.

    For box geoms (ankle/foot): adjusts the X-extent so the foot box
    covers exactly from the heel edge to the toe joint, ensuring
    foot-toe geometry continuity.  Also shifts the toe geom in Y/Z
    to compensate for the raw SMPL-X offset, keeping it visually
    aligned with the foot box while preserving 0.00 cm MPJPE.

    Args:
        updated_body: XML body Element with updated per-subject positions.
        template_body: XML body Element with original template positions.
    """
    body_name = updated_body.get("name")

    # Match children between updated and template trees
    template_children = {c.get("name"): c for c in template_body if c.tag == "body"}

    for child_el in updated_body:
        if child_el.tag != "body":
            continue
        child_name = child_el.get("name")
        tpl_child = template_children.get(child_name)
        if tpl_child is None:
            continue

        # Compute scale and direction change from child body offset change
        new_cpos = np.array([float(x) for x in child_el.get("pos").split()])
        tpl_cpos = np.array([float(x) for x in tpl_child.get("pos").split()])
        tpl_dist = np.linalg.norm(tpl_cpos)
        if tpl_dist < 1e-6:
            _scale_geoms(child_el, tpl_child)
            continue
        new_dist = np.linalg.norm(new_cpos)
        scale = new_dist / tpl_dist

        # Rotation that maps template bone direction → new bone direction
        R = _rotation_matrix_from_to(tpl_cpos, new_cpos)

        # Scale geoms in this body that represent the link to this child
        for geom in updated_body.findall("geom"):
            gtype = geom.get("type", "sphere")

            if gtype == "capsule" and geom.get("fromto"):
                # Rotate endpoints to new bone direction, then scale for new length
                ft = np.array([float(x) for x in geom.get("fromto").split()])
                p1 = scale * (R @ ft[:3])
                p2 = scale * (R @ ft[3:])
                geom.set("fromto", " ".join(f"{x:.6f}" for x in np.concatenate([p1, p2])))

            elif gtype == "box" and body_name in ("L_Ankle", "R_Ankle"):
                # Foot box: extend from heel edge (fixed) to toe joint
                tpl_geom = template_body.find("geom[@type='box']")
                tpl_gpos = np.array([float(x) for x in tpl_geom.get("pos").split()])
                tpl_gsize = np.array([float(x) for x in tpl_geom.get("size").split()])

                heel_x = tpl_gpos[0] - tpl_gsize[0]   # back of foot (fixed)
                toe_x = new_cpos[0]                     # front → toe joint

                g_pos = np.array([float(x) for x in geom.get("pos").split()])
                g_size = np.array([float(x) for x in geom.get("size").split()])
                g_pos[0] = (heel_x + toe_x) / 2.0
                g_size[0] = max((toe_x - heel_x) / 2.0, 0.01)

                geom.set("pos", f"{g_pos[0]:.4f} {g_pos[1]:.4f} {g_pos[2]:.4f}")
                geom.set("size", f"{g_size[0]:.4f} {g_size[1]:.4f} {g_size[2]:.4f}")

        # Adjust toe geom position for visual continuity with foot box.
        # The toe body stays at raw SMPL-X position (for 0.00 cm MPJPE),
        # but the geom is shifted in Y/Z so it appears aligned with the foot.
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
                # Compensate Y/Z for body offset difference vs template
                tg_pos[1] = tpl_tg_pos[1] + (tpl_cpos[1] - new_cpos[1])
                tg_pos[2] = tpl_tg_pos[2] + (tpl_cpos[2] - new_cpos[2])
                toe_geom.set(
                    "pos", f"{tg_pos[0]:.4f} {tg_pos[1]:.4f} {tg_pos[2]:.4f}"
                )

        # Recurse into child
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


def _fix_toe_positions(pelvis, template_pelvis):
    """
    Fix toe body positions to use template-proportional offsets.

    SMPL-X T-pose foot joints have lateral (Y) and downward (Z) components
    because the foot is slightly angled in SMPL-X rest pose.  This makes the
    toe body appear disconnected from the foot box in the viewer.

    Fix: scale the TEMPLATE toe offset by the forward (X) foot-length ratio,
    preserving the template's Y/Z proportions (flat-footed appearance).
    """
    for side in ("L", "R"):
        ankle_name = f"{side}_Ankle"
        toe_name = f"{side}_Toe"

        ankle = _find_body(pelvis, ankle_name)
        tpl_ankle = _find_body(template_pelvis, ankle_name)
        if ankle is None or tpl_ankle is None:
            continue

        toe = next((c for c in ankle if c.tag == "body" and c.get("name") == toe_name), None)
        tpl_toe = next((c for c in tpl_ankle if c.tag == "body" and c.get("name") == toe_name), None)
        if toe is None or tpl_toe is None:
            continue

        current_pos = np.array([float(x) for x in toe.get("pos").split()])
        tpl_pos = np.array([float(x) for x in tpl_toe.get("pos").split()])

        # Scale the template offset by the forward foot-length ratio
        if abs(tpl_pos[0]) > 1e-6:
            scale = current_pos[0] / tpl_pos[0]
            fixed = tpl_pos * scale
            toe.set("pos", f"{fixed[0]:.6f} {fixed[1]:.6f} {fixed[2]:.6f}")


def generate_smpl_xml(betas, output_path=None, template_path=None):
    """
    Generate an MJCF XML file with body offsets matching the given SMPL-X betas.

    Args:
        betas: (10,) numpy array of shape parameters
        output_path: where to write the XML (if None, returns XML string)
        template_path: path to template smpl.xml

    Returns:
        output_path if writing to file, or XML string if output_path is None
    """
    if template_path is None:
        template_path = _TEMPLATE_XML

    # Parse template
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Keep a deep copy of the template for geom scaling reference
    template_root = copy.deepcopy(root)

    # Compute SMPL-X rest-pose joints
    joints = _compute_smplx_joints(betas)

    # Find the root body (Pelvis) in worldbody
    worldbody = root.find("worldbody")
    pelvis = worldbody.find("body")
    assert pelvis.get("name") == "Pelvis", f"Expected Pelvis, got {pelvis.get('name')}"

    # Pelvis stays at origin (freejoint handles translation)
    pelvis.set("pos", "0 0 0")

    # Update all child bodies recursively
    pelvis_idx = BODY_TO_SMPLX["Pelvis"]  # 0
    for child in pelvis:
        if child.tag == "body":
            _update_body_pos(child, joints, pelvis_idx)

    # Scale geoms to match per-subject proportions (foot-toe continuity, limb capsules)
    template_pelvis = template_root.find("worldbody/body")
    _scale_geoms(pelvis, template_pelvis)

    # Note: _fix_toe_positions() was previously called here to flatten
    # the toe body offset for visual continuity with the foot box.
    # Removed because it shifts the L_Toe/R_Toe body origins ~4.5cm
    # from their true SMPL-X positions, introducing MPJPE error on
    # L_Foot and R_Foot joints. The raw SMPL-X offsets are now used
    # directly; any minor visual gap is acceptable.

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        tree.write(output_path, xml_declaration=False)
        return output_path
    else:
        return ET.tostring(root, encoding="unicode")


def get_smplx_body_offset(betas):
    """
    Get the rest-pose pelvis position for given betas.

    This offset must be added to SMPL-X trans to get the correct
    world position of the pelvis in Newton.

    Args:
        betas: (10,) numpy array

    Returns:
        offset: (3,) pelvis position in SMPL-X coords
    """
    joints = _compute_smplx_joints(betas)
    return joints[0]  # pelvis rest position


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate per-subject SMPL XML")
    parser.add_argument("--betas", nargs=10, type=float, required=True,
                        help="10 SMPL-X beta values")
    parser.add_argument("--output", required=True, help="Output XML path")
    parser.add_argument("--template", default=None, help="Template XML path")
    args = parser.parse_args()

    betas = np.array(args.betas, dtype=np.float64)
    out = generate_smpl_xml(betas, output_path=args.output, template_path=args.template)
    print(f"Generated: {out}")
