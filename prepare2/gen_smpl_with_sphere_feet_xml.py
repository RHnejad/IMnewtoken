"""
Generate per-subject SMPL XML with sphere-cluster feet.

This wraps prepare2/gen_smpl_xml.py and replaces ankle/toe box foot geoms with
sphere clusters (heel + two ball spheres + toe sphere) to make contact more
forgiving for small foot-angle errors.

Default cache directory is separate from standard XML cache:
  prepare2/xml_cache_sphere_feet
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prepare2.gen_smpl_xml import generate_smpl_xml  # noqa: E402


def _betas_hash(betas: np.ndarray) -> str:
    # Include version tag to avoid stale cache if sphere layout changes.
    key = b"sphere_feet_v1|" + np.asarray(betas, dtype=np.float64).tobytes()
    return hashlib.md5(key).hexdigest()[:12]


def _parse_vec3(raw: str) -> np.ndarray:
    vals = [float(x) for x in raw.split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 values, got {len(vals)} from {raw!r}")
    return np.asarray(vals, dtype=np.float64)


def _fmt_vec3(v: np.ndarray) -> str:
    return f"{float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}"


def _copy_geom_attrs(src: ET.Element, dst: ET.Element) -> None:
    # Keep contact/material attributes from source geom.
    keep = (
        "contype",
        "conaffinity",
        "density",
        "friction",
        "solimp",
        "solref",
        "margin",
        "condim",
        "material",
        "rgba",
        "group",
        "class",
    )
    for k in keep:
        if k in src.attrib:
            dst.set(k, src.get(k))


def _make_sphere_geom(name: str, center: np.ndarray, radius: float, src: ET.Element) -> ET.Element:
    g = ET.Element("geom")
    g.set("name", name)
    g.set("type", "sphere")
    g.set("pos", _fmt_vec3(center))
    g.set("size", f"{float(radius):.6f}")
    _copy_geom_attrs(src, g)
    return g


def _first_box_geom(body: ET.Element) -> Optional[ET.Element]:
    for geom in body.findall("geom"):
        if geom.get("type", "sphere") == "box":
            return geom
    return None


def _remove_box_geoms(body: ET.Element) -> None:
    for geom in list(body.findall("geom")):
        if geom.get("type", "sphere") == "box":
            body.remove(geom)


def _replace_one_foot_with_spheres(root: ET.Element, side: str) -> bool:
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

    sx = float(max(1e-4, foot_size[0]))
    sy = float(max(1e-4, foot_size[1]))
    sz = float(max(1e-4, foot_size[2]))
    tsx = float(max(1e-4, toe_size[0]))
    tsy = float(max(1e-4, toe_size[1]))
    tsz = float(max(1e-4, toe_size[2]))

    # Four-sphere foot cluster: heel + two forefoot balls + toe.
    y_sign = 1.0 if side == "L" else -1.0
    heel = np.asarray([foot_pos[0] - 0.58 * sx, foot_pos[1], foot_pos[2] - 0.18 * sz], dtype=np.float64)
    ball_in = np.asarray(
        [foot_pos[0] + 0.24 * sx, foot_pos[1] + y_sign * 0.32 * sy, foot_pos[2] - 0.15 * sz], dtype=np.float64
    )
    ball_out = np.asarray(
        [foot_pos[0] + 0.24 * sx, foot_pos[1] - y_sign * 0.32 * sy, foot_pos[2] - 0.15 * sz], dtype=np.float64
    )
    toe_center = np.asarray(
        [toe_pos[0] + 0.08 * tsx, toe_pos[1], toe_pos[2] - 0.18 * tsz], dtype=np.float64
    )

    r_heel = max(0.006, min(0.90 * sz, 0.45 * sy))
    r_ball = max(0.006, min(1.00 * sz, 0.40 * sy))
    r_toe = max(0.006, min(1.05 * tsz, 0.45 * tsy))

    _remove_box_geoms(ankle)
    _remove_box_geoms(toe)

    ankle.append(_make_sphere_geom(f"{side}_foot_heel_sphere", heel, r_heel, foot_box))
    ankle.append(_make_sphere_geom(f"{side}_foot_ball_inner_sphere", ball_in, r_ball, foot_box))
    ankle.append(_make_sphere_geom(f"{side}_foot_ball_outer_sphere", ball_out, r_ball, foot_box))
    toe.append(_make_sphere_geom(f"{side}_foot_toe_sphere", toe_center, r_toe, toe_box))
    return True


def _replace_feet_with_sphere_clusters(root: ET.Element) -> int:
    count = 0
    for side in ("L", "R"):
        if _replace_one_foot_with_spheres(root, side):
            count += 1
    return count


def generate_smpl_with_sphere_feet_xml(
    betas: np.ndarray,
    output_path: Optional[str] = None,
    template_path: Optional[str] = None,
) -> str:
    """
    Generate per-subject XML with sphere-cluster feet.

    If output_path is None, returns XML string.
    Otherwise writes XML and returns output path.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
            tmp_path = tmp.name

        generate_smpl_xml(betas, output_path=tmp_path, template_path=template_path)
        tree = ET.parse(tmp_path)
        root = tree.getroot()
        changed = _replace_feet_with_sphere_clusters(root)
        if changed == 0:
            raise RuntimeError("Foot box geoms not found; sphere-foot conversion failed.")

        if output_path is None:
            return ET.tostring(root, encoding="unicode")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        tree.write(output_path, xml_declaration=False)
        return output_path
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def get_or_create_sphere_feet_xml(
    betas: np.ndarray,
    cache_dir: Optional[str] = None,
    template_path: Optional[str] = None,
) -> str:
    """
    Get or create cached per-subject sphere-feet XML.
    """
    if cache_dir is None:
        cache_dir = os.path.join(PROJECT_ROOT, "prepare2", "xml_cache_sphere_feet")
    elif not os.path.isabs(cache_dir):
        cache_dir = os.path.join(PROJECT_ROOT, cache_dir)

    os.makedirs(cache_dir, exist_ok=True)
    h = _betas_hash(np.asarray(betas, dtype=np.float64))
    out_path = os.path.join(cache_dir, f"smpl_sphere_{h}.xml")
    if not os.path.exists(out_path):
        generate_smpl_with_sphere_feet_xml(
            np.asarray(betas, dtype=np.float64),
            output_path=out_path,
            template_path=template_path,
        )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate per-subject SMPL XML with sphere feet.")
    parser.add_argument("--betas", nargs=10, type=float, required=True, help="10 SMPL-X beta values")
    parser.add_argument("--output", default=None, help="Output XML path (optional).")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: prepare2/xml_cache_sphere_feet). Used when --output is omitted.",
    )
    parser.add_argument("--template", default=None, help="Template XML path.")
    parser.add_argument("--force", action="store_true", help="Overwrite --output if it exists.")
    args = parser.parse_args()

    betas = np.asarray(args.betas, dtype=np.float64)
    if args.output is not None:
        out = args.output
        if os.path.exists(out) and not args.force:
            print(f"Exists, skipping (use --force): {out}")
            return
        generated = generate_smpl_with_sphere_feet_xml(
            betas,
            output_path=out,
            template_path=args.template,
        )
    else:
        generated = get_or_create_sphere_feet_xml(
            betas,
            cache_dir=args.cache_dir,
            template_path=args.template,
        )
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
