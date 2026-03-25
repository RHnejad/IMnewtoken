"""
log_xml_bone_dims.py — Copy per-subject XMLs for a retargeted dataset and
measure bone lengths (parent→child distance in Newton frame).

Outputs:
  <output-dir>/
      xmls/         — copies of every unique per-subject XML
      bone_dims.csv — one row per unique subject (betas hash), columns = bone names
      consistency.log — mean/std per bone across all subjects; flags >10% CV

Usage (conda activate mimickit):
    python eval_pipeline/log_xml_bone_dims.py \\
        --retargeted-dir data/retargeted_v2/generated_interx \\
        --output-dir     data/xml_generated/generated_interx

    python eval_pipeline/log_xml_bone_dims.py \\
        --retargeted-dir data/retargeted_v2/generated_interhuman \\
        --output-dir     data/xml_generated/generated_interhuman

    python eval_pipeline/log_xml_bone_dims.py \\
        --retargeted-dir data/retargeted_v2/interx \\
        --output-dir     data/xml_generated/gt_interx

    python eval_pipeline/log_xml_bone_dims.py \\
        --retargeted-dir data/retargeted_v2/interhuman \\
        --output-dir     data/xml_generated/gt_interhuman
"""
import os
import sys
import argparse
import shutil
import hashlib
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

XML_CACHE_DIR = os.path.join(PROJECT_ROOT, "prepare2", "xml_cache")

# Newton D6 body order (DFS, 24 bodies, index 0 = Pelvis)
# Parent→child bone names for logging
BONE_NAMES = [
    "Pelvis-L_Hip", "L_Hip-L_Knee", "L_Knee-L_Ankle", "L_Ankle-L_Toe",
    "Pelvis-R_Hip", "R_Hip-R_Knee", "R_Knee-R_Ankle", "R_Ankle-R_Toe",
    "Pelvis-Torso", "Torso-Spine", "Spine-Chest", "Chest-Neck",
    "Neck-Head",
    "Chest-L_Thorax", "L_Thorax-L_Shoulder", "L_Shoulder-L_Elbow",
    "L_Elbow-L_Wrist", "L_Wrist-L_Hand",
    "Chest-R_Thorax", "R_Thorax-R_Shoulder", "R_Shoulder-R_Elbow",
    "R_Elbow-R_Wrist", "R_Wrist-R_Hand",
]


def parse_xml_bone_lengths(xml_path: str) -> dict:
    """Parse an MJCF XML and return a dict {bone_name: length_m}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # Collect body positions by name using DFS
    body_positions = {}

    def _collect(el, parent_pos):
        if el.tag != "body":
            return
        name = el.get("name", "?")
        pos_str = el.get("pos", "0 0 0")
        pos = np.array([float(x) for x in pos_str.split()])
        world_pos = parent_pos + pos
        body_positions[name] = world_pos
        for child in el:
            _collect(child, world_pos)

    pelvis = worldbody.find("body")
    if pelvis is None:
        return {}
    root_pos = np.zeros(3)
    body_positions[pelvis.get("name", "Pelvis")] = root_pos
    for child in pelvis:
        _collect(child, root_pos)

    # Compute bone lengths from pairs
    lengths = {}
    for bone in BONE_NAMES:
        parent_name, child_name = bone.split("-", 1)
        p = body_positions.get(parent_name)
        c = body_positions.get(child_name)
        if p is not None and c is not None:
            lengths[bone] = float(np.linalg.norm(c - p))
        else:
            lengths[bone] = float("nan")
    return lengths


def betas_hash(betas: np.ndarray) -> str:
    return hashlib.md5(betas.tobytes()).hexdigest()[:12]


def find_betas_files(retargeted_dir: str):
    """Return list of (clip_person_prefix, betas_path) for all *_betas.npy files."""
    results = []
    for fname in sorted(os.listdir(retargeted_dir)):
        if fname.endswith("_betas.npy"):
            prefix = fname[: -len("_betas.npy")]
            results.append((prefix, os.path.join(retargeted_dir, fname)))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-subject bone dim logs from xml_cache"
    )
    parser.add_argument(
        "--retargeted-dir",
        required=True,
        help="Directory with *_betas.npy files (from retarget.py output)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write xmls/, bone_dims.csv, consistency.log",
    )
    parser.add_argument(
        "--xml-cache-dir",
        default=XML_CACHE_DIR,
        help=f"Directory with cached per-subject XMLs (default: {XML_CACHE_DIR})",
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=0.10,
        help="Flag bones with coefficient of variation > this value (default: 0.10 = 10%%)",
    )
    args = parser.parse_args()

    retargeted_dir = args.retargeted_dir
    if not os.path.isabs(retargeted_dir):
        retargeted_dir = os.path.join(PROJECT_ROOT, retargeted_dir)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    xml_cache_dir = args.xml_cache_dir

    xml_out_dir = os.path.join(output_dir, "xmls")
    os.makedirs(xml_out_dir, exist_ok=True)

    betas_files = find_betas_files(retargeted_dir)
    if not betas_files:
        print(f"ERROR: No *_betas.npy files found in {retargeted_dir}")
        sys.exit(1)

    print(f"Found {len(betas_files)} betas files in {retargeted_dir}")

    # Collect unique subjects
    seen_hashes = {}  # hash → (xml_path, betas)
    subject_lengths = []  # list of (hash, {bone: length})

    for prefix, betas_path in betas_files:
        betas = np.load(betas_path)
        h = betas_hash(betas)
        if h not in seen_hashes:
            xml_name = f"smpl_{h}.xml"
            xml_src = os.path.join(xml_cache_dir, xml_name)
            if not os.path.isfile(xml_src):
                # Try to generate on-the-fly
                try:
                    from prepare2.retarget import get_or_create_xml
                    xml_src = get_or_create_xml(betas, cache_dir=xml_cache_dir)
                    print(f"  Generated XML for {h}")
                except Exception as e:
                    print(f"  WARNING: Could not find or generate XML for {h}: {e}")
                    seen_hashes[h] = None
                    continue

            # Copy to output
            xml_dst = os.path.join(xml_out_dir, xml_name)
            shutil.copy2(xml_src, xml_dst)
            seen_hashes[h] = xml_src

        xml_path = seen_hashes.get(h)
        if xml_path:
            lengths = parse_xml_bone_lengths(xml_path)
            subject_lengths.append((h, betas_path, lengths))

    print(f"Unique subjects: {len({h for h, _, _ in subject_lengths})}")
    print(f"Total betas files processed: {len(subject_lengths)}")

    # Write bone_dims.csv
    csv_path = os.path.join(output_dir, "bone_dims.csv")
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["betas_hash", "source_file"] + BONE_NAMES)
        for h, src, lengths in subject_lengths:
            row = [h, os.path.basename(src)] + [f"{lengths.get(b, float('nan')):.6f}" for b in BONE_NAMES]
            writer.writerow(row)
    print(f"Saved bone_dims.csv → {csv_path}")

    # Consistency analysis
    bone_values = defaultdict(list)
    for _, _, lengths in subject_lengths:
        for bone in BONE_NAMES:
            v = lengths.get(bone, float("nan"))
            if not np.isnan(v):
                bone_values[bone].append(v)

    log_path = os.path.join(output_dir, "consistency.log")
    with open(log_path, "w") as lf:
        lf.write(f"Bone dimension consistency report\n")
        lf.write(f"Retargeted dir : {retargeted_dir}\n")
        lf.write(f"Unique subjects: {len({h for h, _, _ in subject_lengths})}\n")
        lf.write(f"Total samples  : {len(subject_lengths)}\n")
        lf.write(f"CV threshold   : {args.cv_threshold*100:.0f}%\n\n")
        lf.write(f"{'Bone':<30} {'Mean(m)':>9} {'Std(m)':>8} {'CV%':>7} {'Min':>8} {'Max':>8} {'Flag'}\n")
        lf.write("-" * 85 + "\n")
        flagged = []
        for bone in BONE_NAMES:
            vals = np.array(bone_values[bone])
            if len(vals) == 0:
                lf.write(f"{'MISSING: ' + bone:<30}\n")
                continue
            mean_v = vals.mean()
            std_v = vals.std()
            cv = std_v / mean_v if mean_v > 1e-9 else 0.0
            flag = "*** HIGH VARIANCE ***" if cv > args.cv_threshold else ""
            if flag:
                flagged.append(bone)
            lf.write(
                f"{bone:<30} {mean_v:>9.4f} {std_v:>8.5f} {cv*100:>6.1f}% "
                f"{vals.min():>8.4f} {vals.max():>8.4f}  {flag}\n"
            )
        lf.write("\n")
        if flagged:
            lf.write(f"FLAGGED ({len(flagged)} bones with CV > {args.cv_threshold*100:.0f}%):\n")
            for b in flagged:
                lf.write(f"  {b}\n")
        else:
            lf.write("All bones within consistency threshold ✓\n")

    print(f"Saved consistency.log → {log_path}")
    if flagged:
        print(f"  ⚠ {len(flagged)} bones flagged as high-variance (see log)")
    else:
        print("  ✓ All bones within consistency threshold")


if __name__ == "__main__":
    main()
