#!/usr/bin/env python3
"""Batch-convert InterX ground-truth motions to a single merged PHC pkl.

InterX is a SMPL-X dataset captured at 60 fps.  Each clip has two separate
npz files (P1.npz / P2.npz) under motions/{clip_id}/.  This script:

  1. Reads P1.npz / P2.npz for every clip in the requested split
  2. Converts Y-up SMPL-X → Z-up SMPL (dropping hand/face joints)
  3. Downsamples from 60 → 30 fps
  4. Saves one staging pkl per clip (resume-safe, subprocess-batched for OOM)
  5. Merges all staging pkls into one combined pkl for PHC inference

Output motion keys: {clip_id}_p0 / {clip_id}_p1

Usage (from repo root, inside Docker):
    # Test split (default), subprocess-batched to avoid OOM:
    python prepare7/batch_convert_interx_to_phc.py --subprocess-batch-size 300

    # Specific split:
    python prepare7/batch_convert_interx_to_phc.py --split train

    # Specific clips only:
    python prepare7/batch_convert_interx_to_phc.py --clip-ids G001T000A000R004,G001T000A001R005

    # Merge only (staging files already present):
    python prepare7/batch_convert_interx_to_phc.py --merge-only

    # Custom paths:
    python prepare7/batch_convert_interx_to_phc.py \\
        --motions-dir   data/InterX/motions \\
        --splits-dir    data/InterX/splits \\
        --smpl-data-dir PHC/data/smpl \\
        --staging-dir   PHC/output/interx_gt/staging \\
        --output        PHC/output/interx_gt/test.pkl
"""

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from interhuman_to_phc import convert_person  # noqa: E402

# InterX is Y-up SMPL-X; PHC expects Z-up.
# R_YUPTOZUP is the 120° cyclic rotation [0.5,0.5,0.5,0.5] (xyzw).
# It maps: X→Y→Z→X, i.e. (x,y,z)_yup → (z,x,y)_zup.
# Verified: R @ [x, y_height, z] → [z, x, y_height] where z_new=y_old=pelvis height ✓
_R_YUPTOZUP = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()

_SRC_FPS  = 60    # InterX capture rate
_DST_FPS  = 30    # PHC target rate
_STEP     = _SRC_FPS // _DST_FPS   # = 2


# --------------------------------------------------------------------------- #
# Per-person loading + coordinate conversion                                   #
# --------------------------------------------------------------------------- #

def load_person_npz(npz_path: Path) -> dict:
    """Load one InterX person npz and convert to Z-up SMPL format."""
    d = np.load(str(npz_path), allow_pickle=True)

    trans_yup     = d["trans"].astype(np.float64)        # (T, 3) Y-up
    root_yup      = d["root_orient"].astype(np.float64)  # (T, 3) Y-up axis-angle
    pose_body     = d["pose_body"].astype(np.float64)    # (T, 21, 3)
    betas         = d["betas"].astype(np.float64).reshape(-1)[:10]  # (10,)
    gender        = str(d["gender"])

    T = trans_yup.shape[0]

    # --- Downsample 60 → 30 fps ---
    trans_yup = trans_yup[::_STEP]
    root_yup  = root_yup[::_STEP]
    pose_body = pose_body[::_STEP]
    T         = trans_yup.shape[0]

    # --- Y-up → Z-up for root translation ---
    # R_YUPTOZUP @ col-vec: new = R @ old → apply to each row via matmul
    trans_zup = (trans_yup @ _R_YUPTOZUP.T)   # (T, 3)

    # --- Y-up → Z-up for root orientation ---
    # R_pose_zup = R_yup2zup @ R_pose_yup @ R_yup2zup.T
    R_root = sRot.from_rotvec(root_yup).as_matrix()       # (T, 3, 3)
    R_root_zup = _R_YUPTOZUP @ R_root @ _R_YUPTOZUP.T     # (T, 3, 3)
    root_zup = sRot.from_matrix(R_root_zup).as_rotvec()   # (T, 3)

    # --- Body joints: local rotations, no frame transform needed ---
    pose_body_flat = pose_body.reshape(T, 63)              # (T, 63)

    return {
        "trans":       trans_zup,
        "root_orient": root_zup,
        "pose_body":   pose_body_flat,
        "betas":       betas,
        "gender":      gender,
    }


# --------------------------------------------------------------------------- #
# Conversion step                                                              #
# --------------------------------------------------------------------------- #

def convert_step(args):
    smpl_dir  = str(args.smpl_data_dir)
    os.makedirs(args.staging_dir, exist_ok=True)

    if args.clip_ids:
        clip_ids = [c.strip() for c in args.clip_ids.split(",")]
    else:
        split_file = args.splits_dir / f"{args.split}.txt"
        with open(split_file) as f:
            clip_ids = [l.strip() for l in f if l.strip()]

    print(f"[*] {len(clip_ids)} clips ({args.split} split) → staging: {args.staging_dir}")

    errors, skipped, converted = [], 0, 0

    for clip_id in tqdm(clip_ids, desc="Converting"):
        staging_path = args.staging_dir / f"{clip_id}.pkl"
        if staging_path.exists() and not args.force:
            skipped += 1
            continue

        clip_dir = args.motions_dir / clip_id
        if not clip_dir.exists():
            tqdm.write(f"  [warn] missing dir: {clip_dir}")
            errors.append(clip_id)
            continue

        try:
            clip_data = {}
            for npz_name, person_key in [("P1.npz", "p0"), ("P2.npz", "p1")]:
                npz_path = clip_dir / npz_name
                if not npz_path.exists():
                    tqdm.write(f"  [warn] missing: {npz_path}")
                    continue
                person_data = load_person_npz(npz_path)
                clip_data[f"{clip_id}_{person_key}"] = convert_person(
                    person_data, smpl_dir, _DST_FPS
                )
            if clip_data:
                joblib.dump(clip_data, str(staging_path), compress=0)
                converted += 1
            else:
                errors.append(clip_id)
        except Exception as exc:
            tqdm.write(f"  [error] {clip_id}: {exc}")
            errors.append(clip_id)
        finally:
            try:
                del clip_data, person_data
            except NameError:
                pass
            gc.collect()

    print(f"\n[convert] done — converted={converted}  skipped={skipped}  errors={len(errors)}")
    if errors:
        print(f"  failed: {errors[:10]}{'...' if len(errors) > 10 else ''}")
    return errors


# --------------------------------------------------------------------------- #
# Subprocess-batched conversion                                                #
# --------------------------------------------------------------------------- #

def subprocess_convert_step(args):
    if args.clip_ids:
        all_ids = [c.strip() for c in args.clip_ids.split(",")]
    else:
        split_file = args.splits_dir / f"{args.split}.txt"
        with open(split_file) as f:
            all_ids = [l.strip() for l in f if l.strip()]

    if not args.force:
        pending = [c for c in all_ids if not (args.staging_dir / f"{c}.pkl").exists()]
    else:
        pending = all_ids

    if not pending:
        print("[subprocess-convert] All clips already converted.")
        return []

    batch_size = args.subprocess_batch_size
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    print(f"[subprocess-convert] {len(pending)} pending → {len(batches)} batches of ≤{batch_size}")

    all_errors = []
    for i, batch in enumerate(batches):
        print(f"  [batch {i+1}/{len(batches)}]  {batch[0]}..{batch[-1]}")
        cmd = [
            sys.executable, __file__,
            "--motions-dir",    str(args.motions_dir),
            "--splits-dir",     str(args.splits_dir),
            "--smpl-data-dir",  str(args.smpl_data_dir),
            "--staging-dir",    str(args.staging_dir),
            "--output",         str(args.output),
            "--clip-ids",       ",".join(batch),
            "--merge-only-skip",
        ]
        if args.force:
            cmd.append("--force")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [warn] batch {i+1} exited with code {result.returncode}")
            all_errors.extend(batch)

    return all_errors


# --------------------------------------------------------------------------- #
# Merge step                                                                   #
# --------------------------------------------------------------------------- #

def merge_step(args):
    staging_files = sorted(args.staging_dir.glob("*.pkl"))
    if not staging_files:
        print(f"[merge] No staging files in {args.staging_dir}")
        return

    if args.output.exists() and not args.force:
        print(f"[merge] Output exists: {args.output}  (use --force to overwrite)")
        return

    print(f"[merge] Merging {len(staging_files)} staging files → {args.output}")
    os.makedirs(args.output.parent, exist_ok=True)

    merged = {}
    for sp in tqdm(staging_files, desc="Merging"):
        try:
            merged.update(joblib.load(str(sp)))
        except Exception as exc:
            tqdm.write(f"  [warn] skipping {sp.name}: {exc}")

    joblib.dump(merged, str(args.output))
    print(f"[merge] {len(merged)} motions saved → {args.output}")

    first = next(iter(merged.values()))
    rz = first["root_trans_offset"][:, 2]
    print(f"  Sanity: root Z  min={rz.min():.3f}  mean={rz.mean():.3f}  (expect ~0.85–1.1)")

    n_envs = min(len(merged), 16)
    rel_out = os.path.relpath(str(args.output), str(_REPO_ROOT / "PHC"))
    print(f"""
PHC inference command (from PHC/ directory):

  python phc/run_hydra.py \\
      learning=im_pnn exp_name=phc_kp_pnn_iccv \\
      epoch=-1 test=True im_eval=True \\
      env=env_im_pnn \\
      robot.freeze_hand=True robot.box_body=False env.obs_v=7 \\
      env.motion_file={rel_out} \\
      env.num_prim=4 \\
      env.num_envs={n_envs} \\
      env.enableEarlyTermination=False \\
      ++collect_dataset=True \\
      headless=True
""")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert InterX GT motions to PHC pkl (resume-safe)"
    )
    parser.add_argument(
        "--motions-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "InterX" / "motions",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "InterX" / "splits",
    )
    parser.add_argument(
        "--smpl-data-dir",
        type=Path,
        default=_REPO_ROOT / "PHC" / "data" / "smpl",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=_REPO_ROOT / "PHC" / "output" / "interx_gt" / "staging",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "PHC" / "output" / "interx_gt" / "test.pkl",
        help="Final merged pkl path for PHC",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train", "val", "all"],
        default="test",
    )
    parser.add_argument(
        "--clip-ids",
        type=str,
        default=None,
        help="Comma-separated clip IDs (overrides --split)",
    )
    parser.add_argument(
        "--subprocess-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Process in subprocess batches of N to avoid OOM (recommended: 200–300)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip conversion, only merge existing staging files",
    )
    parser.add_argument(
        "--merge-only-skip",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    if args.merge_only_skip:
        convert_step(args)
        return

    if not args.merge_only:
        if args.subprocess_batch_size:
            subprocess_convert_step(args)
        else:
            convert_step(args)

    merge_step(args)


if __name__ == "__main__":
    main()
