#!/usr/bin/env python3
"""Batch-convert generated InterMask/InterGen motion predictions to PHC pkl format.

Three sets are supported (all stored under data/generated/ in the repo):

  data/generated/interhuman/                  InterMask predictions for InterHuman
  data/generated/interx/pkl/                  InterMask predictions for InterX
  data/generated/intergen_interhuman_evalpolicy/  InterGen predictions for InterHuman

Each source pkl stores one dyadic clip with person1/person2 dicts containing:
  root_orient  (T, 3)   axis-angle, Z-up
  pose_body    (T, 63)  21-joint axis-angle
  trans        (T, 3)   root translation, Z-up, metres
  betas        (10,)    SMPL shape params
  positions_zup (T, 22, 3)  joint positions (not used by PHC converter)

InterMask files encode arrays as JSON-compatible __ndarray__ dicts; InterGen files
store plain numpy arrays.  Generated motions are already at 30 fps — no resampling.

Output per dataset:
  PHC/output/generated/{dataset_tag}/staging/{clip_name}.pkl  (per-clip, resume-safe)
  PHC/output/generated/{dataset_tag}.pkl                      (merged, ready for PHC)

Motion keys in merged pkl: {clip_name}_p0 / {clip_name}_p1

Usage (from repo root, inside Docker):
    # All three datasets, subprocess-batched to avoid OOM:
    python prepare7/batch_convert_generated_to_phc.py --subprocess-batch-size 300

    # Single dataset only:
    python prepare7/batch_convert_generated_to_phc.py --dataset interhuman_intermask

    # Merge only (skip conversion):
    python prepare7/batch_convert_generated_to_phc.py --merge-only
"""

import argparse
import gc
import os
import pickle
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from interhuman_to_phc import convert_person  # noqa: E402

# --------------------------------------------------------------------------- #
# Dataset registry                                                              #
# --------------------------------------------------------------------------- #
DATASETS = {
    "interhuman_intermask": {
        "source_dir": _REPO_ROOT / "data" / "generated" / "interhuman",
        "decode_ndarray": True,
        "fps": 30,
        "description": "InterHuman / InterMask predictions",
    },
    "interx_intermask": {
        "source_dir": _REPO_ROOT / "data" / "generated" / "interx" / "pkl",
        "decode_ndarray": True,
        "fps": 30,
        "description": "InterX / InterMask predictions",
    },
    "interhuman_intergen": {
        "source_dir": _REPO_ROOT / "data" / "generated" / "intergen_interhuman_evalpolicy",
        "decode_ndarray": False,
        "fps": 30,
        "description": "InterHuman / InterGen predictions",
    },
}

_GENERATED_OUT = _REPO_ROOT / "PHC" / "output" / "generated"


# --------------------------------------------------------------------------- #
# Array decoding                                                                #
# --------------------------------------------------------------------------- #

def _decode(obj):
    """Decode InterMask's JSON-serialised ndarray → numpy array."""
    if isinstance(obj, dict) and "__ndarray__" in obj:
        return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


def load_persons(pkl_path: Path, decode: bool) -> tuple:
    """Return (persons_dict, clip_name) from a generated prediction pkl."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    clip_name = raw.get("clip_name", pkl_path.stem)

    persons = {}
    for key_in, key_out in [("person1", "p0"), ("person2", "p1")]:
        if key_in not in raw:
            continue
        pd = raw[key_in]
        if decode:
            pd = {k: _decode(v) for k, v in pd.items()}
        # Ensure float64 numpy arrays (convert_person expects this)
        pd = {
            k: np.asarray(v, dtype=np.float64) if isinstance(v, np.ndarray) else v
            for k, v in pd.items()
        }
        persons[key_out] = pd

    return persons, clip_name


# --------------------------------------------------------------------------- #
# Conversion step                                                               #
# --------------------------------------------------------------------------- #

def convert_step(args, dataset_tag: str):
    cfg = DATASETS[dataset_tag]
    source_dir: Path = cfg["source_dir"]
    decode: bool = cfg["decode_ndarray"]
    fps: int = cfg["fps"]
    smpl_dir = str(args.smpl_data_dir)

    staging_dir = _GENERATED_OUT / dataset_tag / "staging"
    os.makedirs(staging_dir, exist_ok=True)

    if args.clip_ids:
        clip_ids = args.clip_ids.split(",")
        pkl_paths = [source_dir / f"{c}.pkl" for c in clip_ids]
    else:
        pkl_paths = sorted(source_dir.glob("*.pkl"))

    print(f"[{dataset_tag}] {len(pkl_paths)} clips → staging: {staging_dir}")

    errors, skipped, converted = [], 0, 0

    for pkl_path in tqdm(pkl_paths, desc=f"Converting {dataset_tag}"):
        staging_path = staging_dir / pkl_path.name
        if staging_path.exists() and not args.force:
            skipped += 1
            continue
        if not pkl_path.exists():
            tqdm.write(f"  [warn] missing: {pkl_path}")
            errors.append(pkl_path.stem)
            continue

        try:
            persons, clip_name = load_persons(pkl_path, decode)
            clip_data = {}
            for person_key, person_data in persons.items():
                clip_data[f"{clip_name}_{person_key}"] = convert_person(
                    person_data, smpl_dir, fps
                )
            joblib.dump(clip_data, str(staging_path), compress=0)
            converted += 1
        except Exception as exc:
            tqdm.write(f"  [error] {pkl_path.stem}: {exc}")
            errors.append(pkl_path.stem)
        finally:
            try:
                del clip_data, persons
            except NameError:
                pass
            gc.collect()

    print(
        f"[{dataset_tag}] done — converted={converted}  skipped={skipped}  errors={len(errors)}"
    )
    if errors:
        print(f"  failed: {errors[:10]}{'...' if len(errors) > 10 else ''}")
    return errors


def subprocess_convert_step(args, dataset_tag: str):
    """Split pending clips into subprocess batches to avoid cumulative OOM."""
    cfg = DATASETS[dataset_tag]
    source_dir: Path = cfg["source_dir"]
    staging_dir = _GENERATED_OUT / dataset_tag / "staging"
    os.makedirs(staging_dir, exist_ok=True)

    all_stems = sorted([p.stem for p in source_dir.glob("*.pkl")])
    if not args.force:
        pending = [s for s in all_stems if not (staging_dir / f"{s}.pkl").exists()]
    else:
        pending = all_stems

    if not pending:
        print(f"[{dataset_tag}] All clips already converted.")
        return []

    batch_size = args.subprocess_batch_size
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    print(
        f"[{dataset_tag}] {len(pending)} pending → {len(batches)} batches of ≤{batch_size}"
    )

    all_errors = []
    for i, batch in enumerate(batches):
        print(f"  [batch {i+1}/{len(batches)}]  {batch[0]}..{batch[-1]}")
        cmd = [
            sys.executable, __file__,
            "--smpl-data-dir",      str(args.smpl_data_dir),
            "--dataset",            dataset_tag,
            "--clip-ids",           ",".join(batch),
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
# Merge step                                                                    #
# --------------------------------------------------------------------------- #

def merge_step(args, dataset_tag: str):
    staging_dir = _GENERATED_OUT / dataset_tag / "staging"
    output_pkl = _GENERATED_OUT / f"{dataset_tag}.pkl"

    staging_files = sorted(staging_dir.glob("*.pkl"))
    if not staging_files:
        print(f"[{dataset_tag}] No staging files in {staging_dir}")
        return

    if output_pkl.exists() and not args.force:
        print(f"[{dataset_tag}] Output exists: {output_pkl}  (use --force to overwrite)")
        return

    print(f"[{dataset_tag}] Merging {len(staging_files)} staging files → {output_pkl}")
    os.makedirs(output_pkl.parent, exist_ok=True)

    merged = {}
    for sp in tqdm(staging_files, desc=f"Merging {dataset_tag}"):
        try:
            merged.update(joblib.load(str(sp)))
        except Exception as exc:
            tqdm.write(f"  [warn] skipping {sp.name}: {exc}")

    joblib.dump(merged, str(output_pkl))
    print(f"[{dataset_tag}] {len(merged)} motions saved → {output_pkl}")

    first = next(iter(merged.values()))
    rz = first["root_trans_offset"][:, 2]
    print(f"  Sanity: root Z  min={rz.min():.3f}  mean={rz.mean():.3f}")

    n_envs = min(len(merged), 16)
    rel_out = os.path.relpath(str(output_pkl), str(_REPO_ROOT / "PHC"))
    print(f"""
PHC inference ({dataset_tag}) — from PHC/ directory:

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
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Convert generated InterMask/InterGen predictions to PHC pkl format"
    )
    parser.add_argument(
        "--smpl-data-dir",
        type=Path,
        default=_REPO_ROOT / "PHC" / "data" / "smpl",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to process (default: all)",
    )
    parser.add_argument(
        "--clip-ids",
        type=str,
        default=None,
        help="Comma-separated clip stems (default: all in source directory)",
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
        help=argparse.SUPPRESS,  # internal flag for subprocess batches
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert / overwrite existing files",
    )
    args = parser.parse_args()

    datasets_to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    if args.merge_only_skip:
        # Called from a subprocess batch — only convert, never merge
        for tag in datasets_to_process:
            convert_step(args, tag)
        return

    for tag in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"  {tag}: {DATASETS[tag]['description']}")
        print(f"{'='*60}")

        if not args.merge_only:
            if args.subprocess_batch_size:
                subprocess_convert_step(args, tag)
            else:
                convert_step(args, tag)

        merge_step(args, tag)


if __name__ == "__main__":
    main()
