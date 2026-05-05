#!/usr/bin/env python3
"""Batch-convert all InterHuman clips to a single merged PHC pkl.

Saves one small pkl per clip to a staging directory (resume-safe, low memory),
then merges them into the final combined pkl that PHC's motion library can load.

Each clip's two persons become separate entries keyed as ``{clip_id}_p0``
and ``{clip_id}_p1``.

Usage (from repo root, inside Docker):
    # Full dataset (auto-resumes if interrupted):
    python prepare7/batch_convert_interhuman_to_phc.py

    # Full dataset in subprocess batches of 200 clips (avoids OOM for large datasets):
    python prepare7/batch_convert_interhuman_to_phc.py --subprocess-batch-size 200

    # Specific clips only:
    python prepare7/batch_convert_interhuman_to_phc.py --clip-ids 10,11,12

    # Only merge already-converted staging files (skip conversion):
    python prepare7/batch_convert_interhuman_to_phc.py --merge-only

    # Custom paths:
    python prepare7/batch_convert_interhuman_to_phc.py \\
        --interhuman-dir  InterHuman_dataset/motions \\
        --smpl-data-dir   PHC/data/smpl \\
        --staging-dir     PHC/output/interhuman/staging \\
        --output          PHC/output/interhuman/all_clips.pkl
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

# --------------------------------------------------------------------------- #
# Locate interhuman_to_phc.py (repo root) and import convert_person           #
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from interhuman_to_phc import convert_person  # noqa: E402


def load_raw_pkl(pkl_path: Path) -> tuple:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    fps_raw = float(raw.get("mocap_framerate", 59.94))
    step = round(fps_raw / 30)
    fps = 30

    persons = {}
    for key_in, key_out in [("person1", "p0"), ("person2", "p1")]:
        if key_in not in raw:
            continue
        persons[key_out] = {
            k: v[::step]
            if hasattr(v, "__len__") and len(np.shape(v)) >= 1 and np.shape(v)[0] > 1
            else v
            for k, v in raw[key_in].items()
        }
    return persons, fps


def convert_step(args):
    """Convert each clip to its own small pkl in the staging directory."""
    smpl_dir = str(args.smpl_data_dir)
    os.makedirs(args.staging_dir, exist_ok=True)

    if args.clip_ids:
        clip_ids = args.clip_ids.split(",")
    else:
        clip_ids = sorted(
            [p.stem for p in args.interhuman_dir.glob("*.pkl")],
            key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
        )

    print(f"[*] {len(clip_ids)} clips to convert → staging: {args.staging_dir}")

    errors = []
    skipped = 0
    converted = 0

    for clip_id in tqdm(clip_ids, desc="Converting"):
        staging_path = args.staging_dir / f"{clip_id}.pkl"
        if staging_path.exists() and not args.force:
            skipped += 1
            continue

        pkl_path = args.interhuman_dir / f"{clip_id}.pkl"
        if not pkl_path.exists():
            tqdm.write(f"  [warn] missing: {pkl_path}")
            errors.append(clip_id)
            continue

        try:
            persons, fps = load_raw_pkl(pkl_path)
            clip_data = {}
            for person_key, person_data in persons.items():
                clip_data[f"{clip_id}_{person_key}"] = convert_person(
                    person_data, smpl_dir, fps
                )
            # No compression — faster and avoids memory spike during compress
            joblib.dump(clip_data, str(staging_path), compress=0)
            converted += 1
        except Exception as exc:
            tqdm.write(f"  [error] {clip_id}: {exc}")
            errors.append(clip_id)
        finally:
            # Force Python to release memory before the next clip
            try:
                del clip_data, persons
            except NameError:
                pass
            gc.collect()

    print(f"\n[convert] done — converted={converted}  skipped={skipped}  errors={len(errors)}")
    if errors:
        print(f"  failed clips: {errors}")
    return errors


def subprocess_convert_step(args):
    """Split clip list into batches and run each batch in a fresh subprocess.

    Each subprocess terminates after its batch, fully freeing Python/SMPL memory
    at the OS level. This prevents cumulative memory growth that causes OOM kills
    when converting thousands of clips sequentially in a single process.
    """
    if args.clip_ids:
        all_clip_ids = args.clip_ids.split(",")
    else:
        all_clip_ids = sorted(
            [p.stem for p in args.interhuman_dir.glob("*.pkl")],
            key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
        )

    # Filter out already-done clips (avoids spawning empty subprocesses)
    if not args.force:
        pending = [
            c for c in all_clip_ids
            if not (args.staging_dir / f"{c}.pkl").exists()
        ]
    else:
        pending = all_clip_ids

    if not pending:
        print("[subprocess-convert] All clips already converted, nothing to do.")
        return []

    batch_size = args.subprocess_batch_size
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    print(
        f"[subprocess-convert] {len(pending)} pending clips → "
        f"{len(batches)} batches of ≤{batch_size}"
    )

    all_errors = []
    for i, batch in enumerate(batches):
        print(f"\n  [batch {i+1}/{len(batches)}]  clips {batch[0]}..{batch[-1]}")
        cmd = [
            sys.executable, __file__,
            "--interhuman-dir", str(args.interhuman_dir),
            "--smpl-data-dir",  str(args.smpl_data_dir),
            "--staging-dir",    str(args.staging_dir),
            "--output",         str(args.output),
            "--clip-ids",       ",".join(batch),
            "--merge-only-skip",   # skip merge inside the subprocess
        ]
        if args.force:
            cmd.append("--force")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [warn] batch {i+1} exited with code {result.returncode}")
            # Don't abort — staging files already saved are resumable
            all_errors.extend(batch)

    return all_errors


def merge_step(args):
    """Merge all staging pkls into one combined pkl for PHC."""
    staging_files = sorted(args.staging_dir.glob("*.pkl"))
    if not staging_files:
        print(f"[merge] No staging files found in {args.staging_dir}")
        return

    if args.output.exists() and not args.force:
        print(f"[merge] Output already exists: {args.output}  (use --force to overwrite)")
        return

    print(f"[merge] Merging {len(staging_files)} staging files → {args.output}")
    os.makedirs(args.output.parent, exist_ok=True)

    merged = {}
    for staging_path in tqdm(staging_files, desc="Merging"):
        try:
            clip_data = joblib.load(str(staging_path))
            merged.update(clip_data)
        except Exception as exc:
            tqdm.write(f"  [warn] skipping {staging_path.name}: {exc}")

    joblib.dump(merged, str(args.output))
    print(f"[merge] {len(merged)} motions saved → {args.output}")

    # Sanity check on first entry
    first_key = next(iter(merged))
    first = merged[first_key]
    root_z = first["root_trans_offset"][:, 2]
    print(f"\nSanity ({first_key}): root Z  min={root_z.min():.3f}  mean={root_z.mean():.3f}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert InterHuman clips to a single PHC pkl (resume-safe)"
    )
    parser.add_argument(
        "--interhuman-dir",
        type=Path,
        default=_REPO_ROOT / "InterHuman_dataset" / "motions",
    )
    parser.add_argument(
        "--smpl-data-dir",
        type=Path,
        default=_REPO_ROOT / "PHC" / "data" / "smpl",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=_REPO_ROOT / "PHC" / "output" / "interhuman" / "staging",
        help="Directory for per-clip intermediate pkls (resume-safe)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "PHC" / "output" / "interhuman" / "all_clips.pkl",
        help="Final merged pkl path for PHC",
    )
    parser.add_argument(
        "--clip-ids",
        type=str,
        default=None,
        help="Comma-separated clip IDs (default: all .pkl files)",
    )
    parser.add_argument(
        "--subprocess-batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Process clips in subprocess batches of N to avoid OOM on large datasets. "
            "Each subprocess terminates after its batch, releasing all memory. "
            "Recommended: 200–500. Default: disabled (single process)."
        ),
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip conversion, only merge existing staging files",
    )
    parser.add_argument(
        "--merge-only-skip",
        action="store_true",
        help=argparse.SUPPRESS,  # internal flag used by subprocess batches
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert / overwrite existing files",
    )
    args = parser.parse_args()

    if args.merge_only_skip:
        # Called internally by subprocess_convert_step — only convert, never merge
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
