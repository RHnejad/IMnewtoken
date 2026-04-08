#!/usr/bin/env python3
"""List available InterX clip ids from one or more H5 sources."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prepare_mesh_contact.mesh_contact_pipeline import _interx_h5_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List InterX clips from available H5 files")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "Inter-X_Dataset"),
        help="InterX root containing processed motions H5",
    )
    parser.add_argument("--h5-file", type=str, default=None, help="Optional explicit H5 file")
    parser.add_argument("--count-only", action="store_true", help="Only print total clip count")
    parser.add_argument("--json", action="store_true", help="Print JSON summary instead of plain clip ids")
    parser.add_argument("--limit", type=int, default=None, help="Optional clip id output limit")
    parser.add_argument("--out-file", type=str, default=None, help="Optional text file to write clip ids to")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import h5py

    h5_paths = _interx_h5_candidates(args.data_root, args.h5_file)
    clip_ids: List[str] = []
    seen = set()
    per_file: Dict[str, int] = {}

    for path in h5_paths:
        with h5py.File(path, "r") as hf:
            keys = sorted(hf.keys())
        per_file[path] = len(keys)
        for key in keys:
            if key not in seen:
                seen.add(key)
                clip_ids.append(key)

    clip_ids.sort()
    if args.limit is not None and args.limit > 0:
        clip_ids = clip_ids[: args.limit]

    if args.out_file is not None:
        os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as f:
            for clip_id in clip_ids:
                f.write(f"{clip_id}\n")

    if args.count_only:
        print(len(clip_ids))
        return

    if args.json:
        payload = {
            "data_root": args.data_root,
            "h5_files": h5_paths,
            "per_file_counts": per_file,
            "num_unique_clips": len(clip_ids),
            "clip_ids": clip_ids,
        }
        print(json.dumps(payload, indent=2))
        return

    for clip_id in clip_ids:
        print(clip_id)


if __name__ == "__main__":
    main()
