#!/usr/bin/env python3
"""List all clip IDs from the InterX H5 file(s) and write to stdout or a file.

Usage:
    python list_interx_clips.py                                 # uses default paths
    python list_interx_clips.py --h5 data/Inter-X_Dataset/processed/inter-x.h5
    python list_interx_clips.py --output output/mesh_contact/logs/interx_all_clips.txt
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, default=None,
                        help="Explicit H5 path. If not given, searches default locations.")
    parser.add_argument("--data-root", type=str,
                        default="data/Inter-X_Dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Write clip IDs to this file (default: stdout)")
    args = parser.parse_args()

    import h5py

    if args.h5:
        h5_files = [args.h5]
    else:
        candidates = [
            os.path.join(args.data_root, "processed", "inter-x.h5"),
            os.path.join(args.data_root, "processed", "motions", "train.h5"),
            os.path.join(args.data_root, "processed", "motions", "val.h5"),
            os.path.join(args.data_root, "processed", "motions", "test.h5"),
        ]
        h5_files = [p for p in candidates if os.path.isfile(p)]

    all_keys = set()
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as hf:
            keys = list(hf.keys())
            print(f"# {h5_path}: {len(keys)} clips", file=sys.stderr)
            all_keys.update(keys)

    sorted_keys = sorted(all_keys)
    print(f"# Total unique clips: {len(sorted_keys)}", file=sys.stderr)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            for k in sorted_keys:
                f.write(k + "\n")
        print(f"# Written to {args.output}", file=sys.stderr)
    else:
        for k in sorted_keys:
            print(k)


if __name__ == "__main__":
    main()
