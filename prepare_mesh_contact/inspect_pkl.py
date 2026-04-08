#!/usr/bin/env python3
"""Inspect InterHuman .pkl files to understand their structure and beta shapes.

Usage:
    python prepare_mesh_contact/inspect_pkl.py
    python prepare_mesh_contact/inspect_pkl.py --clip 7605
    python prepare_mesh_contact/inspect_pkl.py --clip 1000 --data-root data/InterHuman
"""
import argparse
import os
import pickle
import sys
import numpy as np


def inspect_pkl(pkl_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"File: {pkl_path}")
    print(f"{'='*60}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    print(f"Type: {type(raw).__name__}")

    if isinstance(raw, dict):
        print(f"Top-level keys: {list(raw.keys())}")
        for key in sorted(raw.keys()):
            val = raw[key]
            if isinstance(val, dict):
                print(f"\n  [{key}] (dict) keys: {list(val.keys())}")
                for k2 in sorted(val.keys()):
                    v2 = val[k2]
                    if isinstance(v2, np.ndarray):
                        print(f"    {k2:20s}  shape={v2.shape}  dtype={v2.dtype}  "
                              f"min={v2.min():.4f}  max={v2.max():.4f}  "
                              f"mean={v2.mean():.4f}")
                    elif isinstance(v2, (int, float, str, bool)):
                        print(f"    {k2:20s}  = {v2}")
                    else:
                        print(f"    {k2:20s}  type={type(v2).__name__}")
            elif isinstance(val, np.ndarray):
                print(f"\n  [{key}]  shape={val.shape}  dtype={val.dtype}")
            elif isinstance(val, (int, float, str, bool)):
                print(f"\n  [{key}]  = {val}")
            else:
                print(f"\n  [{key}]  type={type(val).__name__}")
    else:
        print(f"  Not a dict, type={type(raw).__name__}")
        if hasattr(raw, '__len__'):
            print(f"  Length: {len(raw)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=str, default=None,
                        help="Specific clip ID to inspect")
    parser.add_argument("--data-root", type=str, default="data/InterHuman")
    parser.add_argument("--num-random", type=int, default=5,
                        help="Number of random clips to inspect if --clip not given")
    args = parser.parse_args()

    if args.clip:
        clips = [args.clip]
    else:
        motions_dir = os.path.join(args.data_root, "motions")
        all_pkls = [f.replace(".pkl", "") for f in os.listdir(motions_dir) if f.endswith(".pkl")]
        all_pkls.sort()
        # Pick a spread: first, last, and random
        n = min(args.num_random, len(all_pkls))
        indices = np.linspace(0, len(all_pkls)-1, n, dtype=int)
        clips = [all_pkls[i] for i in indices]
        print(f"Inspecting {n} clips from {len(all_pkls)} total: {clips}")

    for clip_id in clips:
        pkl_path = os.path.join(args.data_root, "motions", f"{clip_id}.pkl")
        if not os.path.isfile(pkl_path):
            print(f"NOT FOUND: {pkl_path}")
            continue
        inspect_pkl(pkl_path)

    # Summary: check betas consistency across ALL clips
    print(f"\n{'='*60}")
    print("Beta shapes survey (all clips):")
    print(f"{'='*60}")
    motions_dir = os.path.join(args.data_root, "motions")
    all_pkls = sorted([f for f in os.listdir(motions_dir) if f.endswith(".pkl")])

    betas_shapes = {}
    betas_all_zero_count = 0
    betas_nonzero_count = 0
    errors = []

    for fname in all_pkls:
        try:
            with open(os.path.join(motions_dir, fname), "rb") as f:
                raw = pickle.load(f)
            for pkey in ("person1", "person2"):
                if pkey in raw and isinstance(raw[pkey], dict):
                    if "betas" in raw[pkey]:
                        b = np.asarray(raw[pkey]["betas"])
                        shape_key = str(b.shape)
                        betas_shapes[shape_key] = betas_shapes.get(shape_key, 0) + 1
                        if np.allclose(b, 0):
                            betas_all_zero_count += 1
                        else:
                            betas_nonzero_count += 1
                    else:
                        betas_shapes["MISSING"] = betas_shapes.get("MISSING", 0) + 1
        except Exception as e:
            errors.append((fname, str(e)))

    print(f"  Betas shape distribution:")
    for shape, count in sorted(betas_shapes.items(), key=lambda x: -x[1]):
        print(f"    {shape:30s}  count={count}")
    print(f"  All-zero betas: {betas_all_zero_count}")
    print(f"  Non-zero betas: {betas_nonzero_count}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for fname, err in errors[:10]:
            print(f"    {fname}: {err}")

    # ── NaN/Inf/sanity survey across all params ──────────────────
    print(f"\n{'='*60}")
    print("Parameter sanity survey (all clips):")
    print(f"{'='*60}")
    nan_clips = []
    inf_clips = []
    bad_pose_clips = []
    param_keys = ("trans", "root_orient", "pose_body", "betas")

    for fname in all_pkls:
        clip_id = fname.replace(".pkl", "")
        try:
            with open(os.path.join(motions_dir, fname), "rb") as f:
                raw = pickle.load(f)
            for pkey in ("person1", "person2"):
                if pkey not in raw or not isinstance(raw[pkey], dict):
                    continue
                d = raw[pkey]
                for pk in param_keys:
                    if pk not in d:
                        continue
                    arr = np.asarray(d[pk])
                    if np.isnan(arr).any():
                        nan_clips.append((clip_id, pkey, pk))
                    if np.isinf(arr).any():
                        inf_clips.append((clip_id, pkey, pk))
                # Check pose_body range: axis-angle values > pi (3.14) are suspicious
                if "pose_body" in d:
                    pb = np.asarray(d["pose_body"])
                    max_val = np.abs(pb).max()
                    if max_val > 10.0:  # axis-angle rarely exceeds ~3.14
                        bad_pose_clips.append((clip_id, pkey, f"pose_body max={max_val:.2f}"))
                # Check trans range: extreme values (>100m) are suspicious
                if "trans" in d:
                    tr = np.asarray(d["trans"])
                    max_tr = np.abs(tr).max()
                    if max_tr > 100.0:
                        bad_pose_clips.append((clip_id, pkey, f"trans max={max_tr:.2f}"))
        except Exception:
            pass

    print(f"  Clips with NaN: {len(nan_clips)}")
    for clip_id, pkey, pk in nan_clips[:10]:
        print(f"    {clip_id} {pkey} {pk}")
    print(f"  Clips with Inf: {len(inf_clips)}")
    for clip_id, pkey, pk in inf_clips[:10]:
        print(f"    {clip_id} {pkey} {pk}")
    print(f"  Clips with suspicious values: {len(bad_pose_clips)}")
    for clip_id, pkey, msg in bad_pose_clips[:10]:
        print(f"    {clip_id} {pkey}: {msg}")


if __name__ == "__main__":
    main()
