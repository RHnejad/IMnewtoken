#!/usr/bin/env python3
"""Compare a known-good AMASS .motion file with our InterHuman .motion file."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "ProtoMotions"))
import torch
import numpy as np

def describe_motion(path, label):
    print(f"\n{'='*60}")
    print(f"  {label}: {path}")
    print(f"{'='*60}")
    d = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(d, dict):
        print(f"Keys: {sorted(d.keys())}")
        for k, v in sorted(d.items()):
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]")
            elif isinstance(v, (int, float, bool)):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: type={type(v)}")

        # Detailed position analysis
        if "rigid_body_pos" in d:
            pos = d["rigid_body_pos"]  # (T, Nb, 3)
            T, Nb, _ = pos.shape
            print(f"\nPosition analysis (T={T}, Nb={Nb}):")
            print(f"  X range: [{pos[:,:,0].min():.4f}, {pos[:,:,0].max():.4f}]")
            print(f"  Y range: [{pos[:,:,1].min():.4f}, {pos[:,:,1].max():.4f}]")
            print(f"  Z range: [{pos[:,:,2].min():.4f}, {pos[:,:,2].max():.4f}]")

            # Frame 0, body 0 (pelvis)
            print(f"\n  Frame 0, Pelvis pos: {pos[0, 0, :].numpy()}")
            # Min height per axis
            for ax, name in enumerate(["X", "Y", "Z"]):
                min_val = pos[:, :, ax].min().item()
                print(f"  Min {name} across all bodies/frames: {min_val:.4f}")

            # Check which axis has height-like range (should be ~0 to ~1.8)
            for ax, name in enumerate(["X", "Y", "Z"]):
                spread = pos[:, :, ax].max().item() - pos[:, :, ax].min().item()
                print(f"  {name} spread: {spread:.4f}")

        if "rigid_body_rot" in d:
            rot = d["rigid_body_rot"]
            print(f"\nRotation (quat) analysis:")
            print(f"  Shape: {rot.shape}")
            # Check if w-last or w-first by looking at frame 0 pelvis
            print(f"  Frame 0, Pelvis quat: {rot[0, 0, :].numpy()}")
            norms = torch.norm(rot[0], dim=-1)
            print(f"  Frame 0, quat norms (should be ~1): min={norms.min():.6f}, max={norms.max():.6f}")
    else:
        print(f"Type: {type(d)}")
        print(f"Has to_dict: {hasattr(d, 'to_dict')}")

if __name__ == "__main__":
    amass_path = "prepare7/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion"
    ih_path = "prepare7/data/interhuman_test/1_person1.motion"

    if len(sys.argv) > 1:
        amass_path = sys.argv[1]
    if len(sys.argv) > 2:
        ih_path = sys.argv[2]

    describe_motion(amass_path, "AMASS reference")
    describe_motion(ih_path, "InterHuman converted")
