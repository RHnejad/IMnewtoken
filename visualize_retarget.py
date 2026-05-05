#!/usr/bin/env python3
"""Visualize a clip from a PHC retarget pkl using body_pos (T x 24 x 3).

Usage:
    python visualize_retarget.py retarget_<timestamp>.pkl
    python visualize_retarget.py retarget_<timestamp>.pkl --clip 1000_p0
    python visualize_retarget.py retarget_<timestamp>.pkl --index 5
    python visualize_retarget.py retarget_<timestamp>.pkl --list
"""

import argparse
import sys

import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SMPL parent→child joint connections
EDGES = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),
    (7,10),(8,11),(9,12),(9,13),(9,14),(12,15),(13,16),(14,17),
    (16,18),(17,19),(18,20),(19,21),(20,22),(21,23),
]

LEFT_JOINTS  = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT_JOINTS = {2, 5, 8, 11, 14, 17, 19, 21, 23}


def parse_args():
    p = argparse.ArgumentParser(description="Animate PHC retarget body_pos")
    p.add_argument("pkl", help="Path to retarget_<timestamp>.pkl")
    p.add_argument("--clip", default=None, help="Motion key to visualize (e.g. 1000_p0)")
    p.add_argument("--index", type=int, default=0, help="Clip index (default: 0)")
    p.add_argument("--list", action="store_true", help="List all available clip keys and exit")
    p.add_argument("--fps", type=float, default=30.0, help="Playback fps (default: 30)")
    p.add_argument("--save", default=None, help="Save animation to this file (e.g. out.gif)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.pkl} ...")
    data = joblib.load(args.pkl)
    keys = list(data["key_names"])

    if args.list:
        for i, k in enumerate(keys):
            bp = data["body_pos"][i]
            print(f"  [{i:4d}]  {k}  (T={bp.shape[0]})")
        return

    if args.clip is not None:
        if args.clip not in keys:
            print(f"[error] Key '{args.clip}' not found. Use --list to see available keys.")
            sys.exit(1)
        idx = keys.index(args.clip)
    else:
        idx = args.index

    key = keys[idx]
    body_pos = data["body_pos"][idx]   # (T, 24, 3)
    T = body_pos.shape[0]
    print(f"Animating '{key}'  ({T} frames @ {args.fps} fps)")

    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Compute fixed axis limits from full sequence
    pad = 0.3
    xmin, xmax = body_pos[:, :, 0].min() - pad, body_pos[:, :, 0].max() + pad
    ymin, ymax = body_pos[:, :, 1].min() - pad, body_pos[:, :, 1].max() + pad
    zmin, zmax = max(0, body_pos[:, :, 2].min() - pad), body_pos[:, :, 2].max() + pad

    def update(t):
        ax.cla()
        joints = body_pos[t]   # (24, 3)

        for i, j in EDGES:
            color = "blue" if i in LEFT_JOINTS or j in LEFT_JOINTS else \
                    "red"  if i in RIGHT_JOINTS or j in RIGHT_JOINTS else "gray"
            ax.plot([joints[i,0], joints[j,0]],
                    [joints[i,1], joints[j,1]],
                    [joints[i,2], joints[j,2]], color=color, linewidth=2)

        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=15, c="black", zorder=5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"{key}   frame {t+1}/{T}")

    if args.save:
        from PIL import Image
        from tqdm import tqdm as _tqdm
        frames_img = []
        print(f"Rendering {T} frames ...")
        for t in _tqdm(range(T)):
            update(t)
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf)
            frames_img.append(img.convert("RGB"))
        plt.close(fig)
        duration_ms = int(1000.0 / args.fps)
        frames_img[0].save(
            args.save,
            save_all=True,
            append_images=frames_img[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"Saved to {args.save}")
    else:
        anim = FuncAnimation(fig, update, frames=T, interval=1000.0 / args.fps, blit=False)
        plt.show()


if __name__ == "__main__":
    main()
