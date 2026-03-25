"""
Single-clip skyhook debug runner: compute + visualize in one command.

Workflow:
  1) Optionally recompute skyhook metrics for one clip/person using
     prepare2/compute_skyhook_metrics.py
  2) Generate per-frame CSV + diagnostic plots from *_skyhook_metrics.npz

Designed for quick instability diagnosis (e.g., exploding root forces).
"""
import os
import sys
import csv
import json
import argparse
import subprocess
import shutil
from typing import List, Optional

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPUTE_SCRIPT = os.path.join(PROJECT_ROOT, "prepare2", "compute_skyhook_metrics.py")


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _discover_persons(clip_id: str, out_dir: str, person_arg: Optional[int]) -> List[int]:
    if person_arg is not None:
        return [person_arg]
    found = []
    for p in (0, 1):
        npz_path = os.path.join(out_dir, f"{clip_id}_person{p}_skyhook_metrics.npz")
        if os.path.exists(npz_path):
            found.append(p)
    return found


def _run_compute(args, out_dir: str):
    cmd = [
        sys.executable,
        COMPUTE_SCRIPT,
        "--dataset",
        args.dataset,
        "--clip",
        args.clip,
        "--output-dir",
        out_dir,
        "--input",
        args.input,
        "--fps",
        str(args.fps),
        "--downsample",
        str(args.downsample),
        "--diff-method",
        args.diff_method,
        "--gpu",
        args.gpu,
        "--force",
    ]
    if args.person is not None:
        cmd.extend(["--person", str(args.person)])
    if args.disable_mpjpe:
        cmd.append("--disable-mpjpe")
    if args.ground_fix:
        cmd.append("--ground-fix")
        cmd.extend(["--ground-clearance", str(args.ground_clearance)])
    if args.trim_edges > 0:
        cmd.extend(["--trim-edges", str(args.trim_edges)])
    if args.balance_hold_frames > 0:
        cmd.extend(["--balance-hold-frames", str(args.balance_hold_frames)])
    if args.balance_transition_frames > 0:
        cmd.extend(["--balance-transition-frames", str(args.balance_transition_frames)])
    if abs(float(args.balance_knee_bend_deg) - 20.0) > 1e-6:
        cmd.extend(["--balance-knee-bend-deg", str(args.balance_knee_bend_deg)])

    print("Running compute step:")
    print("  " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _write_frame_csv(clip_id: str, person_idx: int, out_dir: str, npz_data):
    csv_path = os.path.join(out_dir, f"{clip_id}_person{person_idx}_skyhook_per_frame.csv")
    frame = np.asarray(npz_data["frame"]).astype(np.int32)
    root_f_xyz = np.asarray(npz_data["root_force_xyz_N"])
    root_t_xyz = np.asarray(npz_data["root_torque_xyz_Nm"])
    root_f_l2 = np.asarray(npz_data["root_force_l2"])
    root_t_l2 = np.asarray(npz_data["root_torque_l2"])
    root_w_l2 = np.asarray(npz_data["root_wrench_l2"])
    mpjpe = np.asarray(npz_data["mpjpe_per_frame_m"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "root_fx_N",
                "root_fy_N",
                "root_fz_N",
                "root_force_l2_N",
                "root_tx_Nm",
                "root_ty_Nm",
                "root_tz_Nm",
                "root_torque_l2_Nm",
                "root_wrench_l2",
                "mpjpe_m",
            ]
        )
        for i in range(frame.shape[0]):
            writer.writerow(
                [
                    int(frame[i]),
                    float(root_f_xyz[i, 0]),
                    float(root_f_xyz[i, 1]),
                    float(root_f_xyz[i, 2]),
                    float(root_f_l2[i]),
                    float(root_t_xyz[i, 0]),
                    float(root_t_xyz[i, 1]),
                    float(root_t_xyz[i, 2]),
                    float(root_t_l2[i]),
                    float(root_w_l2[i]),
                    float(mpjpe[i]) if np.isfinite(mpjpe[i]) else "",
                ]
            )

    return csv_path


def _plot_person(clip_id: str, person_idx: int, out_dir: str, npz_data, meta: dict):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipping plot for person {person_idx}: {exc}")
        return None

    frame = np.asarray(npz_data["frame"]).astype(np.int32)
    root_f_xyz = np.asarray(npz_data["root_force_xyz_N"])
    root_t_xyz = np.asarray(npz_data["root_torque_xyz_Nm"])
    root_f_l2 = np.asarray(npz_data["root_force_l2"])
    root_t_l2 = np.asarray(npz_data["root_torque_l2"])
    root_w_l2 = np.asarray(npz_data["root_wrench_l2"])
    mpjpe = np.asarray(npz_data["mpjpe_per_frame_m"])

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"Skyhook Diagnostics — clip {clip_id} person {person_idx}",
        fontsize=14,
    )

    # Root force components
    axes[0].plot(frame, root_f_xyz[:, 0], label="Fx", linewidth=1.0)
    axes[0].plot(frame, root_f_xyz[:, 1], label="Fy", linewidth=1.0)
    axes[0].plot(frame, root_f_xyz[:, 2], label="Fz", linewidth=1.0)
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Root Force Components")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    # Root force norm
    axes[1].plot(frame, root_f_l2, color="tab:red", linewidth=1.2, label="||Froot||")
    axes[1].set_ylabel("N")
    axes[1].set_title("Root Force L2 (linear)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    # Root force norm (log)
    axes[2].plot(frame, np.maximum(root_f_l2, 1e-12), color="tab:red", linewidth=1.2)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("N (log)")
    axes[2].set_title("Root Force L2 (log scale)")
    axes[2].grid(alpha=0.25, which="both")

    # Root torque / wrench
    axes[3].plot(frame, root_t_xyz[:, 0], label="Tx", linewidth=0.9)
    axes[3].plot(frame, root_t_xyz[:, 1], label="Ty", linewidth=0.9)
    axes[3].plot(frame, root_t_xyz[:, 2], label="Tz", linewidth=0.9)
    axes[3].plot(frame, root_t_l2, label="||Troot||", linewidth=1.2, color="tab:purple")
    axes[3].plot(frame, root_w_l2, label="||Wroot||", linewidth=1.2, color="tab:orange")
    axes[3].set_ylabel("Nm / mixed")
    axes[3].set_title("Root Torque / Wrench")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.25)

    # MPJPE
    valid = np.isfinite(mpjpe)
    if valid.any():
        axes[4].plot(frame[valid], mpjpe[valid], color="tab:green", linewidth=1.2)
    axes[4].set_ylabel("m")
    axes[4].set_xlabel("Frame")
    axes[4].set_title("MPJPE per frame")
    axes[4].grid(alpha=0.25)

    # Annotate top force spike
    if root_f_l2.size > 0:
        i_peak = int(np.argmax(root_f_l2))
        peak_val = float(root_f_l2[i_peak])
        axes[1].axvline(frame[i_peak], color="black", alpha=0.3, linestyle="--")
        axes[1].text(
            frame[i_peak],
            peak_val,
            f" peak@{frame[i_peak]}: {peak_val:.3e} N",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    txt = (
        f"mean||F||={meta.get('skyhook_residual_force_mean_N', float('nan')):.3e} N, "
        f"max||F||={meta.get('skyhook_residual_force_max_N', float('nan')):.3e} N, "
        f"mean MPJPE={meta.get('mpjpe_mean_m', float('nan')):.3e} m"
    )
    fig.text(0.01, 0.005, txt, fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    png_path = os.path.join(out_dir, f"{clip_id}_person{person_idx}_skyhook_diagnostic.png")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


def _plot_combined(clip_id: str, out_dir: str, person_data: dict):
    if not person_data:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Skyhook Comparison — clip {clip_id}", fontsize=14)

    for p, data in sorted(person_data.items()):
        frame = data["frame"]
        rf = data["root_force_l2"]
        rw = data["root_wrench_l2"]
        axes[0].plot(frame, rf, linewidth=1.2, label=f"person {p}")
        axes[1].plot(frame, np.maximum(rw, 1e-12), linewidth=1.2, label=f"person {p}")

    axes[0].set_title("Root Force L2")
    axes[0].set_ylabel("N")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].set_title("Root Wrench L2 (log)")
    axes[1].set_ylabel("mixed (log)")
    axes[1].set_xlabel("Frame")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, f"{clip_id}_skyhook_comparison.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compute + visualize skyhook diagnostics for one clip."
    )
    parser.add_argument("--dataset", default="interhuman", choices=["interhuman", "interx"])
    parser.add_argument("--clip", default="1000")
    parser.add_argument("--person", type=int, default=None, choices=[0, 1])
    parser.add_argument("--output-dir", default="data/test/skyhook")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional directory with existing *_skyhook_metrics.* files to reuse/copy.",
    )
    parser.add_argument("--input", default="retargeted", choices=["retargeted", "dataset", "auto"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--diff-method", default="spline", choices=["spline", "fd"])
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--disable-mpjpe", action="store_true")
    parser.add_argument("--ground-fix", action="store_true")
    parser.add_argument("--ground-clearance", type=float, default=0.01)
    parser.add_argument("--trim-edges", type=int, default=0)
    parser.add_argument("--balance-hold-frames", type=int, default=0)
    parser.add_argument("--balance-transition-frames", type=int, default=0)
    parser.add_argument("--balance-knee-bend-deg", type=float, default=20.0)
    parser.add_argument(
        "--no-recompute",
        action="store_true",
        help="Skip compute step and only visualize existing NPZ files.",
    )
    args = parser.parse_args()

    out_dir = _resolve(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    source_dir = _resolve(args.source_dir) if args.source_dir else None

    compute_rc = None
    if not args.no_recompute:
        compute_rc = _run_compute(args, out_dir)
        if compute_rc != 0:
            print(
                f"Compute step failed with code {compute_rc}. "
                "Will still attempt visualization from existing files."
            )

    persons = _discover_persons(args.clip, out_dir, args.person)
    if not persons and source_dir is not None and os.path.isdir(source_dir):
        target_persons = [args.person] if args.person is not None else [0, 1]
        copied = 0
        for p in target_persons:
            for ext in ("npz", "json"):
                src = os.path.join(source_dir, f"{args.clip}_person{p}_skyhook_metrics.{ext}")
                dst = os.path.join(out_dir, f"{args.clip}_person{p}_skyhook_metrics.{ext}")
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1
        if copied > 0:
            print(f"Copied {copied} existing metric files from {source_dir}")
        persons = _discover_persons(args.clip, out_dir, args.person)

    if not persons:
        raise FileNotFoundError(
            f"No skyhook NPZ files found for clip {args.clip} in {out_dir}"
        )

    person_plot_paths = []
    person_data_for_combined = {}
    run_summary = {
        "dataset": args.dataset,
        "clip": args.clip,
        "output_dir": out_dir,
        "persons": persons,
        "compute_return_code": compute_rc,
        "plots": [],
        "csv_files": [],
        "per_person": {},
    }

    for p in persons:
        npz_path = os.path.join(out_dir, f"{args.clip}_person{p}_skyhook_metrics.npz")
        json_path = os.path.join(out_dir, f"{args.clip}_person{p}_skyhook_metrics.json")
        if not os.path.exists(npz_path):
            print(f"Missing NPZ for person {p}: {npz_path}")
            continue

        with np.load(npz_path) as d:
            arrays = {
                "frame": np.asarray(d["frame"]),
                "root_force_l2": np.asarray(d["root_force_l2"]),
                "root_wrench_l2": np.asarray(d["root_wrench_l2"]),
            }
            person_data_for_combined[p] = arrays

            csv_path = _write_frame_csv(args.clip, p, out_dir, d)
            run_summary["csv_files"].append(csv_path)

            meta = {}
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            plot_path = _plot_person(args.clip, p, out_dir, d, meta)
            if plot_path:
                person_plot_paths.append(plot_path)
                run_summary["plots"].append(plot_path)

            run_summary["per_person"][str(p)] = {
                "npz_path": npz_path,
                "json_path": json_path if os.path.exists(json_path) else None,
                "n_frames": int(arrays["frame"].shape[0]),
                "mean_root_force_l2_N": float(np.mean(arrays["root_force_l2"])),
                "max_root_force_l2_N": float(np.max(arrays["root_force_l2"])),
            }

    combined = _plot_combined(args.clip, out_dir, person_data_for_combined)
    if combined:
        run_summary["plots"].append(combined)

    summary_path = os.path.join(out_dir, f"{args.clip}_skyhook_debug_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\nSaved skyhook debug outputs:")
    print(f"  Summary: {summary_path}")
    for p in sorted(run_summary["per_person"].keys()):
        info = run_summary["per_person"][p]
        print(
            f"  person {p}: n={info['n_frames']} "
            f"mean||F||={info['mean_root_force_l2_N']:.3e} "
            f"max||F||={info['max_root_force_l2_N']:.3e}"
        )
    for p in run_summary["plots"]:
        print(f"  plot: {p}")
    for c in run_summary["csv_files"]:
        print(f"  csv: {c}")


if __name__ == "__main__":
    main()
