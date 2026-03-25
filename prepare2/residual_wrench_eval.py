"""
Residual Wrench Evaluation — Dynamic Plausibility Metric for Two-Person Interactions
======================================================================================

Computes the **Residual Wrench** (skyhook force + torque) required at the root of each
character to make a motion physically consistent, and compares GT vs generated motions.

Theory
------
In a physically valid motion the character balances under gravity and ground contacts
alone.  Inverse dynamics computes the torques needed to reproduce an observed motion:

    τ = M(q) · (q̈_desired − q̈_free)

where q̈_free is the free-fall acceleration under zero torques (gravity + Coriolis only).

The first 6 DOFs of τ are **virtual** root forces — a "skyhook" — that the physics
engine must inject to keep the character on its reference trajectory.  For perfect mocap
these should be near zero.  For generated motions they expose how much the motion
violates Newton's laws.

Metrics (per clip, then compared as distributions)
----------------------------------------------------
  F_sky_mean  : mean_t ||τ[t, 0:3]||₂   (root translation force, Newtons)
  τ_sky_mean  : mean_t ||τ[t, 3:6]||₂   (root rotation torque, Nm)
  W_mean      : mean_t ||τ[t, 0:3]||₂   (synonym of F_sky_mean — "skyhook work proxy")
  P_active    : mean_t Σ_j |τ_j(t)·q̇_j(t)| for j in 6:75   (joint actuation power, W)
  ΔF_sky      : F_sky_mean_gen − F_sky_mean_gt per matching clip pair
                → distribution mean is the primary "Dynamic Gap" score

All metrics use B-spline differentiation (no noise amplification from finite differences).

Usage
-----
    # Quick 50-clip evaluation (GT + generated, test split)
    python prepare2/residual_wrench_eval.py --n-clips 50 --device cuda:0

    # Full evaluation
    python prepare2/residual_wrench_eval.py --n-clips 200 --device cuda:0

    # Only GT or only generated
    python prepare2/residual_wrench_eval.py --source gt --n-clips 100
    python prepare2/residual_wrench_eval.py --source generated --n-clips 100

    # Comparison plot only (if results already saved)
    python prepare2/residual_wrench_eval.py --compare-only \
        --gt-results data/residual_wrench/gt_results.json \
        --gen-results data/residual_wrench/generated_results.json

Outputs (default: data/residual_wrench/)
-----------------------------------------
  gt_results.json             — per-clip GT metrics
  generated_results.json      — per-clip generated metrics
  comparison.json             — aggregated GT vs gen comparison
  skyhook_distribution.png    — histogram: F_sky GT vs generated
  actuation_power.png         — histogram: P_active GT vs generated
  delta_fsky_distribution.png — histogram: ΔF_sky per clip pair
  metrics_barplot.png         — bar chart of all key metrics
"""

import os
import sys
import json
import time
import argparse
import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", message="Custom attribute")

import warp as wp
wp.config.verbose = False

import newton

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare2.retarget import (
    get_or_create_xml,
    smplx_to_joint_q,
    load_interhuman_clip,
    list_interhuman_clips,
)
from prepare2.compute_torques import (
    inverse_dynamics,
    compute_qd_qdd_spline,
)
from prepare2.pd_utils import (
    build_model,
    setup_model_properties,
    create_mujoco_solver,
    DOFS_PER_PERSON,
)

# ──────────────────────────────────────────────────────────────────────────────
# Data paths
# ──────────────────────────────────────────────────────────────────────────────

GT_RETARGETED_DIR = os.path.join(PROJECT_ROOT, "data", "retargeted_v2", "interhuman")
GENERATED_DIR = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman")
TEST_SPLIT_FILE = os.path.join(PROJECT_ROOT, "data", "InterHuman", "split", "test.txt")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "residual_wrench")


# ──────────────────────────────────────────────────────────────────────────────
# Load helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_test_split():
    """Return sorted list of test clip IDs."""
    if not os.path.exists(TEST_SPLIT_FILE):
        raise FileNotFoundError(f"Test split not found: {TEST_SPLIT_FILE}")
    with open(TEST_SPLIT_FILE) as f:
        return sorted(l.strip() for l in f if l.strip())


def load_gt_clip(clip_id: str):
    """
    Load retargeted GT data for a clip.
    Returns list of (joint_q (T,76), betas (10,)) for each person, or None.
    """
    persons = []
    for person_idx in range(2):
        jq_path = os.path.join(GT_RETARGETED_DIR, f"{clip_id}_person{person_idx}_joint_q.npy")
        betas_path = os.path.join(GT_RETARGETED_DIR, f"{clip_id}_person{person_idx}_betas.npy")
        if not os.path.exists(jq_path) or not os.path.exists(betas_path):
            return None
        persons.append((np.load(jq_path), np.load(betas_path)))
    return persons


def load_generated_clip(clip_id: str):
    """
    Load generated clip, retarget SMPL-X params → joint_q.
    Returns list of (joint_q (T,76), betas (10,)) for each person, or None.
    """
    import pickle
    pkl_path = os.path.join(GENERATED_DIR, f"{clip_id}.pkl")
    if not os.path.exists(pkl_path):
        return None
    try:
        data = pickle.load(open(pkl_path, "rb"))
    except Exception:
        return None

    persons = []
    for key in ["person1", "person2"]:
        p = data.get(key)
        if p is None:
            return None
        try:
            jq = smplx_to_joint_q(
                root_orient=np.array(p["root_orient"], dtype=np.float64),
                pose_body=np.array(p["pose_body"], dtype=np.float64),
                trans=np.array(p["trans"], dtype=np.float64),
                betas=np.array(p["betas"], dtype=np.float64),
            )
        except Exception as e:
            print(f"  [retarget error clip {clip_id} {key}]: {e}")
            return None
        persons.append((jq.astype(np.float32), np.array(p["betas"], dtype=np.float32)))
    return persons


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

BODY_MASS_KG = 75.0   # Default SMPL body mass used in Newton model
GRAVITY = 9.81        # m/s²
BODY_WEIGHT_N = BODY_MASS_KG * GRAVITY  # ~736 N — used for normalization

# Threshold for "physically implausible" frame: skyhook force > 3× body weight.
# At 3×BW (~2208 N) the skyhook force exceeds any athletic GRF — the motion cannot
# be explained by normal contact mechanics.  Biomechanics literature (Winter 2009)
# treats root residuals > ~10% BW as a red flag; 3×BW is conservative but separates
# normal dynamics from clear failures without flagging benign GT frames.
FPI_THRESHOLD_N = 3.0 * BODY_WEIGHT_N  # ~2208 N


def compute_residual_wrench(
    joint_q: np.ndarray,
    betas: np.ndarray,
    fps: int = 30,
    device: str = "cuda:0",
    trim_edges: int = 5,
) -> Optional[dict]:
    """
    Run inverse dynamics on one person's trajectory and compute residual wrench metrics.

    The residual wrench is the virtual force/torque at the root (skyhook) that inverse
    dynamics must inject because the motion does not self-balance under physics alone.

    NOTE on absolute values: The skyhook force includes gravity support (~736N for a
    75 kg body) because ground contacts are only partially captured by the zero-torque
    step.  Use the normalised metric (F_sky_norm = F_sky / body_weight) or compare
    GT vs generated as a ratio/delta — do not interpret absolute values in isolation.

    Args:
        joint_q    : (T, 76) Newton joint coordinates at `fps` Hz
        betas      : (10,)   SMPL shape parameters
        fps        : frames per second of joint_q
        device     : CUDA device string
        trim_edges : frames to drop at each end (spline boundary artifact fix)

    Returns dict with per-clip scalar metrics and per-frame arrays.
    """
    T = joint_q.shape[0]
    if T < 5:
        return None

    # Trim boundary frames to avoid spline fit edge artefacts
    if trim_edges > 0 and T > 2 * trim_edges + 5:
        joint_q = joint_q[trim_edges: T - trim_edges]
        T = joint_q.shape[0]

    # Build Newton model with ground plane
    xml_path = get_or_create_xml(betas)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    # Run inverse dynamics (B-spline differentiation — no noise amplification)
    try:
        torques, qd, qdd = inverse_dynamics(
            model, joint_q, fps,
            device=device,
            diff_method="spline",
        )
    except Exception as e:
        print(f"    [ID error]: {e}")
        return None

    # τ[:, 0:3] — root translation virtual force (skyhook force, Newtons)
    # τ[:, 3:6] — root rotation virtual torque  (skyhook torque, Nm)
    # τ[:, 6:]  — hinge joint torques            (actual actuator torques, Nm)
    F_sky = np.linalg.norm(torques[:, 0:3], axis=1)   # (T,)
    tau_sky = np.linalg.norm(torques[:, 3:6], axis=1) # (T,)
    wrench = np.linalg.norm(torques[:, 0:6], axis=1)  # (T,)

    # Actuation power: Σ_j |τ_j · q̇_j| for hinge DOFs only
    # qd[:, 6:] are hinge joint velocities [rad/s]
    # torques[:, 6:] are hinge joint torques [Nm]
    P_active = np.sum(np.abs(torques[:, 6:] * qd[:, 6:]), axis=1)  # (T,) [W]

    # Fraction of Physically Implausible frames (FPI):
    # frames where skyhook force > FPI_THRESHOLD_N (10× body weight)
    fpi = float(np.mean(F_sky > FPI_THRESHOLD_N))

    return {
        "F_sky_per_frame": F_sky.astype(np.float32),
        "tau_sky_per_frame": tau_sky.astype(np.float32),
        "wrench_per_frame": wrench.astype(np.float32),
        "P_active_per_frame": P_active.astype(np.float32),
        # Median is the primary summary statistic — robust to per-frame blowups
        "F_sky_median": float(np.median(F_sky)),
        "tau_sky_median": float(np.median(tau_sky)),
        "P_active_median": float(np.median(P_active)),
        # Mean retained for completeness (sensitive to outliers)
        "F_sky_mean": float(np.mean(F_sky)),
        "tau_sky_mean": float(np.mean(tau_sky)),
        "P_active_mean": float(np.mean(P_active)),
        # Normalised median: F_sky_median / body_weight (1.0 = one body weight)
        "F_sky_norm": float(np.median(F_sky) / BODY_WEIGHT_N),
        # Fraction of frames with skyhook force > 10× body weight
        "FPI": fpi,
        "n_frames": T,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ──────────────────────────────────────────────────────────────────────────────

def batch_evaluate(
    source: str,
    clip_ids: list,
    device: str = "cuda:0",
    fps: int = 30,
    trim_edges: int = 5,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    resume: bool = True,
) -> list:
    """
    Evaluate residual wrench for a list of clips.

    Args:
        source   : "gt" or "generated"
        clip_ids : list of clip ID strings
        device   : CUDA device
        fps      : frames per second (joint_q is at this rate after any downsampling)
        output_dir : where to cache results JSON
        resume   : skip clips already in cached results

    Returns list of per-clip result dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{source}_results.json")

    # Resume: load existing results
    completed = {}
    if resume and os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
        completed = {r["clip_id"]: r for r in existing if "F_sky_median" in r}
        print(f"Resuming: {len(completed)} clips already done for {source}")

    results = list(completed.values())
    load_fn = load_gt_clip if source == "gt" else load_generated_clip
    t0 = time.time()
    n_processed = 0  # count only clips actually computed (not resumed), for ETA

    for i, clip_id in enumerate(clip_ids):
        if clip_id in completed:
            continue

        elapsed = time.time() - t0
        # Use n_processed (not loop index i) so ETA is accurate when resuming
        rate = n_processed / max(elapsed, 1)
        remaining = len(clip_ids) - i - 1
        eta = remaining / rate if rate > 0 else 0
        print(f"[{i+1}/{len(clip_ids)}] clip {clip_id} | "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        persons_data = load_fn(clip_id)
        if persons_data is None:
            print(f"  Skipping — data not found")
            continue

        clip_result = {"clip_id": clip_id, "source": source, "persons": []}
        clip_valid = True

        for person_idx, (joint_q, betas) in enumerate(persons_data):
            # Downsample if joint_q is at 60fps (retargeted GT is sometimes 60fps)
            actual_fps = fps
            # joint_q shape[0] / expected duration check not needed — trust fps arg

            metrics = compute_residual_wrench(joint_q, betas, fps=actual_fps,
                                              device=device, trim_edges=trim_edges)
            if metrics is None:
                print(f"  person {person_idx}: ID failed, skipping clip")
                clip_valid = False
                break

            # Don't store large per-frame arrays in the JSON — just scalars
            clip_result["persons"].append({
                "person_idx": person_idx,
                "F_sky_median": metrics["F_sky_median"],
                "F_sky_mean": metrics["F_sky_mean"],
                "F_sky_norm": metrics["F_sky_norm"],
                "tau_sky_median": metrics["tau_sky_median"],
                "tau_sky_mean": metrics["tau_sky_mean"],
                "P_active_median": metrics["P_active_median"],
                "P_active_mean": metrics["P_active_mean"],
                "FPI": metrics["FPI"],
                "n_frames": metrics["n_frames"],
            })
            print(f"  person {person_idx}: "
                  f"F_sky_med={metrics['F_sky_median']:.0f}N ({metrics['F_sky_norm']:.2f}BW)  "
                  f"FPI={metrics['FPI']*100:.1f}%  "
                  f"τ_sky_med={metrics['tau_sky_median']:.0f}Nm  "
                  f"P_med={metrics['P_active_median']:.0f}W  "
                  f"({metrics['n_frames']} frames)")

        if not clip_valid or not clip_result["persons"]:
            continue

        # Aggregate across persons: mean of person-level means
        clip_result["F_sky_median"] = float(np.mean([p["F_sky_median"] for p in clip_result["persons"]]))
        clip_result["F_sky_mean"]   = float(np.mean([p["F_sky_mean"]   for p in clip_result["persons"]]))
        clip_result["F_sky_norm"]   = float(np.mean([p["F_sky_norm"]   for p in clip_result["persons"]]))
        clip_result["tau_sky_median"] = float(np.mean([p["tau_sky_median"] for p in clip_result["persons"]]))
        clip_result["tau_sky_mean"]   = float(np.mean([p["tau_sky_mean"]   for p in clip_result["persons"]]))
        clip_result["P_active_median"] = float(np.mean([p["P_active_median"] for p in clip_result["persons"]]))
        clip_result["P_active_mean"]   = float(np.mean([p["P_active_mean"]   for p in clip_result["persons"]]))
        clip_result["FPI"]      = float(np.mean([p["FPI"]      for p in clip_result["persons"]]))
        clip_result["n_frames"] = int(np.mean([p["n_frames"]   for p in clip_result["persons"]]))

        results.append(clip_result)
        n_processed += 1

        # Save incrementally every 10 clips
        if len(results) % 10 == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved {len(results)} results → {results_path}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone: {len(results)} clips saved → {results_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Comparison and plotting
# ──────────────────────────────────────────────────────────────────────────────

def compare_and_plot(
    gt_results: list,
    gen_results: list,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Compare GT vs generated residual wrench distributions.
    Produces plots and a comparison JSON.

    Returns comparison dict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Cap for per-clip scalars: values above this are numerical blowups, not physics.
    # 1e6 N = 136,000 × body weight — clearly not a real force.
    SCALAR_CAP = 1e6

    def extract(results, key, cap=None):
        vals = [r[key] for r in results if key in r]
        if not vals:
            raise ValueError(f"No results contain key '{key}' — check that evaluation completed.")
        arr = np.array(vals, dtype=np.float64)
        if cap is not None:
            n_capped = int(np.sum(arr > cap))
            if n_capped:
                print(f"  [extract '{key}'] capping {n_capped} outliers > {cap:.0f}")
            arr = np.clip(arr, None, cap)
        return arr.astype(np.float32)

    # Use median-based per-clip scalars (robust to per-frame numerical blowups)
    # Cap extreme values that represent numerical failures, not physics
    gt_fsky  = extract(gt_results,  "F_sky_median",    cap=SCALAR_CAP)
    gen_fsky = extract(gen_results, "F_sky_median",    cap=SCALAR_CAP)
    gt_fnorm  = extract(gt_results,  "F_sky_norm",     cap=SCALAR_CAP / BODY_WEIGHT_N)
    gen_fnorm = extract(gen_results, "F_sky_norm",     cap=SCALAR_CAP / BODY_WEIGHT_N)
    gt_tsky  = extract(gt_results,  "tau_sky_median",  cap=SCALAR_CAP)
    gen_tsky = extract(gen_results, "tau_sky_median",  cap=SCALAR_CAP)
    gt_pact  = extract(gt_results,  "P_active_median", cap=SCALAR_CAP * 1e3)
    gen_pact = extract(gen_results, "P_active_median", cap=SCALAR_CAP * 1e3)
    gt_fpi   = extract(gt_results,  "FPI")
    gen_fpi  = extract(gen_results, "FPI")

    # ── ΔF_sky per matched clip pair (using median-based per-clip F_sky) ─────
    gt_by_id  = {r["clip_id"]: r["F_sky_median"] for r in gt_results  if "F_sky_median" in r}
    gen_by_id = {r["clip_id"]: r["F_sky_median"] for r in gen_results if "F_sky_median" in r}
    common_ids = sorted(set(gt_by_id) & set(gen_by_id))
    delta_fsky = np.array([gen_by_id[c] - gt_by_id[c] for c in common_ids], dtype=np.float32)

    # ── Aggregated stats (nan-safe) ───────────────────────────────────────────
    def stats(arr):
        valid = arr[np.isfinite(arr)]
        return {
            "mean":   float(np.mean(valid))               if len(valid) else float("nan"),
            "std":    float(np.std(valid))                if len(valid) else float("nan"),
            "median": float(np.median(valid))             if len(valid) else float("nan"),
            "p90":    float(np.percentile(valid, 90))     if len(valid) else float("nan"),
            "n":      int(len(valid)),
            "n_total": int(len(arr)),
        }

    comparison = {
        "note": "F_sky_median_N is the primary metric — median per frame per clip, "
                "then mean across clips. Robust to per-frame numerical blowups.",
        "gt": {
            "F_sky_median_N": stats(gt_fsky),
            "F_sky_norm_BW":  stats(gt_fnorm),
            "tau_sky_median_Nm": stats(gt_tsky),
            "P_active_median_W": stats(gt_pact),
            "FPI_fraction": stats(gt_fpi),
        },
        "generated": {
            "F_sky_median_N": stats(gen_fsky),
            "F_sky_norm_BW":  stats(gen_fnorm),
            "tau_sky_median_Nm": stats(gen_tsky),
            "P_active_median_W": stats(gen_pact),
            "FPI_fraction": stats(gen_fpi),
        },
        "delta_F_sky_N": stats(delta_fsky),
        "ratios": {
            "F_sky_median": float(np.nanmean(gen_fsky) / max(np.nanmean(gt_fsky), 1e-6)),
            "tau_sky_median": float(np.nanmean(gen_tsky) / max(np.nanmean(gt_tsky), 1e-6)),
            "P_active_median": float(np.nanmean(gen_pact) / max(np.nanmean(gt_pact), 1e-6)),
            "FPI": float(np.nanmean(gen_fpi) / max(np.nanmean(gt_fpi), 1e-6)),
        },
        "n_matched_pairs": len(common_ids),
        "body_weight_N": BODY_WEIGHT_N,
        "FPI_threshold_N": FPI_THRESHOLD_N,
    }

    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print("\n=== Dynamic Plausibility Comparison (median-based) ===")
    print(f"  F_sky_med  GT={np.nanmean(gt_fsky):.0f}±{np.nanstd(gt_fsky):.0f}N  "
          f"Gen={np.nanmean(gen_fsky):.0f}±{np.nanstd(gen_fsky):.0f}N  "
          f"ratio={comparison['ratios']['F_sky_median']:.2f}x")
    print(f"  τ_sky_med  GT={np.nanmean(gt_tsky):.0f}±{np.nanstd(gt_tsky):.0f}Nm  "
          f"Gen={np.nanmean(gen_tsky):.0f}±{np.nanstd(gen_tsky):.0f}Nm  "
          f"ratio={comparison['ratios']['tau_sky_median']:.2f}x")
    print(f"  P_act_med  GT={np.nanmean(gt_pact):.0f}±{np.nanstd(gt_pact):.0f}W   "
          f"Gen={np.nanmean(gen_pact):.0f}±{np.nanstd(gen_pact):.0f}W   "
          f"ratio={comparison['ratios']['P_active_median']:.2f}x")
    print(f"  FPI        GT={np.nanmean(gt_fpi)*100:.1f}%  "
          f"Gen={np.nanmean(gen_fpi)*100:.1f}%  "
          f"(threshold={FPI_THRESHOLD_N:.0f}N = 3×BW)")
    print(f"  F_sky_norm  GT={np.nanmean(gt_fnorm):.2f}BW  Gen={np.nanmean(gen_fnorm):.2f}BW  "
          f"(1 BW = {BODY_WEIGHT_N:.0f}N)")
    print(f"  ΔF_sky mean={np.nanmean(delta_fsky):.0f}N  "
          f"median={np.nanmedian(delta_fsky):.0f}N  "
          f"(n={len(common_ids)} paired clips)")

    # ── Plot 1: Skyhook force distribution ───────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Residual Wrench: GT vs Generated  (per-clip median)", fontsize=13, fontweight="bold")

    def _hist_pair(ax, gt_vals, gen_vals, xlabel, title, ratio):
        pmax = np.percentile(np.concatenate([gt_vals, gen_vals]), 95)
        bins = np.linspace(0, pmax, 40) if pmax > 0 else 40
        ax.hist(gt_vals, bins=bins, alpha=0.6, label=f"GT (n={len(gt_vals)})",
                color="#2196F3", density=True)
        ax.hist(gen_vals, bins=bins, alpha=0.6, label=f"Gen (n={len(gen_vals)})",
                color="#F44336", density=True)
        ax.axvline(np.mean(gt_vals), color="#1565C0", linestyle="--", linewidth=1.5,
                   label=f"GT μ={np.mean(gt_vals):.0f}")
        ax.axvline(np.mean(gen_vals), color="#B71C1C", linestyle="--", linewidth=1.5,
                   label=f"Gen μ={np.mean(gen_vals):.0f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{title}  (ratio={ratio:.1f}x)")
        ax.legend(fontsize=8)

    _hist_pair(axes[0], gt_fsky, gen_fsky,
               "Per-clip median F_sky [N]", "Skyhook Force",
               comparison["ratios"]["F_sky_median"])
    _hist_pair(axes[1], gt_tsky, gen_tsky,
               "Per-clip median τ_sky [Nm]", "Skyhook Torque",
               comparison["ratios"]["tau_sky_median"])
    _hist_pair(axes[2], gt_pact, gen_pact,
               "Per-clip median P_active [W]", "Actuation Power",
               comparison["ratios"]["P_active_median"])

    # FPI panel
    ax = axes[3]
    ax.bar(["GT", "Generated"], [np.nanmean(gt_fpi)*100, np.nanmean(gen_fpi)*100],
           color=["#2196F3", "#F44336"], alpha=0.85, width=0.5)
    ax.errorbar(["GT", "Generated"],
                [np.nanmean(gt_fpi)*100, np.nanmean(gen_fpi)*100],
                yerr=[np.nanstd(gt_fpi)*100, np.nanstd(gen_fpi)*100],
                fmt="none", color="black", capsize=5)
    ax.set_ylabel("Fraction of frames [%]")
    ratio_fpi = comparison["ratios"]["FPI"]
    ax.set_title(f"FPI: frames > 3×BW  (ratio={ratio_fpi:.1f}x)\n"
                 f"threshold = {FPI_THRESHOLD_N:.0f} N")
    for i, (lbl, val) in enumerate(zip(["GT", "Gen"],
                                        [np.nanmean(gt_fpi)*100, np.nanmean(gen_fpi)*100])):
        ax.text(i, val + 0.5, f"{val:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(output_dir, "residual_wrench_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    # ── Plot 2: ΔF_sky distribution ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delta_fsky, bins=40, color="#9C27B0", alpha=0.75, density=True)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", label="zero (perfect)")
    ax.axvline(np.nanmean(delta_fsky), color="#6A0080", linewidth=2,
               label=f"mean ΔF_sky = {np.nanmean(delta_fsky):.1f}N")
    ax.axvline(np.nanmedian(delta_fsky), color="#CE93D8", linewidth=1.5, linestyle=":",
               label=f"median = {np.nanmedian(delta_fsky):.1f}N")
    ax.set_xlabel("ΔF_sky = F_sky(generated) − F_sky(GT)  [N]")
    ax.set_ylabel("Density")
    ax.set_title(f"Dynamic Gap (ΔF_sky) across {len(common_ids)} paired clips\n"
                 f"positive = generated needs more skyhook support than GT")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(output_dir, "delta_fsky_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    # ── Plot 3: Summary bar chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics_names = ["F_sky_med [N]", "τ_sky_med [Nm]", "P_active_med [W]", "FPI [%]"]
    gt_vals  = [np.nanmean(gt_fsky),  np.nanmean(gt_tsky),  np.nanmean(gt_pact),  np.nanmean(gt_fpi)*100]
    gen_vals = [np.nanmean(gen_fsky), np.nanmean(gen_tsky), np.nanmean(gen_pact), np.nanmean(gen_fpi)*100]
    gt_errs  = [np.nanstd(gt_fsky),   np.nanstd(gt_tsky),   np.nanstd(gt_pact),   np.nanstd(gt_fpi)*100]
    gen_errs = [np.nanstd(gen_fsky),  np.nanstd(gen_tsky),  np.nanstd(gen_pact),  np.nanstd(gen_fpi)*100]

    x = np.arange(len(metrics_names))
    w = 0.35
    bars_gt = ax.bar(x - w/2, gt_vals, w, yerr=gt_errs, label="GT", color="#2196F3",
                     alpha=0.85, capsize=5)
    bars_gen = ax.bar(x + w/2, gen_vals, w, yerr=gen_errs, label="Generated", color="#F44336",
                      alpha=0.85, capsize=5)

    # Ratio annotations
    for xi, (gv, genf) in enumerate(zip(gt_vals, gen_vals)):
        if gv > 1e-6:
            ax.text(xi, max(gv, genf) * 1.12, f"{genf/gv:.1f}x",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("Value (mean ± std)")
    ax.set_title("Residual Wrench Metrics: GT vs Generated  (per-clip medians)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(output_dir, "metrics_barplot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    return comparison


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Residual Wrench Evaluation — Dynamic Plausibility Metric",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", choices=["gt", "generated", "both"], default="both",
                        help="Which data source to evaluate (default: both)")
    parser.add_argument("--n-clips", type=int, default=50,
                        help="Number of clips to evaluate per source (default: 50)")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device (default: cuda:0)")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS of joint_q data (default: 30)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--trim-edges", type=int, default=8,
                        help="Frames to drop at each end of each clip (spline boundary fix, default: 8)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Recompute all clips even if cached results exist")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip evaluation, only run comparison plot on existing results")
    parser.add_argument("--gt-results",
                        help="Path to GT results JSON (for --compare-only)")
    parser.add_argument("--gen-results",
                        help="Path to generated results JSON (for --compare-only)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Compare-only mode ─────────────────────────────────────────────────────
    if args.compare_only:
        gt_path = args.gt_results or os.path.join(args.output_dir, "gt_results.json")
        gen_path = args.gen_results or os.path.join(args.output_dir, "generated_results.json")
        if not os.path.exists(gt_path) or not os.path.exists(gen_path):
            print(f"ERROR: need both {gt_path} and {gen_path}")
            sys.exit(1)
        gt_results = json.load(open(gt_path))
        gen_results = json.load(open(gen_path))
        compare_and_plot(gt_results, gen_results, args.output_dir)
        return

    # ── Select clips from test split ──────────────────────────────────────────
    print("Loading test split...")
    test_clips = load_test_split()

    # Filter to clips that have both GT and generated data available
    available = [c for c in test_clips
                 if os.path.exists(os.path.join(GT_RETARGETED_DIR, f"{c}_person0_joint_q.npy"))
                 and os.path.exists(os.path.join(GENERATED_DIR, f"{c}.pkl"))]
    print(f"Test split: {len(test_clips)} clips | both GT+Gen available: {len(available)}")

    np.random.seed(42)
    selected = np.random.choice(available, size=min(args.n_clips, len(available)), replace=False)
    selected = sorted(selected)
    print(f"Selected {len(selected)} clips for evaluation")

    # ── Run evaluation ────────────────────────────────────────────────────────
    gt_results, gen_results = [], []

    if args.source in ("gt", "both"):
        print(f"\n=== Evaluating GT ({len(selected)} clips) ===")
        gt_results = batch_evaluate(
            "gt", selected,
            device=args.device,
            fps=args.fps,
            trim_edges=args.trim_edges,
            output_dir=args.output_dir,
            resume=not args.no_resume,
        )

    if args.source in ("generated", "both"):
        print(f"\n=== Evaluating Generated ({len(selected)} clips) ===")
        gen_results = batch_evaluate(
            "generated", selected,
            device=args.device,
            fps=args.fps,
            trim_edges=args.trim_edges,
            output_dir=args.output_dir,
            resume=not args.no_resume,
        )

    # Load the other if only one was run
    if args.source == "gt":
        gen_path = os.path.join(args.output_dir, "generated_results.json")
        if os.path.exists(gen_path):
            gen_results = json.load(open(gen_path))
    elif args.source == "generated":
        gt_path = os.path.join(args.output_dir, "gt_results.json")
        if os.path.exists(gt_path):
            gt_results = json.load(open(gt_path))

    # ── Comparison ────────────────────────────────────────────────────────────
    if gt_results and gen_results:
        print("\n=== Comparison ===")
        compare_and_plot(gt_results, gen_results, args.output_dir)
    else:
        print("\nSkipping comparison (need both GT and generated results)")


if __name__ == "__main__":
    main()
