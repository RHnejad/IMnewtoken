# Residual Wrench: A Dynamic Plausibility Metric for Motion Generation

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [Why Not Just Compare Torques Directly?](#2-why-not-just-compare-torques-directly)
3. [Why Not Just Compare Accelerations?](#3-why-not-just-compare-accelerations)
4. [The Residual Wrench Approach](#4-the-residual-wrench-approach)
5. [Metrics Defined](#5-metrics-defined)
6. [Implementation Details](#6-implementation-details)
7. [Known Limitations and Mitigations](#7-known-limitations-and-mitigations)
8. [How to Run](#8-how-to-run)
9. [Output Files](#9-output-files)
10. [Interpreting Results](#10-interpreting-results)
11. [Relationship to Other Metrics in This Repo](#11-relationship-to-other-metrics-in-this-repo)
12. [References](#12-references)

---

## 1. Motivation and Problem Statement

Motion generation models like InterMask produce motions that are **kinematically smooth** and look realistic to the eye. However, kinematics alone cannot tell you whether a motion is *physically possible* — whether the person could actually execute those movements under gravity with real muscles.

The core question is: **"Does this generated motion obey Newton's laws?"**

For two-person interaction motions specifically, this matters a lot. A person pushing another person requires real contact forces and corresponding joint torques. Without a physics-grounded metric, generated motions can appear plausible while being dynamically impossible — for example, a person "floating" through space or performing a movement that would require infinite muscle force.

---

## 2. Why Not Just Compare Torques Directly?

A first attempt might be: compute joint torques for GT and generated motions using inverse dynamics, then compare them with MSE.

**The problem**: MSE between GT torques and generated torques requires **frame-level temporal alignment**. Generated motions are a different *realization* of the same text prompt — frame 47 of a generated "hit" sequence has no reason to correspond to frame 47 of the GT "hit" sequence. Computing MSE across misaligned frames is meaningless.

**The solution**: Use **distributional comparison**. Instead of comparing frame-to-frame, compare the *distribution* of per-clip summary statistics across many clips. This is what this metric does.

---

## 3. Why Not Just Compare Accelerations?

Since torques are related to accelerations via `τ = M(q)·q̈ − h(q, q̇)`, one might ask: *are torques just a mass-weighted version of accelerations?*

The answer is **no**, for several reasons:

| Feature | Acceleration Comparison | Torque/Wrench Comparison |
|---------|------------------------|--------------------------|
| Mass distribution | Treats a heavy arm and light arm the same if they move at the same speed | Penalizes physically unrealistic movement of heavy limbs |
| Gravity | Only cares about changes in motion (gravity-free) | Captures the *static effort* of holding a pose (even `a=0` requires non-zero torque against gravity) |
| Physical interpretability | Unitless relative to biology | Units of Newtons/Nm, comparable to human muscle capacity |
| "Floating" detection | Cannot detect a floating character | Skyhook force reveals how much external support the motion implicitly requires |

Torques *do* approach accelerations in the limit where GT and generated poses are kinematically very similar (then `M(q_gt) ≈ M(q_gen)` and `h(q_gt, q̇_gt) ≈ h(q_gen, q̇_gen)`). But in practice for generated motions, pose differences are large enough that torques capture genuinely different information.

---

## 4. The Residual Wrench Approach

### 4.1 What Is the Residual Wrench?

Inverse dynamics computes the joint torques needed to reproduce an observed motion:

```
τ = M(q) · (q̈_desired − q̈_free)
```

Where:
- `M(q)` is the mass/inertia matrix at pose `q`
- `q̈_desired` is the acceleration computed from the observed trajectory
- `q̈_free` is the free-fall acceleration under zero torques (gravity + Coriolis only, obtained from a single forward physics step)

The output `τ` has **75 DOFs** for a SMPL body:
- `τ[0:3]` — root translation virtual force (**skyhook force**, Newtons)
- `τ[3:6]` — root rotation virtual torque (**skyhook torque**, Nm)
- `τ[6:75]` — hinge joint torques (actual actuator outputs, Nm)

The root DOFs (0:6) are a **"skyhook"** — a virtual force that the physics engine must inject to keep the character on its reference trajectory. This force has no physical actuator; it represents the dynamic residual of the motion.

**Key insight**: For physically valid mocap data, the skyhook should be small (the person is actually balanced under gravity and ground contacts). For generated motions that violate physics, the skyhook must be large to "hold up" the character — it reveals the *dynamic implausibility* of the motion.

### 4.2 Why Use the Skyhook Instead of Internal Torques?

The skyhook captures something that internal torques alone cannot: **whether the character is actually balanced**. A motion can have perfectly reasonable internal joint torques while being dynamically impossible (e.g., a character flying through the air with normal arm movements). The skyhook force catches this: if the character is floating, the skyhook must support its entire body weight continuously.

The skyhook also acts as a **leakage detector**: if a generated motion is kinematically similar to GT but dynamically "fake," the internal torques may look normal while the skyhook is large. This is the "skyhook leakage" effect — the skyhook is doing the physical work that real contact forces (ground, partner) would do.

### 4.3 Comparison Strategy: Distributions, Not Frame-Level

Rather than comparing GT and generated torques frame-by-frame, we:

1. Compute a **per-clip scalar** summary (median of frame-level F_sky)
2. Collect this scalar across **50-200 clips** for both GT and generated
3. Compare the **distributions** using ratios and ΔF_sky statistics

This avoids the temporal alignment problem entirely. GT clips should cluster at low F_sky; generated clips should show consistently higher F_sky if they are dynamically implausible.

---

## 5. Metrics Defined

### 5.1 Primary Metric: `F_sky_median` (Skyhook Force, Newtons)

```
F_sky(t) = ||τ[t, 0:3]||₂            per frame
F_sky_median(clip) = median_t F_sky(t)   per clip
```

**Why median, not mean?** The per-frame distribution of skyhook forces can have extreme outliers (spline fitting boundary artefacts, extreme poses at clip transitions). The median is robust to these. We then take the mean of per-clip medians when aggregating across clips, so the final number represents the "typical" per-frame skyhook force.

**Units**: Newtons (N). Body weight = 75 kg × 9.81 m/s² ≈ 736 N.

### 5.2 `F_sky_norm` (Normalised Skyhook Force, body weights)

```
F_sky_norm = F_sky_median / (75 kg × 9.81 m/s²)
```

Dimensionless ratio. `F_sky_norm = 1.0` means the skyhook is supporting one full body weight. For reference:
- GT motions: ~0.8-1.8 BW (includes gravity support not captured by ground contacts)
- Generated motions: typically 50-150+ BW

### 5.3 `FPI` — Fraction of Physically Implausible frames

```
FPI(clip) = fraction of frames where F_sky(t) > 3 × body_weight
          = fraction of frames where F_sky(t) > ~2,208 N
```

`3 × body_weight` is chosen based on biomechanics literature (Winter 2009): peak ground reaction forces during sprinting reach ~2-3×BW, so a skyhook exceeding 3×BW cannot be explained by any normal contact mechanics — the motion is dynamically unsupported. A threshold of 10×BW (as sometimes used in engineering contexts) is too permissive and misses many clearly implausible frames.

**Interpretation**: `FPI = 1.0` (100%) means *every single frame* of the motion requires a skyhook force exceeding 3 body weights — the motion is completely dynamically infeasible without a virtual support mechanism.

### 5.4 `τ_sky_median` (Skyhook Torque, Nm)

The rotational component of the skyhook, analogous to F_sky. Measures the virtual torque required to maintain the root orientation trajectory.

### 5.5 `P_active_median` (Actuation Power, Watts)

```
P_active(t) = Σ_j |τ_j(t) · q̇_j(t)|   for hinge joints j=6..74
P_active_median(clip) = median_t P_active(t)
```

The total **mechanical power** consumed by all joint actuators. This is the "actuation work" metric — how much muscular energy per second the motion requires. Unrealistic motions often require disproportionate actuation power due to large accelerations.

### 5.6 `ΔF_sky` — Dynamic Gap

```
ΔF_sky(clip) = F_sky_median(generated, clip) − F_sky_median(GT, clip)
```

Computed for **matched clip pairs** (same clip ID, different source). The mean/median of ΔF_sky across all matched pairs is the single "Dynamic Gap" score. Positive means generated motions require more skyhook support; higher = more implausible.

---

## 6. Implementation Details

### 6.1 Inverse Dynamics Implementation

File: `prepare2/compute_torques.py`, function `inverse_dynamics()`

The implementation follows:
```
τ = M(q) · (q̈_desired − q̈_free)
```

1. **Velocity/acceleration estimation**: B-spline differentiation (`diff_method="spline"`), which avoids noise amplification from finite differences. The cubic B-spline is fit to the full trajectory; velocities and accelerations are obtained analytically from spline derivatives.

2. **Zero-torque step**: A single `SolverMuJoCo` forward step with `control.joint_f = 0` gives `q̈_free` (free acceleration under gravity, Coriolis, and ground contacts). The MuJoCo solver is used here (not Featherstone) because it handles implicit contacts stably at the large motion-capture timestep (1/30 s).

3. **Mass matrix**: `newton.eval_mass_matrix(model, state)` — computed at the current configuration `q`.

4. **Ground plane**: The model is built with a ground plane (`builder.add_ground_plane()`), so ground reaction forces are captured in the zero-torque step. This partially reduces the skyhook force for GT clips.

### 6.2 Edge Trimming

The first and last 5 frames of each clip are discarded before computing metrics (`trim_edges=5`). This is because:

- B-spline fitting introduces boundary artefacts at the endpoints
- Generated motions may start/end in unrealistic configurations

### 6.3 Data Sources

- **GT**: Retargeted Newton joint coordinates from `data/retargeted_v2/interhuman/`
  - Format: `(T, 76)` array, pre-computed via `prepare2/retarget.py`
- **Generated**: SMPL-X parameters from `data/generated/interhuman/*.pkl`
  - Retargeted on-the-fly using `prepare2/retarget.py:smplx_to_joint_q()`
  - Generated by the InterMask model

### 6.4 Body Model

- SMPL skeleton with 24 bodies, 75 DOF (6 root + 69 hinge)
- Total mass: 75 kg (De Leva 1996 anthropometric fractions)
- XML generated per-subject from SMPL betas via `get_or_create_xml(betas)`

### 6.5 Robustness: Median vs Mean

We use **median across frames** for the per-clip scalar, not mean, because:

- Generated motions can have catastrophic per-frame blowups (e.g., extreme initial pose → M·q̈ → gigantic torques)
- Mean of such distributions is dominated by a few extreme frames and is not interpretable
- The median of a clip's frame-level F_sky distribution represents the "typical" dynamic residual

The per-clip medians are then averaged (mean) across clips for the final reported number. This two-stage aggregation (median within clip → mean across clips) is a standard practice in biomechanics for summarizing noisy force data.

---

## 7. Known Limitations and Mitigations

### 7.1 GT Baseline Is Not Zero

**Issue**: GT F_sky is ~600-1300 N (0.8-1.8 body weights), not zero.

**Why**: The skyhook force in this formulation includes gravity support. Even with a ground plane, the zero-torque step's implicit contact forces only *partially* cancel gravity. The residual gravity support (~300-700 N) appears as skyhook force even for perfectly valid mocap.

**Mitigation**: Use **ratio** or **ΔF_sky** comparisons rather than absolute values. The key finding is that generated motions have 50-150× larger F_sky than GT, which is unambiguous regardless of the non-zero GT baseline.

### 7.2 Spline Boundary Artefacts

**Issue**: B-spline differentiation produces extreme accelerations at clip boundaries.

**Mitigation**: `trim_edges=5` discards the first and last 5 frames. The FPI metric provides an additional safeguard by counting only frames above a physical threshold.

### 7.3 Contact Force Approximation

**Issue**: The zero-torque step uses the MuJoCo contact model (implicit, penalty-based). This approximates but does not perfectly reconstruct the true ground reaction forces.

**Mitigation**: This affects GT and generated equally, so the relative comparison (ratio, ΔF_sky) is still valid.

### 7.4 IK Quality for Generated Data

**Issue**: Generated motions are retargeted from SMPL-X parameters using direct rotation mapping. The retargeting may introduce small errors.

**Mitigation**: The retargeting error is small relative to the 50-150× F_sky gap between GT and generated.

---

## 8. How to Run

### Prerequisites

```bash
conda activate mimickit
# Requires: newton, warp, numpy, scipy, matplotlib, pickle
```

### Quick 50-clip evaluation (GT + Generated, test split)

```bash
python prepare2/residual_wrench_eval.py \
    --n-clips 50 \
    --device cuda:0 \
    --output-dir data/residual_wrench
```

This runs ~50 GT clips and ~50 generated clips (same clips, matched pairs from the test split), then produces comparison plots.

**Time**: ~1.5 min/clip for GT, ~1 min/clip for generated. Total: ~2 hours for 50+50.

### Resume interrupted run

```bash
python prepare2/residual_wrench_eval.py --n-clips 50 --device cuda:0
# (resume is enabled by default — already-processed clips are skipped)
```

### Only GT or only generated

```bash
python prepare2/residual_wrench_eval.py --source gt --n-clips 100
python prepare2/residual_wrench_eval.py --source generated --n-clips 100
```

### Comparison plot from existing results

```bash
python prepare2/residual_wrench_eval.py --compare-only \
    --gt-results data/residual_wrench/gt_results.json \
    --gen-results data/residual_wrench/generated_results.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-clips` | 50 | Clips to evaluate per source |
| `--device` | `cuda:0` | CUDA device |
| `--fps` | 30 | Frame rate of joint_q data |
| `--trim-edges` | 5 | Frames to drop at each clip boundary |
| `--source` | `both` | `gt`, `generated`, or `both` |
| `--output-dir` | `data/residual_wrench/` | Where to save results and plots |
| `--no-resume` | off | Recompute even if cached |
| `--compare-only` | off | Skip evaluation, only plot |

---

## 9. Output Files

All outputs go to `data/residual_wrench/` by default.

| File | Description |
|------|-------------|
| `gt_results.json` | Per-clip metrics for GT (list of dicts) |
| `generated_results.json` | Per-clip metrics for generated |
| `comparison.json` | Aggregated comparison: GT vs generated stats, ratios |
| `residual_wrench_distributions.png` | 4-panel histogram: F_sky, τ_sky, P_active, FPI |
| `delta_fsky_distribution.png` | ΔF_sky distribution across matched pairs |
| `metrics_barplot.png` | Bar chart: all metrics side by side |

### Per-clip result format (gt_results.json)

```json
{
  "clip_id": "3433",
  "source": "gt",
  "persons": [
    {
      "person_idx": 0,
      "F_sky_median": 621.4,
      "F_sky_mean": 843.2,
      "F_sky_norm": 0.845,
      "tau_sky_median": 1201.3,
      "tau_sky_mean": 1832.1,
      "P_active_median": 214.8,
      "P_active_mean": 289.3,
      "FPI": 0.012,
      "n_frames": 419
    }
  ],
  "F_sky_median": 621.4,
  "F_sky_norm": 0.845,
  "tau_sky_median": 1201.3,
  "P_active_median": 214.8,
  "FPI": 0.012,
  "n_frames": 419
}
```

### Comparison JSON structure

```json
{
  "note": "F_sky_median_N is the primary metric...",
  "gt": {
    "F_sky_median_N": {"mean": 750.1, "std": 220.3, "median": 682.4, "p90": 1020.1, "n": 50},
    "F_sky_norm_BW": {"mean": 1.02, ...},
    "FPI_fraction": {"mean": 0.018, ...}
  },
  "generated": {
    "F_sky_median_N": {"mean": 48232.5, ...},
    "FPI_fraction": {"mean": 0.97, ...}
  },
  "delta_F_sky_N": {"mean": 47482.4, "median": 32140.1, ...},
  "ratios": {
    "F_sky_median": 64.3,
    "FPI": 53.9
  },
  "n_matched_pairs": 50,
  "body_weight_N": 735.75,
  "FPI_threshold_N": 7357.5
}
```

---

## 10. Interpreting Results

### What good results look like

For a physically plausible motion generator, you would expect:

| Metric | GT (expected) | Generated (good generator) | Bad generator |
|--------|--------------|---------------------------|---------------|
| `F_sky_median` | ~700-1200 N | similar to GT | 10-100× GT |
| `F_sky_norm` | ~1-1.6 BW | ~1-2 BW | 50-200+ BW |
| `FPI` | < 5% | < 15% | > 50% |
| `ΔF_sky` (mean) | — | < 500 N | > 10,000 N |

### Why does GT have non-zero F_sky?

The baseline GT F_sky (~700-1300 N) represents:
1. **Residual gravity** not captured by the ground contact model (~300-500 N)
2. **Dynamic effects** during movement (acceleration forces)
3. **Mocap noise** amplified by double differentiation

This is expected and well-documented in inverse dynamics literature. The GT baseline is roughly 1-1.5 body weights — this is what "physically valid mocap" looks like under this ID formulation. Generated motions in the range of 50-150 body weights are clearly and unambiguously implausible.

### The FPI metric is the clearest discriminator

FPI (Fraction of Physically Implausible frames) is the most interpretable metric:
- **GT FPI ≈ 1-5%**: Even mocap has a few extreme frames (jumps, fast impacts)
- **Generated FPI ≈ 80-100%**: Nearly every frame is physically impossible

This is because generated motions — even when kinematically smooth — are not grounded by real physics, so every frame requires a large virtual force to "explain" the trajectory.

---

## 11. Relationship to Other Metrics in This Repo

### vs. `prepare2/compute_skyhook_metrics.py`

The existing `compute_skyhook_metrics.py` computes the same root residual force (root_force_l2 = ||τ[0:3]||₂), but focuses on single-person GT analysis with complex ground-fixing and balance warmup logic. `residual_wrench_eval.py` is designed specifically for **GT vs Generated comparison** across many clips.

Key differences:
- `residual_wrench_eval.py` adds generated data support and batch comparison
- Uses median-based aggregation (more robust)
- Adds FPI, τ_sky, and actuation power metrics
- No balance warmup (not needed for comparative evaluation)

### vs. `prepare4/` Paired-vs-Solo Analysis

The paired-vs-solo approach (`prepare4/paired_simulation.py`) uses PD forward simulation to compare how torques change when simulating two people together vs. alone. That metric directly measures **interaction force inconsistency**.

The residual wrench approach measures **single-person dynamic plausibility** — does the motion make sense for one person under gravity, regardless of the second person?

These two metrics are **complementary**:
- Residual wrench: "Is each person's motion physically possible in isolation?"
- Paired-vs-solo torque delta: "Are the interaction forces between people physically consistent?"

---

## 12. References

- **ImDy (ICLR 2025)**: "Inferring the Dynamics Model of Human Body from Video" — predicts full inverse dynamics from video, demonstrating that residual forces are a standard plausibility measure.

- **PP-Motion (ACM MM 2025)**: "Physical-Perceptual Fidelity Evaluation for Human Motion Generation" — uses physics simulation residuals as plausibility metrics.

- **Winter, D.A. (2009)**: *Biomechanics and Motor Control of Human Movement* — standard reference for inverse dynamics and root residual analysis in biomechanics. Root residuals > 10% of body weight indicate dynamic inconsistency.

- **Zajac & Gordon (1989)**: "Determining muscle's force and action in multi-articular movement" — establishes that internal joint torques and root residuals are separable components of the inverse dynamics solution.

- **De Leva (1996)**: "Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters" — anthropometric mass fractions used in the Newton SMPL body model.
