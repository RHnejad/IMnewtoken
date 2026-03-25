# Newton Force Analysis — 4-Method Interaction Force Pipeline

Computes interaction forces **F\_{B→A}** and **F\_{A→B}** between two interacting persons using four complementary biomechanics methods, compared against the naive COM-based approach.

## Why 4 Methods?

The naive COM approach (`dyadic_physics.md`) is **underdetermined when both persons are grounded** — which is the most common case. Each method below attacks this problem differently, with different assumptions and trade-offs. Comparing them reveals which forces are robust across methods and which are method-dependent artifacts.

| Method | Approach | Newton's 3rd Law | Works When Both Grounded | Requires |
|--------|----------|------------------|--------------------------|----------|
| 1. Contact Sensors | Direct MuJoCo measurement | Enforced by solver | Yes | MuJoCo solver |
| 2. Inverse Dynamics | τ = M(q̈ − q̈\_free) | Not enforced | Partial (skyhook = ground + interaction mixed) | Newton mass matrix |
| 3. RRA | Adjust COM to minimize residuals | Not enforced | Yes (reduced residuals) | Method 2 output |
| 4. Optimization ID | Constrained QP decomposition | **Enforced by construction** | Yes | Method 2 output |
| Naive COM | F = m·a when floating | Checked, not enforced | **No** (underdetermined) | Nothing |

## Methods

### Method 1: Contact Sensor Forces

Runs a **2-person PD-tracked simulation** in MuJoCo and reads contact forces directly from `SensorContact` objects:
- **Foot→ground** sensors: GRF per foot (4 foot bodies × 2 persons)
- **Hand→body** sensors: inter-person contact at hands

**Limitation**: Only measures contact at sensor-configured body pairs (hands, feet). Misses torso-to-torso contact, shoulder pushes, etc.

### Method 2: Inverse Dynamics with Zero-Phase Filtering

Reuses `prepare2/compute_torques.py::inverse_dynamics()`:
1. **Zero-phase Butterworth** filter on joint positions (cutoff 6 Hz, order 4) — removes noise *before* differentiation
2. **B-spline** fitting → analytic 1st/2nd derivatives (avoids O(1/dt²) noise amplification)
3. **τ = M(q) · (q̈ − q̈\_free)** where q̈\_free = acceleration under zero torque (captures gravity + Coriolis + ground contacts)

Root DOFs 0:5 output **"skyhook" forces** — virtual forces needed to hold the pelvis at the reference trajectory. These represent unbalanced external forces (gravity compensation + interaction + ground contact errors).

### Method 3: Residual Reduction Algorithm (RRA)

Inspired by OpenSim RRA. Minimizes non-physical root residuals from Method 2:
1. Parameterize COM offset as a **B-spline** with N knots (default 20)
2. Optimize with **L-BFGS-B**: `min ||root\_forces(q + Δcom)||² + λ·||Δcom||²`
3. Re-run I.D. with adjusted kinematics

The **remaining root residual after RRA** ≈ estimate of true external forces.

### Method 4: Optimization-Based Inverse Dynamics

Frame-by-frame **constrained decomposition** of root residuals into ground reaction + interaction:

```
root_residual_A = F_ground_A + F_{B→A}
root_residual_B = F_ground_B - F_{B→A}    ← Newton's 3rd law
```

Single variable `F_int` (3D) is optimized per frame with:
- **Newton's 3rd law**: enforced by construction (only 1 F\_int, applied as ±)
- **GRF non-negativity**: ground pushes up, penalty for negative F\_ground\_z
- **Floating person**: if not grounded, F\_ground → 0 (penalized)

## Outputs

### Plots

| File | Contents |
|------|----------|
| `newton_analysis_clip_{id}.png` | Main multi-panel figure (all methods, heatmaps, comparison) |
| `skeleton_keyframes_clip_{id}.png` | 2D stick figures of both persons at key moments |
| `mass_uncertainty_clip_{id}.png` | Root force ±1σ bands (if `--mass-uncertainty`) |

### Main Figure Panels (top to bottom, shared time axis)

1. **Contact state** — ground contact per person (from foot heights)
2. **Angular position heatmap** — joint angle magnitudes over time (Person A)
3. **Torque heatmap** — per-joint torque magnitudes over time (Person A)
4. **Method 1** — hand contact sensor force magnitude
5. **Method 2** — root residual (skyhook) magnitude per person
6. **Method 3** — RRA before/after residual comparison
7. **Method 4** — optimized F\_{B→A} components (X, Y, Z) + magnitude
8. **Comparison** — all methods' interaction force estimates overlaid
9. **Naive COM** — contact regimes + solvable-only force estimates

### Skeleton Keyframes

- **Both persons** drawn as 2D stick figures (blue = A, red = B)
- Frames selected by: evenly spaced (~1s apart) + auto-detected peaks (max interaction force, max torque)
- Each subplot labeled with timestamp

## Usage

### Environment Setup

```bash
runai bash intintermask
source /mnt/vita/scratch/vita-staff/users/rh/miniconda3/etc/profile.d/conda.sh
cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/default_intermask/InterMask/
conda activate newton_env
```

### Commands

```bash
# All 4 methods on one clip (default)
python physics_analysis/newton_force_analysis.py --clip 1000

# Only specific methods
python physics_analysis/newton_force_analysis.py --clip 1000 --methods 2 4

# Multiple clips
python physics_analysis/newton_force_analysis.py --clips 1000 2000 3000

# With mass uncertainty analysis
python physics_analysis/newton_force_analysis.py --clip 1000 --mass-uncertainty --mc-samples 20

# Custom output directory
python physics_analysis/newton_force_analysis.py --clip 1000 --output-dir results/my_analysis
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--clip` | str | `"1000"` | Single clip ID to analyze |
| `--clips` | str[] | — | Multiple clip IDs (space-separated) |
| `--data-dir` | str | `data/InterHuman` | Directory containing `motions/{id}.pkl` files |
| `--output-dir` | str | `physics_analysis/newton_results` | Where to save plots and data |
| `--methods` | int[] | `1 2 3 4` | Which methods to run. Methods 3 and 4 require Method 2 (auto-included) |
| `--solver` | str | `auto` | Physics solver: `mujoco`, `featherstone`, or `auto` (try MuJoCo first). Method 1 requires MuJoCo |
| `--mass-uncertainty` | flag | off | Run Monte Carlo mass perturbation analysis on Method 2 |
| `--mc-samples` | int | `10` | Number of Monte Carlo samples for mass uncertainty |
| `--device` | str | `cuda:0` | GPU device |

### Method Dependencies

```
Method 1 (contact sensors)     → standalone, needs MuJoCo
Method 2 (inverse dynamics)    → standalone
Method 3 (RRA)                 → requires Method 2
Method 4 (optimization ID)     → requires Method 2
```

If you request `--methods 3 4`, Method 2 runs automatically as a prerequisite.

## Key Equations

**Inverse Dynamics (Method 2)**:
```
τ = M(q) · (q̈_desired − q̈_free)
```
where `q̈_free` = acceleration under zero torque (1 forward step with τ=0).

**RRA objective (Method 3)**:
```
min_Δcom  ||τ_root(q + Δcom)||² + λ·||Δcom||²
```

**Optimization decomposition (Method 4)**:
```
min_{F_int}  w_fint·||F_int||² + w_res·(GRF penalties)
subject to:  F_ground_A = root_res_A − F_int
             F_ground_B = root_res_B + F_int    (Newton's 3rd)
             F_ground_z ≥ 0                      (ground pushes up)
```

## File Dependencies

| File | Role |
|------|------|
| `prepare2/retarget.py` | SMPL-X → Newton joint\_q conversion, data loading |
| `prepare2/pd_utils.py` | Model building, PD gains, contact sensors |
| `prepare2/compute_torques.py` | Inverse dynamics, B-spline derivatives, Butterworth filter |
| `newton_vqvae/physics_losses.py` | Segment mass ratios (De Leva 1996) |
| `physics_analysis/dyadic_physics.md` | Theoretical framework for naive COM approach |

## Data Format

**Input**: Raw SMPL-X pkl files at `{data_dir}/motions/{clip_id}.pkl` containing per-person:
- `root_orient` (T, 3) — axis-angle root orientation
- `pose_body` (T, 63) — 21 joint axis-angles
- `trans` (T, 3) — root translation
- `betas` (10,) — SMPL-X shape parameters

**Coordinate system**: Newton/InterHuman uses Z-up. SMPL-X uses Y-up. Conversion handled by `retarget.py` via `R_ROT = [[0,0,1],[1,0,0],[0,1,0]]`.
