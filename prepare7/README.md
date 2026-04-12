# prepare7: PP-Motion Physics Plausibility Metric via ProtoMotions

## Goal

Implement PP-Motion's (ACM MM 2025) automatic physics plausibility metric using
NVIDIA ProtoMotions' pre-trained universal SMPL motion tracker. Evaluate tracking
MPJPE on GT InterHuman motions vs InterMask-generated motions as a physics
plausibility score.

**Metric:** Per-person solo tracking MPJPE (meters). Lower = more physically plausible.

## Pipeline Overview

```
InterHuman .pkl  -->  convert_interhuman_to_proto.py  -->  .motion files
                                                              |
                      submit_eval.sh (GPU node)  <------------+
                           |
                      run_evaluation.py  -->  per-batch JSON
                           |
                      merge_eval_batches.py  -->  eval_{gt,generated}.json
                           |
                      compare_results.py  -->  comparison.json + plots
```

## First-Time Setup

All scripts require the `protomotions` conda env on shared storage. Create it once:

```bash
# Inside: runai bash intintermask
cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken
bash prepare7/setup_deps.sh            # creates env + installs all deps
bash prepare7/setup_deps.sh --force    # recreate from scratch
```

This installs:
- PyTorch with CUDA
- ProtoMotions (editable install)
- Newton simulator (commit `e7a737c`, as tested by ProtoMotions)
- lightning, scikit-image, warp, dm_control, and all other deps from `requirements_newton.txt`

The env lives at `/mnt/vita/scratch/.../miniconda3/envs/protomotions` and survives
container restarts. Override with `CONDA_ENV=<name>` if needed.

## Scripts

| Script | Purpose | Runs on |
|--------|---------|---------|
| `setup_deps.sh` | Create/verify `protomotions` conda env with all deps | GPU (RunAI) |
| `convert_interhuman_to_proto.py` | Convert InterHuman .pkl -> ProtoMotions .motion | CPU (haas001) |
| `run_evaluation.py` | Run ProtoMotions tracker, save per-motion MPJPE | GPU (RunAI) |
| `merge_eval_batches.py` | Merge batched eval JSONs into one result | GPU (RunAI) |
| `submit_eval.sh` | Orchestrates batched evaluation on GPU node | GPU (RunAI) |
| `run_pp_motion.sh` | End-to-end pipeline (convert + eval + compare) | GPU (RunAI) |
| `compare_results.py` | Compare GT vs generated tracking MPJPE | CPU (haas001) |
| `record_video.py` | Record MP4 videos from .motion files (headless) | CPU or GPU |
| `debug_verify_conversion.py` | Verify conversion coordinate correctness | CPU |
| `debug_compare_motions.py` | Compare AMASS reference vs InterHuman .motion | CPU |

## Data

| Directory | Contents | Count |
|-----------|----------|-------|
| `data/interhuman_gt_motions/` | GT InterHuman .motion files | ~15,616 |
| `data/interhuman_gen_motions/` | InterMask generated .motion files | ~2,196 |
| `data/interhuman_test/` | Small test subset | 6 |

## Usage

### 1. Convert InterHuman motions (CPU)

```bash
cd prepare7/ProtoMotions
python ../../prepare7/convert_interhuman_to_proto.py \
    --interhuman-dir ../../data/InterHuman \
    --output-dir ../data/interhuman_gt_motions \
    --output-fps 30
```

### 2. Run evaluation (GPU node)

```bash
# Inside: runai bash intintermask
cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken

bash prepare7/submit_eval.sh test    # quick sanity check (6 motions)
bash prepare7/submit_eval.sh gt      # full GT eval (batched, ~15k motions)
bash prepare7/submit_eval.sh gen     # generated eval (~2k motions)
bash prepare7/submit_eval.sh both    # gt + gen + comparison
```

### 3. Compare results (CPU)

```bash
python prepare7/compare_results.py \
    --gt-json prepare7/output/eval_gt.json \
    --gen-json prepare7/output/eval_generated.json \
    --output-dir prepare7/output/comparison
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONDA_ENV` | `protomotions` | Conda env name (from user's miniconda on shared storage) |
| `NUM_ENVS` | 64 | Parallel simulation environments |
| `BATCH_SIZE` | 2000 | Max motions per eval batch (to avoid OOM) |

## Environment Details

All shell scripts source conda from the persistent shared storage path:
```
/mnt/vita/scratch/vita-staff/users/rh/miniconda3/etc/profile.d/conda.sh
```

The `protomotions` env is created by `setup_deps.sh` and contains:
- Python 3.11
- PyTorch with CUDA
- Newton simulator (commit `e7a737c` — must match ProtoMotions)
- lightning (PyTorch Lightning / Fabric)
- scikit-image, warp-lang, dm_control, and other ProtoMotions deps

On startup, `submit_eval.sh` verifies critical imports (`lightning`, `newton`, `warp`)
and exits with an error message pointing to `setup_deps.sh` if any are missing.

RunAI container workaround: `USER` and `TORCHINDUCTOR_CACHE_DIR` are exported
to avoid `getpwuid` crashes from missing passwd entries (PyTorch 2.10+).

## Logging

All scripts produce structured logs (both stdout and file):

| Log file | Source |
|----------|--------|
| `output/submit_eval_YYYYMMDD_HHMMSS.log` | Shell orchestrator (master log) |
| `output/eval_*.log` | Per-batch `run_evaluation.py` output |
| `output/eval_*_merge.log` | Batch merge step |
| `output/comparison/comparison.log` | GT vs generated comparison |
| `data/*/convert.log` | Conversion script |

**Checking for errors after a run:**

```bash
# Quick check: any errors in the master log?
grep -i "error\|fail\|exception" prepare7/output/submit_eval_*.log

# Check per-batch logs
grep -i "error\|fail\|exception" prepare7/output/eval_gt*.log

# Check final results
python -c "import json; d=json.load(open('prepare7/output/eval_gt.json')); print(f'Success: {d[\"success_rate\"]:.4f}, N={d[\"num_motions\"]}')"
```

## Video Recording

Record MP4 visualizations from `.motion` files, fully headless (no display, X11, or EGL needed):

```bash
# Inside: runai bash intintermask (or any machine with the conda env)
cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken

# Default: 3D skeleton stick-figure (matplotlib, works everywhere)
bash prepare7/submit_eval.sh record                              # 3 test clips
bash prepare7/submit_eval.sh record prepare7/data/interhuman_gt_motions 5  # 5 GT clips

# Or call directly with more options:
python prepare7/record_video.py \
    --motion-dir prepare7/data/interhuman_test \
    --output-dir prepare7/output/videos \
    --num-motions 3 --subsample 2 --trail 3

# MuJoCo renderer (prettier, but needs working EGL/osmesa in container):
bash prepare7/submit_eval.sh record prepare7/data/interhuman_test 3 mujoco
```

**Renderers:**
- `skeleton` (default): Matplotlib 3D stick-figure with color-coded limbs and ghost trails.
  Zero GL dependencies, works on any machine. Subsamples frames for speed (`--subsample 2`).
- `mujoco`: Full MuJoCo humanoid mesh rendering via EGL. Requires `MUJOCO_GL=egl` and
  working NVIDIA EGL libraries in the container.

Videos are saved to `prepare7/output/videos/`.

## Coordinate System

InterHuman .pkl data is Z-up (root_orient and trans pre-rotated via R_x(+90)).
The ProtoMotions MJCF body model (`smpl_humanoid.xml`) uses Z-up body-local
offsets. InterHuman data feeds directly into the standard AMASS conversion
pipeline (including rot1) without any coordinate undo.

**Key insight:** The R_x(-90) undo that would convert back to "AMASS Y-up" is
WRONG here, because the MJCF body model is natively Z-up in its local frame.
InterHuman's Z-up root_orient combined with rot1 produces correct Z-up world
output. This was validated by comparing body-Z-to-world-Z alignment (0.9998).

## Evaluation Metrics

- **gt_error**: MPJPE between simulated and reference joint positions (meters)
- **gr_error**: Global root position error
- **max_joint_error**: Maximum per-joint error across all joints
- **success_rate**: Fraction of motions where gt_error_max stays below 0.5m throughout
- **normalized_jerk**: Smoothness metric
- **action_delta**: Policy action smoothness (rad, deg)

## Batching

The full GT dataset (15,616 motions) causes CUDA OOM on a 32GB V100 when loaded
at once. The eval pipeline splits into batches of BATCH_SIZE motions via symlink
subdirectories, evaluates each batch independently, then merges results with
`merge_eval_batches.py`. The merge recomputes aggregate statistics from per-motion
data to ensure correctness.

## Output Format

`eval_*.json` structure:
```json
{
  "aggregate": {"eval/success_rate": 0.33, "eval/gt_error/mean": 0.54, ...},
  "per_motion": {
    "1_person1": {"file": "...", "failed": true, "gt_error_mean": 1.34, ...},
    ...
  },
  "success_rate": 0.33,
  "num_motions": 6
}
```

## Dependencies

- ProtoMotions (submodule at `prepare7/ProtoMotions/`)
- Pre-trained SMPL tracker checkpoint at `ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt`
- `protomotions` conda env (created by `setup_deps.sh`)
- Newton simulator commit `e7a737c` (installed by `setup_deps.sh`)
