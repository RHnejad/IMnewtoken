# PHC-style Physics Tracker for Newton

Physics-based motion tracker that simulates a humanoid following a reference motion, producing the **closest physically feasible version**. The tracking error (MPJPE) serves as a physical plausibility metric — inspired by [PP-Motion](https://arxiv.org/abs/2508.08179).

## How it works

1. Build a Newton/MuJoCo humanoid with SMPL body shape
2. At each frame, apply PD torques targeting the reference joint angles
3. Physics (gravity, ground contacts, inertia) corrects impossible motions
4. The simulated trajectory = closest physically feasible motion
5. `||reference - simulated||` = physical plausibility error

PD gains are matched to [PHC](https://github.com/ZhengyiLuo/PHC) (Hip/Knee kp=800, Torso kp=1000), giving ~2x better tracking than the old prepare4/ gains.

## Quick Start

```bash
conda activate mimickit

# Track a GT clip (person 1)
python prepare5/run_phc_tracker.py --clip-id 1129 --source gt

# Track a generated clip
python prepare5/run_phc_tracker.py --clip-id 1129 --source generated

# Compare PHC gains vs old PD tracker
python prepare5/run_phc_tracker.py --clip-id 1129 --source gt --compare-old

# Track both persons with inter-body contacts
python prepare5/run_phc_tracker.py --clip-id 1129 --source gt --paired

# Visualize in Newton viewer (run tracker + open viewer)
python prepare5/visualize_newton_tracking.py --clip-id 1129 --source gt --run

# Visualize saved results in Newton viewer
python prepare5/visualize_newton_tracking.py \
    --result output/phc_tracker/clip_1129_gt/phc_result.npz \
    --clip-id 1129 --source gt

# Newton viewer — paired mode
python prepare5/visualize_newton_tracking.py --clip-id 1129 --source gt --run --paired

# Newton viewer — show only simulated humanoid
python prepare5/visualize_newton_tracking.py --clip-id 1129 --source gt --run --sim-only

# Matplotlib static snapshots (alternative)
python prepare5/visualize_tracking.py --clip-id 1129 --source gt

# Matplotlib MP4 (requires ffmpeg)
python prepare5/visualize_tracking.py --clip-id 1129 --source gt --mp4
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--clip-id N` | InterHuman clip ID (default: 1129) |
| `--source gt\|generated` | Data source |
| `--device cuda:N` | GPU device (default: cuda:0) |
| `--paired` | Simulate both persons together (inter-body contacts) |
| `--compare-old` | Also run old PD tracker for comparison |
| `--gain-preset phc\|old` | PD gain set: `phc` (default, higher) or `old` (prepare2 gains) |
| `--gain-scale F` | Multiply all gains by F (default: 1.0) |
| `--use-builtin-pd` | Use Newton's built-in PD instead of explicit PD |

## Results

Tested on InterHuman clips (loaded by ID, **no train/test split enforced**):

| Clip | PHC Gains | Old Gains | Improvement |
|------|-----------|-----------|-------------|
| 1129 GT | 132 mm | 253 mm | 1.9x |
| 1000 GT | 75 mm | 70 mm | ~1x |
| 1129 Gen | 105 mm | 193 mm | 1.8x |

PHC with RL achieves ~20mm. The gap is because we use zero-residual PD (no learned corrections). Next step: optimize per-frame PD target residuals.

## Output Files

Per-clip output in `output/phc_tracker/clip_{id}_{source}/`:

| File | Contents |
|------|----------|
| `phc_result.npz` | `sim_positions`, `ref_positions`, `sim_joint_q`, `torques` |
| `metrics.json` | MPJPE, root drift, reward, terminated_at |
| `phc_tracking.png` | Per-frame MPJPE + root drift + reward time-series |
| `phc_per_joint_mpjpe.png` | Per-joint tracking error bar chart |
| `phc_vs_old_comparison.png` | PHC vs old tracker comparison (if `--compare-old`) |
| `tracking_snapshots.png` | 3D skeleton overlay at key frames (via `visualize_tracking.py`) |
| `tracking_animation.mp4` | Side-by-side animation (via `visualize_tracking.py --mp4`) |

## File Structure

```
prepare5/
├── phc_config.py                 # PD gains, reward weights, simulation params
├── phc_reward.py                 # PHC imitation reward function
├── phc_tracker.py                # Main tracker (solo + paired)
├── run_phc_tracker.py            # CLI entry point
├── visualize_newton_tracking.py  # Newton GL viewer (ref vs sim side-by-side)
├── visualize_tracking.py         # Matplotlib skeleton visualization
└── PHC/                          # Cloned PHC repo (reference only)
```

## Key Differences from prepare4/

| | prepare4/ (old) | prepare5/ (PHC-style) |
|---|---|---|
| **PD gains** | Hip kp=300 | Hip kp=800 (PHC-matched) |
| **Root gains** | pos=2000, rot=1000 | pos=5000, rot=2000 |
| **Joint limits** | Enabled | Disabled (critical for generated data) |
| **Architecture** | Monolithic functions | `PHCTracker` class |
| **Metrics** | Torque-based | MPJPE + PHC reward |
| **Multi-person** | Separate paired_simulation.py | Built into `track_paired()` |
