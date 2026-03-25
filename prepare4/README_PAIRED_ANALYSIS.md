# Paired vs Solo Torque Analysis for Two-Person Interaction Plausibility

A physics-based evaluation pipeline that reveals whether generated human-human interaction motions are physically plausible by comparing joint torques across three simulation scenarios.

## Motivation

Motion generation models (like InterMask) produce kinematically smooth outputs that *look* realistic, but may violate fundamental physics — especially for two-person interactions where contact forces between partners are essential. A person pushing another needs the reaction force from the partner to maintain balance; without it, the motion is physically impossible.

**Key insight:** By simulating each person both *with* and *without* their interaction partner, we can quantify whether the generated interaction forces are physically consistent. For ground-truth (mocap) motions, the torque difference between paired and solo simulations reflects real contact forces. For generated motions, large or inconsistent differences reveal physical implausibility.

### Inspiration

- **PP-Motion** (ACM MM 2025): Uses physics simulation to find the closest physically-valid motion, then measures the distance as a holistic plausibility metric.
- **PhysiInter** (arXiv 2025): Simulates two-person interactions in IsaacGym with inter-body collisions.
- **ImDy** (ICLR 2025): Predicts dynamics from motion using inverse dynamics and learned models.

## Method

### Three Simulation Scenarios

For each motion clip containing persons A and B:

1. **Paired**: Both persons simulated together in a single Newton/MuJoCo scene with inter-body collision detection enabled. Contact forces between persons are resolved by the physics engine.

2. **Solo A**: Person A simulated alone (person B removed). The same PD controller tracks person A's reference trajectory, but without any contact forces from person B.

3. **Solo B**: Person B simulated alone (person A removed).

### Physics Setup

- **Simulator**: [Newton](https://github.com/newton-physics/newton) (NVIDIA Warp + MuJoCo solver)
- **Simulation rate**: 480 Hz (16 substeps per motion frame at 30 fps)
- **Controller**: PD tracking with per-joint gains
- **Body model**: 24-body SMPL skeleton, 75 DOF (6 root + 69 hinge)
- **Mass model**: De Leva 1996 anthropometric mass fractions (75 kg total)
- **Collision**: Inter-person collisions enabled via `contype=1, conaffinity=1`; self-collisions disabled per person

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Torque Delta (TD)** | \|τ\_paired − τ\_solo\| per DOF per frame | How much the interaction changes required torques |
| **Root Force Delta** | \|F\_root\_paired − F\_root\_solo\| (DOFs 0-5) | Interaction force proxy at the pelvis |
| **Newton's 3rd Law Violation (N3LV)** | \|F\_int\_A + F\_int\_B\| / (\|F\_int\_A\| + \|F\_int\_B\| + ε) | 0 = perfect 3rd law compliance, 1 = total violation |
| **Solo Impossibility Index (SII)** | Fraction of frames where \|F\_root\_solo\| > 2 × body weight | Frames physically impossible without partner |
| **Biomechanical Plausibility Score (BPS)** | Fraction of frames exceeding human joint torque limits | Torques beyond what human muscles can produce |
| **Contact-Torque Correlation (CTC)** | Pearson(torque delta, inter-person proximity) | Whether torque changes correlate with physical closeness |

#### Biomechanical Joint Torque Limits (De Leva / Anderson et al.)

| Joint | Limit (Nm) | Joint | Limit (Nm) |
|-------|-----------|-------|-----------|
| Hip | 250 | Shoulder | 77 |
| Knee | 260 | Elbow | 80 |
| Ankle | 140 | Wrist | 18 |

## Results Summary

Evaluated on 196 GT clips and 200 Generated clips from InterHuman:

| Metric | GT | Generated | Ratio |
|--------|-----|-----------|-------|
| **Hinge Torque Delta (A)** | 4.59 Nm | 145.54 Nm | **31.7x** |
| **Hinge Torque Delta (B)** | 4.34 Nm | 144.52 Nm | **33.3x** |
| **Root Force Delta (A)** | 48.9 N | 434.1 N | **8.9x** |
| **BPS Paired (A)** | 1.3% | 51.3% | **40.1x** |
| **Paired Hinge Torque (A)** | 11.0 Nm | 145.0 Nm | **13.2x** |
| **Solo Hinge Torque (A)** | 8.3 Nm | 7.8 Nm | ~1x |
| **N3LV** | 0.75 | 0.79 | ~1x |

### Per-Body-Group Torque Delta

| Body Group | GT (Nm) | Generated (Nm) | Ratio |
|------------|---------|----------------|-------|
| L_Leg | 9.91 | 156.69 | 15.8x |
| R_Leg | 10.32 | 169.30 | 16.4x |
| Spine/Torso | 1.58 | 247.27 | **156.3x** |
| L_Arm | 1.63 | 83.75 | 51.5x |
| R_Arm | 1.70 | 77.66 | 45.7x |

### Key Findings

1. **Generated motions appear plausible in isolation but collapse when simulated together.** Solo torques are nearly identical (~8 Nm for both GT and Generated), but paired torques explode to ~145 Nm for Generated vs ~11 Nm for GT.

2. **51% of generated motion frames violate biomechanical torque limits** when both persons are simulated together, vs only 1.3% for GT.

3. **Spine/Torso shows the largest discrepancy** (156x), indicating that generated motions produce severe inter-person penetration in the torso region, requiring enormous corrective torques.

4. **The torque delta is 30x+ higher for generated motions**, meaning generated interactions are fundamentally inconsistent with the physics of contact — the motions cannot be made physically valid without large modifications.

## Usage

### Prerequisites

```bash
conda activate mimickit
# Requires: newton, warp, numpy, scipy, matplotlib
```

### Single Clip Analysis

```bash
# Analyze a specific GT clip
python prepare4/run_paired_analysis.py --clip-id 1129 --source gt

# Analyze a generated clip
python prepare4/run_paired_analysis.py --clip-id 1129 --source generated

# Specify GPU and output directory
python prepare4/run_paired_analysis.py --clip-id 1129 --source gt --device cuda:1 \
    --output-dir output/my_analysis
```

Output per clip (in `output/paired_analysis/clip_{id}_{source}/`):
- `torque_timeseries_A.png` — Paired vs solo torque time-series for person A
- `torque_timeseries_B.png` — Same for person B
- `root_forces.png` — Root translational forces: paired vs solo vs delta
- `delta_heatmap_A.png` — Joint × time heatmap of |τ\_paired − τ\_solo|
- `delta_heatmap_B.png` — Same for person B
- `newton3_compliance.png` — Newton's 3rd law analysis
- `metrics.txt` — All metrics in text format
- `paired_vs_solo.npz` — Raw torque arrays for downstream analysis

### Batch Evaluation

```bash
# Run 200 GT clips (writes to data/paired_eval_gt/)
PYTHONUNBUFFERED=1 nohup python -u prepare4/run_paired_analysis.py \
    --batch --source gt --n-clips 200 \
    > logs/batch_gt.log 2>&1 &

# Run 200 Generated clips on second GPU (writes to data/paired_eval_generated/)
PYTHONUNBUFFERED=1 nohup python -u prepare4/run_paired_analysis.py \
    --batch --source generated --n-clips 200 --device cuda:1 \
    > logs/batch_gen.log 2>&1 &

# Resume interrupted batch (skips completed clips)
python -u prepare4/run_paired_analysis.py --batch --source gt --n-clips 200 --resume
```

Output: `data/paired_eval_{source}/paired_eval_results.json` with:
- `aggregated`: Mean/std/median/P90 across all clips for each metric
- `per_clip`: Per-clip metrics, clip IDs, text labels

**Speed**: ~50s/clip for GT, ~2.5min/clip for Generated (IK retargeting overhead). 200 clips ≈ ~1.5h (GT) / ~8h (Generated).

Intermediate results saved every 20 clips for crash recovery.

### GT vs Generated Comparison

```bash
python prepare4/run_paired_analysis.py --compare \
    --gen-dir data/paired_eval_generated
```

Output (in `output/paired_comparison/`):
- `gt_vs_gen_metrics.png` — Bar chart of all key metrics side by side
- `per_group_torque_delta.png` — Per-body-group torque delta comparison
- `torque_delta_distribution.png` — Histogram of torque delta across clips
- `bps_distribution.png` — Histogram of biomechanical violations across clips
- `paired_vs_solo_scatter.png` — Scatter plot: paired vs solo torque per clip
- `sii_distribution.png` — Solo Impossibility Index distribution
- `comparison.json` — Machine-readable comparison data

## Architecture

### File Structure

```
prepare4/
├── paired_simulation.py          # Core: paired + solo PD simulation engine
│   ├── pd_forward_torques_paired()  - 2-person PD sim with contacts
│   └── compute_paired_vs_solo()     - Orchestrates all 3 scenarios
├── interaction_metrics.py        # All comparison metrics
│   ├── torque_delta()
│   ├── root_force_delta()
│   ├── newtons_third_law_violation()
│   ├── solo_impossibility_index()
│   ├── biomechanical_plausibility_score()
│   ├── contact_torque_correlation()
│   └── compute_all_metrics()
├── batch_paired_evaluation.py    # Batch processing with resume
│   ├── process_clip_paired()
│   └── batch_evaluate()
├── plot_paired_torques.py        # All visualization
│   ├── plot_paired_vs_solo_timeseries()
│   ├── plot_root_force_comparison()
│   ├── plot_torque_delta_heatmap()
│   ├── plot_newton3_compliance()
│   ├── plot_gt_vs_gen_comparison()
│   ├── plot_sii_distribution()
│   └── plot_per_group_torque_delta()
├── run_paired_analysis.py        # CLI orchestration
│   ├── analyze_single_clip()
│   ├── run_batch()
│   ├── run_comparison()
│   └── main()
└── dynamics.py                   # Modified: added set_segment_masses_multi()
```

### Dependencies on Existing Code

```
prepare2/pd_utils.py         → build_model, setup_model_properties,
                                create_mujoco_solver, build_pd_gains,
                                compute_all_pd_torques_np, init_state
prepare4/dynamics.py         → set_segment_masses_multi, De Leva constants,
                                PD gains, N_JOINT_Q, N_JOINT_QD
prepare4/retarget.py         → rotation_retarget, ik_retarget,
                                load_interhuman_pkl
prepare4/gen_xml.py          → get_or_create_xml (per-subject MJCF)
prepare4/run_full_analysis.py → pd_forward_torques (solo simulation),
                                 compute_torques_for_person, load_gt/gen_persons
```

### Data Flow

```
InterHuman PKL / Generated PKL
    ↓
retarget.py (SMPL-X → Newton joint_q)
    ↓
paired_simulation.py
    ├── pd_forward_torques_paired()  → τ_paired_A, τ_paired_B
    ├── pd_forward_torques()         → τ_solo_A
    └── pd_forward_torques()         → τ_solo_B
    ↓
interaction_metrics.py
    └── compute_all_metrics()  → TD, N3LV, SII, BPS, CTC
    ↓
plot_paired_torques.py / run_paired_analysis.py
    └── per-clip plots, batch aggregation, GT vs Gen comparison
```

## Technical Notes

### Why PD Tracking Instead of Inverse Dynamics?

Inverse dynamics computes τ = M(q)q̈ − h(q, q̇), which produces "skyhook" root forces — virtual forces at the pelvis that include gravity support. These are physically meaningless (no actuator exists there). PD tracking in forward simulation naturally handles gravity and ground contacts via the physics engine, producing physically grounded torques.

### Why CPU PD Torques for Paired Simulation?

The GPU PD kernel (`pd_torque_kernel`) hardcodes root quaternion indices at positions 3-6. For a two-person model, person B's root quaternion is at `COORDS_PER_PERSON + 3` to `COORDS_PER_PERSON + 6`, which the kernel doesn't handle. Using `compute_all_pd_torques_np()` (CPU) correctly handles multi-person DOF/coordinate offsets. This is not a bottleneck — the MuJoCo solver step dominates compute time.

### Interpreting the Torque Delta

The torque delta (paired − solo) is an **indirect proxy** for interaction forces, not a direct measurement. It captures:
- Contact forces resolved by the physics engine when both bodies are present
- Tracking quality differences (contacts may push the simulated body away from reference)
- Collision avoidance forces

For GT motions, the delta should be small and concentrated at contact moments. For generated motions with inter-person penetration, the delta is large everywhere because the physics engine is constantly pushing bodies apart.

### Limitations

1. **Collision geometry**: SMPL bodies are approximated by capsules/boxes — fine-grained hand contacts may be missed.
2. **IK quality for generated data**: Generated motions use IK from positions (~30mm MPJPE), propagating error into torque computation.
3. **PD controller artifact**: The torque delta includes PD tracking errors, not just physics. Stiff PD gains (Kp=300 for hips) can amplify small position differences.
4. **N3LV metric**: Noisy for PD-based analysis because root forces are dominated by gravity support (~300 N) rather than interaction forces (~50 N).
5. **SII with 2×BW threshold**: Too conservative — all clips show 0%. Consider lowering to 1.0×BW or using a percentile-based threshold.

## References

- **PP-Motion**: Zhao et al., "Physical-Perceptual Fidelity Evaluation for Human Motion Generation," ACM MM 2025. [arXiv](https://arxiv.org/abs/2508.08179)
- **PhysiInter**: "PhysiInter: Physically Plausible Two-Person Interaction Generation," arXiv 2025. [arXiv](https://arxiv.org/html/2506.07456)
- **ImDy**: "Inferring the Dynamics Model of Human Body from Video," ICLR 2025. [arXiv](https://arxiv.org/abs/2410.17610)
- **De Leva 1996**: "Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters," J. Biomechanics 29(9).
- **Anderson et al.**: "Maximum voluntary joint torque as a function of joint angle and angular velocity," J. Biomechanics.
- **Newton**: NVIDIA/Google DeepMind/Disney Research. [GitHub](https://github.com/newton-physics/newton)
