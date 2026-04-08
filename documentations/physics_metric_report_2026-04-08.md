# Physics Metric Report

Updated: 2026-04-08

This note consolidates the current status of the physics-oriented evaluation work in this repo:

- paired-vs-solo torque interaction metric (`prepare4/`)
- PHC / PP-Motion-style tracking metric (`prepare5/`)
- ImDy-based dynamics summaries (`eval_pipeline/`)
- residual wrench summaries

It also records what was already run for InterGen versus what was only planned.

---

## 1. Main conclusion

The most complete full-dataset physics report currently available in the repo is the
**paired-vs-solo torque interaction analysis on InterHuman**.

That report is already computed and stored in:

- [paired_eval_results.json](/media/rh/codes/sim/InterMask/data/paired_eval_gt/paired_eval_results.json)
- [paired_eval_results.json](/media/rh/codes/sim/InterMask/data/paired_eval_generated/paired_eval_results.json)
- [comparison.json](/media/rh/codes/sim/InterMask/output/paired_comparison/comparison.json)
- [README_PAIRED_ANALYSIS.md](/media/rh/codes/sim/InterMask/prepare4/README_PAIRED_ANALYSIS.md)

This is the "nice table" source the project already contains.

---

## 2. Metric inventory

| Metric family | Status | Dataset coverage | Main files |
|---|---|---|---|
| Paired-vs-solo torque interaction metric (`prepare4`) | Complete | InterHuman GT + InterMask generated | `data/paired_eval_gt/paired_eval_results.json`, `data/paired_eval_generated/paired_eval_results.json`, `output/paired_comparison/comparison.json` |
| PHC / PP-Motion-style tracking (`prepare5`) | Partial / exploratory | Per-clip only | `output/phc_tracker/clip_*` |
| ImDy summary metrics | Complete for several datasets | GT InterHuman, generated InterHuman, generated InterGen-InterHuman, InterX | `data/imdy_metrics/...` |
| Residual wrench metric | Complete for InterHuman GT vs generated; complete for InterGen generated-only | InterHuman + InterGen | `data/residual_wrench/comparison.json`, `data/residual_wrench_intergen/intergen_results.json` |

---

## 3. Full-dataset paired-vs-solo report

### 3.1 What it is

This is the strongest completed interaction-specific physics metric in the repo.

Method summary:

- simulate the same motion in three settings:
  - paired
  - solo A
  - solo B
- compare the torque and root-force changes induced by the partner
- large paired-vs-solo gaps indicate that the motion is not physically self-consistent

This work is documented in:

- [README_PAIRED_ANALYSIS.md](/media/rh/codes/sim/InterMask/prepare4/README_PAIRED_ANALYSIS.md:1)

### 3.2 Batch coverage

- GT clips analyzed: **196**
- Generated clips analyzed: **200**

Source files:

- [paired_eval_results.json](/media/rh/codes/sim/InterMask/data/paired_eval_gt/paired_eval_results.json:1)
- [paired_eval_results.json](/media/rh/codes/sim/InterMask/data/paired_eval_generated/paired_eval_results.json:1)

### 3.3 Key summary table

| Metric | GT | Generated | Ratio |
|---|---:|---:|---:|
| Hinge Torque Delta (A) | 4.59 Nm | 145.54 Nm | 31.7x |
| Hinge Torque Delta (B) | 4.34 Nm | 144.52 Nm | 33.3x |
| Root Force Delta (A) | 48.92 N | 434.07 N | 8.9x |
| BPS Paired (A) | 1.28% | 51.33% | 40.1x |
| Paired Hinge Torque (A) | 10.97 Nm | 145.01 Nm | 13.2x |
| Solo Hinge Torque (A) | 8.34 Nm | 7.75 Nm | ~1x |
| N3LV mean | 0.751 | 0.787 | ~1x |

These numbers come directly from:

- GT aggregated stats in [paired_eval_results.json](/media/rh/codes/sim/InterMask/data/paired_eval_gt/paired_eval_results.json:1)
- Generated aggregated stats in [paired_eval_results.json](/media/rh/codes/sim/InterMask/data/paired_eval_generated/paired_eval_results.json:1)

### 3.4 Per-body-group torque delta

| Body group | GT (Nm) | Generated (Nm) | Ratio |
|---|---:|---:|---:|
| L Leg | 9.91 | 156.69 | 15.8x |
| R Leg | 10.32 | 169.30 | 16.4x |
| Spine/Torso | 1.58 | 247.27 | 156.3x |
| L Arm | 1.63 | 83.75 | 51.5x |
| R Arm | 1.70 | 77.66 | 45.7x |

These values are also written in:

- [README_PAIRED_ANALYSIS.md](/media/rh/codes/sim/InterMask/prepare4/README_PAIRED_ANALYSIS.md:71)

### 3.5 Interpretation

The main pattern is very strong:

- generated motions look roughly normal in solo simulation
- generated motions become extremely costly when both persons are simulated together
- this means the interaction itself is the problem, not the single-body motion in isolation

In practice, this is why `prepare4` became the winning physics metric in the codebase audit:

- [codebase_audit.md](/media/rh/codes/sim/InterMask/documentations/codebase_audit.md:114)

---

## 4. PHC / PP-Motion-style tracking status

### 4.1 What exists

The repo contains a PHC-style tracking metric inspired by PP-Motion:

- [README.md](/media/rh/codes/sim/InterMask/prepare5/README.md:1)

Available outputs are per-clip only, for example:

- [clip_1000_gt](/media/rh/codes/sim/InterMask/output/phc_tracker/clip_1000_gt)
- [clip_1129_gt](/media/rh/codes/sim/InterMask/output/phc_tracker/clip_1129_gt)
- [clip_1129_generated](/media/rh/codes/sim/InterMask/output/phc_tracker/clip_1129_generated)
- [clip_500_gt](/media/rh/codes/sim/InterMask/output/phc_tracker/clip_500_gt)
- [clip_500_generated](/media/rh/codes/sim/InterMask/output/phc_tracker/clip_500_generated)

Typical files inside each per-clip directory:

- `metrics.json`
- `phc_tracking.png`
- `phc_per_joint_mpjpe.png`
- `phc_result.npz`

### 4.2 What is missing

I do **not** find a finished full-dataset aggregate PHC report analogous to the paired-vs-solo `comparison.json`.

So the PHC / PP-Motion-style metric status is:

- implemented
- tested on several clips
- **not aggregated into a full-dataset table in this repo snapshot**

This matches the handoff note that batch PHC evaluation was a plan:

- [agent_handoff_2026-03-24.md](/media/rh/codes/sim/InterMask/agent_handoff_2026-03-24.md:127)

---

## 5. InterGen status

InterGen was not fully absent. There are already completed runs for some metrics.

### 5.1 InterGen ImDy results exist

Generated InterGen motions were scored with ImDy and compared against GT InterHuman.

Files:

- [summary.json](/media/rh/codes/sim/InterMask/data/imdy_metrics/generated_intergen_interhuman/summary.json)
- [comparison_intergen_interhuman.json](/media/rh/codes/sim/InterMask/data/imdy_metrics/comparison_intergen_interhuman.json)

Coverage:

- requested: **1177**
- succeeded: **1173**
- failed: **4**

Key person-1 summary values:

| Metric | GT InterHuman | InterMask generated | InterGen generated |
|---|---:|---:|---:|
| Mean torque / BW | 0.2592 | 0.2479 | 0.2510 |
| Torque 95th percentile | 386.72 | 362.73 | 365.97 |
| Torque smoothness | 85.83 | 108.19 | 73.81 |

Sources:

- [gt_interhuman summary](/media/rh/codes/sim/InterMask/data/imdy_metrics/gt_interhuman/summary.json)
- [generated_interhuman summary](/media/rh/codes/sim/InterMask/data/imdy_metrics/generated_interhuman/summary.json)
- [generated_intergen_interhuman summary](/media/rh/codes/sim/InterMask/data/imdy_metrics/generated_intergen_interhuman/summary.json)

Interpretation:

- InterGen is closer to GT than InterMask on `torque_smoothness`
- both InterMask and InterGen have lower torque magnitudes than GT under this ImDy metric
- these are single-person dynamics summaries, not interaction-specific contact-consistency metrics

### 5.2 InterGen residual-wrench results exist

There is also an InterGen residual-wrench run:

- [intergen_results.json](/media/rh/codes/sim/InterMask/data/residual_wrench_intergen/intergen_results.json)

Coverage:

- clips: **1096**

Aggregate summary computed from the stored per-clip records:

| Metric | InterGen mean | InterGen median | InterGen p90 |
|---|---:|---:|---:|
| `F_sky_median` (N) | 84033.06 | 80252.40 | 127768.44 |
| `F_sky_norm` (BW) | 114.21 | 109.08 | 173.66 |
| `tau_sky_median` (Nm) | 105275.08 | 92930.17 | 172211.52 |
| `P_active_median` (W) | 52822.98 | 45544.64 | 95340.22 |
| `FPI` | 0.9998 | 1.0 | 1.0 |

### 5.3 What I do **not** find for InterGen

I do **not** find a completed InterGen batch run for the interaction-specific paired-vs-solo torque metric:

- no `data/paired_eval_intergen/...`
- no InterGen-specific paired comparison JSON
- no InterGen paired-vs-solo summary table

So for InterGen, the current state is:

- ImDy: **done**
- residual wrench: **done**
- paired-vs-solo torque interaction metric: **planned / not found as completed**

---

## 6. Relation to earlier plans

The repo notes show that evaluation across multiple generators was already intended:

- [supervisor_meeting_2026-03-25.md](/media/rh/codes/sim/InterMask/supervisor_meeting_2026-03-25.md:171)
- [supervisor_meeting_2026-03-25.tex](/media/rh/codes/sim/InterMask/supervisor_meeting_2026-03-25.tex:537)
- [shimmering-wobbling-volcano.md](/media/rh/codes/sim/InterMask/prepare5/shimmering-wobbling-volcano.md:164)

The clearest missing item is therefore:

1. extend the strongest interaction-specific metric (`prepare4` paired-vs-solo torque) from InterMask-generated InterHuman to InterGen-generated InterHuman

That would give a clean like-for-like comparison:

- GT InterHuman
- InterMask generated InterHuman
- InterGen generated InterHuman

under the same interaction-aware physics metric.

---

## 7. Recommended citation paths inside the repo

If you need the shortest path to the main results again later:

- full paired-vs-solo batch GT: [paired_eval_results.json](/media/rh/codes/sim/InterMask/data/paired_eval_gt/paired_eval_results.json)
- full paired-vs-solo batch generated: [paired_eval_results.json](/media/rh/codes/sim/InterMask/data/paired_eval_generated/paired_eval_results.json)
- final paired comparison report: [comparison.json](/media/rh/codes/sim/InterMask/output/paired_comparison/comparison.json)
- human-readable paired summary table: [README_PAIRED_ANALYSIS.md](/media/rh/codes/sim/InterMask/prepare4/README_PAIRED_ANALYSIS.md)
- InterGen ImDy summary: [summary.json](/media/rh/codes/sim/InterMask/data/imdy_metrics/generated_intergen_interhuman/summary.json)
- InterGen ImDy comparison: [comparison_intergen_interhuman.json](/media/rh/codes/sim/InterMask/data/imdy_metrics/comparison_intergen_interhuman.json)
- InterGen residual wrench results: [intergen_results.json](/media/rh/codes/sim/InterMask/data/residual_wrench_intergen/intergen_results.json)

---

## 8. Bottom line

The repo already contains a real full-dataset report, but it is the
**paired-vs-solo torque interaction report for InterHuman**, not the PHC / PP-Motion-style tracker.

For InterGen, the repo already contains:

- ImDy results
- residual-wrench results

but **not yet** a finished paired-vs-solo torque batch report.
