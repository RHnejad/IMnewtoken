# eval_pipeline/ — InterMask Physics Evaluation Pipeline

Scripts and documentation for evaluating InterMask generated motions using Newton physics.

---

## Overview

```
data/generated/{interx,interhuman}/         ← InterMask network outputs  [DONE]
data/retargeted_v2/{generated_,gt_}{interx,interhuman}/   ← Newton joint_q
data/xml_generated/{generated_,gt_}{interx,interhuman}/   ← per-subject XMLs + bone dim logs
data/compute_torques/{generated_,gt_}{interx,interhuman}/ ← PD torques
data/skyhook_metrics/{generated_,gt_}{interx,interhuman}/ ← physics quality metrics
```

**Conda environments:**
- `intermask` — InterMask model inference (`generate_and_save.py`)
- `mimickit`  — Newton physics steps (retarget, torques, skyhook)

---

## Step 1 — Generate ✅ (conda: intermask)

```bash
cd /media/rh/codes/sim/InterMask

python generate_and_save.py \
    --dataset_name interx --name trans_default \
    --which_epoch best_fid --output_dir data/generated/interx

python generate_and_save.py \
    --dataset_name interhuman --name trans_default \
    --which_epoch best_fid --output_dir data/generated/interhuman
```

Outputs:
- `data/generated/interx/generated.h5`  — `(T, 56, 6)` per clip
- `data/generated/interx/pkl/<clip>.pkl` — `{person1/person2: {root_orient,pose_body,trans,betas}}`
- `data/generated/interhuman/<clip>.pkl` — same format

---

## Step 2 — Retarget SMPL-X → Newton (conda: mimickit)

> **No `--downsample` flag** in retarget.py — it only converts coordinate frames.
> Sampling rate issue only applies in Steps 3–4 (see below).

```bash
# Generated InterX
python prepare2/retarget.py \
    --dataset interx \
    --data_dir data/generated/interx \
    --output_dir data/retargeted_v2/generated_interx

# Generated InterHuman
python prepare2/retarget.py \
    --dataset interhuman \
    --data_dir data/generated/interhuman \
    --output_dir data/retargeted_v2/generated_interhuman

# GT InterX  (default --data_dir data/Inter-X_Dataset)
python prepare2/retarget.py \
    --dataset interx \
    --output_dir data/retargeted_v2/gt_interx

# GT InterHuman  (default --data_dir data/InterHuman)
python prepare2/retarget.py \
    --dataset interhuman \
    --output_dir data/retargeted_v2/gt_interhuman
```

Per-clip outputs: `<clip>_person{0,1}.npy` (T,22,3), `_joint_q.npy` (T,76), `_betas.npy` (10,)

---

## Step 3 — XML Bone Dimensions + Consistency Log (conda: mimickit)

```bash
# Replace <case> with: generated_interx | generated_interhuman | gt_interx | gt_interhuman

python eval_pipeline/log_xml_bone_dims.py \
    --retargeted-dir data/retargeted_v2/<case> \
    --output-dir     data/xml_generated/<case>
```

Outputs per case:
- `xmls/smpl_<hash>.xml` — copies of unique per-subject XMLs
- `bone_dims.csv` — bone lengths (m) per unique subject
- `consistency.log` — mean/std/CV per bone; flags >10% variation

---

## Step 4 — PD Torques (conda: mimickit)

> ⚠ **Sampling rate**: InterMask outputs at **30fps** (already at model rate).
> Use `--downsample 1` for generated data, `--downsample 2` for GT (60fps raw → 30fps).

```bash
# Generated InterX  (--downsample 1 — already 30fps)
python prepare2/compute_torques.py \
    --dataset interx \
    --data-dir  data/retargeted_v2/generated_interx \
    --output-dir data/compute_torques/generated_interx \
    --method pd --save --downsample 1

# Generated InterHuman  (--downsample 1)
python prepare2/compute_torques.py \
    --dataset interhuman \
    --data-dir  data/retargeted_v2/generated_interhuman \
    --output-dir data/compute_torques/generated_interhuman \
    --method pd --save --downsample 1

# GT InterX  (--downsample 2, default — raw 60fps)
python prepare2/compute_torques.py \
    --dataset interx \
    --output-dir data/compute_torques/gt_interx \
    --method pd --save

# GT InterHuman  (--downsample 2, default)
python prepare2/compute_torques.py \
    --dataset interhuman \
    --output-dir data/compute_torques/gt_interhuman \
    --method pd --save
```

Output files: `<clip>_person{0,1}_torques_pd.npy` — (T, 75) joint torques

---

## Step 5 — Skyhook Metrics (conda: mimickit)

> ⚠ **Same sampling rate rule**: `--downsample 1` for generated, `--downsample 2` for GT.

```bash
# Generated InterX
python prepare2/compute_skyhook_metrics.py \
    --dataset interx \
    --data-dir  data/retargeted_v2/generated_interx \
    --output-dir data/skyhook_metrics/generated_interx \
    --downsample 1

# Generated InterHuman
python prepare2/compute_skyhook_metrics.py \
    --dataset interhuman \
    --data-dir  data/retargeted_v2/generated_interhuman \
    --output-dir data/skyhook_metrics/generated_interhuman \
    --downsample 1

# GT InterX
python prepare2/compute_skyhook_metrics.py \
    --dataset interx \
    --output-dir data/skyhook_metrics/gt_interx

# GT InterHuman
python prepare2/compute_skyhook_metrics.py \
    --dataset interhuman \
    --output-dir data/skyhook_metrics/gt_interhuman
```

Per-clip outputs: `<clip>_person{0,1}_skyhook_metrics.npz/.json`
Summary: `summary.csv`, `summary.json`

---

## Newton Visualization (headless mp4, conda: mimickit)

### Option A — Skyhook MP4 (skeleton + force plot)
```bash
python prepare2/visualize_skyhook_mp4.py \
    --clip <CLIP_ID> --dataset interx \
    --retarget-dir data/retargeted_v2/generated_interx \
    --metrics-dir  data/skyhook_metrics/generated_interx \
    --prefer-exact \
    --output /tmp/gen_interx_<CLIP_ID>_skyhook.mp4
```

### Option B — Joint_q playback MP4 (two persons side-by-side)
```bash
# Save mp4 from Newton headless viewer:
python prepare2/visualize_newton.py \
    --clip <CLIP_ID> \
    --data-dir data/retargeted_v2/generated_interx \
    --fps 30 \
    --viewer offscreen --output /tmp/gen_interx_<CLIP_ID>.mp4
```

---

## Step 6 — ImDy Physics Plausibility (conda: intermask)

ImDy inference needs a PyTorch environment with the ImDy dependencies in `prepare5/ImDy`.
In this repo, the validated environment is `intermask`.

```bash
# Generated InterHuman from retargeted positions
python eval_pipeline/imdy_scorer.py \
    --data-dir data/retargeted_v2/generated_interhuman \
    --dataset-type retargeted \
    --output-dir data/imdy_metrics/generated_interhuman \
    --imdy-config prepare5/ImDy/config/IDFD_mkr.yml \
    --imdy-checkpoint prepare5/ImDy/downloaded_checkpoint/imdy_pretrain.pt \
    --device cuda:0

# GT InterHuman from retargeted positions
python eval_pipeline/imdy_scorer.py \
    --data-dir data/retargeted_v2/interhuman \
    --dataset-type retargeted \
    --output-dir data/imdy_metrics/gt_interhuman \
    --imdy-config prepare5/ImDy/config/IDFD_mkr.yml \
    --imdy-checkpoint prepare5/ImDy/downloaded_checkpoint/imdy_pretrain.pt \
    --device cuda:1

# Compare distributions
python eval_pipeline/imdy_scorer.py --compare \
    --gt-dir data/imdy_metrics/gt_interhuman \
    --gen-dir data/imdy_metrics/generated_interhuman \
    --output data/imdy_metrics/comparison_report.json
```

Per-clip output: `<clip_id>_imdy.json`  
Dataset summary: `summary.json`  
Comparison report: `comparison_report.json`

---

## Step 7 — InterX Force × Mesh Contact Analysis (conda: intermask)

This stage aligns per-frame ImDy predictions with precomputed InterX mesh-contact JSONs and is
implemented in `eval_pipeline/force_contact_analysis.py`.

Scope:
- InterX only
- retargeted clips in `data/retargeted_v2/interx`
- mesh-contact JSONs in `output/mesh_contact/interx`

Public subcommands:
- `infer` — save one `.npz` + one `.meta.json` per clip
- `visualize` — save one `*_force_contact.png` per clip
- `analyze` — save one dataset-level JSON with per-clip paired statistics
- `aggregate` — save `action_summary.json` plus action-level plots

Contact policy:
- binary contact: `touching` or `penetrating`
- `barely_touching` is kept separate for visualization shading only
- out-of-range ImDy frame indices are dropped during alignment; they are never zero-filled

Joint labeling:
- torque heatmaps and ranked joint gaps use the 23-joint ImDy torque layout from
  `prepare5/ImDy/models/utils.py:JOINT_NAMES[:23]`
- the final joint in this layout is `jaw`, not `L_Hand`

### 7.1 Infer force arrays

```bash
python eval_pipeline/force_contact_analysis.py infer \
    --output-dir output/force_contact/interx_arrays \
    --device cpu \
    --batch-size 256 \
    --sample-per-action 20 \
    --seed 42
```

Defaults already target the InterX paths:
- `--data-dir data/retargeted_v2/interx`
- `--contact-dir output/mesh_contact/interx`
- `--imdy-config prepare5/ImDy/config/IDFD_mkr.yml`
- `--imdy-checkpoint prepare5/ImDy/downloaded_checkpoint/imdy_pretrain.pt`

Per-clip outputs:
- `<clip>.npz`
  - `torque_p0`, `torque_p1` with shape `(N, 23, 3)`
  - `grf_p0`, `grf_p1` with shape `(N, 2, 24, 3)`
  - `contact_logits_p0`, `contact_logits_p1` with shape `(N, 2, 24, 1)`
  - `frame_indices`
- `<clip>.meta.json`
  - clip ID, action class, preprocessing settings, config/checkpoint paths, seed, exact saved shapes
- `_infer_summary.json`

### 7.2 Visualize force-contact overlays

```bash
python eval_pipeline/force_contact_analysis.py visualize \
    --arrays-dir output/force_contact/interx_arrays \
    --output-dir output/force_contact/interx_plots \
    --clip-ids G005T000A000R002,G035T000A002R013
```

Each plot contains:
- mean torque magnitude for both people
- vertical GRF for both people
- mesh min distance
- contact vertex count
- torque heatmap for person 0
- torque heatmap for person 1

Shading:
- red = touching/penetrating
- orange = barely touching

### 7.3 Analyze contact vs non-contact phases

```bash
python eval_pipeline/force_contact_analysis.py analyze \
    --arrays-dir output/force_contact/interx_arrays \
    --output output/force_contact/interx_analysis/analysis_results.json
```

The saved JSON contains:
- `dataset_stats`
- `joint_gap_ranked`
- `per_clip`
- `processing_summary`

Statistics are computed on clip-level paired summaries, not raw frames, to avoid frame-level pseudo-replication.
Clips without enough contact and non-contact frames are skipped and counted in `processing_summary`.

### 7.4 Aggregate by action category

```bash
python eval_pipeline/force_contact_analysis.py aggregate \
    --analysis-results output/force_contact/interx_analysis/analysis_results.json \
    --output-dir output/force_contact/interx_aggregate
```

Outputs:
- `action_summary.json`
- `torque_gap_by_action.png`
- `contact_rate_vs_torque_gap.png`

### 7.5 Validated smoke test

Validated locally on April 8, 2026 with:
- `infer` on 5 InterX clips
- `visualize` on:
  - `G005T000A000R002` (clear contact)
  - `G035T000A002R013` (no contact)
- `analyze` on the smoke outputs
- `aggregate` on the smoke analysis JSON

Smoke outputs were written under:
- `output/force_contact/interx_smoke/arrays`
- `output/force_contact/interx_smoke/plots`
- `output/force_contact/interx_smoke/analysis`
- `output/force_contact/interx_smoke/aggregate`

---

## Output Summary Table

| Dataset       | Case       | Retargeted | XMLs | Torques | Skyhook |
|---------------|------------|------------|------|---------|---------|
| InterX        | Generated  | `data/retargeted_v2/generated_interx/` | `data/xml_generated/generated_interx/` | `data/compute_torques/generated_interx/` | `data/skyhook_metrics/generated_interx/` |
| InterX        | GT         | `data/retargeted_v2/gt_interx/` | `data/xml_generated/gt_interx/` | `data/compute_torques/gt_interx/` | `data/skyhook_metrics/gt_interx/` |
| InterHuman    | Generated  | `data/retargeted_v2/generated_interhuman/` | `data/xml_generated/generated_interhuman/` | `data/compute_torques/generated_interhuman/` | `data/skyhook_metrics/generated_interhuman/` |
| InterHuman    | GT         | `data/retargeted_v2/gt_interhuman/` | `data/xml_generated/gt_interhuman/` | `data/compute_torques/gt_interhuman/` | `data/skyhook_metrics/gt_interhuman/` |

> GT `data/retargeted_v2/interhuman/` (46k+ files) and `data/retargeted_v2/interx/` were pre-computed.
> For GT: these are symlinked/aliased as `gt_interhuman`/`gt_interx` in the pipeline commands.
