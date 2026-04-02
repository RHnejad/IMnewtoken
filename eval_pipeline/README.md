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

## Step 6 — ImDy Physics Plausibility (conda: torchv2)

ImDy inference needs a PyTorch environment with ImDy dependencies (`prepare5/ImDy`).

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

## Output Summary Table

| Dataset       | Case       | Retargeted | XMLs | Torques | Skyhook |
|---------------|------------|------------|------|---------|---------|
| InterX        | Generated  | `data/retargeted_v2/generated_interx/` | `data/xml_generated/generated_interx/` | `data/compute_torques/generated_interx/` | `data/skyhook_metrics/generated_interx/` |
| InterX        | GT         | `data/retargeted_v2/gt_interx/` | `data/xml_generated/gt_interx/` | `data/compute_torques/gt_interx/` | `data/skyhook_metrics/gt_interx/` |
| InterHuman    | Generated  | `data/retargeted_v2/generated_interhuman/` | `data/xml_generated/generated_interhuman/` | `data/compute_torques/generated_interhuman/` | `data/skyhook_metrics/generated_interhuman/` |
| InterHuman    | GT         | `data/retargeted_v2/gt_interhuman/` | `data/xml_generated/gt_interhuman/` | `data/compute_torques/gt_interhuman/` | `data/skyhook_metrics/gt_interhuman/` |

> GT `data/retargeted_v2/interhuman/` (46k+ files) and `data/retargeted_v2/interx/` were pre-computed.
> For GT: these are symlinked/aliased as `gt_interhuman`/`gt_interx` in the pipeline commands.
