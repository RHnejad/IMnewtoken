# Plan: Fine-tune PHC on InterHuman with Motion-Aware Early Termination

## Context

PHC's pre-trained `phc_kp_pnn_iccv` checkpoint (4 PNN primitives, AMASS-trained) needs to be extended to track InterHuman motions. Two challenges:
1. The model has never seen InterHuman-style interactions → needs a new PNN primitive trained on InterHuman data.
2. Early termination (`env.enableEarlyTermination`) is global — it resets all episodes when any rigid body drifts >0.25m from reference. For InterHuman falling motions, this kills the episode before the fall is learned. The fix must be **per-motion**: falling sequences should never trigger early termination, standing/walking ones should.

---

## Overview of Changes

| Layer | File | What changes |
|---|---|---|
| Conversion | `interhuman_to_phc.py` | Emit `min_root_height` per motion |
| Motion lib | `PHC/phc/utils/motion_lib_base.py` | Load `_motion_allow_early_term` tensor |
| Environment | `PHC/phc/env/tasks/humanoid_im.py` | Post-mask `_terminate_buf` by motion flag |
| Config | `PHC/phc/data/cfg/env/env_im_pnn.yaml` | Document new flag (no change needed) |

---

## Step 1 — Add `min_root_height` to converted PKL

**File:** `interhuman_to_phc.py`, function `convert_person()` (~line 219)

After computing `root_trans_offset`, add one line before `return`:

```python
min_root_height = float(root_trans_offset[:, 2].numpy().min())
```

Add to the returned dict:
```python
"min_root_height": np.float32(min_root_height),
```

**Why `root_trans_offset`:** This is already in the Z-up world frame after the `upright_start` correction. Standing motions have `min ≈ 0.80–0.95 m`; lying/falling motions have `min < 0.35 m`. The sanity check already prints this value, so existing converted files can be audited with `sanity_check()` before re-running.

**Falling threshold:** `0.35 m` is the default. Make it a module-level constant (`FALLING_HEIGHT_THRESH = 0.35`) so it can be adjusted without code changes.

---

## Step 2 — Load per-motion flag in MotionLibBase

**File:** `PHC/phc/utils/motion_lib_base.py`

**2a. Initialize accumulator** — add before the `for f in tqdm(range(...)):` loop (around line 257):
```python
self._motion_allow_early_term_list = []
falling_h_thresh = getattr(config, "falling_height_thresh", 0.35)
```

**2b. Append inside the loop** (after `self._motion_lengths.append(curr_len)`, ~line 280):
```python
raw_min_h = motion_file_data.get("min_root_height", np.inf)
self._motion_allow_early_term_list.append(float(raw_min_h) >= falling_h_thresh)
```

For PKLs that don't have `min_root_height` (AMASS, legacy), `np.inf >= threshold` → `True` → early termination stays on. Backward compatible.

**2c. Convert to tensor** — after the `self._motion_lengths = torch.tensor(...)` block (~line 290):
```python
self._motion_allow_early_term = torch.tensor(
    self._motion_allow_early_term_list, device=self._device, dtype=torch.bool
)  # (num_unique_motions,) — True = allow early term, False = falling motion
```

---

## Step 3 — Motion-aware masking in `_compute_reset`

**File:** `PHC/phc/env/tasks/humanoid_im.py`

**3a. Read config flag in `__init__`** (near line 104 where `_enable_early_termination` is read):
```python
self._use_motion_aware_early_term = cfg["env"].get("useMotionAwareEarlyTerm", False)
```

**3b. Apply mask after JIT calls** — insert between line 1185 (`self._terminate_buf[:] = ...`) and line 1186 (`is_recovery = ...`):

```python
# Per-motion early termination: suppress for falling motions
if self._use_motion_aware_early_term and hasattr(self._motion_lib, '_motion_allow_early_term'):
    allow_early_term = self._motion_lib._motion_allow_early_term[self._sampled_motion_ids]  # (num_envs,) bool
    self._terminate_buf[:] = self._terminate_buf * allow_early_term.long()
    self.reset_buf[:] = torch.where(pass_time, torch.ones_like(self.reset_buf), self._terminate_buf)
```

**Why this works:** `compute_humanoid_im_reset` (JIT, line 1581) sets `reset = terminated | pass_time`. By zeroing `_terminate_buf` for falling motions and recomputing `reset_buf`, we preserve time-limit resets (motion ended) while suppressing fall-drift resets. The `is_recovery` logic at lines 1186–1188 still runs after and correctly overrides cycling states.

**Note:** `pass_time` is defined before both branches of the outer `if self.zero_out_far` block (lines 1124/1148), so it is in scope here.

---

## Step 4 — Fine-tuning: add a 5th PNN primitive

### Strategy
Use the PNN progressive training pattern: **freeze existing 4 columns, add and train column 4 (index 4) on InterHuman**. This is the `fitting=True` mode and avoids catastrophic forgetting entirely.

### 4a. Prepare checkpoint with `forward_pmcp.py`
```bash
cd PHC
python scripts/pmcp/forward_pmcp.py \
    --exp phc_kp_pnn_iccv \
    --epoch -1 \
    --idx 4          # prepare slot for primitive 4
```
This creates `output/HumanoidIm/phc_kp_pnn_iccv_prim4/Humanoid.pth` with primitives 0–3 frozen.

**Critical:** Verify this script handles `num_prim=5` correctly before training. If not, manually copy the checkpoint and set `actors_to_load=4`.

### 4b. Convert full InterHuman training split
```bash
# From repo root
python interhuman_to_phc.py \
    --split train \
    --out PHC/output/interhuman/train_phc.pkl
```
After conversion, run a quick audit:
```python
import joblib
d = joblib.load("PHC/output/interhuman/train_phc.pkl")
heights = [v["min_root_height"] for v in d.values()]
print(f"Falling motions (<0.35m): {sum(h < 0.35 for h in heights)} / {len(heights)}")
```

### 4c. Training command
```bash
cd PHC
python phc/run_hydra.py \
    learning=im_pnn_big \
    exp_name=phc_interhuman_pnn5 \
    env=env_im_pnn \
    robot=smpl_humanoid \
    robot.freeze_hand=True robot.box_body=False \
    env.obs_v=7 \
    env.motion_file=output/interhuman/train_phc.pkl \
    env.num_prim=5 \
    env.training_prim=4 \
    env.fitting=True \
    env.actors_to_load=4 \
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
    env.num_envs=1024 \
    env.enableEarlyTermination=True \
    env.useMotionAwareEarlyTerm=True \
    headless=True
```

**Key overrides explained:**
| Override | Why |
|---|---|
| `num_prim=5` | Total columns (4 existing + 1 new) |
| `training_prim=4` | Train only column 4 (0-indexed) |
| `fitting=True` | Freeze columns 0–3 |
| `actors_to_load=4` | Load existing 4-column weights |
| `env.models=[...]` | Path to existing checkpoint |
| `enableEarlyTermination=True` | Keep global flag on (masked per-motion) |
| `useMotionAwareEarlyTerm=True` | Enable per-motion suppression |

---

## Main Criticalities

### C1 — `forward_pmcp.py` compatibility
The PMCP script was written for AMASS primitives. It may not handle adding a 5th column to a 4-column checkpoint. **Verify before training.** If it fails, manually craft the checkpoint or initialize primitive 4 from random weights and pass the existing checkpoint via `actors_to_load=4`.

### C2 — `pass_time` scoping in `_compute_reset`
`pass_time` is assigned inside `if self.cycle_motion:` / `else:` (lines 1123–1148) but referenced at line 1186. Python scoping means it is visible after the block — but **check that the `cycle_motion=False` path is exercised** during training (it usually is; cycle_motion defaults to False in `env_im_pnn.yaml`).

### C3 — `min_root_height` calibration
After `upright_start=True` conversion and `fix_trans_height`, root Z values may shift. **Do not assume 0.35 m is correct without sampling**. Run the audit in step 4b and inspect 5–10 known falling sequences vs. standing sequences to pick the threshold.

### C4 — InterHuman person-pair dependency
PHC trains each person independently (one env per person). The physics controller has no awareness of the second person's body. This means contact between persons is not simulated — any falling motion that depends on a push from the other person will lack the external force in training, potentially making it unlearnable at first.

### C5 — JIT function signature is unchanged
`compute_humanoid_im_reset` is decorated `@torch.jit.script` and typed `enable_early_termination: bool`. We intentionally do NOT modify it. The post-masking happens in Python after the JIT call. This means early termination fires internally for all envs, then we zero out the result for falling motions. There is no wasted computation — but the distinction matters if you add logging inside the JIT function.

### C6 — Legacy AMASS pkls don't have `min_root_height`
The `np.inf` fallback in Step 2b ensures `allow_early_term = True` for all AMASS motions, so the new code is fully backward compatible.

---

## Pre-Training Verification Checklist

Run these in order before launching a full training job:

1. **Conversion audit** — convert 20 sequences, print `min_root_height`, visually confirm 2–3 sequences labeled as falling vs. standing.
2. **Visualizer check** — run `visualize_interhuman.py` on a falling sequence from the converted pkl; confirm the humanoid lies down without resetting.
3. **Import / init dry run:**
   ```bash
   python phc/run_hydra.py ... env.num_envs=4 test=True headless=True epoch=0
   ```
   Expect clean startup with no key mismatches and `_motion_allow_early_term` populated.
4. **Flag tensor inspection** — add a temporary `print(self._motion_lib._motion_allow_early_term.sum(), '/', len(...))` at the end of `_setup_character_props` to confirm the ratio looks right.
5. **Short training run** — run 5k steps (`max_epochs=5000`) and confirm:
   - Reward climbs above 0.4 for non-falling motions.
   - Falling-motion environments complete their full episode length without resetting.
   - No CUDA errors or shape mismatches.

---

## Critical Files

| File | Lines touched |
|---|---|
| `interhuman_to_phc.py` | `convert_person()` return dict (~line 219) |
| `PHC/phc/utils/motion_lib_base.py` | `load_motions()` loop (lines 257–296) |
| `PHC/phc/env/tasks/humanoid_im.py` | `__init__` (~line 104), `_compute_reset()` (after line 1185) |
| `PHC/phc/data/cfg/env/env_im_pnn.yaml` | Reference only (no code change needed) |
| `PHC/scripts/pmcp/forward_pmcp.py` | Investigate before use (C1) |
