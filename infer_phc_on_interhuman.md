# PHC on InterHuman — Complete Procedure

All commands run **inside the Docker container** (launched via `./docker/run.sh` from the host).

---

## 0. Start the container

```bash
# From the host machine:
cd /home/tommasovan/Repos/IMnewtoken
./docker/run.sh          # GPU + X11 forwarding
# or:
./docker/run.sh --no-gpu # CPU only (no visualization)
```

Inside the container, the repo is mounted at `/workspace/repo`.

```bash
cd /workspace/repo
```

> **Important:** The container works on `/var/tmp/imnewtoken-docker/` (a local non-NFS copy rsynced at startup).
> Output files written inside the container live at that path, not on the NFS repo.
> Use the rsync commands in Section 6 to copy them back.

---

## 1. Visualize a raw InterHuman motion (no conversion needed)

```bash
python visualize_interhuman.py \
    --motion InterHuman_dataset/motions/10.pkl \
    --smpl_data_dir PHC/data/smpl
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--motion` | required | Raw InterHuman `.pkl` or converted PHC `.pkl` |
| `--smpl_data_dir` | `PHC/data/smpl` | Directory with `SMPL_NEUTRAL.pkl` etc. |
| `--person` | `-1` (both) | `0` = person 1 only, `1` = person 2 only |
| `--fps` | `30` | Playback speed |

**Viewer controls:**

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `→` / `←` | Step +1 / −1 frame |
| `R` | Reset to frame 0 |
| `Q` / `Esc` | Quit |

> This visualizer overrides physics every frame — it shows the **reference trajectory** exactly as stored in the dataset, not a physics simulation.

---

## 2. Convert InterHuman motion to PHC format

### Single clip

```bash
python interhuman_to_phc.py \
    --motion InterHuman_dataset/motions/10.pkl \
    --out    PHC/output/interhuman/10_phc.pkl \
    --smpl_data_dir PHC/data/smpl
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--motion` | required | Input InterHuman `.pkl` |
| `--out` | required | Output path for the converted `.pkl` |
| `--smpl_data_dir` | `PHC/data/smpl` | SMPL model directory |
| `--person` | `-1` (both) | `0` or `1` to convert only one person |

The script prints a sanity check with root height stats. Expected values:
- `root Z min` ≈ 0.7–0.9 m (standing)
- `root Z mean` ≈ 0.85 m

If root Z is negative or near zero, the coordinate system is wrong.

The converted pkl contains one entry per person:
```
{
  "interhuman_p0": { pose_quat_global, pose_quat, trans_orig,
                     root_trans_offset, pose_aa, beta, gender, fps },
  "interhuman_p1": { ... }
}
```

### Full dataset (batch, resume-safe)

```bash
python prepare7/batch_convert_interhuman_to_phc.py \
    --subprocess-batch-size 300
```

This splits the dataset into batches of 300 clips and runs each batch in a **fresh subprocess**. When each subprocess exits, Python/SMPL memory is fully released at the OS level — this prevents the OOM kills that occur when converting thousands of clips in a single process.

The script is **resume-safe**: already-converted clips are detected by the presence of their staging pkl and skipped automatically.

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--interhuman-dir` | `InterHuman_dataset/motions` | Source InterHuman directory |
| `--smpl-data-dir` | `PHC/data/smpl` | SMPL model directory |
| `--staging-dir` | `PHC/output/interhuman/staging` | Per-clip intermediate pkls |
| `--output` | `PHC/output/interhuman/all_clips.pkl` | Final merged pkl |
| `--subprocess-batch-size N` | disabled | Process in subprocess batches of N |
| `--clip-ids 10,11,12` | all | Convert specific clips only |
| `--merge-only` | — | Skip conversion, only merge staging files |
| `--force` | — | Re-convert / overwrite existing |

Output keys per clip: `{clip_id}_p0`, `{clip_id}_p1`

---

## 3. Visualize the converted PHC pkl

```bash
python visualize_interhuman.py \
    --motion PHC/output/interhuman/10_phc.pkl \
    --smpl_data_dir PHC/data/smpl
```

Use this to verify the conversion is correct **before** running PHC inference.

---

## 4. Run PHC inference (physics imitation)

The pre-trained checkpoint to use is `phc_kp_pnn_iccv` (keypoint-based PNN, no shape conditioning).
It lives at `PHC/output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth`.

```bash
cd PHC

python phc/run_hydra.py \
    learning=im_pnn exp_name=phc_kp_pnn_iccv \
    epoch=-1 test=True \
    env=env_im_pnn \
    robot.freeze_hand=True robot.box_body=False env.obs_v=7 \
    env.motion_file=output/interhuman/10_phc.pkl \
    env.num_prim=4 \
    env.num_envs=2 headless=False
```

Key overrides:
| Override | Value | Why |
|----------|-------|-----|
| `env.motion_file` | path to converted pkl | relative to `PHC/` directory |
| `env.num_prim=4` | 4 | checkpoint has 4 PNN actors (default config says 3) |
| `env.num_envs` | = number of persons in pkl | 2 for both persons, 1 for single |
| `headless=False` | — | show Isaac Gym viewer window |
| `headless=True` | — | no window, faster (for torque extraction) |
| `env.enableEarlyTermination=False` | — | keep episode running even if humanoid falls |

> **Note:** `env.motion_file` paths are relative to the `PHC/` directory.
> The converted pkl is at `PHC/output/interhuman/10_phc.pkl`, so pass `output/interhuman/10_phc.pkl`.

### Disabling early termination (for falling motions)

By default PHC resets an episode when any rigid body deviates more than 0.5 m from the reference
(`terminationDistance`). Some InterHuman motions contain intentional falls — add
`env.enableEarlyTermination=False` to let them play out fully:

```bash
python phc/run_hydra.py \
    learning=im_pnn exp_name=phc_kp_pnn_iccv \
    epoch=-1 test=True \
    env=env_im_pnn \
    robot.freeze_hand=True robot.box_body=False env.obs_v=7 \
    env.motion_file=output/interhuman/10_phc.pkl \
    env.num_prim=4 \
    env.num_envs=2 headless=False \
    env.enableEarlyTermination=False
```

### Alternative: shape-conditioned model

```bash
python phc/run_hydra.py \
    learning=im_pnn exp_name=phc_shape_pnn_iccv \
    epoch=-1 test=True \
    env=env_im_pnn \
    robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
    env.motion_file=output/interhuman/10_phc.pkl \
    env.num_prim=4 \
    env.num_envs=2 headless=False
```

---

## 5. Quick reference: all available checkpoints

| `exp_name` | Model | Notes |
|------------|-------|-------|
| `phc_kp_pnn_iccv` | SMPL, keypoint obs | Recommended starting point |
| `phc_shape_pnn_iccv` | SMPL + shape, keypoint obs | Shape-conditioned |
| `phc_shape_mcp_iccv` | SMPL + shape, MCP | Needs `robot=smpl_humanoid_shape` |
| `phc_kp_mcp_iccv` | SMPL, keypoint, MCP | Needs `env=env_im_getup_mcp` |
| `phc_3` | SMPL, basic | 3 primitives (`num_prim=3`) |
| `phc_x_pnn` | SMPL-X | Needs `robot=smplx_humanoid` |

Checkpoints are in `PHC/output/HumanoidIm/<exp_name>/Humanoid.pth`.

---

## 6. Run PHC on the full dataset and export retargeted data

This produces a single pickle file with per-clip joint positions, torques, and body positions for every person in the dataset.

### What gets saved

| Field | Shape per clip | Description |
|-------|----------------|-------------|
| `dof_pos` | `T × 69` | Achieved joint positions (exponential map, rad) |
| `torques` | `T × 69` | Applied forces from Isaac Gym PD controller (N·m) |
| `body_pos` | `T × 24 × 3` | 3-D world positions of all 24 SMPL bodies (m) |
| `root_states` | `T × 7` | Root translation (3) + quaternion xyzw (4) |
| `clean_action` | `T × 69` | PD position targets output by the network |
| `obs` | `T × 945` | Policy observation vectors |
| `key_names` | `N` | Motion keys matching the clip order |

### One-command pipeline (inside Docker)

```bash
# Enter the PHC Docker container from the host:
./docker/run.sh

# Inside the container:
bash /workspace/repo/phc_full_dataset.sh
```

The script runs automatically:
1. **Batch conversion** — `prepare7/batch_convert_interhuman_to_phc.py --subprocess-batch-size 300` merges all InterHuman clips into `PHC/output/interhuman/all_clips.pkl`
2. **PHC inference** — headless `im_eval + collect_dataset` mode iterates through every clip and exits when done
3. **NFS sync** — rsyncs output from the local Docker staging area back to the NFS repo

Output file:
```
PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/all_clips/retarget_<timestamp>.pkl
```

### Loading the output

```python
import joblib
import numpy as np

data = joblib.load("PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/all_clips/retarget_<timestamp>.pkl")

keys = data["key_names"]          # e.g. ["1000_p0", "1000_p1", "1001_p0", ...]
for i, key in enumerate(keys):
    dof_pos   = data["dof_pos"][i]    # (T, 69) joint positions
    torques   = data["torques"][i]    # (T, 69) applied torques
    body_pos  = data["body_pos"][i]   # (T, 24, 3) 3-D body positions
    root      = data["root_states"][i]  # (T, 7) root pos + quat
    print(f"{key}: T={dof_pos.shape[0]}  root_z_mean={root[:, 2].mean():.3f} m")
```

### Partial conversion (subset of clips)

```bash
# Convert only clips 10–15:
python prepare7/batch_convert_interhuman_to_phc.py \
    --clip-ids 10,11,12,13,14,15 \
    --output   PHC/output/interhuman/clips_10_15.pkl

# Then run inference on the subset:
cd PHC
python phc/run_hydra.py \
    learning=im_pnn exp_name=phc_kp_pnn_iccv \
    epoch=-1 test=True im_eval=True \
    env=env_im_pnn \
    robot.freeze_hand=True robot.box_body=False env.obs_v=7 \
    env.motion_file=output/interhuman/clips_10_15.pkl \
    env.num_prim=4 env.num_envs=12 \
    env.enableEarlyTermination=False \
    ++collect_dataset=True headless=True
```

### Copying output to the host while inference is still running

The container works on `/var/tmp/imnewtoken-docker/`. To copy output to the NFS repo before the full run finishes:

```bash
# From the HOST (outside the container):
NFS=/home/tommasovan/Repos/IMnewtoken
LOCAL=/var/tmp/imnewtoken-docker

# Copy all retarget pkl files produced so far:
rsync -av --ignore-existing \
    "$LOCAL/PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/" \
    "$NFS/PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/"
```

To check how many clips have been processed from inside the container:

```bash
# From inside the container, count lines in the progress log:
ls PHC/output/interhuman/staging/ | wc -l   # staging pkls (conversion)
ls PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/ | head  # retarget pkls (inference)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `no file to resume` | Wrong `exp_name` | Check checkpoint exists in `PHC/output/HumanoidIm/` |
| `Unexpected key pnn.actors.3.*` | `num_prim` mismatch | Add `env.num_prim=4` |
| Humanoid underground | Root Z too low | Check sanity check output from conversion |
| `einsum size 10 vs 5` | Wrong beta size | Already fixed in `interhuman_to_phc.py` |
| `PyTorch imported before isaacgym` | Import order | Already fixed in scripts |
| Black/blank viewer window | X11/GPU not forwarded | Re-run `./docker/run.sh` with display set |
| Conversion `Killed` with no error | OOM: single process | Use `--subprocess-batch-size 300` |
| Inference loops forever with `....` | `collect_dataset` at wrong config level | Use `++collect_dataset=True` (root level, not `+env.collect_dataset=True`) |
| Output file not on NFS after run | Docker writes to `/var/tmp/` | Run rsync step or use `phc_full_dataset.sh` which does it automatically |
| `Key 'collect_dataset' not in struct` | Hydra strict struct mode | Use `++` prefix (force-override) instead of `+` |

---

## Implementation notes

### SMPL model used
The `phc_kp_pnn_iccv` checkpoint uses **SMPL (not SMPL-X)**. InterHuman sequences are originally in SMPL-X format; `interhuman_to_phc.py` converts them to SMPL by stripping the hand joints and projecting to the 24-body SMPL skeleton. A neutral mean shape (`beta=zeros(10)`) is used since the checkpoint is not shape-conditioned.

### Two-person handling
Each InterHuman pkl contains two persons (`person1`, `person2`). During conversion these become separate SMPL sequences. During inference, each person is simulated as an independent humanoid agent in a parallel Isaac Gym environment. The merged pkl keys are `{clip_id}_p0` and `{clip_id}_p1`.

### Torques vs. dof_force_tensor
PHC uses `control_mode=isaac_pd` — the PD controller runs inside Isaac Gym. The `self.torques` variable is only populated in `pd` mode. For `isaac_pd`, the actual applied forces are read from `self.dof_force_tensor` (Isaac Gym's force sensor). The extraction code uses `dof_force_tensor` to get physically meaningful N·m values.

### Why `++collect_dataset=True` (not `+env.collect_dataset=True`)
PHC reads the flag from the **root-level** Hydra config: `cfg.get("collect_dataset", False)` in `humanoid.py`. Passing `+env.collect_dataset=True` adds it to the `cfg.env` sub-config, which is never read. The `++` prefix overrides at root level, which is where PHC checks.

### OOM during batch conversion
Conversion of 7810 clips in a single Python process accumulates SMPL model state across clips, eventually hitting the RAM limit (~2000–2400 clips on a 64 GB machine). The fix: `--subprocess-batch-size 300` spawns a fresh Python process for each batch. When the subprocess exits, the OS reclaims all memory before the next batch starts.
