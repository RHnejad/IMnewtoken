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

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `no file to resume` | Wrong `exp_name` | Check checkpoint exists in `PHC/output/HumanoidIm/` |
| `Unexpected key pnn.actors.3.*` | `num_prim` mismatch | Add `env.num_prim=4` |
| Humanoid underground | Root Z too low | Check sanity check output from conversion |
| `einsum size 10 vs 5` | Wrong beta size | Already fixed in `interhuman_to_phc.py` |
| `PyTorch imported before isaacgym` | Import order | Already fixed in scripts |
| Black/blank viewer window | X11/GPU not forwarded | Re-run `./docker/run.sh` with display set |
