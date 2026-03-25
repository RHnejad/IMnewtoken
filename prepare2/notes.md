Let me read all the new/updated files carefully.Now let me also check the `retarget.py` and `simulate_torques.py` for any updates:OK, `simulate_torques.py` grew from 275 to 470 lines and `retarget.py` grew from ~200 to 526 lines. Let me quickly scan the changes:Now I have a thorough understanding of all the code. Here's my analysis:

---

## Problems Found

### 1. **`compute_torques.py` inverse dynamics: smoothing applied to qd/qdd instead of joint_q**

Lines 300-306:
```python
qd, qdd = compute_qd_qdd(joint_q, dt)
if smooth and T > 10:
    qd = smooth_trajectory(qd, fps, cutoff=cutoff_hz)
    qdd = smooth_trajectory(qdd, fps, cutoff=cutoff_hz)
```

This filters *after* differentiation. The whole point of filtering is to prevent noise amplification from double finite-differencing. Filtering `qd` and `qdd` after the fact doesn't help — the damage (amplified noise) is already done. The filter should be applied to `joint_q` **before** calling `compute_qd_qdd()`. Additionally, the quaternion columns (3:7) of `joint_q` cannot be filtered with a generic per-column Butterworth — they need to be converted to rotvec, filtered, converted back (as discussed in our previous conversation). The current `smooth_trajectory()` is a generic per-column filter that would break quaternion normalization.

**Fix:** Filter `joint_q` before differentiation, with special handling for the quaternion columns.

### 2. **`compute_torques.py` inverse dynamics: `model.collide(state_0)` in the zero-torque step**

Line 337:
```python
contacts = model.collide(state_0)
solver.step(state_0, state_1, control, contacts, dt)
```

The zero-torque forward step is supposed to measure free acceleration under gravity and Coriolis **only** — no ground contact. But `model.collide()` detects ground contacts, so when the character's feet touch the ground, the contact forces pollute `qdd_free`, making the resulting torques incorrect. For solo inverse dynamics (no ground plane in the model for inverse method — line 601-602 confirms this), this is actually fine since the model is built without `add_ground_plane()` for the inverse method. But `SolverFeatherstone` may still have issues without a ground plane. **This is actually handled correctly** by the `if method == "pd": builder.add_ground_plane()` conditional in `process_clip()`. Good.

However, there's a subtler issue: **the contacts variable could be None or empty when there's no ground plane, and we're passing it to `solver.step()`**. Verify that `SolverFeatherstone.step()` handles `contacts=None` or empty contacts gracefully. The existing code passes `None` in the plan but here passes `model.collide(state_0)` — this should be fine if there are no collision geoms without a ground plane, but it's redundant computation.

### 3. **`compute_torques.py` inverse dynamics: `state_1.joint_qd` pre-zeroed before step**

Line 336:
```python
state_1.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=device)
```

This zeros `state_1.joint_qd` before `solver.step()` writes to it. `solver.step()` should overwrite `state_1` completely, so this line is harmless but wasteful. However, if `SolverFeatherstone.step()` does an *incremental* update (adds to existing `joint_qd`), this zero is necessary. Check Newton docs. Likely fine, just unnecessary.

### 4. **`optimize_interaction.py`: SolverFeatherstone without ground contact**

Line 284:
```python
self.solver = newton.solvers.SolverFeatherstone(self.model)
```

But the model is built with `builder.add_ground_plane()` (line 255). The comment says Featherstone was chosen because it's differentiable. But Featherstone handles contacts very differently from SolverMuJoCo. For two people interacting, you *need* robust contact handling (both ground + inter-person). Featherstone's contact solver may be too basic, leading to interpenetration or instability. 

The code comments say "SolverSemiImplicit works in maximal coords → unstable for articulated MJCF bodies, causes NaN." This is a real finding. But the risk is that Featherstone's contact handling is insufficient for the two-person interaction scenario — the very thing we're trying to capture.

**This is the fundamental tension:** you need differentiability (Featherstone) but also robust contacts (MuJoCo solver). No easy fix — this is acknowledged in the plan. Watch for interpenetration artifacts.

### 5. **`optimize_interaction.py`: Same torque applied for ALL substeps within a frame**

Lines 479-492:
```python
self.control.joint_f = self.combined_frames[f]
for _ in range(self.sim_substeps):
    contacts = self.model.collide(self.states[step_idx])
    self.solver.step(...)
    step_idx += 1
```

The torque is set once per control frame (e.g., once per 24 substeps at 480Hz/20fps). This is correct for feedforward torques, but the PD correction that stabilizes the simulation (present in `batch_sim_solo.py` and `simulate_torques.py`) is **missing here**. The optimization uses pure feedforward (solo + delta) without any PD stabilization. This might cause the simulation to diverge, especially during contacts, making the loss landscape noisy and optimization difficult.

### 6. **`batch_sim_solo.py`: PD correction computed once per frame, not per substep**

Lines 219-269: The PD correction (`tau` based on current state) is computed **once** per frame, then the same `tau` is applied for all `sim_substeps`. In `simulate_torques.py`, PD is recomputed at each substep (lines 390-400). The per-frame approach means the PD correction is stale for 23 of 24 substeps, reducing tracking accuracy. For the feedforward + 10% PD approach, this matters less, but it's inconsistent with the single-person PD tracker.

### 7. **`batch_sim_solo.py`: GPU↔CPU transfer every substep**

Line 266:
```python
control.joint_f = wp.array(tau, dtype=wp.float32, device=device)
```

Inside the substep loop, this creates a new Warp array from numpy every substep. Since `tau` is the same for all substeps within a frame (issue #6), this is just wasted GPU transfer. Move the `wp.array()` call before the substep loop. (The updated `compute_torques.py` PD method correctly uses pre-allocated GPU buffers and Warp kernels to avoid this.)

### 8. **`optimize_interaction.py`: Gradient clipping creates new array inside optimization loop**

Lines 563-566:
```python
self.delta_all.grad = wp.array(
    (g * scale).astype(np.float32),
    dtype=wp.float32, device=self.device,
)
```

This creates a new Warp array for the gradient every time clipping is triggered. It works but is inefficient — should use an in-place Warp kernel to scale the gradient instead of CPU roundtrip (numpy → scale → new wp.array).

### 9. **`optimize_interaction.py` forward mode: delta array converted every frame**

Line 611:
```python
delta_np = self.delta_all.numpy().reshape(self.T, self.n_dof)
```

Called every frame in `_step_forward()`. This is a GPU→CPU transfer every render frame. For forward/playback mode, convert once at init or cache after optimization completes.

### 10. **`batch_sim_solo.py` + `optimize_interaction.py`: FPS/downsample mismatch risk**

`batch_sim_solo.py` defaults to `fps=30, downsample=2` (60→30fps). `optimize_interaction.py` also defaults to `fps=30, downsample=2`. But `compute_torques.py` defaults to `fps=20` and has no downsample. If the solo torques are computed at 20fps (no downsample) and then loaded by `optimize_interaction.py` which downsamples by 2, the torque array length won't match the trajectory length. The `min()` clamp on `self.T` (line 237-240) hides this silently — you get misaligned torques (frame N of torques applied to frame N*2 of the trajectory).

**This is a serious data pipeline bug.** Make sure `compute_torques.py` and the downstream consumers agree on FPS and downsampling. The solo torques should be computed at the same FPS as the optimization expects.

### 11. **`optimize_interaction.py`: No armature consistency with `compute_torques.py` inverse dynamics**

The solo torques from `inverse_dynamics()` are computed using `SolverFeatherstone` on a model **without armature modifications** (the armature is only set in `pd_tracking()`). But `optimize_interaction.py` sets `model.joint_armature` (line 274). The armature changes the effective inertia matrix: `M_effective = M + diag(armature)`. So the solo torques were computed with the natural inertia, but the optimization simulates with augmented inertia. The solo torques will be systematically wrong in the optimization context — they'll under-compensate for the added armature inertia.

**Fix:** Either add the same armature to the model used for inverse dynamics, or remove armature from the optimization model (risky for stability).

### 12. **Minor: `compute_torques.py` inverse dynamics `dt` inconsistency**

Line 338:
```python
solver.step(state_0, state_1, control, contacts, dt)
```

`dt` here is `1/fps` (e.g., 0.05s at 20fps). This is a large timestep for a physics step. The zero-torque forward dynamics step should ideally use a small dt for accuracy. A 0.05s physics step is quite large and may introduce integration error in the `qdd_free` estimate. Consider using the same `sim_dt = 1/480` as the actual simulation, or at minimum noting that this is an approximation.

### 13. **`batch_optimize_interaction.py`: Worker symlink creation race condition**

Lines 120-122:
```python
if os.path.exists(batch_path):
    os.symlink(os.path.abspath(batch_path), solo_path)
```

With multi-GPU workers, two workers might try to create the same symlink simultaneously if processing different clips that share the same person's betas. Unlikely but possible. Use `try/except FileExistsError`.

---

### Summary of severity

**Critical (affect correctness):**
1. Smoothing applied to qd/qdd instead of joint_q (#1)
2. FPS/downsample mismatch between compute_torques and optimize_interaction (#10)
3. Armature mismatch between inverse dynamics and optimization (#11)

**Moderate (affect quality/stability):**
4. No PD stabilization in optimization forward pass (#5)
5. Featherstone contact handling may be insufficient (#4)
6. Large dt in zero-torque step (#12)

**Minor (performance):**
7. Per-frame GPU transfers in batch_sim_solo substep loop (#7)
8. Gradient clipping via CPU roundtrip (#8)
9. Delta array converted every render frame (#9)
10. PD computed once per frame not per substep in batch_sim_solo (#6)