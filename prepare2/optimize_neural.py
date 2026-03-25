"""
optimize_neural.py — Neural actor optimization for interaction torques.

Replaces the flat Δq array in optimize_interaction.py with a learned MLP
actor network.  Gradient flow:

    Actor weights ─[PyTorch]→ Δq ─[scatter]→ Warp array ─[sim tape]→ loss
                                                                       │
    Actor weights ←[.backward]← Δq.grad ←[extract]← tape.backward() ──┘

Benefits over flat trajectory optimization:
    - Parameter sharing across frames → better generalization
    - Conditions on other person's state → learns reactive corrections
    - Compact representation → implicit regularization via network capacity
    - Can transfer across clips via load_model → amortized optimization

Usage:
    # Single-clip training
    python prepare2/optimize_neural.py --clip 1000 --epochs 50

    # Custom hyperparameters
    python prepare2/optimize_neural.py --clip 1000 --epochs 100 \\
        --lr 3e-4 --hidden 256 256 --window 5 --sim-freq 120

    # Fine-tune from pre-trained model
    python prepare2/optimize_neural.py --clip 2000 --epochs 30 \\
        --load-model data/retargeted_v2/interhuman/1000_actor.pt
"""
import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

import warp as wp
wp.config.verbose = False

import warnings
warnings.filterwarnings("ignore", message="Custom attribute")

import newton
from newton import CollisionPipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare2.retarget import get_or_create_xml
from prepare2.optimize_interaction import (
    compose_with_root_pd_kernel,
    position_loss_kernel,
    delta_reg_from_flat_kernel,
    DOFS_PER_PERSON, COORDS_PER_PERSON,
    N_BODIES_PER_PERSON, N_JOINTS, SMPL_TO_BODY,
)
from prepare2.actor_network import (
    InteractionActor, ObservationNormalizer,
    build_observations, build_all_observations,
    scatter_delta_to_flat, extract_hinge_grads,
    N_HINGE_DOFS, OBS_DIM,
)


# ═══════════════════════════════════════════════════════════════
# Neural Interaction Optimizer
# ═══════════════════════════════════════════════════════════════
class NeuralInteractionOptimizer:
    """Train an MLP actor through differentiable physics simulation.

    Instead of optimizing a flat (T × n_dof) delta array directly,
    this optimizer trains a neural network that maps per-frame
    observations to hinge-DOF angle offsets (Δq).  The PD controller
    amplifies Δq by ke_joint (default 200), giving a strong torque
    signal from small angle corrections.

    Training uses exact analytic gradients from Newton's differentiable
    Featherstone solver (via wp.Tape), bridged to PyTorch through a
    manual gradient transfer.

    Observation space (per person, per frame):
        - ref_hinge_angles:        69  (what this person should do)
        - other_person_positions:  66  (where the other person is)
        - solo_hinge_torques:      69  (baseline torques)
        - root_state:               7  (own root pos + quat)
        - normalized_time:          1  (temporal context)
        Total:                    212

    Output: 69 hinge angle offsets → PD(ref + Δq) + solo torques
    """

    def __init__(self, args):
        self.device = getattr(args, "device", "cuda:0")
        self.torch_device = torch.device(
            self.device if "cuda" in self.device else "cpu"
        )
        self.fps = args.fps
        self.downsample = getattr(args, "downsample", 2)
        self._clip_id = args.clip

        # ── Simulation parameters ────────────────────────────
        self.sim_freq = getattr(args, "sim_freq", 120)
        self.sim_substeps = self.sim_freq // self.fps
        self.sim_dt = 1.0 / self.sim_freq
        self.window_size = args.window
        self.torque_limit = 5000.0
        self.reg_lambda = args.reg_lambda
        self.grad_clip = getattr(args, "grad_clip", 100.0)

        # PD gains
        self.ke_root = getattr(args, "ke_root", 5000.0)
        self.kd_root = getattr(args, "kd_root", 500.0)
        self.ke_joint = getattr(args, "ke_joint", 200.0)
        self.kd_joint = getattr(args, "kd_joint", 20.0)

        # ── Load data ────────────────────────────────────────
        data_dir = args.data_dir
        clip_id = args.clip
        self._load_data(data_dir, clip_id)

        # ── Build Newton physics ─────────────────────────────
        self._build_physics()

        # ── Actor network ────────────────────────────────────
        hidden_dims = tuple(getattr(args, "hidden", [256, 256]))
        max_delta = getattr(args, "max_delta", 0.05)
        self.actor = InteractionActor(
            obs_dim=OBS_DIM,
            output_dim=N_HINGE_DOFS,
            hidden_dims=hidden_dims,
            max_delta=max_delta,
        ).to(self.torch_device)

        if getattr(args, "load_model", None):
            self.load_model(args.load_model)

        # ── PyTorch optimizer ────────────────────────────────
        lr = getattr(args, "lr", 1e-4)
        weight_decay = getattr(args, "weight_decay", 1e-5)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.param_grad_clip = getattr(args, "param_grad_clip", 1.0)

        # ── LR scheduler (cosine annealing) ──────────────────
        n_epochs = getattr(args, "epochs", 50)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=n_epochs, eta_min=lr * 0.01,
        )

        # ── Best-model tracking ──────────────────────────────
        self._best_loss = float("inf")
        self._best_state = None
        self._patience = getattr(args, "patience", 10)
        self._no_improve_count = 0

        # ── Observation normalizer ───────────────────────────
        use_norm = getattr(args, "normalize_obs", True)
        if use_norm:
            self.normalizer = ObservationNormalizer(OBS_DIM)
            self._precompute_normalization()
        else:
            self.normalizer = None

        # ── Training state ───────────────────────────────────
        self.train_iter = 0
        self.epoch = 0
        self.loss_history: list = []
        self.n_windows = max(
            1, (self.T + self.window_size - 1) // self.window_size
        )

        print(f"\nNeural optimizer ready:")
        print(f"  Actor:  {self.actor.count_parameters():,} parameters")
        print(f"  Clip:   {clip_id}  ({self.n_persons} persons, "
              f"{self.T} frames)")
        print(f"  Sim:    {self.sim_freq} Hz, "
              f"{self.sim_substeps} substeps/frame")
        print(f"  Train:  {self.n_windows} windows × "
              f"{self.window_size} frames, lr={lr}")
        print(f"  PD:     ke_root={self.ke_root}, ke_joint={self.ke_joint}")
        print(f"  Bound:  max_delta={max_delta:.3f} rad "
              f"(→ max PD torque={max_delta * self.ke_joint:.0f} Nm)")
        print(f"  Norm:   {'ON' if self.normalizer else 'OFF'}")
        print(f"  Sched:  CosineAnnealing, patience={self._patience}")

    # ─────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────
    def _load_data(self, data_dir, clip_id):
        """Load retargeted joint angles, torques, and reference positions."""
        self.all_ref_jq = []
        self.all_torques_solo = []
        self.all_qvel = []
        self.all_xml_paths = []
        self.person_indices = []

        for p_idx in [0, 1]:
            jq_path = os.path.join(
                data_dir, f"{clip_id}_person{p_idx}_joint_q.npy")
            betas_path = os.path.join(
                data_dir, f"{clip_id}_person{p_idx}_betas.npy")
            torques_path = os.path.join(
                data_dir, f"{clip_id}_person{p_idx}_torques_solo.npy")
            qvel_path = os.path.join(
                data_dir, f"{clip_id}_person{p_idx}_qvel.npy")

            # Fallback: compute_torques directory
            if not os.path.exists(torques_path):
                ct_dir = data_dir.replace("retargeted_v2", "compute_torques")
                alt = os.path.join(
                    ct_dir, f"{clip_id}_person{p_idx}_torques_solo.npy")
                if os.path.exists(alt):
                    torques_path = alt
                    alt_qv = os.path.join(
                        ct_dir, f"{clip_id}_person{p_idx}_qvel.npy")
                    if os.path.exists(alt_qv):
                        qvel_path = alt_qv

            needed = [jq_path, betas_path, torques_path]
            if not all(os.path.exists(p_) for p_ in needed):
                continue

            jq = np.load(jq_path).astype(np.float32)
            betas = np.load(betas_path)
            torques = np.load(torques_path).astype(np.float32)
            qvel = (np.load(qvel_path).astype(np.float32)
                    if os.path.exists(qvel_path)
                    else np.zeros_like(torques))
            xml_path = get_or_create_xml(betas)

            if self.downsample > 1:
                jq = jq[::self.downsample]
                torques = torques[::self.downsample]
                qvel = qvel[::self.downsample]

            self.all_ref_jq.append(jq)
            self.all_torques_solo.append(torques)
            self.all_qvel.append(qvel)
            self.all_xml_paths.append(xml_path)
            self.person_indices.append(p_idx)

        if len(self.person_indices) < 2:
            raise FileNotFoundError(
                f"Need both persons for clip {clip_id}. "
                f"Run retarget.py and compute_torques.py first.")

        self.n_persons = 2
        self.T = min(
            min(jq.shape[0] for jq in self.all_ref_jq),
            min(t.shape[0] for t in self.all_torques_solo),
        )

        # Reference body positions (T, 2, 22, 3)
        self.ref_positions = self._load_ref_positions(data_dir, clip_id)
        if self.ref_positions is not None:
            self.T = min(self.T, self.ref_positions.shape[0])
        else:
            raise FileNotFoundError(
                f"Reference positions not found for clip {clip_id}. "
                f"Need {clip_id}_person{{0,1}}.npy in {data_dir}")

        # Trim data arrays to T
        self.all_ref_jq = [jq[:self.T] for jq in self.all_ref_jq]
        self.all_torques_solo = [t[:self.T] for t in self.all_torques_solo]
        self.all_qvel = [v[:self.T] for v in self.all_qvel]

        print(f"Loaded clip {clip_id}: {self.n_persons} persons, "
              f"{self.T} frames, {self.fps} fps")

    def _load_ref_positions(self, data_dir, clip_id):
        """Load (T, 2, 22, 3) reference body positions."""
        parts = []
        for p_idx in self.person_indices:
            path = os.path.join(data_dir, f"{clip_id}_person{p_idx}.npy")
            if os.path.exists(path):
                pos = np.load(path)
                if self.downsample > 1:
                    pos = pos[::self.downsample]
                parts.append(pos)
        if len(parts) < 2:
            return None
        T = min(p.shape[0] for p in parts)
        return np.stack([p[:T] for p in parts], axis=1).astype(np.float32)

    # ─────────────────────────────────────────────────────────
    # Physics engine setup
    # ─────────────────────────────────────────────────────────
    def _build_physics(self):
        """Build Newton model, solver, collision, pre-allocate arrays."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for xml_path in self.all_xml_paths:
            builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(
            device=self.device, requires_grad=True)

        self.n_dof = self.model.joint_dof_count       # 150
        self.n_coords = self.model.joint_coord_count   # 152

        # Disable passive springs
        self.model.mujoco.dof_passive_stiffness.fill_(0.0)
        self.model.mujoco.dof_passive_damping.fill_(0.0)
        self.model.joint_target_ke.fill_(0.0)
        self.model.joint_target_kd.fill_(0.0)

        # Armature for numerical stability
        arm = np.full(self.n_dof, 0.5, dtype=np.float32)
        for i in range(self.n_persons):
            off = i * DOFS_PER_PERSON
            arm[off:off + 6] = 5.0   # higher armature for root
        self.model.joint_armature = wp.array(
            arm, dtype=wp.float32, device=self.device)

        # Solver (differentiable generalized-coordinate solver)
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        # Collision pipeline (requires_grad=True to allocate grad arrays
        # even though collide() runs outside the tape)
        self.collision_pipeline = CollisionPipeline(
            self.model, broad_phase="explicit", requires_grad=True)
        self.contacts = self.collision_pipeline.contacts()

        # Pre-allocate state chain for one window
        max_steps = self.window_size * self.sim_substeps + 1
        self.states = [self.model.state() for _ in range(max_steps)]

        # Control
        self.control = self.model.control()

        # Per-frame combined torques (separate arrays so tape retains
        # correct values for each frame during backward)
        self.combined_frames = [
            wp.zeros(self.n_dof, dtype=wp.float32,
                     device=self.device, requires_grad=True)
            for _ in range(self.window_size)
        ]

        # Solo torques (flat Warp array, no grad)
        solo_flat = np.zeros((self.T, self.n_dof), dtype=np.float32)
        for i in range(self.n_persons):
            d = i * DOFS_PER_PERSON
            solo_flat[:, d:d + DOFS_PER_PERSON] = (
                self.all_torques_solo[i][:self.T])
            solo_flat[:, d:d + 6] = 0.0   # zero root virtual forces
        self.solo_all = wp.array(
            solo_flat.flatten(), dtype=wp.float32, device=self.device)

        # Reference coords (for root PD in compose kernel)
        ref_coords = np.zeros((self.T, self.n_coords), dtype=np.float32)
        for i, jq in enumerate(self.all_ref_jq):
            c = i * COORDS_PER_PERSON
            ref_coords[:, c:c + COORDS_PER_PERSON] = jq[:self.T]
        self.ref_coords_flat = wp.array(
            ref_coords.flatten(), dtype=wp.float32, device=self.device)

        # Reference body positions for loss (one wp.vec3 array per frame)
        ref_flat = self.ref_positions.reshape(self.T, -1, 3)
        self.ref_pos_wp = [
            wp.array(ref_flat[t], dtype=wp.vec3, device=self.device)
            for t in range(self.T)
        ]

        # SMPL → Newton body index mapping
        smpl_body = np.array(
            [SMPL_TO_BODY[j] for j in range(N_JOINTS)], dtype=np.int32)
        self.smpl_to_body_wp = wp.array(
            smpl_body, dtype=wp.int32, device=self.device)

        # Loss scalar (must be pre-allocated, no creation inside tape)
        self.loss = wp.zeros(
            1, dtype=float, device=self.device, requires_grad=True)

        # Placeholder delta_all (overwritten each train_step)
        self.delta_all = wp.zeros(
            self.T * self.n_dof, dtype=wp.float32,
            device=self.device, requires_grad=True)

    def _set_state_from_ref(self, state, frame_idx):
        """Initialize simulation state from reference trajectory."""
        q = np.zeros(self.n_coords, dtype=np.float32)
        qd = np.zeros(self.n_dof, dtype=np.float32)
        for i, jq in enumerate(self.all_ref_jq):
            c = i * COORDS_PER_PERSON
            d = i * DOFS_PER_PERSON
            t = min(frame_idx, jq.shape[0] - 1)
            q[c:c + COORDS_PER_PERSON] = jq[t]
            t_v = min(t, self.all_qvel[i].shape[0] - 1)
            qd[d:d + DOFS_PER_PERSON] = self.all_qvel[i][t_v]
        state.joint_q = wp.array(
            q, dtype=wp.float32, device=self.device)
        state.joint_qd = wp.array(
            qd, dtype=wp.float32, device=self.device)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _precompute_normalization(self):
        """Compute observation statistics from all frames (once)."""
        all_obs = build_all_observations(
            self.all_ref_jq, self.ref_positions,
            self.all_torques_solo, self.T,
        )
        self.normalizer.update(all_obs)
        print(f"  Normalizer: computed stats from {all_obs.shape[0]} "
              f"observations")

    # ─────────────────────────────────────────────────────────
    # Forward simulation (inside wp.Tape)
    # ─────────────────────────────────────────────────────────
    def _forward(self, w_start, w_end):
        """Simulate one window and compute loss.

        MUST be called inside wp.Tape.
        Uses self.delta_all (set by train_step before calling).
        State initialization and collision detection must be done
        BEFORE entering the tape.
        """
        step_idx = 0
        for f, t_frame in enumerate(range(w_start, w_end)):
            dof_offset = t_frame * self.n_dof
            coord_offset = t_frame * self.n_coords

            # Compose torques: root PD + joint PD toward (ref + Δq) + solo
            wp.launch(
                compose_with_root_pd_kernel,
                dim=self.n_dof,
                inputs=[
                    self.states[step_idx].joint_q,
                    self.states[step_idx].joint_qd,
                    self.ref_coords_flat,
                    self.solo_all, self.delta_all,
                    self.combined_frames[f],
                    dof_offset, coord_offset,
                    DOFS_PER_PERSON, COORDS_PER_PERSON,
                    self.n_persons,
                    self.ke_root, self.kd_root,
                    self.ke_joint, self.kd_joint,
                    self.torque_limit,
                ],
                device=self.device,
            )

            self.control.joint_f = self.combined_frames[f]

            # Physics substeps
            for _ in range(self.sim_substeps):
                self.solver.step(
                    self.states[step_idx],
                    self.states[step_idx + 1],
                    self.control, self.contacts, self.sim_dt,
                )
                step_idx += 1

            # Position loss
            if t_frame < len(self.ref_pos_wp):
                wp.launch(
                    position_loss_kernel,
                    dim=self.n_persons * N_JOINTS,
                    inputs=[
                        self.states[step_idx].body_q,
                        self.ref_pos_wp[t_frame],
                        self.smpl_to_body_wp,
                        self.n_persons,
                        N_BODIES_PER_PERSON,
                        N_JOINTS, self.loss,
                    ],
                    device=self.device,
                )

        # L2 regularization on delta
        for t_frame in range(w_start, w_end):
            offset = t_frame * self.n_dof
            wp.launch(
                delta_reg_from_flat_kernel,
                dim=self.n_dof,
                inputs=[self.delta_all, self.reg_lambda,
                        offset, self.n_dof, self.loss],
                device=self.device,
            )

    # ─────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────
    def train_step(self, window_idx):
        """One optimization step on a single window.

        Flow:
            1. Build observations from reference data
            2. Actor forward  →  Δq (PyTorch, has grad_fn)
            3. Scatter Δq into flat array (detached numpy)
            4. Create Warp array with requires_grad
            5. Warp simulation inside tape + tape.backward
            6. Extract hinge gradients from Warp
            7. delta_q.backward(gradient) → actor parameter gradients
            8. PyTorch optimizer.step()

        Returns:
            (loss_value, gradient_norm)
        """
        w_start = window_idx * self.window_size
        w_end = min(w_start + self.window_size, self.T)

        # 1. Observations
        obs_np = build_observations(
            self.all_ref_jq, self.ref_positions,
            self.all_torques_solo,
            w_start, w_end, self.T,
        )
        if self.normalizer:
            obs_np = self.normalizer.normalize(obs_np)

        # 2. Actor forward (PyTorch)
        obs_torch = torch.from_numpy(obs_np).to(self.torch_device)
        delta_q = self.actor(obs_torch)   # (W*2, 69), has grad_fn

        # 3. Scatter to flat delta (detached, no PyTorch graph)
        delta_q_np = delta_q.detach().cpu().numpy()
        delta_flat_np = scatter_delta_to_flat(
            delta_q_np, w_start, w_end, self.T, self.n_dof,
        )

        # 4. Create Warp array
        delta_wp = wp.array(
            delta_flat_np, dtype=wp.float32,
            device=self.device, requires_grad=True,
        )
        self.delta_all = delta_wp

        # 5. Initialize state + collision (OUTSIDE tape)
        self._set_state_from_ref(self.states[0], w_start)
        self.collision_pipeline.collide(self.states[0], self.contacts)
        self.loss.zero_()

        # 6. Warp forward + backward (INSIDE tape)
        tape = wp.Tape()
        with tape:
            self._forward(w_start, w_end)
        tape.backward(self.loss)

        loss_val = float(self.loss.numpy()[0])

        # 7. Extract hinge gradients from Warp
        if delta_wp.grad is not None:
            grad_flat = delta_wp.grad.numpy()
        else:
            grad_flat = np.zeros(self.T * self.n_dof, dtype=np.float32)
            print("  WARNING: no gradient from Warp tape")

        hinge_grads = extract_hinge_grads(
            grad_flat, w_start, w_end, self.n_dof,
        )

        # Clip gradient norm
        grad_norm = float(np.linalg.norm(hinge_grads))
        if grad_norm > self.grad_clip:
            hinge_grads = hinge_grads * (self.grad_clip / grad_norm)

        # 8. PyTorch backward through actor
        grad_torch = torch.from_numpy(hinge_grads).to(self.torch_device)
        self.actor_optimizer.zero_grad()
        delta_q.backward(gradient=grad_torch)

        # Clip PyTorch parameter gradients (prevents weight explosion)
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.param_grad_clip)

        # 9. Optimizer step
        self.actor_optimizer.step()

        tape.zero()
        self.train_iter += 1
        self.loss_history.append(loss_val)

        return loss_val, grad_norm

    def train_epoch(self):
        """One pass through all windows. Returns mean epoch loss."""
        losses = []
        for w in range(self.n_windows):
            loss, _ = self.train_step(w)
            losses.append(loss)
        self.epoch += 1
        return float(np.mean(losses))

    def train(self, n_epochs):
        """Full training loop with LR scheduling, best-model tracking,
        and early stopping."""
        print(f"\nTraining: {n_epochs} epochs × "
              f"{self.n_windows} windows = "
              f"{n_epochs * self.n_windows} steps")

        t_start = time.time()
        for ep in range(n_epochs):
            ep_loss = self.train_epoch()

            # LR scheduler step (per epoch)
            self.scheduler.step()
            current_lr = self.actor_optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t_start
            rate = self.train_iter / elapsed if elapsed > 0 else 0

            # ── Best-model tracking ──────────────────────
            marker = ""
            if ep_loss < self._best_loss:
                self._best_loss = ep_loss
                self._best_state = {
                    k: v.clone() for k, v in
                    self.actor.state_dict().items()
                }
                self._no_improve_count = 0
                marker = " ★ best"
            else:
                self._no_improve_count += 1
                if self._no_improve_count >= self._patience:
                    print(f"  Early stopping: no improvement for "
                          f"{self._patience} epochs")
                    break

            # ── Estimate Δq magnitude from latest step ─────
            delta_np = self.generate_all_deltas()
            dq_max = float(np.abs(delta_np[:, 6:]).max())
            dq_mean = float(np.abs(delta_np[:, 6:]).mean())

            # Print every epoch
            print(f"  Epoch {ep + 1:3d}/{n_epochs} | "
                  f"loss={ep_loss:.6f} | "
                  f"lr={current_lr:.2e} | "
                  f"Δq={dq_mean:.4f}/{dq_max:.4f} | "
                  f"{rate:.1f} it/s | "
                  f"{elapsed:.0f}s{marker}", flush=True)

        # Restore best model
        if self._best_state is not None:
            self.actor.load_state_dict(self._best_state)
            print(f"  Restored best model (loss={self._best_loss:.6f})")

        total = time.time() - t_start
        print(f"\nTraining complete: {self.train_iter} steps, "
              f"{total:.1f}s total, "
              f"best loss={self._best_loss:.6f}")

    # ─────────────────────────────────────────────────────────
    # Inference & saving
    # ─────────────────────────────────────────────────────────
    def generate_all_deltas(self):
        """Run actor on all frames → (T, n_dof) numpy array."""
        self.actor.eval()
        with torch.no_grad():
            all_obs = build_all_observations(
                self.all_ref_jq, self.ref_positions,
                self.all_torques_solo, self.T,
            )
            if self.normalizer:
                all_obs = self.normalizer.normalize(all_obs)
            obs_torch = torch.from_numpy(all_obs).to(self.torch_device)
            delta_q = self.actor(obs_torch).cpu().numpy()  # (T*2, 69)
        self.actor.train()

        delta_flat = scatter_delta_to_flat(
            delta_q, 0, self.T, self.T, self.n_dof,
        )
        return delta_flat.reshape(self.T, self.n_dof)

    def save_results(self, output_dir):
        """Save actor model and generated deltas.

        Produces files compatible with the existing pipeline:
            {clip}_person{p}_delta_torques.npy
            {clip}_person{p}_torques_full.npy
            {clip}_actor.pt   (model checkpoint)
        """
        os.makedirs(output_dir, exist_ok=True)
        clip_id = self._clip_id

        # ── Save actor checkpoint ────────────────────────────
        model_path = os.path.join(output_dir, f"{clip_id}_actor.pt")
        save_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "epoch": self.epoch,
            "train_iter": self.train_iter,
            "loss_history": self.loss_history,
            "clip_id": clip_id,
            "obs_dim": self.actor.obs_dim,
            "output_dim": self.actor.output_dim,
            "max_delta": self.actor.max_delta,
        }
        if self.normalizer:
            save_dict["normalizer"] = self.normalizer.state_dict()
        torch.save(save_dict, model_path)
        print(f"  Actor checkpoint: {model_path}")

        # ── Generate and save deltas ─────────────────────────
        delta_np = self.generate_all_deltas()

        for i, p_idx in enumerate(self.person_indices):
            d = i * DOFS_PER_PERSON
            delta_person = delta_np[:, d:d + DOFS_PER_PERSON]
            solo_person = self.all_torques_solo[i][:self.T]

            # Full torques = solo + delta (only hinges, root stays zero)
            full_person = solo_person.copy()
            full_person[:, 6:] += delta_person[:, 6:]

            delta_path = os.path.join(
                output_dir, f"{clip_id}_person{p_idx}_delta_torques.npy")
            full_path = os.path.join(
                output_dir, f"{clip_id}_person{p_idx}_torques_full.npy")

            np.save(delta_path, delta_person.astype(np.float32))
            np.save(full_path, full_person.astype(np.float32))

        # ── Print stats ──────────────────────────────────────
        print(f"\nResults for clip {clip_id}:")
        for i, p_idx in enumerate(self.person_indices):
            d = i * DOFS_PER_PERSON
            delta = delta_np[:, d:d + DOFS_PER_PERSON]
            abs_d = np.abs(delta)
            print(f"  Person {p_idx}: "
                  f"hinge Δq mean={abs_d[:, 6:].mean():.4f} "
                  f"max={abs_d[:, 6:].max():.4f} rad, "
                  f"PD torque ≈ {abs_d[:, 6:].mean() * self.ke_joint:.1f} Nm")

    def load_model(self, path):
        """Load pre-trained actor weights."""
        checkpoint = torch.load(
            path, map_location=self.torch_device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        if "normalizer" in checkpoint and self.normalizer:
            self.normalizer.load_state_dict(checkpoint["normalizer"])
        ep = checkpoint.get("epoch", "?")
        print(f"  Loaded actor from {path} (epoch {ep})")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Neural actor optimization for interaction torques",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--clip", type=str, default="1000",
                        help="Clip ID")
    parser.add_argument("--data-dir", type=str,
                        default="data/retargeted_v2/interhuman",
                        help="Data directory with retargeted files")

    # Simulation
    parser.add_argument("--fps", type=int, default=30,
                        help="Data frame rate")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample factor (2 = 60→30fps)")
    parser.add_argument("--sim-freq", type=int, default=120,
                        help="Simulation frequency in Hz "
                             "(substeps = sim_freq / fps)")
    parser.add_argument("--window", type=int, default=5,
                        help="Frames per optimization window")

    # PD gains
    parser.add_argument("--ke-root", type=float, default=5000.0,
                        help="Root PD position gain")
    parser.add_argument("--kd-root", type=float, default=500.0,
                        help="Root PD damping gain")
    parser.add_argument("--ke-joint", type=float, default=200.0,
                        help="Joint PD position gain")
    parser.add_argument("--kd-joint", type=float, default=20.0,
                        help="Joint PD damping gain")

    # Actor network
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer dimensions")
    parser.add_argument("--max-delta", type=float, default=0.05,
                        help="Max Δq magnitude in radians (tanh-bounded). "
                             "0.05 rad × ke_joint=200 → max 10 Nm correction")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Pre-trained actor checkpoint (.pt)")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Adam weight decay")
    parser.add_argument("--reg-lambda", type=float, default=0.01,
                        help="L2 regularization on Δq")
    parser.add_argument("--grad-clip", type=float, default=10.0,
                        help="Warp gradient norm clipping threshold")
    parser.add_argument("--param-grad-clip", type=float, default=1.0,
                        help="PyTorch parameter gradient norm clipping")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--normalize-obs", action="store_true",
                        default=True,
                        help="Normalize observations (recommended)")
    parser.add_argument("--no-normalize-obs", dest="normalize_obs",
                        action="store_false",
                        help="Disable observation normalization")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as data-dir)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU device")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")

    args = parser.parse_args()

    # Resolve paths
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    if not os.path.isdir(data_dir):
        if os.path.isdir(args.data_dir):
            data_dir = args.data_dir
        else:
            print(f"ERROR: data directory not found: {data_dir}")
            sys.exit(1)
    args.data_dir = data_dir

    output_dir = args.output_dir or data_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    # Check existing output
    out_check = os.path.join(
        output_dir, f"{args.clip}_person0_delta_torques.npy")
    if os.path.exists(out_check) and not args.force:
        print(f"Output exists: {out_check}")
        print("Use --force to overwrite")
        sys.exit(0)

    # Initialize Warp
    wp.init()

    # Train
    optimizer = NeuralInteractionOptimizer(args)
    optimizer.train(args.epochs)
    optimizer.save_results(output_dir)


if __name__ == "__main__":
    main()
