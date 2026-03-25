"""
view_gt_vs_gen.py — Newton GUI viewer comparing GT vs Generated motions.

Shows multiple representations side-by-side at different Y offsets:
  1. GT (Newton body) — ground truth mocap at Y=0
  2. Gen (Newton body) — generated positions solved via IK at Y=+offset
  3. Gen (Raw positions) — the ACTUAL InterMask output: raw 22-joint positions
     drawn as colored stick figures at Y=+2*offset.
  4. ID Torque Sim — GT re-simulated through physics with inverse dynamics
     torques (optional, --id-torque)
  5. PD Torque Sim — GT re-simulated through physics with PD tracking
     (optional, --pd-torque)

The torque-driven cases (#4 and #5) pass computed torques to the Newton
physics simulator (SolverMuJoCo) and record the resulting motion. This
shows how the character moves when driven by actual physical forces
(gravity, contacts, inertia) rather than kinematic FK.

Usage:
    # All three groups
    conda run -n mimickit --no-capture-output python prepare4/view_gt_vs_gen.py --clip 1

    # Only GT + Raw positions (skip the broken SMPL body)
    conda run -n mimickit --no-capture-output python prepare4/view_gt_vs_gen.py --clip 1 --no-smpl-body

    # With torque-driven simulations (adds rows for ID and PD sim)
    conda run -n mimickit --no-capture-output python prepare4/view_gt_vs_gen.py --clip 1 --id-torque --pd-torque

    # Only GT
    conda run -n mimickit --no-capture-output python prepare4/view_gt_vs_gen.py --clip 1 --gt-only
"""
import os
import sys
import time
import pickle
import warnings
import numpy as np

# Enable pyglet EGL headless mode BEFORE any pyglet/Newton import
# so it works over SSH without X11 OpenGL context
if "--save-mp4" in sys.argv or "--headless" in sys.argv:
    import pyglet
    pyglet.options["headless"] = True

import warp as wp
wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton
import newton.examples

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare4.retarget import rotation_retarget, ik_retarget
from prepare4.gen_xml import get_or_create_xml


# ═══════════════════════════════════════════════════════════════
# SMPL-X 22-joint kinematic tree for stick-figure drawing
# ═══════════════════════════════════════════════════════════════

SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),       # Pelvis → L_Hip, R_Hip, Spine1
    (1, 4), (2, 5), (3, 6),       # → L_Knee, R_Knee, Spine2
    (4, 7), (5, 8), (6, 9),       # → L_Ankle, R_Ankle, Spine3
    (7, 10), (8, 11), (9, 12),    # → L_Foot, R_Foot, Neck
    (9, 13), (9, 14),             # → L_Collar, R_Collar
    (12, 15),                      # → Head
    (13, 16), (14, 17),           # → L_Shoulder, R_Shoulder
    (16, 18), (17, 19),           # → L_Elbow, R_Elbow
    (18, 20), (19, 21),           # → L_Wrist, R_Wrist
]


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_gt_persons(clip_id):
    """Load GT SMPL-X params → list of (joint_q, betas, label)."""
    from prepare4.retarget import load_interhuman_pkl
    gt_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
    persons = load_interhuman_pkl(gt_dir, str(clip_id))
    if persons is None:
        return []
    results = []
    for i, p in enumerate(persons):
        jq = rotation_retarget(p["root_orient"], p["pose_body"], p["trans"], p["betas"])
        results.append((jq, p["betas"], f"GT_p{i}"))
    return results


def load_gen_persons(clip_id):
    """Load Generated positions → IK solve → list of (joint_q, betas, label).
    Uses position-based IK to get proper SMPL bodies from the 22-joint positions
    (the actual InterMask output), instead of the wrong rot6d conversion."""
    path = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman", f"{clip_id}.pkl")
    if not os.path.isfile(path):
        return []
    with open(path, "rb") as f:
        raw = pickle.load(f)
    results = []
    for i, pkey in enumerate(["person1", "person2"]):
        if pkey not in raw:
            continue
        p = raw[pkey]
        if "positions_zup" not in p:
            print(f"  WARNING: {pkey} has no positions_zup, skipping IK")
            continue
        positions_zup = p["positions_zup"].astype(np.float64)
        betas = p["betas"].astype(np.float64)
        print(f"  IK solving Gen {pkey}: {positions_zup.shape[0]} frames ...")
        jq, fk_pos = ik_retarget(positions_zup, betas, ik_iters=50,
                                  sequential=True)
        # Report IK error
        err = np.linalg.norm(fk_pos - positions_zup, axis=-1).mean()
        print(f"    IK MPJPE: {err*1000:.1f}mm")
        results.append((jq, betas, f"Gen_p{i}"))
    return results


def load_gen_positions(clip_id):
    """
    Load raw 22-joint positions from generated pkl.
    Returns list of (positions_zup, label).
    positions_zup: (T, 22, 3) in Z-up world frame — the actual InterMask output.
    """
    path = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman", f"{clip_id}.pkl")
    if not os.path.isfile(path):
        return []
    with open(path, "rb") as f:
        raw = pickle.load(f)
    results = []
    for i, pkey in enumerate(["person1", "person2"]):
        if pkey not in raw:
            continue
        p = raw[pkey]
        if "positions_zup" not in p:
            print(f"  WARNING: {pkey} has no 'positions_zup'. "
                  "Re-run generate_and_save.py to include raw positions.")
            continue
        pos = p["positions_zup"].astype(np.float32)  # (T, 22, 3)
        results.append((pos, f"Pos_p{i}"))
    return results


def load_gt_positions(clip_id):
    """
    Load GT 22-joint positions from raw NPY files (Z-up world frame).
    These are the same joint positions that InterMask trains on, before
    the canonicalization (centering, facing normalization) in process_motion_np.
    Returns list of (positions_zup, label).
    """
    gt_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
    results = []
    for pidx in [1, 2]:
        npy_path = os.path.join(gt_dir, "motions_processed",
                                f"person{pidx}", f"{clip_id}.npy")
        if not os.path.isfile(npy_path):
            return []

        # Raw NPY first 66 dims are positions in Z-up world frame
        npy_raw = np.load(npy_path).astype(np.float32)
        positions_zup = npy_raw[:, :66].reshape(-1, 22, 3)  # (T, 22, 3) Z-up
        results.append((positions_zup, f"GTPos_p{pidx - 1}"))

    return results


def load_rec_positions(clip_id):
    """
    Load positions from the reconstructed_dataset (same data used by InterMask
    MP4 visualizations). These are in Y-up processed frame (262-dim, first 66 cols).
    Converts to Z-up for Newton viewer.
    Returns list of (positions_zup, label).
    """
    INV_TRANS = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    rec_dir = os.path.join(PROJECT_ROOT, "data", "reconstructed_dataset",
                           "interhuman", "processed_262")
    results = []
    for pidx in [1, 2]:
        npy_path = os.path.join(rec_dir, f"{clip_id}_person{pidx}.npy")
        if not os.path.isfile(npy_path):
            return []
        motion_262 = np.load(npy_path).astype(np.float32)
        pos_yup = motion_262[:, :66].reshape(-1, 22, 3)
        pos_zup = np.einsum("mn,...n->...m", INV_TRANS, pos_yup)
        results.append((pos_zup, f"RecPos_p{pidx - 1}"))
    return results


# ═══════════════════════════════════════════════════════════════
# Torque-driven pre-simulation
# ═══════════════════════════════════════════════════════════════

def presimulate_torque(joint_q, betas, mode="pd", dt=1/30, device="cuda:0",
                       verbose=True, settle_frames=15):
    """Pre-simulate a trajectory driven by torques through Newton physics.

    Uses Newton's built-in PD actuators (joint_target_ke/kd with
    JointTargetMode.POSITION) so MuJoCo handles PD internally at every
    integration substep — no custom warp kernels needed.

    For "pd" mode: built-in PD tracks the reference trajectory.
    For "id" mode: inverse dynamics torques provide feedforward via
    control.joint_f on hinges, with built-in PD for drift correction.

    A static settle phase holds the initial pose for ``settle_frames``
    before motion begins, allowing the physics engine to establish
    ground contacts and equilibrium (prevents falling at start).

    Args:
        joint_q: (T, 76) reference joint coordinates
        betas: (10,) SMPL-X shape parameters
        mode: "id" (ID feedforward + built-in PD) or "pd" (pure built-in PD)
        dt: timestep (1/fps)
        device: CUDA device
        verbose: print progress
        settle_frames: number of frames to hold initial pose for settling
                       (default 15 ≈ 0.5s at 30fps)

    Returns:
        sim_jq: (T, 76) simulated joint coordinates
    """
    from newton import JointTargetMode
    from prepare4.dynamics import (
        inverse_dynamics, set_segment_masses, _setup_model_for_id,
        PD_GAINS, ROOT_POS_KP, ROOT_POS_KD, ROOT_ROT_KP, ROOT_ROT_KD,
        DEFAULT_TORQUE_LIMIT, BODY_NAMES as DYN_BODY_NAMES,
        N_JOINT_Q, N_JOINT_QD,
    )

    T = joint_q.shape[0]
    fps = round(1.0 / dt)
    sim_freq = 480
    sim_steps = sim_freq // fps
    dt_sim = 1.0 / sim_freq

    # ── Build model ───────────────────────────────────────────
    xml_path = get_or_create_xml(betas)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    set_segment_masses(model, total_mass=75.0, verbose=False)
    _setup_model_for_id(model, device=device)

    n_dof = model.joint_dof_count

    # ── Configure built-in PD actuators ───────────────────────
    kp_np = np.zeros(n_dof, dtype=np.float32)
    kd_np = np.zeros(n_dof, dtype=np.float32)
    mode_np = np.full(n_dof, int(JointTargetMode.POSITION), dtype=np.int32)

    kp_np[:3] = ROOT_POS_KP
    kd_np[:3] = ROOT_POS_KD
    kp_np[3:6] = ROOT_ROT_KP
    kd_np[3:6] = ROOT_ROT_KD
    for b_idx, name in enumerate(DYN_BODY_NAMES[1:]):
        s = 6 + b_idx * 3
        kp_val, kd_val = PD_GAINS.get(name, (100, 10))
        kp_np[s:s + 3] = kp_val
        kd_np[s:s + 3] = kd_val

    model.joint_target_ke = wp.array(kp_np, dtype=wp.float32, device=device)
    model.joint_target_kd = wp.array(kd_np, dtype=wp.float32, device=device)
    model.joint_target_mode = wp.array(mode_np, dtype=wp.int32, device=device)

    # ── ID feedforward torques (pre-computed) ─────────────────
    id_tau_np = None
    if mode == "id":
        if verbose:
            print("  Computing inverse dynamics torques...")
        id_tau_np, _, _ = inverse_dynamics(
            joint_q, dt, betas, device=device, verbose=verbose,
        )
        id_tau_np = id_tau_np.astype(np.float32)

    # ── Solver + initial state ────────────────────────────────
    solver = newton.solvers.SolverMuJoCo(
        model, solver="newton",
        njmax=450, nconmax=150,
        impratio=10, iterations=100, ls_iterations=50,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    jq_f32 = joint_q.astype(np.float32)
    state_0.joint_q = wp.array(jq_f32[0], dtype=wp.float32, device=device)
    state_0.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=device)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # Pre-build per-frame target arrays (joint_target_pos is in qd-space,
    # i.e. 75 DOFs: 6 root + 69 hinge — NOT 76 joint_q)
    # For the root, target_pos[0:3] = translation, target_pos[3:6] = rotation
    # For hinges, target_pos[6:] = hinge angles (= joint_q[7:])

    sim_jq = np.zeros((T, N_JOINT_Q), dtype=np.float32)
    sim_jq[0] = jq_f32[0]

    if verbose:
        print(f"  Pre-simulating {T} frames ({mode} mode, "
              f"{sim_steps} substeps/frame, {sim_freq} Hz, "
              f"built-in PD, settle={settle_frames} frames)...")

    # ── Settle phase: hold first frame to establish contacts ──
    if settle_frames > 0:
        ref0 = jq_f32[0]
        settle_target = np.zeros(n_dof, dtype=np.float32)
        settle_target[:3] = ref0[:3]
        settle_target[3:6] = ref0[3:6]
        settle_target[6:] = ref0[7:]
        control.joint_target_pos = wp.array(
            settle_target, dtype=wp.float32, device=device)

        for sf in range(settle_frames):
            for sub in range(sim_steps):
                state_0.clear_forces()
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, dt_sim)
                state_0, state_1 = state_1, state_0

        if verbose:
            cq = state_0.joint_q.numpy()
            settle_pos_err = np.linalg.norm(cq[:3] - ref0[:3])
            settle_hinge_err = (np.abs(cq[7:] - ref0[7:]).mean()
                                * 180 / np.pi)
            print(f"    Settle done ({settle_frames} frames): "
                  f"pos_err={settle_pos_err * 100:.1f}cm "
                  f"hinge_err={settle_hinge_err:.1f}°")

    # ── Simulation loop ───────────────────────────────────────
    ff_buf = np.zeros(n_dof, dtype=np.float32)
    for t in range(1, T):
        ref = jq_f32[t]

        # Build target_pos in qd-space (75 DOFs)
        target_pos = np.zeros(n_dof, dtype=np.float32)
        target_pos[:3] = ref[:3]            # root translation
        target_pos[3:6] = ref[3:6]          # root orientation (first 3 of quat)
        target_pos[6:] = ref[7:]            # hinge angles

        control.joint_target_pos = wp.array(
            target_pos, dtype=wp.float32, device=device)

        # For ID mode: feedforward hinge torques via joint_f
        if mode == "id" and id_tau_np is not None:
            ff_buf[:] = 0.0
            ff_buf[6:] = id_tau_np[t, 6:]
            np.clip(ff_buf, -DEFAULT_TORQUE_LIMIT, DEFAULT_TORQUE_LIMIT,
                    out=ff_buf)
            control.joint_f = wp.array(
                ff_buf, dtype=wp.float32, device=device)

        for sub in range(sim_steps):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt_sim)
            state_0, state_1 = state_1, state_0

        sim_jq[t] = state_0.joint_q.numpy()

        if verbose and t % 50 == 0:
            pos_err = np.linalg.norm(sim_jq[t, :3] - ref[:3])
            hinge_err = (np.abs(sim_jq[t, 7:] - ref[7:]).mean()
                         * 180 / np.pi)
            print(f"    Frame {t}/{T}: pos_err={pos_err * 100:.1f}cm "
                  f"hinge_err={hinge_err:.1f}°")

    if verbose:
        all_pos_err = np.linalg.norm(
            sim_jq[:, :3] - jq_f32[:, :3], axis=-1,
        )
        all_hinge_err = (np.abs(sim_jq[:, 7:] - jq_f32[:, 7:]).mean()
                         * 180 / np.pi)
        print(f"  Done. Mean pos_err={all_pos_err.mean() * 100:.1f}cm, "
              f"mean hinge_err={all_hinge_err:.1f}°")

    return sim_jq


# ═══════════════════════════════════════════════════════════════
# Viewer
# ═══════════════════════════════════════════════════════════════

class GTvsGenVisualizer:
    """Newton viewer: GT bodies + optional Gen bodies + raw position stick-figures."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = args.fps
        self.device = args.device
        self.sim_time = 0.0
        self._wall_start = None

        clip_id = args.clip
        fix_ground = getattr(args, 'fix_ground', True)
        show_smpl_body = getattr(args, 'smpl_body', True)

        # ── Compute dynamic y_offset from position extents ───
        y_offset = args.y_offset
        if y_offset <= 0:
            # Auto-compute: measure Y-extent of all positions
            gt_positions_raw = load_gt_positions(clip_id)
            gen_positions_raw = load_gen_positions(clip_id)
            all_y_extents = []
            for pos_list in [gt_positions_raw, gen_positions_raw]:
                if not pos_list:
                    continue
                all_y = np.concatenate([p[:, :, 1] for p, _ in pos_list])
                extent = float(all_y.max() - all_y.min())
                all_y_extents.append(extent)
            if all_y_extents:
                max_extent = max(all_y_extents)
                # Add 1m margin on each side
                y_offset = max_extent + 2.0
            else:
                y_offset = 4.0  # fallback
            print(f"  Auto y_offset: {y_offset:.1f}m "
                  f"(max Y-extent={max(all_y_extents) if all_y_extents else 0:.1f}m)")
        args.y_offset = y_offset  # store back for legend text

        # ── Load articulated-body entries (GT + optionally Gen SMPL) ─
        all_entries = []  # list of (joint_q, betas, label, y_offset)

        if not args.gen_only:
            gt_persons = load_gt_persons(clip_id)
            for jq, betas, label in gt_persons:
                if args.person is not None and not label.endswith(f"p{args.person}"):
                    continue
                # GT PKL is 60fps, downsample 2x to match InterMask's 30fps
                jq = jq[::2]
                all_entries.append((jq, betas, label, 0.0))
            print(f"GT: {len([e for e in all_entries if 'GT' in e[2]])} person(s) "
                  f"(60fps→30fps downsampled)")

        if not args.gt_only and show_smpl_body:
            gen_persons = load_gen_persons(clip_id)
            for jq, betas, label in gen_persons:
                if args.person is not None and not label.endswith(f"p{args.person}"):
                    continue
                all_entries.append((jq, betas, label, args.y_offset))
            print(f"Gen (SMPL body via IK from positions): "
                  f"{len([e for e in all_entries if 'Gen' in e[2]])} person(s)")

        # ── Torque-driven pre-simulation entries ──────────────
        # Compute next available positive Y-offset slot for torque groups
        max_pos_slot = 0
        if not args.gt_only and show_smpl_body:
            max_pos_slot = 1
        if not args.gt_only:
            max_pos_slot = max(max_pos_slot, 2 if show_smpl_body else 1)
        if getattr(args, 'rec_data', False):
            max_pos_slot = max(max_pos_slot,
                               3 if show_smpl_body else 2)
        next_torque_slot = max_pos_slot + 1

        self.torque_groups = []  # (label, y_off, mode) for gui legend

        if getattr(args, 'id_torque', False) and not args.gen_only:
            gt_body_entries = [(jq, b, l, y)
                               for jq, b, l, y in all_entries
                               if l.startswith("GT")]
            if gt_body_entries:
                print(f"\n── ID Torque Pre-simulation ──")
                for jq, betas, label, _ in gt_body_entries:
                    sim_jq = presimulate_torque(
                        jq, betas, mode="id",
                        dt=1.0 / args.fps, device=self.device,
                        verbose=True,
                    )
                    y_off = args.y_offset * next_torque_slot
                    sim_label = f"ID_{label}"
                    all_entries.append((sim_jq, betas, sim_label, y_off))
                    self.torque_groups.append(
                        (sim_label, y_off, "id"))
                    print(f"  {sim_label}: {sim_jq.shape[0]} frames "
                          f"at Y={y_off:.1f}")
                next_torque_slot += 1

        if getattr(args, 'pd_torque', False):
            body_entries = [(jq, b, l, y)
                            for jq, b, l, y in all_entries
                            if l.startswith("GT") or l.startswith("Gen")]
            if body_entries:
                print(f"\n── PD Torque Pre-simulation ──")
                for jq, betas, label, _ in body_entries:
                    sim_jq = presimulate_torque(
                        jq, betas, mode="pd",
                        dt=1.0 / args.fps, device=self.device,
                        verbose=True,
                    )
                    y_off = args.y_offset * next_torque_slot
                    sim_label = f"PD_{label}"
                    all_entries.append((sim_jq, betas, sim_label, y_off))
                    self.torque_groups.append(
                        (sim_label, y_off, "pd"))
                    print(f"  {sim_label}: {sim_jq.shape[0]} frames "
                          f"at Y={y_off:.1f}")
                next_torque_slot += 1

        # ── Load raw position stick-figures ──────────────────
        self.pos_entries = []  # list of (positions, label, y_offset)

        # GT positions as stick-figures (same representation as InterMask)
        if not args.gen_only:
            gt_pos_y_offset = args.y_offset * (-1)  # GT positions at negative Y
            gt_positions = load_gt_positions(clip_id)
            for pos, label in gt_positions:
                if args.person is not None and not label.endswith(f"p{int(label[-1])}"):
                    continue
                pos_shifted = pos.copy()
                pos_shifted[:, :, 1] += gt_pos_y_offset
                self.pos_entries.append((pos_shifted, label, gt_pos_y_offset))
            if gt_positions:
                print(f"GT (positions stick-figure): {len(gt_positions)} person(s) "
                      f"(Y-offset={gt_pos_y_offset:.1f}m)")

        # Gen positions as stick-figures
        if not args.gt_only:
            pos_y_offset = args.y_offset * (2 if show_smpl_body else 1)
            gen_positions = load_gen_positions(clip_id)
            for pos, label in gen_positions:
                if args.person is not None and not label.endswith(f"p{args.person}"):
                    continue
                # Apply Y-offset to all joint positions
                pos_shifted = pos.copy()
                pos_shifted[:, :, 1] += pos_y_offset
                self.pos_entries.append((pos_shifted, label, pos_y_offset))
            if self.pos_entries:
                gen_count = len([e for e in self.pos_entries if e[1].startswith("Pos")])
                if gen_count > 0:
                    print(f"Gen (raw positions, CORRECT): {gen_count} person(s) "
                          f"(Y-offset={pos_y_offset:.1f}m)")

        # Reconstructed_dataset positions (same data as InterMask MP4s)
        if getattr(args, 'rec_data', False):
            rec_y_offset = args.y_offset * (3 if show_smpl_body else 2)
            rec_positions = load_rec_positions(clip_id)
            for pos, label in rec_positions:
                if args.person is not None and not label.endswith(f"p{args.person}"):
                    continue
                pos_shifted = pos.copy()
                pos_shifted[:, :, 1] += rec_y_offset
                self.pos_entries.append((pos_shifted, label, rec_y_offset))
            if rec_positions:
                print(f"Rec (reconstructed_dataset): {len(rec_positions)} person(s) "
                      f"(Y-offset={rec_y_offset:.1f}m)")

        # ── Motion text ──────────────────────────────────────
        self.motion_texts = []
        self._clip_id = clip_id
        annots_path = os.path.join(PROJECT_ROOT, "data", "InterHuman", "annots", f"{clip_id}.txt")
        if os.path.isfile(annots_path):
            with open(annots_path) as f:
                self.motion_texts = [l.strip() for l in f if l.strip()]
            if self.motion_texts:
                print(f"\nMotion: {self.motion_texts[0]}")

        # ── Need at least something ──────────────────────────
        if not all_entries and not self.pos_entries:
            raise FileNotFoundError(f"No data found for clip {clip_id}")

        self.all_entries = all_entries
        self.n_persons = len(all_entries)

        # ── Fix ground penetration for body entries ──────────
        if fix_ground and all_entries:
            from prepare4.retarget import forward_kinematics as fk
            for group_prefix in ["GT", "Gen", "ID_", "PD_"]:
                group_idxs = [i for i, (_, _, lbl, _) in enumerate(all_entries)
                              if lbl.startswith(group_prefix)]
                if not group_idxs:
                    continue
                global_min_z = 0.0
                for idx in group_idxs:
                    jq, betas, label, y_off = all_entries[idx]
                    pos = fk(jq, betas)
                    min_z = float(pos[:, :, 2].min())
                    global_min_z = min(global_min_z, min_z)
                if global_min_z < 0:
                    z_lift = -global_min_z + 0.005
                    print(f"  {group_prefix}: lifting Z by {z_lift:.4f}m")
                    for idx in group_idxs:
                        jq, betas, label, y_off = all_entries[idx]
                        jq[:, 2] += z_lift
                        all_entries[idx] = (jq, betas, label, y_off)

        # ── Fix ground for position entries ──────────────────
        if fix_ground and self.pos_entries:
            global_min_z = min(float(pos[:, :, 2].min()) for pos, _, _ in self.pos_entries)
            if global_min_z < 0:
                z_lift = -global_min_z + 0.005
                print(f"  Pos: lifting Z by {z_lift:.4f}m")
                for i, (pos, lbl, y_off) in enumerate(self.pos_entries):
                    pos[:, :, 2] += z_lift
                    self.pos_entries[i] = (pos, lbl, y_off)

        # ── Build Newton model (for articulated bodies) ──────
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        if all_entries:
            for jq, betas, label, y_off in all_entries:
                xml_path = get_or_create_xml(betas)
                builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.jqd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32,
                            device=self.device)

        # ── Determine frame count ────────────────────────────
        frame_counts = [jq.shape[0] for jq, _, _, _ in all_entries]
        frame_counts += [pos.shape[0] for pos, _, _ in self.pos_entries]
        self.T = min(frame_counts) if frame_counts else 1
        self.n_per = 76  # joint_q columns per person

        # ── Apply Y-offsets to body joint_q ──────────────────
        for i, (jq, betas, label, y_off) in enumerate(all_entries):
            if y_off != 0:
                jq[:, 1] += y_off
                all_entries[i] = (jq, betas, label, y_off)

        # ── Set initial frame ────────────────────────────────
        self.frame = 0
        self._set_frame(0)
        self.viewer.set_model(self.model)

        # ── Camera ───────────────────────────────────────────
        self._setup_camera()

        print(f"\nClip {clip_id}: {self.n_persons} body(ies) + "
              f"{len(self.pos_entries)} stick-figure(s), "
              f"{self.T} frames ({self.T / self.fps:.1f}s)")
        for jq, betas, label, y_off in all_entries:
            print(f"  {label}: {jq.shape[0]} frames, Y={y_off:.1f}m")
        for pos, label, y_off in self.pos_entries:
            print(f"  {label}: {pos.shape[0]} frames, Y={y_off:.1f}m (stick-figure)")
        print("Close viewer to exit.")

    def _set_frame(self, t):
        """Set articulated bodies to frame t."""
        if not self.all_entries:
            return
        combined_n = self.model.joint_coord_count
        combined_q = np.zeros(combined_n, dtype=np.float32)
        for i, (jq, betas, label, y_off) in enumerate(self.all_entries):
            base = i * self.n_per
            frame_t = min(t, jq.shape[0] - 1)
            combined_q[base:base + self.n_per] = jq[frame_t]
        self.state.joint_q = wp.array(combined_q, dtype=wp.float32,
                                      device=self.device)
        newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

    def _draw_stick_figures(self, t):
        """Draw raw position stick-figures using viewer.log_lines + log_points."""
        # Different colors for GT vs Gen vs Rec positions
        gt_colors = [
            (0.2, 0.5, 1.0),   # blue for GT person 0
            (0.8, 0.2, 0.8),   # purple for GT person 1
        ]
        gen_colors = [
            (0.0, 1.0, 0.4),   # bright green for Gen person 0
            (1.0, 0.6, 0.0),   # orange for Gen person 1
        ]
        rec_colors = [
            (1.0, 1.0, 0.0),   # yellow for Rec person 0
            (0.0, 1.0, 1.0),   # cyan for Rec person 1
        ]
        gt_idx = 0
        gen_idx = 0
        rec_idx = 0
        for pi, (pos, label, y_off) in enumerate(self.pos_entries):
            frame_t = min(t, pos.shape[0] - 1)
            joints = pos[frame_t]  # (22, 3)

            if label.startswith("GTPos"):
                color = gt_colors[gt_idx % len(gt_colors)]
                gt_idx += 1
            elif label.startswith("RecPos"):
                color = rec_colors[rec_idx % len(rec_colors)]
                rec_idx += 1
            else:
                color = gen_colors[gen_idx % len(gen_colors)]
                gen_idx += 1

            # Joints as spheres
            n_joints = joints.shape[0]
            pts = wp.array(joints, dtype=wp.vec3, device=self.device)
            radii = wp.array(np.full(n_joints, 0.03, dtype=np.float32),
                             dtype=wp.float32, device=self.device)
            # log_points needs wp.array for colors (no auto-convert from tuple)
            colors_arr = wp.full(n_joints, wp.vec3(*color),
                                 dtype=wp.vec3, device=self.device)
            self.viewer.log_points(f"pos_{label}", pts, radii, colors_arr)

            # Bones as lines (log_lines auto-converts tuple colors)
            starts_np = np.array([joints[p] for p, c in SMPL_BONES
                                  if p < 22 and c < 22], dtype=np.float32)
            ends_np = np.array([joints[c] for p, c in SMPL_BONES
                                if p < 22 and c < 22], dtype=np.float32)
            starts_wp = wp.array(starts_np, dtype=wp.vec3, device=self.device)
            ends_wp = wp.array(ends_np, dtype=wp.vec3, device=self.device)
            self.viewer.log_lines(f"bone_{label}", starts_wp, ends_wp,
                                  color, width=0.012)

    def _setup_camera(self):
        """Point camera at the scene center, using the selected preset."""
        centers = []
        for jq, _, _, _ in self.all_entries:
            centers.append(jq[0, :3])
        for pos, _, _ in self.pos_entries:
            centers.append(pos[0, 0])
        if not centers:
            return
        center = np.mean(centers, axis=0)
        cx, cy = float(center[0]), float(center[1])

        # Compute scene extent in Y to scale camera distance
        y_vals = [float(c[1]) for c in centers]
        y_extent = max(y_vals) - min(y_vals) if y_vals else 4.0
        # Scale camera distance to fit scene vertically.
        # Sub-linear scaling so torque groups don't push the camera too far.
        scale = max(1.0, (y_extent / 8.0) ** 0.65)

        preset = getattr(self.args, 'cam_preset', 'side')
        if preset == 'side':
            dist = 10.0 * scale
            height = 3.0 * scale
            pitch, yaw = -10.0, 180.0
            cam_pos = wp.vec3(cx + dist, cy, height)
        elif preset == 'top':
            dist = 7.0 * scale
            height = 11.0 * scale
            pitch, yaw = -60.0, 90.0
            cam_pos = wp.vec3(cx, cy - dist * 0.3, height)
        elif preset == 'quarter':
            dist = 8.5 * scale
            height = 4.5 * scale
            pitch, yaw = -20.0, 135.0
            cam_pos = wp.vec3(cx + dist * 0.7, cy - dist * 0.7, height)
        else:  # 'front'
            dist = 5.0 * scale
            height = 2.0 * scale
            pitch, yaw = -15.0, 90.0
            cam_pos = wp.vec3(cx, cy - dist, height)

        # CLI overrides
        if self.args.cam_dist is not None:
            dist = self.args.cam_dist
            # Only override magnitude, keep preset direction
            if preset == 'side':
                cam_pos = wp.vec3(cx + dist, cy, height)
            elif preset == 'top':
                cam_pos = wp.vec3(cx, cy - dist * 0.3, height)
            elif preset == 'quarter':
                cam_pos = wp.vec3(cx + dist * 0.7, cy - dist * 0.7, height)
            else:
                cam_pos = wp.vec3(cx, cy - dist, height)
        if self.args.cam_height is not None:
            cam_pos = wp.vec3(float(cam_pos[0]), float(cam_pos[1]),
                              self.args.cam_height)
        if self.args.cam_pitch is not None:
            pitch = self.args.cam_pitch
        if self.args.cam_yaw is not None:
            yaw = self.args.cam_yaw

        self.viewer.set_camera(cam_pos, pitch, yaw)
        print(f"Camera: preset={preset}, dist={dist:.1f}, "
              f"height={float(cam_pos[2]):.1f}, pitch={pitch:.1f}°, "
              f"yaw={yaw:.1f}°")

    def step(self):
        """Advance one frame."""
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        now = time.perf_counter()
        self.sim_time = now - self._wall_start
        self.frame = int(self.sim_time * self.fps) % self.T
        self._set_frame(self.frame)

    def render(self):
        """Render bodies + stick figures."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self._draw_stick_figures(self.frame)
        self.viewer.end_frame()

    def gui(self, imgui):
        """Side-panel info displayed in the Newton GL viewer window."""
        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(0.4, 1.0, 0.8, 1.0), "[ GT vs GEN VIEWER ]")
        imgui.separator()

        imgui.text(f"Clip:    {self._clip_id}")
        pct = int(100 * self.frame / max(self.T - 1, 1))
        imgui.text(f"Frame:   {self.frame} / {self.T - 1}  ({pct}%)")
        imgui.text(f"FPS:     {self.fps}")
        imgui.text(f"Bodies:  {self.n_persons}")
        imgui.text(f"Sticks:  {len(self.pos_entries)}")
        imgui.separator()

        # Group legend
        imgui.text_colored(
            imgui.ImVec4(1.0, 1.0, 0.5, 1.0), "Groups:")
        for jq, betas, label, y_off in self.all_entries:
            if label.startswith("GT_"):
                color = imgui.ImVec4(0.6, 0.9, 0.6, 1.0)
            elif label.startswith("Gen_"):
                color = imgui.ImVec4(0.9, 0.9, 0.6, 1.0)
            elif label.startswith("ID_"):
                color = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
            elif label.startswith("PD_"):
                color = imgui.ImVec4(0.4, 0.4, 1.0, 1.0)
            else:
                color = imgui.ImVec4(0.7, 0.7, 0.7, 1.0)
            imgui.text_colored(color,
                               f"  {label} [Y={y_off:.1f}]")

        for pos, label, y_off in self.pos_entries:
            if label.startswith("GTPos"):
                color = imgui.ImVec4(0.3, 0.6, 1.0, 1.0)
            elif label.startswith("Pos"):
                color = imgui.ImVec4(0.0, 1.0, 0.4, 1.0)
            elif label.startswith("RecPos"):
                color = imgui.ImVec4(1.0, 1.0, 0.0, 1.0)
            else:
                color = imgui.ImVec4(0.7, 0.7, 0.7, 1.0)
            imgui.text_colored(color,
                               f"  {label} [Y={y_off:.1f}]")

        # Motion description — full text, wrapped
        if self.motion_texts:
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(1.0, 1.0, 0.5, 1.0),
                "Motion Description:")
            imgui.spacing()
            for i, text in enumerate(self.motion_texts, 1):
                imgui.text_colored(
                    imgui.ImVec4(0.7, 0.7, 0.7, 1.0), f"[{i}]")
                imgui.same_line()
                imgui.text_wrapped(text)

        # Torque simulation info
        if self.torque_groups:
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.6, 0.3, 1.0),
                "Torque Simulations:")
            for label, y_off, mode in self.torque_groups:
                mode_str = ("Inverse Dynamics" if mode == "id"
                            else "PD Tracking")
                imgui.text_wrapped(
                    f"  {label}: {mode_str} @ Y={y_off:.1f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip", type=str, required=True, help="Clip ID")
    parser.add_argument("--person", type=int, default=None,
                        help="Person index (0 or 1)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Playback FPS (InterMask data is 30fps)")
    parser.add_argument("--gt-only", action="store_true",
                        help="Show only GT")
    parser.add_argument("--gen-only", action="store_true",
                        help="Show only Generated (skip GT)")
    parser.add_argument("--y-offset", type=float, default=0,
                        help="Y offset between groups (0=auto from motion extent)")
    parser.add_argument("--no-fix-ground", dest="fix_ground",
                        action="store_false",
                        help="Disable auto ground-contact correction")
    parser.add_argument("--no-smpl-body", dest="smpl_body",
                        action="store_false",
                        help="Skip broken SMPL body (Gen group 2)")
    parser.add_argument("--rec-data", action="store_true",
                        help="Also show reconstructed_dataset positions "
                             "(same data as InterMask MP4s)")
    parser.add_argument("--save-mp4", type=str, default=None,
                        help="Save MP4 to this path (headless, no GUI)")
    parser.add_argument("--mp4-width", type=int, default=1280,
                        help="MP4 frame width")
    parser.add_argument("--mp4-height", type=int, default=720,
                        help="MP4 frame height")
    # Camera controls
    parser.add_argument("--cam-preset",
                        choices=["front", "side", "top", "quarter"],
                        default="side",
                        help="Camera preset (default: side)")
    parser.add_argument("--cam-yaw", type=float, default=None,
                        help="Override camera yaw (degrees)")
    parser.add_argument("--cam-pitch", type=float, default=None,
                        help="Override camera pitch (degrees)")
    parser.add_argument("--cam-dist", type=float, default=None,
                        help="Override camera distance")
    parser.add_argument("--cam-height", type=float, default=None,
                        help="Override camera Z height")
    parser.add_argument("--id-torque", action="store_true",
                        help="Add inverse dynamics torque-driven "
                             "simulation (pre-simulates GT through "
                             "Newton physics with ID torques)")
    parser.add_argument("--pd-torque", action="store_true",
                        help="Add PD tracking torque-driven simulation "
                             "(pre-simulates GT through Newton physics "
                             "with PD control)")
    parser.set_defaults(fix_ground=True, smpl_body=True)

    # If --save-mp4 is given, force headless mode
    import sys as _sys
    if "--save-mp4" in _sys.argv:
        # Inject --headless before newton parses args
        if "--headless" not in _sys.argv:
            _sys.argv.append("--headless")

    viewer, args = newton.examples.init(parser)
    if not hasattr(args, 'device') or args.device is None:
        args.device = "cuda:0"
    example = GTvsGenVisualizer(viewer, args)

    if args.save_mp4:
        # Headless MP4 recording loop
        import imageio
        from PIL import Image, ImageDraw, ImageFont
        mp4_path = args.save_mp4
        os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)
        writer = imageio.get_writer(mp4_path, fps=args.fps, codec='libx264',
                                     quality=8)

        # ── Build legend entries ─────────────────────────────
        y_off = args.y_offset
        show_body = getattr(args, 'smpl_body', True)
        legend = []
        if not args.gen_only:
            legend.append(("GT Bodies (SMPL)", (200, 180, 150),
                           f"Y=0"))
            legend.append(("GT Sticks (blue/purple)", (50, 128, 255),
                           f"Y={-y_off:.0f}"))
        if not args.gt_only and show_body:
            legend.append(("Gen Bodies (IK→SMPL)", (180, 200, 150),
                           f"Y={y_off:.0f}"))
        if not args.gt_only:
            pos_y = y_off * (2 if show_body else 1)
            legend.append(("Gen Sticks (green/orange)", (0, 255, 100),
                           f"Y={pos_y:.0f}"))
        if getattr(args, 'rec_data', False):
            rec_y = y_off * (3 if show_body else 2)
            legend.append(("Rec Sticks (yellow/cyan)", (255, 255, 0),
                           f"Y={rec_y:.0f}"))

        # Torque simulation legend entries
        for label, t_y_off, mode in example.torque_groups:
            if mode == "id":
                legend.append(("ID Torque Sim", (255, 100, 100),
                               f"Y={t_y_off:.0f}"))
            else:
                legend.append(("PD Torque Sim", (100, 100, 255),
                               f"Y={t_y_off:.0f}"))

        # Get motion text for title
        annots_path = os.path.join(PROJECT_ROOT, "data", "InterHuman",
                                   "annots", f"{args.clip}.txt")
        motion_text = ""
        if os.path.isfile(annots_path):
            with open(annots_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                motion_text = lines[0]

        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/"
                                      "DejaVuSansMono.ttf", 14)
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/"
                                            "DejaVuSansMono-Bold.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_title = font

        def overlay_legend(frame_np, t, total):
            """Draw legend + frame counter on a frame."""
            img = Image.fromarray(frame_np)
            draw = ImageDraw.Draw(img, "RGBA")

            # Semi-transparent background
            box_w = 400
            motion_lines = []
            if motion_text:
                words = motion_text.split()
                cur_line = ""
                for w in words:
                    test = f"{cur_line} {w}".strip() if cur_line else w
                    if len(test) <= 45:
                        cur_line = test
                    else:
                        if cur_line:
                            motion_lines.append(cur_line)
                        cur_line = w
                if cur_line:
                    motion_lines.append(cur_line)
            box_h = (30 + len(legend) * 20
                     + (len(motion_lines) * 16 + 5
                        if motion_lines else 0))
            draw.rectangle([5, 5, 5 + box_w, 5 + box_h],
                           fill=(0, 0, 0, 160))

            # Title line
            draw.text((10, 8),
                      f"Clip {args.clip}  frame {t}/{total}",
                      fill=(255, 255, 255), font=font_title)
            y_pos = 28
            for mline in motion_lines:
                draw.text((10, y_pos), mline,
                          fill=(200, 200, 200), font=font)
                y_pos += 16

            # Legend entries
            for label, color, y_text in legend:
                draw.rectangle([10, y_pos + 2, 22, y_pos + 14],
                               fill=color)
                draw.text((28, y_pos),
                          f"{label} [{y_text}]",
                          fill=(255, 255, 255), font=font)
                y_pos += 20

            return np.array(img)

        print(f"Recording {example.T} frames to {mp4_path} ...")
        for t in range(example.T):
            sim_time = t / args.fps
            example.frame = t
            example._set_frame(t)

            viewer.begin_frame(sim_time)
            viewer.log_state(example.state)
            example._draw_stick_figures(t)
            viewer.end_frame()

            frame_gpu = viewer.get_frame()
            frame_np = frame_gpu.numpy()  # (H, W, 3) uint8
            frame_np = overlay_legend(frame_np, t, example.T)
            writer.append_data(frame_np)

        writer.close()
        print(f"Saved: {mp4_path}")
    else:
        newton.examples.run(example, args)
