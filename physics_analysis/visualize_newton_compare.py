#!/usr/bin/env python
"""
Newton GL viewer for comparing multiple motion versions (GT vs Generated).

Displays retargeted joint_q motions side-by-side in Newton's OpenGL viewer.
Each version is color-coded (green, red, blue, ...) and offset along Y.
An imgui panel shows version labels, frame counter, and text annotation.

Optionally also renders SMPL mesh from pkl files alongside skeletons.

Usage (requires display, conda activate mimickit):

    # Compare GT vs Generated
    python physics_analysis/visualize_newton_compare.py \
        --clip 4659 \
        --data-dir data/retargeted_v2/gt_from_positions \
                   data/retargeted_v2/gen_from_positions \
        --label GT Gen

    # With SMPL mesh for GT (placed next to skeleton at X offset)
    python physics_analysis/visualize_newton_compare.py \
        --clip 4659 \
        --data-dir data/retargeted_v2/gt_from_positions \
                   data/retargeted_v2/gen_from_positions \
        --label GT Gen \
        --mesh-pkl data/InterHuman/motions/4659.pkl none

    # Single version
    python physics_analysis/visualize_newton_compare.py \
        --clip 4659 \
        --data-dir data/retargeted_v2/gt_from_positions \
        --label GT

    # Adjust layout / speed
    python physics_analysis/visualize_newton_compare.py \
        --clip 4659 \
        --data-dir data/retargeted_v2/gt_from_positions \
                   data/retargeted_v2/gen_from_positions \
        --label GT Gen --spacing 5.0 --fps 30
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np

import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton
import newton.examples

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare2.retarget import get_or_create_xml

# ── Version colors (for both 3D shapes and imgui labels) ─────
VERSION_COLORS = [
    (0.2, 0.85, 0.2),  # green
    (1.0, 0.3, 0.3),   # red
    (0.3, 0.6, 1.0),   # blue
    (1.0, 0.7, 0.2),   # orange
    (0.8, 0.4, 1.0),   # purple
]
N_BODIES_PER_SKEL = 24  # Fixed for SMPL Newton skeleton


def load_annotation(clip_id):
    """Try to load motion text description."""
    path = os.path.join(PROJECT_ROOT, "data", "InterHuman", "annots", f"{clip_id}.txt")
    if os.path.exists(path):
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]
    return []


class CompareVisualizer:
    """Newton GL viewer comparing multiple motion versions side-by-side."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = args.fps
        self.sim_time = 0.0
        self._wall_start = None
        self.device = args.device
        self.spacing = args.spacing
        self._clip_id = args.clip

        data_dirs = args.data_dir
        labels = args.label or [os.path.basename(d.rstrip("/")) for d in data_dirs]
        self.version_labels = labels
        self.n_versions = len(data_dirs)

        # Load annotation
        self.motion_texts = load_annotation(args.clip)

        # ── Load all skeletons from all versions ─────────────
        self.all_joint_q = []      # one (T, 76) per skeleton
        self.all_xml_paths = []    # one xml per skeleton
        self.all_version_idx = []  # version each skeleton belongs to
        self.all_person_idx = []   # person index (0 or 1) within version
        self.persons_per_version = []

        for vi, (data_dir, label) in enumerate(zip(data_dirs, labels)):
            print(f"\n{'─'*50}")
            print(f"Version {vi}: {label}")
            print(f"  dir: {data_dir}")

            n_persons = 0
            for p_idx in [0, 1]:
                jq_path = os.path.join(
                    data_dir, f"{args.clip}_person{p_idx}_joint_q.npy"
                )
                betas_path = os.path.join(
                    data_dir, f"{args.clip}_person{p_idx}_betas.npy"
                )
                if not os.path.exists(jq_path):
                    continue

                jq = np.load(jq_path).astype(np.float32)
                betas = (
                    np.load(betas_path)
                    if os.path.exists(betas_path)
                    else np.zeros(10)
                )
                xml_path = get_or_create_xml(betas)

                print(
                    f"  person{p_idx}: {jq.shape[0]} frames, "
                    f"jq shape={jq.shape}, XML={os.path.basename(xml_path)}"
                )

                self.all_joint_q.append(jq)
                self.all_xml_paths.append(xml_path)
                self.all_version_idx.append(vi)
                self.all_person_idx.append(p_idx)
                n_persons += 1

            if n_persons == 0:
                raise FileNotFoundError(
                    f"No joint_q files for clip {args.clip} in {data_dir}"
                )
            self.persons_per_version.append(n_persons)

        self.n_skeletons = len(self.all_joint_q)

        # ── Frame count ──────────────────────────────────────
        self.T = min(jq.shape[0] for jq in self.all_joint_q)

        # ── Build Newton model (SAME as prepare2/visualize_newton.py) ─
        print(f"\nBuilding {self.n_skeletons}-skeleton Newton model...")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for xml_path in self.all_xml_paths:
            builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)

        self.state = self.model.state()
        self.jqd = wp.zeros(
            self.model.joint_dof_count, dtype=wp.float32, device=self.device
        )

        # ── joint_q size ─────────────────────────────────────
        self.n_per = self.all_joint_q[0].shape[1]
        self.combined_n = self.model.joint_coord_count
        print(
            f"Combined model: {self.combined_n} joint coords "
            f"({self.n_skeletons} × {self.n_per})"
        )
        assert (
            self.combined_n == self.n_skeletons * self.n_per
        ), f"joint coord mismatch: {self.combined_n} != {self.n_skeletons}*{self.n_per}"

        # ── Y offsets: one per version, shared by all persons within it ─
        self.y_offsets = []
        for i in range(self.n_skeletons):
            vi = self.all_version_idx[i]
            y = vi * self.spacing
            self.y_offsets.append(y)

        for i in range(self.n_skeletons):
            vi = self.all_version_idx[i]
            pi = self.all_person_idx[i]
            print(
                f"  skeleton {i} (v{vi} p{pi}): "
                f"Y offset={self.y_offsets[i]:.1f}m, "
                f"root0=({self.all_joint_q[i][0, 0]:.2f}, "
                f"{self.all_joint_q[i][0, 1]:.2f}, "
                f"{self.all_joint_q[i][0, 2]:.2f})"
            )

        # ── Set initial frame ────────────────────────────────
        self._set_frame(0)
        self.frame = 0

        # ── Viewer model + coloring ──────────────────────────
        self.viewer.set_model(self.model)
        self._apply_version_colors()

        # ── Optional mesh ────────────────────────────────────
        self.has_meshes = False
        self.mesh_entries = []
        self._mesh_verts_wp = []
        self.mesh_x_offset = args.mesh_x_offset
        if args.mesh_pkl:
            self._load_meshes(args)

        # ── Camera ───────────────────────────────────────────
        self._setup_camera()

        print(
            f"\nReady: {self.n_versions} version(s), "
            f"{self.T} frames ({self.T / self.fps:.1f}s). "
            f"Close viewer to exit.\n"
        )

    # ── Camera ────────────────────────────────────────────────

    def _setup_camera(self):
        """Point camera at the center of all skeletons."""
        # Average root position from frame 0 (before Y offset)
        roots = np.array([jq[0, :3] for jq in self.all_joint_q])
        center = roots.mean(axis=0)
        # Include Y offsets in the center calculation
        center_y = center[1] + np.mean(self.y_offsets)
        y_range = max(self.y_offsets) - min(self.y_offsets) if len(self.y_offsets) > 1 else 0.0
        cam_dist = max(5.0, y_range * 1.2)

        cam_pos = wp.vec3(
            float(center[0]),
            float(center_y) - cam_dist,
            2.0,
        )
        self.viewer.set_camera(cam_pos, pitch=-15.0, yaw=90.0)

    # ── Shape coloring ────────────────────────────────────────

    def _apply_version_colors(self):
        """Color each version's capsule shapes with a distinct color."""
        shape_body = self.model.shape_body.numpy()
        shape_colors = {}

        for s_idx, body_idx in enumerate(shape_body):
            if body_idx < 0:
                continue  # ground plane

            skel_idx = body_idx // N_BODIES_PER_SKEL
            if skel_idx >= self.n_skeletons:
                continue

            vi = self.all_version_idx[skel_idx]
            pi = self.all_person_idx[skel_idx]
            r, g, b = VERSION_COLORS[vi % len(VERSION_COLORS)]

            # Dim person 1 to distinguish from person 0
            if pi == 1:
                r, g, b = r * 0.6, g * 0.6, b * 0.6

            shape_colors[s_idx] = wp.vec3(r, g, b)

        if shape_colors:
            self.viewer.update_shape_colors(shape_colors)
            print(f"Applied version colors to {len(shape_colors)} shapes")

    # ── Mesh loading ──────────────────────────────────────────

    def _load_meshes(self, args):
        """Load SMPL meshes from pkl and register with the viewer."""
        import torch
        from data.body_model.body_model import BodyModel

        torch_device = torch.device(
            "cuda" if "cuda" in self.device else "cpu"
        )

        print(f"\n{'─'*50}")
        print("Loading SMPL-X body model for mesh rendering...")
        body_model = BodyModel(
            bm_fname=os.path.join(args.body_model_path, "SMPLX_NEUTRAL.npz"),
            num_betas=10,
            num_expressions=10,
            dtype=torch.float32,
        ).to(torch_device)

        faces_np = body_model.f.cpu().numpy().astype(np.int32).flatten()
        self._mesh_faces = wp.from_numpy(faces_np, dtype=int, device=self.device)
        self._id_xform = wp.array(
            [wp.transform_identity()], dtype=wp.transform, device=self.device
        )
        self._one_scale = wp.array(
            [wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=self.device
        )
        self._mesh_mat = wp.array(
            [wp.vec4(0.4, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device
        )

        # Pad pkl list
        pkl_paths = list(args.mesh_pkl)
        while len(pkl_paths) < self.n_versions:
            pkl_paths.append("none")

        for vi, pkl_path in enumerate(pkl_paths):
            if pkl_path.lower() == "none" or not os.path.exists(pkl_path):
                continue

            with open(pkl_path, "rb") as f:
                motion_data = pickle.load(f)

            for pidx, key in enumerate(["person1", "person2"]):
                if key not in motion_data:
                    continue
                d = motion_data[key]
                T_mesh = d["trans"].shape[0]

                trans = torch.from_numpy(d["trans"]).float().to(torch_device)
                root_orient = torch.from_numpy(d["root_orient"]).float().to(
                    torch_device
                )
                pose_body = torch.from_numpy(d["pose_body"]).float().to(
                    torch_device
                )
                betas_t = (
                    torch.from_numpy(d["betas"]).float().unsqueeze(0).to(torch_device)
                )

                # FK in batches
                all_verts = []
                for start in range(0, T_mesh, 64):
                    end = min(start + 64, T_mesh)
                    bs = end - start
                    with torch.no_grad():
                        out = body_model(
                            root_orient=root_orient[start:end],
                            pose_body=pose_body[start:end],
                            betas=betas_t.expand(bs, -1),
                            trans=trans[start:end],
                        )
                    all_verts.append(out.v.cpu().numpy())
                verts = np.concatenate(all_verts, axis=0)  # (T_mesh, V, 3)

                # InterHuman pkl is Z-up; apply version Y offset + mesh X offset
                y_off = vi * self.spacing
                verts[:, :, 1] += y_off
                verts[:, :, 0] += self.mesh_x_offset

                verts_wp = wp.from_numpy(
                    verts.astype(np.float32), dtype=wp.vec3, device=self.device
                )

                r, g, b = VERSION_COLORS[vi % len(VERSION_COLORS)]
                if pidx == 1:
                    r, g, b = r * 0.6, g * 0.6, b * 0.6
                color_wp = wp.array(
                    [wp.vec3(r, g, b)], dtype=wp.vec3, device=self.device
                )

                mesh_name = f"/meshes/v{vi}_p{pidx}"
                inst_name = f"/mesh_inst/v{vi}_p{pidx}"
                self.viewer.log_mesh(
                    mesh_name, verts_wp[0], self._mesh_faces, hidden=True
                )
                self.viewer.log_instances(
                    inst_name,
                    mesh_name,
                    self._id_xform,
                    self._one_scale,
                    color_wp,
                    self._mesh_mat,
                )

                self._mesh_verts_wp.append(verts_wp)
                self.mesh_entries.append((vi, pidx, T_mesh))
                print(
                    f"  Mesh v{vi}/p{pidx}: {T_mesh} frames, "
                    f"{verts.shape[1]} verts"
                )

        self.has_meshes = len(self.mesh_entries) > 0
        if self.has_meshes:
            print(f"Registered {len(self.mesh_entries)} mesh(es)")

    # ── Frame update (same logic as prepare2/visualize_newton.py) ──

    def _set_frame(self, t):
        """Set all skeletons to frame t, with per-version Y offsets."""
        combined_q = np.zeros(self.combined_n, dtype=np.float32)

        for i, jq in enumerate(self.all_joint_q):
            frame = min(t, jq.shape[0] - 1)
            base = i * self.n_per
            combined_q[base : base + self.n_per] = jq[frame]
            # Apply version Y offset to root Y (joint_q index 1)
            combined_q[base + 1] += self.y_offsets[i]

        self.state.joint_q = wp.array(
            combined_q, dtype=wp.float32, device=self.device
        )
        self.jqd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

    # ── Newton callbacks ──────────────────────────────────────

    def step(self):
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = now
        self.sim_time = now - self._wall_start
        self.frame = int(self.sim_time * self.fps) % self.T
        self._set_frame(self.frame)

        # Update meshes
        if self.has_meshes:
            for mi, (vi, pidx, T_mesh) in enumerate(self.mesh_entries):
                f = min(self.frame, T_mesh - 1)
                self.viewer.log_mesh(
                    f"/meshes/v{vi}_p{pidx}",
                    self._mesh_verts_wp[mi][f],
                    self._mesh_faces,
                    hidden=True,
                )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def gui(self, imgui):
        """Side panel with version info, frame counter, annotation."""
        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(0.4, 1.0, 0.8, 1.0), "[ MOTION COMPARISON ]"
        )
        imgui.separator()

        imgui.text(f"Clip:     {self._clip_id}")
        imgui.text(f"Versions: {self.n_versions}")
        imgui.text(f"FPS:      {self.fps}")
        imgui.text(f"Frames:   {self.T}")

        pct = int(100 * self.frame / max(self.T - 1, 1))
        imgui.text(f"Frame:    {self.frame} / {self.T - 1}  ({pct}%)")

        imgui.separator()

        skel_idx = 0
        for vi in range(self.n_versions):
            r, g, b = VERSION_COLORS[vi % len(VERSION_COLORS)]
            color = imgui.ImVec4(r, g, b, 1.0)
            label = self.version_labels[vi]
            n_p = self.persons_per_version[vi]

            imgui.text_colored(color, f"■ [{vi}] {label}")
            imgui.text(f"    persons: {n_p}, Y offset: {vi * self.spacing:.1f}m")

            for _ in range(n_p):
                jq = self.all_joint_q[skel_idx]
                pi = self.all_person_idx[skel_idx]
                tint = "bright" if pi == 0 else "dim"
                ended = " [ended]" if self.frame >= jq.shape[0] else ""
                imgui.text(f"    p{pi} ({tint}): {jq.shape[0]} frames{ended}")
                skel_idx += 1

        if self.has_meshes:
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(0.9, 0.9, 0.5, 1.0), "SMPL Mesh:"
            )
            imgui.text(f"  X offset: +{self.mesh_x_offset:.1f}m")
            for mi, (vi, pidx, T_mesh) in enumerate(self.mesh_entries):
                r, g, b = VERSION_COLORS[vi % len(VERSION_COLORS)]
                color = imgui.ImVec4(r, g, b, 1.0)
                ended = " [ended]" if self.frame >= T_mesh else ""
                imgui.text_colored(
                    color, f"  ■ v{vi}/p{pidx}: {T_mesh} frames{ended}"
                )

        if self.motion_texts:
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(1.0, 1.0, 0.5, 1.0), "Annotation:"
            )
            imgui.spacing()
            for i, text in enumerate(self.motion_texts[:3], 1):
                imgui.text_colored(
                    imgui.ImVec4(0.7, 0.7, 0.7, 1.0), f"[{i}]"
                )
                imgui.same_line()
                imgui.text_wrapped(text)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip", type=str, required=True, help="Clip ID")
    parser.add_argument(
        "--data-dir",
        type=str,
        nargs="+",
        required=True,
        help="Directories with {clip}_person{0,1}_joint_q.npy / _betas.npy",
    )
    parser.add_argument(
        "--label",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each --data-dir",
    )
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS")
    parser.add_argument(
        "--spacing",
        type=float,
        default=3.0,
        help="Y-axis spacing between versions (m)",
    )
    parser.add_argument(
        "--mesh-pkl",
        type=str,
        nargs="+",
        default=None,
        help="PKL path per version for SMPL mesh ('none' to skip)",
    )
    parser.add_argument(
        "--body-model-path",
        type=str,
        default="data/body_model/smplx",
        help="SMPL-X body model directory",
    )
    parser.add_argument(
        "--mesh-x-offset",
        type=float,
        default=3.0,
        help="X offset for mesh copies (m)",
    )

    viewer, args = newton.examples.init(parser)
    if not hasattr(args, "device") or args.device is None:
        args.device = "cuda:0"

    example = CompareVisualizer(viewer, args)
    newton.examples.run(example, args)
