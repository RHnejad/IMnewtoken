"""
Visualize InterHuman or InterX clip with full SMPL-X mesh in Newton viewer.

Directly loads a clip by ID from the dataset and renders in Newton's GL viewer.

Usage (run from InterMask directory WITH DISPLAY):
    cd /media/rh/codes/sim/InterMask
    conda activate mimickit

    # InterHuman (default)
    python prepare/visualize_mesh_newton.py --clip 1000
    python prepare/visualize_mesh_newton.py --clip 1000 --fps 30

    # InterX
    python prepare/visualize_mesh_newton.py --dataset interx --clip P01_R01_FN01_ID01
    python prepare/visualize_mesh_newton.py --dataset interx --clip P01_R01_FN01_ID01 --person 0
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
import h5py
import warp as wp
import newton
import newton.examples

# Add InterMask to path for body model import
INTERMASK_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if INTERMASK_PATH not in sys.path:
    sys.path.insert(0, INTERMASK_PATH)

from data.body_model.body_model import BodyModel


class InterHumanMeshVis:
    """Visualize InterHuman or InterX clip with full SMPL-X mesh in Newton."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device(args.device if args.device else "cuda:0")
        self.torch_device = torch.device("cuda" if "cuda" in str(self.device) else "cpu")

        body_model_path = args.body_model_path
        self.fps = args.fps
        self._dataset = args.dataset
        self._wall_start = None

        # ── Initialize body model ────────────────────────────
        print("Loading SMPL-X body model...")
        self.body_model = BodyModel(
            bm_fname=os.path.join(body_model_path, "SMPLX_NEUTRAL.npz"),
            num_betas=10,
            num_expressions=10,
            dtype=torch.float32,
        ).to(self.torch_device)

        faces_np = self.body_model.f.cpu().numpy().astype(np.int32).flatten()

        # ── Load motion data based on dataset ────────────────
        if args.dataset == "interx":
            person_params = self._load_interx(args)
        else:
            person_params = self._load_interhuman(args)

        # ── Person colors ─────────────────────────────────────
        self.person_colors = [
            wp.vec3(1.0, 0.2, 0.2),  # Person 1: Red
            wp.vec3(0.2, 0.6, 1.0),  # Person 2: Blue
        ]

        # ── Generate meshes for all persons ──────────────────
        # person_params: list of (label, pidx, dict with trans/root_orient/pose_body/betas/pose_hand)
        self.persons_vertices = []
        all_min_z = float("inf")

        for label, pidx, p in person_params:
            T = p["trans"].shape[0]
            print(f"\n{label}: {T} frames")

            trans       = p["trans"].to(self.torch_device)
            root_orient = p["root_orient"].to(self.torch_device)
            pose_body   = p["pose_body"].to(self.torch_device)
            betas       = p["betas"].to(self.torch_device)
            pose_hand   = p["pose_hand"].to(self.torch_device) if p["pose_hand"] is not None else None

            all_verts = []
            batch_size = 64
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                bs  = end - start
                with torch.no_grad():
                    out = self.body_model(
                        root_orient=root_orient[start:end],
                        pose_body=pose_body[start:end],
                        pose_hand=pose_hand[start:end] if pose_hand is not None else None,
                        betas=betas.expand(bs, -1),
                        trans=trans[start:end],
                    )
                all_verts.append(out.v.cpu().numpy())

            verts = np.concatenate(all_verts, axis=0)  # (T, 10475, 3)

            # InterX is in SMPL-X Y-up space; Newton viewer is Z-up.
            # Rotate +90° around X: (x, y, z) → (x, -z, y)
            if self._dataset == "interx":
                verts = np.stack([verts[:, :, 0], -verts[:, :, 2], verts[:, :, 1]], axis=2)

            all_min_z = min(all_min_z, verts[:, :, 2].min())
            self.persons_vertices.append(verts)
            print(f"  Generated {verts.shape[0]} meshes, {verts.shape[1]} vertices each")

        # ── Ground offset ─────────────────────────────────────
        self.ground_offset = -all_min_z + 0.005
        print(f"\nGround offset: {self.ground_offset:.4f} m")
        for i in range(len(self.persons_vertices)):
            self.persons_vertices[i][:, :, 2] += self.ground_offset

        self.num_frames = min(v.shape[0] for v in self.persons_vertices)
        self.n_persons = len(self.persons_vertices)

        # ── Upload to Warp ────────────────────────────────────
        self.persons_verts_wp = []
        for verts in self.persons_vertices:
            wp_v = wp.from_numpy(verts.astype(np.float32), dtype=wp.vec3, device=self.device)
            self.persons_verts_wp.append(wp_v)

        self.faces_wp = wp.from_numpy(faces_np, dtype=int, device=self.device)

        # ── Setup Newton model (ground plane) ─────────────────
        builder = newton.ModelBuilder()
        builder.gravity = 0.0
        ground_body = builder.add_body()
        builder.add_shape_plane(
            body=ground_body,
            xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
            width=10.0,
            length=10.0,
        )
        self.model = builder.finalize()
        self.state = self.model.state()
        self.viewer.set_model(self.model)

        # ── Register meshes with viewer ───────────────────────
        self.identity_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self.ones_scale = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=self.device)
        self.mat_default = wp.array([wp.vec4(0.5, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device)

        for i in range(self.n_persons):
            color = wp.array([self.person_colors[i]], dtype=wp.vec3, device=self.device)
            self.viewer.log_mesh(f"/meshes/person_{i}", self.persons_verts_wp[i][0], self.faces_wp, hidden=True)
            self.viewer.log_instances(f"/person_{i}", f"/meshes/person_{i}",
                                     self.identity_xform, self.ones_scale, color, self.mat_default)

        self.current_frame = 0
        self.time = 0.0
        self.frame_dt = 1.0 / self.fps

        print(f"\nReady! {self.n_persons} person(s), {self.num_frames} frames @ {self.fps} fps")
        print(f"Duration: {self.num_frames / self.fps:.1f} seconds")
        print("Close the viewer window to exit.")

    # ── Dataset loaders ───────────────────────────────────────

    def _load_interhuman(self, args):
        """Load from InterHuman motions/{clip_id}.pkl.

        Returns list of (label, pidx, params_dict).
        """
        clip_id   = args.clip
        data_root = args.data_root

        pkl_path = os.path.join(data_root, "motions", f"{clip_id}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Motion file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            motion_data = pickle.load(f)

        annot_path = os.path.join(data_root, "annots", f"{clip_id}.txt")
        if os.path.exists(annot_path):
            with open(annot_path) as f:
                print(f"Annotation: {f.read().strip()}")

        # Determine which persons to include
        raw_keys = []
        if args.person is not None:
            key = f"person{args.person + 1}"
            if key in motion_data:
                raw_keys.append((key, args.person))
        else:
            for key, idx in [("person1", 0), ("person2", 1)]:
                if key in motion_data:
                    raw_keys.append((key, idx))

        person_params = []
        for key, pidx in raw_keys:
            d = motion_data[key]
            params = {
                "trans":       torch.from_numpy(d["trans"]).float(),
                "root_orient": torch.from_numpy(d["root_orient"]).float(),
                "pose_body":   torch.from_numpy(d["pose_body"]).float(),
                "betas":       torch.from_numpy(d["betas"]).float().unsqueeze(0),
                "pose_hand":   None,
            }
            person_params.append((key, pidx, params))

        return person_params

    def _load_interx(self, args):
        """Load from InterX processed/motions H5 file.

        InterX H5 shape: (T, 56, 6)  — axis-angle params + translation
          axis 2: [:3] = person1, [3:] = person2
          joints 0:     root_orient  (3D axis-angle)
          joints 1-21:  pose_body    (21 × 3 = 63D)
          joints 22-24: jaw + eyes   (skipped, defaulted to 0)
          joints 25-54: pose_hand    (15+15 × 3 = 90D)
          joint  55:    translation  (3D)

        Returns list of (label, pidx, params_dict).
        """
        clip_id   = args.clip
        data_root = args.data_root

        h5_path = args.h5_file if args.h5_file else os.path.join(data_root, "processed", "motions", "inter-x.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"InterX H5 file not found: {h5_path}")

        with h5py.File(h5_path, "r") as hf:
            if clip_id not in hf:
                available = list(hf.keys())[:5]
                raise KeyError(
                    f"Clip '{clip_id}' not found in H5.\n"
                    f"  InterX IDs look like: G001T000A000R000\n"
                    f"  Check splits/ for valid IDs. First 5: {available}"
                )
            motion = hf[clip_id][:].astype(np.float32)  # (T, 56, 6)

        annot_path = os.path.join(data_root, "texts", f"{clip_id}.txt")
        if os.path.exists(annot_path):
            with open(annot_path) as f:
                print(f"Annotation: {f.readline().strip()}")

        # Split persons
        persons_raw = [motion[:, :, :3], motion[:, :, 3:]]  # each (T, 56, 3)

        # Determine which persons to include
        person_indices = [args.person] if args.person is not None else [0, 1]

        person_params = []
        for pidx in person_indices:
            p = persons_raw[pidx]           # (T, 56, 3)
            T = p.shape[0]

            root_orient = torch.from_numpy(p[:, 0, :]).float()                 # (T, 3)
            pose_body   = torch.from_numpy(p[:, 1:22, :].reshape(T, 63)).float()  # (T, 63)
            pose_hand   = torch.from_numpy(p[:, 25:55, :].reshape(T, 90)).float() # (T, 90)
            trans       = torch.from_numpy(p[:, 55, :]).float()                 # (T, 3)
            betas       = torch.zeros(1, 10)                                    # neutral shape

            params = {
                "trans":       trans,
                "root_orient": root_orient,
                "pose_body":   pose_body,
                "betas":       betas,
                "pose_hand":   pose_hand,
            }
            person_params.append((f"person{pidx + 1}", pidx, params))

        return person_params

    # ── Newton callbacks ──────────────────────────────────────

    def step(self):
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = now
        self.time = now - self._wall_start
        frame = int(self.time * self.fps) % self.num_frames

        if frame != self.current_frame:
            self.current_frame = frame
            for i in range(self.n_persons):
                self.viewer.log_mesh(
                    f"/meshes/person_{i}",
                    self.persons_verts_wp[i][frame],
                    self.faces_wp,
                    hidden=True,
                )

    def render(self):
        self.viewer.begin_frame(self.time)

        xform = wp.array([wp.transform_identity()], dtype=wp.transform)
        self.viewer.log_shapes(
            "/ground",
            newton.GeoType.PLANE,
            (10.0, 10.0),
            xform,
            wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3),
            self.mat_default,
        )

        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--dataset", type=str, default="interhuman", choices=["interhuman", "interx"],
                        help="Dataset to visualize")
    parser.add_argument("--clip", type=str, default="1000",
                        help="Clip ID (e.g. '1000' for InterHuman, 'P01_R01_FN01_ID01' for InterX)")
    parser.add_argument("--person", type=int, default=None, choices=[0, 1],
                        help="Person index (0 or 1). Omit for both.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset root (defaults to data/InterHuman or data/Inter-X_Dataset)")
    parser.add_argument("--h5-file", type=str, default=None,
                        help="InterX: explicit path to H5 file (default: <data-root>/processed/motions/inter-x.h5)")
    parser.add_argument("--body-model-path", type=str, default="data/body_model/smplx",
                        help="Path to SMPL-X body model files")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Playback framerate")

    viewer, args = newton.examples.init(parser)
    if not hasattr(args, "device") or args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Apply dataset-specific data_root default
    if args.data_root is None:
        args.data_root = "data/Inter-X_Dataset" if args.dataset == "interx" else "data/InterHuman"

    example = InterHumanMeshVis(viewer, args)
    newton.examples.run(example, args)
