#!/usr/bin/env python
"""
Quick debug script: visualize the template smpl.xml skeleton in rest pose.

This helps determine whether the "lying on ground" issue is in:
1. The template XML definition itself
2. Our per-subject XML generation
3. The rest pose visualization code
"""
import os
import sys
import numpy as np
import warp as wp

import newton
import newton.examples

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TemplateRestPoseViewer:
    """View template smpl.xml in rest pose."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.device = args.device if args else "cuda:0"
        self.sim_time = 0.0
        
        # ── Build model from template XML ──────────────────────
        template_xml = os.path.join(PROJECT_ROOT, "prepare", "assets", "smpl.xml")
        
        print(f"\nLoading template XML: {template_xml}")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(template_xml, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        
        self.state = self.model.state()
        self.n_dof = self.model.joint_dof_count
        
        print(f"Model: {self.n_dof} DOF, {self.model.body_count} bodies")
        
        # ── Set rest pose (all joint angles = 0) ───────────────
        joint_q = np.zeros(self.n_dof, dtype=np.float32)
        self.state.joint_q = wp.array(joint_q, dtype=wp.float32, device=self.device)
        jqd = wp.zeros(self.n_dof, dtype=wp.float32, device=self.device)
        newton.eval_fk(self.model, self.state.joint_q, jqd, self.state)
        
        # Print root position info
        body_q = self.state.body_q.numpy()
        root_pos = body_q[0, :3]
        print(f"\nRoot (Pelvis) position: {root_pos}")
        print(f"  X: {root_pos[0]:.3f}")
        print(f"  Y: {root_pos[1]:.3f}")
        print(f"  Z (height): {root_pos[2]:.3f}")
        
        # Print a few body positions
        print(f"\nBody positions (first 5):")
        for i in range(min(5, self.model.body_count)):
            name = self.model.body_label[i].rsplit('/', 1)[-1]
            pos = body_q[i, :3]
            print(f"  [{i:2d}] {name:15s}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
        
        # Set viewer model and camera
        self.viewer.set_model(self.model)
        self._setup_camera(root_pos)
        
        print("\n=== TEMPLATE REST POSE ===")
        print("View the skeleton in rest pose (template XML).")
        print("Check if skeleton is standing upright or lying on ground.")
        print("Close viewer to exit.\n")

    def _setup_camera(self, root_pos):
        """Position camera to look at the root position."""
        # Z-up: place camera behind (-Y), at chest height (z=1m), yaw=90 looks +Y
        cam_dist = 4.0
        cam_pos = wp.vec3(
            float(root_pos[0]),              # x: centred
            float(root_pos[1]) - cam_dist,   # y: 4 m behind character
            2.0,                             # z: above character height
        )
        self.viewer.set_camera(cam_pos, pitch=-15.0, yaw=90.0)

    def step(self):
        pass  # No animation, just static view

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    if not hasattr(args, 'device') or args.device is None:
        args.device = "cuda:0"
    
    example = TemplateRestPoseViewer(viewer, args)
    newton.examples.run(example, args)
