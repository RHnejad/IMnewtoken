"""
Visualize the per-subject SMPL XML for a given clip/person, or the default template.

Usage:
    # Per-subject skeleton (requires retargeted data)
    python prepare/assets/view_smpl.py --clip 1000 --person 0

    # Default template XML (no data needed)
    python prepare/assets/view_smpl.py --default
"""
import os
import sys
import warnings

import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton
import newton.examples
import newton.solvers

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

TEMPLATE_XML = os.path.join(PROJECT_ROOT, "prepare", "assets", "smpl.xml")
from prepare2.retarget import get_or_create_xml


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        use_default = args.default if args and hasattr(args, 'default') else False

        if use_default:
            xml_path = TEMPLATE_XML
            print(f"[DEFAULT TEMPLATE] XML: {xml_path}")
        else:
            clip_id = str(args.clip) if args else "1000"
            person = args.person if args else 0
            data_dir = args.data_dir if args else os.path.join(PROJECT_ROOT, "data/retargeted_v2/interhuman")
            # Resolve relative paths against PROJECT_ROOT (script may be run from any CWD)
            if not os.path.isabs(data_dir):
                data_dir = os.path.join(PROJECT_ROOT, data_dir)

            betas_path = os.path.join(data_dir, f"{clip_id}_person{person}_betas.npy")
            if not os.path.exists(betas_path):
                raise FileNotFoundError(
                    f"Betas not found: {betas_path}\n"
                    f"Run: python prepare2/retarget.py --dataset interhuman --clip {clip_id}\n"
                    f"Or use --default to view the template XML."
                )

            import numpy as np
            betas = np.load(betas_path)
            xml_path = get_or_create_xml(betas)
            print(f"[PER-SUBJECT] XML: {xml_path}")

        # Pelvis is the root; legs hang ~0.9 m below it in the XML.
        # Lift by 1.0 m so feet start just above z=0 (ground plane).
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(xml_path, xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)),
                         enable_self_collisions=False)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=200, nconmax=100)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)

        # Camera: Z-up, 4m behind (-Y), 1m height, looking toward +Y
        self.viewer.set_camera(wp.vec3(0.0, -4.0, 1.0), pitch=-10.0, yaw=90.0)

        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip", type=str, default="1000", help="Clip ID")
    parser.add_argument("--person", type=int, default=0, choices=[0, 1], help="Person index")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/retargeted_v2/interhuman"),
                        help="Directory containing betas .npy files")
    parser.add_argument("--default", action="store_true",
                        help="View the default template smpl.xml (no data required)")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
