#!/usr/bin/env python
"""
Debug script: Check skeleton positions in rest pose without viewer.

Tests both template and per-subject XMLs to identify whether the
"skeleton lying on ground" issue is in template or per-subject generation.
"""
import os
import sys
import numpy as np
import warp as wp

import newton

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare2.gen_smpl_xml import generate_smpl_xml
from prepare2.retarget import get_or_create_xml


def test_skeleton(xml_path, name):
    """Load XML and check rest pose positions."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"XML: {xml_path}")
    print(f"{'='*60}")
    
    # Build model
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    
    # Create state and set rest pose (all zeros)
    state = model.state()
    n_dof = model.joint_dof_count
    
    joint_q = np.zeros(n_dof, dtype=np.float32)
    state.joint_q = wp.array(joint_q, dtype=wp.float32, device="cuda:0")
    jqd = wp.zeros(n_dof, dtype=wp.float32, device="cuda:0")
    
    # Evaluate FK
    newton.eval_fk(model, state.joint_q, jqd, state)
    
    # Get positions
    body_q = state.body_q.numpy()
    
    print(f"\nModel: {n_dof} DOF, {model.body_count} bodies")
    print(f"\nRoot (Pelvis) position:")
    root_pos = body_q[0, :3]
    print(f"  X (horizontal):           {root_pos[0]:7.3f}")
    print(f"  Y (horizontal):           {root_pos[1]:7.3f}")
    print(f"  Z (height along gravity): {root_pos[2]:7.3f}")
    
    # Find min/max Z positions
    all_z = body_q[:, 2]
    print(f"\nZ coordinate range:")
    print(f"  Min: {all_z.min():7.3f} (feet on ground)")
    print(f"  Max: {all_z.max():7.3f} (top of head)")
    print(f"  Range: {all_z.max() - all_z.min():.3f}")
    
    # Print body positions for key joints
    key_bodies = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "Torso", "Head"]
    print(f"\nKey body positions:")
    for key_body in key_bodies:
        try:
            _short_names = [lbl.rsplit('/', 1)[-1] for lbl in model.body_label]
            idx = _short_names.index(key_body)
            pos = body_q[idx, :3]
            print(f"  {key_body:15s}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
        except (ValueError, IndexError):
            pass
    
    # Check if character is roughly standing or lying
    _short_names = [lbl.rsplit('/', 1)[-1] for lbl in model.body_label]
    head_idx = _short_names.index("Head")
    pelvis_idx = _short_names.index("Pelvis")
    head_pos = body_q[head_idx, :3]
    pelvis_pos = body_q[pelvis_idx, :3]
    
    height_diff = head_pos[2] - pelvis_pos[2]
    if height_diff > 1.0:
        print(f"\n✓ Skeleton appears STANDING (head {height_diff:.2f}m above pelvis)")
    else:
        print(f"\n✗ Skeleton appears LYING DOWN (height diff = {height_diff:.2f}m)")


if __name__ == "__main__":
    # Test template XML
    template_xml = os.path.join(PROJECT_ROOT, "prepare", "assets", "smpl.xml")
    test_skeleton(template_xml, "TEMPLATE (default SMPL)")
    
    # Test per-subject XML with zero betas
    print("\n\nGenerating per-subject XML with betas=0...")
    xml_str = generate_smpl_xml(np.zeros(10))
    
    # Write to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_str)
        temp_xml = f.name
    
    test_skeleton(temp_xml, "PER-SUBJECT (betas=0)")
    
    # Cleanup
    os.unlink(temp_xml)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
