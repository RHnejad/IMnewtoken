
import sys
import os
import torch
import numpy as np
import pickle
import argparse
import smplx
from scipy.spatial.transform import Rotation as R

# Add current directory to path to allow imports from InterMask
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from data.rotation_conversions import rotation_6d_to_matrix
except ImportError:
    sys.path.append(os.path.join(current_dir, 'InterMask'))
    from data.rotation_conversions import rotation_6d_to_matrix

def convert(npy_path, output_prefix, smpl_model_path):
    print(f"Loading {npy_path}...")
    data = np.load(npy_path)
    T = data.shape[0]
    data = data.reshape(T, 2, 56, 6)

    # Initialize SMPL-X model
    body_model = smplx.create(
        model_path=smpl_model_path, 
        model_type='smplx',
        gender='NEUTRAL', 
        batch_size=1
    )
    
    # Get topology (first 24 joints)
    parents = body_model.parents.detach().cpu().numpy()[:24]
    
    # Get rest pose
    rest_output = body_model()
    rest_joints = rest_output.joints.detach().cpu().numpy().squeeze()[:24, :]
    
    # Calculate Offsets
    offsets = np.zeros((24, 3))
    offsets[0] = rest_joints[0] # Root rest pos
    for i in range(1, 24):
        p = parents[i]
        offsets[i] = rest_joints[i] - rest_joints[p]
        
    names = [
        "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg", 
        "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToe", "RightToe", 
        "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm", 
        "LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "LeftThumb", "RightThumb"
    ]

    # Process each person
    for p in range(2):
        print(f"Processing Person {p}...")
        
        # Extract Rotations: 0-24
        rot_6d = torch.tensor(data[:, p, :24, :]) # (T, 24, 6)
        rot_mat = rotation_6d_to_matrix(rot_6d) # (T, 24, 3, 3)
        
        # Convert to Quaternions [x, y, z, w]
        rots_np = rot_mat.numpy().reshape(-1, 3, 3)
        quats = R.from_matrix(rots_np).as_quat().reshape(T, 24, 4) 

        # Extract Translation: 55
        trans = data[:, p, 55, :3] # (T, 3)
        
        output_dict = {
            'rotations': quats,
            'root_pos': trans,
            'offsets': offsets,
            'parents': parents,
            'names': names
        }
        
        out_file = f"{output_prefix}_{p}.pkl"
        with open(out_file, 'wb') as f:
            pickle.dump(output_dict, f)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--smpl_path", type=str, required=True)
    args = parser.parse_args()
    convert(args.npy, args.out, args.smpl_path)
