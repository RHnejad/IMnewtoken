import os
import sys
import torch
import numpy as np
import trimesh
import smplx
from torch.utils.data import DataLoader

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from newton_vqvae.data_adapter import InterHumanPairDataset

# Average human body density = ~1000 kg/m³
HUMAN_DENSITY_KG_M3 = 1000.0

def estimate_mass_from_betas(smpl_model: smplx.SMPLLayer, betas: torch.Tensor) -> float:
    """
    Given betas (shape params), generate the rest-pose SMPL mesh
    and compute its mass assuming uniform density of 1000 kg/m³.
    """
    with torch.no_grad():
        # Get rest-pose mesh (all joint angles zero)
        output = smpl_model(betas=betas)
        vertices = output.vertices[0].cpu().numpy() # (6890, 3)
        faces = smpl_model.faces # (13776, 3)

    # Create trimesh object to easily compute volume of the watertight mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Volume is in cubic meters (since vertices are in meters)
    volume_m3 = mesh.volume
    
    # Mass = Density * Volume
    mass_kg = volume_m3 * HUMAN_DENSITY_KG_M3
    return float(mass_kg)

def main():
    print("Loading InterHuman dataset to get target betas...")
    dataset = InterHumanPairDataset(
        data_root=os.path.join(_PROJECT_ROOT, "data/InterHuman"),
        mode='train',
        window_size=64,
        window_stride=64,
    )
    
    # Initialize SMPL-X model
    model_path = os.path.join(_PROJECT_ROOT, "data", "body_model")
    
    try:
        smpl_model = smplx.create(model_path=model_path, model_type='smplx', gender='neutral', batch_size=1)
        smpl_model.eval()
    except Exception as e:
        print(f"Error loading SMPLX model: {e}")
        return

    # Indices to analyze
    target_indices = [1000, 2000]
    
    print("\n--- Estimated Masses (kg) based on SMPL mesh volume ---")
    print("Assuming average human density = 1000 kg/m³\n")
    print(f"{'Sequence':<10} | {'Person A (kg)':<15} | {'Person B (kg)':<15}")
    print("-" * 45)
    
    for idx in target_indices:
        if idx < len(dataset):
            # Get item
            item = dataset[idx]
            
            # Betas are size (10,)
            betas_A = item['betas_p1'].unsqueeze(0).float() # (1, 10)
            betas_B = item['betas_p2'].unsqueeze(0).float() # (1, 10)
            
            try:
                # Estimate mass
                mass_A = estimate_mass_from_betas(smpl_model, betas_A)
                mass_B = estimate_mass_from_betas(smpl_model, betas_B)
                
                print(f"{idx:<10} | {mass_A:<15.1f} | {mass_B:<15.1f}")
            except Exception as e:
                print(f"Error computing mass for sequence {idx}: {e}")
        else:
            print(f"Index {idx} is out of bounds.")
            
    # As a baseline, compute the mass of a "zero" beta (average) person
    zero_betas = torch.zeros(1, 10).float()
    neutral_mass = estimate_mass_from_betas(smpl_model, zero_betas)
    print("-" * 45)
    print(f"{'Neutral Base':<10} | {'Both people: ' + str(round(neutral_mass, 1)) + ' kg'}")

if __name__ == "__main__":
    main()
