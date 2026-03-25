"""Quick import + forward test for PhysicsRVQVAE."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Step 1: Config imports...")
from newton_vqvae.config import TrainingConfig, PhysicsLossWeights
cfg = TrainingConfig()
print(f"  OK: alpha={cfg.physics_weights.alpha}")

print("Step 2: Data adapter imports...")
from newton_vqvae.data_adapter import MotionNormalizer, LightSAGEConv
norm = MotionNormalizer()
print(f"  OK: mean shape={norm.mean_np.shape}")

print("Step 3: Physics losses...")
from newton_vqvae.physics_losses import PhysicsLoss, PhysicsLossScheduler
ploss = PhysicsLoss(cfg)
print(f"  OK")

print("Step 4: Model import...")
try:
    from newton_vqvae.model import PhysicsRVQVAE
    print("  OK: PhysicsRVQVAE imported")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("Step 5: Model creation (CPU)...")
try:
    model = PhysicsRVQVAE(cfg, device='cpu')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK: {n_params} params")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("Step 6: Forward pass (kinematic only)...")
try:
    import torch
    x = torch.randn(2, 64, 262)
    result = model(x, run_physics=False)
    print(f"  x_hat shape: {result['x_hat'].shape}")
    print(f"  commit_loss: {result['commit_loss'].item():.4f}")
    print(f"  perplexity: {result['perplexity'].item():.1f}")
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("Step 7: Loss computation...")
try:
    from models.losses import Geometric_Losses
    geo = Geometric_Losses('l1_smooth', 22, 'interhuman', 'cpu')
    total_loss, log = model.compute_losses(x, result, geo)
    print(f"  total_loss: {total_loss.item():.4f}")
    print(f"  loss_rec: {log['loss_rec']:.4f}")
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("Step 8: Backward pass...")
try:
    total_loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  grad_norm: {grad_norm:.4f}")
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n=== ALL TESTS PASSED ===")
