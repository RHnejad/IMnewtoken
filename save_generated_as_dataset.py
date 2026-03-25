#!/usr/bin/env python3
"""
save_generated_as_dataset.py

Run InterMask VQ-VAE (or transformer) generation and save the output in the
EXACT same format as the raw InterHuman dataset (motions_processed npy files).

This allows the generated data to be loaded by InterHumanDataset through the
IDENTICAL pipeline used for ground truth data:
    load_motion() → process_motion_np() → normalize → network

Output directory structure (mirrors InterHuman dataset layout):
    {output_dir}/
    ├── motions_processed/
    │   ├── person1/
    │   │   └── {clip}.npy   # (T, 492) raw format
    │   └── person2/
    │       └── {clip}.npy
    ├── annots/
    │   └── {clip}.txt
    ├── split/
    │   └── test.txt
    └── processed_262/       # if --save_processed
        └── {clip}_person{1,2}.npy

Usage:
    # Quick test on 2 sequences with visualization:
    python save_generated_as_dataset.py --dataset_name interhuman \\
        --name vq_default --use_trans False --test_n 2

    # Full dataset (VQ-VAE reconstruction):
    python save_generated_as_dataset.py --dataset_name interhuman \\
        --name vq_default --use_trans False

    # Full dataset (transformer generation):
    python save_generated_as_dataset.py --dataset_name interhuman \\
        --name trans_default

    # With 262-dim processed data for direct MPJPE comparison:
    python save_generated_as_dataset.py --dataset_name interhuman \\
        --name vq_default --use_trans False --save_processed
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
from os.path import join as pjoin
from tqdm import tqdm
import distutils.util

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vq.model import RVQVAE
from models.mask_transformer.transformer import MaskTransformer
from utils.get_opt import get_opt
from utils.utils import fixseed
from data.utils import MotionNormalizer, trans_matrix

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Column count in raw InterHuman npy (matches GT: 492 columns)
# Positions 0:66 + extra joints 66:186 + rot6d 186:312 + extra 312:492
N_RAW_COLS = 492

# Inverse of trans_matrix from data/utils.py:
#   trans_matrix: (x,y,z) → (x, z, -y)  [Y-up to processed]
#   INV_TRANS:    (x,y,z) → (x, -z, y)  [processed to Y-up]
INV_TRANS_MATRIX = np.linalg.inv(trans_matrix.numpy()).astype(np.float64)


# ═══════════════════════════════════════════════════════════════
# Core conversion: 262-dim → raw InterHuman format
# ═══════════════════════════════════════════════════════════════

def decoded_262_to_raw_interhuman(motion_262):
    """
    Convert one person's denormalized 262-dim processed motion back to the
    raw InterHuman motions_processed npy format.

    The 262-dim layout (T frames, output of process_motion_np):
        [0:66]    positions (22×3, in processed coordinate frame)
        [66:132]  velocities (22×3, frame-to-frame differences)
        [132:258] rot6d (21×6, body-local — unchanged by process_motion_np)
        [258:262] foot contacts (4)

    The raw format (T+1 frames, at least 312 columns):
        [0:66]    positions (22×3, in Y-up world frame)
        [66:186]  extra joint positions (zeros for 22-joint generated data)
        [186:312] rot6d (21×6, body-local)

    Undoing process_motion_np:
        process_motion_np applies these transforms to positions:
        1. trans_matrix:   Y-up → processed coords
        2. Floor subtract:  shift so min-y = 0
        3. XZ centering:   shift pelvis(t=0) to XZ origin
        4. Face Z+ rotation: rotate so initial forward = Z+
        5. Drop last frame, compute velocities

        For generated data (already centered, floor-level, facing Z+):
        - Steps 2/3/4 are approximately identity
        - Only trans_matrix needs explicit inversion
        - Round-trip through save → load_motion → process_motion_np is stable

    Args:
        motion_262: (T, 262) float array, denormalized VQ-VAE output

    Returns:
        raw: (T+1, N_RAW_COLS) float32 array in raw InterHuman npy format
    """
    T = motion_262.shape[0]

    # ── Extract components from 262-dim ──
    positions = motion_262[:, :66].reshape(T, 22, 3).astype(np.float64)
    velocities = motion_262[:, 66:132].reshape(T, 22, 3).astype(np.float64)
    rot6d = motion_262[:, 132:258].reshape(T, 21, 6)

    # ── Recover the dropped last frame ──
    # process_motion_np outputs T-1 frames from T input frames:
    #   data_positions[i] = positions[i]     for i = 0..T-2
    #   data_velocities[i] = positions[i+1] - positions[i]  for i = 0..T-2
    # Reconstruct: positions[T-1] = data_positions[T-2] + data_velocities[T-2]
    last_pos = positions[-1:] + velocities[-1:]  # (1, 22, 3)
    full_positions = np.concatenate([positions, last_pos], axis=0)  # (T+1, 22, 3)

    # ── Undo trans_matrix: processed frame → Y-up world frame ──
    full_positions = np.einsum("mn,...n->...m", INV_TRANS_MATRIX, full_positions)

    # ── Pad rot6d to T+1 frames (repeat last frame) ──
    full_rot6d = np.concatenate([rot6d, rot6d[-1:]], axis=0)  # (T+1, 21, 6)

    # ── Construct raw npy ──
    T_full = T + 1
    raw = np.zeros((T_full, N_RAW_COLS), dtype=np.float32)
    raw[:, :66] = full_positions.reshape(T_full, -1).astype(np.float32)
    raw[:, 186:312] = full_rot6d.reshape(T_full, -1).astype(np.float32)

    return raw


# ═══════════════════════════════════════════════════════════════
# Model loading (shared with generate_and_save.py)
# ═══════════════════════════════════════════════════════════════

def load_vq_model(vq_opt, which_epoch, device):
    dim_pose = 12 if vq_opt.dataset_name == "interhuman" else 6
    vq_model = RVQVAE(
        vq_opt, dim_pose, vq_opt.nb_code, vq_opt.code_dim, vq_opt.code_dim,
        vq_opt.down_t, vq_opt.stride_t, vq_opt.width, vq_opt.depth,
        vq_opt.dilation_growth_rate, vq_opt.vq_act, vq_opt.vq_norm,
    )
    ckpt = torch.load(
        pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch),
        map_location='cpu',
    )
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    missing, unexpected = vq_model.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    vq_epoch = ckpt.get('ep', -1)
    print(f'Loaded VQ-VAE {vq_opt.name} (epoch {vq_epoch})')
    return vq_model


def load_trans_model(model_opt, which_model, device):
    clip_version = 'ViT-L/14@336px'
    t2m_transformer = MaskTransformer(
        code_dim=model_opt.code_dim, cond_mode='text',
        latent_dim=model_opt.latent_dim, ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers, num_heads=model_opt.n_heads,
        dropout=model_opt.dropout, clip_dim=768,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version=clip_version, opt=model_opt,
    )
    ckpt = torch.load(
        pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
        map_location=device,
    )
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing, unexpected = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    print(f'Loaded Transformer {model_opt.name} (epoch {ckpt["ep"]})')
    return t2m_transformer


# ═══════════════════════════════════════════════════════════════
# Main generation + save loop
# ═══════════════════════════════════════════════════════════════

def generate_and_save_dataset(opt, net, trans, device, output_dir,
                               time_steps, cond_scale, topkr,
                               save_processed=False, test_n=0):
    """
    Generate InterHuman motions and save in raw dataset format.

    The output directory mirrors the InterHuman dataset layout and can be
    used directly as data_root for InterHumanDataset.

    Args:
        opt: Model options
        net: VQ-VAE model
        trans: Transformer model (None for VQ reconstruction)
        device: torch device
        output_dir: Where to save the dataset
        time_steps, cond_scale, topkr: Transformer sampling params
        save_processed: If True, also save 262-dim per person for comparison
        test_n: If > 0, only process first N clips and visualize
    """
    from data.interhuman import InterHumanDataset
    from torch.utils.data import DataLoader

    normalizer = MotionNormalizer()

    # ── Load test dataset ──
    data_cfg = opt
    data_cfg.mode = "test"
    data_cfg.data_root = 'data/InterHuman'
    data_cfg.joints_num = 22
    data_cfg.cache = True
    dataset = InterHumanDataset(data_cfg, normalize=True)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    net = net.to(device).eval()
    if trans is not None:
        trans = trans.to(device).eval()

    # ── Create output directory structure ──
    motions_p1 = pjoin(output_dir, "motions_processed", "person1")
    motions_p2 = pjoin(output_dir, "motions_processed", "person2")
    annots_dir = pjoin(output_dir, "annots")
    split_dir = pjoin(output_dir, "split")
    os.makedirs(motions_p1, exist_ok=True)
    os.makedirs(motions_p2, exist_ok=True)
    os.makedirs(annots_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    if save_processed:
        proc_dir = pjoin(output_dir, "processed_262")
        os.makedirs(proc_dir, exist_ok=True)

    if test_n > 0:
        vis_dir = pjoin(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    saved = 0
    clip_names = []

    mode_str = f"(test_n={test_n})" if test_n > 0 else "(full)"
    print(f"\nGenerating and saving InterHuman dataset format {mode_str} → {output_dir}")
    print(f"  Raw npy (person1/person2): {motions_p1}")
    print(f"  Annotations: {annots_dir}")
    if save_processed:
        print(f"  Processed 262-dim: {proc_dir}")
    if test_n > 0:
        print(f"  Visualizations: {vis_dir}")

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Generate")):
            if test_n > 0 and i >= test_n:
                break

            name, text, motion1_gt, motion2_gt, motion_lens = data
            T_actual = motion_lens[0].item()

            # ── Generate or reconstruct ──
            if trans is not None:
                # Transformer generation
                ids_length = motion_lens.detach().long().to(device) // 4
                motion_ids = trans.generate(
                    text, ids_length, time_steps, cond_scale,
                    topk_filter_thres=topkr, temperature=1,
                )
                motion_ids1 = motion_ids[:, :motion_ids.shape[1] // 2]
                motion_ids2 = motion_ids[:, motion_ids.shape[1] // 2:]

                motion1_out = net.forward_decoder(motion_ids1.unsqueeze(-1).to(device))
                motion2_out = net.forward_decoder(motion_ids2.unsqueeze(-1).to(device))
            else:
                # VQ-VAE reconstruction
                motion1_out, _, _ = net(motion1_gt[:, :T_actual].float().to(device))
                motion2_out, _, _ = net(motion2_gt[:, :T_actual].float().to(device))

            gen_T = motion1_out.shape[1]

            # ── Denormalize: get 262-dim per person ──
            combined = torch.cat([motion1_out, motion2_out], dim=-1)
            combined = combined.reshape(1, gen_T, 2, -1)
            combined_np = normalizer.backward(combined.cpu().numpy())

            m1_decoded = combined_np[0, :, 0, :]  # (gen_T, 262) person 1
            m2_decoded = combined_np[0, :, 1, :]  # (gen_T, 262) person 2

            # ── Convert 262-dim → raw InterHuman format ──
            raw_p1 = decoded_262_to_raw_interhuman(m1_decoded)   # (gen_T+1, 312)
            raw_p2 = decoded_262_to_raw_interhuman(m2_decoded)   # (gen_T+1, 312)

            # ── Save raw npy files ──
            clip_name = name[0] if isinstance(name, (list, tuple)) else str(i)
            np.save(pjoin(motions_p1, f"{clip_name}.npy"), raw_p1)
            np.save(pjoin(motions_p2, f"{clip_name}.npy"), raw_p2)

            # ── Save text annotation ──
            text_str = text[0] if isinstance(text, (list, tuple)) else text
            with open(pjoin(annots_dir, f"{clip_name}.txt"), 'w') as f:
                f.write(text_str + "\n")

            # ── Optionally save 262-dim processed data ──
            if save_processed:
                np.save(pjoin(proc_dir, f"{clip_name}_person1.npy"), m1_decoded)
                np.save(pjoin(proc_dir, f"{clip_name}_person2.npy"), m2_decoded)

            # ── In test mode: visualize GT vs generated side by side ──
            if test_n > 0:
                _visualize_comparison(
                    motion1_gt, motion2_gt, motion_lens,
                    m1_decoded, m2_decoded,
                    normalizer, text_str, clip_name, vis_dir
                )

            clip_names.append(clip_name)
            saved += 1

    # ── Write split file ──
    with open(pjoin(split_dir, "test.txt"), 'w') as f:
        for cn in sorted(clip_names):
            f.write(cn + "\n")

    # Also write empty ignore list and train/val splits (needed by dataloader)
    open(pjoin(output_dir, "ignore_list.txt"), 'w').close()
    for split_name in ["train.txt", "val.txt"]:
        open(pjoin(split_dir, split_name), 'w').close()

    print(f"\nSaved {saved} clips in InterHuman dataset format:")
    print(f"  Root:    {output_dir}")
    print(f"  Person1: {motions_p1}/ ({saved} npy files)")
    print(f"  Person2: {motions_p2}/ ({saved} npy files)")
    print(f"  Annots:  {annots_dir}/ ({saved} txt files)")
    print(f"  Split:   {pjoin(split_dir, 'test.txt')} ({saved} entries)")
    if save_processed:
        print(f"  262-dim: {proc_dir}/ ({saved*2} npy files)")
    print(f"\nTo load with InterHumanDataset:")
    print(f"  opt.data_root = '{output_dir}'")
    print(f"  opt.mode = 'test'")
    print(f"  dataset = InterHumanDataset(opt, normalize=True)")

    return saved


# ═══════════════════════════════════════════════════════════════
# Visualization: GT vs Generated comparison
# ═══════════════════════════════════════════════════════════════

def _visualize_comparison(motion1_gt, motion2_gt, motion_lens,
                           m1_decoded, m2_decoded,
                           normalizer, text_str, clip_name, vis_dir):
    """
    Create side-by-side GT vs Generated stick figure MP4 comparison.

    Uses InterMask's native plot_3d_motion_2views for consistent visualization.
    """
    from utils.plot_script import plot_3d_motion_2views
    from utils import paramUtil

    T_actual = motion_lens[0].item()

    # ── Denormalize GT ──
    gt_combined = torch.cat([motion1_gt[:, :T_actual], motion2_gt[:, :T_actual]], dim=-1)
    gt_combined = gt_combined.reshape(1, T_actual, 2, -1)
    gt_np = normalizer.backward(gt_combined.cpu().numpy())

    gt_m1 = gt_np[0, :, 0, :]  # (T, 262)
    gt_m2 = gt_np[0, :, 1, :]  # (T, 262)

    T_gen = m1_decoded.shape[0]
    T_vis = min(T_actual, T_gen)

    # ── Extract positions (22, 3) ──
    import scipy.ndimage.filters as filters

    gt_joints1 = gt_m1[:T_vis, :66].reshape(-1, 22, 3)
    gt_joints2 = gt_m2[:T_vis, :66].reshape(-1, 22, 3)
    gen_joints1 = m1_decoded[:T_vis, :66].reshape(-1, 22, 3)
    gen_joints2 = m2_decoded[:T_vis, :66].reshape(-1, 22, 3)

    # Smooth for visualization
    gt_joints1 = filters.gaussian_filter1d(gt_joints1, 1, axis=0, mode='nearest')
    gt_joints2 = filters.gaussian_filter1d(gt_joints2, 1, axis=0, mode='nearest')
    gen_joints1 = filters.gaussian_filter1d(gen_joints1, 1, axis=0, mode='nearest')
    gen_joints2 = filters.gaussian_filter1d(gen_joints2, 1, axis=0, mode='nearest')

    # ── Save GT visualization ──
    gt_path = pjoin(vis_dir, f"{clip_name}_gt.mp4")
    plot_3d_motion_2views(
        gt_path, paramUtil.t2m_kinematic_chain,
        [gt_joints1, gt_joints2],
        title=f"GT: {text_str[:60]}", fps=30,
    )

    # ── Save Generated visualization ──
    gen_path = pjoin(vis_dir, f"{clip_name}_gen.mp4")
    plot_3d_motion_2views(
        gen_path, paramUtil.t2m_kinematic_chain,
        [gen_joints1, gen_joints2],
        title=f"Gen: {text_str[:60]}", fps=30,
    )

    # ── Print per-clip MPJPE ──
    # MPJPE: mean per-joint position error in meters
    pos_err_p1 = np.sqrt(((gt_joints1 - gen_joints1) ** 2).sum(axis=-1)).mean()
    pos_err_p2 = np.sqrt(((gt_joints2 - gen_joints2) ** 2).sum(axis=-1)).mean()
    print(f"  [{clip_name}] MPJPE: p1={pos_err_p1:.4f}m  p2={pos_err_p2:.4f}m")
    print(f"    GT  video: {gt_path}")
    print(f"    Gen video: {gen_path}")


# ═══════════════════════════════════════════════════════════════
# Validation: verify round-trip accuracy
# ═══════════════════════════════════════════════════════════════

def validate_roundtrip(output_dir, n_clips=3):
    """
    Verify that saving → loading through the dataloader reproduces the
    262-dim output accurately.

    Loads a few clips from the saved raw format through load_motion() +
    process_motion_np() and compares against the saved 262-dim data.
    """
    from data.utils import load_motion, process_motion_np

    proc_dir = pjoin(output_dir, "processed_262")
    motions_p1 = pjoin(output_dir, "motions_processed", "person1")

    if not os.path.isdir(proc_dir):
        print("WARNING: --save_processed not used; cannot validate round-trip.")
        print("  Re-run with --save_processed to enable validation.")
        return

    clips = sorted([f.replace(".npy", "") for f in os.listdir(motions_p1)
                     if f.endswith(".npy")])[:n_clips]

    print(f"\n{'='*60}")
    print(f"Validating round-trip on {len(clips)} clips...")

    for clip in clips:
        for pidx in range(2):
            pname = f"person{pidx + 1}"

            # Load saved raw npy through the standard pipeline
            raw_path = pjoin(output_dir, "motions_processed", pname, f"{clip}.npy")
            motion_loaded, _ = load_motion(raw_path, min_length=1, swap=False)
            if motion_loaded is None:
                print(f"  [{clip} {pname}] SKIP: load_motion returned None")
                continue
            motion_processed, _, _ = process_motion_np(motion_loaded, 0.001, 0, n_joints=22)

            # Load the saved 262-dim reference
            ref = np.load(pjoin(proc_dir, f"{clip}_person{pidx + 1}.npy"))

            # Compare (the round-trip may differ by 1 frame due to velocity)
            T_cmp = min(motion_processed.shape[0], ref.shape[0])

            # Position error (first 66 dims)
            pos_err = np.abs(motion_processed[:T_cmp, :66] - ref[:T_cmp, :66]).mean()

            # Rotation error (dims 132:258)
            rot_err = np.abs(motion_processed[:T_cmp, 132:258] - ref[:T_cmp, 132:258]).mean()

            status = "OK" if pos_err < 0.05 and rot_err < 0.01 else "WARN"
            print(f"  [{clip} {pname}] pos_err={pos_err:.6f}  rot_err={rot_err:.6f}  [{status}]")

    print(f"{'='*60}")
    print("Note: small position differences are expected because process_motion_np")
    print("re-computes floor/centering/face-Z+ from the saved positions.")
    print("Rotations should match exactly (body-local, frame-invariant).")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate InterMask motions and save in InterHuman dataset format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="interhuman",
                        choices=["interhuman"],
                        help="Dataset (currently InterHuman only)")
    parser.add_argument("--name", type=str, default="trans_default",
                        help="Experiment name (transformer or VQ-VAE)")
    parser.add_argument("--use_trans", type=distutils.util.strtobool, default=True,
                        help="Use transformer (True) or VQ-VAE reconstruction (False)")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--which_epoch", type=str, default="best_fid",
                        help="Checkpoint to load: best_fid, latest, etc.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/generated/interhuman_dataset)")
    # Transformer sampling params
    parser.add_argument("--cond_scale", type=float, default=2.0)
    parser.add_argument("--time_steps", type=int, default=20)
    parser.add_argument("--topkr", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    # Extra options
    parser.add_argument("--save_processed", action="store_true",
                        help="Also save 262-dim processed data for direct MPJPE comparison")
    parser.add_argument("--validate", action="store_true",
                        help="Validate round-trip accuracy after saving (requires --save_processed)")
    parser.add_argument("--test_n", type=int, default=0,
                        help="Quick test: only process first N clips and visualize them")

    args = parser.parse_args()
    device = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")
    print(f"Device: {device}")

    # ── Load model configs ──
    use_trans = bool(args.use_trans)

    if use_trans:
        trans_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'opt.txt')
        main_opt = get_opt(trans_opt_path, device)
        fixseed(args.seed)

        vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, main_opt.vq_name, 'opt.txt')
        vq_opt = get_opt(vq_opt_path, device)

        main_opt.num_tokens = vq_opt.nb_code
        main_opt.code_dim = vq_opt.code_dim
    else:
        vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'opt.txt')
        main_opt = get_opt(vq_opt_path, device)
        vq_opt = main_opt

    # ── Load models ──
    net = load_vq_model(vq_opt, f"{args.which_epoch}.tar", device)
    transformer = None
    if use_trans:
        transformer = load_trans_model(main_opt, f"{args.which_epoch}.tar", device)

    # ── Output directory ──
    if args.output_dir is None:
        mode = "generated" if use_trans else "reconstructed"
        args.output_dir = pjoin("data", f"{mode}_dataset", args.dataset_name)

    # ── Generate and save ──
    t0 = time.time()
    n = generate_and_save_dataset(
        main_opt, net, transformer, device, args.output_dir,
        args.time_steps, args.cond_scale, args.topkr,
        save_processed=args.save_processed or args.validate,
        test_n=args.test_n,
    )

    elapsed = time.time() - t0
    print(f"\nDone: {n} clips in {elapsed:.1f}s ({elapsed/max(n,1):.2f}s/clip)")

    # ── Validate round-trip ──
    if args.validate:
        validate_roundtrip(args.output_dir)


if __name__ == "__main__":
    main()
