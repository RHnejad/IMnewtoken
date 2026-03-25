"""
generate_and_save.py — Generate motions using InterMask and save decoded pose
sequences in the same format as the original dataset for Newton visualization.

For InterHuman: saves pkl files with {person1: {root_orient, pose_body, trans,
    betas}, person2: {...}} — directly compatible with prepare2/retarget.py.
For InterX: saves H5 file with (T, 56, 6) per clip — directly compatible with
    prepare2/retarget.py's load_interx_clip().

Usage:
    # InterHuman (text-to-motion via transformer)
    python generate_and_save.py --dataset_name interhuman --name trans_default

    # InterX (text-to-motion via transformer)
    python generate_and_save.py --dataset_name interx --name trans_default

    # VQ-VAE reconstruction only (no transformer)
    python generate_and_save.py --dataset_name interhuman --name vq_default --use_trans False

    # Custom output directory
    python generate_and_save.py --dataset_name interhuman --name trans_default --output_dir my_output/
"""
import os
import sys
import pickle
import time
import numpy as np
import torch
from os.path import join as pjoin
from tqdm import tqdm
import distutils.util

from models.vq.model import RVQVAE
from models.mask_transformer.transformer import MaskTransformer
from utils.get_opt import get_opt
from utils.utils import fixseed
from data.utils import MotionNormalizer
import data.rotation_conversions as geometry

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')


# ═══════════════════════════════════════════════════════════════
# Model loading (from eval.py)
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
# InterHuman conversion: VQ output → pkl (retarget.py format)
# ═══════════════════════════════════════════════════════════════

# Inverse of the trans_matrix used in data/utils.py process_motion_np:
#   trans_matrix maps (x,y,z) → (x, z, -y)
#   inv_trans maps  (px,py,pz) → (px, -pz, py)  [back to Z-up world]
INV_TRANS_MATRIX = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0],
], dtype=np.float64)


def _rot6d_to_axis_angle_np(rot6d):
    """Convert (*, 6) rotation 6D numpy → (*, 3) axis-angle numpy."""
    rot6d_t = torch.from_numpy(rot6d).float()
    mat = geometry.rotation_6d_to_matrix(rot6d_t)
    aa = geometry.matrix_to_axis_angle(mat)
    return aa.numpy().astype(np.float64)


def _estimate_root_orient_interhuman(positions_zup):
    """
    Estimate root_orient (axis-angle, T×3) from 22-joint positions in Z-up
    world frame.  Uses hip positions to determine facing direction + yaw.

    root_orient maps SMPL-X body-local (Y-up, Z-forward) → Z-up world.

    Only estimates yaw (no pitch/roll) — sufficient for natural visualization.
    """
    from scipy.spatial.transform import Rotation

    T = positions_zup.shape[0]
    root_orient = np.zeros((T, 3), dtype=np.float64)

    # Joint indices: 1 = L_Hip, 2 = R_Hip
    l_hip = positions_zup[:, 1, :]
    r_hip = positions_zup[:, 2, :]

    for t in range(T):
        across = r_hip[t] - l_hip[t]  # right − left (in Z-up world)
        across[2] = 0.0  # project to XY plane
        norm_a = np.linalg.norm(across)
        if norm_a < 1e-8:
            across = np.array([1.0, 0.0, 0.0])
        else:
            across = across / norm_a

        # forward = up × across  (Z-up: up = [0,0,1])
        forward = np.cross(np.array([0.0, 0.0, 1.0]), across)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward = forward / forward_norm

        # yaw angle of forward in XY plane
        yaw_rad = np.arctan2(forward[1], forward[0]) + np.pi / 2.0

        # root_orient = R_z(yaw) · R_x(90°)
        # R_x(90°) maps SMPL-X Y(up) → Z(up), Z(forward) → -Y
        # R_z(yaw) rotates around Z to correct facing direction
        R_root = Rotation.from_euler('xz', [90.0, np.degrees(yaw_rad)], degrees=True)
        root_orient[t] = R_root.as_rotvec()

    return root_orient


def decoded_interhuman_to_pkl(motion1_decoded, motion2_decoded):
    """
    Convert two decoded InterHuman motions (T, 262) → pkl dict for retarget.py.

    The 262-dim layout: positions(66) + velocities(66) + rotations(126) + fc(4).

    NOTE: The rot6d in the 262-dim format comes from the InterHuman npy pipeline,
    which uses a DIFFERENT body model than SMPL-X. Converting rot6d → axis-angle
    and treating as SMPL-X pose_body is approximate at best. The raw 22-joint
    positions (stored as 'positions_zup') are the faithful InterMask output.

    Returns dict: {person1: {root_orient, pose_body, trans, betas, positions_zup},
                   person2: {...}}
    """
    results = {}
    for pidx, motion in enumerate([motion1_decoded, motion2_decoded]):
        T = motion.shape[0]

        # Extract positions (T, 22, 3) in processed frame
        positions_proc = motion[:, :66].reshape(T, 22, 3).astype(np.float64)

        # Extract body rotations (T, 21, 6) — from InterHuman npy convention
        # WARNING: these are NOT SMPL-X body-local rotations
        rot6d = motion[:, 132:132 + 126].reshape(T, 21, 6)

        # Convert body rotations: 6D → axis-angle → (T, 63)
        # NOTE: approximate — rot6d convention differs from SMPL-X
        pose_body = _rot6d_to_axis_angle_np(rot6d).reshape(T, 63)

        # Convert positions from processed frame back to Z-up world
        # (undo trans_matrix)
        positions_zup = np.einsum("mn,...n->...m", INV_TRANS_MATRIX, positions_proc)

        # trans = pelvis position in Z-up world
        trans = positions_zup[:, 0, :].copy()

        # Estimate root orientation from hip positions
        root_orient = _estimate_root_orient_interhuman(positions_zup)

        results[f'person{pidx + 1}'] = {
            'root_orient': root_orient,       # (T, 3)
            'pose_body': pose_body,            # (T, 63) — approximate, NOT true SMPL-X
            'trans': trans,                    # (T, 3)
            'betas': np.zeros(10, dtype=np.float64),
            'positions_zup': positions_zup,    # (T, 22, 3) — raw InterMask positions, Z-up
        }

    return results


# ═══════════════════════════════════════════════════════════════
# InterX conversion: VQ output → H5 (retarget.py format)
# ═══════════════════════════════════════════════════════════════

def decoded_interx_to_raw(motion1_decoded, motion2_decoded):
    """
    Convert two decoded InterX motions (T, 56, 6) → (T, 56, 6) raw format.

    VQ-VAE (motion_rep=smpl) format:
        joints 0..54: 6D rotation representation
        joint 55: [transl_x, transl_y, transl_z, vel_x, vel_y, vel_z]

    Raw H5 format (per combined clip):
        (T, 56, 6) where dims[:3] = person1 axis-angle, dims[3:6] = person2
        joints 0..54: axis-angle (3D) per person
        joint 55: translation (3D) per person
    """
    T = motion1_decoded.shape[0]
    out = np.zeros((T, 56, 6), dtype=np.float32)

    for pidx, motion in enumerate([motion1_decoded, motion2_decoded]):
        # Rotations for joints 0..54: rot_6d (T, 55, 6) → axis_angle (T, 55, 3)
        rot6d = torch.from_numpy(motion[:, :55, :]).float()
        mat = geometry.rotation_6d_to_matrix(rot6d)
        aa = geometry.matrix_to_axis_angle(mat).numpy()  # (T, 55, 3)

        # Translation from joint 55: first 3 dims
        transl = motion[:, 55, :3]  # (T, 3)

        # Assemble per-person (T, 56, 3)
        per_person = np.zeros((T, 56, 3), dtype=np.float32)
        per_person[:, :55, :] = aa
        per_person[:, 55, :] = transl

        out[:, :, pidx * 3:(pidx + 1) * 3] = per_person

    return out


def decoded_interx_to_pkl(motion1_decoded, motion2_decoded):
    """
    Convert two decoded InterX motions to pkl dict for retarget.py.

    Same as decoded_interx_to_raw but returns pkl format matching InterHuman.
    This is useful for a unified retarget pipeline.
    """
    raw = decoded_interx_to_raw(motion1_decoded, motion2_decoded)
    T = raw.shape[0]
    results = {}

    for pidx in range(2):
        p = raw[:, :, pidx * 3:(pidx + 1) * 3].astype(np.float64)
        results[f'person{pidx + 1}'] = {
            'root_orient': p[:, 0, :],                    # (T, 3)
            'pose_body': p[:, 1:22, :].reshape(T, 63),    # (T, 63)
            'trans': p[:, 55, :],                          # (T, 3)
            'betas': np.zeros(10, dtype=np.float64),
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Main generation loop
# ═══════════════════════════════════════════════════════════════

def generate_interhuman(opt, net, trans, device, output_dir, time_steps, cond_scale, topkr):
    """Generate and save all InterHuman test set motions."""
    from data.interhuman import InterHumanDataset
    from torch.utils.data import DataLoader

    normalizer = MotionNormalizer()

    # Load test dataset
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

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    print(f"\nGenerating InterHuman test set motions → {output_dir}")
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Generate")):
            name, text, motion1_gt, motion2_gt, motion_lens = data
            T_actual = motion_lens[0].item()

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

            # Combine and denormalize: (1, T, 2, 262)
            combined = torch.cat([motion1_out, motion2_out], dim=-1)
            combined = combined.reshape(1, gen_T, 2, -1)
            combined_np = normalizer.backward(combined.cpu().numpy())

            # Extract per-person decoded motions
            m1_decoded = combined_np[0, :, 0, :]  # (T, 262)
            m2_decoded = combined_np[0, :, 1, :]  # (T, 262)

            # Convert to pkl format
            pkl_data = decoded_interhuman_to_pkl(m1_decoded, m2_decoded)

            # Add metadata
            pkl_data['text'] = text[0] if isinstance(text, (list, tuple)) else text
            pkl_data['clip_name'] = name[0] if isinstance(name, (list, tuple)) else name

            # Save
            clip_name = name[0] if isinstance(name, (list, tuple)) else str(i)
            out_path = pjoin(output_dir, f"{clip_name}.pkl")
            with open(out_path, 'wb') as f:
                pickle.dump(pkl_data, f)

            saved += 1

    print(f"Saved {saved} InterHuman generated clips to {output_dir}")
    return saved


def generate_interx(opt, net, trans, device, output_dir, time_steps, cond_scale, topkr):
    """Generate and save all InterX test set motions."""
    import codecs as cs
    import h5py
    from data.interx import Text2MotionDatasetV2HHI, collate_fn
    from torch.utils.data import DataLoader
    from utils.word_vectorizer import WordVectorizer

    data_cfg = opt
    data_cfg.data_root = 'data/Inter-X_Dataset'
    data_cfg.motion_dir = pjoin(data_cfg.data_root, 'processed/motions')
    data_cfg.text_dir = pjoin(data_cfg.data_root, 'processed/texts_processed')
    data_cfg.motion_rep = "smpl"
    data_cfg.joints_num = 56
    data_cfg.max_motion_length = 150
    data_cfg.max_text_len = 35
    data_cfg.unit_length = 4

    w_vectorizer = WordVectorizer(pjoin(data_cfg.data_root, 'processed/glove'), 'hhi_vab')
    dataset = Text2MotionDatasetV2HHI(
        data_cfg,
        pjoin(data_cfg.data_root, 'splits/test.txt'),
        w_vectorizer,
        pjoin(data_cfg.motion_dir, 'test.h5'),
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=collate_fn)

    net = net.to(device).eval()
    if trans is not None:
        trans = trans.to(device).eval()

    os.makedirs(output_dir, exist_ok=True)
    h5_path = pjoin(output_dir, "generated.h5")
    pkl_dir = pjoin(output_dir, "pkl")
    os.makedirs(pkl_dir, exist_ok=True)

    saved = 0

    print(f"\nGenerating InterX test set motions → {output_dir}")
    with torch.no_grad(), h5py.File(h5_path, 'w') as h5f:
        for i, data in enumerate(tqdm(dataloader, desc="Generate")):
            word_emb, pos_ohot, caption, cap_lens, motions, motion_lens, tokens = data
            T_actual = motion_lens[0].item()

            motion1_gt, motion2_gt = motions.split(motions.shape[-1] // 2, dim=-1)

            if trans is not None:
                # Transformer generation
                ids_length = motion_lens.detach().long().to(device) // 4
                motion_ids = trans.generate(
                    caption, ids_length, time_steps, cond_scale,
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
            m1 = motion1_out[0, :gen_T].cpu().numpy()  # (T, 56, 6)
            m2 = motion2_out[0, :gen_T].cpu().numpy()  # (T, 56, 6)

            # Convert to raw (T, 56, 6) format
            raw = decoded_interx_to_raw(m1, m2)

            # Get clip name from dataset
            idx = dataset.pointer + i
            if idx < len(dataset.name_list):
                clip_name = dataset.name_list[idx]
            else:
                clip_name = f"gen_{i:06d}"

            # Save to H5
            h5f.create_dataset(clip_name, data=raw)

            # Also save as pkl for retarget.py
            pkl_data = decoded_interx_to_pkl(m1, m2)
            pkl_data['text'] = caption[0] if isinstance(caption, (list, tuple)) else caption
            pkl_data['clip_name'] = clip_name
            with open(pjoin(pkl_dir, f"{clip_name}.pkl"), 'wb') as f:
                pickle.dump(pkl_data, f)

            saved += 1

    print(f"Saved {saved} InterX generated clips")
    print(f"  H5:  {h5_path}")
    print(f"  PKL: {pkl_dir}/")
    return saved


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate InterMask motions and save in dataset-native format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["interhuman", "interx"])
    parser.add_argument("--name", type=str, default="trans_default",
                        help="Transformer experiment name")
    parser.add_argument("--use_trans", type=distutils.util.strtobool, default=True,
                        help="Use transformer (True) or VQ-VAE reconstruction (False)")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--which_epoch", type=str, default="best_fid",
                        help="Checkpoint to load: best_fid, latest, etc.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/generated/<dataset>)")
    # Transformer sampling params
    parser.add_argument("--cond_scale", type=float, default=2.0)
    parser.add_argument("--time_steps", type=int, default=20)
    parser.add_argument("--topkr", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)

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
        args.output_dir = pjoin("data", mode, args.dataset_name)

    # ── Generate ──
    t0 = time.time()

    if args.dataset_name == "interhuman":
        n = generate_interhuman(
            main_opt, net, transformer, device, args.output_dir,
            args.time_steps, args.cond_scale, args.topkr,
        )
    elif args.dataset_name == "interx":
        n = generate_interx(
            main_opt, net, transformer, device, args.output_dir,
            args.time_steps, args.cond_scale, args.topkr,
        )

    elapsed = time.time() - t0
    print(f"\nDone: {n} clips in {elapsed:.1f}s ({elapsed/max(n,1):.2f}s/clip)")
    print(f"Output: {args.output_dir}")
    print(f"\nTo retarget for Newton visualization:")
    print(f"  python prepare2/retarget.py --dataset {args.dataset_name} "
          f"--data_dir {args.output_dir} --eval")


if __name__ == "__main__":
    main()
