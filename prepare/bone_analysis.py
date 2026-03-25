"""Analyze bone length variance in InterHuman dataset."""
import os, numpy as np

PARENTS = {
    0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
    10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14,
    18: 16, 19: 17, 20: 18, 21: 19,
}
JOINT_NAMES = ['Pelvis','L_Hip','R_Hip','Spine1','L_Knee','R_Knee','Spine2',
               'L_Ankle','R_Ankle','Spine3','L_Foot','R_Foot','Neck',
               'L_Collar','R_Collar','Head','L_Shoulder','R_Shoulder',
               'L_Elbow','R_Elbow','L_Wrist','R_Wrist']

# Load ALL clips
per_clip_bones = {j: [] for j in range(1, 22)}
n_clips = 0
for pdir in ['data/InterHuman/motions_processed/person1',
             'data/InterHuman/motions_processed/person2']:
    if not os.path.isdir(pdir):
        continue
    for fname in sorted(os.listdir(pdir)):
        if not fname.endswith('.npy'):
            continue
        data = np.load(os.path.join(pdir, fname))
        pos = data[:, :66].reshape(-1, 22, 3)
        for j in range(1, 22):
            mean_bone = np.linalg.norm(pos[:, j] - pos[:, PARENTS[j]], axis=-1).mean()
            per_clip_bones[j].append(mean_bone)
        n_clips += 1

print(f"Total clips analyzed: {n_clips}")
print()
hdr = f"{'Link':<25s} | {'Mean':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s} | {'CV%':>8s}"
print(hdr)
print('-' * len(hdr))
for j in range(1, 22):
    vals = np.array(per_clip_bones[j]) * 100
    link = f"{JOINT_NAMES[PARENTS[j]]} -> {JOINT_NAMES[j]}"
    cv = (vals.std() / vals.mean()) * 100
    print(f"{link:<25s} | {vals.mean():>8.2f} | {vals.std():>8.2f} | {vals.min():>8.2f} | {vals.max():>8.2f} | {cv:>8.1f}")

# Within-clip variation
print()
print("--- Within-clip bone length variation (clip 1000, person1) ---")
data = np.load('data/InterHuman/motions_processed/person1/1000.npy')
pos = data[:, :66].reshape(-1, 22, 3)
print(f"Frames: {pos.shape[0]}")
for j in [4, 7, 18, 20]:
    bone = np.linalg.norm(pos[:, j] - pos[:, PARENTS[j]], axis=-1) * 100
    link = f"{JOINT_NAMES[PARENTS[j]]} -> {JOINT_NAMES[j]}"
    print(f"  {link:<25s}: mean={bone.mean():.2f} std={bone.std():.4f} min={bone.min():.2f} max={bone.max():.2f}")
