#!/usr/bin/env bash
# Diagnose a clip: inspect raw data + render multi-angle views with captions.
# Usage: bash prepare_mesh_contact/diagnose_clip.sh [CLIP_ID] [DATASET]
#   CLIP_ID  defaults to 1
#   DATASET  defaults to interhuman

set -euo pipefail

CLIP_ID="${1:-1}"
DATASET="${2:-interhuman}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- activate conda (same paths as run_interhuman_batch.sh) ---
CONDA_BIN="${CONDA_BIN:-/mnt/vita/scratch/vita-staff/users/rh/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-intermask}"
CONDA_SH="${CONDA_SH:-$(dirname "$(dirname "$CONDA_BIN")")/etc/profile.d/conda.sh}"

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
elif [ -f "$CONDA_BIN" ]; then
    export PATH="$(dirname "$CONDA_BIN"):$PATH"
fi

if ! command -v conda &>/dev/null; then
    echo "Conda not found. Set CONDA_BIN or CONDA_SH to a valid installation." >&2
    exit 1
fi

conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

OUT_DIR="output/renders/diagnose_${DATASET}_${CLIP_ID}"
mkdir -p "$OUT_DIR"

echo "=== Step 1: Inspect raw pkl data for clip ${CLIP_ID} ==="
python -c "
import pickle, numpy as np, os, sys
from scipy.spatial.transform import Rotation

dataset = '${DATASET}'
clip_id = '${CLIP_ID}'

if dataset == 'interhuman':
    candidates = [
        f'data/InterHuman/motions/{clip_id}.pkl',
        f'data/InterHuman/{clip_id}.pkl',
    ]
    pkl_path = next((p for p in candidates if os.path.isfile(p)), None)
    if pkl_path is None:
        print(f'ERROR: clip {clip_id} pkl not found'); sys.exit(1)

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    print(f'Pkl path: {pkl_path}')
    print(f'Top-level keys: {list(raw.keys())}')
    print()

    for key in ('person1', 'person2'):
        if key not in raw:
            continue
        p = raw[key]
        print(f'--- {key} ---')
        for field in ('trans', 'root_orient', 'pose_body', 'betas', 'pose_hand', 'gender'):
            if field in p:
                v = np.asarray(p[field])
                print(f'  {field}: shape={v.shape}, dtype={v.dtype}')
                if field == 'root_orient':
                    # Show first 3 frames
                    for t in range(min(3, v.shape[0])):
                        rv = v[t]
                        angle_deg = np.degrees(np.linalg.norm(rv))
                        euler = Rotation.from_rotvec(rv).as_euler('xyz', degrees=True)
                        print(f'    frame {t}: rotvec={rv}, angle={angle_deg:.1f}deg, euler_xyz={euler}')
                if field == 'pose_body':
                    # Show spine joints (indices 0,3,6 = spine1,spine2,spine3 in SMPL-X body joints)
                    # SMPL-X body joint order: 0=pelvis_child(L_Hip)? Actually:
                    # The 21 body joints in SMPL-X (pose_body) are joints 1-21:
                    #   0:L_Hip, 1:R_Hip, 2:Spine1, 3:L_Knee, 4:R_Knee, 5:Spine2,
                    #   6:L_Ankle, 7:R_Ankle, 8:Spine3, 9:L_Foot, 10:R_Foot, 11:Neck,
                    #   12:L_Collar, 13:R_Collar, 14:Head, 15:L_Shoulder, 16:R_Shoulder,
                    #   17:L_Elbow, 18:R_Elbow, 19:L_Wrist, 20:R_Wrist
                    spine_indices = [2, 5, 8, 11, 14]  # Spine1, Spine2, Spine3, Neck, Head
                    spine_names = ['Spine1', 'Spine2', 'Spine3', 'Neck', 'Head']
                    body = v.reshape(-1, 21, 3)
                    for si, sname in zip(spine_indices, spine_names):
                        rv = body[0, si]
                        angle = np.degrees(np.linalg.norm(rv))
                        euler = Rotation.from_rotvec(rv).as_euler('xyz', degrees=True)
                        print(f'    frame0 {sname}(j{si}): rotvec={np.round(rv,4)}, angle={angle:.1f}deg, euler={np.round(euler,1)}')
                if field == 'betas':
                    print(f'    values: {np.round(v[:10] if v.ndim==1 else v[0,:10], 4)}')
                if field == 'trans':
                    print(f'    frame0: {v[0]}')
            else:
                print(f'  {field}: NOT PRESENT')
        print()
else:
    print('InterX clip inspection not implemented in this script')
"

echo ""
echo "=== Step 2: Render clip ${CLIP_ID} from multiple angles (with captions) ==="

for AZIM in -60 0 60 180; do
    echo "  Rendering azim=${AZIM}..."
    python prepare_mesh_contact/render_contact_headless.py \
        --dataset "$DATASET" \
        --clip "$CLIP_ID" \
        --out-dir "${OUT_DIR}/azim_${AZIM}" \
        --max-frames 5 \
        --frame-policy first \
        --show-caption \
        --caption-lines 3 \
        --azim "$AZIM" \
        --dpi 150
done

echo ""
echo "=== Step 3: Re-render batch sample with captions ==="
python prepare_mesh_contact/render_contact_headless.py \
    --batch interhuman \
    --frames-per-clip 1 \
    --max-clips 10 \
    --out-dir output/renders/interhuman_sample_captioned \
    --show-caption \
    --caption-lines 3

echo ""
echo "=== Done ==="
echo "Multi-angle renders: ${OUT_DIR}/"
echo "Batch with captions: output/renders/interhuman_sample_captioned/"
echo ""
echo "Check these files to determine:"
echo "  1. Do captions appear in the info panel?"
echo "  2. Does person2's twist look the same from all angles (real) or only from one angle (projection artifact)?"
