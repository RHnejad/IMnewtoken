#!/bin/bash
# phc_generated_dataset.sh — Run PHC retargeting on generated InterMask/InterGen predictions.
#
# Run this INSIDE the PHC Docker container:
#   ./docker/run.sh          # enter container (syncs data/generated/ automatically)
#   bash /workspace/repo/phc_generated_dataset.sh
#
# Processes three datasets sequentially:
#   1. interhuman_intermask  (InterMask predictions for InterHuman, 1098 clips)
#   2. interx_intermask      (InterMask predictions for InterX,     1706 clips)
#   3. interhuman_intergen   (InterGen predictions for InterHuman,  1098 clips)
#
# Outputs (per dataset):
#   PHC/output/generated/{dataset_tag}.pkl
#       Converted PHC motion pkl (input to PHC inference)
#
#   PHC/output/generated/{dataset_tag}_retarget/retarget_<timestamp>.pkl
#       PHC inference output with dof_pos, torques, body_pos, root_states, etc.
#
# To run a single dataset only, pass its tag as argument:
#   bash phc_generated_dataset.sh interhuman_intermask

set -e
cd /workspace/repo

SMPL_DIR="PHC/data/smpl"
NUM_ENVS=16  # adjust to GPU memory; must be ≤ total motions in pkl

DATASETS=(interhuman_intermask interx_intermask interhuman_intergen)

# If a dataset tag is passed as argument, process only that one
if [ -n "$1" ]; then
    DATASETS=("$1")
fi

# --------------------------------------------------------------------------- #
# Step 1: Convert generated motions to PHC format                              #
# --------------------------------------------------------------------------- #
echo "========================================================"
echo " Step 1: Convert generated predictions → PHC format"
echo "========================================================"

python prepare7/batch_convert_generated_to_phc.py \
    --smpl-data-dir "$SMPL_DIR" \
    --subprocess-batch-size 300

# --------------------------------------------------------------------------- #
# Step 2: PHC inference for each dataset                                        #
# --------------------------------------------------------------------------- #
echo ""
echo "========================================================"
echo " Step 2: PHC inference per dataset"
echo "========================================================"

for DATASET_TAG in "${DATASETS[@]}"; do
    MOTION_PKL="output/generated/${DATASET_TAG}.pkl"
    OUT_LABEL="${DATASET_TAG}_retarget"

    if [ ! -f "/workspace/repo/PHC/${MOTION_PKL}" ]; then
        echo "[skip] ${MOTION_PKL} not found, skipping inference for ${DATASET_TAG}"
        continue
    fi

    echo ""
    echo "--- Inference: ${DATASET_TAG} ---"
    cd /workspace/repo/PHC

    python phc/run_hydra.py \
        learning=im_pnn exp_name=phc_kp_pnn_iccv \
        epoch=-1 test=True im_eval=True \
        env=env_im_pnn \
        robot.freeze_hand=True robot.box_body=False env.obs_v=7 \
        "env.motion_file=${MOTION_PKL}" \
        env.num_prim=4 \
        "env.num_envs=${NUM_ENVS}" \
        env.enableEarlyTermination=False \
        ++collect_dataset=True \
        headless=True

    cd /workspace/repo
done

# --------------------------------------------------------------------------- #
# Step 3: Sync output back to NFS                                              #
# --------------------------------------------------------------------------- #
NFS_REPO="$(cd "$(dirname "$0")" && pwd)"
LOCAL_REPO="/var/tmp/imnewtoken-docker"

if [ -d "$LOCAL_REPO" ] && [ "$LOCAL_REPO" != "$NFS_REPO" ]; then
    echo ""
    echo "========================================================"
    echo " Step 3: Syncing output back to NFS"
    echo "========================================================"
    rsync -av --ignore-existing \
        "$LOCAL_REPO/PHC/output/generated/" \
        "$NFS_REPO/PHC/output/generated/"
    rsync -av --ignore-existing \
        "$LOCAL_REPO/PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/" \
        "$NFS_REPO/PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/"
    echo "[done] Output synced to NFS."
else
    echo "[skip] NFS sync not needed."
fi

echo ""
echo "========================================================"
echo " Done. Retargeted data in:"
echo "   PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/"
echo "========================================================"
