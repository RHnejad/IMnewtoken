#!/bin/bash
# phc_interx_gt_dataset.sh — Run PHC retargeting on InterX GT test split.
#
# Run INSIDE the PHC Docker container:
#   ./docker/run.sh
#   bash /workspace/repo/phc_interx_gt_dataset.sh
#
# Outputs:
#   PHC/output/interx_gt/test.pkl                      — converted PHC motion pkl
#   PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/interx_gt/retarget_<timestamp>.pkl
#       dof_pos, torques, body_pos, root_states, key_names, ...

set -e
cd /workspace/repo

SMPL_DIR="PHC/data/smpl"
COMBINED_PKL="PHC/output/interx_gt/test.pkl"
NUM_ENVS=16   # adjust to GPU memory

# --------------------------------------------------------------------------- #
# Step 1: Convert InterX test split to PHC format                             #
# --------------------------------------------------------------------------- #
echo "========================================================"
echo " Step 1: Convert InterX GT test split → PHC format"
echo "========================================================"

python prepare7/batch_convert_interx_to_phc.py \
    --smpl-data-dir "$SMPL_DIR" \
    --split test \
    --subprocess-batch-size 300

# --------------------------------------------------------------------------- #
# Step 2: PHC inference                                                        #
# --------------------------------------------------------------------------- #
if [ ! -f "/workspace/repo/$COMBINED_PKL" ]; then
    echo "[error] $COMBINED_PKL not found — conversion failed. Aborting."
    exit 1
fi

echo ""
echo "========================================================"
echo " Step 2: PHC inference on InterX GT"
echo "========================================================"

cd /workspace/repo/PHC

# Use a dedicated output label so the retarget pkl lands in interx_gt/ subfolder
# (PHC names the output dir after exp_name; we rename the run dir via env.cfg_save_name)
python phc/run_hydra.py \
    learning=im_pnn exp_name=phc_kp_pnn_iccv \
    epoch=-1 test=True im_eval=True \
    env=env_im_pnn \
    robot.freeze_hand=True robot.box_body=False env.obs_v=7 \
    env.motion_file=output/interx_gt/test.pkl \
    env.num_prim=4 \
    "env.num_envs=${NUM_ENVS}" \
    env.enableEarlyTermination=False \
    ++collect_dataset=True \
    headless=True

cd /workspace/repo

# --------------------------------------------------------------------------- #
# Step 3: Sync output back to NFS                                             #
# --------------------------------------------------------------------------- #
NFS_REPO="$(cd "$(dirname "$0")" && pwd)"
LOCAL_REPO="/var/tmp/imnewtoken-docker"

if [ -d "$LOCAL_REPO" ] && [ "$LOCAL_REPO" != "$NFS_REPO" ]; then
    echo ""
    echo "========================================================"
    echo " Step 3: Syncing output back to NFS"
    echo "========================================================"
    rsync -av --ignore-existing \
        "$LOCAL_REPO/PHC/output/interx_gt/" \
        "$NFS_REPO/PHC/output/interx_gt/"
    rsync -av --ignore-existing \
        "$LOCAL_REPO/PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/" \
        "$NFS_REPO/PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/"
    echo "[done] Output synced to NFS."
fi

echo ""
echo "========================================================"
echo " Done."
echo "========================================================"
