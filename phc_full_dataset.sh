#!/bin/bash
# phc_full_dataset.sh — Run PHC retargeting on the full InterHuman dataset.
#
# Run this INSIDE the PHC Docker container:
#   ./docker/run.sh          # enter container
#   bash /workspace/repo/phc_full_dataset.sh
#
# Outputs:
#   PHC/output/interhuman/all_clips.pkl            — merged PHC motion pkl
#   PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/all_clips/retarget_<timestamp>.pkl
#       Keys per clip (list indexed by key_names):
#         dof_pos     list[T × 69]        joint positions (exponential map, rad)
#         torques     list[T × 69]        applied forces from Isaac Gym PD controller (N·m)
#         body_pos    list[T × 24 × 3]    3-D world positions of all 24 SMPL bodies (m)
#         root_states list[T × 7]         root pos (3) + quaternion xyzw (4)
#         key_names   array[N]            motion keys matching clip order
#         clean_action list[T × 69]       PD position targets (network output)
#         obs         list[T × 945]       policy observation vectors
#         ...

set -e
cd /workspace/repo

INTERHUMAN_DIR="InterHuman_dataset/motions"
SMPL_DIR="PHC/data/smpl"
COMBINED_PKL="PHC/output/interhuman/all_clips.pkl"
NUM_ENVS=16   # adjust to your GPU memory; must be ≤ total number of motions

# --------------------------------------------------------------------------- #
# Step 1: Batch-convert all InterHuman clips to a single PHC pkl              #
# --------------------------------------------------------------------------- #
echo "========================================================"
echo " Step 1: Batch conversion of InterHuman → PHC format"
echo "========================================================"

# The script saves one small pkl per clip in the staging dir before merging,
# so it is safe to interrupt and re-run — already-converted clips are skipped.
python prepare7/batch_convert_interhuman_to_phc.py \
    --interhuman-dir "$INTERHUMAN_DIR" \
    --smpl-data-dir  "$SMPL_DIR" \
    --output         "$COMBINED_PKL" \
    --subprocess-batch-size 300

# --------------------------------------------------------------------------- #
# Step 2: Run PHC inference in headless collect_dataset mode                  #
# --------------------------------------------------------------------------- #
echo ""
echo "========================================================"
echo " Step 2: PHC inference + retargeting data export"
echo "========================================================"

cd /workspace/repo/PHC

# im_eval=True  — iterate through every motion, compute MPJPE, and exit when done
# collect_dataset=True — save dof_pos, torques, body_pos, root_states per clip
python phc/run_hydra.py \
    learning=im_pnn exp_name=phc_kp_pnn_iccv \
    epoch=-1 test=True im_eval=True \
    env=env_im_pnn \
    robot.freeze_hand=True robot.box_body=False env.obs_v=7 \
    env.motion_file=output/interhuman/all_clips.pkl \
    env.num_prim=4 \
    env.num_envs=$NUM_ENVS \
    env.enableEarlyTermination=False \
    ++collect_dataset=True \
    headless=True

echo ""
echo "========================================================"
echo " Done. Retargeted data saved in:"
echo "   PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/all_clips/"
echo "========================================================"

# --------------------------------------------------------------------------- #
# Step 3: Sync output back to NFS (permanent storage)                         #
# --------------------------------------------------------------------------- #
# The container works on /var/tmp/imnewtoken-docker (non-NFS local copy).
# This step copies the generated output back to the NFS repo so it survives
# container restarts and is accessible from other machines.
NFS_REPO="$(cd "$(dirname "$0")" && pwd)"   # real NFS path of this repo
LOCAL_REPO="/var/tmp/imnewtoken-docker"

if [ -d "$LOCAL_REPO" ] && [ "$LOCAL_REPO" != "$NFS_REPO" ]; then
    echo ""
    echo "========================================================"
    echo " Step 3: Syncing output back to NFS"
    echo "   $LOCAL_REPO/PHC/output → $NFS_REPO/PHC/output"
    echo "========================================================"
    rsync -av --ignore-existing \
        "$LOCAL_REPO/PHC/output/interhuman/" \
        "$NFS_REPO/PHC/output/interhuman/"
    rsync -av --ignore-existing \
        "$LOCAL_REPO/PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/" \
        "$NFS_REPO/PHC/output/HumanoidIm/phc_kp_pnn_iccv/phc_act/"
    echo "[done] Output synced to NFS."
else
    echo "[skip] NFS sync not needed (running directly on NFS or LOCAL_REPO not found)."
fi
