#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NFS_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# Local staging area on non-NFS filesystem (needed because Docker daemon
# cannot mount NFS paths with root_squash)
LOCAL_REPO="/var/tmp/imnewtoken-docker"

echo "[*] Syncing repo to local staging area: $LOCAL_REPO"
mkdir -p "$LOCAL_REPO"
rsync -a --delete \
    --exclude='.git/' \
    --exclude='myvenv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='/data/' \
    --exclude='InterHuman_dataset/' \
    "$NFS_REPO/" "$LOCAL_REPO/"

# Sync InterHuman dataset (follows NFS symlink → local copy, skips large zips)
DATASET_SRC="$(realpath "$NFS_REPO/InterHuman_dataset")"
DATASET_DST="$LOCAL_REPO/InterHuman_dataset"
if [ -d "$DATASET_SRC" ]; then
    echo "[*] Syncing InterHuman dataset: $DATASET_SRC → $DATASET_DST"
    [ -L "$DATASET_DST" ] && rm "$DATASET_DST"
    mkdir -p "$DATASET_DST"
    rsync -a \
        --exclude='*.zip' \
        "$DATASET_SRC/" "$DATASET_DST/"
else
    echo "[warn] InterHuman dataset not found at $DATASET_SRC, skipping"
fi

export REPO_PATH="$LOCAL_REPO"

# X11 auth setup
XAUTH=/tmp/.docker.xauth
touch "$XAUTH"
chmod 600 "$XAUTH"
if [ -n "$DISPLAY" ]; then
    xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
fi
xhost +local:docker 2>/dev/null || echo "[warn] xhost non disponibile, continuo comunque"
export XAUTHORITY="$XAUTH"

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

echo "[*] Repo path: $REPO_PATH"
echo "[*] Running as UID=$HOST_UID GID=$HOST_GID"

# build   → only build the image
# --no-gpu → run without GPU (when nvidia-container-toolkit is missing)
# (default) → run with GPU
if [[ "$1" == "build" ]]; then
    echo "[*] Building ProtoMotion image"
    docker compose -f "$SCRIPT_DIR/docker-compose.protomotion.yml" build
elif [[ "$1" == "--no-gpu" ]]; then
    echo "[*] GPU disabled"
    docker compose -f "$SCRIPT_DIR/docker-compose.protomotion.yml" run --rm protomotion bash
else
    echo "[*] GPU enabled"
    docker compose -f "$SCRIPT_DIR/docker-compose.protomotion.yml" run --rm protomotion bash
fi
