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
    --exclude='data/' \
    "$NFS_REPO/" "$LOCAL_REPO/"

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
    echo "[*] Building image"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" build
elif [[ "$1" == "--no-gpu" ]]; then
    echo "[*] GPU disabled (using docker-compose.nogpu.yml)"
    docker compose -f "$SCRIPT_DIR/docker-compose.nogpu.yml" run --rm phc bash
else
    echo "[*] GPU enabled"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm phc bash
fi
