#!/bin/bash
# Entrypoint for the ProtoMotion container.
# Patches the SMPL MJCF joint limits to match the frozen inference config,
# then hands off to whatever command was requested (default: bash).

set -e

MJCF="/workspace/repo/prepare7/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml"
PATCH="/workspace/repo/prepare7/patch_proto_mjcf.py"
CONFIGS="/workspace/repo/prepare7/ProtoMotions/data/pretrained_models/motion_tracker/smpl/resolved_configs_inference.pt"

if [ -f "$CONFIGS" ] && [ -f "$PATCH" ] && [ -f "$MJCF" ]; then
    echo "[entrypoint] Patching MJCF joint limits from frozen inference config..."
    python "$PATCH" && echo "[entrypoint] MJCF patch applied." || echo "[entrypoint] WARNING: MJCF patch failed, continuing anyway."
else
    echo "[entrypoint] Skipping MJCF patch (missing configs, patch script, or MJCF)."
fi

exec "$@"
