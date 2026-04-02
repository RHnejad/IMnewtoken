# Mesh Contact Update + Repo Sync Note (2026-04-02)

## Context
This note documents the mesh-contact pipeline work completed for InterHuman/InterX and the repository sync preparation for mirroring `InterMask` into `IMnewtoken` for GitHub push/EPFL usage.

## What was implemented

### 1) New contact analysis module
Created `prepare_mesh_contact/` with:
- `mesh_contact_pipeline.py`
- `visualize_contact_newton.py`
- `run_contact_visual_examples.sh`
- `README.md`

Capabilities:
- Reconstructs SMPL-X meshes for person1/person2 per frame.
- Computes contact state (`penetrating`, `touching`, `barely_touching`, `not_touching`).
- Supports InterHuman PKL and InterX H5 (single-file and train/val/test split layouts).
- Exports JSON summaries, optional detail pickles, optional colored PLY frame exports.
- Newton GUI overlays for full-sequence inspection, including:
  - contact points
  - closest-point segment
  - 22-joint skeleton for both persons
  - clip text annotation panel

### 2) Penetration-source disambiguation
Updated JSON schema to separate penetration causes:
- `has_inter_person_penetration`
- `has_self_penetration`
- `penetration_source` in `{other_person_only, self_only, both, none}`

Per-frame metrics now include separate counts/depth estimates for:
- inter-person penetration
- self-penetration (optional heuristic mode)

### 3) Self-penetration handling
Added `--self-penetration-mode`:
- `off` (default; recommended for large-scale robust runs)
- `heuristic` (experimental topology-aware nearest non-neighbor detector)

Rationale:
- The heuristic can over-trigger depending on pose/topology and should not be silently mixed into production metrics.
- Defaulting to `off` keeps the core inter-person contact signal stable and reproducible.

### 4) Visualization improvements
Updated Newton viewer overlays to show:
- inter-person penetration points
- self-penetration points (different color)
- penetration source text in GUI

## Why these changes were made
- You requested contact output that distinguishes self-penetration from penetration caused by the other person.
- You requested frame-wise visual verification over full sequences in Newton GUI.
- The separation avoids conflating artifact-like self-collisions with true interaction contact/penetration between P1/P2.

## Repository sync preparation
To keep GitHub pushes clean and prevent accidental inclusion of local generated artifacts, `.gitignore` was extended with:
- `data/generated_intergen/`
- `data/imdy_metrics/`
- `data/residual_wrench_intergen/`
- `data/torque_stats_generated_log.txt`
- `data/body_model/smpl`
- `.codex`
- `.claude/projects/`

These are local/generated/runtime assets and should not be versioned.

## Notes for EPFL cluster usage
- GPU helps SMPL-X forward reconstruction.
- Current contact stage is primarily CPU-side (KD-tree + ray casting), so best throughput usually comes from many parallel CPU workers (optionally hybrid with GPU mesh reconstruction).
