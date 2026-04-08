# InterHuman Generated-vs-GT Mesh-Contact Workflow

Date: 2026-04-08

## Summary

This document captures the generated-vs-GT mesh-contact tooling that was previously living as EPFL-only `prepare_mesh_contact` work and is now folded into the canonical pipeline.

The workflow adds two capabilities on top of the base InterHuman / InterX mesh-contact analyzer:

1. Reconstruct generated InterHuman clips with GT betas via `--betas-from-interhuman-root`
2. Render or summarize generated-vs-GT contact behavior for matching clip ids

## What Was Added

Core pipeline support:

- `prepare_mesh_contact/mesh_contact_pipeline.py`
  - missing-betas fallback when loading InterHuman `.pkl` clips
  - `--betas-from-interhuman-root` for GT-beta override
  - provenance block in output JSON
  - zero-frame clip handling that writes an explicit empty summary instead of crashing
  - `CKDTREE_WORKERS` environment override for SciPy KD-tree queries
  - `max_inside_queries_per_mesh=256`

Operational helpers:

- `prepare_mesh_contact/render_contact_headless.py`
- `prepare_mesh_contact/render_interhuman_generated_vs_gt.py`
- `prepare_mesh_contact/summarize_interhuman_generated_vs_gt.py`
- `prepare_mesh_contact/run_interhuman_generated_vs_gt.sh`
- batch / retry / monitor helpers under `prepare_mesh_contact/*.sh`

## Data Requirements

The generated-vs-GT path assumes the generated InterHuman root is SMPL-X-parameter compatible.

Each generated clip must expose per-person dictionaries with at least:

- `trans`
- `root_orient`
- `pose_body`

Optional:

- `betas`
- hand pose fields supported by `mesh_contact_pipeline.py`

The current local `data/generated_intergen/interhuman/*.pkl` files do **not** satisfy this requirement. They store feature/position outputs such as `features_262`, `positions_yup`, and `positions_zup`, not SMPL-X pose parameters. Those files cannot be passed directly to the mesh-contact reconstruction code.

## Recommended Usage

Single comparison render:

```bash
python prepare_mesh_contact/render_interhuman_generated_vs_gt.py \
  --generated-root /path/to/generated_smplx_interhuman \
  --gt-root data/InterHuman \
  --clip 4278 \
  --frames-per-clip 1 \
  --frame-policy representative \
  --out-dir output/mesh_contact/generated_vs_gt_renders
```

Batch wrapper:

```bash
GENERATED_ROOT=/path/to/generated_smplx_interhuman \
prepare_mesh_contact/run_interhuman_generated_vs_gt.sh
```

If you want summary numbers from already-generated JSON outputs:

```bash
python prepare_mesh_contact/summarize_interhuman_generated_vs_gt.py \
  --generated-json-dir output/mesh_contact/generated_interhuman \
  --gt-json-dir output/mesh_contact/interhuman \
  --output-json output/mesh_contact/generated_vs_gt_summary.json \
  --output-md output/mesh_contact/generated_vs_gt_summary.md
```

## Notes

- This report is intentionally stored under `documentations/` rather than `prepare_mesh_contact/` so the runnable pipeline folder stays code-focused.
- `IMnewtoken/main` should carry the code and this documentation; one-off machine-local artifacts should still stay out of git.
