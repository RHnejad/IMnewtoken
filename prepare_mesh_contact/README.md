# prepare_mesh_contact

SMPL-X mesh reconstruction + mesh-contact analysis for 2-person clips
(InterHuman / InterX).

## What this does

For each frame in a clip, it reconstructs person-1/person-2 SMPL-X meshes and
classifies the interaction state as:

- `penetrating`
- `touching`
- `barely_touching`
- `not_touching`

It also reports:

- minimum mesh distance proxy (`min_distance_m`)
- closest points on person1/person2
- contact / barely-contact / penetrating vertex counts per person
- explicit penetration source split:
  - `other_person_only` (inter-person penetration)
  - `self_only` (self-penetration heuristic)
  - `both`
  - `none`

Optional outputs:

- detailed per-frame vertex-index arrays (`--output-details`)
- colored `.ply` mesh export for one frame (`--export-ply-frame`)

## Method summary

- Mesh reconstruction: local `BodyModel` (`data/body_model/body_model.py`) with
  SMPL-X params from dataset clips.
- Distance/contact: nearest-vertex queries via `scipy.spatial.cKDTree`.
- Penetration check: odd-even ray casting (Moller-Trumbore intersections) on a
  sampled set of vertices (near-distance + uniform coverage).
- Self-penetration check (optional heuristic): nearest non-neighbor surface
  proximity (topology-aware), enabled with `--self-penetration-mode heuristic`.

Notes:

- This fallback avoids extra geometry dependencies (`trimesh`, `rtree`) so it
  runs in the current environment.
- For publication-grade penetration depth, you may replace the penetration
  backend with signed-distance tools (SDF / robust point-mesh distance).

## Usage

Run synthetic sanity checks:

```bash
python prepare_mesh_contact/mesh_contact_pipeline.py --self-test
```

Newton GUI contact-point overlay (full sequence):

```bash
python prepare_mesh_contact/visualize_contact_newton.py \
  --dataset interhuman \
  --clip 7605 \
  --data-root data/InterHuman \
  --body-model-path data/body_model/smplx/SMPLX_NEUTRAL.npz
```

```bash
python prepare_mesh_contact/visualize_contact_newton.py \
  --dataset interx \
  --clip G039T007A025R000 \
  --data-root data/Inter-X_Dataset \
  --body-model-path data/body_model/smplx/SMPLX_NEUTRAL.npz
```

The Newton viewer now overlays:

- contact points (`penetrating`, `touching`, `barely_touching`)
- closest-point segment
- full 22-joint skeleton (joints + bones) for both persons
- clip annotation text in GUI side panel and terminal

Run both demos sequentially:

```bash
prepare_mesh_contact/run_contact_visual_examples.sh
```

The launcher defaults to contact-rich clips:

- InterHuman: `7605`
- InterX: `G039T007A025R000`

Override them if needed:

```bash
INTERHUMAN_CLIP=1000 INTERX_CLIP=G001T000A000R000 prepare_mesh_contact/run_contact_visual_examples.sh
```

If OpenGL cannot connect to a display, use a non-GL backend
(requires `viser` to be installed):

```bash
python prepare_mesh_contact/visualize_contact_newton.py \
  --viewer viser \
  --dataset interhuman \
  --clip 1000 \
  --data-root data/InterHuman \
  --body-model-path data/body_model/smplx/SMPLX_NEUTRAL.npz
```

InterHuman example:

```bash
python prepare_mesh_contact/mesh_contact_pipeline.py \
  --dataset interhuman \
  --clip 1000 \
  --data-root data/InterHuman \
  --body-model-path data/body_model/smplx/SMPLX_NEUTRAL.npz \
  --output-json output/mesh_contact/interhuman_1000_mesh_contact.json \
  --output-details output/mesh_contact/interhuman_1000_mesh_contact_details.pkl \
  --export-ply-frame 0
```

InterX example:

```bash
python prepare_mesh_contact/mesh_contact_pipeline.py \
  --dataset interx \
  --clip G001T000A000R000 \
  --data-root data/Inter-X_Dataset \
  --body-model-path data/body_model/smplx/SMPLX_NEUTRAL.npz \
  --output-json output/mesh_contact/interx_G001T000A000R000_mesh_contact.json
```

InterX loader supports both layouts:

- `processed/motions/inter-x.h5` (single-file layout)
- `processed/motions/train.h5`, `val.h5`, `test.h5` (split layout)

## Important arguments

- `--touching-threshold-m` (default `0.005`)
- `--barely-threshold-m` (default `0.020`)
- `--penetration-probe-distance-m` (default `0.010`)
- `--penetration-min-depth-m` (default `0.002`)
- `--self-penetration-mode` (`off` by default, `heuristic` optional)
- `--self-penetration-threshold-m` (default `0.004`)
- `--self-penetration-k` (default `12`)
- `--self-penetration-normal-dot-max` (default `-0.2`)
- `--frame-start`, `--frame-end`, `--frame-step`

Tune thresholds based on your mesh resolution and downstream tolerance.
