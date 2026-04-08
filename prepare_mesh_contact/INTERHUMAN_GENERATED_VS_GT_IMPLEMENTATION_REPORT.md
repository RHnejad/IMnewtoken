# InterHuman Generated-vs-GT Contact Analysis Implementation Report

## Why this change was needed

The repo already had a GT-oriented SMPL-X mesh contact pipeline for InterHuman and InterX, but it was missing the pieces needed to audit InterMask-generated InterHuman motions against GT in a controlled way:

- generated InterHuman clips live in a different raw layout than GT (`<root>/<clip>.pkl` instead of `<root>/motions/<clip>.pkl`)
- generated InterHuman clips should reuse GT betas for fair mesh reconstruction
- we needed split-aware reporting for GT train/test and generated test coverage
- we needed a deterministic GT-vs-generated comparison renderer that uses the same frame index inside the shared valid prefix

The change also uncovered an important dataset edge case: official GT split clips `3945` and `4106` exist on disk but contain zero frames for both people. Before this work, those clips could never produce JSON outputs because mesh reconstruction failed on zero-length sequences.

## Verified source paths

- GT InterHuman root used by the pipeline:
  - `data/InterHuman`
- Verified local InterMask generated InterHuman root:
  - `/mnt/vita/scratch/vita-staff/users/rh/codes/2026/default_intermask/data/generated/interhuman`
- GT mesh-contact JSON root:
  - `output/mesh_contact/interhuman`
- Generated mesh-contact JSON root introduced by this work:
  - `output/mesh_contact/generated_interhuman`
- GT-vs-generated comparison PNG root introduced by this work:
  - `output/renders/interhuman_gt_vs_generated_contact`
- Summary/report output root introduced by this work:
  - `output/reports/interhuman_generated_vs_gt`

## Design decisions

### 1. Reuse the existing contact core instead of creating a parallel analyzer

The existing `mesh_contact_pipeline.py` already contains the correct mesh reconstruction and inter-person penetration logic, so the safest path was to extend it with generated-data support instead of forking the analysis logic.

### 2. Apply GT betas as an explicit override, not as an implicit side effect

Generated InterHuman analysis now uses a dedicated CLI flag:

- `--betas-from-interhuman-root PATH`

This makes the provenance explicit in the output JSON and keeps the default GT path unchanged.

### 3. Treat zero-frame GT clips as explicit dataset artifacts, not as silent failures

The official GT split members `3945` and `4106` both have:

- `trans: (0, 3)`
- `root_orient: (0, 3)`
- `pose_body: (0, 63)`

for both people. The pipeline now writes valid zero-frame JSON summaries for this case so:

- split coverage can reach 6022 train / 1177 test
- downstream summary scripts do not need special-case missing-file logic for these two clips
- the output makes the zero-frame condition visible instead of hiding it behind a failed batch job

### 4. Keep summary/reporting strict about configuration drift

The summary path fails if:

- thresholds differ across the JSONs being summarized
- `self_penetration_mode != off`

This is important because otherwise GT and generated aggregates could be merged even if they came from incompatible runs.

### 5. Make frame selection deterministic and shared between report and renderer

For GT-vs-generated PNGs, the selected frame is:

1. inside the shared prefix `T_shared = min(T_gt, T_gen)`
2. generated frame must have `has_inter_person_penetration=True`
3. choose highest `inter_person_penetration_depth_est_m`
4. break ties by smaller `min_distance_m`
5. then earliest frame

This logic is shared through `interhuman_generated_vs_gt_utils.py` so the CSV and renderer cannot drift.

## Files changed

### Modified

- `prepare_mesh_contact/mesh_contact_pipeline.py`
  - added `--betas-from-interhuman-root`
  - added InterHuman provenance recording
  - added zero-frame clip support
- `prepare_mesh_contact/run_interhuman_batch.sh`
  - added `--data-root`
  - added `--output-dir`
  - added `--betas-from-interhuman-root`
  - added dual clip discovery for GT and generated InterHuman layouts
- `prepare_mesh_contact/README.md`
  - documented generated-vs-GT orchestration, summary, comparison rendering, and zero-frame handling

### Added

- `prepare_mesh_contact/interhuman_generated_vs_gt_utils.py`
  - shared paths, split loading, threshold checks, frame selection, summary helpers
- `prepare_mesh_contact/summarize_interhuman_generated_vs_gt.py`
  - writes `summary_by_split.csv`, `coverage_gaps.csv`, `common_clip_metrics.csv`, `summary.md`
- `prepare_mesh_contact/render_interhuman_generated_vs_gt.py`
  - renders one GT-vs-generated PNG per selected common test clip
- `prepare_mesh_contact/run_interhuman_generated_vs_gt.sh`
  - orchestrates GT top-up, generated extraction, PNG generation, and reporting
- `prepare_mesh_contact/INTERHUMAN_GENERATED_VS_GT_IMPLEMENTATION_REPORT.md`
  - this report

## Output JSON changes

InterHuman contact JSONs can now include:

```json
"provenance": {
  "source_variant": "gt" | "intermask_generated",
  "betas_override": {
    "enabled": true | false,
    "interhuman_root": "...",
    "matched_people": ["person1", "person2"]
  }
}
```

This was added so the summary layer can distinguish GT vs generated runs and so engineers can verify whether GT beta override was actually applied.

## Current outputs produced in this implementation pass

### GT official split top-up

- `output/mesh_contact/interhuman/3945.json`
- `output/mesh_contact/interhuman/4106.json`

Both are explicit zero-frame JSON summaries.

### Generated InterHuman verification outputs

Full generated contact JSONs produced in this pass:

- `output/mesh_contact/generated_interhuman/1004.json`
- `output/mesh_contact/generated_interhuman/26.json`

Important note:

- `1004` is a non-contact generated sample and exercises the no-penetration path
- `26` is a real penetrating generated sample and exercises the compare-render path

### GT-vs-generated comparison PNG produced in this pass

- `output/renders/interhuman_gt_vs_generated_contact/interhuman_26_frame_00121.png`

For clip `26`, the generated run had 23 inter-person penetration frames and the selection rule chose frame `121`.

### Summary/report outputs produced in this pass

- `output/reports/interhuman_generated_vs_gt/summary_by_split.csv`
- `output/reports/interhuman_generated_vs_gt/coverage_gaps.csv`
- `output/reports/interhuman_generated_vs_gt/common_clip_metrics.csv`
- `output/reports/interhuman_generated_vs_gt/summary.md`

## Current numeric results

These values come from `output/reports/interhuman_generated_vs_gt/summary_by_split.csv` as produced in this pass.

### GT train

- expected clips: 6022
- JSON-covered clips: 6022
- total frames: 1,704,443
- inter-person penetration frames: 482,542
- inter-person penetration frame fraction: 0.2831083233642897
- clips with any inter-person penetration: 3,632

### GT test

- expected clips: 1177
- JSON-covered clips: 1177
- total frames: 354,333
- inter-person penetration frames: 95,881
- inter-person penetration frame fraction: 0.27059573903644313
- clips with any inter-person penetration: 639

### Generated test coverage at the moment of this report

The generated side is only partially populated in this pass because I only ran the new pipeline on selected verification clips, not the full 1098-clip InterMask export.

- expected test clips: 1177
- generated JSON-covered clips currently present: 2
- common GT/generated test clips currently summarized: 2

This means the GT rows above are complete, but the generated rows in the current CSV/Markdown report are only a partial verification snapshot, not the final full-generated benchmark.

## Verification performed

### Static verification

- `python -m py_compile` passed for:
  - `prepare_mesh_contact/mesh_contact_pipeline.py`
  - `prepare_mesh_contact/interhuman_generated_vs_gt_utils.py`
  - `prepare_mesh_contact/summarize_interhuman_generated_vs_gt.py`
  - `prepare_mesh_contact/render_interhuman_generated_vs_gt.py`
- `bash -n` passed for:
  - `prepare_mesh_contact/run_interhuman_batch.sh`
  - `prepare_mesh_contact/run_interhuman_generated_vs_gt.sh`
- `--help` passed for:
  - `prepare_mesh_contact/summarize_interhuman_generated_vs_gt.py`
  - `prepare_mesh_contact/render_interhuman_generated_vs_gt.py`

### Functional verification

- Generated smoke run:
  - clip `1004`
  - verified `source_variant=intermask_generated`
  - verified GT beta override provenance was written
- Zero-frame GT handling:
  - verified `3945.json` and `4106.json` now exist
  - verified both have `num_frames=0` and empty `frames`
- Generated penetration example:
  - clip `26`
  - generated JSON written successfully
  - 23 generated inter-person penetration frames detected
  - selected compare frame = `121`
- Compare renderer:
  - rendered `output/renders/interhuman_gt_vs_generated_contact/interhuman_26_frame_00121.png`
- Summary script:
  - completed successfully
  - wrote all four expected report artifacts

## What is not complete yet

The following was intentionally not completed in this pass:

- the full 1098-clip generated InterHuman batch has not been run yet
- therefore the current generated summary rows are partial
- the compare PNG directory currently contains only the clips that have actually been processed on the generated side
- InterX summary/rendering was intentionally deferred as requested

## How to finish the generated InterHuman run

Run:

```bash
cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken
bash prepare_mesh_contact/run_interhuman_generated_vs_gt.sh \
  --workers 4 \
  --device cuda \
  --batch-size 64
```

Expected full-generated target after that run:

- generated test coverage should reach 1098 clips
- missing generated test clips should settle at 79
- `common_clip_metrics.csv` should expand from the current 2 rows to the full common GT/generated set

## InterX note

Per request, InterX reporting/rendering was deferred. The exact GT completion command is:

```bash
cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken
bash prepare_mesh_contact/run_interx_batch.sh --workers 4 --device cuda --batch-size 64
```

## Reviewer checklist

- Verify the new provenance block is present in generated InterHuman JSONs.
- Verify GT train/test split coverage is now complete because `3945` and `4106` are represented as zero-frame JSONs.
- Verify no summary run mixes threshold configurations or non-`off` self-penetration mode.
- Verify `common_clip_metrics.csv` and comparison PNG filenames agree on selected frame indices.
- Verify generated outputs are not partial before interpreting generated summary rows as final benchmark numbers.
- Verify zero-frame GT clips are acceptable for downstream consumers; if not, downstream readers may need an explicit `num_frames == 0` check.
