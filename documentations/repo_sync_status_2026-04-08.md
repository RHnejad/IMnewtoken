# Repo Sync Status

Updated: 2026-04-08

This note records the current repo topology and the required reconciliation order.

---

## 1. Repos and roles

### A. `/media/rh/codes/sim/InterMask`

Role:

- active local research workspace
- based on upstream InterMask repo
- used for fast iteration and experimentation

Git state at time of audit:

- branch: `main`
- head: `5100c555de9839b325d0f3d6904669698c5c87f5`
- remote: `https://github.com/gohar-malik/InterMask.git`
- working tree: **dirty**

This repo is **not** the canonical personal source of truth.

### B. `/media/rh/codes/sim/IMnewtoken`

Role:

- personal canonical git repo
- the repo intended for GitHub push / pull and EPFL cloning

Git state at time of audit:

- branch: `main`
- head: `36cf0fddce8d58bf6727f651baa1f7edf4e27f8b`
- remote: `https://github.com/RHnejad/IMnewtoken`
- working tree: **clean**

This repo should be treated as the main source of truth going forward.

### C. EPFL clone on `haas001`

Role:

- remote working copy cloned from `IMnewtoken`
- contains additional local-only changes

Current exact state:

- path: `/mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken`
- branch: `main`
- head: `36cf0fddce8d58bf6727f651baa1f7edf4e27f8b`
- remote: `https://github.com/RHnejad/IMnewtoken.git`
- branch relation: `HEAD == origin/main`
- working tree: **dirty**

---

## 2. Main problem

There are currently **three evolving copies**:

1. local upstream-derived `InterMask`
2. personal git repo `IMnewtoken`
3. EPFL clone derived from `IMnewtoken`

The main risk is losing track of:

- which changes already reached personal git
- which changes only exist in local `InterMask`
- which changes only exist on EPFL

---

## 3. What is already synced into `IMnewtoken`

Most of the larger structural work is already in `IMnewtoken`, including:

- `prepare_mesh_contact/`
- `prepare6/`
- `prepare_utils/`
- `newton_vqvae/`
- most of `eval_pipeline/`
- `documentations/` baseline docs
- `physics_analysis/`
- `experiments/`

This is why `InterMask` shows many folders as untracked: they are untracked relative to the
original upstream InterMask repo, but they are already tracked in the personal repo.

---

## 4. Current known unsynced delta: `InterMask` -> `IMnewtoken`

At the time of audit, the **recent work present in `InterMask` but missing from `IMnewtoken`** is:

1. `eval_pipeline/force_contact_analysis.py`
2. updated `eval_pipeline/README.md`
3. `documentations/physics_metric_report_2026-04-08.md`

Checked directly:

- `eval_pipeline/force_contact_analysis.py` is missing from `IMnewtoken`
- `documentations/physics_metric_report_2026-04-08.md` is missing from `IMnewtoken`
- `eval_pipeline/README.md` differs between the two repos

The `prepare_mesh_contact/` files themselves are already present in `IMnewtoken`.

---

## 5. Generated-data risk

There is a generated local folder in `InterMask`:

- `data/force_contact/`

Current ignore behavior in `IMnewtoken`:

- `*.npz` is ignored
- `output/` is ignored
- but `data/force_contact/**/*.meta.json` is **not** ignored by the current `.gitignore`

This means a future raw sync from `InterMask` into `IMnewtoken` could accidentally stage:

- `data/force_contact/.../*.meta.json`

Required ignore fix in `IMnewtoken`:

- add `data/force_contact/`

This should be done before the next sync that copies local working-tree files over.

---

## 5.1 EPFL audit result

The EPFL clone is **not ahead of GitHub**.

Important result:

- local `IMnewtoken` head = `36cf0fd`
- EPFL `IMnewtoken` head = `36cf0fd`
- `origin/main` on EPFL also points at `36cf0fd`

So there are currently **no committed EPFL-only changes**.

What exists on EPFL instead is:

1. a very large set of unstaged modifications affecting tracked files across the repo
2. a smaller set of **untracked** files under `prepare_mesh_contact/`

Observed untracked EPFL-only files:

- `prepare_mesh_contact/INTERHUMAN_GENERATED_VS_GT_IMPLEMENTATION_REPORT.md`
- `prepare_mesh_contact/check_progress.sh`
- `prepare_mesh_contact/diagnose_clip.sh`
- `prepare_mesh_contact/inspect_pkl.py`
- `prepare_mesh_contact/interhuman_generated_vs_gt_utils.py`
- `prepare_mesh_contact/list_interx_clips.py`
- `prepare_mesh_contact/monitor.sh`
- `prepare_mesh_contact/render_contact_headless.py`
- `prepare_mesh_contact/render_interhuman_generated_vs_gt.py`
- `prepare_mesh_contact/retry_failed.sh`
- `prepare_mesh_contact/run_all_batch.sh`
- `prepare_mesh_contact/run_interhuman_batch.sh`
- `prepare_mesh_contact/run_interhuman_generated_vs_gt.sh`
- `prepare_mesh_contact/run_interx_batch.sh`
- `prepare_mesh_contact/smoke_test.sh`
- `prepare_mesh_contact/summarize_interhuman_generated_vs_gt.py`

Interpretation:

- these untracked files are likely the real EPFL-only work to preserve
- the huge number of tracked `M` entries is suspicious and very likely includes
  filesystem-induced noise such as mode changes

Because the tracked-file modifications are so broad, they should **not** be committed blindly.

---

## 6. Practical source-of-truth rule

Going forward:

- `InterMask` = integration sandbox / experimentation tree
- `IMnewtoken` = canonical git repo
- EPFL clone = temporary execution node, never long-term source of truth

Meaning:

- do not treat `InterMask` and EPFL as parallel permanent repositories
- every durable change should end up in `IMnewtoken`
- EPFL changes should arrive back as a branch or patch, not stay only on the cluster

---

## 7. Required reconciliation order

### Step 1. Fold current local-only `InterMask` delta into `IMnewtoken`

Target changes:

- `eval_pipeline/force_contact_analysis.py`
- `eval_pipeline/README.md`
- `documentations/physics_metric_report_2026-04-08.md`
- `.gitignore` fix for `data/force_contact/`

Do this on a dedicated branch in `IMnewtoken`, not directly on `main`.

Suggested branch:

- `sync/intermask-2026-04-08`

### Step 2. Audit EPFL clone directly on the cluster

Required commands on EPFL:

```bash
git status --short --branch
git remote -v
git rev-parse HEAD
git branch --show-current
git log --oneline -1
```

### Step 3. Separate real EPFL work from repo-wide noise

Before committing anything broad on EPFL, run:

```bash
git diff --summary | sed -n '1,120p'
git diff --numstat | sed -n '1,80p'
git diff -- prepare_mesh_contact/README.md | sed -n '1,120p'
git config --show-origin --get core.filemode || echo "core.filemode unset"
git config --show-origin --get core.autocrlf || echo "core.autocrlf unset"
```

If the diff is mostly `mode change`, treat it as repo-noise, not authored work.

### Step 4. Freeze EPFL changes on their own branch

Suggested sequence on EPFL:

```bash
git switch -c sync/epfl-2026-04-08
git add prepare_mesh_contact/INTERHUMAN_GENERATED_VS_GT_IMPLEMENTATION_REPORT.md \
        prepare_mesh_contact/check_progress.sh \
        prepare_mesh_contact/diagnose_clip.sh \
        prepare_mesh_contact/inspect_pkl.py \
        prepare_mesh_contact/interhuman_generated_vs_gt_utils.py \
        prepare_mesh_contact/list_interx_clips.py \
        prepare_mesh_contact/monitor.sh \
        prepare_mesh_contact/render_contact_headless.py \
        prepare_mesh_contact/render_interhuman_generated_vs_gt.py \
        prepare_mesh_contact/retry_failed.sh \
        prepare_mesh_contact/run_all_batch.sh \
        prepare_mesh_contact/run_interhuman_batch.sh \
        prepare_mesh_contact/run_interhuman_generated_vs_gt.sh \
        prepare_mesh_contact/run_interx_batch.sh \
        prepare_mesh_contact/smoke_test.sh \
        prepare_mesh_contact/summarize_interhuman_generated_vs_gt.py
git commit -m "EPFL mesh-contact batch helpers and reports"
git push -u origin sync/epfl-2026-04-08
```

Only commit tracked-file modifications after confirming they are real content changes.

### Step 5. Merge both branches inside `IMnewtoken`

Merge order:

1. local `InterMask` delta -> `IMnewtoken` branch
2. EPFL branch -> `IMnewtoken`
3. resolve conflicts once in the canonical repo
4. push merged result to `origin/main`

### Step 6. Re-clone or hard refresh EPFL only after merge

After `IMnewtoken/main` is updated:

- pull or re-clone on EPFL
- stop carrying long-lived unsynced edits there

---

## 8. Current recommendation

The cleanest path is:

1. update `IMnewtoken` first from the current `InterMask` delta
2. then reconcile EPFL into `IMnewtoken`
3. only after that continue new development

Do **not** continue making unrelated changes in all three places before this is done.

---

## 9. Immediate actionable checklist

### Local machine

- [ ] create sync branch in `IMnewtoken`
- [ ] copy in current missing files from `InterMask`
- [ ] add ignore rule for `data/force_contact/`
- [ ] commit

### EPFL

- [ ] audit git state
- [ ] create `sync/epfl-2026-04-08` branch
- [ ] commit local changes there
- [ ] push branch

### Final

- [ ] merge both sync branches in `IMnewtoken`
- [ ] push merged `main`
- [ ] refresh EPFL clone from updated `main`
