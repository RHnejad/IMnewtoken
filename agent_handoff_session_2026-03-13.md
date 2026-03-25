# Agent Handoff (2026-03-13)

## Core Summary Artifacts
- Global analysis summary: output/newton_analysis/summary.txt
- Eval alignment audit: output/newton_analysis/eval_alignment_audit.txt

## Main Code Updated This Session
- prepare4/run_full_analysis.py

## What Was Implemented
1. Added interaction-force decomposition output (Newton 3rd law constrained split from root residuals).
2. Added dedicated interaction force plot:
   - interaction_forces.png per clip
3. Improved contact modeling with sole-proxy kinematics:
   - contact from sole height + vertical velocity gating
4. Added foot sole acceleration diagnostics focused on directionality:
   - foot_sole_acceleration.png per clip
   - includes liftoff direction checks and v_z vs a_z diagnostics
5. Added eval-aligned mode to match eval.py temporal policy:
   - GT crop to 300 then effective motion_lens = crop - 1
   - expected generated length = (motion_lens // 4) * 4
6. Added per-run eval alignment audit gate:
   - ready_for_torque_vq_phase in eval_alignment_audit.txt

## Eval Alignment Result (latest full run)
- all_gt_match=True
- all_gen_match=True
- ready_for_torque_vq_phase=1

## Per-Clip Output Structure
Each clip folder in output/newton_analysis/clip_* contains:
- torque_comparison.png
- forces.png
- root_residuals.png
- interaction_forces.png
- foot_sole_acceleration.png
- skeleton_keyframes.png
- data.npz
- summary.txt
- newton_video.mp4 (if video generation enabled)

## Default Clips Processed
- 1129 hit
- 1147 pull
- 1187 kick
- 1006 sword
- 1441 strike
- 3017 punch

## Recommended Next Step (for next agent)
- Start from eval-aligned outputs and implement training-data torque extraction pipeline at InterMask FPS=30 with matching temporal policy, then prepare torque-centric VQ-VAE dataset/loader.
