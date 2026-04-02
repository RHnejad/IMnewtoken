# NVIDIA Embodied AI Ecosystem — Relevance to Our Work
*Analysed: 2026-03-29*
*Source: LinkedIn post by Umar Iqbal (NVIDIA), GTC 2026*

---

## Summary Table

| Project | Relevance | Key Action |
|---------|-----------|------------|
| **ProtoMotions** | Very High | MimicKit = production-grade version of our `prepare5/` tracker |
| **Kimodo** | High | Direct competitor to InterMask; add as comparison target for physics metric paper |
| **SONIC** | High | Proves RL > differentiable for tracking; cite as architectural justification |
| **SOMA** | Medium-High | Could replace our custom SMPL-X retargeting pipeline |
| **SOMA Retargeter** | Medium | Newton+Warp-based; alternative to `prepare2/` retargeting |
| **GEM** | Low-Medium | Single-person; interesting constrained-generation architecture |
| **Bones-SEED** | Low | Single-person dataset; could test our metric at scale |

---

## 1. SOMA — Universal Skeleton Representation
**Paper/Repo:** https://github.com/NVlabs/SOMA-X

**What it does:**
Unifies SMPL, SMPL-X, MHR, Anny into a single canonical body topology and rig.
End-to-end differentiable, GPU-accelerated via NVIDIA Warp. Supports analytical
SMPL-to-SOMA conversion at 1279 FPS on RTX 5000 Ada.

**Relevance to our work: MEDIUM-HIGH**
- We do SMPL-X → Newton retargeting manually in `prepare2/`
- SOMA already solves this in a standardised, production-tested way
- Could replace our custom per-subject XML skeleton generation
- Free interoperability with Kimodo, ProtoMotions, SONIC

**Caveat:** Single-body model — does not handle two-person interaction directly.

---

## 2. Kimodo — Text-to-Motion Generation
**Paper:** https://arxiv.org/abs/2603.15546
**Code:** https://github.com/nv-tlabs/kimodo
**Project:** https://research.nvidia.com/labs/sil/projects/kimodo/

**What it does:**
Kinematic motion diffusion model trained on **700 hours** of studio mocap (Bones Rigplay dataset).
Two-stage denoiser: root motion first, then body. Supports text, keyframe, sparse joint,
and 2D path conditioning. Outputs SMPL-X, SOMA, and Unitree G1 formats.
~25x more training data than MDM or MotionDiffuse.

**Relevance to our work: HIGH**
- Direct successor/competitor to InterMask and InterGen
- **Single-person only** — our work uniquely targets two-person interaction
- Our physics metric (ΔF_sky) could evaluate Kimodo outputs:
  - Does 25x more data → more physically plausible motions?
  - This would be a strong comparison in a paper
- Could test: Kimodo (single-person, 700h data) vs InterMask (two-person, 6800 clips) on physics plausibility

---

## 3. ProtoMotions — GPU-Accelerated Humanoid Learning Framework
**Repo:** https://github.com/NVlabs/ProtoMotions

**What it does:**
Modular RL framework for physics-based humanoid control. Supports Newton, MuJoCo,
IsaacGym, IsaacLab. Includes **MimicKit** — a motion imitation sub-framework.
Trains on entire AMASS (40+ hours) in ~12 hours on 4x A100s.
Uses PyRoki for retargeting across robot morphologies.
Demonstrates sim-to-real on Unitree G1 without fine-tuning.

**Relevance to our work: VERY HIGH**
- MimicKit is a production-grade version of what we built in `prepare5/`
- Uses Newton (same physics engine) for simulation
- Uses RL (PPO), not differentiable optimization, for tracking — even though Newton IS differentiable
- This is the strongest external evidence that **RL > differentiable** for motion tracking
- Could adopt ProtoMotions as our tracker backend instead of custom `prepare5/`

**Key implication:** NVIDIA built Newton to be differentiable, but chose RL for the tracking
task. Contacts + long rollouts make autodiff gradients unreliable; RL handles both via sampling.

---

## 4. SONIC — Humanoid Whole-Body Control Foundation Model
**Paper:** https://arxiv.org/abs/2511.07820
**Project:** https://nvlabs.github.io/SONIC/

**What it does:**
Scales motion tracking along three axes: network (1.2M → 42M params), data (700 hours,
100M+ frames), compute (9k GPU hours). Universal kinematic planner bridges tracking to
downstream tasks (VR teleoperation, VLA models). Achieves 100% success on 50 diverse
real-world motion trajectories on Unitree G1. Built on ProtoMotions + RL (PPO).

**Relevance to our work: HIGH**
- Directly validates that tracking error (sim vs ref MPJPE) scales with data + model size
- Our proposed metric (tracker MPJPE gap: GT vs Generated) is conceptually aligned with SONIC's evaluation
- Uses RL, not differentiable physics — further supports dropping `optimize_tracking.py`
- SONIC is single-body + robot-focused; we are two-person + human motion evaluation
- Cite as: "concurrent work validates physics tracking as a scalable plausibility measure"

---

## 5. SOMA Retargeter — Newton + Warp Based
**Repo:** https://github.com/NVIDIA/soma-retargeter

**What it does:**
BVH → humanoid robot motion retargeting, built on Newton and Warp (same stack as us).
Used to retarget Bones-SEED to Unitree G1. Used for Kimodo G1 training data.

**Relevance to our work: MEDIUM**
- We already have working retargeting (0.00 cm MPJPE on InterHuman clips)
- Could simplify our pipeline; worth checking if it handles SMPL-X directly
- Uses same Newton+Warp stack — code should be compatible

---

## 6. Bones-SEED — Large-Scale Motion Dataset
**Dataset:** https://huggingface.co/datasets/bones-studio/seed

**What it does:**
142,220 annotated human motion animations in SOMA format + Unitree G1.
Up to 6 natural-language descriptions per motion. Built from the same data that trained SONIC.
Covers locomotion, everyday activities, object interactions.

**Relevance to our work: LOW**
- All single-person motions — our work requires two-person interaction
- Could use as a large-scale test set to validate our physics metric on solo motions
- If we extend our metric to single-person, this is the obvious dataset to use

---

## 7. GEM — Generalist Motion Estimation + Generation
**Project:** https://research.nvidia.com/labs/dair/gem/
**Repo:** https://github.com/NVlabs/GEM-X

**What it does:**
Unifies motion estimation (from video) and generation (text/music) in a single diffusion model.
Reformulates motion estimation as constrained generation. ICCV 2025 Highlight.
Handles dynamic cameras, recovers global trajectories. Uses SOMA keypoints.

**Relevance to our work: LOW-MEDIUM**
- Single-person only
- Constrained-generation architecture is conceptually interesting
- If we ever want to generate physics-constrained interaction motions (not just evaluate them),
  GEM's approach could inspire adding a physics constraint to the generation process

---

## Key Takeaways for Our Project

### 1. RL vs Differentiable (answers the open question)
NVIDIA built Newton as a differentiable physics engine but chose **RL (PPO via ProtoMotions)**
for motion tracking in both ProtoMotions and SONIC. This is the strongest possible evidence that
differentiable optimization (`prepare5/optimize_tracking.py`) is the wrong tool for motion
tracking. Drop it or treat it as a future experiment. Focus on kinematic PD tracker (`phc_tracker.py`)
or consider adopting MimicKit from ProtoMotions.

### 2. Our Niche is Clear
Every project above is **single-person**. There is no physics-based evaluation metric for
**two-person interaction motion generation** anywhere in this ecosystem. Our work fills a
genuine gap.

### 3. Paper Comparison Targets
If we write a paper, compare our physics metric across:
- InterMask (our baseline, two-person)
- InterGen (another two-person generator)
- Kimodo (single-person, much more data — physics plausibility baseline)

### 4. Infrastructure to Consider Adopting
- SOMA: replace custom XML retargeting in `prepare2/`
- ProtoMotions/MimicKit: replace custom `prepare5/` tracking
- Both run on Newton + Warp — code compatibility is high
