"""
Newton-VQ: Physics-Informed VQ-VAE for InterMask.

Wraps InterMask's RVQVAE architecture and adds differentiable
physics simulation (Newton SolverFeatherstone) during training.
The decoder output is FK-mapped, retargeted to Newton, PD-tracked,
and physics losses backpropagated into encoder/decoder/codebook.

Phase 1: Per-character physics (single body sim)
Phase 2: Interaction coupling (two bodies in same sim)

Environment: conda activate mimickit  (Python >=3.10, Newton + Warp)
"""
