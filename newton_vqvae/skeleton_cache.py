"""
skeleton_cache.py — Cache per-subject MJCF XMLs and Newton models.

Reuses prepare2/gen_smpl_xml.py for XML generation and caches both
the XML files and finalized Newton Models keyed by a betas hash.
This avoids re-building models every batch.
"""
from __future__ import annotations

import os
import sys
import hashlib
from typing import Dict, Tuple, Optional, List

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import warp as wp
import newton

from prepare2.gen_smpl_xml import generate_smpl_xml, get_smplx_body_offset, R_ROT
from prepare2.gen_smpl_with_sphere_feet_xml import get_or_create_sphere_feet_xml
from prepare2.retarget import SMPL_TO_NEWTON, N_SMPL_JOINTS, N_JOINT_Q
from newton_vqvae.config import BODIES_PER_PERSON, DOFS_PER_PERSON, COORDS_PER_PERSON


# ═══════════════════════════════════════════════════════════════
# Betas hashing
# ═══════════════════════════════════════════════════════════════

def betas_hash(betas: np.ndarray) -> str:
    """Stable hash of betas for caching (12-char hex)."""
    return hashlib.md5(betas.astype(np.float64).tobytes()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════
# XML cache
# ═══════════════════════════════════════════════════════════════

_XML_CACHE_DIR = os.path.join(_PROJECT_ROOT, "newton_vqvae", "xml_cache")


def get_or_create_xml(
    betas: np.ndarray,
    cache_dir: Optional[str] = None,
    use_sphere_feet: bool = True,
) -> str:
    """Get/generate a per-subject MJCF XML, cached on disk.

    By default uses sphere-cluster feet instead of box feet to prevent
    contact force explosions at narrow cube-ground edges.
    """
    if cache_dir is None:
        cache_dir = _XML_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    if use_sphere_feet:
        # Sphere-cluster feet: 4 spheres per foot (heel, 2 ball, toe)
        # Much more forgiving contact than box edges
        return get_or_create_sphere_feet_xml(
            np.asarray(betas, dtype=np.float64),
            cache_dir=cache_dir,
        )
    else:
        h = betas_hash(betas)
        xml_path = os.path.join(cache_dir, f"smpl_{h}.xml")
        if not os.path.exists(xml_path):
            generate_smpl_xml(betas, output_path=xml_path)
        return xml_path


def _apply_soft_contact_to_model(model) -> None:
    """Apply contact softening to a Newton model to prevent force explosion.

    Three measures against foot-ground contact instability:
    1. Increased contact margin (0.002 → 0.005) — detect contacts earlier
    2. Contact softening via shape_geo ke/kd dampening
    3. Contact damping ratio increase for energy dissipation

    These prevent the "stiff contact bounce" failure mode where
    a cube foot edge hits hard ground and generates extreme forces.
    """
    # The Newton Featherstone solver uses penalty contacts.
    # We soften them via the model's contact parameters.
    n_shapes = model.shape_count

    # Increase contact margin — contacts detected earlier = gentler onset
    if hasattr(model, 'shape_margin'):
        margins = np.full(n_shapes, 0.005, dtype=np.float32)
        model.shape_margin = wp.array(margins, dtype=wp.float32, device=model.device)

    # Soften contact stiffness (ke) and damping (kd) if available
    if hasattr(model, 'shape_material_ke'):
        ke = np.full(n_shapes, 1000.0, dtype=np.float32)   # softer than default
        kd = np.full(n_shapes, 100.0, dtype=np.float32)    # high damping ratio
        model.shape_material_ke = wp.array(ke, dtype=wp.float32, device=model.device)
        model.shape_material_kd = wp.array(kd, dtype=wp.float32, device=model.device)

    # Increase friction for stable foot plants
    if hasattr(model, 'shape_material_mu'):
        mu = np.full(n_shapes, 1.0, dtype=np.float32)
        model.shape_material_mu = wp.array(mu, dtype=wp.float32, device=model.device)


# ═══════════════════════════════════════════════════════════════
# Newton model cache  (thread-local per device)
# ═══════════════════════════════════════════════════════════════

class SkeletonCache:
    """
    Caches Newton Models and per-subject metadata.

    One cache per training run — call `get_model` with betas and the
    model is built once and reused for all subsequent batches.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._models: Dict[str, newton.Model] = {}
        self._offsets: Dict[str, np.ndarray] = {}
        self._xml_paths: Dict[str, str] = {}

    def get_model(
        self,
        betas: np.ndarray,
        with_ground: bool = True,
        requires_grad: bool = True,
        use_sphere_feet: bool = True,
        soft_contact: bool = True,
    ) -> newton.Model:
        """
        Get or build a Newton model for one person.

        Args:
            betas: (10,) SMPL-X betas
            with_ground: add ground plane (needed for Featherstone contacts)
            requires_grad: enable gradient tracking for wp.Tape
            use_sphere_feet: use sphere-cluster feet (prevents edge-contact explosion)
            soft_contact: apply contact softening parameters
        """
        h = betas_hash(betas)
        if h not in self._models:
            xml_path = get_or_create_xml(betas, use_sphere_feet=use_sphere_feet)
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
            builder.add_mjcf(
                xml_path,
                enable_self_collisions=False,
            )
            if with_ground:
                builder.add_ground_plane()
            model = builder.finalize(device=self.device, requires_grad=requires_grad)

            # Apply contact safety measures
            if soft_contact:
                _apply_soft_contact_to_model(model)

            self._models[h] = model
            self._xml_paths[h] = xml_path
        return self._models[h]

    def get_pair_model(
        self,
        betas1: np.ndarray,
        betas2: np.ndarray,
        requires_grad: bool = True,
    ) -> newton.Model:
        """
        Build a two-person Newton model for Phase 2 interaction.

        Both characters are in the same simulation — they can
        collide/contact through Featherstone penalty contacts.
        """
        h1 = betas_hash(betas1)
        h2 = betas_hash(betas2)
        key = f"pair_{h1}_{h2}"
        if key not in self._models:
            xml1 = get_or_create_xml(betas1)
            xml2 = get_or_create_xml(betas2)
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
            builder.add_mjcf(xml1, enable_self_collisions=False)
            builder.add_mjcf(xml2, enable_self_collisions=False)
            builder.add_ground_plane()
            model = builder.finalize(device=self.device, requires_grad=requires_grad)
            self._models[key] = model
        return self._models[key]

    def get_body_offset(self, betas: np.ndarray) -> np.ndarray:
        """Get SMPL-X body offset (rest-pose pelvis position)."""
        h = betas_hash(betas)
        if h not in self._offsets:
            self._offsets[h] = get_smplx_body_offset(betas)
        return self._offsets[h]

    def clear(self):
        """Clear all cached models (frees GPU memory)."""
        self._models.clear()
        self._offsets.clear()
        self._xml_paths.clear()
