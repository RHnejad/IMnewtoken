"""
Compute per-frame skyhook metrics (root virtual force residuals) and MPJPE.

This script is independent of the optimization loop. It can run from:
  1) Retargeted joint trajectories (*.npy), or
  2) Raw dataset clips (SMPL-X params), converting to joint_q on the fly.

Per clip/person outputs:
  - *_skyhook_metrics.npz  (per-frame arrays + key scalars)
  - *_skyhook_metrics.json (summary metadata)

Dataset outputs:
  - summary.csv
  - summary.json
  - config.json

Skyhook metric (per frame):
  root_force_l2[t] = ||F_root,t||_2, where F_root,t = torques[t, 0:3]

Sequence residual:
  Residual = mean_t root_force_l2[t]
"""
import os
import sys
import csv
import json
import time
import hashlib
import argparse
import warnings
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare2.retarget import (  # noqa: E402
    N_SMPL_JOINTS,
    get_or_create_xml,
    extract_positions_from_fk,
    smplx_to_joint_q,
    load_interhuman_clip,
    load_interx_clip,
    list_interhuman_clips,
    list_interx_clips,
)
from prepare2.compute_torques import inverse_dynamics  # noqa: E402
from prepare2.pd_utils import DOF_NAMES  # noqa: E402


@dataclass
class ClipPersonData:
    clip_id: str
    person_idx: int
    joint_q: np.ndarray
    betas: np.ndarray
    source: str  # "retargeted" | "dataset"


def _resolve_existing_dir(path: Optional[str], default_rel: str) -> str:
    """Resolve an existing input directory (project-relative fallback)."""
    candidate = path or default_rel
    if os.path.isabs(candidate):
        out = candidate
    else:
        out = os.path.join(PROJECT_ROOT, candidate)
        if not os.path.isdir(out) and os.path.isdir(candidate):
            out = candidate
    if not os.path.isdir(out):
        raise FileNotFoundError(f"Directory not found: {out}")
    return out


def _resolve_output_dir(path: Optional[str], default_rel: str) -> str:
    """Resolve an output directory path (project-relative by default)."""
    candidate = path or default_rel
    if os.path.isabs(candidate):
        return candidate
    return os.path.join(PROJECT_ROOT, candidate)


def list_retargeted_clips(data_dir: str) -> List[str]:
    """List clips that have joint_q files for both persons."""
    clips_p0 = set()
    clips_p1 = set()
    for fname in os.listdir(data_dir):
        if fname.endswith("_person0_joint_q.npy"):
            clips_p0.add(fname.replace("_person0_joint_q.npy", ""))
        elif fname.endswith("_person1_joint_q.npy"):
            clips_p1.add(fname.replace("_person1_joint_q.npy", ""))
    return sorted(clips_p0 & clips_p1)


def load_raw_clip(dataset: str, raw_data_dir: str, clip_id: str):
    """Load raw dataset clip data for both persons (or None if missing)."""
    if dataset == "interhuman":
        return load_interhuman_clip(raw_data_dir, clip_id)
    return load_interx_clip(raw_data_dir, clip_id)


def list_raw_clips(dataset: str, raw_data_dir: str) -> List[str]:
    """List clips from raw dataset storage."""
    if dataset == "interhuman":
        return list_interhuman_clips(raw_data_dir)
    return list_interx_clips(raw_data_dir)


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Temporal downsample helper."""
    if factor > 1:
        return arr[::factor]
    return arr


# SMPL joint indices for feet/ankles in the 22-joint skeleton.
FOOT_JOINT_INDICES = np.asarray([7, 8, 10, 11], dtype=np.int32)

# Knee bend DOF labels (mapped to joint_q indices through DOF_NAMES).
BALANCE_KNEE_DOF_LABELS = ("L_Knee_y", "R_Knee_y")

# Approximate kinematic foot flattening DOFs/joints.
FOOT_FLATTEN_ANKLE_DOFS = {
    "L": ("L_Ankle_x", "L_Ankle_y"),
    "R": ("R_Ankle_x", "R_Ankle_y"),
}
FOOT_FLATTEN_KNEE_DOF = {
    "L": "L_Knee_y",
    "R": "R_Knee_y",
}
FOOT_FLATTEN_FOOT_JOINTS = {
    "L": np.asarray([7, 10], dtype=np.int32),  # L_Ankle, L_Foot
    "R": np.asarray([8, 11], dtype=np.int32),  # R_Ankle, R_Foot
}


def _parse_float_sequence(raw: str, expected_len: int, flag_name: str) -> Tuple[float, ...]:
    """Parse float list from CLI string with comma/space separators."""
    toks = str(raw).replace(",", " ").split()
    if len(toks) != expected_len:
        raise ValueError(
            f"{flag_name} expects {expected_len} floats, got {len(toks)}: {raw!r}"
        )
    return tuple(float(x) for x in toks)


def _fmt_float_sequence(values: Tuple[float, ...]) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _soft_contact_xml_path(
    *,
    xml_path: str,
    solimp: Tuple[float, float, float],
    solref: Tuple[float, float],
    margin: Optional[float],
    foot_only: bool,
) -> str:
    """
    Create/reuse a cached XML variant with softer contact params.

    This keeps the base per-subject XML untouched and applies softening only for
    skyhook metric runs where requested.
    """
    key = (
        f"{os.path.abspath(xml_path)}|"
        f"solimp={solimp}|solref={solref}|margin={margin}|foot_only={int(foot_only)}"
    )
    suffix = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    out_path = os.path.join(
        os.path.dirname(xml_path),
        f"{os.path.splitext(os.path.basename(xml_path))[0]}_soft_{suffix}.xml",
    )
    if os.path.exists(out_path):
        return out_path

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _touch_geom(geom_el):
        geom_el.set("solimp", _fmt_float_sequence(solimp))
        geom_el.set("solref", _fmt_float_sequence(solref))
        if margin is not None:
            geom_el.set("margin", f"{float(margin):.6g}")

    if foot_only:
        foot_bodies = {"L_Ankle", "R_Ankle", "L_Toe", "R_Toe"}
        for body in root.findall(".//body"):
            if body.get("name") in foot_bodies:
                for geom in body.findall("geom"):
                    _touch_geom(geom)
    else:
        # Update default geom blocks and all explicit geoms.
        for dgeom in root.findall(".//default/geom"):
            _touch_geom(dgeom)
        for geom in root.findall(".//geom"):
            _touch_geom(geom)

    tree.write(out_path, xml_declaration=False)
    return out_path


def _compute_report_range(
    n_frames: int,
    trim_edges: int,
    base_start: int = 0,
    base_end: Optional[int] = None,
) -> Tuple[int, int]:
    """Return inclusive-start/exclusive-end range used for summary stats."""
    if n_frames <= 0:
        return 0, 0
    bs = int(np.clip(base_start, 0, n_frames))
    be = int(n_frames if base_end is None else np.clip(base_end, bs, n_frames))
    if be <= bs:
        bs, be = 0, n_frames
    trim = max(0, int(trim_edges))
    if trim == 0 or (be - bs) <= (2 * trim):
        return bs, be
    return bs + trim, be - trim


def _min_foot_height_m(positions: np.ndarray) -> float:
    """Compute minimum foot height (Z) from FK positions."""
    if positions.ndim != 3 or positions.shape[0] == 0:
        return float("nan")
    if positions.shape[1] > int(FOOT_JOINT_INDICES.max()):
        z = positions[:, FOOT_JOINT_INDICES, 2]
    else:
        z = positions[:, :, 2]
    return float(np.min(z))


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if arr.size else float("nan")


def _safe_max(arr: np.ndarray) -> float:
    return float(np.max(arr)) if arr.size else float("nan")


def _coord_index_from_hinge_dof_name(dof_name: str) -> Optional[int]:
    """Map hinge DOF name (75-dof space) to joint_q coordinate index (76-coord)."""
    try:
        dof_idx = DOF_NAMES.index(dof_name)
    except ValueError:
        return None
    if dof_idx < 6:
        return None
    # joint_q has an extra quaternion scalar at index 6.
    return dof_idx + 1


def _build_balance_seed_pose(
    q0: np.ndarray,
    knee_bend_deg: float,
) -> np.ndarray:
    """Create a bent-knee seed pose from first sequence frame."""
    q_seed = q0.copy()
    knee_target = np.float32(np.deg2rad(float(knee_bend_deg)))
    for label in BALANCE_KNEE_DOF_LABELS:
        cidx = _coord_index_from_hinge_dof_name(label)
        if cidx is None or cidx < 0 or cidx >= q_seed.shape[0]:
            continue
        # Ensure at least the target bend; keep deeper bends unchanged.
        q_seed[cidx] = np.float32(max(float(q_seed[cidx]), float(knee_target)))
    return q_seed


def _align_single_pose_to_ground(
    model,
    q_pose: np.ndarray,
    device: str,
    target_clearance_m: float,
) -> Tuple[np.ndarray, float, float]:
    """Shift root Z so min foot height of one pose hits target clearance."""
    q = q_pose.copy()
    pos_before = extract_positions_from_fk(model, q[None], device=device)
    min_before = _min_foot_height_m(pos_before)
    if np.isfinite(min_before):
        q[2] += np.float32(float(target_clearance_m) - float(min_before))
    pos_after = extract_positions_from_fk(model, q[None], device=device)
    min_after = _min_foot_height_m(pos_after)
    return q, float(min_before), float(min_after)


def _prepend_balance_warmup(
    joint_q_sequence: np.ndarray,
    q_seed: np.ndarray,
    hold_frames: int,
    transition_frames: int,
) -> Tuple[np.ndarray, int]:
    """Prepend [seed hold + linear blend seed->frame0] before the sequence."""
    hold = max(0, int(hold_frames))
    trans = max(0, int(transition_frames))
    if hold == 0 and trans == 0:
        return joint_q_sequence, 0

    chunks = []
    if hold > 0:
        chunks.append(np.repeat(q_seed[None, :], hold, axis=0).astype(np.float32))
    if trans > 0:
        alpha = np.linspace(0.0, 1.0, trans, endpoint=True, dtype=np.float32)[:, None]
        blend = (1.0 - alpha) * q_seed[None, :] + alpha * joint_q_sequence[0:1, :]
        chunks.append(blend.astype(np.float32))
    chunks.append(joint_q_sequence.astype(np.float32))
    return np.concatenate(chunks, axis=0), (hold + trans)


def _apply_kinematic_foot_flatten(
    *,
    model,
    joint_q: np.ndarray,
    device: str,
    trigger_height_m: float,
    ankle_alpha: float,
    knee_comp: float,
) -> Tuple[np.ndarray, int, int]:
    """
    Approximate kinematic foot flattening pass.

    If a side's ankle/foot joints are near the floor, blend ankle roll/pitch
    toward 0 (flat-foot assumption) and optionally add slight knee bend.
    """
    if joint_q.shape[0] == 0:
        return joint_q, 0, 0

    q = joint_q.astype(np.float32, copy=True)
    pos = extract_positions_from_fk(model, q, device=device)

    left_h = np.min(pos[:, FOOT_FLATTEN_FOOT_JOINTS["L"], 2], axis=1)
    right_h = np.min(pos[:, FOOT_FLATTEN_FOOT_JOINTS["R"], 2], axis=1)
    left_mask = left_h <= float(trigger_height_m)
    right_mask = right_h <= float(trigger_height_m)

    def _flatten_side(side: str, mask: np.ndarray):
        if not np.any(mask):
            return
        delta_mag = np.zeros((int(np.count_nonzero(mask)),), dtype=np.float32)
        for dof_name in FOOT_FLATTEN_ANKLE_DOFS[side]:
            cidx = _coord_index_from_hinge_dof_name(dof_name)
            if cidx is None or cidx < 0 or cidx >= q.shape[1]:
                continue
            old = q[mask, cidx].copy()
            q[mask, cidx] = (1.0 - float(ankle_alpha)) * old
            delta_mag += np.abs(old - q[mask, cidx]).astype(np.float32)

        if float(knee_comp) > 0.0:
            kidx = _coord_index_from_hinge_dof_name(FOOT_FLATTEN_KNEE_DOF[side])
            if kidx is not None and 0 <= kidx < q.shape[1]:
                # Small compensation to keep feet close to the floor after flattening.
                q[mask, kidx] = np.clip(
                    q[mask, kidx] + float(knee_comp) * delta_mag,
                    -0.5,
                    2.6,
                ).astype(np.float32)

    _flatten_side("L", left_mask)
    _flatten_side("R", right_mask)
    return q, int(np.count_nonzero(left_mask)), int(np.count_nonzero(right_mask))


def load_person_from_retargeted(
    data_dir: Optional[str], clip_id: str, person_idx: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load joint_q and betas for one person from retargeted directory."""
    if data_dir is None:
        return None
    jq_path = os.path.join(data_dir, f"{clip_id}_person{person_idx}_joint_q.npy")
    beta_path = os.path.join(data_dir, f"{clip_id}_person{person_idx}_betas.npy")
    if not os.path.exists(jq_path):
        return None
    if not os.path.exists(beta_path):
        return None
    return (
        np.load(jq_path).astype(np.float32),
        np.load(beta_path).astype(np.float64),
    )


def load_person_data(
    *,
    clip_id: str,
    person_idx: int,
    input_mode: str,
    data_dir: Optional[str],
    raw_persons: Optional[list],
) -> Optional[ClipPersonData]:
    """
    Load one clip/person trajectory from requested input mode.

    input_mode:
      - retargeted: require *_joint_q.npy + *_betas.npy
      - dataset:    require raw dataset clip, convert to joint_q
      - auto:       try retargeted first, then dataset
    """
    if input_mode in ("retargeted", "auto"):
        loaded = load_person_from_retargeted(data_dir, clip_id, person_idx)
        if loaded is not None:
            joint_q, betas = loaded
            return ClipPersonData(
                clip_id=clip_id,
                person_idx=person_idx,
                joint_q=joint_q,
                betas=betas,
                source="retargeted",
            )
        if input_mode == "retargeted":
            return None

    # dataset / auto fallback
    if raw_persons is None or person_idx >= len(raw_persons):
        return None

    pdata = raw_persons[person_idx]
    joint_q = smplx_to_joint_q(
        pdata["root_orient"], pdata["pose_body"], pdata["trans"], pdata["betas"]
    ).astype(np.float32)
    betas = np.asarray(pdata["betas"], dtype=np.float64)
    return ClipPersonData(
        clip_id=clip_id,
        person_idx=person_idx,
        joint_q=joint_q,
        betas=betas,
        source="dataset",
    )


class SmplxFk:
    """Lazy SMPL-X FK provider for MPJPE ground-truth positions."""

    def __init__(self):
        self._torch = None
        self._bm = None
        self._init_error = None

    @property
    def available(self) -> bool:
        return self._init_error is None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def _lazy_init(self):
        if self._bm is not None or self._init_error is not None:
            return
        try:
            import torch  # pylint: disable=import-outside-toplevel

            body_model_dir = os.path.join(PROJECT_ROOT, "data", "body_model")
            if body_model_dir not in sys.path:
                sys.path.insert(0, body_model_dir)
            from body_model import BodyModel  # pylint: disable=import-outside-toplevel

            model_path = os.path.join(
                PROJECT_ROOT, "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"
            )
            self._torch = torch
            self._bm = BodyModel(model_path, num_betas=10)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            self._init_error = str(exc)

    def compute_positions(self, person_data: dict) -> Optional[np.ndarray]:
        """Return (T, 22, 3) SMPL-X FK positions in meters (Z-up)."""
        self._lazy_init()
        if self._bm is None:
            return None

        torch = self._torch
        T = int(person_data["root_orient"].shape[0])
        betas = np.asarray(person_data["betas"], dtype=np.float32)
        betas_t = (
            torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(T, -1)
        )

        with torch.no_grad():
            out = self._bm(
                root_orient=torch.tensor(person_data["root_orient"], dtype=torch.float32),
                pose_body=torch.tensor(person_data["pose_body"], dtype=torch.float32),
                betas=betas_t,
                trans=torch.tensor(person_data["trans"], dtype=torch.float32),
            )

        return out.Jtr[:, :N_SMPL_JOINTS].detach().cpu().numpy().astype(np.float32)


def build_single_person_model(
    betas: np.ndarray,
    device: str,
    with_ground: bool,
    *,
    sphere_feet: bool = False,
    sphere_feet_cache_dir: Optional[str] = None,
    contact_soften: bool = False,
    contact_soften_solimp: Tuple[float, float, float] = (0.9, 0.99, 0.003),
    contact_soften_solref: Tuple[float, float] = (0.015, 1.0),
    contact_soften_margin: Optional[float] = None,
    contact_soften_foot_only: bool = False,
):
    """Build one-person Newton model for inverse dynamics / FK."""
    if sphere_feet:
        from prepare2.gen_smpl_with_sphere_feet_xml import (
            get_or_create_sphere_feet_xml,
        )

        cache_dir = sphere_feet_cache_dir
        if cache_dir is not None and not os.path.isabs(cache_dir):
            cache_dir = os.path.join(PROJECT_ROOT, cache_dir)
        xml_path = get_or_create_sphere_feet_xml(betas, cache_dir=cache_dir)
    else:
        xml_path = get_or_create_xml(betas)
    if contact_soften:
        xml_path = _soft_contact_xml_path(
            xml_path=xml_path,
            solimp=contact_soften_solimp,
            solref=contact_soften_solref,
            margin=contact_soften_margin,
            foot_only=bool(contact_soften_foot_only),
        )
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    if with_ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _summary_from_arrays(
    *,
    root_force_l2: np.ndarray,
    root_torque_l2: np.ndarray,
    root_wrench_l2: np.ndarray,
    mpjpe_per_frame: np.ndarray,
    report_start: int,
    report_end: int,
) -> dict:
    """Build summary stats from full arrays and a report window."""
    n_frames = int(root_force_l2.shape[0])
    rs = int(np.clip(report_start, 0, n_frames))
    re = int(np.clip(report_end, rs, n_frames))
    if re <= rs:
        rs, re = 0, n_frames

    rf_report = root_force_l2[rs:re]
    rt_report = root_torque_l2[rs:re]
    rw_report = root_wrench_l2[rs:re]
    mp_report = mpjpe_per_frame[rs:re]

    valid_report = np.isfinite(mp_report)
    valid_full = np.isfinite(mpjpe_per_frame)

    return {
        "n_frames": n_frames,
        "n_report_frames": int(re - rs),
        "report_frame_start": rs,
        "report_frame_end_exclusive": re,
        # Primary metrics: report window (possibly trimmed).
        "skyhook_residual_force_mean_N": _safe_mean(rf_report),
        "skyhook_residual_force_std_N": _safe_std(rf_report),
        "skyhook_residual_force_max_N": _safe_max(rf_report),
        "skyhook_residual_torque_mean_Nm": _safe_mean(rt_report),
        "skyhook_residual_wrench_mean": _safe_mean(rw_report),
        "mpjpe_mean_m": (
            float(np.nanmean(mp_report)) if valid_report.any() else float("nan")
        ),
        "mpjpe_valid": bool(valid_report.any()),
        # Full-sequence reference metrics for diagnostics.
        "skyhook_residual_force_mean_full_N": _safe_mean(root_force_l2),
        "skyhook_residual_force_max_full_N": _safe_max(root_force_l2),
        "mpjpe_mean_full_m": (
            float(np.nanmean(mpjpe_per_frame)) if valid_full.any() else float("nan")
        ),
        "mpjpe_valid_full": bool(valid_full.any()),
    }


def summarize_existing_output(npz_path: str) -> Optional[dict]:
    """Recover summary row from existing NPZ if sidecar JSON is missing."""
    if not os.path.exists(npz_path):
        return None
    try:
        with np.load(npz_path) as d:
            root_force = np.asarray(d["root_force_l2"])
            root_torque = np.asarray(d["root_torque_l2"])
            root_wrench = np.asarray(d["root_wrench_l2"])
            mpjpe = np.asarray(d["mpjpe_per_frame_m"])
            rs = int(d["report_frame_start"]) if "report_frame_start" in d else 0
            re = (
                int(d["report_frame_end_exclusive"])
                if "report_frame_end_exclusive" in d
                else int(root_force.shape[0])
            )
            out = _summary_from_arrays(
                root_force_l2=root_force,
                root_torque_l2=root_torque,
                root_wrench_l2=root_wrench,
                mpjpe_per_frame=mpjpe,
                report_start=rs,
                report_end=re,
            )
            if "trim_edges" in d:
                out["trim_edges"] = int(d["trim_edges"])
            if "sequence_frame_start" in d:
                out["sequence_frame_start"] = int(d["sequence_frame_start"])
            if "sequence_frame_end_exclusive" in d:
                out["sequence_frame_end_exclusive"] = int(d["sequence_frame_end_exclusive"])
            if "balance_hold_frames" in d:
                out["balance_hold_frames"] = int(d["balance_hold_frames"])
            if "balance_transition_frames" in d:
                out["balance_transition_frames"] = int(d["balance_transition_frames"])
            if "balance_knee_bend_deg" in d:
                out["balance_knee_bend_deg"] = float(d["balance_knee_bend_deg"])
            if "seed_min_foot_z_before_m" in d:
                out["seed_min_foot_z_before_m"] = float(d["seed_min_foot_z_before_m"])
            if "seed_min_foot_z_after_m" in d:
                out["seed_min_foot_z_after_m"] = float(d["seed_min_foot_z_after_m"])
            if "ground_offset_m" in d:
                out["ground_offset_m"] = float(d["ground_offset_m"])
            if "min_foot_z_before_m" in d:
                out["min_foot_z_before_m"] = float(d["min_foot_z_before_m"])
            if "min_foot_z_after_m" in d:
                out["min_foot_z_after_m"] = float(d["min_foot_z_after_m"])
            if "sphere_feet" in d:
                out["sphere_feet"] = bool(int(d["sphere_feet"]))
            if "sphere_feet_cache_dir" in d:
                out["sphere_feet_cache_dir"] = str(np.asarray(d["sphere_feet_cache_dir"]).item())
            if "contact_soften" in d:
                out["contact_soften"] = bool(int(d["contact_soften"]))
            if "contact_soften_solimp" in d:
                out["contact_soften_solimp"] = [float(x) for x in np.asarray(d["contact_soften_solimp"]).ravel()]
            if "contact_soften_solref" in d:
                out["contact_soften_solref"] = [float(x) for x in np.asarray(d["contact_soften_solref"]).ravel()]
            if "contact_soften_margin_m" in d:
                out["contact_soften_margin_m"] = float(d["contact_soften_margin_m"])
            if "contact_soften_foot_only" in d:
                out["contact_soften_foot_only"] = bool(int(d["contact_soften_foot_only"]))
            if "foot_flatten" in d:
                out["foot_flatten"] = bool(int(d["foot_flatten"]))
            if "foot_flatten_height_m" in d:
                out["foot_flatten_height_m"] = float(d["foot_flatten_height_m"])
            if "foot_flatten_ankle_alpha" in d:
                out["foot_flatten_ankle_alpha"] = float(d["foot_flatten_ankle_alpha"])
            if "foot_flatten_knee_comp" in d:
                out["foot_flatten_knee_comp"] = float(d["foot_flatten_knee_comp"])
            if "foot_flatten_frames_left" in d:
                out["foot_flatten_frames_left"] = int(d["foot_flatten_frames_left"])
            if "foot_flatten_frames_right" in d:
                out["foot_flatten_frames_right"] = int(d["foot_flatten_frames_right"])
            return out
    except Exception:
        return None


def compute_metrics_for_person(
    *,
    clip_data: ClipPersonData,
    fps: int,
    downsample_factor: int,
    diff_method: str,
    device: str,
    data_dir: Optional[str],
    raw_person_data: Optional[dict],
    smplx_fk: Optional[SmplxFk],
    disable_mpjpe: bool,
    ground_fix: bool,
    ground_clearance: float,
    trim_edges: int,
    balance_hold_frames: int,
    balance_transition_frames: int,
    balance_knee_bend_deg: float,
    sphere_feet: bool,
    sphere_feet_cache_dir: Optional[str],
    contact_soften: bool,
    contact_soften_solimp: Tuple[float, float, float],
    contact_soften_solref: Tuple[float, float],
    contact_soften_margin: Optional[float],
    contact_soften_foot_only: bool,
    foot_flatten: bool,
    foot_flatten_height_m: float,
    foot_flatten_ankle_alpha: float,
    foot_flatten_knee_comp: float,
):
    """Compute per-frame skyhook metrics + MPJPE for one clip/person."""
    joint_q_sequence = downsample(clip_data.joint_q.astype(np.float32), downsample_factor)
    if joint_q_sequence.shape[0] == 0:
        raise ValueError("Empty trajectory after downsampling.")

    # Build model once and optionally raise root translation to eliminate
    # global foot penetration before inverse dynamics.
    model_id = build_single_person_model(
        clip_data.betas,
        device=device,
        with_ground=True,
        sphere_feet=sphere_feet,
        sphere_feet_cache_dir=sphere_feet_cache_dir,
        contact_soften=contact_soften,
        contact_soften_solimp=contact_soften_solimp,
        contact_soften_solref=contact_soften_solref,
        contact_soften_margin=contact_soften_margin,
        contact_soften_foot_only=contact_soften_foot_only,
    )
    seed_min_foot_before_m = float("nan")
    seed_min_foot_after_m = float("nan")
    if max(0, int(balance_hold_frames)) > 0 or max(0, int(balance_transition_frames)) > 0:
        q_seed_before_ground = _build_balance_seed_pose(
            joint_q_sequence[0], knee_bend_deg=balance_knee_bend_deg
        )
        (
            q_seed_on_ground,
            seed_min_foot_before_m,
            seed_min_foot_after_m,
        ) = _align_single_pose_to_ground(
            model_id,
            q_seed_before_ground,
            device=device,
            target_clearance_m=float(ground_clearance),
        )
        joint_q, warmup_frames = _prepend_balance_warmup(
            joint_q_sequence,
            q_seed_on_ground,
            hold_frames=balance_hold_frames,
            transition_frames=balance_transition_frames,
        )
    else:
        joint_q = joint_q_sequence
        warmup_frames = 0
    sequence_frame_start = int(warmup_frames)
    sequence_frame_end = int(sequence_frame_start + joint_q_sequence.shape[0])

    foot_flatten_frames_left = 0
    foot_flatten_frames_right = 0
    if foot_flatten:
        joint_q, foot_flatten_frames_left, foot_flatten_frames_right = _apply_kinematic_foot_flatten(
            model=model_id,
            joint_q=joint_q,
            device=device,
            trigger_height_m=float(foot_flatten_height_m),
            ankle_alpha=float(np.clip(foot_flatten_ankle_alpha, 0.0, 1.0)),
            knee_comp=float(max(0.0, foot_flatten_knee_comp)),
        )

    min_foot_z_before = float("nan")
    min_foot_z_after = float("nan")
    ground_offset_m = 0.0
    if ground_fix:
        pred_before = extract_positions_from_fk(model_id, joint_q, device=device)
        min_foot_z_before = _min_foot_height_m(pred_before)
        ground_offset_m = max(0.0, float(ground_clearance) - min_foot_z_before)
        if ground_offset_m > 0.0:
            joint_q = joint_q.copy()
            joint_q[:, 2] += np.float32(ground_offset_m)
        pred_after = extract_positions_from_fk(model_id, joint_q, device=device)
        min_foot_z_after = _min_foot_height_m(pred_after)

    # Inverse dynamics torques include root virtual forces in DOFs 0:6.
    torques, _, _ = inverse_dynamics(
        model_id,
        joint_q,
        fps,
        device=device,
        diff_method=diff_method,
    )

    # Predicted joint positions from FK on the same model/trajectory.
    pred_positions = extract_positions_from_fk(model_id, joint_q, device=device)

    # Per-frame skyhook arrays
    root_force_xyz = torques[:, 0:3].astype(np.float32)
    root_torque_xyz = torques[:, 3:6].astype(np.float32)
    root_force_l2 = np.linalg.norm(root_force_xyz, axis=1).astype(np.float32)
    root_torque_l2 = np.linalg.norm(root_torque_xyz, axis=1).astype(np.float32)
    root_wrench_l2 = np.linalg.norm(torques[:, 0:6], axis=1).astype(np.float32)

    # MPJPE reference target priority:
    #  1) raw SMPL-X FK (if available)
    #  2) saved retargeted FK positions (*.npy)
    ref_positions = None
    mpjpe_source = "none"
    if not disable_mpjpe:
        if raw_person_data is not None and smplx_fk is not None and smplx_fk.available:
            ref_positions = smplx_fk.compute_positions(raw_person_data)
            if ref_positions is not None:
                ref_positions = downsample(ref_positions, downsample_factor)
                mpjpe_source = "smplx_fk"

        if ref_positions is None and data_dir is not None:
            ref_path = os.path.join(
                data_dir, f"{clip_data.clip_id}_person{clip_data.person_idx}.npy"
            )
            if os.path.exists(ref_path):
                ref_positions = downsample(
                    np.load(ref_path).astype(np.float32), downsample_factor
                )
                mpjpe_source = "retarget_fk"

    T = min(joint_q.shape[0], torques.shape[0], pred_positions.shape[0], sequence_frame_end)
    root_force_xyz = root_force_xyz[:T]
    root_torque_xyz = root_torque_xyz[:T]
    root_force_l2 = root_force_l2[:T]
    root_torque_l2 = root_torque_l2[:T]
    root_wrench_l2 = root_wrench_l2[:T]
    pred_positions = pred_positions[:T]
    sequence_frame_start = int(np.clip(sequence_frame_start, 0, T))
    sequence_frame_end = int(np.clip(sequence_frame_end, sequence_frame_start, T))

    mpjpe_per_frame = np.full((T,), np.nan, dtype=np.float32)
    mpjpe_per_joint = np.full((T, N_SMPL_JOINTS), np.nan, dtype=np.float32)
    if ref_positions is not None and ref_positions.shape[0] > 0:
        seq_len = max(0, sequence_frame_end - sequence_frame_start)
        T2 = min(seq_len, ref_positions.shape[0])
        err = np.linalg.norm(
            pred_positions[sequence_frame_start:sequence_frame_start + T2]
            - ref_positions[:T2],
            axis=-1,
        ).astype(np.float32)
        mpjpe_per_joint[sequence_frame_start:sequence_frame_start + T2] = err
        mpjpe_per_frame[sequence_frame_start:sequence_frame_start + T2] = err.mean(axis=1)

    report_start, report_end = _compute_report_range(
        T,
        trim_edges,
        base_start=sequence_frame_start,
        base_end=sequence_frame_end,
    )

    return {
        "n_frames": int(T),
        "joint_q_used": joint_q[:T].astype(np.float32),
        "betas": clip_data.betas.astype(np.float32),
        "root_force_xyz": root_force_xyz,
        "root_torque_xyz": root_torque_xyz,
        "root_force_l2": root_force_l2,
        "root_torque_l2": root_torque_l2,
        "root_wrench_l2": root_wrench_l2,
        "mpjpe_per_frame_m": mpjpe_per_frame,
        "mpjpe_per_joint_m": mpjpe_per_joint,
        "joint_source": clip_data.source,
        "mpjpe_source": mpjpe_source,
        "trim_edges": int(max(0, trim_edges)),
        "report_frame_start": int(report_start),
        "report_frame_end_exclusive": int(report_end),
        "sequence_frame_start": int(sequence_frame_start),
        "sequence_frame_end_exclusive": int(sequence_frame_end),
        "balance_hold_frames": int(max(0, balance_hold_frames)),
        "balance_transition_frames": int(max(0, balance_transition_frames)),
        "balance_knee_bend_deg": float(balance_knee_bend_deg),
        "seed_min_foot_z_before_m": float(seed_min_foot_before_m),
        "seed_min_foot_z_after_m": float(seed_min_foot_after_m),
        "ground_fix": bool(ground_fix),
        "ground_clearance_m": float(ground_clearance),
        "ground_offset_m": float(ground_offset_m),
        "min_foot_z_before_m": float(min_foot_z_before),
        "min_foot_z_after_m": float(min_foot_z_after),
        "sphere_feet": bool(sphere_feet),
        "sphere_feet_cache_dir": str(sphere_feet_cache_dir or ""),
        "contact_soften": bool(contact_soften),
        "contact_soften_solimp": tuple(float(x) for x in contact_soften_solimp),
        "contact_soften_solref": tuple(float(x) for x in contact_soften_solref),
        "contact_soften_margin_m": (
            float(contact_soften_margin)
            if contact_soften_margin is not None
            else float("nan")
        ),
        "contact_soften_foot_only": bool(contact_soften_foot_only),
        "foot_flatten": bool(foot_flatten),
        "foot_flatten_height_m": float(foot_flatten_height_m),
        "foot_flatten_ankle_alpha": float(foot_flatten_ankle_alpha),
        "foot_flatten_knee_comp": float(foot_flatten_knee_comp),
        "foot_flatten_frames_left": int(foot_flatten_frames_left),
        "foot_flatten_frames_right": int(foot_flatten_frames_right),
    }


def save_person_outputs(
    *,
    clip_id: str,
    person_idx: int,
    output_dir: str,
    metrics: dict,
):
    """Save NPZ + JSON sidecar for one clip/person."""
    base = f"{clip_id}_person{person_idx}_skyhook_metrics"
    npz_path = os.path.join(output_dir, f"{base}.npz")
    json_path = os.path.join(output_dir, f"{base}.json")

    n_frames = metrics["n_frames"]
    frame = np.arange(n_frames, dtype=np.int32)
    np.savez_compressed(
        npz_path,
        frame=frame,
        joint_q_used=metrics["joint_q_used"],
        betas=metrics["betas"],
        root_force_xyz_N=metrics["root_force_xyz"],
        root_torque_xyz_Nm=metrics["root_torque_xyz"],
        root_force_l2=metrics["root_force_l2"],
        root_torque_l2=metrics["root_torque_l2"],
        root_wrench_l2=metrics["root_wrench_l2"],
        mpjpe_per_frame_m=metrics["mpjpe_per_frame_m"],
        mpjpe_per_joint_m=metrics["mpjpe_per_joint_m"],
        trim_edges=np.int32(metrics["trim_edges"]),
        report_frame_start=np.int32(metrics["report_frame_start"]),
        report_frame_end_exclusive=np.int32(metrics["report_frame_end_exclusive"]),
        sequence_frame_start=np.int32(metrics["sequence_frame_start"]),
        sequence_frame_end_exclusive=np.int32(metrics["sequence_frame_end_exclusive"]),
        balance_hold_frames=np.int32(metrics["balance_hold_frames"]),
        balance_transition_frames=np.int32(metrics["balance_transition_frames"]),
        balance_knee_bend_deg=np.float32(metrics["balance_knee_bend_deg"]),
        seed_min_foot_z_before_m=np.float32(metrics["seed_min_foot_z_before_m"]),
        seed_min_foot_z_after_m=np.float32(metrics["seed_min_foot_z_after_m"]),
        ground_fix=np.int32(1 if metrics["ground_fix"] else 0),
        ground_clearance_m=np.float32(metrics["ground_clearance_m"]),
        ground_offset_m=np.float32(metrics["ground_offset_m"]),
        min_foot_z_before_m=np.float32(metrics["min_foot_z_before_m"]),
        min_foot_z_after_m=np.float32(metrics["min_foot_z_after_m"]),
        sphere_feet=np.int32(1 if metrics["sphere_feet"] else 0),
        sphere_feet_cache_dir=np.asarray(metrics["sphere_feet_cache_dir"], dtype=np.str_),
        contact_soften=np.int32(1 if metrics["contact_soften"] else 0),
        contact_soften_solimp=np.asarray(metrics["contact_soften_solimp"], dtype=np.float32),
        contact_soften_solref=np.asarray(metrics["contact_soften_solref"], dtype=np.float32),
        contact_soften_margin_m=np.float32(metrics["contact_soften_margin_m"]),
        contact_soften_foot_only=np.int32(1 if metrics["contact_soften_foot_only"] else 0),
        foot_flatten=np.int32(1 if metrics["foot_flatten"] else 0),
        foot_flatten_height_m=np.float32(metrics["foot_flatten_height_m"]),
        foot_flatten_ankle_alpha=np.float32(metrics["foot_flatten_ankle_alpha"]),
        foot_flatten_knee_comp=np.float32(metrics["foot_flatten_knee_comp"]),
        foot_flatten_frames_left=np.int32(metrics["foot_flatten_frames_left"]),
        foot_flatten_frames_right=np.int32(metrics["foot_flatten_frames_right"]),
    )

    summary = _summary_from_arrays(
        root_force_l2=metrics["root_force_l2"],
        root_torque_l2=metrics["root_torque_l2"],
        root_wrench_l2=metrics["root_wrench_l2"],
        mpjpe_per_frame=metrics["mpjpe_per_frame_m"],
        report_start=metrics["report_frame_start"],
        report_end=metrics["report_frame_end_exclusive"],
    )
    meta = {
        "clip_id": clip_id,
        "person_idx": person_idx,
        "joint_source": metrics["joint_source"],
        "mpjpe_source": metrics["mpjpe_source"],
        "trim_edges": int(metrics["trim_edges"]),
        "sequence_frame_start": int(metrics["sequence_frame_start"]),
        "sequence_frame_end_exclusive": int(metrics["sequence_frame_end_exclusive"]),
        "balance_hold_frames": int(metrics["balance_hold_frames"]),
        "balance_transition_frames": int(metrics["balance_transition_frames"]),
        "balance_knee_bend_deg": float(metrics["balance_knee_bend_deg"]),
        "seed_min_foot_z_before_m": float(metrics["seed_min_foot_z_before_m"]),
        "seed_min_foot_z_after_m": float(metrics["seed_min_foot_z_after_m"]),
        "ground_fix": bool(metrics["ground_fix"]),
        "ground_clearance_m": float(metrics["ground_clearance_m"]),
        "ground_offset_m": float(metrics["ground_offset_m"]),
        "min_foot_z_before_m": float(metrics["min_foot_z_before_m"]),
        "min_foot_z_after_m": float(metrics["min_foot_z_after_m"]),
        "sphere_feet": bool(metrics["sphere_feet"]),
        "sphere_feet_cache_dir": str(metrics["sphere_feet_cache_dir"]),
        "contact_soften": bool(metrics["contact_soften"]),
        "contact_soften_solimp": list(metrics["contact_soften_solimp"]),
        "contact_soften_solref": list(metrics["contact_soften_solref"]),
        "contact_soften_margin_m": float(metrics["contact_soften_margin_m"]),
        "contact_soften_foot_only": bool(metrics["contact_soften_foot_only"]),
        "foot_flatten": bool(metrics["foot_flatten"]),
        "foot_flatten_height_m": float(metrics["foot_flatten_height_m"]),
        "foot_flatten_ankle_alpha": float(metrics["foot_flatten_ankle_alpha"]),
        "foot_flatten_knee_comp": float(metrics["foot_flatten_knee_comp"]),
        "foot_flatten_frames_left": int(metrics["foot_flatten_frames_left"]),
        "foot_flatten_frames_right": int(metrics["foot_flatten_frames_right"]),
        **summary,
        "npz_path": npz_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return npz_path, json_path, meta


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-frame skyhook residual metrics and MPJPE."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["interhuman", "interx"],
        help="Dataset name (used for clip listing and defaults).",
    )
    parser.add_argument(
        "--input",
        default="auto",
        choices=["auto", "retargeted", "dataset"],
        help="Input source: retargeted files, raw dataset, or auto fallback.",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="Single clip ID. If omitted, process all clips.",
    )
    parser.add_argument(
        "--person",
        type=int,
        default=None,
        choices=[0, 1],
        help="Person index. Default: both persons.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with retargeted .npy files (joint_q/betas).",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=None,
        help="Raw dataset root directory for loading SMPL-X clips.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/skyhook_metrics/{dataset}).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Motion FPS used for inverse dynamics (default: 30).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Downsample factor for trajectories (default: 2 -> 60 to 30fps).",
    )
    parser.add_argument(
        "--diff-method",
        default="spline",
        choices=["spline", "fd"],
        help="Differentiation method for inverse dynamics.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Process at most N clips (for smoke tests).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-clip outputs.",
    )
    parser.add_argument(
        "--disable-mpjpe",
        action="store_true",
        help="Skip MPJPE computation and only save skyhook metrics.",
    )
    parser.add_argument(
        "--ground-fix",
        action="store_true",
        help=(
            "Apply a constant upward root-translation offset so feet are at least "
            "--ground-clearance above the ground before inverse dynamics."
        ),
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=0.01,
        help="Minimum desired foot height in meters when --ground-fix is enabled.",
    )
    parser.add_argument(
        "--trim-edges",
        type=int,
        default=0,
        help=(
            "Trim this many frames from each side when reporting summary stats. "
            "Per-frame arrays are always saved untrimmed."
        ),
    )
    parser.add_argument(
        "--balance-hold-frames",
        type=int,
        default=0,
        help="Number of initial frames to hold a bent-knee balanced pose.",
    )
    parser.add_argument(
        "--balance-transition-frames",
        type=int,
        default=0,
        help="Number of frames to blend from balanced pose to sequence frame 0.",
    )
    parser.add_argument(
        "--balance-knee-bend-deg",
        type=float,
        default=20.0,
        help="Target minimum bend angle (deg) for L/R knee in balance warmup.",
    )
    parser.add_argument(
        "--sphere-feet",
        action="store_true",
        help="Build per-subject model using sphere-cluster feet XML.",
    )
    parser.add_argument(
        "--sphere-feet-cache-dir",
        type=str,
        default="prepare2/xml_cache_sphere_feet",
        help="Cache dir for sphere-feet XMLs (separate from default xml_cache).",
    )
    parser.add_argument(
        "--contact-soften",
        action="store_true",
        help="Use a softened-contact XML variant when computing skyhook torques.",
    )
    parser.add_argument(
        "--contact-soften-solimp",
        type=str,
        default="0.8 0.95 0.02",
        help="MuJoCo geom solimp values (3 floats) for softened contacts.",
    )
    parser.add_argument(
        "--contact-soften-solref",
        type=str,
        default="0.03 1.0",
        help="MuJoCo geom solref values (2 floats) for softened contacts.",
    )
    parser.add_argument(
        "--contact-soften-margin",
        type=float,
        default=None,
        help="Optional geom contact margin (meters) for softened contacts.",
    )
    parser.add_argument(
        "--contact-soften-foot-only",
        action="store_true",
        help="Apply softened contact params only to L/R ankle+toe geoms.",
    )
    parser.add_argument(
        "--foot-flatten",
        action="store_true",
        help="Apply approximate kinematic foot flattening before inverse dynamics.",
    )
    parser.add_argument(
        "--foot-flatten-height",
        type=float,
        default=0.035,
        help="Trigger flattening when a side's foot is below this height (m).",
    )
    parser.add_argument(
        "--foot-flatten-ankle-alpha",
        type=float,
        default=0.65,
        help="Blend strength toward flat ankle (0..1).",
    )
    parser.add_argument(
        "--foot-flatten-knee-comp",
        type=float,
        default=0.25,
        help="Knee compensation gain for flattened ankle rotation.",
    )
    parser.add_argument("--gpu", default="cuda:0")
    args = parser.parse_args()

    contact_soften_solimp = _parse_float_sequence(
        args.contact_soften_solimp, 3, "--contact-soften-solimp"
    )
    contact_soften_solref = _parse_float_sequence(
        args.contact_soften_solref, 2, "--contact-soften-solref"
    )
    foot_flatten_ankle_alpha = float(np.clip(args.foot_flatten_ankle_alpha, 0.0, 1.0))
    foot_flatten_knee_comp = float(max(0.0, args.foot_flatten_knee_comp))
    sphere_feet_cache_dir = (
        args.sphere_feet_cache_dir if args.sphere_feet and args.sphere_feet_cache_dir else None
    )

    # Directory defaults
    default_retargeted = f"data/retargeted_v2/{args.dataset}"
    default_raw = (
        "data/InterHuman" if args.dataset == "interhuman" else "data/Inter-X_Dataset"
    )
    data_dir = None
    raw_data_dir = None

    need_retargeted = args.input in ("retargeted", "auto")
    need_raw = args.input in ("dataset", "auto") or not args.disable_mpjpe

    if need_retargeted:
        try:
            data_dir = _resolve_existing_dir(args.data_dir, default_retargeted)
        except FileNotFoundError:
            if args.input == "retargeted":
                raise

    # Even in retargeted mode, raw data is useful for MPJPE (SMPL-X FK target).
    if need_raw:
        try:
            raw_data_dir = _resolve_existing_dir(args.raw_data_dir, default_raw)
        except FileNotFoundError:
            if args.input == "dataset":
                raise

    if data_dir is None and raw_data_dir is None:
        raise FileNotFoundError(
            "No usable input directory found (retargeted and raw missing)."
        )
    output_dir = _resolve_output_dir(
        args.output_dir, f"data/skyhook_metrics/{args.dataset}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Clip list
    if args.clip:
        clips = [args.clip]
    else:
        clips_rt = list_retargeted_clips(data_dir) if data_dir is not None else []
        clips_raw = list_raw_clips(args.dataset, raw_data_dir) if raw_data_dir is not None else []
        if args.input == "retargeted":
            clips = clips_rt
        elif args.input == "dataset":
            clips = clips_raw
        else:
            # auto: union, prioritize deterministic sorted IDs
            clips = sorted(set(clips_rt) | set(clips_raw))

    if args.max_clips is not None:
        clips = clips[: args.max_clips]

    persons = [args.person] if args.person is not None else [0, 1]

    # Save run config
    config = {
        "script": "compute_skyhook_metrics.py",
        "dataset": args.dataset,
        "input": args.input,
        "data_dir": data_dir,
        "raw_data_dir": raw_data_dir,
        "output_dir": output_dir,
        "fps": args.fps,
        "downsample": args.downsample,
        "diff_method": args.diff_method,
        "device": args.gpu,
        "disable_mpjpe": args.disable_mpjpe,
        "ground_fix": args.ground_fix,
        "ground_clearance_m": args.ground_clearance,
        "trim_edges": args.trim_edges,
        "balance_hold_frames": args.balance_hold_frames,
        "balance_transition_frames": args.balance_transition_frames,
        "balance_knee_bend_deg": args.balance_knee_bend_deg,
        "sphere_feet": args.sphere_feet,
        "sphere_feet_cache_dir": sphere_feet_cache_dir,
        "contact_soften": args.contact_soften,
        "contact_soften_solimp": list(contact_soften_solimp),
        "contact_soften_solref": list(contact_soften_solref),
        "contact_soften_margin_m": args.contact_soften_margin,
        "contact_soften_foot_only": args.contact_soften_foot_only,
        "foot_flatten": args.foot_flatten,
        "foot_flatten_height_m": args.foot_flatten_height,
        "foot_flatten_ankle_alpha": foot_flatten_ankle_alpha,
        "foot_flatten_knee_comp": foot_flatten_knee_comp,
        "n_clips": len(clips),
        "persons": persons,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"{'='*68}")
    print("Skyhook Metric Pass")
    print(f"{'='*68}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Input mode:   {args.input}")
    print(f"  Clips:        {len(clips)}")
    print(f"  Persons:      {persons}")
    print(f"  FPS:          {args.fps}")
    print(f"  Downsample:   {args.downsample}x")
    print(f"  Diff method:  {args.diff_method}")
    print(f"  MPJPE:        {'disabled' if args.disable_mpjpe else 'enabled'}")
    print(
        f"  Ground fix:   {'enabled' if args.ground_fix else 'disabled'} "
        f"(clearance={args.ground_clearance:.3f} m)"
    )
    print(
        "  Balance warmup: "
        f"hold={max(0, int(args.balance_hold_frames))}, "
        f"blend={max(0, int(args.balance_transition_frames))}, "
        f"knee={float(args.balance_knee_bend_deg):.1f} deg"
    )
    print(
        "  Sphere feet:    "
        f"{'enabled' if args.sphere_feet else 'disabled'} "
        f"(cache={sphere_feet_cache_dir})"
    )
    print(
        "  Contact soften: "
        f"{'enabled' if args.contact_soften else 'disabled'} "
        f"(solimp={list(contact_soften_solimp)}, "
        f"solref={list(contact_soften_solref)}, "
        f"margin={args.contact_soften_margin}, "
        f"foot_only={args.contact_soften_foot_only})"
    )
    print(
        "  Foot flatten:   "
        f"{'enabled' if args.foot_flatten else 'disabled'} "
        f"(h={float(args.foot_flatten_height):.3f} m, "
        f"ankle_alpha={foot_flatten_ankle_alpha:.2f}, "
        f"knee_comp={foot_flatten_knee_comp:.2f})"
    )
    print(f"  Trim edges:   {max(0, int(args.trim_edges))} frames/side")
    print(f"  Data dir:     {data_dir}")
    print(f"  Raw data dir: {raw_data_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"{'='*68}")

    smplx_fk = None if args.disable_mpjpe else SmplxFk()

    rows = []
    processed = 0
    skipped = 0
    missing = 0
    errors = 0
    t_start = time.time()

    total_tasks = len(clips) * len(persons)
    task_idx = 0

    for clip_id in clips:
        raw_persons = (
            load_raw_clip(args.dataset, raw_data_dir, clip_id)
            if raw_data_dir is not None
            else None
        )
        for person_idx in persons:
            task_idx += 1
            elapsed = time.time() - t_start
            rate = (task_idx / elapsed * 3600.0) if elapsed > 0 else 0.0
            eta_h = ((total_tasks - task_idx) / rate) if rate > 0 else 0.0

            base = f"{clip_id}_person{person_idx}_skyhook_metrics"
            npz_path = os.path.join(output_dir, f"{base}.npz")
            json_path = os.path.join(output_dir, f"{base}.json")
            print(
                f"\n[{task_idx}/{total_tasks}] clip={clip_id} p{person_idx} "
                f"(done={processed}, skip={skipped}, miss={missing}, err={errors}, "
                f"rate={rate:.0f}/h, eta={eta_h:.1f}h)"
            )

            # Skip path (unless --force)
            if os.path.exists(npz_path) and not args.force:
                row = None
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            row = json.load(f)
                    except Exception:
                        row = None
                if row is None:
                    summary = summarize_existing_output(npz_path)
                    if summary is not None:
                        row = {
                            "clip_id": clip_id,
                            "person_idx": person_idx,
                            "joint_source": "unknown",
                            "mpjpe_source": "unknown",
                            "npz_path": npz_path,
                            **summary,
                        }
                if row is not None:
                    rows.append(row)
                skipped += 1
                print("  SKIP: output exists (use --force to recompute)")
                continue

            try:
                clip_person = load_person_data(
                    clip_id=clip_id,
                    person_idx=person_idx,
                    input_mode=args.input,
                    data_dir=data_dir,
                    raw_persons=raw_persons,
                )
                if clip_person is None:
                    missing += 1
                    print("  MISSING: no trajectory for this clip/person")
                    continue

                raw_person = (
                    raw_persons[person_idx]
                    if raw_persons is not None and person_idx < len(raw_persons)
                    else None
                )
                metrics = compute_metrics_for_person(
                    clip_data=clip_person,
                    fps=args.fps,
                    downsample_factor=args.downsample,
                    diff_method=args.diff_method,
                    device=args.gpu,
                    data_dir=data_dir,
                    raw_person_data=raw_person,
                    smplx_fk=smplx_fk,
                    disable_mpjpe=args.disable_mpjpe,
                    ground_fix=args.ground_fix,
                    ground_clearance=args.ground_clearance,
                    trim_edges=args.trim_edges,
                    balance_hold_frames=args.balance_hold_frames,
                    balance_transition_frames=args.balance_transition_frames,
                    balance_knee_bend_deg=args.balance_knee_bend_deg,
                    sphere_feet=args.sphere_feet,
                    sphere_feet_cache_dir=sphere_feet_cache_dir,
                    contact_soften=args.contact_soften,
                    contact_soften_solimp=contact_soften_solimp,
                    contact_soften_solref=contact_soften_solref,
                    contact_soften_margin=args.contact_soften_margin,
                    contact_soften_foot_only=args.contact_soften_foot_only,
                    foot_flatten=args.foot_flatten,
                    foot_flatten_height_m=args.foot_flatten_height,
                    foot_flatten_ankle_alpha=foot_flatten_ankle_alpha,
                    foot_flatten_knee_comp=foot_flatten_knee_comp,
                )
                _, _, row = save_person_outputs(
                    clip_id=clip_id,
                    person_idx=person_idx,
                    output_dir=output_dir,
                    metrics=metrics,
                )
                rows.append(row)
                processed += 1
                print(
                    "  DONE: "
                    f"T={row['n_frames']} "
                    f"seq=[{row.get('sequence_frame_start', 0)},"
                    f"{row.get('sequence_frame_end_exclusive', row['n_frames'])}) "
                    f"report=[{row.get('report_frame_start', 0)},"
                    f"{row.get('report_frame_end_exclusive', row['n_frames'])}) "
                    f"Residual={row['skyhook_residual_force_mean_N']:.3f} N "
                    f"MPJPE={row['mpjpe_mean_m']:.4f} m "
                    f"dz={row.get('ground_offset_m', 0.0):.4f} m "
                    f"flat(L/R)={row.get('foot_flatten_frames_left', 0)}/"
                    f"{row.get('foot_flatten_frames_right', 0)} "
                    f"(joint={row['joint_source']}, ref={row['mpjpe_source']})"
                )

            except Exception as exc:  # pragma: no cover - runtime dependent
                errors += 1
                print(f"  ERROR: {exc}")
                traceback.print_exc()

    # Write summary CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    fieldnames = [
        "clip_id",
        "person_idx",
        "n_frames",
        "n_report_frames",
        "report_frame_start",
        "report_frame_end_exclusive",
        "sequence_frame_start",
        "sequence_frame_end_exclusive",
        "joint_source",
        "mpjpe_source",
        "trim_edges",
        "balance_hold_frames",
        "balance_transition_frames",
        "balance_knee_bend_deg",
        "seed_min_foot_z_before_m",
        "seed_min_foot_z_after_m",
        "ground_fix",
        "ground_clearance_m",
        "ground_offset_m",
        "min_foot_z_before_m",
        "min_foot_z_after_m",
        "sphere_feet",
        "sphere_feet_cache_dir",
        "contact_soften",
        "contact_soften_solimp",
        "contact_soften_solref",
        "contact_soften_margin_m",
        "contact_soften_foot_only",
        "foot_flatten",
        "foot_flatten_height_m",
        "foot_flatten_ankle_alpha",
        "foot_flatten_knee_comp",
        "foot_flatten_frames_left",
        "foot_flatten_frames_right",
        "skyhook_residual_force_mean_N",
        "skyhook_residual_force_std_N",
        "skyhook_residual_force_max_N",
        "skyhook_residual_torque_mean_Nm",
        "skyhook_residual_wrench_mean",
        "mpjpe_mean_m",
        "mpjpe_valid",
        "skyhook_residual_force_mean_full_N",
        "skyhook_residual_force_max_full_N",
        "mpjpe_mean_full_m",
        "mpjpe_valid_full",
        "npz_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Aggregate summary
    elapsed = time.time() - t_start
    force_vals = np.array(
        [r["skyhook_residual_force_mean_N"] for r in rows], dtype=np.float64
    ) if rows else np.array([], dtype=np.float64)
    mpjpe_vals = np.array(
        [r["mpjpe_mean_m"] for r in rows if bool(r.get("mpjpe_valid"))],
        dtype=np.float64,
    ) if rows else np.array([], dtype=np.float64)

    aggregate = {
        "dataset": args.dataset,
        "input": args.input,
        "processed": processed,
        "skipped": skipped,
        "missing": missing,
        "errors": errors,
        "total_tasks": total_tasks,
        "elapsed_sec": elapsed,
        "summary_csv": csv_path,
        "force_residual_mean_N": (
            float(np.mean(force_vals)) if force_vals.size else float("nan")
        ),
        "force_residual_median_N": (
            float(np.median(force_vals)) if force_vals.size else float("nan")
        ),
        "mpjpe_mean_m": (
            float(np.mean(mpjpe_vals)) if mpjpe_vals.size else float("nan")
        ),
        "mpjpe_median_m": (
            float(np.median(mpjpe_vals)) if mpjpe_vals.size else float("nan")
        ),
    }
    if smplx_fk is not None and not smplx_fk.available:
        aggregate["smplx_fk_error"] = smplx_fk.init_error

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*68}")
    print("Skyhook Metric Pass Complete")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Missing:   {missing}")
    print(f"  Errors:    {errors}")
    print(f"  Time:      {elapsed/60.0:.1f} min ({elapsed/3600.0:.2f} h)")
    if force_vals.size:
        print(f"  Mean Residual (N): {np.mean(force_vals):.4f}")
    if mpjpe_vals.size:
        print(f"  Mean MPJPE (m):    {np.mean(mpjpe_vals):.6f}")
    print(f"  Output:    {output_dir}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
