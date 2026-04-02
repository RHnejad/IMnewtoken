"""Physics plausibility metrics for ImDy predictions."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

DEFAULT_G = 9.81
DEFAULT_WEIGHT_KG = 75.0
DEFAULT_VERTICAL_AXIS = 2  # Z-up in this project
DEFAULT_FOOT_JOINTS = (7, 8, 10, 11)
DEFAULT_TORQUE_LIMITS = np.array(
    [
        2500.0,
        2500.0,
        2500.0,
        2600.0,
        2600.0,
        2600.0,
        2500.0,
        2500.0,
        2500.0,
        2000.0,
        2000.0,
        1500.0,
        2000.0,
        2000.0,
        1500.0,
        2000.0,
        2000.0,
        2000.0,
        2000.0,
        1500.0,
        1500.0,
        1200.0,
        1200.0,
    ],
    dtype=np.float32,
)


def _to_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _safe_mean(x: np.ndarray) -> float:
    x = _finite(_to_float_array(x).reshape(-1))
    return float(np.mean(x)) if x.size > 0 else float("nan")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_float_array(a).reshape(-1)
    b = _to_float_array(b).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _torque_magnitude(torque: np.ndarray) -> np.ndarray:
    return np.linalg.norm(_to_float_array(torque), axis=-1)


def _grf_vertical(grf: np.ndarray, vertical_axis: int) -> np.ndarray:
    return _to_float_array(grf)[..., vertical_axis]


def _canonicalize_torque(torque: np.ndarray) -> np.ndarray:
    tor = _to_float_array(torque)
    if tor.ndim == 4 and tor.shape[1] == 1:
        tor = tor[:, 0]
    if tor.ndim != 3 or tor.shape[-1] != 3:
        raise ValueError(f"Expected torque shape (N, 23, 3), got {tor.shape}")
    return tor


def _canonicalize_grf(grf: np.ndarray) -> np.ndarray:
    arr = _to_float_array(grf)
    if arr.ndim != 4 or arr.shape[1] != 2 or arr.shape[-1] != 3:
        raise ValueError(f"Expected grf shape (N, 2, 24, 3), got {arr.shape}")
    return arr


def compute_person_metrics(
    torque: np.ndarray,
    grf: np.ndarray,
    positions_center: np.ndarray | None = None,
    *,
    body_weight_kg: float = DEFAULT_WEIGHT_KG,
    g: float = DEFAULT_G,
    joint_torque_limits: np.ndarray = DEFAULT_TORQUE_LIMITS,
    vertical_axis: int = DEFAULT_VERTICAL_AXIS,
    foot_joints: Tuple[int, ...] = DEFAULT_FOOT_JOINTS,
) -> Dict[str, float]:
    """Compute per-person plausibility metrics from ImDy outputs."""
    tor = _canonicalize_torque(torque)
    grf_arr = _canonicalize_grf(grf)

    tor_mag = _torque_magnitude(tor)  # (N, 23)
    bw_force = max(body_weight_kg * g, 1e-8)

    limits = _to_float_array(joint_torque_limits).reshape(1, -1)
    if limits.shape[1] != tor_mag.shape[1]:
        raise ValueError(
            f"joint_torque_limits has {limits.shape[1]} entries, expected {tor_mag.shape[1]}"
        )

    vertical = _grf_vertical(grf_arr, vertical_axis)  # (N, 2, 24)
    vertical_now = vertical[:, 0]

    metrics: Dict[str, float] = {
        "mean_torque_bw": float(np.mean(tor_mag) / bw_force),
        "torque_95th": float(np.percentile(tor_mag, 95)),
        "torque_violation_rate": float(np.mean(tor_mag > limits)),
        "grf_negative_rate": float(np.mean(vertical < 0.0)),
        "mean_vertical_grf": float(np.mean(vertical_now)),
        "peak_vertical_grf": float(np.percentile(vertical_now, 95)),
        "contact_rate": float(np.mean(vertical_now > 1e-3)),
    }

    if tor.shape[0] > 1:
        dtor = np.diff(tor, axis=0)
        metrics["torque_smoothness"] = float(np.mean(np.linalg.norm(dtor, axis=-1)))
    else:
        metrics["torque_smoothness"] = float("nan")

    if positions_center is not None:
        pos = _to_float_array(positions_center)
        if pos.ndim != 3 or pos.shape[-1] != 3:
            raise ValueError(
                f"Expected positions_center shape (N, J, 3), got {pos.shape}"
            )
        if pos.shape[0] != tor.shape[0]:
            raise ValueError(
                "positions_center and torque must have matching frame count, got "
                f"{pos.shape[0]} vs {tor.shape[0]}"
            )

        foot_height = np.mean(pos[:, foot_joints, vertical_axis], axis=1)
        foot_force = np.sum(np.clip(vertical_now[:, foot_joints], 0.0, None), axis=1)
        metrics["contact_height_consistency"] = _safe_corr(-foot_height, foot_force)
    else:
        metrics["contact_height_consistency"] = float("nan")

    return metrics


def compute_interaction_metrics(
    torque_person1: np.ndarray,
    torque_person2: np.ndarray,
    grf_person1: np.ndarray,
    grf_person2: np.ndarray,
    positions_person1: np.ndarray | None = None,
    positions_person2: np.ndarray | None = None,
    *,
    joint_torque_limits: np.ndarray = DEFAULT_TORQUE_LIMITS,
    vertical_axis: int = DEFAULT_VERTICAL_AXIS,
    proximity_threshold_m: float = 1.0,
) -> Dict[str, float]:
    """Compute pair-level metrics from two-person predictions."""
    tor1 = _canonicalize_torque(torque_person1)
    tor2 = _canonicalize_torque(torque_person2)
    grf1 = _canonicalize_grf(grf_person1)
    grf2 = _canonicalize_grf(grf_person2)

    n = min(tor1.shape[0], tor2.shape[0], grf1.shape[0], grf2.shape[0])
    tor1 = tor1[:n]
    tor2 = tor2[:n]
    grf1 = grf1[:n]
    grf2 = grf2[:n]

    t1_mag = _torque_magnitude(tor1)
    t2_mag = _torque_magnitude(tor2)

    limits = _to_float_array(joint_torque_limits).reshape(1, -1)
    v1 = float(np.mean(t1_mag > limits))
    v2 = float(np.mean(t2_mag > limits))

    m1 = float(np.mean(t1_mag))
    m2 = float(np.mean(t2_mag))

    metrics: Dict[str, float] = {
        "torque_asymmetry": float(m1 / (m2 + 1e-8)),
        "combined_violation_rate": float(0.5 * (v1 + v2)),
    }

    if positions_person1 is not None and positions_person2 is not None:
        p1 = _to_float_array(positions_person1)[:n]
        p2 = _to_float_array(positions_person2)[:n]
        if p1.ndim != 3 or p2.ndim != 3:
            raise ValueError("positions_person1/2 must be shaped (N, J, 3)")

        # Contact proxy: pelvis distance below threshold.
        pelvis_dist = np.linalg.norm(p1[:, 0] - p2[:, 0], axis=-1)
        close_mask = pelvis_dist < proximity_threshold_m
        metrics["close_frame_fraction"] = float(np.mean(close_mask))

        if np.any(close_mask):
            # Use horizontal GRF components and check force cancellation ratio.
            f1_xy = np.sum(grf1[:, 0, :, :2], axis=1)
            f2_xy = np.sum(grf2[:, 0, :, :2], axis=1)
            net_xy = np.linalg.norm(f1_xy + f2_xy, axis=-1)
            sum_xy = np.linalg.norm(f1_xy, axis=-1) + np.linalg.norm(f2_xy, axis=-1) + 1e-8
            metrics["n3l_violation_at_contact"] = float(np.mean((net_xy / sum_xy)[close_mask]))
        else:
            metrics["n3l_violation_at_contact"] = float("nan")
    else:
        metrics["close_frame_fraction"] = float("nan")
        metrics["n3l_violation_at_contact"] = float("nan")

    # Optional global-force balance signal across full body vertical GRFs.
    vgrf1 = _grf_vertical(grf1, vertical_axis)[:, 0].sum(axis=1)
    vgrf2 = _grf_vertical(grf2, vertical_axis)[:, 0].sum(axis=1)
    metrics["vertical_grf_balance"] = float(np.mean(np.abs(vgrf1 - vgrf2) / (np.abs(vgrf1) + np.abs(vgrf2) + 1e-8)))

    return metrics


def aggregate_metric_dicts(metric_dicts: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate clip-level scalar metrics into dataset-level summary stats."""
    buckets: Dict[str, List[float]] = {}
    for row in metric_dicts:
        for key, value in row.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                buckets.setdefault(key, []).append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for key, values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
    return summary


def _wasserstein_1d_manual(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(_finite(_to_float_array(a)))
    b = np.sort(_finite(_to_float_array(b)))
    if a.size == 0 or b.size == 0:
        return float("nan")

    values = np.sort(np.concatenate([a, b]))
    a_cdf = np.searchsorted(a, values, side="right") / a.size
    b_cdf = np.searchsorted(b, values, side="right") / b.size

    deltas = np.diff(values)
    if deltas.size == 0:
        return 0.0
    return float(np.sum(np.abs(a_cdf[:-1] - b_cdf[:-1]) * deltas))


def _ks_2samp_manual(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.sort(_finite(_to_float_array(a)))
    b = np.sort(_finite(_to_float_array(b)))
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")

    values = np.sort(np.concatenate([a, b]))
    a_cdf = np.searchsorted(a, values, side="right") / a.size
    b_cdf = np.searchsorted(b, values, side="right") / b.size
    d = float(np.max(np.abs(a_cdf - b_cdf)))

    # Asymptotic p-value approximation.
    en = np.sqrt((a.size * b.size) / (a.size + b.size))
    p = 2.0 * np.exp(-2.0 * (en * d) ** 2)
    p = float(np.clip(p, 0.0, 1.0))
    return d, p


def compare_metric_distributions(
    gt_metric_dicts: Iterable[Dict[str, float]],
    gen_metric_dicts: Iterable[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Compare GT and generated clip-level metric distributions."""
    gt_values: Dict[str, List[float]] = {}
    gen_values: Dict[str, List[float]] = {}

    for row in gt_metric_dicts:
        for k, v in row.items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                gt_values.setdefault(k, []).append(float(v))

    for row in gen_metric_dicts:
        for k, v in row.items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                gen_values.setdefault(k, []).append(float(v))

    common = sorted(set(gt_values.keys()) & set(gen_values.keys()))
    out: Dict[str, Dict[str, float]] = {}

    scipy_stats = None
    try:
        from scipy import stats as scipy_stats  # type: ignore
    except Exception:
        scipy_stats = None

    for key in common:
        g = np.asarray(gt_values[key], dtype=np.float64)
        p = np.asarray(gen_values[key], dtype=np.float64)

        if scipy_stats is not None:
            wasser = float(scipy_stats.wasserstein_distance(g, p))
            ks_res = scipy_stats.ks_2samp(g, p)
            ks_d = float(ks_res.statistic)
            ks_p = float(ks_res.pvalue)
        else:
            wasser = _wasserstein_1d_manual(g, p)
            ks_d, ks_p = _ks_2samp_manual(g, p)

        out[key] = {
            "gt_count": int(g.size),
            "gen_count": int(p.size),
            "gt_mean": float(np.mean(g)),
            "gen_mean": float(np.mean(p)),
            "mean_delta_gen_minus_gt": float(np.mean(p) - np.mean(g)),
            "wasserstein": wasser,
            "ks_statistic": ks_d,
            "ks_pvalue": ks_p,
        }

    return out
