"""
provenance.py — Output metadata tagging for traceability.

Every data output should include a provenance record so it's clear
which prepare folder / script / parameters generated it.

Usage:
    from prepare_utils.provenance import save_with_provenance, load_with_provenance

    # Save numpy array with metadata sidecar
    save_with_provenance(
        "data/compute_torques/interhuman/1000_person0_torques_pd.npy",
        torques,
        source="prepare2/compute_torques.py",
        params={"fps": 30, "method": "pd", "gain_scale": 1.0},
    )

    # Load and check provenance
    data, meta = load_with_provenance(
        "data/compute_torques/interhuman/1000_person0_torques_pd.npy"
    )
"""
import os
import json
import datetime
import numpy as np


def _meta_path(data_path):
    """Get sidecar metadata path for a given data file."""
    base, ext = os.path.splitext(data_path)
    return base + "_provenance.json"


def save_provenance(data_path, source, params=None, extra=None):
    """Save a provenance sidecar JSON alongside a data file.

    Args:
        data_path: path to the data file (npy, npz, json, pkl, etc.)
        source: which script generated this (e.g., "prepare4/retarget.py")
        params: dict of parameters used (fps, method, clip_id, etc.)
        extra: any additional metadata dict
    """
    meta = {
        "source": source,
        "timestamp": datetime.datetime.now().isoformat(),
        "data_file": os.path.basename(data_path),
    }
    if params:
        meta["params"] = params
    if extra:
        meta.update(extra)

    meta_path = _meta_path(data_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def save_with_provenance(data_path, data, source, params=None, extra=None):
    """Save numpy data + provenance sidecar in one call.

    Args:
        data_path: output path (.npy or .npz)
        data: numpy array or dict of arrays (for npz)
        source: script that produced this
        params: parameter dict
        extra: additional metadata
    """
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    if data_path.endswith(".npz"):
        if isinstance(data, dict):
            np.savez(data_path, **data)
        else:
            np.savez(data_path, data=data)
    else:
        np.save(data_path, data)

    save_provenance(data_path, source, params, extra)


def load_with_provenance(data_path):
    """Load data + its provenance metadata if available.

    Returns:
        (data, meta) where meta is None if no provenance file exists.
    """
    if data_path.endswith(".npz"):
        data = np.load(data_path)
    else:
        data = np.load(data_path)

    meta_path = _meta_path(data_path)
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    return data, meta


def check_provenance(data_path, expected_source=None, expected_params=None):
    """Validate provenance of a data file.

    Args:
        data_path: path to data file
        expected_source: if set, warn if source doesn't match
        expected_params: dict of param keys to validate (e.g., {"fps": 30})

    Returns:
        (ok, warnings) where ok is bool and warnings is list of strings
    """
    meta_path = _meta_path(data_path)
    warnings_list = []

    if not os.path.exists(meta_path):
        warnings_list.append(
            f"No provenance file for {os.path.basename(data_path)}. "
            f"Cannot verify origin or parameters."
        )
        return False, warnings_list

    with open(meta_path, "r") as f:
        meta = json.load(f)

    if expected_source and meta.get("source") != expected_source:
        warnings_list.append(
            f"Source mismatch: expected '{expected_source}', "
            f"got '{meta.get('source')}'"
        )

    if expected_params and "params" in meta:
        for key, expected_val in expected_params.items():
            actual_val = meta["params"].get(key)
            if actual_val is not None and actual_val != expected_val:
                warnings_list.append(
                    f"Param '{key}' mismatch: expected {expected_val}, "
                    f"got {actual_val}"
                )

    return len(warnings_list) == 0, warnings_list
