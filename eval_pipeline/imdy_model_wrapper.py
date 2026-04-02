"""ImDy model loading and inference wrapper for evaluation scripts."""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict

import numpy as np


class _CfgNode(dict):
    """Minimal attribute-access config node for YAML fallback."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @classmethod
    def from_obj(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            node = cls()
            for k, v in obj.items():
                node[k] = cls.from_obj(v)
            return node
        if isinstance(obj, list):
            return [cls.from_obj(v) for v in obj]
        return obj


class ImDyWrapper:
    """Load ImDy checkpoint and run batched inference on marker windows."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        imdy_root: str | None = None,
        use_contact_mask: bool = True,
    ) -> None:
        self.config_path = os.path.abspath(config_path)
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.imdy_root = os.path.abspath(
            imdy_root
            if imdy_root is not None
            else os.path.join(os.path.dirname(__file__), "..", "prepare5", "ImDy")
        )
        self.use_contact_mask = use_contact_mask

        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"ImDy config not found: {self.config_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"ImDy checkpoint not found: {self.checkpoint_path}")
        if not os.path.isdir(self.imdy_root):
            raise FileNotFoundError(f"ImDy root not found: {self.imdy_root}")

        if self.imdy_root not in sys.path:
            sys.path.insert(0, self.imdy_root)

        # ImDy imports OmegaConf in dataset.py for typing/load utilities.
        # Provide a lightweight fallback module when omegaconf is absent.
        if "omegaconf" not in sys.modules:
            try:
                import omegaconf  # noqa: F401
            except Exception:
                fake_omegaconf = types.ModuleType("omegaconf")

                class _DummyOmegaConf:  # pragma: no cover - compatibility shim
                    pass

                fake_omegaconf.OmegaConf = _DummyOmegaConf
                sys.modules["omegaconf"] = fake_omegaconf

        try:
            import torch
            from models import get_model
            from models.utils import joint_torque_limits
        except Exception as exc:
            raise RuntimeError(
                "Failed to import ImDy runtime dependencies. "
                "Use an environment with torch and ImDy model dependencies installed."
            ) from exc

        self.torch = torch

        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(self.config_path)
        except Exception:
            # Fallback for environments that have torch but not omegaconf.
            try:
                import yaml
            except Exception as exc:
                raise RuntimeError(
                    "Need either omegaconf or pyyaml to read ImDy config YAML"
                ) from exc
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = _CfgNode.from_obj(yaml.safe_load(f))

        self.config = self._sync_config(cfg)

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = requested_device

        self.model = get_model(self.config).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        state_dict = self._strip_module_prefix(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.joint_torque_limits = np.asarray(joint_torque_limits, dtype=np.float32)
        self.past_kf = int(self.config.get("PAST_KF", 2))
        self.fut_kf = int(self.config.get("FUTURE_KF", 2))

    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state_dict, dict):
            return state_dict
        if not any(k.startswith("module.") for k in state_dict.keys()):
            return state_dict
        return {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    @staticmethod
    def _sync_config(config: Any) -> Any:
        # Mirror the sync logic used in prepare5/ImDy/main.py.
        config.TRAIN.PAST_KF = config.get("PAST_KF", 2)
        config.TRAIN.FUTURE_KF = config.get("FUTURE_KF", 2)
        config.TRAIN.rot_rep = config.get("rot_rep", "quat")
        config.TRAIN.use_norm = config.get("use_norm", False)

        config.TEST.PAST_KF = config.get("PAST_KF", 2)
        config.TEST.FUTURE_KF = config.get("FUTURE_KF", 2)
        config.TEST.rot_rep = config.get("rot_rep", "quat")
        config.TEST.use_norm = config.get("use_norm", False)

        config.MODEL.PAST_KF = config.get("PAST_KF", 2)
        config.MODEL.FUTURE_KF = config.get("FUTURE_KF", 2)
        config.MODEL.rot_rep = config.get("rot_rep", "quat")
        config.MODEL.use_norm = config.get("use_norm", False)

        config.DATASET.TRAIN.PAST_KF = config.get("PAST_KF", 2)
        config.DATASET.TRAIN.FUTURE_KF = config.get("FUTURE_KF", 2)
        config.DATASET.TRAIN.rot_rep = config.get("rot_rep", "quat")
        config.DATASET.TRAIN.use_norm = config.get("use_norm", False)

        config.DATASET.TEST.PAST_KF = config.get("PAST_KF", 2)
        config.DATASET.TEST.FUTURE_KF = config.get("FUTURE_KF", 2)
        config.DATASET.TEST.rot_rep = config.get("rot_rep", "quat")
        config.DATASET.TEST.use_norm = config.get("use_norm", False)

        return config

    def _to_tensor(self, arr: np.ndarray):
        t = self.torch.as_tensor(arr, dtype=self.torch.float32)
        return t.to(self.device)

    @staticmethod
    def _maybe_squeeze_time_axis(arr: np.ndarray) -> np.ndarray:
        if arr.ndim >= 2 and arr.shape[1] == 1:
            return arr[:, 0]
        return arr

    def predict(self, mkr: np.ndarray, mvel: np.ndarray) -> Dict[str, np.ndarray]:
        """Run one forward pass.

        Args:
            mkr: (B, M, L, 3)
            mvel: (B, M, L, 3)

        Returns:
            Dictionary containing torque/grf/contact and raw head outputs.
        """
        mkr_t = self._to_tensor(mkr)
        mvel_t = self._to_tensor(mvel)

        if mkr_t.ndim != 4 or mvel_t.ndim != 4:
            raise ValueError(
                f"Expected mkr/mvel to be 4D, got {tuple(mkr_t.shape)}, {tuple(mvel_t.shape)}"
            )

        with self.torch.no_grad():
            output = self.model({"mkr": mkr_t, "mvel": mvel_t})

        if "torque" not in output:
            output["torque"] = output["torvec"] * output["tornorm"]
        if "grf" not in output:
            output["grf"] = output["grfvec"] * output["grfnorm"]

        out_np: Dict[str, np.ndarray] = {}
        for key in ["torque", "grf", "contact", "torvec", "tornorm", "grfvec", "grfnorm"]:
            if key in output and output[key] is not None:
                out_np[key] = output[key].detach().cpu().numpy()

        if "torque" in out_np:
            out_np["torque"] = self._maybe_squeeze_time_axis(out_np["torque"])

        if self.use_contact_mask and "contact" in out_np and "grf" in out_np:
            # contact output is raw logits (no sigmoid); logit > 0 = probability > 0.5
            contact_mask = (out_np["contact"] > 0.0).astype(np.float32)
            out_np["grf_masked"] = out_np["grf"] * contact_mask

        return out_np

    def predict_clip(
        self,
        mkr_windows: np.ndarray,
        mvel_windows: np.ndarray,
        batch_size: int = 256,
    ) -> Dict[str, np.ndarray]:
        """Run inference for an entire clip by batching windows."""
        if len(mkr_windows) != len(mvel_windows):
            raise ValueError(
                "mkr_windows and mvel_windows must have the same length, got "
                f"{len(mkr_windows)} and {len(mvel_windows)}"
            )

        if len(mkr_windows) == 0:
            return {}

        merged: Dict[str, list[np.ndarray]] = {}
        for s in range(0, len(mkr_windows), batch_size):
            e = min(s + batch_size, len(mkr_windows))
            batch_out = self.predict(mkr_windows[s:e], mvel_windows[s:e])
            for key, value in batch_out.items():
                merged.setdefault(key, []).append(value)

        return {k: np.concatenate(v, axis=0) for k, v in merged.items()}
