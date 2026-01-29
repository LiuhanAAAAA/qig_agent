# src/evaluators/objective.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np


_CALIBRATOR_CACHE: Dict[str, Any] = {}


def _load_calibrator(model_path: str):
    p = str(Path(model_path))
    if p in _CALIBRATOR_CACHE:
        return _CALIBRATOR_CACHE[p]

    if not Path(p).exists():
        _CALIBRATOR_CACHE[p] = None
        return None

    try:
        obj = joblib.load(p)
    except Exception:
        _CALIBRATOR_CACHE[p] = None
        return None

    if isinstance(obj, dict) and "model" in obj:
        _CALIBRATOR_CACHE[p] = obj
        return obj

    # 兼容旧格式：直接保存 model
    _CALIBRATOR_CACHE[p] = {"model": obj, "feature_names": None, "info": {}}
    return _CALIBRATOR_CACHE[p]


def objective_score(
    task_spec: Dict[str, Any],
    clip: float,
    sharpness: float,
    aesthetic: float,
    penalties: Dict[str, float],
    tags: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    prompt: Optional[str] = None,   
    **kwargs,                      
) -> Tuple[float, List[str]]:
    """
    返回：
      total_score: float 0~1
      tags: list[str]
    """

    tags = list(tags or [])
    metrics = dict(metrics or {})

    clip = float(clip)
    sharpness = float(sharpness)
    aesthetic = float(aesthetic)

    # ---- base score ----
    w = (task_spec.get("soft_preferences") or {}).get("weights", {}) or {}
    w_clip = float(w.get("clip_alignment", 0.45))
    w_sharp = float(w.get("sharpness", 0.25))
    w_aest = float(w.get("aesthetic", 0.30))

    base = w_clip * clip + w_sharp * sharpness + w_aest * aesthetic

    # ---- penalty ----
    pen = 0.0
    if isinstance(penalties, dict):
        for k, v in penalties.items():
            if isinstance(v, (int, float)):
                pen += float(v)
                tags.append(str(k))

    total = float(max(0.0, min(1.0, base - pen)))

    # ---- reward calibration (optional) ----
    cal_cfg = (task_spec.get("reward_calibration") or {})
    enabled = bool(cal_cfg.get("enabled", False))
    model_path = str(cal_cfg.get("model_path", "configs/reward_calibrator.joblib"))

    if enabled:
        payload = _load_calibrator(model_path)
        if payload and payload.get("model") is not None:
            model = payload["model"]

            # feature vector order must match train script
            has_text = 1.0 if bool(metrics.get("has_text", False)) else 0.0
            has_face = 1.0 if bool(metrics.get("has_face", False)) else 0.0
            hard_fail = 1.0 if bool(metrics.get("hard_fail", False)) else 0.0
            pen_sum = float(sum(float(v) for v in penalties.values() if isinstance(v, (int, float)))) if isinstance(penalties, dict) else 0.0

            feat = np.asarray([[
                float(metrics.get("clip_alignment", clip)),
                float(metrics.get("sharpness", sharpness)),
                float(metrics.get("aesthetic", aesthetic)),
                has_text,
                has_face,
                hard_fail,
                pen_sum,
                float(total),
            ]], dtype=np.float32)

            try:
                if hasattr(model, "predict_proba"):
                    p1 = float(model.predict_proba(feat)[0, 1])
                    alpha = float(cal_cfg.get("alpha", 0.35))
                    total = float((1 - alpha) * total + alpha * p1)
                    tags.append("calibrated_lr")
                else:
                    pred = float(model.predict(feat)[0])
                    # fallback
                    alpha = float(cal_cfg.get("alpha", 0.35))
                    total = float((1 - alpha) * total + alpha * (1 / (1 + np.exp(-pred))))
                    tags.append("calibrated_generic")
            except Exception:
                tags.append("calibration_failed")

    total = float(max(0.0, min(1.0, total)))
    tags = sorted(list(set(tags)))
    return total, tags
