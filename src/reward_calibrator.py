# src/reward_calibrator.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
import json

import numpy as np

try:
    import joblib
except Exception:
    joblib = None


DEFAULT_FEATURE_KEYS = [
    # core metrics
    "clip_alignment",
    "sharpness",
    "aesthetic",
    # constraint signals
    "has_text",
    "has_face",
    # penalty summary
    "penalty_sum",
    "num_penalties",
    # prompt stats
    "prompt_len",
    "neg_len",
]


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def build_feature_dict(
    prompt: str,
    metrics: Dict[str, Any],
    penalties: Dict[str, float],
    task_spec: Dict[str, Any],
) -> Dict[str, float]:
    """
    把 (prompt, metrics, penalties) 变成可训练/可推理的特征字典。
    ✅ 这个函数是 Reward Calibration 的“协议”，训练和推理必须一致。
    """
    prompt = prompt or ""
    base_style = task_spec.get("prompt_policy", {}).get("base_style", "") or ""
    negative_prompt = task_spec.get("prompt_policy", {}).get("negative_prompt", "") or ""

    # metrics
    clip_alignment = _safe_float(metrics.get("clip_alignment", 0.0))
    sharpness = _safe_float(metrics.get("sharpness", 0.0))
    aesthetic = _safe_float(metrics.get("aesthetic", 0.0))

    has_text = 1.0 if bool(metrics.get("has_text", False)) else 0.0
    has_face = 1.0 if bool(metrics.get("has_face", False)) else 0.0

    penalty_sum = float(sum(max(0.0, _safe_float(v)) for v in (penalties or {}).values()))
    num_penalties = float(sum(1 for v in (penalties or {}).values() if _safe_float(v) > 0))

    # prompt stats
    prompt_len = float(len(prompt))
    neg_len = float(len(negative_prompt))

    feat = {
        "clip_alignment": clip_alignment,
        "sharpness": sharpness,
        "aesthetic": aesthetic,
        "has_text": has_text,
        "has_face": has_face,
        "penalty_sum": penalty_sum,
        "num_penalties": num_penalties,
        "prompt_len": prompt_len,
        "neg_len": neg_len,
    }

    # 可以扩展一些上下文特征（不影响兼容）
    feat["base_style_len"] = float(len(base_style))
    return feat


def vectorize_feature_dict(feat: Dict[str, float], feature_keys: List[str]) -> np.ndarray:
    return np.array([_safe_float(feat.get(k, 0.0)) for k in feature_keys], dtype=np.float32).reshape(1, -1)


@dataclass
class RewardCalibrator:
    """
    Reward Calibration 模型：把 raw metrics -> calibrated score
    训练脚本会输出：
      - reward_calibrator.joblib
      - reward_calibrator.meta.json
    """
    model: Any
    feature_keys: List[str]

    @staticmethod
    def load(model_path: str | Path) -> Optional["RewardCalibrator"]:
        model_path = Path(model_path)
        meta_path = model_path.with_suffix(".meta.json")
        if not model_path.exists():
            return None
        if joblib is None:
            raise RuntimeError("joblib not installed, cannot load calibrator. pip install joblib")

        feature_keys = DEFAULT_FEATURE_KEYS
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                feature_keys = meta.get("feature_keys", feature_keys)
            except Exception:
                pass

        model = joblib.load(str(model_path))
        return RewardCalibrator(model=model, feature_keys=feature_keys)

    def predict(self, prompt: str, metrics: Dict[str, Any], penalties: Dict[str, float], task_spec: Dict[str, Any]) -> float:
        feat = build_feature_dict(prompt, metrics, penalties, task_spec)
        x = vectorize_feature_dict(feat, self.feature_keys)

        # regression model => float
        y = self.model.predict(x)
        try:
            y = float(y[0])
        except Exception:
            y = float(y)

        # score clamp 到 [0,1]，避免爆炸
        if np.isnan(y) or np.isinf(y):
            y = 0.0
        return max(0.0, min(1.0, float(y)))
