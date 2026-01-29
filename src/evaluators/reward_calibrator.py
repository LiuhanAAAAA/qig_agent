# src/evaluators/reward_calibrator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import glob
import os

import numpy as np

try:
    import joblib
except Exception:
    joblib = None


FEATURE_KEYS = [
    "clip_alignment",
    "sharpness",
    "aesthetic",
    "has_text",
    "has_face",
]


def featurize_metrics(metrics: Dict[str, Any]) -> np.ndarray:
    """
    metrics -> feature vector
    bool 特征转 0/1
    """
    feat = []
    for k in FEATURE_KEYS:
        v = metrics.get(k, 0.0)
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        feat.append(float(v))
    return np.array(feat, dtype=np.float32)


@dataclass
class RewardCalibrator:
    model: Any

    @staticmethod
    def load(path: str) -> "RewardCalibrator":
        if joblib is None:
            raise RuntimeError("joblib not available, please pip install joblib")
        m = joblib.load(path)
        return RewardCalibrator(model=m)

    def predict_proba_from_metrics(self, metrics: Dict[str, Any]) -> float:
        x = featurize_metrics(metrics).reshape(1, -1)
        proba = self.model.predict_proba(x)[0, 1]
        return float(proba)


def _find_results_files(runs_dir: str) -> List[str]:
    pattern = os.path.join(runs_dir, "**", "results.jsonl")
    files = glob.glob(pattern, recursive=True)
    return [f for f in files if os.path.isfile(f) and os.path.getsize(f) > 0]


def build_training_data_from_runs(runs_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 runs/**/results.jsonl 构建 (X, y)
    y 优先用 accepted 字段；
    如果没有 accepted，则用 (hard_fail==False and score>=image_min_score) 推断。
    """
    files = _find_results_files(runs_dir)
    if not files:
        raise RuntimeError(f"No training data found under {runs_dir} (need runs/**/results.jsonl)")

    X, y = [], []
    for fp in files:
        # 直接用spec
        default_min_score = 0.68

        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                metrics = row.get("metrics", {})
                if not metrics:
                    continue

                # label 逻辑
                if "accepted" in row:
                    label = 1 if bool(row["accepted"]) else 0
                else:
                    hard_fail = bool(row.get("hard_fail", False))
                    score = float(row.get("score", 0.0))
                    # 如果 jsonl 里带了阈值，就用；否则用默认
                    min_score = float(row.get("image_min_score", default_min_score))
                    label = 1 if (not hard_fail and score >= min_score) else 0

                X.append(featurize_metrics(metrics))
                y.append(label)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y
