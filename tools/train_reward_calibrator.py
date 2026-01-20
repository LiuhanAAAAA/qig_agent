# tools/train_reward_calibrator.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def _iter_results_jsonl(runs_dir: str) -> List[Dict[str, Any]]:
    runs = Path(runs_dir)
    if not runs.exists():
        return []

    items: List[Dict[str, Any]] = []
    for p in runs.glob("**/results.jsonl"):
        if not p.is_file():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    return items


def _extract_features(row: Dict[str, Any]) -> Tuple[List[float], float]:
    """
    X: features
    y_score: continuous target (for pseudo-label / debug)
    """
    metrics = row.get("metrics") or {}
    # 常见字段（没有就给默认）
    clip = float(metrics.get("clip_alignment", row.get("clip_alignment", 0.0)) or 0.0)
    sharp = float(metrics.get("sharpness", row.get("sharpness", 0.0)) or 0.0)
    aest = float(metrics.get("aesthetic", row.get("aesthetic", 0.0)) or 0.0)

    has_text = 1.0 if bool(metrics.get("has_text", row.get("has_text", False))) else 0.0
    has_face = 1.0 if bool(metrics.get("has_face", row.get("has_face", False))) else 0.0
    hard_fail = 1.0 if bool(row.get("hard_fail", False)) else 0.0

    # penalties（可选）
    penalties = row.get("penalties") or {}
    # 如果 penalties 没写，就从 tags 粗略推断
    pen_sum = 0.0
    if isinstance(penalties, dict):
        pen_sum = float(sum(float(v) for v in penalties.values() if isinstance(v, (int, float))))
    else:
        pen_sum = 0.0

    # total score（你系统输出的总分）
    total = float(row.get("score", row.get("total_score", 0.0)) or 0.0)

    feat = [
        clip, sharp, aest,
        has_text, has_face,
        hard_fail,
        pen_sum,
        total,                 # 把原系统 score 也喂进去（非常强）
    ]
    return feat, total


def build_training_data_from_runs(runs_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rows = _iter_results_jsonl(runs_dir)
    if not rows:
        raise RuntimeError(f"No training data found under {runs_dir} (need runs/**/results.jsonl)")

    X_list: List[List[float]] = []
    score_list: List[float] = []
    y_list: List[int] = []

    # ✅ 优先使用 “accepted” 字段作为真标签（如果你后面写入了）
    for r in rows:
        feat, total = _extract_features(r)
        X_list.append(feat)
        score_list.append(total)

        if "accepted" in r:
            y_list.append(1 if bool(r["accepted"]) else 0)
        else:
            # 临时：先按 total>=0.75 当正样本（很可能导致全 0，所以后面会兜底）
            y_list.append(1 if total >= 0.75 and (not bool(r.get("hard_fail", False))) else 0)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    scores = np.asarray(score_list, dtype=np.float32)

    info = {
        "n": int(len(y)),
        "y_pos_rate": float(y.mean() if len(y) else 0.0),
        "score_min": float(scores.min() if len(scores) else 0.0),
        "score_max": float(scores.max() if len(scores) else 0.0),
    }

    # ✅ 如果只有一个类别：自动构造伪标签（top 30% = 1）
    uniq = np.unique(y)
    if len(uniq) < 2:
        # score 全一样也救不了，只能报错
        if float(scores.max() - scores.min()) < 1e-6:
            raise RuntimeError(
                "Training labels are single-class and scores are constant. "
                "Need more diverse runs or write `accepted` labels into results.jsonl."
            )

        q = float(np.quantile(scores, 0.70))
        y = (scores >= q).astype(np.int64)
        info["pseudo_label"] = True
        info["pseudo_quantile"] = q
        info["y_pos_rate_after_pseudo"] = float(y.mean())
    else:
        info["pseudo_label"] = False

    return X, y, info


def train_lr(X: np.ndarray, y: np.ndarray):
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X, y)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--method", type=str, default="lr", choices=["lr"])
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    X, y, info = build_training_data_from_runs(args.runs_dir)
    print(f"[OK] Loaded training data: X={X.shape}, y_pos_rate={info['y_pos_rate']:.3f}")

    model = train_lr(X, y)

    # quick metrics
    try:
        prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, prob)
        pred = (prob >= 0.5).astype(np.int64)
        acc = accuracy_score(y, pred)
        print(f"[INFO] Train-acc={acc:.4f}  AUC={auc:.4f}")
    except Exception:
        pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "type": "reward_calibrator",
        "method": args.method,
        "model": model,
        "info": info,
        "feature_dim": int(X.shape[1]),
        "feature_names": [
            "clip_alignment",
            "sharpness",
            "aesthetic",
            "has_text",
            "has_face",
            "hard_fail",
            "penalty_sum",
            "base_total_score",
        ],
    }
    joblib.dump(payload, out_path)
    print(f"[OK] Saved calibrator -> {out_path}")


if __name__ == "__main__":
    main()
