# tools/train_calibrator.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib

try:
    import xgboost as xgb
except Exception:
    xgb = None

# 复用你工程里的 feature 协议（训练/推理必须一致）
from src.reward_calibrator import build_feature_dict, DEFAULT_FEATURE_KEYS, vectorize_feature_dict


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_results(run_dir: Path) -> List[Path]:
    if run_dir.is_file() and run_dir.suffix.lower() == ".jsonl":
        return [run_dir]
    files = sorted(run_dir.rglob("results.jsonl"))
    return files


def to_xy(rows: List[Dict[str, Any]], task_spec: Dict[str, Any], feature_keys: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for r in rows:
        prompt = r.get("prompt", "") or ""
        metrics = r.get("metrics", {}) or {}
        tags = r.get("tags", []) or []

        # penalties: 用 tags 恢复一个“近似 penalty summary”
        # （训练时你也可以改成直接读 penalties 字段，若你未来存进去）
        penalties = {}
        for t in tags:
            if t.endswith("_penalty"):
                penalties[t] = 0.10

        feat = build_feature_dict(prompt=prompt, metrics=metrics, penalties=penalties, task_spec=task_spec)
        x = vectorize_feature_dict(feat, feature_keys).reshape(-1)
        y = float(r.get("score", 0.0))  # target 默认用你 objective 分数（可替换成人工/PM评分）
        xs.append(x)
        ys.append(y)

    return np.stack(xs, axis=0), np.array(ys, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="runs", help="runs 目录，或某个 results.jsonl 文件路径")
    ap.add_argument("--spec", type=str, default="configs/miaotu_avatar.yaml", help="task spec yaml，用于保持特征一致")
    ap.add_argument("--model", type=str, choices=["lr", "xgb"], default="lr")
    ap.add_argument("--out", type=str, default="models/reward_calibrator.joblib")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读 spec
    from src.task_spec import TaskSpec
    task_spec = TaskSpec.load(args.spec).raw

    files = collect_results(run_dir)
    if not files:
        raise FileNotFoundError(f"No results.jsonl found under: {run_dir}")

    all_rows: List[Dict[str, Any]] = []
    for f in files:
        all_rows.extend(read_jsonl(f))

    if len(all_rows) < 20:
        print(f"[WARN] data too small: {len(all_rows)} rows. Calibration will be weak (but runnable).")

    feature_keys = DEFAULT_FEATURE_KEYS[:]  # 固定协议（训练=推理）
    X, y = to_xy(all_rows, task_spec=task_spec, feature_keys=feature_keys)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    if args.model == "lr":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=args.seed)),
        ])
    else:
        if xgb is None:
            raise RuntimeError("xgboost not installed. pip install xgboost")
        model = xgb.XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=args.seed,
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    print("=== Reward Calibration Training ===")
    print(f"rows: {len(all_rows)}")
    print(f"model: {args.model}")
    print(f"MSE: {mse:.6f}")

    # 保存
    joblib.dump(model, str(out_path))
    meta = {
        "feature_keys": feature_keys,
        "target": "score",
        "model": args.model,
        "num_rows": int(len(all_rows)),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved: {out_path}")
    print(f"[OK] saved: {out_path.with_suffix('.meta.json')}")
    print("\nNow enable calibration in your yaml:")
    print("reward_calibration:\n  enabled: true\n  model_path: models/reward_calibrator.joblib\n  alpha: 0.65")


if __name__ == "__main__":
    main()
