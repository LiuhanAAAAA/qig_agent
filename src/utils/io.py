import json
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


# =========================
# Run dir utils
# =========================

def make_run_dir(base_dir: str = "runs") -> str:
    """
    创建本次运行的输出目录，例如 runs/20260115_214720
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(base_dir) / ts
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# =========================
# JSONL safe writer
# =========================

def _to_jsonable(x: Any) -> Any:
    """
    把 numpy / torch 的标量转成 Python 原生类型，保证 json.dumps 不炸
    """
    # numpy scalar
    try:
        import numpy as np
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
    except Exception:
        pass

    # torch scalar / tensor
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return _to_jsonable(x.item())
            return x.detach().cpu().tolist()
    except Exception:
        pass

    # 原生类型
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    # dict/list 递归
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]

    # 最后兜底：转字符串（避免炸）
    return str(x)


def append_jsonl(path: str, row: Dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    safe_row = _to_jsonable(row)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_text(path: str, text: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
