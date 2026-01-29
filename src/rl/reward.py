# src/rl/reward.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math


@dataclass
class RewardConfig:
    w_prompt: float = 0.35
    w_image: float = 0.65
    hard_fail_penalty: float = 1.5
    len_target: int = 80
    len_penalty: float = 0.02
    rep_penalty: float = 0.10


def length_penalty(token_len: int, len_target: int, alpha: float) -> float:
    """
    只惩罚“超过目标长度”的部分，避免鼓励模型极短 prompt。
    """
    if token_len <= len_target:
        return 0.0
    over = (token_len - len_target) / max(1.0, float(len_target))
    return alpha * over


def combine_reward(
    cfg: RewardConfig,
    prompt_score: float,
    image_score: float | None,
    token_len: int,
    rep_sim: float,
    hard_fail: bool,
) -> float:
    """
    multi-fidelity reward:
      - 未生图 image_score=None -> 只用 prompt_score
      - 生图 image_score!=None -> prompt+image

    最终 reward 越大越好，PPO 最大化
    """
    r = 0.0

    # 主奖励项
    if image_score is None:
        r += cfg.w_prompt * float(prompt_score)
    else:
        r += cfg.w_prompt * float(prompt_score) + cfg.w_image * float(image_score)

    # 长度正则（轻）
    r -= length_penalty(token_len, cfg.len_target, cfg.len_penalty)

    # 防雷同
    r -= cfg.rep_penalty * float(rep_sim)

    # hard fail 重
    if hard_fail:
        r -= cfg.hard_fail_penalty

    # 数值稳定：避免爆炸
    r = max(-5.0, min(5.0, r))
    return float(r)
