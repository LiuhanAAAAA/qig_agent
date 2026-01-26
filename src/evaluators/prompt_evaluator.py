# src/evaluators/prompt_evaluator.py
from __future__ import annotations

import re
from typing import Any, Dict, List


_BAD_POS_TEXT_WORDS = [
    "watermark", "logo", "signature", "caption", "subtitle", "overlay text",
    "文字", "水印", "字幕", "标志", "logo", "签名"
]

# 这些是你“只禁真人脸”时，推荐在 Negative prompt 里出现的词
_NEG_REAL_FACE_TOKENS = [
    "real human", "photorealistic face", "realistic face", "portrait photo",
    "skin texture", "human face", "real person"
]

_NEG_TEXT_TOKENS = [
    "text", "watermark", "logo", "signature", "caption", "subtitle"
]


def _has_negative_prompt(prompt: str) -> bool:
    return ("negative prompt:" in prompt.lower()) or ("负向" in prompt)


def _split_pos_neg(prompt: str) -> (str, str):
    """
    支持格式：
    - "xxx ...\nNegative prompt: yyy"
    - "xxx ...\n负向：yyy"
    """
    s = prompt.strip()
    m = re.search(r"(negative prompt\s*:|负向\s*[:：])", s, flags=re.IGNORECASE)
    if not m:
        return s, ""
    pos = s[:m.start()].strip()
    neg = s[m.end():].strip()
    return pos, neg


def _approx_len(prompt: str) -> int:
    # 简易 token proxy：按空格切
    return max(1, len(prompt.strip().split()))


def eval_prompts(spec: Dict[str, Any], user_query: str, prompts: List[str]) -> List[Dict[str, Any]]:
    """
    输出字段对齐你的 PPO 训练脚本：
    [
      { "prompt": str, "score": float, "tags": [...], "hard_fail": bool }
    ]
    """
    forbid_text = bool(spec["hard_constraints"]["detect"].get("forbid_text_overlay", False))
    forbid_real_face = bool(spec["hard_constraints"]["detect"].get("forbid_realistic_face", False))

    out: List[Dict[str, Any]] = []

    for p in prompts:
        p = (p or "").strip()
        if not p:
            out.append({"prompt": "", "score": 0.0, "tags": ["empty_prompt"], "hard_fail": True})
            continue

        tags: List[str] = []
        hard_fail = False

        pos, neg = _split_pos_neg(p)
        L = _approx_len(pos)

        # ---------- Positive side checks ----------
        low_pos = pos.lower()
        if any(w in low_pos for w in _BAD_POS_TEXT_WORDS):
            tags.append("pos_mentions_text_overlay")
            # 如果你 spec forbid_text_overlay，这种属于严重错误
            if forbid_text:
                hard_fail = True

        # 真人脸相关词（正向里出现也很危险）
        if forbid_real_face and re.search(r"\b(photo|portrait|real person|realistic face|human face)\b", low_pos):
            tags.append("pos_mentions_real_face")
            hard_fail = True

        # ---------- Negative side checks ----------
        if not _has_negative_prompt(p):
            tags.append("missing_negative_prompt")

        low_neg = neg.lower()
        if forbid_text:
            if not any(t in low_neg for t in _NEG_TEXT_TOKENS):
                tags.append("need_neg_text")
        if forbid_real_face:
            if not any(t in low_neg for t in _NEG_REAL_FACE_TOKENS):
                tags.append("need_neg_realistic_face")

        # ---------- Length / structure ----------
        if L < 12:
            tags.append("prompt_too_short")
        if L > 120:
            tags.append("clip_trunc_risk")

        # ---------- score heuristic ----------
        # 基础分：让 prompt evaluator 只是一个“cheap proxy”
        score = 0.55

        # 结构完整性奖励
        if "high quality" in low_pos or "best quality" in low_pos:
            score += 0.05
        if "cinematic" in low_pos or "lighting" in low_pos:
            score += 0.04
        if "background" in low_pos or "scene" in low_pos:
            score += 0.04

        # 约束满足奖励
        if forbid_text and ("need_neg_text" not in tags):
            score += 0.06
        if forbid_real_face and ("need_neg_realistic_face" not in tags):
            score += 0.06
        if "missing_negative_prompt" not in tags:
            score += 0.04

        # 风险惩罚
        if "prompt_too_short" in tags:
            score -= 0.08
        if "clip_trunc_risk" in tags:
            score -= 0.10
        if "pos_mentions_text_overlay" in tags:
            score -= 0.25
        if "pos_mentions_real_face" in tags:
            score -= 0.30

        # hard_fail 直接砍
        if hard_fail:
            score = min(score, 0.15)

        score = float(max(0.0, min(1.0, score)))

        out.append({
            "prompt": p,
            "score": score,
            "tags": sorted(list(set(tags))),
            "hard_fail": bool(hard_fail),
        })

    return out
