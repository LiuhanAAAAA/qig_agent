# src/evaluators/prompt_evaluator.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import re


# -----------------------------
# helpers
# -----------------------------
_NEG_MARK = "Negative prompt:"


def _split_pos_neg(prompt: str) -> Tuple[str, str]:
    """
    Split prompt into (positive_text, negative_text).
    If no negative section, negative_text = "".
    """
    s = (prompt or "").strip()
    idx = s.lower().find(_NEG_MARK.lower())
    if idx < 0:
        return s, ""
    pos = s[:idx].strip()
    neg = s[idx + len(_NEG_MARK):].strip()
    return pos, neg


def _contains_any(text: str, kws: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in kws)


def _count_hashtags(text: str) -> int:
    return len(re.findall(r"#\w+", text or ""))


def _count_emojis(text: str) -> int:
    # very lightweight emoji-ish detector
    return len(re.findall(r"[\U00010000-\U0010ffff]", text or ""))


def _approx_token_len(text: str) -> int:
    """
    Rough token estimate without extra dependency:
    - English: word count
    - Mixed / symbols: also counts punctuation groups
    """
    if not text:
        return 0
    words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text)
    return int(len(words))


def _append_missing_neg_keywords(neg: str, required: List[str]) -> str:
    """
    Append missing keywords into negative prompt, keeping it compact.
    """
    neg_l = (neg or "").lower()
    missing = [k for k in required if k.lower() not in neg_l]
    if not missing:
        return neg
    extra = ", ".join(missing)
    if neg.strip() == "":
        return extra
    # keep punctuation consistent
    if neg.strip().endswith((",", ";")):
        return neg.strip() + " " + extra
    return neg.strip() + ", " + extra


# -----------------------------
# rule check + scoring
# -----------------------------
def _rule_check(task_spec: Dict[str, Any], prompt: str):
    """
    第一层：规则检查（快 & 稳 & 无幻觉）
    输出：
      ok, issues, tags, hard_fail, rewrite
    """
    issues: List[str] = []
    tags: List[str] = []

    hard = task_spec.get("hard_constraints", {}).get("detect", {})
    forbid_text = bool(hard.get("forbid_text_overlay", False))
    forbid_real_face = bool(hard.get("forbid_realistic_face", False))

    pos, neg = _split_pos_neg(prompt)

    # --- required negative section ---
    if _NEG_MARK.lower() not in (prompt or "").lower():
        issues.append("missing_negative_prompt")
        tags.append("missing_negative")

    # --- negative should cover constraints ---
    required_neg_text = ["watermark", "text", "caption", "subtitle", "logo", "signature", "username"]
    required_neg_face = [
        "photorealistic", "photo", "real person", "human face", "realistic portrait",
        "skin texture", "natural skin", "camera"
    ]

    if forbid_text:
        if not _contains_any((neg or ""), ["watermark", "text", "caption", "subtitle", "logo"]):
            issues.append("text_forbidden_but_negative_missing")
            tags.append("need_neg_text")

    # ✅ 你要求：只禁“真人脸/写真风格”，卡通脸允许
    if forbid_real_face:
        if not _contains_any((neg or ""), ["photorealistic", "photo", "real person", "human face", "portrait"]):
            issues.append("real_face_forbidden_but_negative_missing")
            tags.append("need_neg_face")

    # --- prompt quality heuristics ---
    # Too short => high variance
    if len((pos or "").strip()) < 60:
        issues.append("too_short")
        tags.append("weak_prompt")

    # Too long => truncation risk (你日志 token_len 常到 160)
    tok_est = _approx_token_len(prompt)
    if tok_est >= 140:
        tags.append("clip_trunc_risk")

    if tok_est >= 190:
        issues.append("too_long_high_risk")
        tags.append("too_long_high_risk")

    # Noisy patterns: hashtags / emoji spam / meta chatter
    ht = _count_hashtags(pos)
    em = _count_emojis(pos)
    if ht >= 6 or em >= 10:
        issues.append("noisy_prompt")
        tags.append("noisy_prompt")

    # Prompt asking for overlay text is dangerous if forbid_text
    if _contains_any(pos, ["add a caption", "caption", "add text", "watermark", "subtitle", "logo"]):
        tags.append("pos_mentions_text_overlay")
        if forbid_text:
            issues.append("pos_requests_text_overlay")
            tags.append("hard_forbid_text_req")

    # If prompt explicitly requests photorealistic human portrait while forbidding realistic face => hard fail
    hard_fail = False
    if forbid_real_face:
        if _contains_any(pos, ["photorealistic", "photo", "realistic portrait"]) and _contains_any(pos, ["person", "human", "face", "portrait"]):
            hard_fail = True
            issues.append("pos_requests_realistic_human_face")
            tags.append("hard_forbid_real_face_req")

    # Specificity proxy (very simple):
    # Encourage: subject + style + background/detail
    style_kws = ["cartoon", "anime", "illustration", "2d", "pixel", "chibi", "flat color", "vector"]
    detail_kws = ["background", "lighting", "soft", "pastel", "composition", "close-up", "full-body", "icon", "avatar"]
    spec_hits = 0
    spec_hits += 1 if len(pos.split()) >= 8 else 0
    spec_hits += 1 if _contains_any(pos, style_kws) else 0
    spec_hits += 1 if _contains_any(pos, detail_kws) else 0
    if spec_hits <= 1:
        issues.append("low_specificity")
        tags.append("low_specificity")

    ok = (len([x for x in issues if x.startswith("pos_requests_") or x.endswith("_missing") or x == "missing_negative_prompt"]) == 0) and (not hard_fail)

    # --- rewrite (工业稳) ---
    rewrite = (prompt or "").strip()

    # 1) ensure negative prompt block exists
    if _NEG_MARK.lower() not in rewrite.lower():
        default_neg = ""
        try:
            default_neg = str(task_spec["prompt_policy"]["negative_prompt"]).strip()
        except Exception:
            default_neg = "watermark, text, logo, photorealistic, photo, human face"
        rewrite = rewrite.strip() + f"\n{_NEG_MARK} {default_neg}\n"

        # update split
        pos, neg = _split_pos_neg(rewrite)

    # 2) ensure required negative keywords exist
    if forbid_text:
        neg = _append_missing_neg_keywords(neg, required_neg_text)
    if forbid_real_face:
        # ✅ 这里只是“禁真人脸”，不是禁所有 face
        neg = _append_missing_neg_keywords(neg, required_neg_face)

    # re-compose with same marker
    rewrite = pos.strip() + f"\n{_NEG_MARK} {neg.strip()}\n"

    return ok, issues, tags, hard_fail, rewrite


def eval_prompts(task_spec: Dict[str, Any], prompts: List[str]) -> List[Dict[str, Any]]:
    """
    输出结构化：
    - score: 0~1
    - issues: list[str]
    - hard_fail: bool
    - tags: list[str]
    - rewrite: str（给下一步 select / generation 使用）
    """
    out: List[Dict[str, Any]] = []

    for p in prompts:
        ok, issues, tags, hard_fail, rewrite = _rule_check(task_spec, p)

        # -----------------------------
        # scoring (0~1)
        # -----------------------------
        # Base score
        score = 0.62

        # Hard fails -> very low but not necessarily 0 (so RL still learns)
        if hard_fail:
            score -= 0.55

        # Missing negative is a big deal (your logs show this dominates failure)
        if "missing_negative" in tags:
            score -= 0.30

        # Missing constraint-related neg
        if "need_neg_text" in tags:
            score -= 0.15
        if "need_neg_face" in tags:
            score -= 0.15

        # Quality penalties
        if "weak_prompt" in tags:
            score -= 0.10
        if "low_specificity" in tags:
            score -= 0.12
        if "clip_trunc_risk" in tags:
            score -= 0.06
        if "too_long_high_risk" in tags:
            score -= 0.12
        if "noisy_prompt" in tags:
            score -= 0.08

        # Positive mentions text overlay while forbidden -> extra penalty
        if "hard_forbid_text_req" in tags:
            score -= 0.25

        # Bonuses: having explicit style words helps stable generation
        pos, _neg = _split_pos_neg(rewrite)
        if _contains_any(pos, ["avatar", "icon", "mascot"]):
            score += 0.04
        if _contains_any(pos, ["cartoon", "anime", "illustration", "2d", "vector"]):
            score += 0.05

        if ok:
            score += 0.05

        # clamp
        score = max(0.0, min(1.0, float(score)))

        out.append({
            "prompt": p,
            "score": score,
            "issues": issues,
            "tags": tags,
            "hard_fail": bool(hard_fail),
            "rewrite": rewrite,
        })

    return out
