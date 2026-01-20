# src/prompt_evaluator.py
from typing import Dict, Any, List


def _rule_check(task_spec: Dict[str, Any], prompt: str):
    """
    第一层：规则检查（快 & 稳 & 无幻觉）
    """
    issues = []
    tags = []

    hard = task_spec.get("hard_constraints", {}).get("detect", {})

    # 规则：必须包含 Negative prompt
    if "Negative prompt:" not in prompt:
        issues.append("missing_negative_prompt")
        tags.append("missing_negative")

    # 规则：禁止真人脸 → negative 必须出现 face/photo
    if hard.get("forbid_realistic_face", False):
        if "photo" not in prompt.lower() and "realistic face" not in prompt.lower():
            issues.append("face_forbidden_but_negative_missing")
            tags.append("face_avoid_missing")

    # 规则：禁止文字/水印 → negative 必须出现 watermark/text
    if hard.get("forbid_text_overlay", False):
        if "watermark" not in prompt.lower() and "text" not in prompt.lower():
            issues.append("text_forbidden_but_negative_missing")
            tags.append("text_avoid_missing")

    # 简单长度约束（避免 prompt 太短导致发散）
    if len(prompt) < 80:
        issues.append("too_short")
        tags.append("weak_prompt")

    ok = len(issues) == 0
    return ok, issues, tags


def eval_prompts(task_spec: Dict[str, Any], prompts: List[str]) -> List[Dict[str, Any]]:
    """
    输出结构化：
    - score: 0~1
    - issues: list[str]
    - rewrite: str（给下一步 select 使用）
    """
    out = []
    for p in prompts:
        ok, issues, tags = _rule_check(task_spec, p)

        # ✅ score 设计：规则不过就低分，但不直接归零（避免全死）
        base = 0.85 if ok else 0.35
        penalty = min(0.25, 0.05 * len(issues))
        score = max(0.0, base - penalty)

        # rewrite：规则不过时做一次自动补全（工业稳）
        rewrite = p
        if "Negative prompt:" not in rewrite:
            neg = task_spec["prompt_policy"]["negative_prompt"].strip()
            rewrite = rewrite.strip() + f"\n. Negative prompt: {neg}\n"

        out.append({
            "prompt": p,
            "score": float(score),
            "issues": issues,
            "tags": tags,
            "rewrite": rewrite
        })
    return out
