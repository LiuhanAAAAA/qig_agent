# src/evaluators/taxonomy.py
from __future__ import annotations
from typing import Dict, Any, List, Optional


def taxonomy_from_metrics(task_spec_or_metrics: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    ✅ 兼容两种调用方式：
    - taxonomy_from_metrics(metrics)
    - taxonomy_from_metrics(task_spec, metrics)

    你现在报错就是因为你项目里函数签名变成了第二种，但调用还是第一种。
    我这里直接做成“自动兼容”，以后不再因为签名不一致炸。
    """
    if metrics is None:
        metrics = task_spec_or_metrics

    tags: List[str] = []
    if float(metrics.get("sharpness", 1.0)) < 0.25:
        tags.append("blurry")
    if float(metrics.get("clip_alignment", 1.0)) < 0.45:
        tags.append("low_clip")
    if float(metrics.get("aesthetic", 1.0)) < 0.35:
        tags.append("low_aesthetic")
    if bool(metrics.get("has_text", False)):
        tags.append("has_text")
    if bool(metrics.get("has_face", False)):
        tags.append("has_face")
    return tags


def autofix_prompt(task_spec: Dict[str, Any], prompt: str, tags: List[str]) -> str:
    """
    根据 tags 给 prompt 自动“补丁”，比如 blurry -> add 'sharp focus'
    """
    fix_cfg = task_spec.get("taxonomy_autofix", {}) or {}
    add_pos: List[str] = []
    add_neg: List[str] = []

    for t in tags:
        if t not in fix_cfg:
            continue
        ap = (fix_cfg[t] or {}).get("add_positive", "")
        an = (fix_cfg[t] or {}).get("add_negative", "")
        if ap:
            add_pos.append(str(ap))
        if an:
            add_neg.append(str(an))

    if not add_pos and not add_neg:
        return prompt

    new_prompt = prompt.strip()

    # 简单拼接正向补丁
    if add_pos:
        new_prompt = new_prompt + ", " + ", ".join(add_pos)

    # 如果原 prompt 有 Negative prompt 段，则也拼进去
    if add_neg:
        if "Negative prompt:" in new_prompt:
            new_prompt = new_prompt + ", " + ", ".join(add_neg)
        else:
            new_prompt = new_prompt + "\nNegative prompt: " + ", ".join(add_neg)

    return new_prompt

def build_prior_avoidance_negative(
    task_spec: Dict[str, Any],
    failure_tags_or_counts,
    top_k_tags: int = 3
) -> str:
    """
    ✅ 兼容多种调用形式：
    - build_prior_avoidance_negative(spec, ["blurry","has_text"], top_k_tags=3)
    - build_prior_avoidance_negative(spec, {"blurry": 12, "has_text": 3}, top_k_tags=3)

    返回：拼接后的 negative prompt string
    """

    policy = (task_spec.get("prompt_policy") or {})
    base_neg = str(policy.get("negative_prompt", "")).strip()

    tag2neg = (task_spec.get("prior_avoidance") or {}).get("tag_to_negative", {}) or {}

    # -------- 1) 统一抽取 tags --------
    tags = []
    if isinstance(failure_tags_or_counts, dict):
        # failure_counts: {"tag": count, ...}
        # 取 top_k_tags 个出现最多的 tag
        items = sorted(
            [(k, int(v)) for k, v in failure_tags_or_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )
        tags = [k for k, _ in items[:max(0, int(top_k_tags))]]
    elif isinstance(failure_tags_or_counts, (list, tuple)):
        tags = list(failure_tags_or_counts)[:max(0, int(top_k_tags))]
    else:
        tags = []

    # -------- 2) 拼 extra negative --------
    extra = []
    for t in tags:
        v = tag2neg.get(t, "")
        if v:
            extra.append(str(v))

    # -------- 3) base + extra 去重拼接 --------
    all_neg = []
    if base_neg:
        all_neg.append(base_neg)
    for x in extra:
        if x not in all_neg:
            all_neg.append(x)

    return ", ".join(all_neg).strip()



