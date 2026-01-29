# src/generators/prompt_generator.py
from __future__ import annotations
from typing import Dict, Any, List
import random

from src.memory.prompt_bank import PromptBank
from src.evaluators.taxonomy import build_prior_avoidance_negative
from pathlib import Path
from src.memory.prompt_bank import PromptBank


def _build_base_prompt(spec: Dict[str, Any], user_query: str) -> str:
    base_style = spec.get("prompt_policy", {}).get("base_style", "").strip()
    neg = spec.get("prompt_policy", {}).get("negative_prompt", "").strip()

    prompt = f"{user_query}, {base_style}\n. Negative prompt: {neg}\n"
    return prompt


def _expand_agent_generate_variants(spec: Dict[str, Any], user_query: str, k: int) -> List[str]:
    """
    Expand Agent：扩写出不同角度 prompt,用policy llm
    """
    seeds = [
        "a chibi character smiling, energetic and bright",
        "a minimalist geometric icon, cute and playful",
        "a soft plush animal avatar, warm and adorable",
        "a cute mascot head icon, clean and simple",
        "a flat illustration sticker style, simple outline",
        "a kawaii animal face icon, centered and clear",
    ]

    random.shuffle(seeds)
    seeds = seeds[: max(1, k)]

    out = []
    for s in seeds:
        out.append(_build_base_prompt(spec, f"{s}, {user_query}"))
    return out


def _policy_agent_apply_prior_avoidance(
    spec: Dict[str, Any],
    run_dir: str,
    prompts: List[str],
) -> List[str]:
    """
    Policy Agent：基于 memory 的 failure tag 分布，进行首次生成“先验规避”
    """
    mem_cfg = spec.get("memory", {})
    if not mem_cfg.get("enabled", False):
        return prompts

    # 用本 run 的 prompt_bank.sqlite 做统计（你也可以换成全局数据库）
    db_path = str(Path(run_dir) / "prompt_bank.sqlite")
    bank = PromptBank(db_path)
    # bank = PromptBank(run_dir)
    task_name = spec.get("task_name", "default")
    stats = bank.stats_for_task(task_name)

    failure_counts = stats.get("failure_tags", {}) if isinstance(stats, dict) else {}
    neg_extra = build_prior_avoidance_negative(spec, failure_counts, top_k_tags=3)

    if not neg_extra:
        return prompts

    fixed = []
    for p in prompts:
        # 把 prior avoidance 追加到 Negative prompt 后面
        if "Negative prompt:" in p:
            head, neg = p.split("Negative prompt:", 1)
            neg = neg.strip().rstrip(",")
            neg = neg + ", " + neg_extra
            p2 = head.rstrip() + "\n. Negative prompt: " + neg + "\n"
            fixed.append(p2)
        else:
            # 保底兜底
            neg0 = spec.get("prompt_policy", {}).get("negative_prompt", "")
            fixed.append(p.strip() + f"\n. Negative prompt: {neg0}, {neg_extra}\n")

    return fixed


def generate_prompt_candidates(
    spec: Dict[str, Any],
    user_query: str,
    k: int,
    retrieved: List[Dict[str, Any]] | None = None,
    run_dir: str = ".",
) -> List[str]:
    """
    主入口：graph.py 
    """
    retrieved = retrieved or []

    # 1) Expand：先生成候选
    prompts = _expand_agent_generate_variants(spec, user_query, k=k)

    # 2) 结合 retrieved memory few-shot（把历史高分 prompt prepend）
    if retrieved:
        # retrieved 是 PromptBank.retrieve_similar 的输出 dict 列表
        top = [r["prompt"] for r in retrieved if "prompt" in r][:2]
        prompts = top + prompts

    # 3) Policy：先验规避
    prompts = _policy_agent_apply_prior_avoidance(spec, run_dir, prompts)

    return prompts[:k]
