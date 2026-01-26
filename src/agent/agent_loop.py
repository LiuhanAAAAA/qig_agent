# src/agent/agent_loop.py
from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.utils.logger import log_info
from src.evaluators.image_evaluator import eval_images
from src.evaluators.prompt_evaluator import eval_prompts
from src.rl.reward import RewardConfig, combine_reward

from .skill_library import SkillLibrary
from .trajectory_memory import TrajectoryMemory


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_jsonl_append(path: str, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class AgentLoopConfig:
    enabled: bool = True
    max_steps: int = 4                 # 最多修复轮数
    max_images: int = 4                # 每个 episode 的总出图预算
    images_per_step: int = 1           # 每轮先出几张
    stop_score: float = 0.82           # 达标分数停止
    stop_if_no_hard_fail: bool = True  # 达标 + 没硬失败 才停
    ucb_c: float = 1.0                 # UCB exploration
    mutate_on_failure: bool = True     # 自我进化
    mutate_min_trials: int = 6
    mutate_max_success_rate: float = 0.20


def run_agent_episode(
    *,
    spec: Dict[str, Any],
    user_query: str,
    init_prompt: str,
    img_gen: Any,  # ImageGenerator
    tokenizer: Any,
    reward_cfg: RewardConfig,
    skill_lib: SkillLibrary,
    memory: TrajectoryMemory,
    out_dir: str,
    global_step: int,
    log_jsonl: str,
    cfg: AgentLoopConfig,
) -> Dict[str, Any]:
    """
    单个 query 的 episode：
      prompt -> image -> eval(tags) -> skill -> prompt' -> ...
    返回最终 prompt / reward / best_score / tags / trajectory
    """
    episode_id = f"{_ts()}_{global_step:06d}"
    ep_dir = Path(out_dir) / f"episode_{episode_id}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    prompt_cur = init_prompt.strip()
    if not prompt_cur:
        prompt_cur = init_prompt

    # 轨迹记录
    trajectory: List[Dict[str, Any]] = []
    used_images = 0

    best_reward = -1e9
    best_prompt = prompt_cur
    best_image_score: Optional[float] = None
    best_tags: List[str] = []

    # 每步都允许 prompt evaluator 先给一些 tags（cheap）
    # 注意：你的 eval_prompts 输出结构以仓库为准，这里按你现有的用法写
    def _eval_prompt_one(p: str) -> Dict[str, Any]:
        res = eval_prompts(spec, user_query, [p])
        if isinstance(res, list) and len(res) > 0:
            return res[0]
        # 兜底结构（尽量别触发）
        return {"prompt": p, "score": 0.0, "tags": [], "hard_fail": False}

    # 初始化：先 ensure structure（很关键，不然 hard constraint 反复爆炸）
    ensure_skill = None
    for s in skill_lib.skills:
        if s.skill_id == "ensure_structure":
            ensure_skill = s
            break
    if ensure_skill is not None:
        prompt_cur = ensure_skill.apply(prompt_cur, spec)

    # loop
    for step in range(int(cfg.max_steps)):
        # ---- prompt eval (cheap) ----
        pinfo = _eval_prompt_one(prompt_cur)
        prompt_score = float(pinfo.get("score", 0.0))
        tags_p = list(pinfo.get("tags", []))
        hard_p = bool(pinfo.get("hard_fail", False))

        # ---- image rollout (expensive) ----
        img_paths: List[str] = []
        image_score: Optional[float] = None
        tags_i: List[str] = []
        hard_i = False

        n_gen = min(int(cfg.images_per_step), max(0, int(cfg.max_images) - used_images))
        if n_gen > 0:
            cand_dir = ep_dir / f"step_{step:02d}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            img_paths = img_gen.generate(prompt_cur, out_dir=str(cand_dir), n=n_gen)
            used_images += len(img_paths)

            scores = []
            for pth in img_paths:
                res_i = eval_images(spec, prompt_cur, [pth])[0]
                scores.append(float(res_i["score"]))
                hard_i = bool(hard_i or res_i.get("hard_fail", False))
                tags_i.extend(list(res_i.get("tags", [])))

            if len(scores) > 0:
                scores_sorted = sorted(scores, reverse=True)
                top2 = scores_sorted[: min(2, len(scores_sorted))]
                image_score = float(sum(top2) / len(top2))

        # ---- reward ----
        token_len = 0
        try:
            token_len = int(tokenizer(prompt_cur, return_tensors="pt")["input_ids"].shape[-1])
        except Exception:
            token_len = int(len(prompt_cur.split()))

        hard_fail = bool(hard_p or hard_i)

        # rep_sim 这里不在 episode 内算（留给外层 batch 统一算）
        reward = combine_reward(
            reward_cfg,
            prompt_score=prompt_score,
            image_score=image_score,
            token_len=token_len,
            rep_sim=0.0,
            hard_fail=hard_fail,
        )

        tags_all = sorted(list(set(tags_p + tags_i)))

        # 更新 best
        if float(reward) > float(best_reward):
            best_reward = float(reward)
            best_prompt = prompt_cur
            best_image_score = image_score
            best_tags = tags_all

        # ---- log step ----
        row = {
            "episode_id": episode_id,
            "global_step": global_step,
            "user_query": user_query,
            "step": step,
            "prompt": prompt_cur,
            "prompt_score": prompt_score,
            "image_score": image_score,
            "reward": float(reward),
            "token_len": token_len,
            "tags_prompt": tags_p,
            "tags_image": sorted(list(set(tags_i))),
            "tags_all": tags_all,
            "hard_fail": hard_fail,
            "used_images": used_images,
            "img_paths": img_paths,
        }
        trajectory.append(row)
        _safe_jsonl_append(log_jsonl, row)

        # ---- stop condition ----
        if image_score is not None and float(image_score) >= float(cfg.stop_score):
            if (not cfg.stop_if_no_hard_fail) or (cfg.stop_if_no_hard_fail and not hard_fail):
                break

        if used_images >= int(cfg.max_images):
            break

        # ---- choose repair skill ----
        # 用 trajectory memory 做 UCB
        skill = skill_lib.choose_skill_ucb(tags_all, memory.get_stats(), ucb_c=float(cfg.ucb_c))

        # 应用 skill
        before_reward = float(reward)
        prompt_next = skill.apply(prompt_cur, spec)

        # 如果 skill 没产生变化，强制加 ensure_structure（避免无效循环）
        if prompt_next.strip() == prompt_cur.strip():
            prompt_next = ensure_skill.apply(prompt_cur, spec) if ensure_skill is not None else prompt_cur

        # 记录 skill 行为（下一步再评估 reward，先不 update）
        # 这里做一个“预测更新”：用下轮 reward - 本轮 reward 更新 memory
        # 但我们更稳：等下轮算完 reward 后再 update（因此这里先缓存）
        row_skill = {
            "episode_id": episode_id,
            "global_step": global_step,
            "user_query": user_query,
            "step": step,
            "action": "apply_skill",
            "skill_id": skill.skill_id,
            "skill_desc": skill.desc,
            "tags_used": tags_all,
            "reward_before": before_reward,
            "prompt_before": prompt_cur,
            "prompt_after": prompt_next,
        }
        _safe_jsonl_append(log_jsonl, row_skill)

        # 下一轮
        prompt_cur = prompt_next

        # self-evolution：如果某些 tag 长期无效，给它变异技能
        if cfg.mutate_on_failure:
            # 只在 hard tag 上变异（你后续可以扩大）
            for t in tags_all:
                if memory.should_mutate(t, min_trials=int(cfg.mutate_min_trials), max_success_rate=float(cfg.mutate_max_success_rate)):
                    new_skill = skill_lib.mutate_new_skill(t)
                    if new_skill is not None:
                        log_info(f"[AgentLoop] mutated new skill for tag={t}: {new_skill.skill_id}")
                        break

        # 注意：memory 的 update 放在外层训练脚本里（拿到 before/after reward）

    return {
        "episode_id": episode_id,
        "best_prompt": best_prompt,
        "best_reward": float(best_reward),
        "best_image_score": best_image_score,
        "best_tags": best_tags,
        "trajectory": trajectory,
        "used_images": used_images,
    }
