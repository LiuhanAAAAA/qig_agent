# src/llm/gepa_optimizer.py
from __future__ import annotations

import json
import inspect
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logger import log_info, log_warn
from src.rl.reward import RewardConfig, combine_reward
from src.evaluators.prompt_evaluator import eval_prompts
from src.evaluators.image_evaluator import eval_images

# agent components
from src.agent.skill_library import SkillLibrary
from src.agent.skill_selector import SkillSelector
from src.agent.trajectory_memory import TrajectoryMemory

try:
    from src.llm.clients import LLMClient  # type: ignore
except Exception:
    LLMClient = Any  # fallback


# -----------------------
# Adapter for evaluator signatures
# -----------------------
def _call_eval_prompts(task_spec: Dict[str, Any], user_query: str, prompts: List[str]) -> List[Dict[str, Any]]:
    try:
        sig = inspect.signature(eval_prompts)
        if len(sig.parameters) == 2:
            return eval_prompts(user_query, prompts)  # type: ignore
    except Exception:
        pass
    return eval_prompts(task_spec, user_query, prompts)  # type: ignore


def _call_eval_images(task_spec: Dict[str, Any], prompt: str, img_paths: List[str]) -> List[Dict[str, Any]]:
    try:
        sig = inspect.signature(eval_images)
        if len(sig.parameters) == 2:
            return eval_images(prompt, img_paths)  # type: ignore
    except Exception:
        pass
    return eval_images(task_spec, prompt, img_paths)  # type: ignore


# -----------------------
# Config
# -----------------------
@dataclass
class GEPAConfig:
    population_size: int = 12
    generations: int = 5
    elite_size: int = 3
    crossover_rate: float = 0.35
    mutation_rate: float = 0.9

    # mutation chooser
    llm_mutation_rate: float = 0.65  # otherwise use skill mutation
    max_words: int = 45

    # multi-fidelity inside GEPA
    image_eval_topk: int = 3
    images_per_prompt: int = 1

    # robustness
    init_fill_max_trials: int = 80          # 防止填充 population 死循环
    child_retry_limit: int = 200            # 防止生成 child 死循环
    max_prompt_chars: int = 8000            # prompt 太长直接硬截断，避免 split 卡死

    # bookkeeping
    random_seed: int = 42


# -----------------------
# Utils
# -----------------------
def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _clean_llm_output(s: str) -> str:
    s = (s or "").strip()

    # strip markdown fences
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            s = "\n".join(lines[1:-1]).strip()

    # 删除常见废话引导
    s = re.sub(r"^\s*(final\s*:|answer\s*:)\s*", "", s, flags=re.I).strip()
    return s

# def _extract_prompt_only(text: str) -> str:
#     raw = _clean_llm_output(text)

#     # JSON
#     try:
#         obj = json.loads(raw)
#         if isinstance(obj, dict):
#             for k in ["prompt", "positive_prompt", "pos_prompt"]:
#                 v = obj.get(k, None)
#                 if isinstance(v, str) and v.strip():
#                     return v.strip()
#     except Exception:
#         pass

#     # "prompt: ..."
#     m = re.search(r"(?is)\bprompt\s*:\s*(.+)$", raw)
#     if m:
#         s = m.group(1).strip()
#         s = re.split(r"(?is)\bnegative\s*prompt\s*:", s)[0].strip()
#         return s

#     # split off negative prompt
#     raw = re.split(r"(?is)\bnegative\s*prompt\s*:", raw)[0].strip()

#     # first non-empty line
#     for line in raw.splitlines():
#         line = line.strip()
#         if line:
#             return line

#     return raw.strip()


def _extract_prompt_only(text: str) -> str:
    """
    强化：只保留「正向 prompt」，把解释性句子砍掉
    """
    raw = _clean_llm_output(text)

    # 1) JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for k in ["prompt", "positive_prompt", "pos_prompt"]:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    raw = v.strip()
    except Exception:
        pass

    # 2) 如果包含 prompt: 取后半段
    m = re.search(r"(?is)\bprompt\s*:\s*(.+)$", raw)
    if m:
        raw = m.group(1).strip()

    # 3) 切掉 negative prompt 之后的内容
    raw = re.split(r"(?is)\bnegative\s*prompt\s*:", raw)[0].strip()

    # 4) 删除解释型废话句式
    trash_patterns = [
        r"(?is)\bthe response is written.*$",
        r"(?is)\bthis prompt.*$",
        r"(?is)\bwithout including.*$",
        r"(?is)\bhashtags.*$",
        r"(?is)\bexplanation.*$",
    ]
    for pat in trash_patterns:
        raw = re.sub(pat, "", raw).strip()

    # 5) 只取第一行
    first_line = raw.splitlines()[0].strip() if raw.splitlines() else raw.strip()
    return first_line

def _word_count_fast(s: str) -> int:
    """fast count words by scanning non-space segments (no split-all)"""
    if not s:
        return 0
    n = 0
    in_tok = False
    for ch in s:
        if ch.isspace():
            in_tok = False
        else:
            if not in_tok:
                n += 1
                in_tok = True
    return n


def _truncate_by_words(text: str, max_words: int, max_chars: int | None = None) -> str:
    """
    - 按 words 截断（英文）
    - 按 chars 截断（中文/无空格场景 & 防止超长）
    """
    if not isinstance(text, str):
        text = str(text)

    s = (text or "").strip()

    # 1) 先做字符级截断（最硬的上限，避免你那种超长废话）
    if max_chars is not None and max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars].strip()

    # 2) 再做 word 级截断（英文有效）
    if max_words is not None and max_words > 0:
        toks = s.split()
        if len(toks) > max_words:
            s = " ".join(toks[:max_words]).strip()

    # 3) 清理结尾的逗号/分号
    s = s.rstrip(" ,;，；")

    return s


def _split_fragments(prompt: str) -> List[str]:
    parts = re.split(r"[，,;；]+", (prompt or "").strip())
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def _crossover(a: str, b: str, max_words: int, max_chars: int) -> str:
    pa = _split_fragments(a)
    pb = _split_fragments(b)
    if not pa and not pb:
        return ""
    if not pa:
        return _truncate_by_words(b, max_words, max_chars=max_chars)
    if not pb:
        return _truncate_by_words(a, max_words, max_chars=max_chars)

    keep_a = max(1, int(len(pa) * 0.6))
    keep_b = max(1, int(len(pb) * 0.4))
    cand = random.sample(pa, min(len(pa), keep_a)) + random.sample(pb, min(len(pb), keep_b))

    seen = set()
    merged: List[str] = []
    for x in cand:
        key = x.lower()
        if key not in seen:
            seen.add(key)
            merged.append(x)

    out = ", ".join(merged).strip()
    return _truncate_by_words(out, max_words, max_chars=max_chars)


def _normalize_skill_names(skill_lib: SkillLibrary, items: List[Any]) -> List[str]:
    """
    兼容 SkillLibrary.available_skills_for_tags 返回：
      - ["name1","name2"] 或
      - [Skill(...), Skill(...)]
    """
    out: List[str] = []
    for it in items:
        if isinstance(it, str):
            out.append(it)
        else:
            name = getattr(it, "name", None)
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
    # 去重保序
    out = list(dict.fromkeys(out))
    # 过滤掉不存在的技能名
    valid = getattr(skill_lib, "skills", {})
    out = [n for n in out if n in valid]
    return out


# -----------------------
# GEPA Optimizer
# -----------------------
class GEPAOptimizer:
    """
    GEPA-style prompt optimizer (Genetic evolution + optional LLM mutation + skill mutation fallback)
    """

    def __init__(
        self,
        task_spec: Dict[str, Any],
        reward_cfg: RewardConfig,
        img_gen: Any,
        llm: Any,
        memory: TrajectoryMemory,
        skill_lib: SkillLibrary,
        selector: SkillSelector,
        gepa_cfg: GEPAConfig,
    ):
        self.task_spec = task_spec
        self.reward_cfg = reward_cfg
        self.img_gen = img_gen
        self.llm = llm
        self.memory = memory
        self.skill_lib = skill_lib
        self.selector = selector
        self.cfg = gepa_cfg

        random.seed(int(self.cfg.random_seed))
        np.random.seed(int(self.cfg.random_seed))

    # ---------- mutations ----------
    def _apply_one_skill(self, parent_prompt: str, tags: List[str]) -> Tuple[str, Optional[str]]:
        """
        - available_skills_for_tags 可能返回 Skill 对象列表 -> 转成 skill_name 列表
        - selector.select 输入必须是 name list
        """
        parent_prompt = (parent_prompt or "").strip()

        cand_raw: List[Any] = []
        try:
            cand_raw = self.skill_lib.available_skills_for_tags(tags or [])
        except Exception:
            cand_raw = []

        cand = _normalize_skill_names(self.skill_lib, cand_raw)

        # 如果没有 tag 匹配到技能，则允许挑一些“通用技能”
        if not cand:
            all_names = list(getattr(self.skill_lib, "skills", {}).keys())
            # 这里不要全放进 selector，挑一小撮避免 UCB 初期膨胀
            if all_names:
                random.shuffle(all_names)
                cand = all_names[: min(6, len(all_names))]

        if not cand:
            return parent_prompt, None

        # selector 返回 skill_name
        skill_name = self.selector.select(cand)
        if not isinstance(skill_name, str):
            # 极端情况兜底
            skill_name = str(getattr(skill_name, "name", "")) or ""

        skill_name = skill_name.strip()
        skill = getattr(self.skill_lib, "skills", {}).get(skill_name, None)
        if skill is None:
            return parent_prompt, None

        try:
            out = skill.apply(parent_prompt, {"tags": tags})
            out = out if isinstance(out, str) else str(out)
            out = (out or "").strip()
            if not out:
                return parent_prompt, skill_name
            out = _truncate_by_words(out, int(self.cfg.max_words), max_chars=int(self.cfg.max_prompt_chars))
            return out, skill_name
        except Exception as e:
            log_warn(f"[GEPA] skill mutation failed ({skill_name}): {e}")
            return parent_prompt, skill_name

    def _llm_mutation(self, user_query: str, parent_prompt: str, tags: List[str]) -> Tuple[str, Optional[str]]:
        """
        LLM mutation: ask LLM to rewrite prompt.
        If fails, fallback to skill mutation.
        """
        tags_str = ", ".join(sorted(set(tags or []))) if tags else "none"
        sys = (
            # "You are a prompt optimization assistant for image generation.\n"
            # "Rewrite the prompt to better match the user query while keeping it concise.\n"
            # "Return ONLY the positive prompt text (no explanation, no markdown).\n"
            # "Avoid repeating quality phrases.\n"
            # f"Keep <= {int(self.cfg.max_words)} words if possible."
            "You are rewriting an image generation POSITIVE prompt.\n"
            "Return ONLY the prompt text.\n"
            "- No explanation.\n"
            "- No hashtags.\n"
            "- No 'negative prompt'.\n"
            "- No markdown.\n"
            "- Keep it short and descriptive.\n"
            f"- Max {int(self.cfg.max_words)} words.\n"
        )
        user = (
            f"User query:\n{user_query}\n\n"
            f"Current prompt:\n{parent_prompt}\n\n"
            f"Detected issues/tags:\n{tags_str}\n\n"
            "Rewrite the prompt now."
        )
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

        try:
            out = self.llm.chat(msgs, temperature=0.5, max_tokens=220)
            new_prompt = _extract_prompt_only(out).strip()
            if not new_prompt:
                raise ValueError("empty LLM output")

            new_prompt = _truncate_by_words(new_prompt, int(self.cfg.max_words), max_chars=int(self.cfg.max_prompt_chars))
            return new_prompt, None
        except Exception as e:
            log_warn(f"[GEPA] LLM mutation failed ({type(e).__name__}: {e}). Fallback -> skill mutation.")
            return self._apply_one_skill(parent_prompt, tags)

    # ---------- evaluation ----------
    def _prompt_rep_sim(self, prompts: List[str]) -> List[float]:
        sets = [set((x or "").lower().split()) for x in prompts]
        sims: List[float] = []
        for i in range(len(prompts)):
            s_i = sets[i]
            best = 0.0
            for j in range(len(prompts)):
                if i == j:
                    continue
                s_j = sets[j]
                inter = len(s_i & s_j)
                union = max(1, len(s_i | s_j))
                best = max(best, inter / union)
            sims.append(float(best))
        return sims

    def _eval_population(self, user_query: str, prompts: List[str], episode_dir: Path) -> List[Dict[str, Any]]:
        pe = _call_eval_prompts(self.task_spec, user_query, prompts)
        pe_map = {x.get("prompt", ""): x for x in pe}

        order = list(prompts)
        order.sort(key=lambda p: float(pe_map.get(p, {}).get("score", 0.0)), reverse=True)
        topk = max(0, min(int(self.cfg.image_eval_topk), len(order)))
        img_candidates = set(order[:topk])

        rep_sim = self._prompt_rep_sim(prompts)
        results: List[Dict[str, Any]] = []

        for i, p in enumerate(prompts):
            pi = pe_map.get(p, {"prompt": p, "score": 0.0, "tags": [], "hard_fail": False})
            prompt_score = float(pi.get("score", 0.0))
            tags_p = list(pi.get("tags", [])) if isinstance(pi.get("tags", []), list) else []
            hard_p = bool(pi.get("hard_fail", False))

            image_score: Optional[float] = None
            hard_i = False
            img_paths: List[str] = []
            tags_i: List[str] = []

            if p in img_candidates and int(self.cfg.images_per_prompt) > 0:
                out_dir = episode_dir / "gepa_img" / f"cand_{i:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)

                img_paths = self.img_gen.generate(p, out_dir=str(out_dir), n=int(self.cfg.images_per_prompt))

                scores: List[float] = []
                for path in img_paths:
                    res = _call_eval_images(self.task_spec, p, [path])[0]
                    scores.append(float(res.get("score", 0.0)))
                    if res.get("hard_fail", False):
                        hard_i = True
                    tags_i.extend(list(res.get("tags", [])))

                if scores:
                    scores_sorted = sorted(scores, reverse=True)
                    top2 = scores_sorted[: min(2, len(scores_sorted))]
                    image_score = float(sum(top2) / len(top2))

            hard_fail = bool(hard_p or hard_i)
            tags_all = sorted(list(set(tags_p + tags_i)))

            token_len = _word_count_fast(p)
            r = combine_reward(
                self.reward_cfg,
                prompt_score=prompt_score,
                image_score=image_score,
                token_len=token_len,
                rep_sim=float(rep_sim[i]),
                hard_fail=hard_fail,
            )

            results.append(
                {
                    "prompt": p,
                    "prompt_score": prompt_score,
                    "image_score": image_score,
                    "reward": float(r),
                    "hard_fail": hard_fail,
                    "tags": tags_all,
                    "img_paths": img_paths,
                }
            )

        return results

    # ---------- main optimize ----------
    def optimize(self, user_query: str, seed_prompts: List[str], episode_dir: Path) -> Dict[str, Any]:
        episode_dir = Path(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)

        seeds = [s.strip() for s in (seed_prompts or []) if (s or "").strip()]
        seeds = list(dict.fromkeys(seeds))
        if not seeds:
            return {"best_prompt": "", "best_reward": 0.0, "history": []}

        # init population
        pop: List[str] = seeds[:]
        trials = 0
        while len(pop) < int(self.cfg.population_size) and trials < int(self.cfg.init_fill_max_trials):
            trials += 1
            base = random.choice(seeds)
            mutated, _ = self._apply_one_skill(base, tags=[])
            mutated = _truncate_by_words(mutated, int(self.cfg.max_words), max_chars=int(self.cfg.max_prompt_chars))
            if mutated and mutated not in pop:
                pop.append(mutated)

        if len(pop) < max(2, int(self.cfg.elite_size)):
            # 兜底：避免 population 太小导致进化过程异常
            pop = seeds[: max(2, min(len(seeds), int(self.cfg.population_size)))]

        history: List[Dict[str, Any]] = []
        best_prompt = pop[0]
        best_reward = -1e9

        for g in range(int(self.cfg.generations)):
            scored = self._eval_population(user_query, pop, episode_dir=episode_dir / f"gen_{g:02d}")
            scored.sort(key=lambda x: float(x["reward"]), reverse=True)

            elite = scored[: max(1, min(int(self.cfg.elite_size), len(scored)))]
            gen_best = elite[0]
            if float(gen_best["reward"]) > best_reward:
                best_reward = float(gen_best["reward"])
                best_prompt = str(gen_best["prompt"])

            history.append(
                {
                    "generation": g,
                    "best_reward": float(gen_best["reward"]),
                    "best_prompt": str(gen_best["prompt"]),
                    "mean_reward": float(np.mean([x["reward"] for x in scored])),
                }
            )

            log_info(
                f"[GEPA] gen={g} best_reward={float(gen_best['reward']):.4f} "
                f"mean_reward={float(history[-1]['mean_reward']):.4f} "
                f"best_prompt_words={_word_count_fast(str(gen_best['prompt']))}"
            )

            # next population
            new_pop: List[str] = [x["prompt"] for x in elite]
            reward_map = {x["prompt"]: x for x in scored}

            pool = scored[: max(2, len(scored) // 2)]
            pool_prompts = [x["prompt"] for x in pool]

            def sample_parent() -> str:
                return random.choice(pool_prompts)

            attempts = 0
            while len(new_pop) < int(self.cfg.population_size) and attempts < int(self.cfg.child_retry_limit):
                attempts += 1
                p1 = sample_parent()
                p2 = sample_parent()
                child = p1

                if random.random() < float(self.cfg.crossover_rate):
                    child = _crossover(p1, p2, max_words=int(self.cfg.max_words), max_chars=int(self.cfg.max_prompt_chars))

                used_skill: Optional[str] = None
                if random.random() < float(self.cfg.mutation_rate):
                    tags = reward_map.get(p1, {}).get("tags", [])
                    if random.random() < float(self.cfg.llm_mutation_rate):
                        child, used_skill = self._llm_mutation(user_query, child, tags)
                    else:
                        child, used_skill = self._apply_one_skill(child, tags)

                child = _truncate_by_words(child, int(self.cfg.max_words), max_chars=int(self.cfg.max_prompt_chars))
                if not child:
                    continue
                if child in new_pop:
                    continue

                # record patch
                try:
                    self.memory.record_patch(
                        user_query=user_query,
                        parent_prompt=p1,
                        new_prompt=child,
                        tags=list(reward_map.get(p1, {}).get("tags", [])),
                        meta={"generation": g, "used_skill": used_skill},
                    )
                except Exception:
                    pass

                new_pop.append(child)

            pop = new_pop

        return {"best_prompt": best_prompt, "best_reward": float(best_reward), "history": history}
