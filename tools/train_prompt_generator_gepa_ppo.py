# tools/train_prompt_generator_gepa_ppo.py
from __future__ import annotations

"""
GEPA + PPO training loop for prompt generator.

Key features (non缩水版):
- Repo bootstrap: always runnable by `python tools/train_prompt_generator_gepa_ppo.py ...`
- GEPA prompt search loop (self-improving, tag-aware mutation)
- Multi-fidelity evaluation (prompt evaluator -> topK -> image evaluator)
- PPO update on PromptPolicy (actor) with value head (critic)
- Trajectory logging (jsonl) + memory (jsonl)

Windows-friendly: no need `pip install -e .`, no reliance on `tools` being a package.
"""

# -----------------------
# ✅ bootstrap: ensure repo root is on sys.path
# -----------------------
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# make relative paths stable
os.chdir(REPO_ROOT)

# -----------------------
# Standard libs
# -----------------------
import argparse
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -----------------------
# Third-party
# -----------------------
import numpy as np
import torch
import yaml

# -----------------------
# Project imports
# -----------------------
from src.utils.logger import log_info
from src.generators.image_generator import ImageGenerator

# PPO policy（保持你仓库结构）
from src.rl.prompt_policy import PromptPolicy, PolicyConfig
from src.rl.reward import RewardConfig, combine_reward

# evaluators
from src.evaluators.image_evaluator import eval_images
from src.evaluators.prompt_evaluator import eval_prompts

# agent components
from src.agent.skill_library import SkillLibrary
from src.agent.skill_selector import SkillSelector
from src.agent.trajectory_memory import TrajectoryMemory

# GEPA optimizer is in src/llm in your repo
from src.llm.gepa_optimizer import GEPAOptimizer, GEPAConfig

# LLM client (Qwen via Ollama/vLLM)
from src.llm.clients import build_llm_client


# -----------------------
# adapter (兼容 eval_prompts 的不同签名)
# -----------------------
def _call_eval_prompts(spec: Dict[str, Any], user_query: str, prompts: List[str]):
    sig = inspect.signature(eval_prompts)
    n = len(sig.parameters)

    if n == 3:
        return eval_prompts(spec, user_query, prompts)

    if n == 2:
        names = list(sig.parameters.keys())
        if "spec" in names[0] or "task" in names[0]:
            return eval_prompts(spec, prompts)
        if "query" in names[0] or "user" in names[0]:
            return eval_prompts(user_query, prompts)
        return eval_prompts(spec, prompts)

    raise TypeError(f"eval_prompts signature not supported: {sig}")


# -----------------------
# Utils
# -----------------------
def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_jsonl_append(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_rep_sim(prompts: List[str]) -> List[float]:
    sets = [set(x.lower().split()) for x in prompts]
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


def _pad_1d(ids: torch.Tensor, pad_id: int, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    L = int(ids.shape[0])
    out = torch.full((max_len,), pad_id, dtype=torch.long)
    out[:L] = ids
    mask = torch.zeros((max_len,), dtype=torch.long)
    mask[:L] = 1
    return out, mask


def _filter_kwargs_for_dataclass(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make YAML -> dataclass robust.
    Example: PolicyConfig expects model_id, but yaml uses model_name_or_path.
    We map known aliases and drop unknown keys.
    """
    dd = dict(d or {})
    # aliases
    if cls.__name__ == "PolicyConfig":
        if "model_id" not in dd and "model_name_or_path" in dd:
            dd["model_id"] = dd.pop("model_name_or_path")
        if "model_id" not in dd and "pretrained_model_name_or_path" in dd:
            dd["model_id"] = dd.pop("pretrained_model_name_or_path")

    if cls.__name__ in ("GEPAConfig", "RewardConfig"):
        # keep as-is, but still filter below
        pass

    sig = inspect.signature(cls)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in dd.items() if k in valid}


# -----------------------
# PPO Core
# -----------------------
@dataclass
class PPOHyper:
    learning_rate: float
    num_ppo_epochs: int
    mini_batch_size: int
    clip_range: float
    vf_coef: float
    ent_coef: float
    kl_coef: float
    max_grad_norm: float


class ActorCritic(torch.nn.Module):
    """
    ✅ Use the same base model for policy logits and add a value head.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

        base = self._get_base_model()
        hidden_size = None
        if hasattr(base, "config") and hasattr(base.config, "hidden_size"):
            hidden_size = int(base.config.hidden_size)
        elif hasattr(base, "config") and hasattr(base.config, "n_embd"):
            hidden_size = int(base.config.n_embd)

        if hidden_size is None:
            raise RuntimeError("Cannot infer hidden_size from base model config")

        self.value_head = torch.nn.Linear(hidden_size, 1)

    def _get_base_model(self) -> torch.nn.Module:
        if hasattr(self.model, "pretrained_model"):
            return getattr(self.model, "pretrained_model")
        return self.model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        base = self._get_base_model()
        out = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        logits = out.logits
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()

        hs = out.hidden_states
        if hs is None or len(hs) == 0:
            raise RuntimeError("Model did not return hidden_states.")

        last_h = hs[-1]
        last_h = last_h.to(dtype=self.value_head.weight.dtype)
        values = self.value_head(last_h).squeeze(-1)  # [B,S]
        return logits, values


def _selective_logprob_sum(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    query_lens: List[int],
    resp_lens: List[int],
) -> torch.Tensor:
    logp_all = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [B, S-1, V]
    tgt = input_ids[:, 1:]  # [B, S-1]
    tgt_logp = logp_all.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # [B,S-1]

    B, S1 = tgt_logp.shape
    mask = torch.zeros((B, S1), dtype=torch.float32, device=tgt_logp.device)

    for i, (ql, rl) in enumerate(zip(query_lens, resp_lens)):
        start = max(0, ql - 1)
        end = min(S1, start + rl)
        if end > start:
            mask[i, start:end] = 1.0

    return (tgt_logp * mask).sum(dim=1)


def _gather_last_value(values: torch.Tensor, query_lens: List[int], resp_lens: List[int]) -> torch.Tensor:
    out = []
    for i, (ql, rl) in enumerate(zip(query_lens, resp_lens)):
        pos = max(0, ql + rl - 1)
        pos = min(pos, values.shape[1] - 1)
        out.append(values[i, pos])
    return torch.stack(out, dim=0)


def ppo_update_step(
    ac: ActorCritic,
    optimizer: torch.optim.Optimizer,
    ppo: PPOHyper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    query_lens: List[int],
    resp_lens: List[int],
    rewards: torch.Tensor,
) -> Dict[str, float]:
    device = next(ac.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    rewards = rewards.to(device)

    ac.train()

    with torch.no_grad():
        logits_old, values_old = ac(input_ids, attention_mask)
        old_logp_sum = _selective_logprob_sum(logits_old, input_ids, query_lens, resp_lens)
        old_v = _gather_last_value(values_old, query_lens, resp_lens)

    B = int(input_ids.shape[0])
    mb = max(1, min(int(ppo.mini_batch_size), B))

    stats = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "mean_reward": float(rewards.mean().item()),
    }
    n_steps = 0

    for _epoch in range(int(ppo.num_ppo_epochs)):
        perm = torch.randperm(B, device=device)
        for start in range(0, B, mb):
            idx = perm[start:start + mb]
            ql_mb = [query_lens[i] for i in idx.tolist()]
            rl_mb = [resp_lens[i] for i in idx.tolist()]

            logits, values = ac(input_ids[idx], attention_mask[idx])
            logp_sum = _selective_logprob_sum(logits, input_ids[idx], ql_mb, rl_mb)
            v = _gather_last_value(values, ql_mb, rl_mb)

            adv = (rewards[idx] - old_v[idx]).detach()
            ratio = torch.exp(logp_sum - old_logp_sum[idx])

            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - ppo.clip_range, 1.0 + ppo.clip_range) * adv
            policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

            value_loss = 0.5 * torch.mean((v - rewards[idx]) ** 2)
            approx_kl = torch.mean(old_logp_sum[idx] - logp_sum)
            clipfrac = torch.mean((torch.abs(ratio - 1.0) > ppo.clip_range).float())

            loss = policy_loss + ppo.vf_coef * value_loss + ppo.kl_coef * approx_kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), ppo.max_grad_norm)
            optimizer.step()

            stats["loss"] += float(loss.item())
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["approx_kl"] += float(approx_kl.item())
            stats["clipfrac"] += float(clipfrac.item())
            n_steps += 1

    for k in ["loss", "policy_loss", "value_loss", "approx_kl", "clipfrac"]:
        stats[k] /= max(1, n_steps)

    return stats


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gepa_ppo_prompt.yaml", type=str)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()

    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    log_info(f"[GEPA+PPO] repo_root={REPO_ROOT}")
    log_info(f"[GEPA+PPO] config={cfg_path}")

    # ---------- run dir ----------
    out_root = Path(cfg["run"]["out_root"])
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()

    run_dir = out_root / _now_ts()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"[GEPA+PPO] run_dir={run_dir}")

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    # ---------- task spec ----------
    from src.task_spec import TaskSpec
    spec = TaskSpec.load(cfg["task"]["task_spec_path"]).raw
    user_queries: List[str] = list(cfg["task"]["user_queries"])

    # ---------- policy ----------
    policy_kwargs = _filter_kwargs_for_dataclass(PolicyConfig, cfg["policy_model"])
    policy_cfg = PolicyConfig(**policy_kwargs)
    policy = PromptPolicy(policy_cfg)

    # ---------- actor-critic ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ac = ActorCritic(policy.model).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in ac.parameters() if p.requires_grad],
        lr=float(cfg["ppo"]["learning_rate"]),
    )

    # ---------- PPO hypers ----------
    ppo_h = PPOHyper(
        learning_rate=float(cfg["ppo"]["learning_rate"]),
        num_ppo_epochs=int(cfg["ppo"]["num_ppo_epochs"]),
        mini_batch_size=int(cfg["ppo"]["mini_batch_size"]),
        clip_range=float(cfg["ppo"]["clip_range"]),
        vf_coef=float(cfg["ppo"].get("vf_coef", 0.5)),
        ent_coef=float(cfg["ppo"].get("ent_coef", 0.0)),
        kl_coef=float(cfg["ppo"]["kl_coef"]),
        max_grad_norm=float(cfg["ppo"].get("max_grad_norm", 1.0)),
    )

    # ---------- image generator ----------
    img_gen = ImageGenerator(spec)

    # ---------- reward ----------
    r_kwargs = _filter_kwargs_for_dataclass(RewardConfig, cfg["reward"])
    r_cfg = RewardConfig(**r_kwargs)

    # ---------- GEPA components ----------
    llm = build_llm_client(cfg["llm"])
    memory = TrajectoryMemory(path=str(run_dir / "trajectory_memory.jsonl"))
    skill_lib = SkillLibrary()
    selector = SkillSelector(c=float(cfg["skills"].get("ucb_c", 1.4)))

    gepa_kwargs = _filter_kwargs_for_dataclass(GEPAConfig, cfg["gepa"])
    gepa_cfg = GEPAConfig(**gepa_kwargs)

    gepa = GEPAOptimizer(
        task_spec=spec,
        reward_cfg=r_cfg,
        img_gen=img_gen,
        llm=llm,
        memory=memory,
        skill_lib=skill_lib,
        selector=selector,
        gepa_cfg=gepa_cfg,
    )

    pad_id = policy.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = int(policy.tokenizer.eos_token_id)

    log_path = str(run_dir / "train_log.jsonl")
    global_step = 0

    total_updates = int(cfg["ppo"]["total_updates"])
    num_candidates = int(cfg["multi_fidelity"]["num_prompt_candidates"])
    topk_img = int(cfg["multi_fidelity"]["topk_to_generate_images"])
    n_img = int(cfg["multi_fidelity"]["images_per_prompt"])

    for upd in range(total_updates):
        for user_query in user_queries:
            ep_dir = run_dir / f"upd_{upd:04d}" / f"q_{abs(hash(user_query)) % 100000:05d}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            # 1) seed candidates from current policy
            seeds = policy.sample(user_query, num_samples=num_candidates)
            seeds = [c.strip() for c in seeds if c and c.strip()]
            if len(seeds) == 0:
                continue

            # 2) GEPA loop (self-improving prompt search)
            gepa_res = gepa.optimize(user_query=user_query, seed_prompts=seeds, episode_dir=ep_dir)
            best_prompt = gepa_res["best_prompt"]
            best_reward = float(gepa_res["best_reward"])

            # 3) PPO rollout set:
            #    best prompt + original seeds -> batch
            candidates = list(dict.fromkeys([best_prompt] + seeds))  # unique
            p_eval = _call_eval_prompts(spec, user_query, candidates)
            prompts_only = [x["prompt"] for x in p_eval]
            rep_sims = _prompt_rep_sim(prompts_only)

            # rank by prompt score
            order = list(range(len(p_eval)))
            order.sort(key=lambda i: float(p_eval[i].get("score", 0.0)), reverse=True)
            selected_for_image = set(order[: max(1, min(topk_img, len(order)))])

            samples_q: List[torch.Tensor] = []
            samples_r: List[torch.Tensor] = []
            rewards: List[float] = []

            for i, item in enumerate(p_eval):
                prompt = item["prompt"]
                ps = float(item.get("score", 0.0))
                hard_p = bool(item.get("hard_fail", False))
                tags_p = item.get("tags", [])

                q_ids, r_ids = policy.encode_query_response(user_query, prompt)
                token_len = int(r_ids.shape[0])

                image_score: Optional[float] = None
                hard_i = False
                tags_i: List[str] = []
                img_paths: List[str] = []

                if i in selected_for_image:
                    subdir = ep_dir / f"ppo_cand_{i:02d}"
                    img_paths = img_gen.generate(prompt, out_dir=str(subdir), n=n_img)

                    scores = []
                    for pth in img_paths:
                        res = eval_images(spec, prompt, [pth])[0]
                        scores.append(float(res["score"]))
                        if res.get("hard_fail", False):
                            hard_i = True
                        tags_i.extend(list(res.get("tags", [])))

                    if len(scores) > 0:
                        scores_sorted = sorted(scores, reverse=True)
                        top2 = scores_sorted[: min(2, len(scores_sorted))]
                        image_score = float(sum(top2) / len(top2))

                hard_fail = bool(hard_p or hard_i)
                rep_sim = float(rep_sims[i])

                r = combine_reward(
                    r_cfg,
                    prompt_score=ps,
                    image_score=image_score,
                    token_len=token_len,
                    rep_sim=rep_sim,
                    hard_fail=hard_fail,
                )

                samples_q.append(q_ids.cpu())
                samples_r.append(r_ids.cpu())
                rewards.append(float(r))

                _safe_jsonl_append(log_path, {
                    "global_step": global_step,
                    "update": upd,
                    "user_query": user_query,
                    "prompt": prompt,
                    "prompt_score": ps,
                    "image_score": image_score,
                    "reward": float(r),
                    "token_len": token_len,
                    "rep_sim": rep_sim,
                    "hard_fail": hard_fail,
                    "tags_prompt": tags_p,
                    "tags_image": sorted(list(set(tags_i))),
                    "img_paths": img_paths,
                    "gepa_best_reward": best_reward,
                    "gepa_history": gepa_res.get("history", []),
                })

            # 4) build padded batch for PPO update
            seqs: List[torch.Tensor] = []
            q_lens: List[int] = []
            r_lens: List[int] = []

            for q, r in zip(samples_q, samples_r):
                ql = int(q.shape[0])
                rl = int(r.shape[0])
                q_lens.append(ql)
                r_lens.append(rl)
                seqs.append(torch.cat([q, r], dim=0))

            max_len = max(int(s.shape[0]) for s in seqs)
            input_ids_list = []
            attn_list = []

            for s in seqs:
                ids, am = _pad_1d(s, pad_id=pad_id, max_len=max_len)
                input_ids_list.append(ids)
                attn_list.append(am)

            input_ids = torch.stack(input_ids_list, dim=0)
            attention_mask = torch.stack(attn_list, dim=0)
            reward_t = torch.tensor(rewards, dtype=torch.float32)

            # 5) PPO update
            stats = ppo_update_step(
                ac=ac,
                optimizer=optimizer,
                ppo=ppo_h,
                input_ids=input_ids,
                attention_mask=attention_mask,
                query_lens=q_lens,
                resp_lens=r_lens,
                rewards=reward_t,
            )

            global_step += 1
            log_info(
                f"[GEPA+PPO] upd={upd} step={global_step} "
                f"gepa_best={best_reward:.4f} "
                f"mean_reward={stats['mean_reward']:.4f} "
                f"loss={stats['loss']:.4f} kl={stats['approx_kl']:.4f} clipfrac={stats['clipfrac']:.4f}"
            )

            # checkpoint
            if (global_step % int(cfg["logging"]["save_every"])) == 0:
                ckpt_dir = run_dir / "checkpoints" / f"step_{global_step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.model.save_pretrained(str(ckpt_dir))
                policy.tokenizer.save_pretrained(str(ckpt_dir))
                torch.save(ac.value_head.state_dict(), ckpt_dir / "value_head.pt")
                log_info(f"[GEPA+PPO] saved -> {ckpt_dir}")

    log_info("[GEPA+PPO] ✅ Finished training")
    log_info(f"[GEPA+PPO] logs -> {log_path}")
    log_info(f"[GEPA+PPO] memory -> {run_dir / 'trajectory_memory.jsonl'}")


if __name__ == "__main__":
    main()
