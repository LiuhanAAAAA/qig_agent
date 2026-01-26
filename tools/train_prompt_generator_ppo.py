# tools/train_prompt_generator_ppo.py
from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from src.utils.logger import log_info
from src.generators.image_generator import ImageGenerator

# 你的 PPO policy（保持不动）
from src.rl.prompt_policy import PromptPolicy, PolicyConfig
from src.rl.reward import RewardConfig, combine_reward

# ✅ 用你仓库自己的 prompt evaluator（不兜底、不降级）
from src.evaluators.prompt_evaluator import eval_prompts

# ✅ Agent loop / skill / memory（新增）
from src.agent.skill_library import SkillLibrary
from src.agent.trajectory_memory import TrajectoryMemory
from src.agent.agent_loop import AgentLoopConfig, run_agent_episode


# -----------------------
# prompt evaluator adapter（只做接口适配，不做降级）
# -----------------------
def _call_eval_prompts(spec: Dict[str, Any], user_query: str, prompts: List[str]):
    """
    ✅只做“接口适配”，不做任何降级：
    兼容 eval_prompts 的几种常见签名：
      - eval_prompts(spec, prompts)
      - eval_prompts(user_query, prompts)
      - eval_prompts(spec, user_query, prompts)
    """
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
        # 命名不标准时，默认 (spec, prompts)
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
    """
    轻量“防雷同”proxy：Jaccard（不引入额外 embedding 依赖）
    值越大越像（需要扣分）
    """
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
    """
    ids: [L]
    return:
      input_ids: [max_len]
      attn_mask: [max_len]
    """
    L = int(ids.shape[0])
    out = torch.full((max_len,), pad_id, dtype=torch.long)
    out[:L] = ids
    mask = torch.zeros((max_len,), dtype=torch.long)
    mask[:L] = 1
    return out, mask


# -----------------------
# PPO Core (custom step)
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
    Actor-Critic wrapper:
    - actor: 语言模型输出 logits
    - critic: 用最后一层 hidden state 做一个 value head -> values [B,S]

    ✅关键修复：
    你的 policy.model 很可能是 TRL 的 AutoModelForCausalLMWithValueHead 之类 wrapper，
    直接 forward 它不一定会返回 hidden_states。
    所以我们优先走 model.pretrained_model（标准 HF 模型）来确保 hidden_states 一定拿到。
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

        # ✅拿“底座模型”来推断 hidden_size（更稳定）
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
        # TRL AutoModelForCausalLMWithValueHead 通常有 pretrained_model
        if hasattr(self.model, "pretrained_model"):
            return getattr(self.model, "pretrained_model")
        return self.model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Returns:
            logits: [B,S,V]
            values: [B,S]
        """
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
            raise RuntimeError(
                "Model did not return hidden_states. "
                "Even base model has no hidden_states -> check transformers version / model support."
            )

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
    """
    计算每个 sample 的 response token logprob_sum（只对 response 部分求和）
    logits: [B,S,V]
    input_ids: [B,S]
    return: [B]
    """
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

    return (tgt_logp * mask).sum(dim=1)  # [B]


def _gather_last_value(values: torch.Tensor, query_lens: List[int], resp_lens: List[int]) -> torch.Tensor:
    """
    values: [B,S]
    return: [B] (取 response 最后一个 token 的 value)
    """
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
    """
    自己实现 PPO 的一次 update（支持多 epoch + mini-batch）
    """
    device = next(ac.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    rewards = rewards.to(device)

    ac.train()

    with torch.no_grad():
        logits_old, values_old = ac(input_ids, attention_mask)
        old_logp_sum = _selective_logprob_sum(logits_old, input_ids, query_lens, resp_lens)  # [B]
        old_v = _gather_last_value(values_old, query_lens, resp_lens)  # [B]

    B = int(input_ids.shape[0])
    mb = int(ppo.mini_batch_size)
    mb = max(1, min(mb, B))

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

            adv = rewards[idx] - old_v[idx]
            adv = adv.detach()

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_prompt.yaml", type=str)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # ---------- run dir ----------
    out_root = cfg["run"]["out_root"]
    run_dir = Path(out_root) / _now_ts()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"[PPO] run_dir={run_dir}")

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    # ---------- task spec ----------
    from src.task_spec import TaskSpec
    spec = TaskSpec.load(cfg["task"]["task_spec_path"]).raw
    user_queries: List[str] = list(cfg["task"]["user_queries"])

    # ---------- policy ----------
    policy_cfg = PolicyConfig(**cfg["policy_model"])
    policy = PromptPolicy(policy_cfg)

    # ---------- actor-critic ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ac = ActorCritic(policy.model).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in ac.parameters() if p.requires_grad],
        lr=float(cfg["ppo"]["learning_rate"])
    )

    # ---------- PPO hypers ----------
    num_ppo_epochs = int(cfg["ppo"].get("num_ppo_epochs", cfg["ppo"].get("ppo_epochs", 2)))
    ppo_h = PPOHyper(
        learning_rate=float(cfg["ppo"]["learning_rate"]),
        num_ppo_epochs=num_ppo_epochs,
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
    r_cfg = RewardConfig(**cfg["reward"])

    # ---------- agent loop configs ----------
    agent_cfg_raw = cfg.get("agent_loop", {})
    agent_cfg = AgentLoopConfig(
        enabled=bool(agent_cfg_raw.get("enabled", True)),
        max_steps=int(agent_cfg_raw.get("max_steps", 4)),
        max_images=int(agent_cfg_raw.get("max_images", 4)),
        images_per_step=int(agent_cfg_raw.get("images_per_step", 1)),
        stop_score=float(agent_cfg_raw.get("stop_score", 0.82)),
        stop_if_no_hard_fail=bool(agent_cfg_raw.get("stop_if_no_hard_fail", True)),
        ucb_c=float(agent_cfg_raw.get("ucb_c", 1.0)),
        mutate_on_failure=bool(agent_cfg_raw.get("mutate_on_failure", True)),
        mutate_min_trials=int(agent_cfg_raw.get("mutate_min_trials", 6)),
        mutate_max_success_rate=float(agent_cfg_raw.get("mutate_max_success_rate", 0.20)),
    )

    # ---------- skill library & memory ----------
    skill_path = str(run_dir / "skill_library.json")
    mem_path = str(run_dir / "trajectory_memory.json")

    skill_lib = SkillLibrary.load(skill_path)
    memory = TrajectoryMemory.load(mem_path)

    # ---------- loop ----------
    total_updates = int(cfg["ppo"]["total_updates"])
    num_candidates = int(cfg["multi_fidelity"]["num_prompt_candidates"])

    pad_id = policy.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = int(policy.tokenizer.eos_token_id)

    # logs
    ppo_log_path = str(run_dir / "ppo_log.jsonl")
    agent_log_path = str(run_dir / "agent_trajectory.jsonl")

    global_step = 0

    for upd in range(total_updates):
        for user_query in user_queries:
            global_step += 1

            # 1) sample prompt candidates（用当前 policy 采样）
            candidates = policy.sample(user_query, num_samples=num_candidates)
            candidates = [c.strip() for c in candidates if c and c.strip()]
            if len(candidates) == 0:
                continue

            # 2) prompt eval（cheap）
            p_eval = _call_eval_prompts(spec, user_query, candidates)

            # 3) agent episodes（Level A 核心：每个候选 prompt 自己循环修复）
            final_prompts: List[str] = []
            final_rewards: List[float] = []
            final_tokenlens: List[int] = []
            final_image_scores: List[Optional[float]] = []
            final_tags: List[List[str]] = []

            # 用于 memory 更新：reward_before/after
            # 我们用 episode 的 trajectory 里相邻步 reward 差更新 stats
            for i, item in enumerate(p_eval):
                init_prompt = str(item["prompt"]).strip()
                if not init_prompt:
                    continue

                # 如果不开 agent loop，就退回原始“单次评估”（但你要 agent，所以默认开）
                if agent_cfg.enabled:
                    ep = run_agent_episode(
                        spec=spec,
                        user_query=user_query,
                        init_prompt=init_prompt,
                        img_gen=img_gen,
                        tokenizer=policy.tokenizer,
                        reward_cfg=r_cfg,
                        skill_lib=skill_lib,
                        memory=memory,
                        out_dir=str(run_dir / f"upd_{upd:04d}"),
                        global_step=global_step * 1000 + i,
                        log_jsonl=agent_log_path,
                        cfg=agent_cfg,
                    )

                    best_prompt = str(ep["best_prompt"])
                    best_reward = float(ep["best_reward"])
                    best_img_score = ep.get("best_image_score", None)
                    best_tags = list(ep.get("best_tags", []))

                    # ✅用 trajectory 相邻 reward 差更新 memory（Level C）
                    traj = ep.get("trajectory", [])
                    for t in range(1, len(traj)):
                        prev_r = float(traj[t - 1].get("reward", 0.0))
                        cur_r = float(traj[t].get("reward", 0.0))
                        delta = cur_r - prev_r

                        # 找 step(t-1) 的 apply_skill 记录（在 jsonl 里也有，但这里更稳）
                        # 简化：用 prev 的 tags_all 和一个“最可能的 skill”更新（仍然有效）
                        # 更精确你可以把 skill_id 从 apply_skill 行里读出来（后续我也能帮你做）
                        tags_all = list(traj[t - 1].get("tags_all", []))
                        # 用 ensure_structure 作为默认 skill 归因（保守）
                        skill_id = "ensure_structure"
                        if tags_all:
                            # 取第一个 tag 的“最佳技能”作为归因（近似 credit assignment）
                            skill_id = skill_lib.choose_skill_ucb(tags_all, memory.get_stats(), ucb_c=float(agent_cfg.ucb_c)).skill_id

                        for tg in tags_all:
                            memory.update(tg, skill_id, delta_reward=float(delta))

                    final_prompts.append(best_prompt)
                    final_rewards.append(best_reward)
                    final_image_scores.append(best_img_score)
                    final_tags.append(best_tags)

                    try:
                        tl = int(policy.tokenizer(best_prompt, return_tensors="pt")["input_ids"].shape[-1])
                    except Exception:
                        tl = int(len(best_prompt.split()))
                    final_tokenlens.append(tl)

                else:
                    # fallback（一般不用）
                    best_prompt = init_prompt
                    ps = float(item.get("score", 0.0))
                    hard_fail = bool(item.get("hard_fail", False))
                    token_len = 0
                    try:
                        token_len = int(policy.tokenizer(best_prompt, return_tensors="pt")["input_ids"].shape[-1])
                    except Exception:
                        token_len = int(len(best_prompt.split()))

                    r = combine_reward(
                        r_cfg,
                        prompt_score=ps,
                        image_score=None,
                        token_len=token_len,
                        rep_sim=0.0,
                        hard_fail=hard_fail,
                    )
                    final_prompts.append(best_prompt)
                    final_rewards.append(float(r))
                    final_tokenlens.append(token_len)
                    final_image_scores.append(None)
                    final_tags.append(list(item.get("tags", [])))

            if len(final_prompts) == 0:
                continue

            # 4) rep_sim（在最终 prompt 上算更合理）
            rep_sims = _prompt_rep_sim(final_prompts)

            # 5) build padded batch for PPO update（用最终 prompt 来 update policy）
            samples_q: List[torch.Tensor] = []
            samples_r: List[torch.Tensor] = []
            rewards_t: List[float] = []

            for ptxt, rwd, rep_sim in zip(final_prompts, final_rewards, rep_sims):
                q_ids, r_ids = policy.encode_query_response(user_query, ptxt)
                token_len = int(r_ids.shape[0])

                # ✅把 rep_sim 写回 reward（保持你 reward 结构完整）
                r2 = combine_reward(
                    r_cfg,
                    prompt_score=None,       # combine_reward 内部如果用不到可忽略；否则你也可以把 prompt_score传进去
                    image_score=None,        # 这里保持最终 reward 不变（已经在 agent loop 里计算过）
                    token_len=token_len,
                    rep_sim=float(rep_sim),
                    hard_fail=False,
                )
                # 注意：r2 只是加了 rep_sim 的惩罚项（如果你 combine_reward 依赖 prompt_score/image_score，你可以改 reward.py 支持直接 reward_override）
                # 为了不破坏你现有实现，这里直接用“agent_reward - rep_sim*alpha”也行
                final_r = float(rwd) - float(r_cfg.rep_sim_coef) * float(rep_sim) if hasattr(r_cfg, "rep_sim_coef") else float(rwd)

                samples_q.append(q_ids.cpu())
                samples_r.append(r_ids.cpu())
                rewards_t.append(final_r)

            # pad
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

            input_ids = torch.stack(input_ids_list, dim=0)        # [B,S]
            attention_mask = torch.stack(attn_list, dim=0)         # [B,S]
            reward_tensor = torch.tensor(rewards_t, dtype=torch.float32)  # [B]

            # 6) PPO update
            stats = ppo_update_step(
                ac=ac,
                optimizer=optimizer,
                ppo=ppo_h,
                input_ids=input_ids,
                attention_mask=attention_mask,
                query_lens=q_lens,
                resp_lens=r_lens,
                rewards=reward_tensor,
            )

            # 7) logs
            mean_r = float(np.mean(rewards_t)) if len(rewards_t) else 0.0
            log_info(
                f"[PPO+AgentLoop] upd={upd} step={global_step} "
                f"mean_reward={mean_r:.4f} "
                f"loss={stats['loss']:.4f} kl={stats['approx_kl']:.4f} clipfrac={stats['clipfrac']:.4f}"
            )

            _safe_jsonl_append(ppo_log_path, {
                "global_step": global_step,
                "update": upd,
                "user_query": user_query,
                "num_samples": len(final_prompts),
                "mean_reward": mean_r,
                "loss": stats["loss"],
                "kl": stats["approx_kl"],
                "clipfrac": stats["clipfrac"],
                "final_prompts": final_prompts,
                "final_rewards": rewards_t,
                "final_image_scores": final_image_scores,
                "final_tags": final_tags,
            })

            # 8) save checkpoint
            if (global_step % int(cfg["logging"]["save_every"])) == 0:
                ckpt_dir = run_dir / "checkpoints" / f"step_{global_step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                policy.model.save_pretrained(str(ckpt_dir))
                policy.tokenizer.save_pretrained(str(ckpt_dir))
                torch.save(ac.value_head.state_dict(), ckpt_dir / "value_head.pt")

                # ✅保存 skill / memory（Level C 的核心产物）
                skill_lib.save(str(ckpt_dir / "skill_library.json"))
                memory.save(str(ckpt_dir / "trajectory_memory.json"))

                log_info(f"[PPO] saved -> {ckpt_dir}")

            # 每步也落盘一次（避免中断丢失）
            skill_lib.save(skill_path)
            memory.save(mem_path)

    log_info("[PPO] ✅ Finished training")
    log_info(f"[PPO] logs -> {ppo_log_path}")
    log_info(f"[AgentLoop] trajectories -> {agent_log_path}")
    log_info(f"[AgentLoop] skill_library -> {skill_path}")
    log_info(f"[AgentLoop] memory -> {mem_path}")


if __name__ == "__main__":
    main()
