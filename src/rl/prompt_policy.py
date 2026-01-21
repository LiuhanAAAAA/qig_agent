# src/rl/prompt_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None

try:
    from trl import AutoModelForCausalLMWithValueHead
except Exception:
    AutoModelForCausalLMWithValueHead = None


@dataclass
class PolicyConfig:
    model_id: str
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_new_tokens: int = 160
    temperature: float = 0.9
    top_p: float = 0.92


class PromptPolicy:
    """
    PPO policy: 给一个 user_query，输出一个“正向 prompt”（不含多余解释）。
    """

    def __init__(self, cfg: PolicyConfig):
        if AutoModelForCausalLMWithValueHead is None:
            raise RuntimeError("TRL not installed. Please `pip install trl` first.")

        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load base model (optionally 4bit)
        quant_kwargs: Dict[str, Any] = {}
        if cfg.use_4bit and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                quant_kwargs["device_map"] = "auto"
            except Exception:
                # bitsandbytes 不可用就退化（仍可跑，但显存压力大）
                quant_kwargs = {}

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            **quant_kwargs,
        )

        # LoRA
        if (LoraConfig is not None) and (get_peft_model is not None):
            lora = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            base = get_peft_model(base, lora)

        # value head wrapper (for PPO)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(base)

        if self.device == "cuda" and "device_map" not in quant_kwargs:
            self.model = self.model.to(self.device)

        self.model.train()

    def build_input(self, user_query: str) -> str:
        """
        你工业 demo 最好固定“输出格式”，这样 reward 才稳定。
        """
        return (
            "You are a professional text-to-image prompt engineer.\n"
            "Task: write ONE single high-quality POSITIVE prompt for the user request.\n"
            "Rules:\n"
            "- Output ONLY the prompt, no explanations.\n"
            "- Keep it compact but descriptive.\n"
            "- Avoid political symbols, watermark, logo, text overlays.\n"
            f"User request: {user_query}\n"
            "Prompt:"
        )

    @torch.no_grad()
    def sample(self, user_query: str, num_samples: int = 1) -> List[str]:
        prompt_in = self.build_input(user_query)
        enc = self.tokenizer(prompt_in, return_tensors="pt").to(self.device)

        outs = self.model.generate(
            **enc,
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=True,
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            pad_token_id=self.tokenizer.eos_token_id,
        )

        results: List[str] = []
        for i in range(min(num_samples, outs.shape[0])):
            text = self.tokenizer.decode(outs[i], skip_special_tokens=True)
            # 只取 "Prompt:" 后面的部分
            if "Prompt:" in text:
                text = text.split("Prompt:", 1)[-1].strip()
            results.append(text.strip())
        return results

    def encode_query_response(self, user_query: str, response: str):
        """
        PPOTrainer 需要 query_tensors / response_tensors
        """
        query_text = self.build_input(user_query)
        q = self.tokenizer(query_text, return_tensors="pt")["input_ids"][0]
        r = self.tokenizer(response, return_tensors="pt")["input_ids"][0]
        return q, r
