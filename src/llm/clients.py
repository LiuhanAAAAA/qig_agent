# src/llm/clients.py
from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import requests

from src.utils.logger import log_info, log_warn


class LLMClient(Protocol):
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 256,
    ) -> str:
        ...


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:4b"
    timeout: int = 600
    connect_timeout: int = 5
    retries: int = 2
    retry_backoff: float = 2.0
    think: bool = False   # ✅ 关键：关闭 thinking，保证 content 不为空


class OllamaClient:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self.url = self.cfg.base_url.rstrip("/") + "/api/chat"

        log_info(
            f"[LLM] backend=ollama base_url={self.cfg.base_url} "
            f"model={self.cfg.model} timeout={self.cfg.timeout}s think={self.cfg.think}"
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 256,
    ) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "think": bool(self.cfg.think),  # ✅ 关键
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }

        last_err: Optional[Exception] = None
        for attempt in range(int(self.cfg.retries) + 1):
            try:
                r = requests.post(
                    self.url,
                    json=payload,
                    timeout=(int(self.cfg.connect_timeout), int(self.cfg.timeout)),
                )
                r.raise_for_status()
                data = r.json()

                msg = (data or {}).get("message", {}) or {}
                content = (msg.get("content") or "").strip()

                # ✅ 如果你不小心开了 think=true，这里给个兜底（但默认你应该关闭 think）
                if not content:
                    thinking = (msg.get("thinking") or "").strip()
                    if thinking:
                        log_warn("[OllamaClient] content empty but thinking exists. You should set think=false.")
                        # 不建议用 thinking 当输出，因为里面可能是过程，不是最终 prompt
                        # 这里直接判空让上层 fallback
                    raw = json.dumps(data, ensure_ascii=False)[:500]
                    log_warn(f"[OllamaClient] empty content, raw={raw}")
                    return ""

                return content

            except Exception as e:
                last_err = e
                if attempt < int(self.cfg.retries):
                    wait = float(self.cfg.retry_backoff) * (attempt + 1)
                    log_warn(f"[OllamaClient] request failed ({type(e).__name__}: {e}), retry in {wait:.1f}s ...")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"OllamaClient chat failed: {last_err}")


def build_llm_client(cfg: Dict[str, Any]) -> LLMClient:
    backend = (cfg.get("backend") or "ollama").lower().strip()

    if backend == "ollama":
        ocfg = OllamaConfig(
            base_url=str(cfg.get("base_url", "http://localhost:11434")),
            model=str(cfg.get("model", "qwen3:4b")),
            timeout=int(cfg.get("timeout", 600)),
            connect_timeout=int(cfg.get("connect_timeout", 5)),
            retries=int(cfg.get("retries", 2)),
            retry_backoff=float(cfg.get("retry_backoff", 2.0)),
            think=bool(cfg.get("think", False)),  # ✅ 默认 False
        )
        return OllamaClient(ocfg)

    raise ValueError(f"Unknown llm backend: {backend}")
