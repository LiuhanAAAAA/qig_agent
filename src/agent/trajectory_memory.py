# src/agent/trajectory_memory.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryMemory:
    path: str

    def append(self, row: Dict[str, Any]) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_patch(
        self,
        user_query: str,
        before_prompt: str,
        after_prompt: str,
        tags: List[str],
        skill_name: str,
        reward_before: float,
        reward_after: float,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        self.append({
            "user_query": user_query,
            "before_prompt": before_prompt,
            "after_prompt": after_prompt,
            "tags": tags,
            "skill": skill_name,
            "reward_before": float(reward_before),
            "reward_after": float(reward_after),
            "reward_gain": float(reward_after - reward_before),
            "extra": extra or {},
        })
