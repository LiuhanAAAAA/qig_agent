# src/agent/skill_selector.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SkillStats:
    n: int = 0
    mean_gain: float = 0.0


@dataclass
class SkillSelector:
    """
    ✅ UCB1 selector：让 Agent “越用越聪明”
    """
    c: float = 1.4
    stats: Dict[str, SkillStats] = field(default_factory=dict)
    total_tries: int = 0

    def update(self, skill_name: str, reward_gain: float) -> None:
        self.total_tries += 1
        st = self.stats.get(skill_name, SkillStats())
        st.n += 1
        st.mean_gain += (reward_gain - st.mean_gain) / float(st.n)
        self.stats[skill_name] = st

    def select(self, skill_names: List[str]) -> Optional[str]:
        if len(skill_names) == 0:
            return None

        # cold-start：先都试一遍
        for s in skill_names:
            if self.stats.get(s, SkillStats()).n == 0:
                return s

        # UCB score
        best_s = None
        best_v = -1e9
        for s in skill_names:
            st = self.stats.get(s, SkillStats())
            bonus = self.c * math.sqrt(math.log(max(1, self.total_tries)) / float(st.n))
            v = st.mean_gain + bonus
            if v > best_v:
                best_v = v
                best_s = s
        return best_s
