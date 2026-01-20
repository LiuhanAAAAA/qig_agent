from dataclasses import dataclass
from typing import Dict, Any
import yaml

@dataclass
class TaskSpec:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "TaskSpec":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TaskSpec(raw=data)

    def __getitem__(self, k):
        return self.raw[k]
