import random
from typing import List, Dict

def mutate_prompt(prompt: str, task_spec: Dict, m: int = 2) -> List[str]:
    """
    简单 mutation：替换/插入风格词、构图词，让你有“算法味”的可控探索。
    """
    inserts = [
        "soft pastel color palette",
        "high contrast clean silhouette",
        "minimalist modern design",
        "studio lighting, high clarity",
        "centered composition, close-up portrait"
    ]
    outs = []
    for _ in range(m):
        add = random.choice(inserts)
        outs.append(prompt + ", " + add)
    return outs
