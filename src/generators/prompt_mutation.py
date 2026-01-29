import random
from typing import List, Dict

def mutate_prompt(prompt: str, task_spec: Dict, m: int = 2) -> List[str]:
    """
    后面没用这个了。用GEPA mutation LLM了。
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
