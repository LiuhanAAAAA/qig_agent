# src/llm/prompt_templates.py
from __future__ import annotations

from typing import Any, Dict, List


def build_gepa_reflection_messages(
    user_query: str,
    current_prompt: str,
    tags: List[str],
    task_spec: Dict[str, Any],
) -> List[Dict[str, str]]:
    forbid_text = bool(task_spec["hard_constraints"]["detect"].get("forbid_text_overlay", False))
    forbid_real_face = bool(task_spec["hard_constraints"]["detect"].get("forbid_realistic_face", False))

    system = (
        "You are a senior prompt optimizer for text-to-image generation.\n"
        "You must rewrite prompts to satisfy constraints and maximize image quality.\n"
        "Return ONLY the rewritten prompt. Do not add explanations.\n"
        "If you include a negative prompt, format it as:\n"
        "Negative prompt: xxx\n"
    )

    constraints = []
    if forbid_text:
        constraints.append("No text overlay, watermark, logo, caption, subtitles.")
    if forbid_real_face:
        constraints.append("No photorealistic human faces / real-person portrait photos. Cartoon/anime faces are allowed.")

    user = (
        f"User query:\n{user_query}\n\n"
        f"Current prompt:\n{current_prompt}\n\n"
        f"Observed failure tags:\n{tags}\n\n"
        f"Constraints:\n- " + "\n- ".join(constraints) + "\n\n"
        "Rewrite the prompt to fix the failures.\n"
        "Keep it concise (<= 110 words for positive prompt).\n"
        "Ensure negative prompt explicitly suppresses the forbidden elements.\n"
        "Output ONLY the rewritten prompt."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
