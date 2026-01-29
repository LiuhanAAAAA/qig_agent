# src/agent/skill_library.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


@dataclass
class Skill:
    name: str
    description: str
    trigger_tags: List[str]
    apply: Callable[[str, Dict], str]


def _split_pos_neg(prompt: str) -> Tuple[str, str]:
    s = (prompt or "").strip()
    m = re.search(r"(negative prompt\s*:|负向\s*[:：])", s, flags=re.IGNORECASE)
    if not m:
        return s, ""
    pos = s[:m.start()].strip()
    neg = s[m.end():].strip()
    return pos, neg


def _compose(pos: str, neg: str) -> str:
    pos = (pos or "").strip()
    neg = (neg or "").strip()
    if not neg:
        return pos
    return f"{pos}\nNegative prompt: {neg}"


def _ensure_neg_tokens(prompt: str, tokens: List[str], prepend: bool = True) -> str:
    pos, neg = _split_pos_neg(prompt)
    neg_low = neg.lower()
    need = [t for t in tokens if t.lower() not in neg_low]
    if len(need) == 0:
        return prompt
    add = ", ".join(need)
    if not neg.strip():
        neg = add
    else:
        neg = (add + ", " + neg) if prepend else (neg + ", " + add)
    return _compose(pos, neg)


def _remove_pos_words(prompt: str, patterns: List[str]) -> str:
    pos, neg = _split_pos_neg(prompt)
    for pat in patterns:
        pos = re.sub(pat, "", pos, flags=re.IGNORECASE)
    pos = re.sub(r"\s{2,}", " ", pos).strip()
    return _compose(pos, neg)


def _shorten_prompt(prompt: str, max_words: int = 110) -> str:
    pos, neg = _split_pos_neg(prompt)
    words = pos.split()
    if len(words) <= max_words:
        return prompt
    pos2 = " ".join(words[:max_words])
    return _compose(pos2, neg)


def _expand_prompt(prompt: str) -> str:
    """
    prompt 太短时给一个稳的扩展.规则 patch
    """
    pos, neg = _split_pos_neg(prompt)
    if len(pos.split()) >= 12:
        return prompt
    pos2 = pos.strip()
    pos2 += ", high quality, cinematic lighting, detailed scene, clean composition, sharp focus"
    return _compose(pos2, neg)


class SkillLibrary:
    """
    ✅ Level B：tag -> skill（规则技能）

    """
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self._register_builtin()

    def _register(self, skill: Skill):
        self.skills[skill.name] = skill

    def _register_builtin(self):
        # 1) 禁文字 overlay
        self._register(Skill(
            name="add_neg_text",
            description="Add strong negative tokens to suppress watermark/text overlay",
            trigger_tags=["need_neg_text", "hard_forbid_text", "pos_mentions_text_overlay"],
            apply=lambda prompt, ctx: _ensure_neg_tokens(
                prompt,
                tokens=["text", "watermark", "logo", "signature", "caption", "subtitle"],
                prepend=True
            )
        ))

        # 2)  禁真人脸
        self._register(Skill(
            name="add_neg_realistic_face",
            description="Add negative tokens to suppress photorealistic human faces only",
            trigger_tags=["need_neg_realistic_face", "hard_forbid_realistic_face", "pos_mentions_real_face"],
            apply=lambda prompt, ctx: _ensure_neg_tokens(
                prompt,
                tokens=[
                    "real human", "portrait photo", "photorealistic face",
                    "realistic face", "human face", "skin texture", "real person"
                ],
                prepend=True
            )
        ))

        # 3) 正向里删掉“引导生成文字”的词
        self._register(Skill(
            name="remove_pos_text_words",
            description="Remove words that encourage text overlay in positive prompt",
            trigger_tags=["pos_mentions_text_overlay"],
            apply=lambda prompt, ctx: _remove_pos_words(
                prompt,
                patterns=[r"\b(watermark|logo|signature|caption|subtitle|overlay text|text)\b", r"(文字|水印|字幕|签名|标志)"]
            )
        ))

        # 4) prompt 太短：扩写
        self._register(Skill(
            name="expand_prompt",
            description="Expand prompt when too short (add quality + composition hints)",
            trigger_tags=["prompt_too_short"],
            apply=lambda prompt, ctx: _expand_prompt(prompt)
        ))

        # 5) prompt 太长：截断
        self._register(Skill(
            name="shorten_prompt",
            description="Shorten prompt to reduce CLIP truncation risk",
            trigger_tags=["clip_trunc_risk"],
            apply=lambda prompt, ctx: _shorten_prompt(prompt, max_words=int(ctx.get("max_words", 110)))
        ))

    def available_skills_for_tags(self, tags: List[str]) -> List[Skill]:
        tset = set(tags)
        out = []
        for s in self.skills.values():
            if any(t in tset for t in s.trigger_tags):
                out.append(s)
        return out
