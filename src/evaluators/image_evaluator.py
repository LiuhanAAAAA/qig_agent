# src/evaluators/image_evaluator.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import open_clip

from .objective import objective_score
from .taxonomy import taxonomy_from_metrics


def sharpness_score(path: str) -> float:
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.tanh(var / 300.0))


def detect_text_simple(path: str) -> bool:
    img = cv2.imread(path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    ratio = edges.mean() / 255.0
    return bool(ratio > 0.08)


def detect_face_any(path: str) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
    """
    Detect any face-like region (cartoon/realistic both may trigger).
    Returns: (has_face, bboxes)
    """
    img = cv2.imread(path)
    if img is None:
        return False, []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    bboxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    return bool(len(bboxes) > 0), bboxes


def aesthetic_dummy(path: str) -> float:
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean() / 255.0
    std = gray.std() / 255.0
    val = 0.5 * mean + 0.5 * std
    return float(max(0.0, min(1.0, val)))


def _split_pos_neg(prompt: str) -> Tuple[str, str]:
    s = (prompt or "").strip()
    idx = s.lower().find("negative prompt:")
    if idx < 0:
        return s, ""
    pos = s[:idx].strip()
    neg = s[idx + len("negative prompt:"):].strip()
    return pos, neg


def _sanitize_for_clip(text: str) -> str:
    """
    CLIP 对齐用：只保留“正向语义”，避免 hashtags / 垃圾 token / negative 词污染。
    """
    t = (text or "").strip()
    # remove hashtags
    t = " ".join([w for w in t.split() if not w.startswith("#")])
    # remove bracket noise like [1080p] [END] etc
    t = cv2.regex.substitute if False else t  # keep for compatibility (no-op)
    t = t.replace("[END", "").replace("] [", " ").replace("[", " ").replace("]", " ")
    # collapse spaces
    t = " ".join(t.split())
    return t[:300]  # keep it short


# -----------------------------
# CLIP singleton (IMPORTANT for speed)
# -----------------------------
_GLOBAL_CLIP: Optional["CLIPModel"] = None


class CLIPModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._text_cache: Dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def encode_image(self, image_path: str) -> torch.Tensor:
        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        img_f = self.model.encode_image(image)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        return img_f  # [1,D]

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        key = text.strip().lower()
        if key in self._text_cache:
            return self._text_cache[key]
        toks = self.tokenizer([text]).to(self.device)
        txt_f = self.model.encode_text(toks)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        self._text_cache[key] = txt_f
        return txt_f  # [1,D]

    @torch.inference_mode()
    def score(self, prompt_text: str, image_path: str) -> float:
        img_f = self.encode_image(image_path)
        txt_f = self.encode_text(prompt_text)
        sim = (img_f * txt_f).sum(dim=-1).item()
        return float((sim + 1) / 2)

    @torch.inference_mode()
    def similarity(self, image_path: str, texts: List[str]) -> List[float]:
        img_f = self.encode_image(image_path)
        outs: List[float] = []
        for t in texts:
            txt_f = self.encode_text(t)
            sim = (img_f * txt_f).sum(dim=-1).item()
            outs.append(float(sim))
        return outs


def _get_clipper() -> CLIPModel:
    global _GLOBAL_CLIP
    if _GLOBAL_CLIP is None:
        _GLOBAL_CLIP = CLIPModel()
    return _GLOBAL_CLIP


def detect_realistic_human_face(
    image_path: str,
    clipper: CLIPModel,
    face_bboxes: List[Tuple[int, int, int, int]],
    threshold: float = 0.65,
) -> Tuple[bool, float]:
    """
    ✅ 只禁“真人脸/写真风格的脸”，卡通脸允许
    流程：
      1) 先检测到 face bbox（任何脸）
      2) 对最大 face crop 做 CLIP 二分类：
         photorealistic human face  vs  cartoon/anime face
    返回：(is_realistic_face, realism_prob)
    """
    if not face_bboxes:
        return False, 0.0

    # choose largest face
    x, y, w, h = max(face_bboxes, key=lambda b: b[2] * b[3])

    # read & crop
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # expand a bit
    pad = int(0.25 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    crop = img.crop((x0, y0, x1, y1))

    # temp pathless CLIP: run preprocess directly by PIL -> tensor
    # We'll reuse clipper's preprocess by writing small helper
    # Create an in-memory image embedding by temporarily saving is overkill;
    # simplest: save crop to numpy and reopen not needed; instead use preprocess on PIL:
    with torch.inference_mode():
        crop_t = clipper.preprocess(crop).unsqueeze(0).to(clipper.device)
        img_f = clipper.model.encode_image(crop_t)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        photo_texts = [
            "a photorealistic photo of a human face",
            "a realistic portrait photo of a person",
            "a DSLR photo of a human face, natural skin texture",
        ]
        cartoon_texts = [
            "a cartoon face",
            "an anime character face",
            "a cute illustration of an animal character",
            "a chibi style mascot face",
        ]

        sims_photo = []
        sims_cart = []
        for t in photo_texts:
            txt_f = clipper.encode_text(t)
            sims_photo.append((img_f * txt_f).sum(dim=-1).item())
        for t in cartoon_texts:
            txt_f = clipper.encode_text(t)
            sims_cart.append((img_f * txt_f).sum(dim=-1).item())

        s_photo = float(np.mean(sims_photo))
        s_cart = float(np.mean(sims_cart))

        # logistic on margin
        margin = s_photo - s_cart
        realism_prob = float(1.0 / (1.0 + np.exp(-5.0 * margin)))  # steeper

    return bool(realism_prob >= threshold), float(realism_prob)


def eval_images(task_spec: Dict[str, Any], prompt: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    返回：
      - score
      - metrics
      - tags
      - hard_fail
    """
    clipper = _get_clipper()
    out: List[Dict[str, Any]] = []

    hard = task_spec.get("hard_constraints", {}).get("detect", {})
    forbid_text = bool(hard.get("forbid_text_overlay", False))

    # ✅ 注意：这里是 forbid_realistic_face（不是 forbid_any_face）
    forbid_real_face = bool(hard.get("forbid_realistic_face", False))

    pos, _neg = _split_pos_neg(prompt)
    clip_text = _sanitize_for_clip(pos)

    for p in image_paths:
        # alignment: use only positive prompt
        clip = clipper.score(clip_text, p)
        sharp = sharpness_score(p)
        aest = aesthetic_dummy(p)

        has_text = detect_text_simple(p) if forbid_text else False

        # face-any then realistic-face
        has_face_any, bboxes = detect_face_any(p)
        is_real_face = False
        realism_prob = 0.0
        if forbid_real_face and has_face_any:
            is_real_face, realism_prob = detect_realistic_human_face(p, clipper, bboxes, threshold=0.65)

        # hard constraint fail tags
        hard_fail: List[str] = []
        if has_text and forbid_text:
            hard_fail.append("hard_forbid_text")
        if is_real_face and forbid_real_face:
            hard_fail.append("hard_forbid_face")  # keep old tag name for compatibility

        penalties: Dict[str, float] = {}

        if sharp < 0.25:
            penalties["blurry_penalty"] = 0.10
        if clip < 0.45:
            penalties["low_clip_penalty"] = 0.08
        if has_text:
            penalties["text_penalty"] = 0.20

        # ✅ 只对“真人脸”处罚
        if is_real_face:
            penalties["face_penalty"] = 0.25

        metrics = {
            "clip_alignment": float(clip),
            "sharpness": float(sharp),
            "aesthetic": float(aest),
            "has_text": bool(has_text),

            # compatibility key:
            "has_face": bool(has_face_any),  # whether any face-like region exists
            # new keys:
            "has_realistic_face": bool(is_real_face),
            "real_face_prob": float(realism_prob),
        }

        total, tags = objective_score(
            task_spec,
            clip, sharp, aest,
            penalties,
            prompt=prompt,
            metrics=metrics,
            tags=[]
        )

        tax_tags = taxonomy_from_metrics(task_spec, metrics)

        # add explicit tags
        extra_tags: List[str] = []
        if has_face_any:
            extra_tags.append("has_face")
        if is_real_face:
            extra_tags.append("has_realistic_face")

        tags = sorted(list(set(tags + tax_tags + hard_fail + extra_tags)))

        out.append({
            "image_path": p,
            "score": float(total),
            "metrics": metrics,
            "tags": tags,
            "hard_fail": bool(len(hard_fail) > 0)
        })

    return out
