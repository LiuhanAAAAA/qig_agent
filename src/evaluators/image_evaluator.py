# src/evaluators/image_evaluator.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
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
    """
    ⚠️ 这个是非常粗糙的 text/overlay proxy，
    先用边缘密度做判断（极轻量）。
    """
    img = cv2.imread(path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    ratio = edges.mean() / 255.0
    return bool(ratio > 0.08)


def _haar_face_boxes(path: str) -> List[Tuple[int, int, int, int]]:
    img = cv2.imread(path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    out = []
    for (x, y, w, h) in faces:
        out.append((int(x), int(y), int(w), int(h)))
    return out


class CLIPModel:
    """
    用于：
    1) prompt-image alignment 分数
    2) “真人脸 vs 非真人脸(卡通/插画)”判别（轻量）
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.inference_mode()
    def score(self, prompt: str, image_path: str) -> float:
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = self.tokenizer([prompt]).to(self.device)

        img_f = self.model.encode_image(image)
        txt_f = self.model.encode_text(text)

        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        sim = (img_f * txt_f).sum(dim=-1).item()
        return float((sim + 1) / 2)

    @torch.inference_mode()
    def realistic_face_prob(self, image_path: str) -> float:
        """
        ✅ 只要“像照片的真人脸”概率
        不是做人脸检测，而是“整体视觉风格是否像真人头像照片”
        """
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        prompts = [
            "a photorealistic close-up portrait photo of a real human face",
            "a real person portrait photo, realistic skin texture, DSLR photo",
            "a cartoon face illustration, anime portrait, 2d drawing",
            "a stylized character illustration, painting, digital art",
        ]
        text = self.tokenizer(prompts).to(self.device)

        img_f = self.model.encode_image(image)
        txt_f = self.model.encode_text(text)

        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

        sims = (img_f @ txt_f.t()).squeeze(0)  # [4]
        sims = sims.float()

        # photo score vs cartoon score
        photo_score = float(torch.max(sims[0:2]).item())
        cartoon_score = float(torch.max(sims[2:4]).item())

        # logistic-like
        margin = photo_score - cartoon_score
        prob = 1.0 / (1.0 + np.exp(-3.0 * margin))
        return float(prob)


def aesthetic_dummy(path: str) -> float:
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean() / 255.0
    std = gray.std() / 255.0
    val = 0.5 * mean + 0.5 * std
    return float(max(0.0, min(1.0, val)))


def detect_realistic_face(path: str, clipper: CLIPModel) -> bool:
    """
    ✅ 只禁“真人脸”：两个条件同时满足才判定
    A) Haar 能检测到疑似脸框
    B) CLIP 判定整体风格偏真人头像照片
    """
    boxes = _haar_face_boxes(path)
    if len(boxes) == 0:
        return False

    # CLIP photo portrait probability
    prob = clipper.realistic_face_prob(path)
    return bool(prob >= 0.65)


def eval_images(
    task_spec: Dict[str, Any],
    prompt: str,
    image_paths: List[str]
) -> List[Dict[str, Any]]:
    clipper = CLIPModel()
    out: List[Dict[str, Any]] = []

    forbid_text = bool(task_spec["hard_constraints"]["detect"].get("forbid_text_overlay", False))
    # ✅ 只禁“真人脸”
    forbid_real_face = bool(task_spec["hard_constraints"]["detect"].get("forbid_realistic_face", False))

    for p in image_paths:
        clip = clipper.score(prompt, p)
        sharp = sharpness_score(p)
        aest = aesthetic_dummy(p)

        has_text = detect_text_simple(p) if forbid_text else False
        has_real_face = detect_realistic_face(p, clipper) if forbid_real_face else False

        hard_fail: List[str] = []
        if has_text and forbid_text:
            hard_fail.append("hard_forbid_text")
        if has_real_face and forbid_real_face:
            hard_fail.append("hard_forbid_realistic_face")

        penalties: Dict[str, float] = {}
        if sharp < 0.25:
            penalties["blurry_penalty"] = 0.10
        if clip < 0.45:
            penalties["low_clip_penalty"] = 0.08
        if has_text:
            penalties["text_penalty"] = 0.20
        if has_real_face:
            penalties["real_face_penalty"] = 0.25

        metrics = {
            "clip_alignment": float(clip),
            "sharpness": float(sharp),
            "aesthetic": float(aest),
            "has_text": bool(has_text),
            "has_realistic_face": bool(has_real_face),
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
        tags = sorted(list(set(tags + tax_tags + hard_fail)))

        out.append({
            "image_path": p,
            "score": float(total),
            "metrics": metrics,
            "tags": tags,
            "hard_fail": bool(len(hard_fail) > 0)
        })

    return out
