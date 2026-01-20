# src/evaluators/image_evaluator.py
from __future__ import annotations

from typing import Dict, Any, List
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


def detect_face_simple(path: str) -> bool:
    img = cv2.imread(path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return bool(len(faces) > 0)


class CLIPModel:
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


def aesthetic_dummy(path: str) -> float:
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean() / 255.0
    std = gray.std() / 255.0
    val = 0.5 * mean + 0.5 * std
    return float(max(0.0, min(1.0, val)))


def eval_images(task_spec: Dict[str, Any], prompt: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    clipper = CLIPModel()
    out: List[Dict[str, Any]] = []

    forbid_text = bool(task_spec["hard_constraints"]["detect"].get("forbid_text_overlay", False))
    forbid_face = bool(task_spec["hard_constraints"]["detect"].get("forbid_realistic_face", False))

    for p in image_paths:
        clip = clipper.score(prompt, p)
        sharp = sharpness_score(p)
        aest = aesthetic_dummy(p)

        has_text = detect_text_simple(p) if forbid_text else False
        has_face = detect_face_simple(p) if forbid_face else False

        # hard constraint fail tags
        hard_fail: List[str] = []
        if has_text and forbid_text:
            hard_fail.append("hard_forbid_text")
        if has_face and forbid_face:
            hard_fail.append("hard_forbid_face")

        penalties: Dict[str, float] = {}
        if sharp < 0.25:
            penalties["blurry_penalty"] = 0.10
        if clip < 0.45:
            penalties["low_clip_penalty"] = 0.08
        if has_text:
            penalties["text_penalty"] = 0.20
        if has_face:
            penalties["face_penalty"] = 0.25

        metrics = {
            "clip_alignment": float(clip),
            "sharpness": float(sharp),
            "aesthetic": float(aest),
            "has_text": bool(has_text),
            "has_face": bool(has_face),
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
