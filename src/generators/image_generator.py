# src/generators/image_generator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from src.utils.logger import log_info


class ImageGenerator:


    def __init__(self, task_spec: Dict[str, Any]):
        self.spec = task_spec
        gen_cfg = (task_spec.get("generation") or {})
        enforce = (task_spec.get("hard_constraints") or {}).get("enforce", {}) or {}

        # ---- 基本生成参数 ----
        size = enforce.get("image_size", [1024, 1024])
        self.width = int(size[0])
        self.height = int(size[1])
        self.steps = int(gen_cfg.get("steps", 25))
        self.guidance_scale = float(gen_cfg.get("guidance_scale", 7.0))

      
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_cpu_offload = bool(gen_cfg.get("enable_cpu_offload", True))


        self.decode_device = str(gen_cfg.get("decode_device", "auto")).lower()

        model_id = gen_cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        log_info(f"[ImageGenerator] model_id={model_id}")
        log_info(f"[ImageGenerator] size={self.width}x{self.height} steps={self.steps}")
        log_info(
            f"[ImageGenerator] device={self.device} cpu_offload={self.enable_cpu_offload} decode_device={self.decode_device}"
        )

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None,
        )

        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            self.pipe.vae.enable_slicing()
        except Exception:
            pass
        try:
            self.pipe.vae.enable_tiling()
        except Exception:
            pass

        if self.enable_cpu_offload and self.device == "cuda":
           
            self.pipe.enable_model_cpu_offload()
            log_info("[INFO] enable_model_cpu_offload ON")
        else:
            self.pipe.to(self.device)

        # 黑图稳定性：把 VAE 的 dtype 上浮到 fp32（不会强制搬设备）
        try:
            self.pipe.vae.to(dtype=torch.float32)
            log_info("[INFO] VAE upcast to fp32 (stable decode)")
        except Exception:
            pass

    def _get_execution_device(self) -> torch.device:
       
        exec_dev = getattr(self.pipe, "_execution_device", None)
        if exec_dev is None:
            exec_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return exec_dev

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        返回 decoded image tensor: [B, 3, H, W]
        """
        vae = self.pipe.vae

        # SDXL scaling
        scaling = getattr(self.pipe.vae.config, "scaling_factor", 0.13025)
        exec_dev = self._get_execution_device()

        # ---- 决定 decode_device ----
        mode = self.decode_device

        if mode == "cpu":
            vae.to("cpu", dtype=torch.float32)
            target_device = torch.device("cpu")
            target_dtype = torch.float32

        elif mode == "cuda":
            if torch.cuda.is_available():
                vae.to("cuda")
                target_device = torch.device("cuda")
                # VAE dtype：
                target_dtype = next(vae.parameters()).dtype
            else:
                target_device = torch.device("cpu")
                target_dtype = torch.float32

        else:
            target_device = exec_dev
            # dtype 用 VAE 当前 dtype（一般 fp32）
            try:
                target_dtype = next(vae.parameters()).dtype
            except Exception:
                target_dtype = torch.float32

        latents = (latents / scaling).to(device=target_device, dtype=target_dtype)

        decoded = vae.decode(latents, return_dict=False)[0]
        return decoded

    @torch.no_grad()
    def generate(self, prompt: str, out_dir: str, n: int = 1) -> List[str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        result = self.pipe(
            prompt=prompt,
            num_inference_steps=self.steps,
            width=self.width,
            height=self.height,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=int(n),
            output_type="latent",
        )
        latents = result.images  # latent tensor [B, C, H/8, W/8]

        decoded = self._decode_latents(latents)

        # postprocess -> PIL
        images: List[Image.Image] = self.pipe.image_processor.postprocess(
            decoded, output_type="pil"
        )

        paths: List[str] = []
        for i, img in enumerate(images[:n]):
            p = out_path / f"gen_{i:02d}.png"
            img.save(p)
            paths.append(str(p))
        return paths
