#!/usr/bin/env python3
"""
Flux2Klein Style Transfer — 最简推理脚本
用法:
    python infer.py --content content.jpg --style style.jpg --output result.png
    python infer.py --content content.jpg --style style.jpg --output result.png --seed 42 --steps 20 --cfg 4
"""

import argparse
import os

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

# ─── 配置 ────────────────────────────────────────────────────────────────────

BASE_MODEL_ID        = os.getenv("BASE_MODEL_ID",        "black-forest-labs/FLUX.2-klein-9B")
TUNED_REPO_ID        = os.getenv("TUNED_REPO_ID",        "wyjlu/omnistyle2-klein9b-base")
TUNED_WEIGHTS_FILE   = os.getenv("TUNED_WEIGHTS_FILENAME","step-3000.safetensors")
HF_TOKEN             = os.getenv("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DEFAULT_PROMPT = "Transfer the style of Figure 2 into Figure 1"

# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image) -> Image.Image:
    """Center-crop to 1:1, resize to 1024×1024."""
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w + side) // 2, (h + side) // 2))
    return img.resize((1024, 1024), Image.Resampling.LANCZOS)


def download_base_model():
    """下载基础模型，返回本地路径字典。"""
    cache_dir = snapshot_download(
        repo_id=BASE_MODEL_ID,
        token=HF_TOKEN or None,
        allow_patterns=[
            "text_encoder/*.safetensors",
            "transformer/*.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "tokenizer/*",
        ],
    )
    from pathlib import Path
    root = Path(cache_dir)
    return {
        "cache_dir":          str(root),
        "text_encoder_paths": sorted(str(p) for p in (root / "text_encoder").glob("*.safetensors")),
        "transformer_paths":  sorted(str(p) for p in (root / "transformer").glob("*.safetensors")),
        "vae_path":           str(root / "vae" / "diffusion_pytorch_model.safetensors"),
    }


def load_pipeline():
    """加载 pipeline 并注入 LoRA 权重。"""
    from diffsynth.core import load_state_dict
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    print("[1/3] Resolving model files...")
    base = download_base_model()
    tuned = hf_hub_download(
        repo_id=TUNED_REPO_ID,
        filename=TUNED_WEIGHTS_FILE,
        token=HF_TOKEN or None,
    )
    print(f"      text_encoder : {len(base['text_encoder_paths'])} shard(s)")
    print(f"      transformer  : {len(base['transformer_paths'])} shard(s)")
    print(f"      vae          : {base['vae_path']}")
    print(f"      tuned weights: {tuned}")

    print("[2/3] Loading pipeline...")
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=DTYPE,
        device=DEVICE,
        model_configs=[
            ModelConfig(path=base["text_encoder_paths"]),
            ModelConfig(path=base["transformer_paths"]),
            ModelConfig(path=base["vae_path"]),
        ],
        tokenizer_config=ModelConfig(
            model_id=BASE_MODEL_ID,
            origin_file_pattern="tokenizer/",
        ),
    )

    print("[3/3] Injecting tuned weights...")
    state_dict = load_state_dict(tuned, torch_dtype=DTYPE)
    pipe.dit.load_state_dict(state_dict)

    return pipe


# ─── 推理入口 ─────────────────────────────────────────────────────────────────

def run(content_path: str, style_path: str, output_path: str,
        seed: int = 1, steps: int = 20, cfg: float = 4.0):

    content = preprocess(Image.open(content_path))
    style   = preprocess(Image.open(style_path))

    pipe = load_pipeline()

    print(f"\nGenerating: seed={seed}, steps={steps}, cfg={cfg}")
    output = pipe(
        DEFAULT_PROMPT,
        edit_image=[content, style],
        seed=seed,
        rand_device="cuda" if DEVICE.startswith("cuda") else "cpu",
        num_inference_steps=steps,
        cfg_scale=cfg,
        height=1024,
        width=1024,
    )

    if isinstance(output, list):
        output = output[0]

    output.save(output_path)
    print(f"Saved → {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ.setdefault("DISABLE_FLASH_ATTN", "1")
    os.environ.setdefault("XFORMERS_DISABLED",  "1")

    parser = argparse.ArgumentParser(description="Flux2Klein Style Transfer")
    parser.add_argument("--content", required=True,  help="Path to content image")
    parser.add_argument("--style",   required=True,  help="Path to style image")
    parser.add_argument("--output",  required=True,  help="Path for output image (e.g. result.png)")
    parser.add_argument("--seed",    type=int,   default=1,   help="Random seed (default: 1)")
    parser.add_argument("--steps",   type=int,   default=20,  help="Inference steps (default: 20)")
    parser.add_argument("--cfg",     type=float, default=4.0, help="CFG scale (default: 4.0)")
    args = parser.parse_args()

    run(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
    )
