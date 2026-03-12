"""
Generate synthetic training images using Wan2.2 text-to-image.

Wan2.2 is a video model (WanPipeline) that generates single images when num_frames=1.

Usage:
    python generate_images.py --prompts-file prompts.json --output-dir ./raw/images
    python generate_images.py --prompts-file prompts.json --output-dir ./raw/images --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers  

Dependencies:
    pip install git+https://github.com/huggingface/diffusers transformers accelerate torch ftfy
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default: 14B model (~80GB VRAM)
# For 24GB VRAM, use: export WAN_MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B-Diffusers (~8GB VRAM)
DEFAULT_MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, deformed, ugly, worst quality, overexposed, "
    "subtitles, watermark, text, banner"
)


def load_pipeline(model_id: str, device: str = "cuda") -> WanPipeline:
    logger.info(f"Loading VAE from {model_id}")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

    logger.info(f"Loading pipeline from {model_id}")
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to(device)
    return pipe


def generate_image(
    pipe: WanPipeline,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    height: int = 720,
    width: int = 1280,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 40,
):
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=1,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    frame = output.frames[0][0]
    if isinstance(frame, Image.Image):
        return frame
    frame = np.squeeze(frame)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(frame)


def load_prompts(prompts_file: str) -> list[dict]:
    """Load prompts from a JSON file.

    Expected format:
    [
        {"id": "scene_001", "prompt": "A warehouse with stacked boxes and a forklift"},
        {"id": "scene_002", "prompt": "An outdoor construction site with safety cones"},
        ...
    ]
    """
    with open(prompts_file) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic images with Wan2.2")
    parser.add_argument("--prompts-file", required=True, help="Path to JSON file with prompts")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated images")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    pipe = load_pipeline(args.model_id, device=args.device)

    manifest = []
    for i, entry in enumerate(prompts):
        prompt_id = entry.get("id", f"image_{i:04d}")
        prompt_text = entry["prompt"]
        output_path = output_dir / f"{prompt_id}.png"

        if output_path.exists():
            logger.info(f"[{i+1}/{len(prompts)}] Skipping {prompt_id} (already exists)")
            manifest.append({"id": prompt_id, "prompt": prompt_text, "image": str(output_path)})
            continue

        logger.info(f"[{i+1}/{len(prompts)}] Generating: {prompt_text[:80]}...")
        image = generate_image(
            pipe,
            prompt=prompt_text,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        )
        image.save(output_path)
        manifest.append({"id": prompt_id, "prompt": prompt_text, "image": str(output_path)})
        logger.info(f"  Saved to {output_path}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
