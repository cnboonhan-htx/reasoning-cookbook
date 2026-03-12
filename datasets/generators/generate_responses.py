"""
Generate VLM reasoning responses for synthetic images (Step 2 & 3 of PLAN.md).

Reads the manifest.json produced by generate_images.py, sends each image + questions
to a teacher VLM via OpenAI-compatible API, and outputs a HuggingFace dataset
matching the trl-lib/llava-instruct-mix schema.

Usage:
    python generate_responses.py --manifest-file ./raw/images/manifest.json --output-dir ./raw/sft_dataset
    python generate_responses.py --manifest-file ./raw/images/manifest.json --output-dir ./raw/sft_dataset --base-url http://127.0.0.1:8317/v1
    python generate_responses.py --manifest-file ./raw/images/manifest.json --output-dir ./raw/sft_dataset --base-url http://127.0.0.1:8317/v1 --model-id claude-haiku-4-5-20251001

Environment variables:
    OPENAI_API_KEY      - API key for the VLM endpoint (required)
    OPENAI_BASE_URL     - Base URL for OpenAI-compatible endpoint (default: https://api.openai.com/v1)
    VLM_MODEL_ID        - Model ID to use (default: claude-haiku-4-5-20251001)

Dependencies:
    pip install openai datasets Pillow
"""

import argparse
import base64
import json
import logging
import os
from pathlib import Path

from datasets import Dataset, Features, Image, Sequence, Value, load_from_disk
from openai import OpenAI
from PIL import Image as PILImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("VLM_MODEL_ID", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """\
You are a visual reasoning assistant. When given an image and a question, \
think step by step about what you observe, then provide a concise answer.

You MUST format your response exactly as:
<think>your reasoning here</think>
<answer>your answer here</answer>"""

QUESTION_TEMPLATES = [
    "Describe the scene in detail. What objects are present and where are they located?",
    "Identify any potential safety hazards or risks visible in this scene.",
    "If an autonomous agent were navigating this environment, what obstacles should it be aware of?",
    "What actions could a robot perform in this scene? List the interactive objects.",
    "Describe the spatial relationships between the main objects in this image.",
]

DATASET_FEATURES = Features({
    "images": [Image(decode=True)],
    "prompt": [{"role": Value("string"), "content": Value("string")}],
    "completion": [{"role": Value("string"), "content": Value("string")}],
})


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vlm(client: OpenAI, model: str, image_path: str, question: str) -> str:
    image_b64 = encode_image_base64(image_path)
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    {"type": "text", "text": question},
                ],
            },
        ],
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Generate VLM responses for synthetic images")
    parser.add_argument("--manifest-file", required=True, help="Path to manifest.json from generate_images.py")
    parser.add_argument("--output-dir", required=True, help="Path to output HuggingFace dataset directory")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="VLM model ID")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--questions-file", default=None, help="JSON file with custom questions (list of strings). If not provided, uses built-in templates.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY environment variable is required")
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    with open(args.manifest_file) as f:
        manifest = json.load(f)
    logger.info(f"Loaded {len(manifest)} entries from {args.manifest_file}")

    if args.questions_file:
        with open(args.questions_file) as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} custom questions")
    else:
        questions = QUESTION_TEMPLATES
        logger.info(f"Using {len(questions)} built-in question templates")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load existing progress if dataset exists (resumable)
    rows = []
    seen = set()
    progress_file = output_path / "_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            rows = json.load(f)
        seen = {(r["_image_path"], r["_question"]) for r in rows}
        logger.info(f"Resuming: {len(rows)} entries already generated")

    total = len(manifest) * len(questions)
    done = len(seen)

    for entry in manifest:
        image_path = entry["image"]
        prompt_id = entry["id"]

        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}, skipping {prompt_id}")
            continue

        for question in questions:
            if (image_path, question) in seen:
                continue

            done += 1
            logger.info(f"[{done}/{total}] {prompt_id}: {question[:60]}...")

            try:
                response = query_vlm(client, args.model_id, image_path, question)
                rows.append({
                    "_image_path": image_path,
                    "_question": question,
                    "prompt": [{"role": "user", "content": question}],
                    "completion": [{"role": "assistant", "content": response}],
                })

                # Save progress after each response
                with open(progress_file, "w") as f:
                    json.dump(rows, f, indent=2)
            except Exception as e:
                logger.error(f"Failed for {prompt_id}: {e}")
                continue

    # Copy images into output directory and build final HuggingFace dataset
    logger.info(f"Building HuggingFace dataset with {len(rows)} entries...")
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_lists = []
    for r in rows:
        src = Path(r["_image_path"])
        dst = images_dir / src.name
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)
        image_lists.append([str(dst)])

    dataset_dict = {
        "images": image_lists,
        "prompt": [r["prompt"] for r in rows],
        "completion": [r["completion"] for r in rows],
    }
    ds = Dataset.from_dict(dataset_dict, features=DATASET_FEATURES)
    ds.save_to_disk(str(output_path))

    # Clean up progress file
    progress_file.unlink(missing_ok=True)

    logger.info(f"Done. Dataset saved to {output_path} ({len(ds)} rows)")
    logger.info(f"Schema: {ds.features}")


if __name__ == "__main__":
    main()
