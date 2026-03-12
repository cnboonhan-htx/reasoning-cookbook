"""
Generate VLM reasoning responses for synthetic images (Step 2 & 3 of PLAN.md).

Reads the manifest.json produced by generate_images.py, sends each image + questions
to a teacher VLM via OpenAI-compatible API, and outputs a LLaVA-style SFT dataset.

Usage:
    python generate_responses.py --manifest-file ./raw/images/manifest.json --output-file ./raw/sft_dataset.json
    python generate_responses.py --manifest-file ./raw/images/manifest.json --output-file ./raw/sft_dataset.json --base-url http://127.0.0.1:3456/v1
    python generate_responses.py --manifest-file ./raw/images/manifest.json --output-file ./raw/sft_dataset.json --base-url http://127.0.0.1:3456/v1 --model-id claude-haiku-4

Environment variables:
    OPENAI_API_KEY      - API key for the VLM endpoint
    OPENAI_BASE_URL     - Base URL for OpenAI-compatible endpoint (default: https://api.openai.com/v1)
    VLM_MODEL_ID        - Model ID to use (default: claude-haiku-4)

Dependencies:
    pip install openai
"""

import argparse
import json
import logging
import os
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("VLM_MODEL_ID", "claude-haiku-4")

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



def query_vlm(client: OpenAI, model: str, image_path: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    return response.choices[0].message.content


def build_llava_entry(image_path: str, question: str, response: str) -> dict:
    """Package into LLaVA SFT format (Step 3)."""
    return {
        "image": image_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": response},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate VLM responses for synthetic images")
    parser.add_argument("--manifest-file", required=True, help="Path to manifest.json from generate_images.py")
    parser.add_argument("--output-file", required=True, help="Path to output LLaVA SFT JSON dataset")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="VLM model ID")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--questions-file", default=None, help="JSON file with custom questions (list of strings). If not provided, uses built-in templates.")
    args = parser.parse_args()

    client_kwargs = {"api_key": os.environ.get("OPENAI_API_KEY", "none")}
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

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing progress if output file exists (resumable)
    dataset = []
    seen = set()
    if output_path.exists():
        with open(output_path) as f:
            dataset = json.load(f)
        seen = {(e["image"], e["conversations"][0]["value"]) for e in dataset}
        logger.info(f"Resuming: {len(dataset)} entries already generated")

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
                llava_entry = build_llava_entry(image_path, question, response)
                dataset.append(llava_entry)

                # Save after each response for resumability
                with open(output_path, "w") as f:
                    json.dump(dataset, f, indent=2)
            except Exception as e:
                logger.error(f"Failed for {prompt_id}: {e}")
                continue

    logger.info(f"Done. {len(dataset)} total entries saved to {output_path}")


if __name__ == "__main__":
    main()
