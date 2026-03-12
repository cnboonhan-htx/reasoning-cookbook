"""
Visualize the sft_dataset using renumics-spotlight with images rendered properly.

Usage:
    python visualize_dataset.py ./raw/sft_dataset
    python visualize_dataset.py ./raw/sft_dataset --host 0.0.0.0 --port 8080
"""

import argparse
import io

import pandas as pd
from datasets import load_from_disk
from renumics import spotlight
from renumics.spotlight import Image


def main():
    parser = argparse.ArgumentParser(description="Visualize sft_dataset with Spotlight")
    parser.add_argument("dataset_path", help="Path to the HuggingFace dataset directory")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_path)

    # Build a DataFrame with the first image extracted as PNG bytes for Spotlight
    records = []
    for row in ds:
        img_bytes = None
        if row["images"]:
            pil_img = row["images"][0]
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

        prompt_text = "\n".join(f"[{m['role']}] {m['content']}" for m in row["prompt"])
        completion_text = "\n".join(f"[{m['role']}] {m['content']}" for m in row["completion"])

        records.append({
            "image": img_bytes,
            "prompt": prompt_text,
            "completion": completion_text,
        })

    df = pd.DataFrame(records)

    spotlight.show(
        df,
        dtype={"image": Image, "prompt": str, "completion": str},
        host=args.host,
        port=args.port,
        wait=True,
    )


if __name__ == "__main__":
    main()
