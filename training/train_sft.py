"""
SFT training script for Cosmos-Reason2 using TRL + QLoRA.

Based on: https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/notebooks/trl_sft.ipynb

Usage:
    python train_sft.py
    python train_sft.py --dataset-name my-org/my-dataset
    python train_sft.py --dataset-name ./local/sft_dataset --from-disk
    python train_sft.py --max-steps 100 --batch-size 4 --lr 1e-4

Environment variables:
    DATASET_NAME     - HuggingFace dataset name or local path (default: trl-lib/llava-instruct-mix)
    HF_TOKEN         - HuggingFace token for gated models/datasets
"""

import argparse
import os

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="SFT training for Cosmos-Reason2")
    parser.add_argument(
        "--dataset-name",
        default=os.environ.get("DATASET_NAME", "trl-lib/llava-instruct-mix"),
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument("--from-disk", action="store_true", help="Load dataset from disk instead of HF Hub")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to use")
    parser.add_argument("--model-name", default="nvidia/Cosmos-Reason2-2B", help="Base model name")
    parser.add_argument("--output-dir", default="/workspace/outputs/Cosmos-Reason2-2B-trl-sft")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    if args.from_disk:
        train_dataset = load_from_disk(args.dataset_name)
    else:
        train_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    print(f"Dataset loaded: {len(train_dataset)} samples")

    # Load model with QLoRA 4-bit quantization
    print(f"Loading model: {args.model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "down_proj",
            "o_proj",
            "k_proj",
            "q_proj",
            "gate_proj",
            "up_proj",
            "v_proj",
        ],
    )

    # Training config
    training_args = SFTConfig(
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        optim="adamw_8bit",
        max_length=None,
        output_dir=args.output_dir,
        logging_steps=1,
        report_to="tensorboard",
        bf16=True,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    # Log GPU info
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    trainer_stats = trainer.train()

    # Log training stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']:.1f} seconds used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")

    # Save model
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
