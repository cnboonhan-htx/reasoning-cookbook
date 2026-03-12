# Datasets

Dataset download, transformation, and validation utilities. Post-processing scripts to align collected data with the formats needed for TRL and cosmos-rl training.

## Setup

```bash
conda create -n reasoning-cookbook python=3.10
conda activate reasoning-cookbook 
pip install git+https://github.com/huggingface/diffusers transformers accelerate torch ftfy huggingface_hub openai datasets Pillow renumics-spotlight
```

## Example Datasets
```
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers
hf download trl-lib/llava-instruct-mix --repo-type dataset
hf download liuhaotian/LLaVA-Instruct-150K --repo-type dataset
wget http://images.cocodataset.org/zips/train2017.zip -O /tmp/media.zip && unzip -q /tmp/media.zip -d /tmp # Copy it to your destination
```

## Dataset Synthesis Pipeline
```
python generate_images.py --prompts-file example_prompts.json --output-dir ./raw/images --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers 
python generate_responses.py --manifest-file ./raw/images/manifest.json --output-dir ./raw/sft_dataset --base-url http://127.0.0.1:8317/v1 --model-id claude-haiku-4-5-20251001
```

## Visualize Datasets
```
python visualize_dataset.py ./raw/sft_dataset
```

## Push to HF
```
python -c "from datasets import load_from_disk; load_from_disk('./raw/sft_dataset').push_to_hub('cnboonhan-htx/reasoning-cookbook')" 
```