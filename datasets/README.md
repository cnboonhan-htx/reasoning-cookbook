# Datasets

Dataset download, transformation, and validation utilities. Post-processing scripts to align collected data with the formats needed for TRL and cosmos-rl training.

## Setup

```bash
conda create -n reasoning-cookbook python=3.10
conda activate reasoning-cookbook 
pip install git+https://github.com/huggingface/diffusers transformers accelerate torch ftfy huggingface_hub openai
```

## Example Datasets
```
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers
hf download trl-lib/llava-instruct-mix --repo-type dataset
hf download liuhaotian/LLaVA-Instruct-150K --repo-type dataset
wget http://images.cocodataset.org/zips/train2017.zip -O /tmp/media.zip && unzip -q /tmp/media.zip -d /tmp # Copy it to your destination
```
