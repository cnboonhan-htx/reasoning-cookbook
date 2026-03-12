# Build:
#   docker build --build-arg CUDA_VERSION=12.8.0 -t reasoning-cookbook-datasets datasets/
#
# Run (image generation):
#   docker run --gpus all -v $(pwd)/datasets/raw:/workspace/datasets/raw \
#     reasoning-cookbook-datasets generators/generate_images.py \
#     --prompts-file generators/example_prompts.json --output-dir raw/images

ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git wget unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers \
    transformers accelerate torch ftfy huggingface_hub openai

WORKDIR /workspace
COPY . /workspace/datasets

ENTRYPOINT ["python"]
