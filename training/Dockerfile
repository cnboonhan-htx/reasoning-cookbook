# Build:
#   docker build -t cosmos-reason2-sft training/
#
# Run (default dataset: trl-lib/llava-instruct-mix):
#   docker run --gpus all -v $(pwd)/outputs:/workspace/outputs cosmos-reason2-sft
#
# Run (custom HF dataset):
#   docker run --gpus all -e DATASET_NAME=my-org/my-dataset -v $(pwd)/outputs:/workspace/outputs cosmos-reason2-sft
#
# Run (local dataset from disk):
#   docker run --gpus all -v $(pwd)/datasets/generators/raw/sft_dataset:/data/sft_dataset -v $(pwd)/outputs:/workspace/outputs \
#     cosmos-reason2-sft --dataset-name /data/sft_dataset --from-disk
#
# Run (override training params):
#   docker run --gpus all -v $(pwd)/outputs:/workspace/outputs cosmos-reason2-sft --max-steps 100 --batch-size 4 --lr 1e-4

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Install PyTorch with CUDA 12.8 support (H100 compatible)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install training dependencies
RUN pip install --no-cache-dir \
    "trl[peft]==0.26.1" \
    "bitsandbytes==0.49.0" \
    "tensorboard==2.20.0" \
    transformers \
    accelerate \
    datasets \
    huggingface_hub \
    Pillow \
    qwen-vl-utils

WORKDIR /workspace
COPY train_sft.py /workspace/train_sft.py

ENV DATASET_NAME=trl-lib/llava-instruct-mix

ENTRYPOINT ["python", "train_sft.py"]
