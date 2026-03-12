# Environment

SimWorld installation/setup scripts and agent interaction layer. Includes an agent interface that connects model inference to SimWorld actions (observe scene, reason, act), basic primitives (move agent, interact with objects, capture observations), and a simple evaluation loop (scenario -> observation -> model reasoning -> action -> outcome).

## Setup

```bash
# Clone submodule
git submodule update --init --recursive environment/SimWorld

# Create and activate conda environment
conda create -n simworld python=3.10
conda activate simworld

# Install SimWorld
cd SimWorld
pip install -e .

# Download Unreal Backend
wget https://huggingface.co/datasets/SimWorld-AI/SimWorld/resolve/main/Base/Linux.zip
unzip Linux.zip
./Linux/SimWorld.sh /Game/Maps/demo_1.umap
python simworld_iteration_1.pssssss
```
