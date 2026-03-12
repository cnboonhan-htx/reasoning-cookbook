# Dataset Generation Plan

## Training Approaches Comparison

| Aspect | TRL (Notebooks) | Cosmos-RL |
|---|---|---|
| **SFT** | `SFTTrainer` with QLoRA, HF conversation format | Native async framework, LLaVA JSON or HF Arrow format |
| **RL** | `GRPOTrainer` with QLoRA (WIP) | Native GRPO with vLLM rollouts (coming soon) |
| **SFT Dataset** | `trl-lib/llava-instruct-mix` — HF messages with `prompt` (list of role/content dicts) + `images` | LLaVA JSON — `conversations` (list of value dicts) + `image` path, media base64-encoded |
| **GRPO Dataset** | `multimodal-open-r1-8k-verified` — `problem`, `image`, `solution` fields, preprocessed into `prompt` string via chat template | Same concept, but RL section not yet released |
| **Reward Functions** | `format_reward` (checks `<think>/<answer>` tags) + `len_reward` (penalizes overthinking, verifies math correctness) | Config-based, `single_choice` reward referenced |
| **Maturity** | Fully working notebooks | SFT works, RL "coming soon" |

## Datasets Needed

### 1. SFT Format (for supervised fine-tuning)
- **LLaVA-style JSON**: `conversations` (user/assistant turns) + `image`/`video` paths
- **HF conversation format**: `prompt` (list of message dicts with typed content) + `images` (PIL)

### 2. GRPO Format (for RL fine-tuning)
- **Problem + image + ground-truth solution** — the solution is used by reward functions to score model completions
- Model must output in `<think>...</think><answer>...</answer>` format

## Synthetic Dataset Generation Plan

### Step 1: Image Generation with Wan2.2
- Use Wan2.2 T2V pipeline (`WanPipeline` with `num_frames=1`) to generate single synthetic images (not video) from text prompts
- Write diverse scene prompts covering target domains (warehouses, roads, construction, factories, etc.)
- Output: `manifest.json` mapping `{id, prompt, image_path}` per generated image
- Script: `generators/generate_images.py`

### Step 2: VLM Response Generation (Teacher Distillation)
- Feed each generated image + task-specific questions to a teacher VLM via OpenAI-compatible API
- Generate `<think>...</think><answer>...</answer>` reasoning chains as training targets
- Question types: spatial reasoning, object identification, safety assessment, navigation
- Output: `{image, question, response}` triples
- Script: `generators/generate_responses.py`

### Step 3: SFT Dataset Construction
- Package `(image, question, response)` triples into LLaVA JSON format:
  ```json
  {"conversations": [{"value": "<question>"}, {"value": "<think>...</think><answer>...</answer>"}], "image": "scene_001.png"}
  ```
- Script: `converters/to_llava_sft.py`

### Step 4: GRPO Dataset Construction
- Extract ground-truth answers from teacher responses and package as:
  ```json
  {"problem": "<question>", "image": "scene_001.png", "solution": "<ground_truth>"}
  ```
- Reuse the existing `format_reward` + adapt `len_reward` to non-math domains (e.g., exact match or semantic similarity instead of `math_verify`)
- Script: `converters/to_grpo.py`

### Step 5: Public Dataset Transformation
- Download and transform existing public datasets to supplement synthetic data:
  - **Nexar collision prediction** (already has a converter in the repo) — dashcam video classification
  - **LLaVA-Instruct-150K** — general visual instruction following
  - **multimodal-open-r1-8k-verified** — math reasoning with images
- Write converters to align these into the two target formats above
- Script: `converters/from_public.py`

### Step 6: Dataset Validation & Splitting
- Validate all entries (images loadable, conversations well-formed, answers non-empty)
- Train/val/test splits
- Statistics reporting (distribution of question types, answer lengths, scene complexity)
- Script: `validation/validate_dataset.py`

## Suggested Directory Structure

```
datasets/
├── README.md
├── PLAN.md
├── Dockerfile
├── generators/
│   ├── generate_images.py       # Step 1: Wan2.2 text-to-image generation
│   ├── generate_responses.py    # Step 2: VLM teacher distillation
│   └── example_prompts.json     # Scene prompt templates
├── converters/
│   ├── to_llava_sft.py          # Step 3: convert to LLaVA SFT format
│   ├── to_grpo.py               # Step 4: convert to GRPO format
│   └── from_public.py           # Step 5: transform public datasets
├── validation/
│   └── validate_dataset.py      # Step 6: entry validation & statistics
└── raw/                         # Downloaded/generated raw data (gitignored)
```
