# SOP-Guided Response Generation Plan

## Concept

Inject relevant SOP context into the VLM system prompt so the teacher model generates responses aligned with domain-specific procedures and expectations.

## Approach

### 1. SOP Document Store
- Place SOP documents (markdown, PDF, text) in `sop/documents/`
- Each SOP covers a domain/task (e.g., `warehouse_safety.md`, `forklift_operation.md`, `road_hazards.md`)

### 2. SOP-to-Prompt Mapping
- Create a mapping file (`sop/sop_index.json`) that links scene categories to relevant SOPs:
  ```json
  {
    "warehouse": ["warehouse_safety.md", "forklift_operation.md"],
    "road": ["road_hazards.md", "traffic_rules.md"],
    "construction": ["construction_safety.md"]
  }
  ```
- Each prompt in `datasets/generators/example_prompts.json` gets a `category` field to look up which SOPs apply

### 3. Context Injection into `generate_responses.py`
- Before querying the VLM, look up the scene's category -> retrieve relevant SOP documents
- Append the SOP content to the system prompt as reference context:
  ```
  You are a visual reasoning assistant. [existing prompt]

  Use the following Standard Operating Procedures as ground truth
  when assessing the scene:

  --- SOP: Warehouse Safety ---
  [SOP content here]
  ---
  ```
- The teacher VLM now reasons against the SOPs (e.g., "the forklift is parked incorrectly per SOP section 3.2")

### 4. SOP-Specific Questions
- Extend question templates to be SOP-aware. Instead of generic "identify safety hazards", ask:
  - "Based on the safety procedures, is the equipment positioned correctly?"
  - "List any violations of standard operating procedures visible in this scene."
- These could live in the SOP index alongside each SOP, or as a separate `sop_questions.json`

### 5. Optional: RAG for Large SOP Sets
- If SOPs are too large to fit in a single prompt, use simple keyword/embedding retrieval to select the most relevant sections
- For a first pass, just concatenate full SOPs (most are short enough) -- add RAG only if context limits become an issue

## Changes Required

| File | Change |
|---|---|
| `datasets/generators/example_prompts.json` | Add `category` field per prompt |
| `datasets/generators/generate_responses.py` | Add `--sop-dir` and `--sop-index` args, inject SOP context into system prompt |
| `sop/documents/` | Place SOP markdown files |
| `sop/sop_index.json` | Category -> SOP file mapping + optional SOP-specific questions |

## Flow

```
prompts.json (with category)
        |
        v
generate_images.py -> images + manifest.json
        |
        v
generate_responses.py
   |-- lookup category from manifest
   |-- load relevant SOPs from sop/documents/
   |-- inject SOP context into system prompt
   |-- query VLM with image + SOP-aware question
   +-- output sft_dataset.json (LLaVA format)
```
