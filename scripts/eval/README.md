# Evaluation Pipeline

This directory contains the end-to-end evaluation script for LLaVA-LE.

## Overview

The pipeline has two main stages:

### 1. Answer Generation — `llava/eval/model_vqa_lunar.py`

Runs a LLaVA-style model on the lunar evaluation set and writes each model's
answers to a `.jsonl` file. This is done for three models:

| Model | Description |
|---|---|
| **LLaVA-LE-S1** | Stage 1 fine-tuned LoRA checkpoint |
| **LLaVA-LE-S2** | Stage 2 fine-tuned LoRA checkpoint |
| **LLaVA** | Base LLaVA (no fine-tuning), vision-language baseline |

GPT and Gemini text-only answer files are generated separately and must exist
before running the judge step.

### 2. Judge Scoring — `llava/eval/caption_judge.py`

Sends all five model answers (LLaVA-LE-S1, LLaVA-LE-S2, GPT-TEXT-ONLY,
GEMINI-TEXT-ONLY, LLaVA) together with the reference caption to a GPT judge
model. The judge scores each answer on a 1–10 scale and produces:

- `judge_scores.jsonl` — per-question scores for all models
- `judge_summary.json` — aggregated statistics (mean, std, per question-type breakdown)

## Usage

```bash
export OPENAI_API_KEY="sk-..."
bash scripts/eval/run_eval.sh [OPTIONS]
```

### Common examples

Run the full pipeline with default paths:
```bash
bash scripts/eval/run_eval.sh
```

Override checkpoint paths:
```bash
bash scripts/eval/run_eval.sh \
  --stage1-model-path LLaVA-LE/checkpoints/stage_1/my_stage1_checkpoint \
  --stage2-model-path LLaVA-LE/checkpoints/stage_2/my_stage2_checkpoint
```

Skip answer generation and only run the judge (answers already exist):
```bash
bash scripts/eval/run_eval.sh --skip-generate
```

Use a different judge model and output directory:
```bash
bash scripts/eval/run_eval.sh \
  --judge-model gpt-4-turbo \
  --output-dir ./my_eval_outputs
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--stage1-model-path` | `LLaVA-LE/checkpoints/stage_1/llava-v1.5-13b-task-lora-stage1_20260221_001701` | Stage 1 LoRA checkpoint |
| `--stage2-model-path` | `LLaVA-LE/checkpoints/stage_2/llava-v1.5-13b-task-lora-stage2_20260225_221920` | Stage 2 LoRA checkpoint |
| `--model-base` | `liuhaotian/llava-v1.5-13b` | Base model ID (HuggingFace or local) |
| `--question-file` | `./dataset/data/lunar_eval_questions.json` | Evaluation question file |
| `--image-folder` | `./dataset/data/lumina_96k/data` | Evaluation image directory |
| `--temperature` | `0.2` | Sampling temperature for generation |
| `--conv-mode` | `vicuna_v1` | Conversation template |
| `--output-dir` | `./eval_outputs` | Root output directory |
| `--judge-model` | `gpt-4o` | OpenAI model used as judge |
| `--gpt-text-only-answers` | `<output-dir>/model_answers/gpt_new_answers.json` | GPT text-only answers file |
| `--gemini-text-only-answers` | `<output-dir>/model_answers/text_gemini_answers.jsonl` | Gemini text-only answers file |
| `--skip-generate` | — | Skip steps 1–3; only run the judge |

## Output structure

```
eval_outputs/
├── model_answers/
│   ├── stage1_answers.jsonl          # LLaVA-LE-S1
│   ├── stage2_answers.jsonl          # LLaVA-LE-S2
│   ├── basellava_answers.jsonl       # LLaVA (base)
│   ├── gpt_new_answers.json          # GPT-TEXT-ONLY  (pre-existing)
│   └── text_gemini_answers.jsonl     # GEMINI-TEXT-ONLY (pre-existing)
└── judge/
    ├── judge_scores.jsonl
    └── judge_summary.json
```
