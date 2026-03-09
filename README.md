# LLaVA-LE: Large Language-and-Vision Assistant for Lunar Exploration

**LLaVA-LE** is a domain-adapted vision-language model (VLM) for lunar surface analysis, built on top of [LLaVA-v1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b). It is fine-tuned in two stages using parameter-efficient LoRA adapters on the **LUCID** dataset — a curated collection of lunar imagery paired with descriptive captions (Stage 1) and instruction-following Q&A (Stage 2). LLaVA-LE enables detailed, grounded responses to natural-language questions about geological features, rock formations, crater morphology, and surface characteristics visible in lunar imagery.

---

## Updates

- **[2026-02-21] Codebase released.** The full training and evaluation code for LLaVA-LE is now publicly available in this repository.
- 🤗 **[2026-02-21] LUCID dataset on Hugging Face.** The LUCID dataset ([`pcvlab/lucid`](https://huggingface.co/datasets/pcvlab/lucid)) is publicly available and is directly integrated into this codebase — no local data files or image folders are required.

---

## Environment Setup

### Requirements

- Python 3.10
- PyTorch 2.1.2 (CUDA 12.1)
- CUDA-capable GPU(s) — multi-GPU recommended for training

### 1. Create and activate the conda environment

```bash
conda create -n llavale python=3.10 -y
conda activate llavale
```

### 2. Install PyTorch

```bash
pip install torch==2.1.2 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install the package and core dependencies

```bash
pip install -e .
```

### 4. Install training extras (DeepSpeed, Flash Attention, WandB)

```bash
pip install -e ".[train]"
```

> Flash Attention 2 (`flash-attn`) requires a CUDA-capable GPU and may take several minutes to compile. If compilation fails, ensure `ninja` is installed and your CUDA toolkit matches the PyTorch CUDA version.

---

## Training

Training is done in two sequential stages using DeepSpeed ZeRO-3. Both stages stream data directly from HuggingFace — no local dataset download is needed.

### Stage 1 — Caption Alignment

Aligns the LoRA adapters and multimodal projector to lunar imagery using ~96 K caption-style samples (`stage1_captions` config).

```bash
bash scripts/train/train_stage_1.sh
```

Checkpoints are saved automatically under:

```
checkpoints/stage_1/<run_name>_<YYYYMMDD_HHMMSS>/
```

### Stage 2 — Instruction Tuning

Fine-tunes on ~81 K lunar Q&A instruction turns (`stage2_qa` config), warm-started from the Stage 1 LoRA weights.

```bash
# Using the default Stage 1 checkpoint
bash scripts/train/train_stage_2.sh

# Using a custom Stage 1 checkpoint
bash scripts/train/train_stage_2.sh \
  --lora-weight-path checkpoints/stage_1/<your_stage1_run>
```

Checkpoints are saved under:

```
checkpoints/stage_2/<run_name>_<YYYYMMDD_HHMMSS>/
```

For a full parameter reference (learning rates, LoRA settings, DeepSpeed config, etc.), see [scripts/train/README.md](scripts/train/README.md).

---

## Evaluation

Evaluation uses a GPT-4o judge to score model answers on the LUCID evaluation split (50 images, 190 questions, `evaluation` config). The pipeline generates answers for LLaVA-LE-S1, LLaVA-LE-S2, and base LLaVA, then scores all five models (including pre-existing GPT and Gemini text-only answers) in a single judge pass.

### Prerequisites

1. An OpenAI API key with access to `gpt-4o`.
2. Pre-generated GPT and Gemini text-only answer files (see [scripts/eval/README.md](scripts/eval/README.md) for expected file paths).

### Run

```bash
export OPENAI_API_KEY="sk-..."
bash scripts/eval/run_eval.sh
```

Override checkpoint paths if needed:

```bash
bash scripts/eval/run_eval.sh \
  --stage1-model-path checkpoints/stage_1/<your_stage1_run> \
  --stage2-model-path checkpoints/stage_2/<your_stage2_run>
```

Skip answer generation if `.jsonl` files already exist:

```bash
bash scripts/eval/run_eval.sh --skip-generate
```

Results are written to `eval_outputs/judge/`:

```
eval_outputs/
├── model_answers/
│   ├── stage1_answers.jsonl
│   ├── stage2_answers.jsonl
│   └── basellava_answers.jsonl
└── judge/
    ├── judge_scores.jsonl     # per-question scores for all models
    └── judge_summary.json     # aggregated mean / std per model
```

For all available flags, see [scripts/eval/README.md](scripts/eval/README.md).

---

## Citation

If you use LLaVA-LE or the LUCID dataset in your work, please cite:

```bibtex
@article{inal2025llavale,
  title       = {LLaVA-LE: Large Language-and-Vision Assistant for Lunar Exploration},
  author      = {Inal, Gokce and Navard, Pouyan and Yilmaz, Alper},
  journal     = {arXiv preprint},
  year        = {2025},
  note        = {Under review},
  institution = {The Ohio State University}
}
```
