#!/usr/bin/env bash
# =============================================================================
# run_eval.sh  —  Full evaluation pipeline for LLaVA-LE
#
# Steps
#   1. Generate answers for LLaVA-LE-S1  (Stage 1 fine-tuned model)
#   2. Generate answers for LLaVA-LE-S2  (Stage 2 fine-tuned model)
#   3. Generate answers for base LLaVA   (no fine-tuning, vision baseline)
#   4. Run caption_judge.py to score all models with a GPT judge
#
# Evaluation questions and images are streamed directly from the LUCID
# HuggingFace dataset — no local data files are required.
# GPT/Gemini text-only answer files must already exist before running step 4.
# Pass --skip-generate to jump straight to the judge step.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
STAGE1_MODEL_PATH="LLaVA-LE/checkpoints/stage_1/llava-v1.5-13b-task-lora-stage1_20260221_001701"
STAGE2_MODEL_PATH="LLaVA-LE/checkpoints/stage_2/llava-v1.5-13b-task-lora-stage2_20260225_221920"
MODEL_BASE="liuhaotian/llava-v1.5-13b"
HF_DATASET="pcvlab/lucid"
TEMPERATURE="0.2"
CONV_MODE="vicuna_v1"
OUTPUT_DIR="./eval_outputs"
JUDGE_MODEL="gpt-4o"

# Paths for pre-existing text-only answer files (GPT / Gemini)
GPT_TEXT_ONLY_ANSWERS="${OUTPUT_DIR}/model_answers/gpt_new_answers.json"
GEMINI_TEXT_ONLY_ANSWERS="${OUTPUT_DIR}/model_answers/text_gemini_answers.jsonl"

SKIP_GENERATE=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bash run_eval.sh [OPTIONS]

Model paths:
  --stage1-model-path PATH       Path to the Stage 1 LoRA checkpoint
                                 (default: LLaVA-LE/checkpoints/stage_1/llava-v1.5-13b-task-lora-stage1_20260221_001701)
  --stage2-model-path PATH       Path to the Stage 2 LoRA checkpoint
                                 (default: LLaVA-LE/checkpoints/stage_2/llava-v1.5-13b-task-lora-stage2_20260225_221920)
  --model-base MODEL             HuggingFace base model ID or local path
                                 (default: ${MODEL_BASE})

Dataset:
  --hf-dataset REPO              HuggingFace dataset repo ID for evaluation questions and images
                                 (default: ${HF_DATASET})

Generation:
  --temperature FLOAT            Sampling temperature (default: ${TEMPERATURE})
  --conv-mode MODE               Conversation mode (default: ${CONV_MODE})

Output:
  --output-dir PATH              Root directory for all eval outputs
                                 (default: ${OUTPUT_DIR})

Judge:
  --judge-model MODEL            OpenAI model used as judge (default: ${JUDGE_MODEL})
  --gpt-text-only-answers PATH   Path to pre-existing GPT text-only answer file
                                 (default: ${GPT_TEXT_ONLY_ANSWERS})
  --gemini-text-only-answers PATH  Path to pre-existing Gemini text-only answer file
                                 (default: ${GEMINI_TEXT_ONLY_ANSWERS})

Flags:
  --skip-generate                Skip answer generation; run judge step only
  -h, --help                     Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage1-model-path)          STAGE1_MODEL_PATH="$2";          shift 2 ;;
        --stage2-model-path)          STAGE2_MODEL_PATH="$2";          shift 2 ;;
        --model-base)                 MODEL_BASE="$2";                  shift 2 ;;
        --hf-dataset)                 HF_DATASET="$2";                  shift 2 ;;
        --temperature)                TEMPERATURE="$2";                 shift 2 ;;
        --conv-mode)                  CONV_MODE="$2";                   shift 2 ;;
        --output-dir)                 OUTPUT_DIR="$2";                  shift 2 ;;
        --judge-model)                JUDGE_MODEL="$2";                 shift 2 ;;
        --gpt-text-only-answers)      GPT_TEXT_ONLY_ANSWERS="$2";      shift 2 ;;
        --gemini-text-only-answers)   GEMINI_TEXT_ONLY_ANSWERS="$2";   shift 2 ;;
        --skip-generate)              SKIP_GENERATE=true;               shift   ;;
        -h|--help)                    usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

# Derived output paths
ANSWERS_DIR="${OUTPUT_DIR}/model_answers"
JUDGE_DIR="${OUTPUT_DIR}/judge"
S1_ANSWERS="${ANSWERS_DIR}/stage1_answers.jsonl"
S2_ANSWERS="${ANSWERS_DIR}/stage2_answers.jsonl"
LLAVA_ANSWERS="${ANSWERS_DIR}/basellava_answers.jsonl"

mkdir -p "${ANSWERS_DIR}" "${JUDGE_DIR}"

# ---------------------------------------------------------------------------
# Check API key
# ---------------------------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it before running this script." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1-3: Generate model answers
# ---------------------------------------------------------------------------
if [[ "${SKIP_GENERATE}" == false ]]; then

    echo "=== [1/4] Generating LLaVA-LE-S1 answers ==="
    python llava/eval/model_vqa_lunar.py \
        --model-path    "${STAGE1_MODEL_PATH}" \
        --model-base    "${MODEL_BASE}" \
        --hf-dataset    "${HF_DATASET}" \
        --answers-file  "${S1_ANSWERS}" \
        --conv-mode     "${CONV_MODE}" \
        --temperature   "${TEMPERATURE}"

    echo "=== [2/4] Generating LLaVA-LE-S2 answers ==="
    python llava/eval/model_vqa_lunar.py \
        --model-path    "${STAGE2_MODEL_PATH}" \
        --model-base    "${MODEL_BASE}" \
        --hf-dataset    "${HF_DATASET}" \
        --answers-file  "${S2_ANSWERS}" \
        --conv-mode     "${CONV_MODE}" \
        --temperature   "${TEMPERATURE}"

    echo "=== [3/4] Generating base LLaVA answers ==="
    python llava/eval/model_vqa_lunar.py \
        --model-path    "${MODEL_BASE}" \
        --hf-dataset    "${HF_DATASET}" \
        --answers-file  "${LLAVA_ANSWERS}" \
        --conv-mode     "${CONV_MODE}" \
        --temperature   "${TEMPERATURE}"

else
    echo "=== Skipping answer generation (--skip-generate) ==="
fi

# ---------------------------------------------------------------------------
# Step 4: Judge all answers
# ---------------------------------------------------------------------------
echo "=== [4/4] Running caption judge ==="

for f in "${S1_ANSWERS}" "${S2_ANSWERS}" "${LLAVA_ANSWERS}" \
          "${GPT_TEXT_ONLY_ANSWERS}" "${GEMINI_TEXT_ONLY_ANSWERS}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: Required answer file not found: ${f}" >&2
        exit 1
    fi
done

python llava/eval/caption_judge.py \
    --hf-dataset                "${HF_DATASET}" \
    --LLaVA-LE-S1-answers       "${S1_ANSWERS}" \
    --LLaVA-LE-S2-answers       "${S2_ANSWERS}" \
    --GPT-TEXT-ONLY-answers     "${GPT_TEXT_ONLY_ANSWERS}" \
    --GEMINI-TEXT-ONLY-answers  "${GEMINI_TEXT_ONLY_ANSWERS}" \
    --LLaVA-answers             "${LLAVA_ANSWERS}" \
    --scores-file               "${JUDGE_DIR}/judge_scores.jsonl" \
    --summary-file              "${JUDGE_DIR}/judge_summary.json" \
    --judge-model               "${JUDGE_MODEL}"

echo "=== Evaluation complete. Results in ${JUDGE_DIR}/ ==="
