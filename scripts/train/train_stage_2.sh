#!/bin/bash

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
LORA_WEIGHT_PATH="LLaVA-LE/checkpoints/stage_1/llava-v1.5-13b-task-lora-stage1_20260221_001701"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lora-weight-path) LORA_WEIGHT_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
deepspeed llava/train/train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path liuhaotian/llava-v1.5-13b \
  --lora_weight_path "${LORA_WEIGHT_PATH}" \
  --lora_enable True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.00 \
  --mm_projector_lr 2e-5 \
  --version v1 \
  --data_path ./dataset/data/instruction_stage2_19550.json \
  --image_folder ./dataset/data/lumina_96k/data \
  --vision_tower openai/clip-vit-large-patch14 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --log_gradients False \
  --logging_strategy steps \
  --logging_steps 10 \
  --max_grad_norm 1.0 \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --report_to none
