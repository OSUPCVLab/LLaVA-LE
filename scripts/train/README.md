# Training Pipeline

This directory contains the two-stage LoRA fine-tuning scripts for LLaVA-LE on
the lunar domain dataset.

---

## Two-Stage Training Overview

| Stage | Script | Dataset | Purpose |
|---|---|---|---|
| **Stage 1** | `train_stage_1.sh` | `alignment_stage1_76000.json` | Align the model to lunar imagery via caption-style supervision |
| **Stage 2** | `train_stage_2.sh` | `instruction_stage2_19550.json` | Instruction-tune on lunar Q&A, initialised from the Stage 1 LoRA weights |

Both scripts call `llava/train/train_mem.py` (which in turn calls `llava/train/train.py`
with Flash Attention 2 enabled).
`train.py` **automatically appends a timestamp** to the output directory, so you
never need to pass `--output_dir` from the scripts.

---

## Usage

### Stage 1
```bash
bash scripts/train/train_stage_1.sh
```
No required arguments — all defaults are set inside the script.

### Stage 2
```bash
# Use the default Stage 1 checkpoint
bash scripts/train/train_stage_2.sh

# Use a custom Stage 1 checkpoint
bash scripts/train/train_stage_2.sh \
  --lora-weight-path LLaVA-LE/checkpoints/stage_1/my_stage1_checkpoint
```

---

## Parameter Reference

Parameters are grouped by the dataclass they belong to in `train.py`.

### Model (`ModelArguments`)

| Parameter | Stage 1 value | Stage 2 value | Description |
|---|---|---|---|
| `--model_name_or_path` | `liuhaotian/llava-v1.5-13b` | `liuhaotian/llava-v1.5-13b` | HuggingFace model ID or local path for the base LLM |
| `--version` | `v1` | `v1` | Conversation template version; `v1` maps to the Vicuna-style prompt |
| `--vision_tower` | `openai/clip-vit-large-patch14` | same | Vision encoder used to embed images |
| `--mm_projector_type` | `mlp2x_gelu` | same | Architecture of the multimodal projector; `mlp2x_gelu` is a 2-layer MLP with GELU activation |
| `--mm_vision_select_layer` | `-2` | same | Which layer of the vision tower to extract features from; `-2` is the second-to-last layer |
| `--mm_use_im_start_end` | `False` | same | Whether to wrap image tokens with special `<im_start>` / `<im_end>` tokens |
| `--mm_use_im_patch_token` | `False` | same | Whether to use explicit patch tokens in the sequence |

### Data (`DataArguments`)

| Parameter | Stage 1 value | Stage 2 value | Description |
|---|---|---|---|
| `--data_path` | `alignment_stage1_76000.json` | `instruction_stage2_19550.json` | Path to the training JSON file (list of conversation dicts) |
| `--image_folder` | `./dataset/data/lumina_96k/data` | same | Root directory containing all training images |
| `--image_aspect_ratio` | `pad` | `pad` | How to handle non-square images; `pad` pads to square rather than cropping |
| `--lazy_preprocess` | `True` | `True` | Tokenise samples on-the-fly instead of upfront; saves RAM at the cost of slightly slower first epoch |

### LoRA (`TrainingArguments`)

| Parameter | Stage 1 value | Stage 2 value | Description |
|---|---|---|---|
| `--lora_enable` | `True` | `True` | Enable LoRA; only a small set of adapter weights are trained instead of the full model |
| `--lora_r` | `64` | `64` | LoRA rank — controls the size (and expressiveness) of the adapter matrices |
| `--lora_alpha` | `128` | `128` | LoRA scaling factor; effective learning scale is `lora_alpha / lora_r` (here: 2×) |
| `--lora_dropout` | *(not set)* | `0.00` | Dropout applied to LoRA activations; `0.00` disables it for stable fine-tuning |
| `--lora_weight_path` | *(not set)* | Stage 1 checkpoint | Path to existing LoRA weights to load before training; used in Stage 2 to warm-start from Stage 1 |

### Multimodal Projector

| Parameter | Stage 1 value | Stage 2 value | Description |
|---|---|---|---|
| `--mm_projector_lr` | `2e-5` | `2e-5` | Separate learning rate for the multimodal projector; allows the projector to adapt faster than the LoRA adapters |

### Optimization (`TrainingArguments`)

| Parameter | Stage 1 value | Stage 2 value | Description |
|---|---|---|---|
| `--num_train_epochs` | `4` | `5` | Total number of passes over the training dataset |
| `--per_device_train_batch_size` | `16` | `16` | Batch size per GPU |
| `--gradient_accumulation_steps` | `1` | `1` | Accumulate gradients over N steps before an optimizer update; effective batch size = `per_device × gpus × this` |
| `--learning_rate` | `1e-4` | `2e-5` | Peak learning rate; Stage 2 is 5× lower to preserve Stage 1 alignment |
| `--weight_decay` | `0.0` | *(not set)* | L2 regularization coefficient |
| `--warmup_ratio` | `0.05` | `0.03` | Fraction of total steps spent linearly warming up the learning rate |
| `--lr_scheduler_type` | `cosine` | `cosine` | LR schedule after warmup; `cosine` decays smoothly to near-zero |
| `--max_grad_norm` | `1.0` | `1.0` | Gradient clipping threshold; prevents exploding gradients |
| `--bf16` | `True` | `True` | Train in bfloat16; lower memory than fp32, more stable than fp16 on Ampere+ GPUs |
| `--tf32` | `True` | *(not set)* | Enable TF32 matrix multiplications on Ampere+ GPUs for additional speed |

### Memory & I/O

| Parameter | Stage 1 value | Stage 2 value | Description |
|---|---|---|---|
| `--model_max_length` | `2048` | `2048` | Maximum token sequence length; sequences are right-padded or truncated |
| `--gradient_checkpointing` | `True` | `True` | Recompute activations during the backward pass to save GPU memory at the cost of ~30% more compute |
| `--dataloader_num_workers` | `4` | *(not set)* | Number of CPU workers for data loading |
| `--group_by_modality_length` | `True` | `True` | Batch samples of similar token length together to reduce padding waste |
| `--deepspeed` | `./scripts/zero3.json` | same | DeepSpeed config; ZeRO-3 shards optimizer states, gradients, and parameters across GPUs |

### Checkpointing (`TrainingArguments`)

| Parameter | Stage 1 value | Stage 2 value | Description |
|---|---|---|---|
| `--save_strategy` | `steps` | *(not set)* | When to save checkpoints: `steps` saves every `--save_steps` steps |
| `--save_steps` | `5000` | *(not set)* | Save a checkpoint every N optimizer steps |
| `--save_total_limit` | `2` | *(not set)* | Maximum number of checkpoints to keep on disk; older ones are deleted |
| `--evaluation_strategy` | `no` | *(not set)* | When to run evaluation; `no` disables it |
| `--per_device_eval_batch_size` | `4` | *(not set)* | Eval batch size per GPU (only relevant when evaluation is enabled) |

### Logging

| Parameter | Stage 1 value | Stage 2 value | Description |
|---|---|---|---|
| `--logging_steps` | `1` | `10` | Log training metrics every N steps |
| `--logging_strategy` | *(not set)* | `steps` | When to log; `steps` logs every `--logging_steps` steps |
| `--log_gradients` | `False` | `False` | Enable gradient norm logging via `GradientLoggingCallback`; when `True`, logs are saved to `<output_dir>/gradients/` |
| `--report_to` | `none` | `none` | Disable all external logging backends (e.g. WandB, TensorBoard) |

---

## Output

`train.py` constructs the output path automatically as:

```
checkpoints/
├── stage_1/<base_name>_<YYYYMMDD_HHMMSS>/    # Stage 1 run
│   ├── adapter_model.bin
│   ├── non_lora_trainables.bin
│   ├── loss_log.json
│   └── gradients/                            # only present if --log_gradients True
└── stage_2/<base_name>_<YYYYMMDD_HHMMSS>/    # Stage 2 run
    ├── adapter_model.bin
    ├── non_lora_trainables.bin
    ├── loss_log.json
    └── gradients/                            # only present if --log_gradients True
```

The timestamp suffix is only added on fresh runs; if an existing `checkpoint-*`
subdirectory is found, training resumes from the latest checkpoint without
renaming the directory.
