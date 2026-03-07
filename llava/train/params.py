from collections import defaultdict

def _clean_name(name: str) -> str:
    """
    Remove PEFT wrapper prefixes so parameter names
    match the LLaVA architecture the user sees.
    """
    for prefix in [
        "base_model.model.",
        "model.model.",
        "model."
    ]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def print_trainable_layers(model):
    groups = defaultdict(list)
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        clean_name = _clean_name(name)

        total_params += param.numel()
        if not param.requires_grad:
            continue

        trainable_params += param.numel()

        if "lora_" in clean_name:
            groups["LoRA adapters (LLaMA)"].append(clean_name)
        elif clean_name.startswith("mm_projector"):
            groups["Multimodal projector"].append(clean_name)
        elif clean_name.startswith("vision_tower"):
            groups["Vision tower"].append(clean_name)
        elif clean_name.startswith("lm_head"):
            groups["LM head"].append(clean_name)
        else:
            groups["Other trainable params"].append(clean_name)

    print("\n🔍 Trainable parameter groups (PEFT-aware):\n")

    for group, names in groups.items():
        print(f"▶ {group} ({len(names)} tensors)")
        for n in names:
            print(f"   • {n}")
        print()

    print("📊 Parameter summary:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    if total_params > 0:
        print(f"   Trainable ratio:      {100 * trainable_params / total_params:.4f}%\n")
    else:
        print("   Trainable ratio:      N/A (ZeRO-3 sharded)\n")


