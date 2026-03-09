from __future__ import annotations

import argparse
import os
import json
import torch
from tqdm import tqdm
import shortuuid

from datasets import load_dataset

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
from transformers import set_seed, logging

logging.set_verbosity_error()


def eval_model(args: argparse.Namespace) -> None:
    """Run a LLaVA model over the LUCID evaluation split and write answers.

    Loads the evaluation questions and images directly from the
    ``pcvlab/lucid`` HuggingFace dataset (``evaluation`` config, ``test``
    split) so no local data files or image folders are required.  Iterates
    over all questions, generates one answer per question, and streams results
    to ``args.answers_file`` in JSONL format.

    Each output record contains the following fields:

    - ``question_id``: identifier copied from the dataset.
    - ``prompt``: the raw question text sent to the model (without image tokens).
    - ``text``: the model's generated answer.
    - ``answer_id``: a unique UUID for this answer.
    - ``model_id``: the model name derived from ``args.model_path``.
    - ``metadata``: empty dict reserved for future use.

    Args:
        args: Parsed CLI arguments.  Expected attributes:

            - ``model_path`` (str): HuggingFace model ID or local checkpoint path.
            - ``model_base`` (str | None): Base model path required for LoRA checkpoints.
            - ``hf_dataset`` (str): HuggingFace dataset repo ID (e.g. ``"pcvlab/lucid"``).
            - ``answers_file`` (str): Output JSONL path for generated answers.
            - ``conv_mode`` (str): Conversation template key (e.g. ``"vicuna_v1"``).
            - ``temperature`` (float): Sampling temperature; ``0`` → greedy decoding.
            - ``top_p`` (float | None): Nucleus sampling probability threshold.
            - ``num_beams`` (int): Number of beams for beam search (``1`` → greedy).
    """
    set_seed(0)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Load the full evaluation split from HuggingFace (50 images, 190 questions)
    eval_ds = load_dataset(args.hf_dataset, "evaluation", split="test")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for sample in tqdm(eval_ds):
        idx = sample["question_id"]
        image: Image.Image = sample["image"]  # PIL Image provided by HF
        qs = sample["text"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        # Panchromatic images arrive as mode "L"; convert to RGB for CLIP
        image_tensor = process_images([image.convert("RGB")], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model ID or local path to the checkpoint.")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path required when loading a LoRA checkpoint.")
    parser.add_argument("--hf-dataset", type=str, default="pcvlab/lucid",
                        help="HuggingFace dataset repo ID (default: pcvlab/lucid).")
    parser.add_argument("--answers-file", type=str, required=True,
                        help="Output JSONL path where generated answers are written.")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1",
                        help="Conversation template key (default: vicuna_v1).")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature; 0 enables greedy decoding.")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Nucleus sampling probability threshold.")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search (1 = greedy).")
    args = parser.parse_args()

    eval_model(args)
