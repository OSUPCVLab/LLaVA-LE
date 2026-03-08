from __future__ import annotations

import argparse
import os
import json
import math
import torch
from tqdm import tqdm
from typing import Any
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
from transformers import set_seed, logging

logging.set_verbosity_error()


def split_list(lst: list[Any], n: int) -> list[list[Any]]:
    """Split a list into ``n`` roughly equal-sized chunks.

    Args:
        lst: The list to split.
        n: Number of chunks to produce.

    Returns:
        A list of ``n`` sublists.  The last chunk may be smaller than the rest
        if ``len(lst)`` is not divisible by ``n``.
    """
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: list[Any], n: int, k: int) -> list[Any]:
    """Return the ``k``-th chunk from a list split into ``n`` chunks.

    Args:
        lst: The full list to split.
        n: Total number of chunks.
        k: Zero-based index of the desired chunk.

    Returns:
        The ``k``-th sublist produced by :func:`split_list`.
    """
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args: argparse.Namespace) -> None:
    """Run a LLaVA model over the evaluation question set and write answers.

    Loads the model and tokenizer from ``args.model_path``, iterates over the
    questions in ``args.question_file`` (optionally restricted to a single
    chunk for distributed evaluation), generates one answer per question, and
    streams results to ``args.answers_file`` in JSONL format.

    Each output record contains the following fields:

    - ``question_id``: identifier copied from the input question.
    - ``prompt``: the raw question text sent to the model (without image tokens).
    - ``text``: the model's generated answer.
    - ``answer_id``: a unique UUID for this answer.
    - ``model_id``: the model name derived from ``args.model_path``.
    - ``metadata``: empty dict reserved for future use.

    Args:
        args: Parsed CLI arguments.  Expected attributes:

            - ``model_path`` (str): HuggingFace model ID or local checkpoint path.
            - ``model_base`` (str | None): Base model path required for LoRA checkpoints.
            - ``question_file`` (str): Path to the JSON question file.
            - ``image_folder`` (str): Root directory containing evaluation images.
            - ``answers_file`` (str): Output JSONL path for generated answers.
            - ``conv_mode`` (str): Conversation template key (e.g. ``"vicuna_v1"``).
            - ``num_chunks`` (int): Total number of chunks for distributed eval.
            - ``chunk_idx`` (int): Zero-based index of the chunk to process.
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

    # Load questions from JSON
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
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

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

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
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Root directory containing evaluation images.")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Path to the JSON evaluation question file.")
    parser.add_argument("--answers-file", type=str, required=True,
                        help="Output JSONL path where generated answers are written.")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1",
                        help="Conversation template key (default: vicuna_v1).")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Total number of chunks for distributed evaluation.")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Zero-based index of the chunk to process.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature; 0 enables greedy decoding.")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Nucleus sampling probability threshold.")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search (1 = greedy).")
    args = parser.parse_args()

    eval_model(args)
