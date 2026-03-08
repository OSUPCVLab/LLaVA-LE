import os
import json
import argparse
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Judge prompt – do NOT modify this block
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """
You are an expert planetary scientist evaluating responses to lunar surface questions. The user asks the question on observing an image.
Each question is paired with a scientifically accurate caption describing the image which serves as the reference.
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

Core Rule: The highest score must go to the most direct, concise, and accurate answer that clearly states the central geological inference.

Score each response based on the following criteria:

1.\tDirectness - Highest Weight
The response must answer only what is asked.
Prefer the shortest and most concise answer that directly addresses the question with a clear generalized conclusion.

2.\tFocus on the Exact Question
Answers that answer the question correctly in the fewest necessary words should receive higher scores.
The answer that addresses only what is being asked scores higher.
Do not reward additional geological context, broader history, age interpretation, tectonics, or speculative extensions unless explicitly required by the question.

3.\tPrinciple Extraction
Reward answers that summarize the key implication cleanly. Answers that closely paraphrase the caption must receive noticeably lower scores, even if correct.

4.\tConcision and Clarity
Prefer the shortest and most concise answer that directly addresses the question with a clear generalized conclusion.
Prefer responses that are technically accurate while being clear and concise. 
High scoring responses are compact, direct, short, and technically accurate.

5.\tScientific Accuracy
Incorrect or physically implausible claims must receive strong penalties.

Scoring Guidance:
Highest scores should go to answers that are concise, observation grounded, directly responsive, and interpret only to the degree required by the question.
Lower scores should go to answers that speculate, overextend conclusions, introduce unrelated processes, or lack clarity and precision.

Penalty Criteria:
Apply strong penalties for direct contradiction of the caption or physically implausible claims
Apply strong penalties for caption paraphrasing.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MODEL_KEYS = ["LLaVA-LE-S1", "LLaVA-LE-S2", "GPT-TEXT-ONLY", "GEMINI-TEXT-ONLY", "LLaVA"]

def load_jsonl(path):
    """Load a .jsonl file and return a list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_answers(path):
    """
    Load answers file (.json or .jsonl).
    Returns dict: question_id (int/str) -> answer text.
    """
    path = str(path)
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
    else:
        data = load_jsonl(path)

    result = {}
    for item in data:
        qid = item["question_id"]
        text = item.get("text") or item.get("answer") or ""
        result[qid] = text
    return result


def create_evaluation_prompt(question, fig_caption, question_type, answers_dict):
    """
    Build the user-turn prompt for the GPT-4 judge.
    answers_dict: {model_key: answer_text, ...}
    """
    parts = [
        "[Reference Caption]",
        fig_caption,
        "",
        f"[Question Type]: {question_type}",
        "",
        "[Question]",
        question,
        "",
    ]

    for key in MODEL_KEYS:
        answer = answers_dict.get(key, "No answer provided")
        parts += [
            f"[{key}]",
            answer,
            f"[End of {key}]",
            "",
        ]

    format_str = " ".join(f"{k}=<score>" for k in MODEL_KEYS)
    example_scores = [8, 7, 9, 6, 8]
    example_str = " ".join(f"{k}={s}" for k, s in zip(MODEL_KEYS, example_scores))

    parts += [
        "[Evaluation Task]",
        "Evaluate each model's answer using ONLY the caption above as reference.",
        "Output format (follow EXACTLY — do not add any text before the scores block):",
        format_str,
        f"Example: {example_str}",
        "Then on the next line(s): A comparative reasoning paragraph explaining why each model received its score.",
    ]

    return "\n".join(parts)


def parse_scores(raw_eval, n=5):
    """
    Extract scores from key=value format anywhere in the output.
    Looks for patterns like: LLaVA-LE-S1=8 LLaVA-LE-S2=7 ... (case-insensitive, allows spaces around =)
    Falls back to first-line space-separated parse if key=value not found.
    Returns list of n floats in MODEL_KEYS order.
    """
    import re

    # Primary: match each MODEL_KEY=<num> anywhere in the text
    escaped_keys = [re.escape(k) for k in MODEL_KEYS]
    pattern = re.compile(
        r"(" + "|".join(escaped_keys) + r")\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(raw_eval)

    if matches:
        score_map = {}
        for key_found, val in matches:
            for k in MODEL_KEYS:
                if k.lower() == key_found.lower():
                    score_map[k] = float(val)
                    break
        if len(score_map) == n:
            return [score_map[k] for k in MODEL_KEYS]
        print(f"  Warning: found only {len(score_map)} model=score pairs, expected {n}. Found: {score_map}")
        return [score_map.get(k, 0.0) for k in MODEL_KEYS]

    # Fallback: first line space-separated numbers
    print("  Warning: key=value pattern not found, attempting fallback parse.")
    try:
        first_line = raw_eval.strip().split("\n")[0].strip()
        # Strip any leading label like "Scores:"
        first_line = re.sub(r"^[^0-9]*", "", first_line)
        tokens = first_line.split()
        scores = [float(t) for t in tokens if re.match(r"^[0-9]+(?:\.[0-9]+)?$", t)]
        if len(scores) == n:
            return scores
        print(f"  Fallback also failed. Got {len(scores)} values: {scores}")
    except Exception as e:
        print(f"  Error in fallback parse: {e} | raw: {raw_eval[:200]}")

    return [0.0] * n


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_questions(samples, client, judge_model):
    results = []
    print(f"\nEvaluating {len(samples)} questions with {judge_model} as judge …")

    for sample in tqdm(samples):
        prompt = create_evaluation_prompt(
            question=sample["question"],
            fig_caption=sample["fig_caption"],
            question_type=sample["type"],
            answers_dict=sample["answers"],
        )

        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        raw_eval = response.choices[0].message.content.strip()
        scores_list = parse_scores(raw_eval)

        result = deepcopy(sample)
        result["judge_output"] = raw_eval
        result["scores"] = {key: scores_list[i] for i, key in enumerate(MODEL_KEYS)}
        # question_id and question_type are already in sample; make them explicit at top level
        result["question_id"] = sample["question_id"]
        result["question_type"] = sample["type"]
        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_model_summaries(results, client, judge_model):
    """Compute per-model aggregate stats and generate a short LLM justification."""
    aggregated = {key: defaultdict(list) for key in MODEL_KEYS}

    for r in results:
        q_type = r["type"]
        for key in MODEL_KEYS:
            score = r["scores"][key]
            aggregated[key][q_type].append(score)
            aggregated[key]["overall"].append(score)

    # Stats
    model_stats = {}
    for key in MODEL_KEYS:
        model_stats[key] = {}
        for cat in list(aggregated[key].keys()):
            vals = aggregated[key][cat]
            model_stats[key][cat] = {"avg": sum(vals) / len(vals), "n": len(vals)}

    # LLM justifications
    print("\nGenerating per-model summary justifications …")
    summaries = {}

    for key in tqdm(MODEL_KEYS):
        stats = model_stats[key]
        lines = []
        for cat, v in stats.items():
            lines.append(f"  {cat}: {v['avg']:.2f}/10 ({v['n']} questions)")

        summary_prompt = (
            f"You evaluated multiple AI models on lunar science questions. "
            f"Below are the aggregated scores for an anonymous model ({key}).\n"
            + "\n".join(lines)
            + "\n\nWrite a concise 2-3 sentence justification covering: "
            "main strengths, main weaknesses, and overall assessment."
        )

        resp = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are an AI evaluation expert."},
                {"role": "user", "content": summary_prompt},
            ],
            temperature=0.0,
        )
        justification = resp.choices[0].message.content.strip()

        summaries[key] = {
            "stats": stats,
            "justification": justification,
        }

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    client = OpenAI(api_key=args.api_key or os.environ["OPENAI_API_KEY"])
    judge_model = args.judge_model  # e.g. "gpt-4o"

    # ── Load questions ────────────────────────────────────────────────────────
    with open(args.question_file) as f:
        question_data = json.load(f)

    # Build a lookup: question_id -> question record
    question_map = {q["question_id"]: q for q in question_data}

    # ── Load all 5 model answer files ─────────────────────────────────────────
    answer_files = [
        args.LLaVA_LE_S1_answers,
        args.LLaVA_LE_S2_answers,
        args.GPT_TEXT_ONLY_answers,
        args.GEMINI_TEXT_ONLY_answers,
        args.LLaVA_answers,
    ]

    print("Loading model answers …")
    model_answers = {}
    for key, path in zip(MODEL_KEYS, answer_files):
        if path:
            data = load_answers(path)
            model_answers[key] = data
            print(f"  {key}: {len(data)} answers loaded from {path}")
        else:
            model_answers[key] = {}
            print(f"  {key}: no file provided – will use 'No answer'")

    # ── Prepare samples ───────────────────────────────────────────────────────
    samples = []
    missing_caption = 0
    for qid, q in question_map.items():
        # fig_caption may be stored directly on the question or not present
        fig_caption = q.get("fig_caption") or q.get("caption") or ""
        if not fig_caption:
            missing_caption += 1

        answers_dict = {
            key: model_answers[key].get(qid, "No answer provided")
            for key in MODEL_KEYS
        }

        samples.append(
            {
                "question_id": qid,
                "question": q.get("text", ""),
                "fig_caption": fig_caption,
                "type": q.get("type", "unknown"),
                "answers": answers_dict,
            }
        )

    if missing_caption:
        print(
            f"  Warning: {missing_caption} questions have no fig_caption. "
            "Evaluation quality will be reduced for those."
        )

    print(f"Prepared {len(samples)} questions for evaluation.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate_questions(samples, client, judge_model)

    # ── Save per-question scores ──────────────────────────────────────────────
    Path(args.scores_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.scores_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-question scores → {args.scores_file}")

    # ── Generate & save summaries ─────────────────────────────────────────────
    summaries = generate_model_summaries(results, client, judge_model)

    with open(args.summary_file, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Model summaries      → {args.summary_file}")

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    for key in MODEL_KEYS:
        s = summaries[key]
        overall = s["stats"].get("overall", {})
        print(f"\n{key.upper()}  |  Overall avg: {overall.get('avg', 0):.2f}/10  ({overall.get('n', 0)} questions)")
        print(f"  {s['justification']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blind multi-model evaluation using GPT-4 as judge (caption-only)."
    )
    parser.add_argument("--question-file", required=True,
                        help="JSON file with questions, types, and fig_captions.")
    parser.add_argument("--LLaVA-LE-S1-answers", required=True, help="Answers file for LLaVA-LE stage 1 model.")
    parser.add_argument("--LLaVA-LE-S2-answers", required=True, help="Answers file for LLaVA-LE stage 2 model.")
    parser.add_argument("--GPT-TEXT-ONLY-answers", required=True, help="Answers file for GPT text-only.")
    parser.add_argument("--GEMINI-TEXT-ONLY-answers", required=True, help="Answers file for Gemini text-only.")
    parser.add_argument("--LLaVA-answers", required=True, help="Answers file for base LLaVA.")
    parser.add_argument("--scores-file", required=True,
                        help="Output JSONL: one record per question with scores.")
    parser.add_argument("--summary-file", required=True,
                        help="Output JSON: per-model aggregated stats + justification.")
    parser.add_argument("--judge-model", default="gpt-4o",
                        help="OpenAI model to use as judge (default: gpt-4o).")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var).")
    args = parser.parse_args()
    main(args)