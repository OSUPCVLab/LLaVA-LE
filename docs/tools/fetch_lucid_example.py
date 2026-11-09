"""Fetch one LUCID example (image + Stage-1 caption + Stage-2 QA) from Hugging Face.

Queries the public parquet shards of `pcvlab/lucid` remotely with DuckDB, so only
the matching rows are downloaded (no multi-GB dataset download).

Usage:
    pip install duckdb
    python docs/tools/fetch_lucid_example.py --id N36.600_E179.000 --image pancro/r02_c11.png \
        --out-dir docs/tools/examples

Writes:
    <out-dir>/<id>_<row_col>.png   the 224x224 panchromatic patch
    <out-dir>/<id>_<row_col>.json  {"id", "image_path", "caption", "qa": [[q, a], ...]}
"""

import argparse
import json
import os
import re

import duckdb

BASE = "https://huggingface.co/datasets/pcvlab/lucid/resolve/main/data/"
STAGE1 = [f"{BASE}stage1_captions-{i:05d}-of-00010.parquet" for i in range(10)]
STAGE2 = [f"{BASE}stage2_qa-{i:05d}-of-00003.parquet" for i in range(3)]


def clean(text):
    return text.replace("<image>", "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help='Region id, e.g. "N36.600_E179.000"')
    ap.add_argument("--image", required=True, help='Patch path within the region, e.g. "pancro/r02_c11.png"')
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "examples"))
    args = ap.parse_args()

    image_path = f"data/{args.id}/{args.image}"
    if "'" in image_path:
        raise SystemExit("invalid id/image")

    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")

    print(f"Searching Stage 1 captions for {image_path} ...")
    s1 = con.sql(
        f"SELECT image, conversations FROM read_parquet({STAGE1}) WHERE image_path = '{image_path}' LIMIT 1"
    ).fetchall()
    if not s1:
        raise SystemExit(f"{image_path} not found in stage1_captions")
    image, conv1 = s1[0]

    print("Searching Stage 2 QA ...")
    s2 = con.sql(
        f"SELECT conversations FROM read_parquet({STAGE2}) WHERE image_path = '{image_path}' LIMIT 1"
    ).fetchall()
    conv2 = s2[0][0] if s2 else []
    if not s2:
        print("  (no Stage 2 QA for this patch; 'qa' will be empty)")

    caption = clean(next(t["value"] for t in conv1 if t["from"] == "gpt"))
    qa = [
        [clean(conv2[i]["value"]), clean(conv2[i + 1]["value"])]
        for i in range(0, len(conv2) - 1, 2)
        if conv2[i]["from"] == "human" and conv2[i + 1]["from"] == "gpt"
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.id}_{re.sub(r'[^A-Za-z0-9]+', '_', os.path.splitext(os.path.basename(args.image))[0])}"
    png = os.path.join(args.out_dir, stem + ".png")
    with open(png, "wb") as f:
        f.write(image["bytes"])
    rec = {"id": args.id, "image_path": image_path, "caption": caption, "qa": qa}
    with open(os.path.join(args.out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {png} and {stem}.json\n")
    print("CAPTION:\n ", caption, "\n")
    for i, (q, a) in enumerate(qa):
        print(f"QA[{i}]  Q: {q}\n        A: {a}")


if __name__ == "__main__":
    main()
