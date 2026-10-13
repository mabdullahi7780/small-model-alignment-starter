"""Prepare single-turn SFT pairs from UltraChat 200k (Task 3).

Each conversation contributes its first user message and first assistant reply. A pair is
kept only when the formatted prompt, the response and EOS fit in `max_length` tokens, so
every training example ends with an EOS the model can learn to predict. Prompts that also
appear in an evaluation set are dropped, and so are repeated prompts. Validation pairs come
from the test_sft split and training pairs from train_sft.

Usage:
    python -m pa2.sft_data --config configs/sft.yaml --out-dir outputs/sft/data
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from pa2.batching import encode
from pa2.formatting import format_prompt

SOURCE = "HuggingFaceH4/ultrachat_200k"
EVAL_FILES = ("eval_sets/alpaca_eval_300_seed42.json", "eval_sets/hh_rlhf_eval_300_seed42.json")
DECODING_CHECK_SIZE = 50


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def eval_prompt_keys(paths=EVAL_FILES) -> set[str]:
    keys = set()
    for path in paths:
        keys.update(normalize(row["instruction"]) for row in json.loads(Path(path).read_text()))
    return keys


def first_turn(messages: list[dict]) -> tuple[str, str] | None:
    """The first user message and the assistant reply that follows it, or None."""

    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    prompt, response = messages[0]["content"].strip(), messages[1]["content"].strip()
    if not prompt or not response:
        return None
    return prompt, response


def select_pairs(rows, tokenizer, *, count, max_length, seed, excluded, prefix, stats: Counter) -> list[dict]:
    """Walk the rows in a seeded random order and keep the first `count` usable pairs."""

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    kept, seen = [], set()
    for index in order:
        if len(kept) == count:
            break
        stats["scanned"] += 1
        row = rows[index]
        pair = first_turn(row["messages"])
        if pair is None:
            stats["not_a_user_assistant_turn"] += 1
            continue
        prompt, response = pair
        key = normalize(prompt)
        if key in excluded:
            stats["eval_overlap"] += 1
            continue
        if key in seen:
            stats["duplicate_prompt"] += 1
            continue
        prompt_tokens = len(encode(tokenizer, format_prompt(prompt)))
        response_tokens = len(encode(tokenizer, response)) + 1  # + EOS
        if prompt_tokens + response_tokens > max_length:
            stats["too_long"] += 1
            continue
        seen.add(key)
        kept.append(
            {
                "id": f"{prefix}-{len(kept):05d}",
                "prompt": prompt,
                "response": response,
                "source_id": row.get("prompt_id"),
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
            }
        )
    if len(kept) < count:
        raise RuntimeError(f"only {len(kept)} usable pairs, {count} requested")
    stats["kept"] = len(kept)
    return kept


def length_summary(pairs: list[dict]) -> dict:
    totals = [pair["prompt_tokens"] + pair["response_tokens"] for pair in pairs]
    return {
        "total_tokens_mean": round(statistics.fmean(totals), 1),
        "total_tokens_median": statistics.median(totals),
        "total_tokens_max": max(totals),
        "response_tokens_mean": round(statistics.fmean(pair["response_tokens"] for pair in pairs), 1),
        "prompt_tokens_mean": round(statistics.fmean(pair["prompt_tokens"] for pair in pairs), 1),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    import yaml
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    revision = HfApi().dataset_info(SOURCE).sha
    excluded = eval_prompt_keys()
    common = {"max_length": config["max_length"], "seed": config["seed"]}

    val_stats, train_stats = Counter(), Counter()
    val_rows = load_dataset(SOURCE, split="test_sft", revision=revision)
    val = select_pairs(val_rows, tokenizer, count=config["val_examples"], excluded=excluded,
                       prefix="sft-val", stats=val_stats, **common)
    excluded |= {normalize(pair["prompt"]) for pair in val}
    train_rows = load_dataset(SOURCE, split="train_sft", revision=revision)
    train = select_pairs(train_rows, tokenizer, count=config["train_examples"], excluded=excluded,
                         prefix="sft-train", stats=train_stats, **common)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    # Held-out prompts for choosing the decoding setting, so the evaluation sets are never used for it.
    check = [{"id": pair["id"], "instruction": pair["prompt"], "dataset": "ultrachat_val"} for pair in val[:DECODING_CHECK_SIZE]]
    (out_dir / "decoding_check_prompts.json").write_text(json.dumps(check, indent=2, ensure_ascii=False) + "\n")

    stats = {
        "source": SOURCE,
        "revision": revision,
        "max_length": config["max_length"],
        "seed": config["seed"],
        "excluded_eval_files": list(EVAL_FILES),
        "val": {"split": "test_sft", "available": len(val_rows), **val_stats, **length_summary(val)},
        "train": {"split": "train_sft", "available": len(train_rows), **train_stats, **length_summary(train)},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
