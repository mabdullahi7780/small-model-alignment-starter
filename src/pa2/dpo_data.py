"""Prepare single-turn preference pairs for the two DPO runs (Task 4).

hh: Anthropic/hh-rlhf, harmless-base and helpful-base in equal shares. Each row stores two
    full transcripts ("\\n\\nHuman: ...\\n\\nAssistant: ..."); only rows with exactly one human
    turn are kept, and the prompt is the human message. Validation pairs come from the test
    files, training pairs from the train files.
ultrafeedback: HuggingFaceH4/ultrafeedback_binarized train_prefs. Ties (equal scores) are
    dropped. Validation and training pairs are disjoint draws from the same split.

Every kept pair has a formatted prompt of at most `max_prompt_length` tokens and both
prompt + response + EOS sequences within `max_length`, so no response is truncated. Prompts
that appear in an evaluation set are excluded, and so are repeated prompts.

Usage:
    python -m pa2.dpo_data --config configs/dpo_hh.yaml --out-dir outputs/dpo/hh/data
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from pa2.batching import encode
from pa2.formatting import format_prompt
from pa2.sft_data import eval_prompt_keys, normalize

HUMAN, ASSISTANT = "\n\nHuman:", "\n\nAssistant:"
HH_REPO = "Anthropic/hh-rlhf"
HH_SUBSETS = ("harmless-base", "helpful-base")
UF_REPO = "HuggingFaceH4/ultrafeedback_binarized"


def parse_hh(chosen_text: str, rejected_text: str) -> tuple[str, str, str] | None:
    """(prompt, chosen, rejected) for a single-turn HH row, or None."""

    for text in (chosen_text, rejected_text):
        if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
            return None
    cut = chosen_text.index(ASSISTANT) + len(ASSISTANT)
    if rejected_text[:cut] != chosen_text[:cut]:
        return None
    prompt = chosen_text[: cut - len(ASSISTANT)].replace(HUMAN, "", 1).strip()
    chosen, rejected = chosen_text[cut:].strip(), rejected_text[cut:].strip()
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return prompt, chosen, rejected


def parse_ultrafeedback(row: dict) -> tuple[str, str, str] | None:
    """(prompt, chosen, rejected) for a single-turn, non-tied UltraFeedback row, or None."""

    if row["score_chosen"] == row["score_rejected"]:
        return None
    replies = []
    for messages in (row["chosen"], row["rejected"]):
        if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            return None
        replies.append(messages[1]["content"].strip())
    prompt = row["prompt"].strip()
    chosen, rejected = replies
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return prompt, chosen, rejected


def select_pairs(candidates, tokenizer, *, count, max_length, max_prompt_length, seed, excluded, prefix, stats, source):
    """Walk the candidates in a seeded order and keep the first `count` usable pairs.

    candidates: list of raw items; `source` turns one into (prompt, chosen, rejected) or None.
    """

    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)
    kept = []
    for index in order:
        if len(kept) == count:
            break
        stats["scanned"] += 1
        parsed = source(candidates[index])
        if parsed is None:
            stats["unusable_row"] += 1
            continue
        prompt, chosen, rejected = parsed
        key = normalize(prompt)
        if key in excluded:
            stats["excluded_prompt"] += 1
            continue
        prompt_tokens = len(encode(tokenizer, format_prompt(prompt)))
        chosen_tokens = len(encode(tokenizer, chosen)) + 1  # + EOS
        rejected_tokens = len(encode(tokenizer, rejected)) + 1
        if prompt_tokens > max_prompt_length:
            stats["prompt_too_long"] += 1
            continue
        if prompt_tokens + max(chosen_tokens, rejected_tokens) > max_length:
            stats["response_too_long"] += 1
            continue
        excluded.add(key)
        kept.append({
            "id": f"{prefix}-{len(kept):05d}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_tokens": prompt_tokens,
            "chosen_tokens": chosen_tokens,
            "rejected_tokens": rejected_tokens,
        })
    if len(kept) < count:
        raise RuntimeError(f"{prefix}: only {len(kept)} usable pairs, {count} requested")
    stats["kept"] += len(kept)
    return kept


def summary(pairs: list[dict]) -> dict:
    return {
        "pairs": len(pairs),
        "prompt_tokens_mean": round(statistics.fmean(p["prompt_tokens"] for p in pairs), 1),
        "chosen_tokens_mean": round(statistics.fmean(p["chosen_tokens"] for p in pairs), 1),
        "rejected_tokens_mean": round(statistics.fmean(p["rejected_tokens"] for p in pairs), 1),
        "chosen_longer_share": round(sum(p["chosen_tokens"] > p["rejected_tokens"] for p in pairs) / len(pairs), 3),
    }


def prepare_hh(config, tokenizer, excluded) -> tuple[list[dict], list[dict], dict]:
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(HH_REPO).sha
    common = {"max_length": config["max_length"], "max_prompt_length": config["max_prompt_length"],
              "seed": config["seed"], "source": lambda row: parse_hh(row["chosen"], row["rejected"])}
    per_subset_val = config["val_pairs"] // len(HH_SUBSETS)
    per_subset_train = config["train_pairs"] // len(HH_SUBSETS)
    train, val, stats = [], [], {"source": HH_REPO, "revision": revision}
    for subset in HH_SUBSETS:
        rows = {}
        for split in ("test", "train"):
            path = hf_hub_download(HH_REPO, f"{subset}/{split}.jsonl.gz", repo_type="dataset", revision=revision)
            with gzip.open(path, "rt") as handle:
                rows[split] = [json.loads(line) for line in handle]
        val_stats, train_stats = Counter(), Counter()
        val += select_pairs(rows["test"], tokenizer, count=per_subset_val, excluded=excluded,
                            prefix=f"hh-val-{subset}", stats=val_stats, **common)
        train += select_pairs(rows["train"], tokenizer, count=per_subset_train, excluded=excluded,
                              prefix=f"hh-train-{subset}", stats=train_stats, **common)
        stats[subset] = {"test_rows": len(rows["test"]), "train_rows": len(rows["train"]),
                         "val": dict(val_stats), "train": dict(train_stats)}
    return train, val, stats


def prepare_ultrafeedback(config, tokenizer, excluded) -> tuple[list[dict], list[dict], dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(UF_REPO)
    file = next(s.rfilename for s in info.siblings if s.rfilename.startswith("data/train_prefs"))
    rows = pq.read_table(hf_hub_download(UF_REPO, file, repo_type="dataset", revision=info.sha)).to_pylist()
    common = {"max_length": config["max_length"], "max_prompt_length": config["max_prompt_length"],
              "seed": config["seed"], "source": parse_ultrafeedback}
    val_stats, train_stats = Counter(), Counter()
    # One seeded walk: validation takes the first usable pairs; training continues with a
    # different shuffle and skips every validation prompt through `excluded`.
    val = select_pairs(rows, tokenizer, count=config["val_pairs"], excluded=excluded,
                       prefix="uf-val", stats=val_stats, **common)
    train = select_pairs(rows, tokenizer, count=config["train_pairs"], excluded=excluded,
                         prefix="uf-train", stats=train_stats, **{**common, "seed": config["seed"] + 1})
    ties = sum(row["score_chosen"] == row["score_rejected"] for row in rows)
    stats = {"source": UF_REPO, "revision": info.sha, "file": file, "rows": len(rows), "tied_rows": ties,
             "val": dict(val_stats), "train": dict(train_stats)}
    return train, val, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    import yaml
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    excluded = eval_prompt_keys()

    prepare = {"hh": prepare_hh, "ultrafeedback": prepare_ultrafeedback}[config["dataset"]]
    train, val, stats = prepare(config, tokenizer, excluded)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    stats.update({"dataset": config["dataset"], "max_length": config["max_length"],
                  "max_prompt_length": config["max_prompt_length"], "seed": config["seed"],
                  "train_summary": summary(train), "val_summary": summary(val)})
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
