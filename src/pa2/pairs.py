"""Blind pairwise comparison files (Task 6).

For every prompt answered by two models, a seeded coin flip decides which answer is shown as
Response A and which as Response B. The mapping from position to model is kept under
`hidden`, which the judge never sees: `judge_view` is the only way a pair reaches the judge,
and it copies an explicit allow-list of fields.

Usage:
    python -m pa2.pairs --x outputs/gen/dpo_hh_hh.json --name-x dpo_hh --y outputs/gen/sft_hh.json --name-y sft \\
        --prompts eval_sets/hh_rlhf_eval_300_seed42.json --comparison dpo_hh_vs_sft --out outputs/pairs/dpo_hh_vs_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

JUDGE_FIELDS = ("pair_id", "instruction", "response_a", "response_b", "category", "expected_behavior")


def build_pairs(records_x, records_y, name_x, name_y, comparison, seed=42, prompt_rows=None) -> list[dict]:
    """One blind pair per prompt; both answer lists must cover the same prompts in the same order."""

    if [r["id"] for r in records_x] != [r["id"] for r in records_y]:
        raise ValueError("the two generation files must answer the same prompts in the same order")
    extra = {row["id"]: row for row in (prompt_rows or [])}
    rng = random.Random(f"{seed}-{comparison}")
    pairs = []
    for x, y in zip(records_x, records_y):
        if x["instruction"] != y["instruction"]:
            raise ValueError(f"{x['id']}: instructions differ between the two files")
        x_first = rng.random() < 0.5
        (a, name_a), (b, name_b) = ((x, name_x), (y, name_y)) if x_first else ((y, name_y), (x, name_x))
        pair = {
            "pair_id": x["id"],
            "comparison": comparison,
            "dataset": x.get("dataset"),
            "instruction": x["instruction"],
            "response_a": a["output"],
            "response_b": b["output"],
            "hidden": {"a": name_a, "b": name_b, "words_a": len(a["output"].split()), "words_b": len(b["output"].split())},
        }
        for field in ("category", "expected_behavior"):
            if field in extra.get(x["id"], {}):
                pair[field] = extra[x["id"]][field]
        pairs.append(pair)
    return pairs


def judge_view(pair: dict) -> dict:
    """The only fields of a pair that may be shown to a judge; model identities never are."""

    return {field: pair[field] for field in JUDGE_FIELDS if field in pair}


def balance(pairs: list[dict], name: str) -> float:
    """Share of pairs in which `name` is shown as Response A."""

    return sum(pair["hidden"]["a"] == name for pair in pairs) / max(len(pairs), 1)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--x", required=True, help="generation file of the model being evaluated (DPO or LoRA)")
    parser.add_argument("--name-x", required=True)
    parser.add_argument("--y", required=True, help="generation file of the baseline (SFT or base SmolLM2)")
    parser.add_argument("--name-y", required=True)
    parser.add_argument("--prompts", required=True, help="the prompt file, for category and expected_behavior")
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    pairs = build_pairs(
        json.loads(Path(args.x).read_text()),
        json.loads(Path(args.y).read_text()),
        args.name_x,
        args.name_y,
        args.comparison,
        args.seed,
        json.loads(Path(args.prompts).read_text()),
    )
    write_jsonl(Path(args.out), pairs)
    print(json.dumps({"comparison": args.comparison, "pairs": len(pairs),
                      f"{args.name_x}_shown_as_a": round(balance(pairs, args.name_x), 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
