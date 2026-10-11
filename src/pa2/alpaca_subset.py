"""Build the fixed AlpacaEval evaluation subset: 300 of the 805 prompts, chosen with seed 42.

Usage (from the repository root):
    python -m pa2.alpaca_subset --out eval_sets/alpaca_eval_300_seed42.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SOURCE_REPO = "tatsu-lab/alpaca_eval"
SOURCE_FILE = "alpaca_eval.json"
DEFAULT_SIZE = 300
DEFAULT_SEED = 42


def select_subset(rows: list[dict], size: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> list[dict]:
    """Pick `size` rows with a seeded sampler and return them in their original order.

    Only the instructions are kept; AlpacaEval's reference outputs are not used.
    """

    if size > len(rows):
        raise ValueError(f"cannot sample {size} prompts from {len(rows)}")
    chosen = sorted(random.Random(seed).sample(range(len(rows)), size))
    return [
        {
            "id": f"alpaca-eval-{seed}-{position:04d}",
            "instruction": rows[index]["instruction"],
            "dataset": "alpaca_eval",
            "source_dataset": rows[index]["dataset"],
            "source_index": index,
        }
        for position, index in enumerate(chosen)
    ]


def download_source() -> tuple[list[dict], dict]:
    """Fetch the raw 805-prompt file from the Hub.

    The raw JSON is read directly because the Hub repository's loading script is not
    supported by datasets 4.x.
    """

    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(SOURCE_REPO).sha
    path = hf_hub_download(SOURCE_REPO, SOURCE_FILE, repo_type="dataset", revision=revision)
    raw = Path(path).read_bytes()
    source = {
        "repo": SOURCE_REPO,
        "file": SOURCE_FILE,
        "revision": revision,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return json.loads(raw), source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="eval_sets/alpaca_eval_300_seed42.json")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    rows, source = download_source()
    subset = select_subset(rows, args.size, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n")

    meta = {
        "source": source,
        "source_rows": len(rows),
        "size": args.size,
        "seed": args.seed,
        "selection": "sorted(random.Random(seed).sample(range(source_rows), size))",
        "source_mix": _count(row["source_dataset"] for row in subset),
    }
    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {len(subset)} prompts to {out_path}")
    return 0


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


if __name__ == "__main__":
    raise SystemExit(main())
