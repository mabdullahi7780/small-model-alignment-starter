"""Batched greedy generation for a fixed prompt file (Task 2 baseline, Task 6 evaluation).

Prompts are left-padded so every row's next token is predicted at the same column; the
attention mask hides the pads and the position ids are counted over real tokens only.

Usage (from the repository root):
    python -m pa2.generate --model Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base \\
        --prompts eval_sets/alpaca_eval_300_seed42.json --out outputs/baseline/base_greedy.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from pa2.formatting import format_prompt, stop_token_ids

TURN_START = "<|im_start|>"


def load_prompts(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = json.loads(Path(path).read_text())
    for row in rows:
        if "id" not in row or "instruction" not in row:
            raise ValueError(f"{path}: every row needs 'id' and 'instruction'")
    return rows[:limit] if limit else rows


def cut_at_stop(token_ids: list[int], stop_ids: list[int], eos_id: int | None) -> tuple[list[int], str]:
    """Keep the generated tokens before the first stop token and say why generation ended."""

    for position, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            return token_ids[:position], "eos" if token_id == eos_id else "turn_end"
    return token_ids, "max_new_tokens"


def has_repetition_loop(text: str, ngram: int = 6, min_repeats: int = 3) -> bool:
    """True when some run of `ngram` words appears `min_repeats` or more times."""

    words = text.split()
    if len(words) < ngram * min_repeats:
        return False
    counts = Counter(tuple(words[i : i + ngram]) for i in range(len(words) - ngram + 1))
    return max(counts.values()) >= min_repeats


def generate_responses(
    model,
    tokenizer,
    instructions: list[str],
    *,
    batch_size: int = 32,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    formatter=format_prompt,
) -> list[dict]:
    """Greedy-decode one response per instruction, returned in the input order.

    `formatter` turns an instruction into the prompt text: our shared format by default, or
    a model's own chat template (SmolLM2 in Task 5).
    """

    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = stop_token_ids(tokenizer)
    turn_start_id = tokenizer.convert_tokens_to_ids(TURN_START)
    device = next(model.parameters()).device

    texts = [formatter(instruction) for instruction in instructions]
    lengths = [len(tokenizer(text, add_special_tokens=False).input_ids) for text in texts]
    # Longest prompts first: similar lengths share a batch (less padding), and an
    # out-of-memory error shows up on the first batch rather than the last.
    order = sorted(range(len(texts)), key=lambda index: -lengths[index])

    results: list[dict | None] = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        encoded = tokenizer(
            [texts[index] for index in batch_indices],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded.input_ids.shape[1] :].tolist()
        for row, index in enumerate(batch_indices):
            kept, stop_reason = cut_at_stop(new_tokens[row], stop_ids, tokenizer.eos_token_id)
            text = tokenizer.decode(kept, skip_special_tokens=True).strip()
            results[index] = {
                "output": text,
                "prompt_tokens": lengths[index],
                "new_tokens": len(kept),
                "stop_reason": stop_reason,
                "wrote_new_turn": turn_start_id in kept,
                "repetition_loop": has_repetition_loop(text),
            }
        done = min(start + batch_size, len(order))
        print(f"generated {done}/{len(order)}", flush=True)
    return results


def summarize(records: list[dict]) -> dict:
    count = len(records)
    new_tokens = [record["new_tokens"] for record in records]
    return {
        "count": count,
        "stopped_on_its_own": sum(r["stop_reason"] != "max_new_tokens" for r in records) / count,
        "hit_max_new_tokens": sum(r["stop_reason"] == "max_new_tokens" for r in records) / count,
        "wrote_new_turn": sum(r["wrote_new_turn"] for r in records) / count,
        "repetition_loop": sum(r["repetition_loop"] for r in records) / count,
        "empty_output": sum(not r["output"] for r in records) / count,
        "mean_new_tokens": statistics.fmean(new_tokens),
        "median_new_tokens": statistics.median(new_tokens),
        "mean_words": statistics.fmean(len(r["output"].split()) for r in records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hub id or local checkpoint directory")
    parser.add_argument("--tokenizer", default=None, help="defaults to --model")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory to apply on top of --model")
    parser.add_argument("--prompt-format", default="chatml", choices=["chatml", "chat_template"],
                        help="chatml: the shared Qwen format; chat_template: the tokenizer's own template")
    parser.add_argument("--generator", required=True, help="name stored in every record")
    parser.add_argument("--prompts", required=True, help="JSON list with id and instruction")
    parser.add_argument("--dataset", default=None, help="defaults to each prompt's 'dataset' field")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity check)")
    parser.add_argument("--show", type=int, default=0, help="print this many prompt and output pairs")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.skip_existing and out_path.exists():
        print(f"{out_path} exists, skipping")
        return 0

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(args.prompts, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to(args.device).eval()
    if args.prompt_format == "chat_template":
        def formatter(instruction):
            return tokenizer.apply_chat_template([{"role": "user", "content": instruction.strip()}],
                                                 tokenize=False, add_generation_prompt=True)
    else:
        formatter = format_prompt

    started = time.time()
    outputs = generate_responses(
        model,
        tokenizer,
        [row["instruction"] for row in prompts],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        formatter=formatter,
    )
    seconds = time.time() - started

    records = [
        {
            "id": row["id"],
            "instruction": row["instruction"],
            "output": result["output"],
            "generator": args.generator,
            "dataset": args.dataset or row.get("dataset", Path(args.prompts).stem),
            **{key: value for key, value in result.items() if key != "output"},
        }
        for row, result in zip(prompts, outputs)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    meta = {
        "generator": args.generator,
        "model": args.model,
        "prompts_file": args.prompts,
        "prompts_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        "decoding": {
            "strategy": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "stop_tokens": tokenizer.convert_ids_to_tokens(stop_token_ids(tokenizer)),
            "prompt_format": formatter("{instruction}"),
        },
        "adapter": args.adapter,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "seconds": round(seconds, 1),
        "stats": summarize(records),
    }
    out_path.with_name(out_path.stem + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta["stats"], indent=2))
    for record in records[: args.show]:
        print(f"\n### {record['id']} [{record['stop_reason']}, {record['new_tokens']} tokens]")
        print("PROMPT:", record["instruction"])
        print("OUTPUT:", record["output"][:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
