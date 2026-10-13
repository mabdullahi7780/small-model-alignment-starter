"""Supervised fine-tuning with a hand-written PyTorch training loop (Task 3).

fp32 master weights with fp16 autocast and dynamic loss scaling, gradient accumulation with
an exact token-mean loss per optimizer step, gradient clipping, AdamW with linear warmup and
cosine decay, validation every `eval_every` steps, and only the best and final checkpoints.

Usage:
    python -m pa2.train_sft --config configs/sft.yaml --data-dir outputs/sft/data --out-dir outputs/sft/run
    python -m pa2.train_sft ... --max-steps 20 --no-save      # short timing run
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import torch

from pa2.batching import TokenizedPair, collate, pad_token_id, tokenize_pair
from pa2.formatting import format_prompt
from pa2.losses import response_logprobs_from_logits


class MetricsWriter:
    """Appends one JSON object per line and flushes, so partial runs keep their history."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a")

    def log(self, record: dict) -> None:
        self.handle.write(json.dumps(record) + "\n")
        self.handle.flush()
        print(json.dumps(record), flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenize_examples(tokenizer, rows: list[dict], max_length: int) -> list[TokenizedPair]:
    return [tokenize_pair(tokenizer, format_prompt(row["prompt"]), row["response"], max_length) for row in rows]


def warmup_cosine(total_steps: int, warmup_steps: int):
    """Multiplier on the peak learning rate after `step` optimizer steps."""

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return multiplier


def param_groups(model, weight_decay: float) -> list[dict]:
    """Weight decay on matrices only; biases and norm weights (1-D tensors) are not decayed."""

    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 else no_decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def autocast(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def length_sorted_batches(pairs: list[TokenizedPair], batch_size: int, pad_id: int) -> list[dict]:
    order = sorted(range(len(pairs)), key=lambda index: len(pairs[index].input_ids))
    return [collate([pairs[i] for i in order[start : start + batch_size]], pad_id) for start in range(0, len(order), batch_size)]


@torch.no_grad()
def evaluate(model, batches: list[dict], device: torch.device, use_amp: bool) -> dict:
    """Token-mean response NLL over the whole validation set."""

    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in batches:
        with autocast(device, use_amp):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
        logps, counts = response_logprobs_from_logits(logits, batch["labels"].to(device))
        total_nll -= logps.sum().item()
        total_tokens += int(counts.sum().item())
    model.train()
    loss = total_nll / max(total_tokens, 1)
    return {"val_loss": loss, "val_perplexity": math.exp(min(loss, 50.0)), "val_tokens": total_tokens}


def save_checkpoint(model, tokenizer, path: Path, info: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    (path / "checkpoint_info.json").write_text(json.dumps(info, indent=2) + "\n")


def peak_memory_gib(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 2),
    }


def train(config: dict, data_dir: Path, out_dir: Path, max_steps: int | None = None, save: bool = True) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and config.get("precision") == "fp16_autocast"

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForCausalLM.from_pretrained(config["model"], dtype=torch.float32).to(device)
    model.config.use_cache = False
    if config.get("gradient_checkpointing"):
        # Recompute activations in the backward pass instead of storing them: less memory, more compute.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    pad_id = pad_token_id(tokenizer)
    train_pairs = tokenize_examples(tokenizer, read_jsonl(data_dir / "train.jsonl")[: config["train_examples"]], config["max_length"])
    val_pairs = tokenize_examples(tokenizer, read_jsonl(data_dir / "val.jsonl")[: config["val_examples"]], config["max_length"])
    val_batches = length_sorted_batches(val_pairs, config["eval_batch_size"], pad_id)

    batch_size, accum = config["per_device_batch_size"], config["grad_accum_steps"]
    micro_per_epoch = math.ceil(len(train_pairs) / batch_size)
    total_steps = math.ceil(micro_per_epoch / accum) * config["epochs"]
    steps_to_run = min(max_steps, total_steps) if max_steps else total_steps
    warmup_steps = max(1, round(config["warmup_ratio"] * total_steps))

    optimizer = torch.optim.AdamW(
        param_groups(model, config["weight_decay"]), lr=config["lr"], betas=tuple(config["betas"]), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps, warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**config, "total_steps": total_steps, "warmup_steps": warmup_steps,
                                                     "train_pairs": len(train_pairs), "val_pairs": len(val_pairs)}, indent=2) + "\n")
    metrics = MetricsWriter(out_dir / "metrics.jsonl")

    eval_started = time.time()
    initial = evaluate(model, val_batches, device, use_amp)
    eval_seconds = time.time() - eval_started
    metrics.log({"type": "eval", "step": 0, **initial, "eval_seconds": round(eval_seconds, 1)})
    best = {"step": 0, "val_loss": initial["val_loss"]}

    generator = torch.Generator().manual_seed(config["seed"])
    step, skipped_steps, tokens_seen = 0, 0, 0
    last_eval, pause_seconds = initial, 0.0  # pause_seconds: time spent evaluating and saving
    train_started = time.time()
    window_started = time.time()
    done = False
    for epoch in range(config["epochs"]):
        order = torch.randperm(len(train_pairs), generator=generator).tolist()
        micro_batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
        for window_start in range(0, len(micro_batches), accum):
            window = [collate([train_pairs[i] for i in indices], pad_id) for indices in micro_batches[window_start : window_start + accum]]
            # Normalising by the window's total response tokens makes the accumulated gradient
            # exactly the gradient of the token-mean loss over all examples in the step.
            window_tokens = max(sum(int(batch["labels"][:, 1:].ne(-100).sum()) for batch in window), 1)
            window_nll = 0.0
            for batch in window:
                with autocast(device, use_amp):
                    logits = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    ).logits
                logps, _ = response_logprobs_from_logits(logits, batch["labels"].to(device))
                loss = -logps.sum() / window_tokens
                scaler.scale(loss).backward()
                window_nll -= logps.sum().item()
                tokens_seen += int(batch["attention_mask"].sum())

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                skipped_steps += 1  # inf/NaN gradients: the scaler skipped this update
            optimizer.zero_grad(set_to_none=True)
            lr_used = scheduler.get_last_lr()[0]
            scheduler.step()
            step += 1

            if step == 1 or step % config["log_every"] == 0 or step == steps_to_run:
                elapsed = time.time() - train_started
                metrics.log({
                    "type": "train", "step": step, "epoch": epoch, "train_loss": window_nll / window_tokens,
                    "lr": lr_used, "grad_norm": float(grad_norm), "loss_scale": scaler.get_scale() if use_amp else 1.0,
                    "skipped_steps": skipped_steps, "seconds_per_step": round((time.time() - window_started) / (1 if step == 1 else config["log_every"]), 3),
                    "elapsed_seconds": round(elapsed, 1), "tokens_per_second": round(tokens_seen / max(elapsed, 1e-9)),
                    **peak_memory_gib(device),
                })
                window_started = time.time()

            if step % config["eval_every"] == 0 or step == steps_to_run:
                paused = time.time()
                last_eval = evaluate(model, val_batches, device, use_amp)
                metrics.log({"type": "eval", "step": step, **last_eval})
                if last_eval["val_loss"] < best["val_loss"]:
                    best = {"step": step, "val_loss": last_eval["val_loss"]}
                    if save:
                        save_checkpoint(model, tokenizer, out_dir / "best", {"step": step, **last_eval})
                pause_seconds += time.time() - paused
                window_started = time.time()
            if step >= steps_to_run:
                done = True
                break
        if done:
            break

    train_seconds = time.time() - train_started - pause_seconds
    if save:
        if best["step"] == 0:
            save_checkpoint(model, tokenizer, out_dir / "best", {"step": step, "note": "validation never improved"})
        save_checkpoint(model, tokenizer, out_dir / "final", {"step": step, **last_eval})

    seconds_per_step = train_seconds / max(step, 1)
    evals_in_full_run = total_steps // config["eval_every"] + 2
    summary = {
        "steps_run": step,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "initial_val_loss": initial["val_loss"],
        "best_step": best["step"],
        "best_val_loss": best["val_loss"],
        "final_val_loss": last_eval["val_loss"],
        "skipped_steps": skipped_steps,
        "train_seconds": round(train_seconds, 1),
        "seconds_per_step": round(seconds_per_step, 3),
        "eval_seconds": round(eval_seconds, 1),
        "projected_full_run_hours": round((seconds_per_step * total_steps + eval_seconds * evals_in_full_run) / 3600, 2),
        **peak_memory_gib(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (timing run)")
    parser.add_argument("--no-save", action="store_true", help="do not write checkpoints")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        config[key] = yaml.safe_load(value)
    train(config, Path(args.data_dir), Path(args.out_dir), args.max_steps, save=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
