"""Direct Preference Optimization with a hand-written PyTorch loop (Task 4).

The policy starts from the SFT checkpoint and trains on `policy_device`; a second copy of the
same checkpoint is the frozen reference on `reference_device`. Both run the identical compute
path (fp32 weights, fp16 autocast), so at step 0 every implicit reward is exactly zero.

Per pair: the chosen and rejected rows go through each model in one forward pass, their
response log-probabilities are summed, and the loss is -log sigmoid(beta * (policy gap -
reference gap)). Gradients flow into the policy only. Validation loss, reward accuracy and
reward margin are tracked every `eval_every` steps; the best checkpoint is the one with the
highest validation reward accuracy (ties broken by lower validation loss).

Usage:
    python -m pa2.train_dpo --config configs/dpo_hh.yaml --sft-model <sft best dir> \\
        --data-dir outputs/dpo/hh/data --out-dir outputs/dpo/hh/run
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from pa2.batching import TokenizedPair, collate, pad_token_id, tokenize_pair
from pa2.formatting import format_prompt
from pa2.losses import dpo_loss_per_pair, dpo_metrics, dpo_rewards, response_logprobs_from_logits
from pa2.train_sft import MetricsWriter, autocast, peak_memory_gib, read_jsonl, save_checkpoint, set_seed

Pair = tuple[TokenizedPair, TokenizedPair]


def tokenize_preferences(tokenizer, rows: list[dict], max_length: int) -> list[Pair]:
    pairs = []
    for row in rows:
        prompt = format_prompt(row["prompt"])
        pairs.append((tokenize_pair(tokenizer, prompt, row["chosen"], max_length),
                      tokenize_pair(tokenizer, prompt, row["rejected"], max_length)))
    return pairs


def pairs_batch(pairs: list[Pair], pad_id: int) -> dict:
    """Chosen rows first, then rejected rows, right-padded together."""

    return collate([chosen for chosen, _ in pairs] + [rejected for _, rejected in pairs], pad_id)


def sequence_logps(model, batch: dict, device: torch.device, use_amp: bool) -> torch.Tensor:
    with autocast(device, use_amp):
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
    logps, _ = response_logprobs_from_logits(logits, batch["labels"].to(device))
    return logps


@torch.no_grad()
def score_pairs(model, pairs: list[Pair], pad_id: int, device: torch.device, use_amp: bool, batch_pairs: int):
    """Chosen and rejected log-probabilities for every pair, on the CPU, in input order."""

    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    chosen, rejected = [], []
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start : start + batch_pairs]
        logps = sequence_logps(model, pairs_batch(chunk, pad_id), device, use_amp).float().cpu()
        chosen.append(logps[: len(chunk)])
        rejected.append(logps[len(chunk) :])
    model.train(was_training)
    return torch.cat(chosen), torch.cat(rejected)


def resolve_devices(config: dict) -> tuple[torch.device, torch.device]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cpu")
    policy = torch.device(config["policy_device"])
    reference = torch.device(config["reference_device"])
    if reference.index is not None and reference.index >= torch.cuda.device_count():
        reference = policy  # one GPU: both models share it
    return policy, reference


def train(config: dict, sft_model: str, data_dir: Path, out_dir: Path, max_steps: int | None = None, save: bool = True) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(config["seed"])
    policy_device, reference_device = resolve_devices(config)
    use_amp = policy_device.type == "cuda" and config.get("precision") == "fp16_autocast"
    beta = config["beta"]

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    policy = AutoModelForCausalLM.from_pretrained(sft_model, dtype=torch.float32).to(policy_device)
    policy.config.use_cache = False
    policy.train()
    reference = AutoModelForCausalLM.from_pretrained(sft_model, dtype=torch.float32).to(reference_device)
    reference.config.use_cache = False
    reference.eval()
    reference.requires_grad_(False)

    pad_id = pad_token_id(tokenizer)
    train_pairs = tokenize_preferences(tokenizer, read_jsonl(data_dir / "train.jsonl")[: config["train_pairs"]], config["max_length"])
    val_pairs = tokenize_preferences(tokenizer, read_jsonl(data_dir / "val.jsonl")[: config["val_pairs"]], config["max_length"])

    pairs_per_micro, accum = config["per_device_batch_size"], config["grad_accum_steps"]
    micro_per_epoch = math.ceil(len(train_pairs) / pairs_per_micro)
    total_steps = math.ceil(micro_per_epoch / accum) * config["epochs"]
    steps_to_run = min(max_steps, total_steps) if max_steps else total_steps

    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                  eps=1e-8, weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**config, "sft_model": str(sft_model), "total_steps": total_steps,
                                                     "train_pairs_loaded": len(train_pairs), "val_pairs_loaded": len(val_pairs),
                                                     "policy_device": str(policy_device), "reference_device": str(reference_device)},
                                                    indent=2) + "\n")
    metrics = MetricsWriter(out_dir / "metrics.jsonl")

    # The reference is frozen, so its validation scores are computed once and reused.
    eval_batch = config["eval_batch_pairs"]
    started = time.time()
    val_ref_chosen, val_ref_rejected = score_pairs(reference, val_pairs, pad_id, reference_device, use_amp, eval_batch)

    def evaluate() -> dict:
        chosen, rejected = score_pairs(policy, val_pairs, pad_id, policy_device, use_amp, eval_batch)
        return dpo_metrics(chosen, rejected, val_ref_chosen, val_ref_rejected, beta)

    initial = {"step": 0, **evaluate()}
    eval_seconds = time.time() - started
    metrics.log({"type": "eval", **initial, "eval_seconds": round(eval_seconds, 1)})
    best = last_eval = initial

    generator = torch.Generator().manual_seed(config["seed"])
    step, skipped_steps, pause_seconds = 0, 0, 0.0
    train_started = window_started = time.time()
    done = False
    for epoch in range(config["epochs"]):
        order = torch.randperm(len(train_pairs), generator=generator).tolist()
        micro_batches = [order[start : start + pairs_per_micro] for start in range(0, len(order), pairs_per_micro)]
        for window_start in range(0, len(micro_batches), accum):
            window = micro_batches[window_start : window_start + accum]
            window_pairs = sum(len(indices) for indices in window)
            logged = {"loss": 0.0, "correct": 0, "margin": 0.0, "chosen": 0.0, "rejected": 0.0}
            for indices in window:
                batch = pairs_batch([train_pairs[i] for i in indices], pad_id)
                count = len(indices)
                # Reference first: its GPU works while the policy's forward pass is queued.
                with torch.no_grad():
                    reference_logps = sequence_logps(reference, batch, reference_device, use_amp)
                policy_logps = sequence_logps(policy, batch, policy_device, use_amp)
                reference_logps = reference_logps.to(policy_device)
                args = (policy_logps[:count], policy_logps[count:], reference_logps[:count], reference_logps[count:], beta)
                losses = dpo_loss_per_pair(*args)
                scaler.scale(losses.sum() / window_pairs).backward()
                with torch.no_grad():
                    chosen_reward, rejected_reward = dpo_rewards(*args)
                    logged["loss"] += losses.sum().item()
                    logged["correct"] += int((chosen_reward > rejected_reward).sum())
                    logged["margin"] += (chosen_reward - rejected_reward).sum().item()
                    logged["chosen"] += chosen_reward.sum().item()
                    logged["rejected"] += rejected_reward.sum().item()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["grad_clip"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                skipped_steps += 1
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == 1 or step % config["log_every"] == 0 or step == steps_to_run:
                elapsed = time.time() - train_started - pause_seconds
                memory = {f"policy_{k}": v for k, v in peak_memory_gib(policy_device).items()}
                if reference_device != policy_device:
                    memory.update({f"reference_{k}": v for k, v in peak_memory_gib(reference_device).items()})
                metrics.log({
                    "type": "train", "step": step, "epoch": epoch, "loss": logged["loss"] / window_pairs,
                    "reward_accuracy": logged["correct"] / window_pairs, "reward_margin": logged["margin"] / window_pairs,
                    "chosen_reward": logged["chosen"] / window_pairs, "rejected_reward": logged["rejected"] / window_pairs,
                    "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm),
                    "loss_scale": scaler.get_scale() if use_amp else 1.0, "skipped_steps": skipped_steps,
                    "seconds_per_step": round((time.time() - window_started) / (1 if step == 1 else config["log_every"]), 3),
                    "elapsed_seconds": round(elapsed, 1), **memory,
                })
                window_started = time.time()

            if step % config["eval_every"] == 0 or step == steps_to_run:
                paused = time.time()
                last_eval = {"step": step, **evaluate()}
                metrics.log({"type": "eval", **last_eval})
                if (last_eval["reward_accuracy"], -last_eval["loss"]) > (best["reward_accuracy"], -best["loss"]):
                    best = last_eval
                    if save:
                        save_checkpoint(policy, tokenizer, out_dir / "best", last_eval)
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
            save_checkpoint(policy, tokenizer, out_dir / "best", {"step": step, "note": "validation accuracy never improved"})
        save_checkpoint(policy, tokenizer, out_dir / "final", last_eval)

    seconds_per_step = train_seconds / max(step, 1)
    evals_in_full_run = total_steps // config["eval_every"] + 2
    summary = {
        "dataset": config["dataset"],
        "steps_run": step,
        "total_steps": total_steps,
        "initial": initial,
        "best": best,
        "final": last_eval,
        "skipped_steps": skipped_steps,
        "train_seconds": round(train_seconds, 1),
        "seconds_per_step": round(seconds_per_step, 3),
        "eval_seconds": round(eval_seconds, 1),
        "projected_full_run_hours": round((seconds_per_step * total_steps + eval_seconds * evals_in_full_run) / 3600, 2),
        **{f"policy_{k}": v for k, v in peak_memory_gib(policy_device).items()},
        **({f"reference_{k}": v for k, v in peak_memory_gib(reference_device).items()} if reference_device != policy_device else {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-model", required=True, help="directory of the SFT checkpoint (policy start and reference)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (timing run)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        config[key] = yaml.safe_load(value)
    train(config, args.sft_model, Path(args.data_dir), Path(args.out_dir), args.max_steps, save=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
