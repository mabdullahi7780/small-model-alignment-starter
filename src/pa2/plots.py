"""Training curves for the report.

Usage:
    python -m pa2.plots sft --metrics outputs/sft/run/metrics.jsonl --out outputs/sft/sft_curves.png
    python -m pa2.plots dpo --metrics outputs/dpo/hh/run/metrics.jsonl --out outputs/dpo/hh_curves.png --title "DPO on HH-RLHF"
    python -m pa2.plots lora --metrics outputs/lora/run/log_history.json --out outputs/lora/lora_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_metrics(path: str | Path) -> tuple[list[dict], list[dict]]:
    train, evals = [], []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            (train if record["type"] == "train" else evals).append(record)
    return train, evals


def ema(values: list[float], smoothing: float) -> list[float]:
    smoothed, current = [], None
    for value in values:
        current = value if current is None else smoothing * current + (1 - smoothing) * value
        smoothed.append(current)
    return smoothed


def plot_sft(metrics_path: str | Path, out_path: str | Path, smoothing: float = 0.6) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, evals = read_metrics(metrics_path)
    steps = [record["step"] for record in train]
    losses = [record["train_loss"] for record in train]
    best = min(evals, key=lambda record: record["val_loss"])

    fig, (ax_loss, ax_lr) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(steps, losses, color="tab:blue", alpha=0.3, linewidth=1, label="train (logged steps)")
    ax_loss.plot(steps, ema(losses, smoothing), color="tab:blue", linewidth=2, label=f"train (EMA {smoothing})")
    ax_loss.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], "o-", color="tab:orange", label="validation (1k pairs)")
    ax_loss.axvline(best["step"], color="grey", linestyle="--", linewidth=1)
    ax_loss.annotate(f"best checkpoint\nstep {best['step']}, {best['val_loss']:.3f}", (best["step"], best["val_loss"]),
                     textcoords="offset points", xytext=(-90, 25), fontsize=8)
    ax_loss.set_xlabel("optimizer step (32 examples each)")
    ax_loss.set_ylabel("loss (nats per response token)")
    ax_loss.set_title("SFT: response-only cross-entropy")
    ax_loss.legend(fontsize=8)

    ax_lr.plot(steps, [record["lr"] for record in train], color="tab:green")
    ax_lr.set_xlabel("optimizer step")
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("Schedule: linear warmup, cosine decay")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_dpo(metrics_path: str | Path, out_path: str | Path, title: str = "DPO", smoothing: float = 0.6) -> Path:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, evals = read_metrics(metrics_path)
    steps = [record["step"] for record in train]
    eval_steps = [record["step"] for record in evals]
    best = max(evals, key=lambda record: (record["reward_accuracy"], -record["loss"]))

    fig, (ax_loss, ax_acc, ax_reward) = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in ((ax_loss, "loss", "DPO loss"), (ax_acc, "reward_accuracy", "reward accuracy")):
        values = [record[key] for record in train]
        ax.plot(steps, values, color="tab:blue", alpha=0.3, linewidth=1, label="train (32-pair window)")
        ax.plot(steps, ema(values, smoothing), color="tab:blue", linewidth=2, label=f"train (EMA {smoothing})")
        ax.plot(eval_steps, [record[key] for record in evals], "o-", color="tab:orange", label="validation (1k pairs)")
        ax.axvline(best["step"], color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("optimizer step (32 pairs each)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    ax_loss.axhline(math.log(2), color="black", linestyle=":", linewidth=1)
    ax_loss.text(steps[-1] if steps else 0, math.log(2), " ln 2 (no change from SFT)", fontsize=8, va="top", ha="right")
    ax_loss.set_title(f"{title}: loss")
    ax_acc.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax_acc.set_title(f"{title}: reward accuracy (best checkpoint: step {best['step']}, {best['reward_accuracy']:.3f})")

    ax_reward.plot(eval_steps, [record["chosen_reward"] for record in evals], "o-", color="tab:green", label="chosen reward")
    ax_reward.plot(eval_steps, [record["rejected_reward"] for record in evals], "o-", color="tab:red", label="rejected reward")
    ax_reward.plot(eval_steps, [record["reward_margin"] for record in evals], "o-", color="tab:purple", label="margin (chosen - rejected)")
    ax_reward.axhline(0, color="black", linestyle=":", linewidth=1)
    ax_reward.set_xlabel("optimizer step")
    ax_reward.set_ylabel("implicit reward, beta x (log pi - log pi_ref)")
    ax_reward.set_title(f"{title}: validation rewards")
    ax_reward.legend(fontsize=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_lora(log_history_path: str | Path, out_path: str | Path, title: str = "LoRA-DPO persona") -> Path:
    """Curves from TRL's log history: training rows carry `loss`, evaluation rows `eval_loss`."""

    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads(Path(log_history_path).read_text())
    train = [row for row in history if "loss" in row and "rewards/accuracies" in row]
    evals = [row for row in history if "eval_loss" in row]

    fig, (ax_loss, ax_acc, ax_reward) = plt.subplots(1, 3, figsize=(15, 4))
    for ax, train_key, eval_key, label in ((ax_loss, "loss", "eval_loss", "DPO loss"),
                                           (ax_acc, "rewards/accuracies", "eval_rewards/accuracies", "reward accuracy")):
        ax.plot([r["step"] for r in train], [r[train_key] for r in train], color="tab:blue", alpha=0.35, linewidth=1, label="train (8-pair step)")
        ax.plot([r["step"] for r in train], ema([r[train_key] for r in train], 0.6), color="tab:blue", linewidth=2, label="train (EMA 0.6)")
        ax.plot([r["step"] for r in evals], [r[eval_key] for r in evals], "o-", color="tab:orange", label="held-out (20 pairs)")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    ax_loss.axhline(math.log(2), color="black", linestyle=":", linewidth=1)
    ax_loss.set_title(f"{title}: loss")
    ax_acc.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax_acc.set_title(f"{title}: reward accuracy")

    steps = [r["step"] for r in evals]
    ax_reward.plot(steps, [r["eval_rewards/chosen"] for r in evals], "o-", color="tab:green", label="chosen (persona) reward")
    ax_reward.plot(steps, [r["eval_rewards/rejected"] for r in evals], "o-", color="tab:red", label="rejected (neutral) reward")
    ax_reward.plot(steps, [r["eval_rewards/margins"] for r in evals], "o-", color="tab:purple", label="margin")
    ax_reward.axhline(0, color="black", linestyle=":", linewidth=1)
    ax_reward.set_xlabel("optimizer step")
    ax_reward.set_ylabel("implicit reward, beta x (log pi - log pi_ref)")
    ax_reward.set_title(f"{title}: held-out rewards")
    ax_reward.legend(fontsize=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["sft", "dpo", "lora"])
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="DPO")
    args = parser.parse_args(argv)
    if args.kind == "sft":
        print("wrote", plot_sft(args.metrics, args.out))
    elif args.kind == "dpo":
        print("wrote", plot_dpo(args.metrics, args.out, args.title))
    else:
        print("wrote", plot_lora(args.metrics, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
