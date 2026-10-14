"""LoRA persona tuning with TRL's DPOTrainer (Task 5, "Idiot Sandwich").

HuggingFaceTB/SmolLM2-360M-Instruct learns a Gordon Ramsay speaking style from the provided
preference pairs, where the chosen and rejected answers give the same advice in different
tones. Only LoRA adapters are trained. TRL computes the reference log-probabilities by running
the same network with the adapters switched off, so no second model copy is needed. This is
the one task in the assignment that is meant to use TRL and PEFT.

Usage:
    python -m pa2.lora_persona --config configs/lora_persona.yaml --out-dir outputs/lora/run
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_pairs(path: str | Path) -> list[dict]:
    rows = json.loads(Path(path).read_text())
    for row in rows:
        if not {"prompt", "chosen", "rejected"} <= set(row):
            raise ValueError(f"{path}: every row needs prompt, chosen and rejected")
    return rows


def split_pairs(rows: list[dict], val_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded split: the first `val_size` rows of a shuffled order are held out for evaluation."""

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    held_out = set(order[:val_size])
    train = [rows[i] for i in range(len(rows)) if i not in held_out]
    val = [rows[i] for i in sorted(held_out)]
    return train, val


def to_conversational(row: dict) -> dict:
    """TRL's conversational preference format; the trainer applies the model's chat template."""

    return {
        "prompt": [{"role": "user", "content": row["prompt"].strip()}],
        "chosen": [{"role": "assistant", "content": row["chosen"].strip()}],
        "rejected": [{"role": "assistant", "content": row["rejected"].strip()}],
    }


class TorchLogProbFallback:
    """Plain PyTorch version of TRL's chunked log-probability kernel.

    TRL's DPOTrainer computes per-token log-probabilities with a Triton kernel, and Triton is
    only available on Linux. Where the kernel cannot be imported, this computes the same values
    (log-softmax of hidden @ W^T, gathered at the target tokens) with ordinary autograd.
    """

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, softcap=None, logit_scale=1.0, outputs=("log_probs",)):
        import torch

        logits = hidden @ weight.t()
        if bias is not None:
            logits = logits + bias
        logits = logits.float() * logit_scale
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        logp = torch.log_softmax(logits / temperature, dim=-1)
        token = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1) if "entropy" in outputs else None
        sq = torch.logsumexp(2 * logp, -1) if "log_sum_sq_probs" in outputs else None
        mean = logits.mean(-1).detach() if "mean_logits" in outputs else None
        top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
        return token, entropy, sq, mean, top1


def ensure_logprob_kernel() -> str:
    """Use TRL's Triton kernel when it loaded; otherwise install the PyTorch fallback."""

    import trl.trainer.utils as trl_utils

    if getattr(trl_utils, "_ChunkedLogProbFunction", None) is None:
        trl_utils._ChunkedLogProbFunction = TorchLogProbFallback
        return "pytorch fallback"
    return "trl triton kernel"


def lora_config(config: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "trainable_share": round(trainable / total, 5)}


def train(config: dict, out_dir: Path, max_steps: int | None = None) -> dict:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(config["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel = ensure_logprob_kernel()
    print(json.dumps({"logprob_kernel": kernel}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], dtype=torch.float32)
    model.config.use_cache = False

    train_rows, val_rows = split_pairs(load_pairs(config["data"]), config["val_pairs"], config["seed"])
    (out_dir / "split.json").write_text(json.dumps(
        {"train_prompts": [r["prompt"] for r in train_rows], "val_prompts": [r["prompt"] for r in val_rows]}, indent=2) + "\n")
    train_dataset = Dataset.from_list([to_conversational(r) for r in train_rows])
    eval_dataset = Dataset.from_list([to_conversational(r) for r in val_rows])

    use_fp16 = torch.cuda.is_available() and config.get("precision") == "fp16"
    args = DPOConfig(
        output_dir=str(out_dir / "trainer"),
        beta=config["beta"],
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        per_device_train_batch_size=config["per_device_batch_size"],
        per_device_eval_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["grad_accum_steps"],
        num_train_epochs=config["epochs"],
        max_steps=max_steps or -1,
        max_length=config["max_length"],
        max_grad_norm=config["grad_clip"],
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="no",
        report_to="none",
        seed=config["seed"],
        fp16=use_fp16,
        bf16=False,
        disable_dropout=False,  # keep the LoRA dropout active; the base model has no dropout of its own
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config(config),
    )
    parameters = count_parameters(trainer.model)
    print(json.dumps({"parameters": parameters}), flush=True)

    # With B = 0 the adapted model equals the base model, so this must give ln 2 and zero rewards.
    initial = trainer.evaluate()
    result = trainer.train()
    final = trainer.evaluate()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n")

    summary = {
        "model": config["model"],
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "parameters": parameters,
        "steps": trainer.state.global_step,
        "train_runtime_seconds": round(result.metrics.get("train_runtime", 0.0), 1),
        "initial_eval": initial,
        "final_eval": final,
        "fp16": use_fp16,
        "logprob_kernel": kernel,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/lora_persona.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        config[key] = yaml.safe_load(value)
    train(config, Path(args.out_dir), args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
