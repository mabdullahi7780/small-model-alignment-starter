"""A tiny end-to-end SFT run on CPU: the loop learns, logs, evaluates and saves (Task 3)."""

from __future__ import annotations

import json
from collections import Counter

import pytest
import torch

from pa2 import plots, sft_data, train_sft


def test_warmup_then_cosine_schedule():
    multiplier = train_sft.warmup_cosine(total_steps=100, warmup_steps=4)

    assert [round(multiplier(step), 2) for step in range(4)] == [0.25, 0.5, 0.75, 1.0]
    assert multiplier(4) == pytest.approx(1.0)
    assert multiplier(52) == pytest.approx(0.5, abs=0.02)
    assert multiplier(100) == pytest.approx(0.0, abs=1e-6)


def test_first_turn_and_pair_selection_filters():
    class WordTokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return text.split()

    rows = [
        {"prompt_id": "a", "messages": [{"role": "user", "content": "short question"}, {"role": "assistant", "content": "short answer"}]},
        {"prompt_id": "b", "messages": [{"role": "user", "content": "Short   QUESTION"}, {"role": "assistant", "content": "dup"}]},
        {"prompt_id": "c", "messages": [{"role": "assistant", "content": "starts wrong"}, {"role": "user", "content": "x"}]},
        {"prompt_id": "d", "messages": [{"role": "user", "content": "too long " * 20}, {"role": "assistant", "content": "y"}]},
        {"prompt_id": "e", "messages": [{"role": "user", "content": "an eval prompt"}, {"role": "assistant", "content": "z"}]},
        {"prompt_id": "f", "messages": [{"role": "user", "content": "fine"}, {"role": "assistant", "content": "ok"}]},
    ]
    stats = Counter()

    kept = sft_data.select_pairs(rows, WordTokenizer(), count=2, max_length=12, seed=0,
                                 excluded={"an eval prompt"}, prefix="t", stats=stats)

    assert {pair["source_id"] for pair in kept} <= {"a", "b", "f"}
    assert len({sft_data.normalize(pair["prompt"]) for pair in kept}) == 2
    assert stats["kept"] == 2


def tiny_setup(tmp_path):
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except OSError:
        pytest.skip("Qwen tokenizer not available offline")
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=True,
    )
    model_dir = tmp_path / "tiny"
    transformers.Qwen2ForCausalLM(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    words = ["red", "blue", "green", "gold"]
    rows = [{"prompt": f"Repeat the word {w} twice.", "response": f"{w} {w}"} for w in words * 16]
    for name, subset in (("train.jsonl", rows), ("val.jsonl", rows[:8])):
        (data_dir / name).write_text("".join(json.dumps(row) + "\n" for row in subset))
    config = {
        "model": str(model_dir), "seed": 0, "max_length": 64, "train_examples": 64, "val_examples": 8,
        "per_device_batch_size": 2, "grad_accum_steps": 2, "epochs": 2, "lr": 3e-3, "betas": [0.9, 0.999],
        "weight_decay": 0.1, "warmup_ratio": 0.1, "grad_clip": 1.0, "precision": "fp16_autocast",
        "eval_every": 8, "eval_batch_size": 4, "log_every": 4,
    }
    return config, data_dir


def test_tiny_run_learns_and_writes_metrics_checkpoints_and_plot(tmp_path):
    config, data_dir = tiny_setup(tmp_path)
    out_dir = tmp_path / "run"

    summary = train_sft.train(config, data_dir, out_dir)

    records = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in records if r["type"] == "eval"]
    assert summary["total_steps"] == 32 and summary["steps_run"] == 32
    assert summary["best_val_loss"] < summary["initial_val_loss"] - 1.0  # nats; 11.9 -> 8.9 on this toy task
    assert [r["step"] for r in evals] == [0, 8, 16, 24, 32]
    assert (out_dir / "best" / "model.safetensors").exists() and (out_dir / "final" / "model.safetensors").exists()
    assert json.loads((out_dir / "best" / "checkpoint_info.json").read_text())["step"] == summary["best_step"]
    assert plots.plot_sft(out_dir / "metrics.jsonl", tmp_path / "curves.png").stat().st_size > 10_000


def test_timing_run_stops_early_and_saves_nothing(tmp_path):
    config, data_dir = tiny_setup(tmp_path)
    out_dir = tmp_path / "smoke"

    summary = train_sft.train(config, data_dir, out_dir, max_steps=3, save=False)

    assert summary["steps_run"] == 3 and summary["projected_full_run_hours"] >= 0
    assert not (out_dir / "best").exists() and not (out_dir / "final").exists()
