"""LoRA persona tuning with TRL (Task 5): data handling, adapter size and a tiny end-to-end run."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import torch
import yaml

from pa2 import generate, lora_persona, plots

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((REPO_ROOT / "configs" / "lora_persona.yaml").read_text())
PAIRS = REPO_ROOT / "data" / "persona_lora" / "ramsay_style_preferences.json"
EVAL_SET = REPO_ROOT / "eval_sets" / "persona_style_eval_40.json"


def lora_parameter_count(hidden, intermediate, layers, heads, kv_heads, rank):
    """r * (d_in + d_out) per adapted projection, summed over the seven projections of every block."""

    head_dim = hidden // heads
    kv = kv_heads * head_dim
    shapes = [(hidden, hidden), (hidden, kv), (hidden, kv), (hidden, hidden),  # q, k, v, o
              (hidden, intermediate), (hidden, intermediate), (intermediate, hidden)]  # gate, up, down
    return layers * sum(rank * (d_in + d_out) for d_in, d_out in shapes)


def test_split_is_seeded_disjoint_and_complete():
    rows = lora_persona.load_pairs(PAIRS)

    train, val = lora_persona.split_pairs(rows, CONFIG["val_pairs"], CONFIG["seed"])
    again, _ = lora_persona.split_pairs(rows, CONFIG["val_pairs"], CONFIG["seed"])

    assert len(rows) == 220 and len(train) == 200 and len(val) == 20
    assert train == again
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in val}


def test_logprob_fallback_matches_a_direct_log_softmax():
    torch.manual_seed(0)
    hidden, weight, targets = torch.randn(5, 8), torch.randn(11, 8), torch.tensor([0, 3, 10, 2, 7])

    token, entropy, *_ = lora_persona.TorchLogProbFallback.apply(hidden, weight, None, targets, 1.0, 4, None, 1.0, ("log_probs", "entropy"))

    expected = torch.log_softmax(hidden @ weight.t(), dim=-1)
    torch.testing.assert_close(token, expected[torch.arange(5), targets])
    torch.testing.assert_close(entropy, -(expected.exp() * expected).sum(-1))


def test_rows_become_trl_conversational_records():
    record = lora_persona.to_conversational({"prompt": " Q? ", "chosen": "Loud answer!", "rejected": "Calm answer."})

    assert record == {
        "prompt": [{"role": "user", "content": "Q?"}],
        "chosen": [{"role": "assistant", "content": "Loud answer!"}],
        "rejected": [{"role": "assistant", "content": "Calm answer."}],
    }


def test_adapter_targets_every_linear_projection_and_is_about_2_percent_of_smollm2():
    config = lora_persona.lora_config(CONFIG)

    assert set(config.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert (config.r, config.lora_alpha, config.lora_dropout) == (16, 32, 0.05)
    # SmolLM2-360M: hidden 960, MLP 2560, 32 blocks, 15 heads, 5 key/value heads
    assert lora_parameter_count(960, 2560, 32, 15, 5, 16) == 8_683_520


def test_persona_eval_set_is_balanced_and_unseen_in_training():
    rows = json.loads(EVAL_SET.read_text())
    train_prompts = [r["prompt"] for r in lora_persona.load_pairs(PAIRS)]
    words = lambda text: set(re.findall(r"[a-z0-9']+", text.lower()))  # noqa: E731

    assert len(rows) == 40 and len({r["instruction"] for r in rows}) == 40
    assert {c: sum(r["category"] == c for r in rows) for c in ("cooking", "coding", "general", "safety")} == \
        {"cooking": 10, "coding": 10, "general": 10, "safety": 10}
    for row in rows:
        overlap = max(len(words(row["instruction"]) & words(p)) / len(words(row["instruction"]) | words(p)) for p in train_prompts)
        assert overlap < 0.35, row["instruction"]


def test_tiny_lora_dpo_run_starts_at_ln2_learns_saves_and_generates(tmp_path):
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(CONFIG["model"])
    except OSError:
        pytest.skip("SmolLM2 tokenizer not available offline")
    torch.manual_seed(0)
    tiny = transformers.LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=True)
    model_dir = tmp_path / "tiny"
    transformers.LlamaForCausalLM(tiny).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = {**CONFIG, "model": str(model_dir), "data": str(PAIRS), "per_device_batch_size": 2,
              "grad_accum_steps": 1, "eval_steps": 2, "lr": 1e-3}

    summary = lora_persona.train(config, tmp_path / "run", max_steps=4)

    assert summary["parameters"]["trainable"] == lora_parameter_count(32, 64, 2, 4, 2, 16)
    assert summary["logprob_kernel"] in {"trl triton kernel", "pytorch fallback"}
    assert summary["initial_eval"]["eval_loss"] == pytest.approx(math.log(2), abs=1e-5)
    assert summary["initial_eval"]["eval_rewards/accuracies"] == 0.0
    assert summary["final_eval"]["eval_rewards/margins"] > 0
    assert (tmp_path / "run" / "adapter" / "adapter_model.safetensors").exists()
    assert plots.plot_lora(tmp_path / "run" / "log_history.json", tmp_path / "lora.png").stat().st_size > 10_000

    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"id": "p0", "instruction": "How do I boil an egg?", "dataset": "toy"}]))
    out = tmp_path / "gen" / "lora.json"
    assert generate.main(["--model", str(model_dir), "--adapter", str(tmp_path / "run" / "adapter"),
                          "--prompt-format", "chat_template", "--generator", "lora", "--prompts", str(prompts),
                          "--out", str(out), "--device", "cpu", "--max-new-tokens", "4"]) == 0
    meta = json.loads(out.with_name("lora.meta.json").read_text())
    assert meta["decoding"]["prompt_format"].startswith("<|im_start|>system")
    assert json.loads(out.read_text())[0]["generator"] == "lora"
