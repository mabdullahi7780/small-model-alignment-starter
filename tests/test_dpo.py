"""DPO loss, metrics, batching, data parsing and a tiny end-to-end run (Task 4)."""

from __future__ import annotations

import json
import math
from collections import Counter

import pytest
import torch

import submission_adapter as adapter
from pa2 import dpo_data, plots, train_dpo
from pa2.batching import IGNORE_INDEX
from pa2.losses import dpo_loss_per_pair, dpo_metrics, pair_logps
from tests.toy_lm import ToyCausalLM, ToyTokenizer

t = torch.tensor


def test_worked_example_and_its_redo():
    # policy chosen -50, rejected -60; reference chosen -52, rejected -58: z = 0.1 * (10 - 6) = 0.4
    loss = adapter.compute_dpo_loss_from_logps(t([-50.0]), t([-60.0]), t([-52.0]), t([-58.0]), beta=0.1)
    redo = adapter.compute_dpo_loss_from_logps(t([-49.0]), t([-61.0]), t([-52.0]), t([-58.0]), beta=0.1)

    torch.testing.assert_close(loss, -torch.log(torch.sigmoid(t(0.4))))
    assert loss.item() == pytest.approx(0.5130, abs=1e-4)
    assert redo.item() == pytest.approx(0.4375, abs=1e-4)
    metrics = dpo_metrics(t([-50.0]), t([-60.0]), t([-52.0]), t([-58.0]), beta=0.1)
    assert metrics["chosen_reward"] == pytest.approx(0.2) and metrics["rejected_reward"] == pytest.approx(-0.2)
    assert metrics["reward_margin"] == pytest.approx(0.4) and metrics["reward_accuracy"] == 1.0


def test_identical_models_give_ln2_zero_accuracy_and_zero_margin():
    logps = t([-12.0, -40.0, -7.5])

    metrics = dpo_metrics(logps, logps - 3, logps, logps - 3, beta=0.1)

    assert metrics["loss"] == pytest.approx(math.log(2))
    assert metrics["reward_accuracy"] == 0.0  # strict r_w > r_l, so ties count as wrong
    assert metrics["reward_margin"] == 0.0 and metrics["max_abs_z"] == 0.0


def test_batch_mean_keeps_row_order():
    pc, pr, rc, rr = t([-5.0, -9.0, -3.0]), t([-6.0, -4.0, -8.0]), t([-5.5, -8.0, -3.0]), t([-5.0, -5.0, -7.0])

    per_pair = dpo_loss_per_pair(pc, pr, rc, rr, beta=0.2)
    permutation = [2, 0, 1]
    permuted = dpo_loss_per_pair(pc[permutation], pr[permutation], rc[permutation], rr[permutation], beta=0.2)

    torch.testing.assert_close(adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta=0.2), per_pair.mean())
    torch.testing.assert_close(permuted, per_pair[permutation])


def test_gradient_raises_chosen_lowers_rejected_and_focuses_on_mistakes():
    def weight(z_value):
        policy_chosen = t([z_value * 10.0], requires_grad=True)  # beta 0.1, so z = policy_chosen / 10
        loss = adapter.compute_dpo_loss_from_logps(policy_chosen, t([0.0]), t([0.0]), t([0.0]), beta=0.1)
        loss.backward()
        return -policy_chosen.grad.item() / 0.1  # = sigmoid(-z)

    assert weight(0.0) == pytest.approx(0.5)
    assert weight(-5.0) / weight(5.0) == pytest.approx(math.exp(5), rel=1e-3)  # about 148 times
    policy_rejected = t([0.0], requires_grad=True)
    adapter.compute_dpo_loss_from_logps(t([0.0]), policy_rejected, t([0.0]), t([0.0]), beta=0.1).backward()
    assert policy_rejected.grad.item() > 0  # gradient descent lowers the rejected log-probability


def test_dpo_batch_shares_the_prompt_and_differs_in_responses():
    tokenizer = ToyTokenizer()
    batch = adapter.build_dpo_batch(tokenizer, ["Q: ", "Task: "], ["yes", "good"], ["no", "bad"], max_length=12)

    for row, prompt in enumerate(["Q: ", "Task: "]):
        n_prompt = len(tokenizer.encode(prompt))
        for side in ("chosen", "rejected"):
            assert batch[side]["labels"][row, :n_prompt].eq(IGNORE_INDEX).all()
            assert torch.equal(batch["chosen"]["input_ids"][row, :n_prompt], batch["rejected"]["input_ids"][row, :n_prompt])
    assert not torch.equal(batch["chosen"]["labels"], batch["rejected"]["labels"])


def test_identical_policy_and_reference_give_ln2_and_only_the_policy_gets_gradients():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.6)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.6)
    args = (tokenizer, ["Q: ", "Task: "], ["yes", "good"], ["no", "bad"], 0.1, 16)

    loss = adapter.compute_dpo_loss(policy, reference, *args)
    loss.backward()

    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert policy.logit_bias.grad is not None and policy.logit_bias.grad.abs().sum() > 0
    assert reference.logit_bias.grad is None


def test_compute_dpo_loss_equals_the_loss_from_separately_scored_logps():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.9)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.2)
    prompts, chosen, rejected = ["Q: ", "Pick: "], ["yes", "a"], ["no", "longer answer"]

    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, 0.3, 24)
    expected = adapter.compute_dpo_loss_from_logps(
        adapter.compute_response_logprobs(policy, tokenizer, prompts, chosen, 24),
        adapter.compute_response_logprobs(policy, tokenizer, prompts, rejected, 24),
        adapter.compute_response_logprobs(reference, tokenizer, prompts, chosen, 24),
        adapter.compute_response_logprobs(reference, tokenizer, prompts, rejected, 24),
        0.3,
    )
    torch.testing.assert_close(loss, expected)


def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except OSError:
        pytest.skip("Qwen tokenizer not available offline")


def tiny_qwen(tokenizer, seed=0):
    import transformers

    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=True,
    )
    return transformers.Qwen2ForCausalLM(config)


def test_stacked_chosen_and_rejected_forward_equals_separate_scoring():
    tokenizer = qwen_tokenizer()
    model = tiny_qwen(tokenizer).eval()
    prompts = ["<|im_start|>user\nIs the sky blue?<|im_end|>\n<|im_start|>assistant\n"]
    batch = adapter.build_dpo_batch(tokenizer, prompts, ["Yes, on a clear day it looks blue."], ["No."], 64)
    narrow = {side: {k: v[:, : int(batch[side]["attention_mask"].sum())] for k, v in batch[side].items()} for side in batch}

    chosen, rejected = pair_logps(model, narrow["chosen"], narrow["rejected"])

    torch.testing.assert_close(chosen, adapter.compute_response_logprobs(model, tokenizer, prompts, ["Yes, on a clear day it looks blue."], 64), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rejected, adapter.compute_response_logprobs(model, tokenizer, prompts, ["No."], 64), rtol=1e-4, atol=1e-4)


def test_hh_parser_keeps_only_clean_single_turn_rows():
    good = ("\n\nHuman: Is it safe?\n\nAssistant: Yes, mostly.", "\n\nHuman: Is it safe?\n\nAssistant: No idea.")
    multi = ("\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: Is it safe?\n\nAssistant: Yes.", good[1])
    other_prompt = (good[0], "\n\nHuman: Something else\n\nAssistant: No idea.")
    same = (good[0], good[0])

    assert dpo_data.parse_hh(*good) == ("Is it safe?", "Yes, mostly.", "No idea.")
    assert dpo_data.parse_hh(*multi) is None
    assert dpo_data.parse_hh(*other_prompt) is None
    assert dpo_data.parse_hh(*same) is None


def test_ultrafeedback_parser_drops_ties_and_multi_turn_rows():
    def row(score_chosen=8.0, score_rejected=3.0, extra_turn=False):
        chosen = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "good"}]
        rejected = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "bad"}]
        if extra_turn:
            chosen = chosen + [{"role": "user", "content": "more"}, {"role": "assistant", "content": "x"}]
        return {"prompt": "Q", "chosen": chosen, "rejected": rejected, "score_chosen": score_chosen, "score_rejected": score_rejected}

    assert dpo_data.parse_ultrafeedback(row()) == ("Q", "good", "bad")
    assert dpo_data.parse_ultrafeedback(row(score_rejected=8.0)) is None
    assert dpo_data.parse_ultrafeedback(row(extra_turn=True)) is None


def test_pair_selection_applies_length_limits_and_exclusions():
    class WordTokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return text.split()

    rows = [("short prompt", "ok", "no"), ("long " * 30, "ok", "no"), ("fits", "ok " * 40, "no"),
            ("an eval prompt", "ok", "no"), ("another prompt", "fine", "bad")]
    stats = Counter()

    kept = dpo_data.select_pairs(rows, WordTokenizer(), count=2, max_length=20, max_prompt_length=12, seed=0,
                                 excluded={"an eval prompt"}, prefix="t", stats=stats, source=lambda r: r)

    assert {pair["prompt"] for pair in kept} == {"short prompt", "another prompt"}
    assert stats["kept"] == 2


def test_tiny_dpo_run_starts_exactly_at_ln2_and_learns_the_preference(tmp_path):
    tokenizer = qwen_tokenizer()
    sft_dir = tmp_path / "sft"
    tiny_qwen(tokenizer).save_pretrained(sft_dir)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    topics = ["the sky", "grass", "snow", "coal", "the sea", "blood", "milk", "the sun"]
    rows = [{"prompt": f"Describe {topic}.", "chosen": f"A short honest description of {topic}.", "rejected": "No."}
            for topic in topics * 5]
    for name, subset in (("train.jsonl", rows), ("val.jsonl", rows[:8])):
        (data_dir / name).write_text("".join(json.dumps(row) + "\n" for row in subset))
    config = {
        "dataset": "toy", "tokenizer": "Qwen/Qwen2.5-0.5B", "seed": 0, "max_length": 64, "max_prompt_length": 32,
        "train_pairs": 40, "val_pairs": 8, "beta": 0.1, "per_device_batch_size": 1, "grad_accum_steps": 4,
        "epochs": 1, "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0, "grad_clip": 1.0,
        "precision": "fp16_autocast", "policy_device": "cuda:0", "reference_device": "cuda:1",
        "eval_every": 5, "eval_batch_pairs": 2, "log_every": 2,
    }
    out_dir = tmp_path / "run"

    summary = train_dpo.train(config, str(sft_dir), data_dir, out_dir)

    assert summary["initial"]["loss"] == pytest.approx(math.log(2), abs=1e-6)
    assert summary["initial"]["reward_accuracy"] == 0.0 and summary["initial"]["max_abs_z"] == 0.0
    assert summary["total_steps"] == 10 and summary["steps_run"] == 10
    assert summary["best"]["reward_accuracy"] > 0.5 and summary["best"]["reward_margin"] > 0
    assert summary["final"]["loss"] < math.log(2)
    assert (out_dir / "best" / "model.safetensors").exists() and (out_dir / "final" / "model.safetensors").exists()
    assert plots.plot_dpo(out_dir / "metrics.jsonl", tmp_path / "dpo.png").stat().st_size > 10_000
