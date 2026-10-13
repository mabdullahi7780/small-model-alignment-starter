"""Edge cases for SFT batching, response log-probabilities and the SFT loss (Task 3)."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import torch

import submission_adapter as adapter
from pa2.batching import IGNORE_INDEX, build_sft_batch
from tests.toy_lm import ToyCausalLM, ToyTokenizer

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "pa2"


class PadIsEosTokenizer(ToyTokenizer):
    """Like Qwen: the pad token and the EOS token share one id."""

    pad_token_id = 1


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


def lengths(tokenizer, prompt, response):
    return len(tokenizer.encode(prompt)), len(tokenizer.encode(response))


def manual_logprob(model, tokenizer, prompt, response):
    """The toy model's logits ignore position, so log p(token) is a fixed lookup."""

    log_probs = torch.log_softmax(model.logit_bias.detach().float(), dim=-1)
    target = tokenizer.encode(response) + [tokenizer.eos_token_id]
    return float(sum(log_probs[token] for token in target))


def test_rows_are_padded_to_max_length_with_eos_labelled_and_pads_ignored():
    tokenizer = ToyTokenizer()
    batch = build_sft_batch(tokenizer, ["Task: ", "Q: "], ["ok", "yes"], max_length=12)

    assert batch["input_ids"].shape == (2, 12)
    for row, (prompt, response) in enumerate([("Task: ", "ok"), ("Q: ", "yes")]):
        n_prompt, n_response = lengths(tokenizer, prompt, response)
        end = n_prompt + n_response + 1
        assert batch["labels"][row, :n_prompt].eq(IGNORE_INDEX).all()
        assert batch["labels"][row, end - 1] == tokenizer.eos_token_id
        assert batch["labels"][row, end:].eq(IGNORE_INDEX).all()
        assert batch["attention_mask"][row, :end].eq(1).all() and batch["attention_mask"][row, end:].eq(0).all()


def test_pad_equal_to_eos_keeps_the_real_eos_label():
    tokenizer = PadIsEosTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["Q: "], ["yes"], max_length=10)

    n_prompt, n_response = lengths(tokenizer, "Q: ", "yes")
    eos_position = n_prompt + n_response
    assert batch["labels"][0, eos_position] == tokenizer.eos_token_id
    assert batch["labels"][0, eos_position + 1 :].eq(IGNORE_INDEX).all()
    assert batch["input_ids"][0, eos_position + 1 :].eq(tokenizer.eos_token_id).all()  # pads share the id


def test_tokenizer_without_pad_token_still_masks_padding():
    tokenizer = NoPadTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["Q: ", "Task: "], ["a", "bb"], max_length=10)

    assert batch["attention_mask"][0].sum() == len(tokenizer.encode("Q: a")) + 1
    assert batch["labels"][0, batch["attention_mask"][0].eq(0)].eq(IGNORE_INDEX).all()


def test_tokenizer_without_eos_appends_nothing():
    tokenizer = NoEosTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["Q: "], ["yes"], max_length=10)

    assert batch["labels"][0].ne(IGNORE_INDEX).sum() == len(tokenizer.encode("yes"))


@pytest.mark.parametrize("extra", [0, 1, 3])
def test_truncation_boundary_and_eos_loss(extra):
    tokenizer = ToyTokenizer()
    prompt, response = "Q: ", "abcd"
    n_prompt, n_response = lengths(tokenizer, prompt, response)
    full = n_prompt + n_response + 1
    max_length = full - extra

    batch = adapter.build_sft_batch(tokenizer, [prompt], [response], max_length)

    labelled = batch["labels"][0].ne(IGNORE_INDEX)
    assert batch["input_ids"].shape == (1, max_length)
    assert labelled.sum() == n_response + 1 - extra
    has_eos = bool(batch["labels"][0, labelled][-1] == tokenizer.eos_token_id)
    assert has_eos == (extra == 0)


def test_prompt_longer_than_max_length_leaves_no_response_and_finite_loss():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.3)
    prompt = "a long prompt that does not fit"

    batch = adapter.build_sft_batch(tokenizer, [prompt], ["yes"], max_length=8)
    logps = adapter.compute_response_logprobs(model, tokenizer, [prompt], ["yes"], max_length=8)
    loss = adapter.compute_sft_loss(model, tokenizer, [prompt], ["yes"], max_length=8)

    assert batch["labels"].eq(IGNORE_INDEX).all()
    assert logps.item() == 0.0
    assert torch.isfinite(loss)


def test_logprobs_match_a_hand_computation_and_keep_row_order():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.8)
    prompts = ["Task: ", "Q: ", "Longer prompt here: "]
    responses = ["ok", "a much longer answer", "z"]

    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=40)
    reversed_logps = adapter.compute_response_logprobs(model, tokenizer, prompts[::-1], responses[::-1], max_length=40)

    expected = [manual_logprob(model, tokenizer, p, r) for p, r in zip(prompts, responses)]
    torch.testing.assert_close(logps, torch.tensor(expected), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reversed_logps, logps.flip(0))


def test_sft_loss_is_the_token_mean_and_differs_from_the_example_mean():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    prompts, responses = ["Q: ", "Q: "], ["a", "a much longer answer"]

    loss = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=40)
    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=40)
    tokens = sum(len(tokenizer.encode(r)) + 1 for r in responses)

    torch.testing.assert_close(loss, -logps.sum() / tokens)
    assert not math.isclose(loss.item(), -logps.mean().item(), rel_tol=1e-3)


def test_right_padding_does_not_change_a_row_in_a_real_attention_model():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except OSError:
        pytest.skip("Qwen tokenizer not available offline")
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256, tie_word_embeddings=True,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    prompts = ["<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", "<|im_start|>user\nExplain tides in detail<|im_end|>\n<|im_start|>assistant\n"]
    responses = ["Hello.", "Tides come from the moon's pull on the oceans, which raises a bulge of water."]

    together = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=96)
    alone = torch.stack([adapter.compute_response_logprobs(model, tokenizer, [p], [r], max_length=96)[0] for p, r in zip(prompts, responses)])

    torch.testing.assert_close(together, alone, rtol=1e-4, atol=1e-4)


def test_hand_written_training_code_uses_no_prebuilt_trainer():
    banned = re.compile(r"^\s*(from|import)\s+trl\b|\bSFTTrainer\b|\bDPOTrainer\b|transformers\s+import\s+.*\bTrainer\b|\bTrainer\(", re.M)
    allowed = {"lora_persona.py"}  # Task 5 is required to use TRL
    for path in SRC_DIR.glob("*.py"):
        if path.name not in allowed:
            assert not banned.search(path.read_text()), f"{path.name} must not use a prebuilt trainer"
