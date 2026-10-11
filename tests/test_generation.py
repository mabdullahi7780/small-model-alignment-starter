"""Checks for the prompt format, the AlpacaEval subset and batched generation (Task 2)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from pa2 import alpaca_subset, generate
from pa2.formatting import format_prompt, stop_token_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSET_FILE = REPO_ROOT / "eval_sets" / "alpaca_eval_300_seed42.json"


def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except OSError:
        pytest.skip("Qwen tokenizer not available offline")


def tiny_qwen(tokenizer):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config).eval()


def test_format_prompt_is_the_exact_chat_layout():
    assert format_prompt("  What is a ball?\n") == (
        "<|im_start|>user\nWhat is a ball?<|im_end|>\n<|im_start|>assistant\n"
    )


def test_qwen_stop_tokens_are_endoftext_and_im_end():
    tokenizer = qwen_tokenizer()

    assert stop_token_ids(tokenizer) == [151643, 151645]
    # Qwen pads with its EOS token, which is why masks are built from positions, not ids.
    assert tokenizer.pad_token_id == tokenizer.eos_token_id == 151643


def test_subset_selection_is_seeded_sorted_and_unique():
    rows = [{"instruction": f"prompt {index}", "dataset": f"source{index % 3}"} for index in range(805)]

    first = alpaca_subset.select_subset(rows, size=300, seed=42)
    second = alpaca_subset.select_subset(rows, size=300, seed=42)
    other_seed = alpaca_subset.select_subset(rows, size=300, seed=7)

    assert first == second
    assert first != other_seed
    assert [row["source_index"] for row in first] == sorted(row["source_index"] for row in first)
    assert len({row["source_index"] for row in first}) == 300
    assert first[0]["id"] == "alpaca-eval-42-0000" and first[-1]["id"] == "alpaca-eval-42-0299"


def test_committed_subset_matches_the_seed_42_rule():
    rows = json.loads(SUBSET_FILE.read_text())

    expected_indices = sorted(random.Random(42).sample(range(805), 300))
    assert [row["source_index"] for row in rows] == expected_indices
    assert len({row["instruction"] for row in rows}) == 300
    assert all(row["dataset"] == "alpaca_eval" for row in rows)


def test_cut_at_stop_reports_why_generation_ended():
    stop_ids, eos_id = [151643, 151645], 151643

    assert generate.cut_at_stop([5, 6, 151643, 7], stop_ids, eos_id) == ([5, 6], "eos")
    assert generate.cut_at_stop([5, 151645, 151643], stop_ids, eos_id) == ([5], "turn_end")
    assert generate.cut_at_stop([5, 6, 7], stop_ids, eos_id) == ([5, 6, 7], "max_new_tokens")


def test_repetition_loop_detector():
    assert generate.has_repetition_loop("the cat sat on the mat and " * 4)
    assert not generate.has_repetition_loop("one two three four five six seven eight nine ten eleven twelve")


def test_left_padded_batch_matches_one_prompt_at_a_time():
    tokenizer = qwen_tokenizer()
    model = tiny_qwen(tokenizer)
    instructions = ["What is a ball?", "Explain why the sky looks blue during the day but red at sunset."]

    batched = generate.generate_responses(model, tokenizer, instructions, batch_size=2, max_new_tokens=6)
    single = [
        generate.generate_responses(model, tokenizer, [instruction], batch_size=1, max_new_tokens=6)[0]
        for instruction in instructions
    ]

    assert [row["output"] for row in batched] == [row["output"] for row in single]
    assert [row["new_tokens"] for row in batched] == [row["new_tokens"] for row in single]


def test_cli_writes_records_in_prompt_order_with_required_fields(tmp_path):
    tokenizer = qwen_tokenizer()
    model_dir = tmp_path / "tiny"
    tiny_qwen(tokenizer).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    prompts = [
        {"id": f"p{index}", "instruction": text, "dataset": "toy"}
        for index, text in enumerate(["Hi", "Write a long sentence about the sea and its tides.", "Why?"])
    ]
    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(json.dumps(prompts))
    out = tmp_path / "out" / "toy.json"

    exit_code = generate.main(
        [
            "--model", str(model_dir), "--generator", "tiny", "--prompts", str(prompts_file),
            "--out", str(out), "--device", "cpu", "--batch-size", "2", "--max-new-tokens", "4",
        ]
    )

    records = json.loads(out.read_text())
    meta = json.loads(out.with_name("toy.meta.json").read_text())
    assert exit_code == 0
    assert [record["id"] for record in records] == ["p0", "p1", "p2"]
    for record in records:
        assert {"id", "instruction", "output", "generator", "dataset"} <= set(record)
        assert record["generator"] == "tiny" and record["dataset"] == "toy"
        assert record["new_tokens"] <= 4
    assert meta["decoding"]["strategy"] == "greedy"
    assert meta["stats"]["count"] == 3
