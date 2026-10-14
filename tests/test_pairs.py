"""Blind A/B pair files (Task 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pa2 import pairs

REPO_ROOT = Path(__file__).resolve().parents[1]


def answers(prefix, count=300):
    return [{"id": f"p{i:03d}", "instruction": f"question {i}", "output": f"{prefix} answer {i}", "dataset": "toy"}
            for i in range(count)]


def test_positions_are_seeded_balanced_and_the_hidden_mapping_round_trips():
    dpo, sft = answers("from-x"), answers("from-y")

    built = pairs.build_pairs(dpo, sft, "dpo_hh", "sft", "dpo_hh_vs_sft")
    again = pairs.build_pairs(dpo, sft, "dpo_hh", "sft", "dpo_hh_vs_sft")
    other = pairs.build_pairs(dpo, sft, "dpo_hh", "sft", "another_comparison")

    assert built == again and built != other
    assert 0.4 < pairs.balance(built, "dpo_hh") < 0.6
    lookup = {"dpo_hh": {r["id"]: r["output"] for r in dpo}, "sft": {r["id"]: r["output"] for r in sft}}
    for pair in built:
        assert pair["response_a"] == lookup[pair["hidden"]["a"]][pair["pair_id"]]
        assert pair["response_b"] == lookup[pair["hidden"]["b"]][pair["pair_id"]]
        assert {pair["hidden"]["a"], pair["hidden"]["b"]} == {"dpo_hh", "sft"}


def test_the_judge_view_never_contains_model_identities():
    built = pairs.build_pairs(answers("x", 20), answers("y", 20), "dpo_ultrafeedback", "sft", "uf")

    for pair in built:
        view = pairs.judge_view(pair)
        assert set(view) <= set(pairs.JUDGE_FIELDS)
        text = json.dumps(view)
        assert "dpo_ultrafeedback" not in text and '"sft"' not in text and "hidden" not in text


def test_mismatched_prompt_files_are_rejected():
    with pytest.raises(ValueError):
        pairs.build_pairs(answers("x", 5), answers("y", 5)[::-1], "a", "b", "c")


def test_categories_and_expected_behaviour_come_from_the_prompt_file(tmp_path):
    prompt_rows = json.loads((REPO_ROOT / "eval_sets" / "hh_rlhf_eval_300_seed42.json").read_text())[:4]
    x = [{"id": r["id"], "instruction": r["instruction"], "output": "x", "dataset": r["dataset"]} for r in prompt_rows]
    y = [{**r, "output": "y"} for r in x]
    (tmp_path / "x.json").write_text(json.dumps(x))
    (tmp_path / "y.json").write_text(json.dumps(y))
    (tmp_path / "prompts.json").write_text(json.dumps(prompt_rows))
    out = tmp_path / "pairs.jsonl"

    assert pairs.main(["--x", str(tmp_path / "x.json"), "--name-x", "dpo_hh", "--y", str(tmp_path / "y.json"),
                       "--name-y", "sft", "--prompts", str(tmp_path / "prompts.json"),
                       "--comparison", "dpo_hh_vs_sft", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 4
    assert rows[0]["category"] == prompt_rows[0]["category"]
    assert rows[0]["expected_behavior"] == prompt_rows[0]["expected_behavior"]
    assert "expected_behavior" in pairs.judge_view(rows[0])
