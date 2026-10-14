"""The pairwise judge (Task 7), exercised with fake judges: no network or Kaggle access needed."""

from __future__ import annotations

import json
import re

import pytest

from pa2 import judge, pairs


def make_pairs(count=12, comparison="dpo_hh_vs_sft", evaluated="dpo_hh", baseline="sft", with_safety=True):
    answers_x = [{"id": f"hh-{i:03d}", "instruction": f"question {i}", "output": f"EVALUATED answer {i} " + "word " * (i % 3), "dataset": "toy"} for i in range(count)]
    answers_y = [{"id": f"hh-{i:03d}", "instruction": f"question {i}", "output": f"BASELINE answer {i}", "dataset": "toy"} for i in range(count)]
    prompt_rows = [{"id": f"hh-{i:03d}", "category": "cyber_abuse" if i % 2 else "benign_helpfulness",
                    "expected_behavior": "Refuse cyber abuse while offering defensive alternatives."} for i in range(count)] if with_safety else []
    return pairs.build_pairs(answers_x, answers_y, evaluated, baseline, comparison, prompt_rows=prompt_rows)


class PreferMarker:
    """Picks whichever response contains `marker`; records every prompt it receives."""

    def __init__(self, marker="EVALUATED"):
        self.marker, self.prompts = marker, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        items = re.split(r"### Item \d+: pair_id = ", prompt)[1:]
        verdicts = []
        for item in items:
            pair_id = item.split("\n", 1)[0].strip()
            response_a = item.split("Response A:", 1)[1].split("Response B:", 1)[0]
            verdicts.append({"pair_id": pair_id, "winner": "A" if self.marker in response_a else "B", "reason": "fake"})
        return "Here you go:\n```json\n" + json.dumps(verdicts) + "\n```"


def test_prompt_shows_rubric_ids_and_expected_behaviour_but_never_model_names():
    batch = make_pairs(5)

    prompt = judge.build_prompt(batch, "quality")

    for pair in batch:
        assert f"pair_id = {pair['pair_id']}" in prompt
    assert "helpfulness" in prompt and "over-refusal" in prompt and "Expected behaviour:" in prompt
    assert "dpo_hh" not in prompt and "sft" not in prompt.lower().replace("soft", "") and "hidden" not in prompt
    style = judge.build_prompt(make_pairs(2, "lora_vs_base_persona", "smollm2_lora", "smollm2_base", False), "persona_style")
    assert "Gordon Ramsay" in style and "smollm2" not in style


@pytest.mark.parametrize("reply, expected", [
    ('[{"pair_id": "x1", "winner": "A"}, {"pair_id": "x2", "winner": "tie"}]', {"x1": "A", "x2": "tie"}),
    ('```json\n[{"pair_id": "x1", "winner": "Response B"}, {"pair_id": "x2", "winner": "b"}]\n```', {"x1": "B", "x2": "B"}),
    ('[{"pair_id": "garbled", "winner": "A"}, {"pair_id": "x2", "winner": "B"}]', {"x1": "A", "x2": "B"}),
])
def test_reply_parsing_is_tolerant_of_format(reply, expected):
    verdicts = judge.parse_verdicts(reply, ["x1", "x2"])

    assert {k: v["winner"] for k, v in verdicts.items()} == expected


@pytest.mark.parametrize("reply", ["no json here", '[{"pair_id": "x1", "winner": "A"}]', '[{"pair_id": "x1", "winner": "maybe"}, {"pair_id": "x2", "winner": "A"}]'])
def test_incomplete_or_invalid_replies_raise(reply):
    with pytest.raises(judge.JudgeParseError):
        judge.parse_verdicts(reply, ["x1", "x2"])


def test_batches_of_five_sleep_before_every_request_and_map_verdicts_back_to_models(tmp_path):
    batch = make_pairs(12)
    fake, sleeps = PreferMarker(), []

    counts = judge.run_judging(batch, tmp_path / "j.jsonl", tmp_path / "f.jsonl", fake, "fake-judge",
                               batch_size=5, sleep_seconds=5, sleep=sleeps.append, log=lambda _: None)

    rows = judge.read_jsonl(tmp_path / "j.jsonl")
    assert counts["requests"] == 3 and len(fake.prompts) == 3 and sleeps == [5, 5, 5]
    assert len(rows) == 12 and all(row["winner_model"] == "dpo_hh" for row in rows)
    summary = judge.summarize(rows, "quality", 12)
    assert summary["wins"] == 12 and summary["losses"] == 0 and summary["win_rate_without_ties"] == 1.0
    assert 0 < summary["position_a_share"] < 1  # the evaluated model sat in both positions


def test_failed_attempts_are_retried_then_logged_and_resume_skips_finished_pairs(tmp_path):
    batch = make_pairs(10)
    calls = {"n": 0}
    good = PreferMarker()

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] in (1, 2):
            raise RuntimeError("request denied")
        return good(prompt)

    sleeps = []
    judge.run_judging(batch, tmp_path / "j.jsonl", tmp_path / "f.jsonl", flaky, "fake", batch_size=5,
                      sleep_seconds=5, max_retries=3, sleep=sleeps.append, log=lambda _: None)
    rows = judge.read_jsonl(tmp_path / "j.jsonl")
    assert calls["n"] == 4 and len(sleeps) == 4  # 3 attempts for batch 1, 1 for batch 2
    assert [row["attempts"] for row in rows[:5]] == [3] * 5 and not (tmp_path / "f.jsonl").exists()

    def broken(prompt):
        raise RuntimeError("proxy down")

    more = make_pairs(13)  # 3 new pairs on top of the 10 already judged
    counts = judge.run_judging(more, tmp_path / "j.jsonl", tmp_path / "f.jsonl", broken, "fake", batch_size=5,
                               sleep_seconds=5, max_retries=3, sleep=lambda _: None, log=lambda _: None)
    assert counts["skipped"] == 10 and counts["requests"] == 3 and counts["failed_batches"] == 1
    failure = judge.read_jsonl(tmp_path / "f.jsonl")[0]
    assert failure["pair_ids"] == ["hh-010", "hh-011", "hh-012"] and failure["attempts"] == 3
    summary = judge.summarize(judge.read_jsonl(tmp_path / "j.jsonl"), "quality", 13, failures=1)
    assert summary["unknown"] == 3 and summary["judged_valid"] == 10 and summary["failed_batches"] == 1

    counts = judge.run_judging(more, tmp_path / "j.jsonl", tmp_path / "f.jsonl", good, "fake", batch_size=5,
                               sleep_seconds=5, sleep=lambda _: None, log=lambda _: None)
    assert counts["skipped"] == 10 and counts["requests"] == 1  # only the unknowns are judged again
    assert judge.summarize(judge.read_jsonl(tmp_path / "j.jsonl"), "quality", 13)["unknown"] == 0


def test_kaggle_judge_falls_back_when_temperature_is_rejected(monkeypatch):
    import contextlib
    import sys
    import types

    calls = []

    class FakeLLM:
        def prompt(self, text, **options):
            calls.append(options)
            if "temperature" in options:
                raise ValueError("unsupported option")
            return "ok"

    fake = types.SimpleNamespace(llms={"m": FakeLLM()}, chats=types.SimpleNamespace(new=lambda name: contextlib.nullcontext()))
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", fake)

    client = judge.KaggleBenchmarksJudge("m")

    assert client("hi") == "ok" and client("again") == "ok"
    assert calls == [{"temperature": 0.0}, {}, {}]


def test_summary_rates_categories_and_length_split():
    def row(pair_id, verdict, model_a, words_a, words_b, category):
        winner = {"A": model_a, "B": "sft" if model_a == "dpo" else "dpo", "tie": "tie"}[verdict]
        return {"pair_id": pair_id, "comparison": "c", "criterion": "quality", "category": category, "judge_model": "j",
                "verdict": verdict, "winner_model": winner, "model_a": model_a, "model_b": "sft" if model_a == "dpo" else "dpo",
                "words_a": words_a, "words_b": words_b, "reason": "", "attempts": 1}

    rows = [row("1", "A", "dpo", 200, 100, "x"), row("2", "B", "sft", 100, 200, "x"),   # dpo wins twice, longer both times
            row("3", "A", "sft", 100, 90, "y"), row("4", "tie", "dpo", 80, 100, "y")]   # dpo loses once, ties once, shorter
    summary = judge.summarize(rows, "quality", 4)

    assert (summary["evaluated"], summary["baseline"]) == ("dpo", "sft")
    assert (summary["wins"], summary["losses"], summary["ties"]) == (2, 1, 1)
    assert summary["win_rate_without_ties"] == pytest.approx(2 / 3)
    assert summary["win_rate_ties_as_half"] == pytest.approx(2.5 / 4)
    assert summary["length"]["evaluated_longer"]["wins"] == 2 and summary["length"]["evaluated_not_longer"]["losses"] == 1
    assert summary["by_category"]["x"]["wins"] == 2 and summary["by_category"]["y"]["ties"] == 1


def test_cli_follows_the_manual_flags_and_refuses_short_sleeps(tmp_path):
    pairs.write_jsonl(tmp_path / "pairs.jsonl", make_pairs(6))
    args = ["--pairs_path", str(tmp_path / "pairs.jsonl"), "--output_path", str(tmp_path / "out" / "judged.jsonl"),
            "--judge_model", "google/gemini-3.6-flash", "--batch_size", "5", "--sleep_seconds", "5",
            "--max_retries", "3", "--reuse_judgments"]
    judge_fn = PreferMarker()
    original_sleep = judge.time.sleep
    judge.time.sleep = lambda _: None
    try:
        assert judge.main(args, judge_factory=lambda model: judge_fn) == 0
    finally:
        judge.time.sleep = original_sleep
    assert len(judge_fn.prompts) == 2
    summary = json.loads((tmp_path / "out" / "judged_summary.json").read_text())
    assert summary["judge_model"] == "google/gemini-3.6-flash" and summary["wins"] == 6
    with pytest.raises(SystemExit):
        judge.main([*args[:-4], "--sleep_seconds", "1"], judge_factory=lambda model: judge_fn)
