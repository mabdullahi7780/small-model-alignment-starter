"""Checks for the Task 1 environment report."""

from __future__ import annotations

import json

import torch

from pa2 import env_report


def test_package_versions_reports_installed_and_missing_packages():
    versions = env_report.package_versions(("torch", "a-package-that-does-not-exist"))

    assert versions["torch"] is not None
    assert torch.__version__.startswith(versions["torch"].split("+")[0])
    assert versions["a-package-that-does-not-exist"] is None


def test_report_flags_missing_gpus_and_a_reinstalled_torch():
    report = env_report.build_report(torch_before="0.0.0")

    problems = env_report.check_report(report, expect_gpus=2)

    assert report["torch_unchanged"] is False
    assert any("torch changed" in problem for problem in problems)
    if not torch.cuda.is_available():
        assert any("expected 2 GPUs" in problem for problem in problems)


def test_main_writes_json_and_passes_without_expectations(tmp_path):
    out = tmp_path / "setup" / "report.json"

    exit_code = env_report.main(["--out", str(out), "--torch-before", torch.__version__])

    report = json.loads(out.read_text())
    assert exit_code == 0
    assert report["torch_unchanged"] is True
    assert report["problems"] == []
    assert set(env_report.PACKAGES) <= set(report["packages"])
