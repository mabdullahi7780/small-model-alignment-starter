"""Record the runtime facts Task 1 asks for (GPUs, package versions, commit) as JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "trl",
    "peft",
    "huggingface_hub",
    "safetensors",
    "bitsandbytes",
)


def package_versions(names=PACKAGES) -> dict[str, str | None]:
    """Installed version of each package, or None when it is missing."""

    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_inventory() -> list[dict]:
    """One entry per visible CUDA device, as PyTorch sees it."""

    import torch

    if not torch.cuda.is_available():
        return []
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gib": round(props.total_memory / 2**30, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                # Native bf16 tensor cores start at compute capability 8.0 (Ampere).
                # A T4 is 7.5, so training uses fp16 autocast with loss scaling.
                "native_bf16": props.major >= 8,
            }
        )
    return gpus


def git_commit(repo_dir: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_report(repo_dir: str | Path = ".", torch_before: str | None = None) -> dict:
    import torch

    gpus = gpu_inventory()
    versions = package_versions()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(repo_dir),
        "torch_preinstalled": torch_before,
        "torch_unchanged": None if torch_before is None else torch_before == torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "packages": versions,
    }


def check_report(report: dict, expect_gpus: int) -> list[str]:
    """Problems that should stop the setup notebook; empty when everything is as required."""

    problems = []
    if expect_gpus and report["gpu_count"] != expect_gpus:
        problems.append(f"expected {expect_gpus} GPUs, found {report['gpu_count']}")
    if expect_gpus and not all("T4" in gpu["name"] for gpu in report["gpus"]):
        problems.append("expected Tesla T4 GPUs: " + ", ".join(gpu["name"] for gpu in report["gpus"]))
    if report["torch_unchanged"] is False:
        problems.append(
            f"torch changed from {report['torch_preinstalled']} to {report['packages']['torch']}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="path of the JSON report to write")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--torch-before", default=None, help="torch version before installing requirements")
    parser.add_argument("--expect-gpus", type=int, default=0, help="required GPU count; 0 skips the check")
    args = parser.parse_args(argv)

    report = build_report(args.repo_dir, args.torch_before)
    report["problems"] = check_report(report, args.expect_gpus)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    for problem in report["problems"]:
        print(f"SETUP CHECK FAILED: {problem}", file=sys.stderr)
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
