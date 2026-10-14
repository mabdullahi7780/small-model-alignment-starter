"""Entry point named in the manual (section 13.2); the implementation is src/pa2/judge.py.

    python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/dpo_vs_sft_pairs.jsonl \
        --output_path outputs/eval/dpo_vs_sft_judged.jsonl --judge_model google/gemini-3.6-flash \
        --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pa2.judge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
