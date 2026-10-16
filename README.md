# Small-Model Alignment: SFT and DPO on Qwen2.5-0.5B

The CS 5326 Programming Assignment 2 solution. `Qwen/Qwen2.5-0.5B` is fine-tuned on 30,000 instruction-response pairs (SFT). Two from-scratch DPO runs follow, from the SFT checkpoint: one on HH-RLHF pairs and one on UltraFeedback pairs. Separately, a LoRA adapter is trained on `HuggingFaceTB/SmolLM2-360M-Instruct` with TRL's `DPOTrainer` on the provided persona pairs. All models are evaluated with blind pairwise judgments from `google/gemini-3.6-flash` through Kaggle's `kaggle_benchmarks`.

The report is [`REPORT.pdf`](REPORT.pdf); its LaTeX source is [`REPORT.tex`](REPORT.tex) and its figures are in `report_assets/`.

## Results at a glance

| Comparison | Prompts | Wins / ties / losses | Win rate (ties as half) | Win rate (excluding ties) |
|---|---|---|---|---|
| HH-DPO vs SFT | 300 items, 185 distinct (HH safety set) | 179 / 35 / 86 | 65.5% | 67.5% |
| UltraFeedback-DPO vs SFT | 300 (AlpacaEval, seed 42) | 108 / 59 / 133 | 45.8% | 44.8% |
| LoRA-DPO vs SmolLM2, persona style | 40 | 24 / 16 / 0 | 80.0% | 100% |
| LoRA-DPO vs SmolLM2, usefulness and safety | 40 | 13 / 2 / 25 | 35.0% | 34.2% |

## Layout

```text
submission_adapter.py        grading interface: six thin wrappers around src/pa2
src/pa2/
  formatting.py              the shared prompt format and stop tokens
  batching.py                SFT and DPO batches: separate tokenization, EOS before truncation, position-based masks
  losses.py                  response log-probabilities, SFT loss (token mean), DPO loss, rewards and metrics
  sft_data.py, train_sft.py  UltraChat preparation and the hand-written SFT loop
  dpo_data.py, train_dpo.py  HH-RLHF / UltraFeedback preparation and the hand-written two-GPU DPO loop
  lora_persona.py            TRL DPOTrainer + PEFT LoRA on SmolLM2 (the only use of TRL)
  alpaca_subset.py           the seed-42 AlpacaEval subset
  generate.py                batched greedy generation (left padding, optional chat template, PEFT LoRA adapter)
  pairs.py, judge.py         blind A/B pair files and the resumable batched judge
  env_report.py, plots.py    environment report and training curves
scripts/judge_pairwise_kaggle.py   judge entry point with the manual's command-line flags
configs/                     sft.yaml, dpo_hh.yaml, dpo_ultrafeedback.yaml, lora_persona.yaml
eval_sets/                   hh_rlhf_eval_300_seed42.json (provided), alpaca_eval_300_seed42.json, persona_style_eval_40.json
notebooks/                   01_setup to 07_judge: one Kaggle notebook per task
results/                     metrics, summaries, generations, pair files and judgments from the Kaggle runs
report_assets/               training-curve figures used in the report
REPORT.pdf, REPORT.tex       the report and its LaTeX source
tests/                       public adapter tests plus tests for every module
```

## Reproducing the runs on Kaggle

Notebooks 01 to 06 clone this repository, keep Kaggle's preinstalled PyTorch (pinned with a pip constraint while installing `requirements.txt`), and install the package with `pip install -e . --no-deps`; 07 clones the repository and imports it directly. Run them in order, attaching each earlier notebook's saved output where listed:

| Notebook | Task | Accelerator | Attach as input | Time |
|---|---|---|---|---|
| `01_setup.ipynb` | 1: environment check | GPU T4 x2 | none | 10 min |
| `02_baseline.ipynb` | 2: base model on AlpacaEval-300 | GPU T4 x2 | none | 15 min |
| `03_sft.ipynb` | 3: SFT | GPU T4 x2 | none | 1.5 h |
| `04_dpo.ipynb` | 4: HH and UltraFeedback DPO | GPU T4 x2 | `03_sft` output | 2.6 h |
| `05_lora_persona.ipynb` | 5: LoRA persona DPO | GPU T4 x2 | none | 10 min |
| `06_generate.ipynb` | 6: evaluation outputs and pair files | GPU T4 x2 | outputs of 03, 04, 05 | 20 min |
| `07_judge.ipynb` | 7: judging | none, in a benchmark notebook (kaggle.com/benchmarks/tasks/new) | `06_generate` output | 30 min |

Each notebook writes durable files under `/kaggle/working/outputs/` and a small zip of its results. Checkpoints stay in the notebook output for the next notebook to attach.

## Tests

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The full suite (`PYTHONPATH=src:. python -m pytest -q tests/`) covers the edge cases the grader describes: truncation at the limit, a pad token equal to EOS, a tokenizer without a pad token, mixed lengths, row order, and gradient flow. It also covers DPO's exact ln 2 start, the data parsers, tiny end-to-end SFT, DPO and LoRA runs, generation, pair building and the judge. The SFT reduction is the mean over all unmasked response tokens.

## Fixed files

- `eval_sets/hh_rlhf_eval_300_seed42.json`: the provided synthetic helpful/harmless evaluation set.
- `eval_sets/alpaca_eval_300_seed42.json`: 300 of the 805 AlpacaEval prompts, chosen with seed 42; `alpaca_eval_300_seed42.meta.json` records the source revision and the selection rule.
- `eval_sets/persona_style_eval_40.json`: 40 persona evaluation prompts (cooking, coding, general help, safety-sensitive).
- `data/persona_lora/ramsay_style_preferences.json`: the provided persona preference pairs.
- `docs/assignment_manual.pdf`: the assignment manual.
