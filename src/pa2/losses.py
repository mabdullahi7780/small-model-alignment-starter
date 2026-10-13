"""Response-only log probabilities and the SFT objective (DPO objectives are added in Task 4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pa2.batching import IGNORE_INDEX, build_sft_batch


def response_logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: the summed log probability of the labelled tokens, and how many there are.

    The output at position t predicts the token at t + 1, so the last logit and the first
    label are dropped before comparing. Unlabelled positions (-100) contribute nothing.
    """

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    token_nll = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shifted_labels)
    counts = shifted_labels.ne(IGNORE_INDEX).sum(dim=-1)
    return -token_nll.sum(dim=-1), counts


def batch_response_logprobs(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    logits = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    ).logits
    return response_logprobs_from_logits(logits, batch["labels"].to(device))


def token_mean_nll(logps: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """The SFT reduction used throughout: total response NLL over total response tokens."""

    return -logps.sum() / counts.sum().clamp_min(1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    """log pi(response | prompt) summed over response tokens, one value per input row."""

    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logps, _ = batch_response_logprobs(model, batch)
    return logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logps, counts = batch_response_logprobs(model, batch)
    return token_mean_nll(logps, counts)
