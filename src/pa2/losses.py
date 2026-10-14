"""Response-only log probabilities, the SFT objective and the DPO objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pa2.batching import IGNORE_INDEX, build_dpo_batch, build_sft_batch


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


def dpo_logits(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """z = beta * [(log pi(y_w) - log pi(y_l)) - (log pi_ref(y_w) - log pi_ref(y_l))], one per pair."""

    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    return beta * (policy_gap - reference_gap)


def dpo_loss_per_pair(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """-log sigmoid(z) per pair; logsigmoid is the numerically stable form."""

    z = dpo_logits(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta)
    return -F.logsigmoid(z)


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """The batch mean of the DPO objective."""

    return dpo_loss_per_pair(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    ).mean()


def dpo_rewards(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """Implicit rewards r = beta * (log pi - log pi_ref) for the chosen and rejected responses."""

    chosen = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected = beta * (policy_rejected_logps - reference_rejected_logps)
    return chosen, rejected


def pair_logps(model, chosen_batch: dict, rejected_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Score the chosen and rejected rows of each pair in one forward pass.

    Both batches are padded to a common width and stacked, so one call returns the
    response log-probabilities of every chosen row and every rejected row.
    """

    width = max(chosen_batch["input_ids"].size(1), rejected_batch["input_ids"].size(1))

    def widen(batch, key, value):
        tensor = batch[key]
        extra = width - tensor.size(1)
        return F.pad(tensor, (0, extra), value=value) if extra else tensor

    # The id written into the extra columns is irrelevant: they get attention 0 and label -100.
    stacked = {
        "input_ids": torch.cat([widen(chosen_batch, "input_ids", 0), widen(rejected_batch, "input_ids", 0)]),
        "attention_mask": torch.cat([widen(chosen_batch, "attention_mask", 0), widen(rejected_batch, "attention_mask", 0)]),
        "labels": torch.cat([widen(chosen_batch, "labels", IGNORE_INDEX), widen(rejected_batch, "labels", IGNORE_INDEX)]),
    }
    logps, _ = batch_response_logprobs(model, stacked)
    count = chosen_batch["input_ids"].size(0)
    return logps[:count], logps[count:]


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length):
    """Scalar mean DPO loss; the reference model is evaluated without building a graph."""

    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    policy_chosen, policy_rejected = pair_logps(policy_model, batch["chosen"], batch["rejected"])
    with torch.no_grad():
        reference_chosen, reference_rejected = pair_logps(reference_model, batch["chosen"], batch["rejected"])
    device = policy_chosen.device
    return compute_dpo_loss_from_logps(
        policy_chosen, policy_rejected, reference_chosen.to(device), reference_rejected.to(device), beta
    )


def dpo_metrics(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta) -> dict:
    """Loss, reward accuracy (strict r_w > r_l, so ties count as wrong), margin and both mean rewards."""

    args = (policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta)
    z = dpo_logits(*args)
    chosen, rejected = dpo_rewards(*args)
    return {
        "loss": float((-F.logsigmoid(z)).mean()),
        "reward_accuracy": float((chosen > rejected).float().mean()),
        "reward_margin": float((chosen - rejected).mean()),
        "chosen_reward": float(chosen.mean()),
        "rejected_reward": float(rejected.mean()),
        "max_abs_z": float(z.abs().max()),
    }
