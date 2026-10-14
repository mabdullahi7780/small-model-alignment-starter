"""Tokenize prompt and response pairs into right-padded, response-masked LM batches.

Labels and attention masks are built from positions, never from token ids. Qwen's pad token
is its EOS token, so an id-based mask would also hide the EOS that ends every response.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

IGNORE_INDEX = -100


@dataclass
class TokenizedPair:
    input_ids: list[int]
    prompt_length: int  # prompt tokens kept after truncation
    response_length: int  # response tokens kept after truncation, EOS included when it survives


def encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def tokenize_pair(tokenizer, prompt: str, response: str, max_length: int) -> TokenizedPair:
    """Prompt ids, then response ids, then EOS, cut to `max_length` from the right.

    Prompt and response are tokenized separately so the boundary is exact, and EOS is
    appended before truncation, so a response that does not fit loses its EOS.
    """

    prompt_ids = encode(tokenizer, prompt)
    response_ids = encode(tokenizer, response)
    if tokenizer.eos_token_id is not None:
        response_ids.append(tokenizer.eos_token_id)
    input_ids = (prompt_ids + response_ids)[:max_length]
    prompt_length = min(len(prompt_ids), max_length)
    return TokenizedPair(input_ids, prompt_length, len(input_ids) - prompt_length)


def pad_token_id(tokenizer) -> int:
    """The id written into padded positions; it never matters which, because they are masked."""

    for candidate in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if candidate is not None:
            return candidate
    return 0


def collate(pairs: list[TokenizedPair], pad_id: int, width: int | None = None) -> dict[str, torch.Tensor]:
    """Right-pad to `width` (default: the longest row) and label only the response span."""

    width = max(width or max(len(pair.input_ids) for pair in pairs), 1)
    input_ids = torch.full((len(pairs), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(pairs), width), dtype=torch.long)
    labels = torch.full((len(pairs), width), IGNORE_INDEX, dtype=torch.long)
    for row, pair in enumerate(pairs):
        length = len(pair.input_ids)
        ids = torch.tensor(pair.input_ids, dtype=torch.long)
        input_ids[row, :length] = ids
        attention_mask[row, :length] = 1
        start, end = pair.prompt_length, pair.prompt_length + pair.response_length
        labels[row, start:end] = ids[start:end]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_sft_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True) -> dict[str, torch.Tensor]:
    """input_ids, attention_mask and labels, with -100 on prompt and padding positions.

    The grading interface pads every row to exactly `max_length`; the training loop passes
    `pad_to_max_length=False` and pads only to the longest row, which gives the same loss.
    """

    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    pairs = [tokenize_pair(tokenizer, prompt, response, max_length) for prompt, response in zip(prompts, responses)]
    return collate(pairs, pad_token_id(tokenizer), max_length if pad_to_max_length else None)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to_max_length=True):
    """Two response-masked LM batches over the same prompts: {"chosen": ..., "rejected": ...}.

    Row i of both batches belongs to preference pair i, and the prompt is tokenized the same
    way in both, so the two sequence scores differ only through the responses.
    """

    if not len(prompts) == len(chosen_responses) == len(rejected_responses):
        raise ValueError("prompts, chosen_responses and rejected_responses must have the same length")
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length, pad_to_max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length, pad_to_max_length),
    }
