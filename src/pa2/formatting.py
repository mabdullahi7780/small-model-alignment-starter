"""The one prompt format shared by the base model, SFT, both DPO runs and evaluation.

Qwen's chat markers are single special tokens, so user text can never imitate the
boundary between the prompt and the response. No system prompt is used.
"""

from __future__ import annotations

USER_START = "<|im_start|>user\n"
TURN_END = "<|im_end|>\n"
ASSISTANT_START = "<|im_start|>assistant\n"

# Generation stops on either token: <|endoftext|> is the tokenizer EOS that SFT appends
# to every response, and <|im_end|> is the turn marker a base model may emit on its own.
STOP_TOKENS = ("<|endoftext|>", "<|im_end|>")


def format_prompt(instruction: str) -> str:
    """Wrap one instruction so the model's next token starts the assistant response."""

    return f"{USER_START}{instruction.strip()}{TURN_END}{ASSISTANT_START}"


def stop_token_ids(tokenizer) -> list[int]:
    """Ids of the stop tokens this tokenizer knows, plus its EOS, without duplicates."""

    ids = []
    for token in STOP_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            ids.append(token_id)
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return list(dict.fromkeys(ids))
