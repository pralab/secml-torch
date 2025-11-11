"""Mock classes for testing language models."""

from types import SimpleNamespace
from typing import Any

import torch


class MockHFTokenizer:
    """Fake tokenizer replicating minimal HF interface."""

    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>" # noqa: S105
    eos_token = "</s>" # noqa: S105

    def __call__(
        self,
        texts: list[str],
        return_tensors: str = "pt",
        **kwargs: Any,# noqa: ANN401
    ) -> dict[str, torch.Tensor]:
        """Return fake tokenized tensors."""
        batch = len(texts)
        seq_len = max(len(t) for t in texts)
        input_ids = torch.randint(2, 50, (batch, seq_len))
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn}

    def batch_decode(
        self,
        ids: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs: Any, # noqa: ANN401
    ) -> list[str]:
        """Return dummy decoded strings."""
        return ["decoded text" for _ in range(ids.size(0))]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str:
        """Return concatenated message contents."""
        return " ".join(m["content"] for m in messages)


class MockHFModel(torch.nn.Module):
    """Fake causal LM returning random logits and hidden states."""

    def __init__(self) -> None:
        """Initialize fake model."""
        super().__init__()
        self.dtype = torch.float32

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any, # noqa: ANN401
    ) -> SimpleNamespace:
        """Return random logits and hidden states."""
        b, t = input_ids.shape
        device = input_ids.device
        logits = torch.randn(b, t, 100, device=device)
        hidden_states = [torch.randn(b, t, 16, device=device) for _ in range(3)]
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any, # noqa: ANN401
    ) -> torch.Tensor:
        """Return random generated token IDs."""
        b, t = input_ids.shape
        return torch.randint(2, 50, (b, t + 5))
