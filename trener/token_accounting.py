"""Accounting of completion and processed tokens (Appendix B of the article).

* **Completion tokens** — number of generated completion tokens (both roles for
  JOLT/JOLT+). Prompts and privileged information are not included.
* **Processed training tokens** — all non-padding positions processed by the
  model: prefill and decode of generation, teacher-forced scoring passes and
  optimization passes (repeated under gradient accumulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenCounter:
    """Two token-consumption counters of a run."""

    completion_tokens: int = 0
    processed_tokens: int = 0

    def add_completion(self, n: int) -> None:
        """Adds generated completion tokens."""
        if n < 0:
            raise ValueError(f"number of tokens cannot be negative: {n}")
        self.completion_tokens += int(n)

    def add_processed(self, n: int) -> None:
        """Adds positions processed by the model."""
        if n < 0:
            raise ValueError(f"number of tokens cannot be negative: {n}")
        self.processed_tokens += int(n)

    def snapshot(self) -> dict[str, int]:
        """Current state of the counters."""
        return {
            "completion_tokens": self.completion_tokens,
            "processed_tokens": self.processed_tokens,
        }

    def reset(self) -> None:
        """Resets both counters to zero."""
        self.completion_tokens = 0
        self.processed_tokens = 0

    def state_dict(self) -> dict[str, int]:
        """State for a checkpoint (resume)."""
        return self.snapshot()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restores the state from a checkpoint."""
        self.completion_tokens = int(state.get("completion_tokens", 0))
        self.processed_tokens = int(state.get("processed_tokens", 0))


__all__ = ["TokenCounter"]
