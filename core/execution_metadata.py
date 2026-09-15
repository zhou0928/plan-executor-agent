"""Execution metadata for the agent node's `usage` output.

Dify reads `execution_metadata` out of a trailing JSON message to fill the
node's token/price panel; without it the panel stays empty. Shape mirrors
Dify's official agent strategies. Kept dependency-free (no pydantic) so core/
stays importable in tests without dify_plugin installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ExecutionMetadata:
    total_price: float = 0.0
    currency: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    prompt_unit_price: float = 0.0
    prompt_price_unit: float = 0.0
    prompt_price: float = 0.0
    completion_tokens: int = 0
    completion_unit_price: float = 0.0
    completion_price_unit: float = 0.0
    completion_price: float = 0.0
    latency: float = 0.0

    @classmethod
    def from_llm_usage(cls, usage: Any) -> "ExecutionMetadata":
        """Build from an SDK LLMUsage (duck-typed; None yields all-defaults)."""
        if usage is None:
            return cls()
        return cls(
            total_price=float(usage.total_price),
            currency=str(usage.currency),
            total_tokens=int(usage.total_tokens),
            prompt_tokens=int(usage.prompt_tokens),
            prompt_unit_price=float(usage.prompt_unit_price),
            prompt_price_unit=float(usage.prompt_price_unit),
            prompt_price=float(usage.prompt_price),
            completion_tokens=int(usage.completion_tokens),
            completion_unit_price=float(usage.completion_unit_price),
            completion_price_unit=float(usage.completion_price_unit),
            completion_price=float(usage.completion_price),
            latency=float(usage.latency),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
