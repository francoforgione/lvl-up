"""Token accounting and cost estimation for Claude API calls."""

from dataclasses import dataclass

# USD per 1M tokens, (input, output). Keyed by model id prefix so both the alias
# and the dated full id resolve to the same entry.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
}


@dataclass
class TokenUsage:
    """Accumulates token counts across the iterations of one agentic turn."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage) -> None:
        """Add one API response's `usage` object to the running total."""
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate_cost(model: str, usage: TokenUsage) -> float:
    """Estimate the USD cost of a turn. Returns 0.0 for unknown models."""
    prices = next((p for prefix, p in PRICING_PER_MTOK.items() if model.startswith(prefix)), None)
    if prices is None:
        return 0.0
    input_price, output_price = prices
    return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000
