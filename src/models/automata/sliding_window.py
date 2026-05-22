from __future__ import annotations


class SlidingWindow:
    """Extract fixed-length overlapping symbolic patterns."""

    def __init__(self, size: int, stride: int = 1):
        if size <= 0:
            raise ValueError("size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.size = size
        self.stride = stride

    def transform(self, symbols: list[str]) -> list[str]:
        if len(symbols) < self.size:
            return []

        return [
            "".join(symbols[start : start + self.size])
            for start in range(0, len(symbols) - self.size + 1, self.stride)
        ]
