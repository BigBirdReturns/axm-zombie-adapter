from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class Metrics:
    tokens_per_sec: float
    p50_latency_ms: float
    errors: int

class Monitor:
    """
    Placeholder monitor.

    Wire this into your inference server logs or metrics endpoint.
    """

    def __init__(self) -> None:
        self._errors = 0

    def sample(self) -> Metrics:
        # Replace with real sampling.
        return Metrics(tokens_per_sec=0.0, p50_latency_ms=0.0, errors=self._errors)

    def loop(self, interval_s: float = 5.0) -> None:
        while True:
            m = self.sample()
            print(f"tps={m.tokens_per_sec:.2f} p50_ms={m.p50_latency_ms:.1f} errors={m.errors}")
            time.sleep(interval_s)
