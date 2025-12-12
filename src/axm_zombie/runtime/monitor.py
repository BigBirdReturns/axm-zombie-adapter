from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class Metrics:
    tps: float
    p50_latency_ms: float
    p95_latency_ms: float
    oom_events: int

class SimpleMonitor:
    """Minimal monitoring interface placeholder."""

    def __init__(self) -> None:
        self.oom_events = 0

    def sample(self) -> Metrics:
        return Metrics(tps=0.0, p50_latency_ms=0.0, p95_latency_ms=0.0, oom_events=self.oom_events)

    def run_loop(self, interval_s: float = 5.0) -> None:
        while True:
            m = self.sample()
            print(f"tps={m.tps:.2f} p50={m.p50_latency_ms:.1f}ms p95={m.p95_latency_ms:.1f}ms oom={m.oom_events}")
            time.sleep(interval_s)
