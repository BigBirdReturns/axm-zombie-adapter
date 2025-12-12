from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    vram_gb: float
    mem_bw_gbps: float

@dataclass(frozen=True)
class Link:
    type: str
    to: str
    gbps: float

@dataclass(frozen=True)
class Node:
    id: str
    host: str
    gpus: List[GPU]
    links: List[Link]

@dataclass(frozen=True)
class ModelSpec:
    name: str
    dtype: str = "fp16"
    kv_cache: str = "fp16"

@dataclass(frozen=True)
class Policy:
    target_tps: float = 18.0
    max_latency_ms: float = 250.0
    prefer_bandwidth: bool = True
    allow_cpu_offload: bool = False

@dataclass(frozen=True)
class Manifest:
    cluster_name: str
    nodes: List[Node]
    model: ModelSpec
    policy: Policy

@dataclass(frozen=True)
class Placement:
    node: str
    gpu: int

@dataclass(frozen=True)
class Stage:
    stage: int
    placement: List[Placement]

@dataclass(frozen=True)
class Plan:
    model: str
    pipeline_stages: List[Stage]
    tensor_parallel_groups: List[List[Placement]]
    kv_cache_policy: Dict[str, Any]
    routing: Dict[str, Any]
    health: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "pipeline_stages": [
                {"stage": s.stage, "placement": [{"node": p.node, "gpu": p.gpu} for p in s.placement]}
                for s in self.pipeline_stages
            ],
            "tensor_parallel_groups": [
                [{"node": p.node, "gpu": p.gpu} for p in group]
                for group in self.tensor_parallel_groups
            ],
            "kv_cache_policy": self.kv_cache_policy,
            "routing": self.routing,
            "health": self.health,
        }
