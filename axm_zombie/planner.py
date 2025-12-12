from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

from .types import Manifest, Placement, Plan, Stage
from .cost import cross_node_penalty

def _flatten_gpus(m: Manifest) -> List[Tuple[str, int, float, float]]:
    items: List[Tuple[str, int, float, float]] = []
    for node in m.nodes:
        for gpu in node.gpus:
            items.append((node.id, gpu.index, gpu.vram_gb, gpu.mem_bw_gbps))
    return items

def _link_gbps(m: Manifest, a: str, b: str) -> float:
    if a == b:
        return 1e9
    for node in m.nodes:
        if node.id == a:
            for link in node.links:
                if link.to == b:
                    return link.gbps
    return 0.0

class Planner:
    """
    Planner v0.1

    Goal: produce a feasible, deterministic plan with minimal cross-node boundaries.

    Assumptions:
    - Inference only
    - Pipeline parallelism across nodes
    - Tensor parallelism only within a node (unless explicitly extended later)
    """

    def build(self, m: Manifest) -> Plan:
        gpus = _flatten_gpus(m)
        if not gpus:
            raise ValueError("No GPUs found in manifest")

        # Sort by bandwidth first, then VRAM, then stable identifiers.
        gpus_sorted = sorted(gpus, key=lambda x: (-x[3], -x[2], x[0], x[1]))

        # Heuristic: group GPUs by node for tensor-parallel groups.
        by_node = {}
        for node_id, gpu_idx, vram, bw in gpus_sorted:
            by_node.setdefault(node_id, []).append(Placement(node=node_id, gpu=gpu_idx))

        # Build pipeline stages: each node becomes a stage.
        stages: List[Stage] = []
        for i, node_id in enumerate(sorted(by_node.keys())):
            stages.append(Stage(stage=i, placement=by_node[node_id]))

        # Estimate cross-node activation penalty to decide if we should merge stages.
        # We keep this simple: if any inter-node link is too slow, collapse to single stage.
        activation_gb_per_token = 0.002  # coarse default; override later with model-specific estimator
        total_penalty = 0.0
        for i in range(len(stages) - 1):
            a = stages[i].placement[0].node
            b = stages[i + 1].placement[0].node
            gbps = _link_gbps(m, a, b)
            total_penalty += cross_node_penalty(gbps, activation_gb_per_token, m.policy.target_tps)

        if total_penalty > 5e5 and len(stages) > 1:
            # Collapse everything into one stage on the highest bandwidth node group.
            best_node = max(by_node.keys(), key=lambda nid: sum(1 for p in by_node[nid]))
            stages = [Stage(stage=0, placement=by_node[best_node])]
            tensor_groups = [by_node[best_node]]
        else:
            tensor_groups = [placements for _, placements in sorted(by_node.items(), key=lambda x: x[0])]

        kv_cache_policy = {
            "pin_hot": True,
            "spill_allowed": m.policy.allow_cpu_offload,
            "per_stage_reserve_gb": 4,
        }
        routing = {
            "rag_async": True,
            "tools_on_cpu": True,
            "max_inflight": 8,
        }
        health = {
            "replan_on_gpu_oom": True,
            "replan_on_node_loss": True,
        }

        return Plan(
            model=m.model.name,
            pipeline_stages=stages,
            tensor_parallel_groups=tensor_groups,
            kv_cache_policy=kv_cache_policy,
            routing=routing,
            health=health,
        )
