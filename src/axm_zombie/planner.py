from __future__ import annotations
from typing import Any, Dict, List
import math

from .placement import (
    BYTES_PER_GB,
    PlacementRefusal,
    build_seats,
    prove_placement,
    resolve_exact_weight_bytes,
    resolve_model_identity,
)

# WL-02 removed `_estimate_model_bytes` rather than keeping it beside the exact
# path. It inferred a parameter count from the model name and fell through to a
# silent 30B default for anything it did not recognise, then that number
# authorized a GPU placement. The resulting plan was byte-identical in shape to
# a correct one, so the guess was never surfaced as uncertainty - it was
# surfaced as a placement. Nothing here may size a placement from a name again;
# `placement.resolve_exact_weight_bytes` refuses instead.

def _cluster_gpus(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Seats in the cluster, with declared physical-accelerator identity.

    Kept as the plan's `cluster.gpus` shape. `build_seats` adds
    `accelerator_uuid` and refuses a manifest that seats one board twice.
    """
    return build_seats(manifest)

def _link_bw_between(manifest: Dict[str, Any], a: str, b: str) -> float:
    if a == b:
        return float("inf")
    for node in manifest["cluster"]["nodes"]:
        if node["id"] != a:
            continue
        for link in node.get("links", []):
            if link.get("to") == b and link.get("gbps") is not None:
                return float(link["gbps"])
    return 0.0

def _score_stage_boundary(manifest: Dict[str, Any], left_nodes: List[str], right_nodes: List[str], activation_gb_per_token: float, target_tps: float) -> float:
    if not left_nodes or not right_nodes:
        return 0.0
    bw = 0.0
    for a in left_nodes:
        for b in right_nodes:
            bw = max(bw, _link_bw_between(manifest, a, b))
    if bw <= 0.0:
        return 1e9
    required_gbps = activation_gb_per_token * target_tps * 8.0
    util = required_gbps / bw
    if util > 0.7:
        return 1e6 * util
    return 1e3 * util

def build_plan(manifest: Dict[str, Any]) -> Dict[str, Any]:
    model = manifest["model"]
    policy = manifest["policy"]
    gpus = _cluster_gpus(manifest)

    # Placement is authorized by the measured object or not at all.
    model_bytes = resolve_exact_weight_bytes(model)
    model_identity = resolve_model_identity(model)
    model_gb = model_bytes / BYTES_PER_GB

    reserve_gb_per_gpu = float(policy.get("reserve_gb_per_gpu", 4.0))

    # There is deliberately no pooled-capacity gate here. The v1 planner
    # admitted a placement when `sum(vram) - reserve * count` cleared the model
    # size, which is a claim about an address space that does not exist. The
    # proof below assigns every byte to a named device and checks each stage
    # against its own seats instead.

    node_ids = sorted(list({g["node"] for g in gpus}))
    multi_node = len(node_ids) > 1

    stage_count = len(node_ids) if multi_node else max(1, math.ceil(len(gpus) / 2))

    stages: List[List[Dict[str, Any]]] = []
    if multi_node:
        for nid in node_ids:
            stages.append([g for g in gpus if g["node"] == nid])
    else:
        def sort_key(g):
            bw = g["mem_bw_gbps"]
            bwv = float(bw) if bw is not None else 0.0
            return (bwv, g["vram_gb"])
        sorted_gpus = sorted(gpus, key=sort_key, reverse=True)
        chunk = math.ceil(len(sorted_gpus) / stage_count)
        for i in range(stage_count):
            stages.append(sorted_gpus[i * chunk : (i + 1) * chunk])

    activation_gb_per_token = 0.002
    boundary_costs: List[float] = []
    for i in range(len(stages) - 1):
        left_nodes = sorted(list({g["node"] for g in stages[i]}))
        right_nodes = sorted(list({g["node"] for g in stages[i + 1]}))
        boundary_costs.append(
            _score_stage_boundary(
                manifest, left_nodes, right_nodes, activation_gb_per_token, float(policy.get("target_tps", 18))
            )
        )

    placement_proof = prove_placement(
        manifest, stages, model_bytes, model_identity, reserve_gb_per_gpu
    )

    return {
        "schema_version": 1,
        "model": {
            "name": model["name"],
            "dtype": model["dtype"],
            "kv_cache": model.get("kv_cache", model["dtype"]),
            "estimated_model_gb": round(model_gb, 2),
            "model_size_bytes": model_bytes,
            "model_object_sha256": model_identity,
        },
        "cluster": {
            "name": manifest["cluster"].get("name", "zombie"),
            "nodes": node_ids,
            "gpus": gpus,
        },
        "policy": policy,
        "pipeline_stages": [
            {"stage": i, "placement": [{"node": g["node"], "gpu": g["gpu"]} for g in stage]}
            for i, stage in enumerate(stages)
        ],
        "placement_proof": placement_proof,
        "kv_cache_policy": {
            "reserve_gb_per_gpu": reserve_gb_per_gpu,
            "spill_allowed": bool(policy.get("allow_cpu_offload", False)),
        },
        "health": {"replan_on_gpu_oom": True, "replan_on_node_loss": True},
        "diagnostics": {
            "stage_boundary_costs": boundary_costs,
            "notes": "Planner is heuristic. Tune KV cache and activation sizing per model for accurate TPS estimates.",
        },
    }
