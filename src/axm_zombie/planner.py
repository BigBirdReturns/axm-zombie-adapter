from __future__ import annotations
from typing import Any, Dict, List
import math
import re

_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8": 1.0,
    "int4": 0.5,
    "q4": 0.5,
}

# Ordered longest-first so "70b" is not shadowed by "7b".
_KNOWN_PARAMS = (
    ("70b", 70_000_000_000),
    ("34b", 34_000_000_000),
    ("13b", 13_000_000_000),
    ("7b", 7_000_000_000),
)

# "8x7b", "4 x 22b": a mixture-of-experts multiplier makes a bare parameter
# substring a lie. mixtral-8x7b is ~46.7B, not 7B.
_MOE_MULTIPLIER = re.compile(r"\d+\s*x\s*\d+\s*b\b")


def _bytes_per_param(dtype: str) -> float:
    return _BYTES_PER_PARAM.get(str(dtype).lower(), 2.0)


def _estimate_model_bytes(model: Dict[str, Any]) -> int:
    """Model size in bytes.

    Declared size always wins over inference from the name. When the size can be
    neither declared nor inferred unambiguously, this refuses instead of
    guessing: a wrong guess here is emitted as a confident placement plan that
    OOMs on contact with real hardware.
    """
    dtype = model.get("dtype", "")
    name = str(model.get("name", ""))

    if model.get("bytes") is not None:
        return int(model["bytes"])
    if model.get("params") is not None:
        return int(float(model["params"]) * _bytes_per_param(dtype))

    lowered = name.lower()
    if _MOE_MULTIPLIER.search(lowered):
        raise ValueError(
            f"Model {name!r} carries a mixture-of-experts multiplier, so its "
            "parameter count cannot be read from its name. Declare model.params "
            "(total parameters, not per-expert) or model.bytes in the manifest."
        )
    for token, params in _KNOWN_PARAMS:
        if token in lowered:
            return int(params * _bytes_per_param(dtype))
    raise ValueError(
        f"Unknown model size for {name!r}. The planner will not assume a "
        "default parameter count, because the resulting plan would look "
        "identical to a correct one. Declare model.params or model.bytes "
        "in the manifest."
    )

def _cluster_gpus(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    for node in manifest["cluster"]["nodes"]:
        for gpu in node["gpus"]:
            gpus.append({
                "node": node["id"],
                "host": node.get("host"),
                "gpu": int(gpu["index"]),
                "name": gpu.get("name", "GPU"),
                "vram_gb": float(gpu["vram_gb"]),
                "mem_bw_gbps": gpu.get("mem_bw_gbps"),
            })
    return gpus

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

    total_vram = sum(g["vram_gb"] for g in gpus)
    model_bytes = _estimate_model_bytes(model)
    model_gb = model_bytes / (1024**3)

    reserve_gb_per_gpu = float(policy.get("reserve_gb_per_gpu", 4.0))
    usable_vram = max(0.0, total_vram - reserve_gb_per_gpu * len(gpus))
    if usable_vram < model_gb * 1.05:
        raise ValueError(
            f"Insufficient VRAM for rough model estimate. Usable ~{usable_vram:.1f} GB, model ~{model_gb:.1f} GB."
        )

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

    return {
        "schema_version": 1,
        "model": {
            "name": model["name"],
            "dtype": model["dtype"],
            "kv_cache": model.get("kv_cache", model["dtype"]),
            "estimated_model_gb": round(model_gb, 2),
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
