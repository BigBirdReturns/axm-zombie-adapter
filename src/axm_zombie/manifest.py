from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

REQUIRED_TOP = ["cluster", "model", "policy"]

def load_manifest(path: str, strict: bool = False) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a YAML mapping.")
    for k in REQUIRED_TOP:
        if k not in data:
            raise ValueError(f"Missing top-level key: {k}")

    cluster = data["cluster"]
    if "nodes" not in cluster or not isinstance(cluster["nodes"], list) or not cluster["nodes"]:
        raise ValueError("cluster.nodes must be a non-empty list.")

    for node in cluster["nodes"]:
        if "id" not in node:
            raise ValueError("Each node must have an id.")
        if "gpus" not in node or not isinstance(node["gpus"], list) or not node["gpus"]:
            raise ValueError("Each node must have a non-empty gpus list.")
        for gpu in node["gpus"]:
            for req in ["index", "vram_gb"]:
                if req not in gpu:
                    raise ValueError(f"GPU entry on node {node['id']} missing {req}.")
            gpu.setdefault("name", "GPU")
            gpu.setdefault("mem_bw_gbps", None)
        node.setdefault("links", [])
        node.setdefault("notes", "")
        node.setdefault("host", None)

    model = data["model"]
    for req in ["name", "dtype"]:
        if req not in model:
            raise ValueError(f"model.{req} is required.")
    model.setdefault("kv_cache", model["dtype"])

    policy = data["policy"]
    policy.setdefault("target_tps", 18)
    policy.setdefault("max_latency_ms", 250)
    policy.setdefault("prefer_bandwidth", True)
    policy.setdefault("allow_cpu_offload", False)
    policy.setdefault("reserve_gb_per_gpu", 4.0)

    if strict:
        for node in cluster["nodes"]:
            for gpu in node["gpus"]:
                if gpu.get("mem_bw_gbps") is None:
                    raise ValueError("strict mode: gpu.mem_bw_gbps is required for all gpus.")
        if all(len(n.get("links", [])) == 0 for n in cluster["nodes"]) and len(cluster["nodes"]) > 1:
            raise ValueError("strict mode: cluster has multiple nodes but no links defined.")

    return data
