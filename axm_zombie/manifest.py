from __future__ import annotations

from typing import Any, Dict, List
import yaml

from .types import GPU, Link, Manifest, ModelSpec, Node, Policy

def load_manifest(path: str) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cluster = raw.get("cluster") or {}
    model = raw.get("model") or {}
    policy = raw.get("policy") or {}

    nodes_raw = cluster.get("nodes") or []
    nodes: List[Node] = []

    for n in nodes_raw:
        gpus_raw = n.get("gpus") or []
        links_raw = n.get("links") or []
        gpus = [
            GPU(
                index=int(g.get("index", 0)),
                name=str(g.get("name", "GPU")),
                vram_gb=float(g.get("vram_gb", 0)),
                mem_bw_gbps=float(g.get("mem_bw_gbps", 0)),
            )
            for g in gpus_raw
        ]
        links = [
            Link(
                type=str(l.get("type", "ethernet")),
                to=str(l.get("to", "")),
                gbps=float(l.get("gbps", 0)),
            )
            for l in links_raw
        ]
        nodes.append(
            Node(
                id=str(n.get("id")),
                host=str(n.get("host", "")),
                gpus=gpus,
                links=links,
            )
        )

    ms = ModelSpec(
        name=str(model.get("name", "unknown-model")),
        dtype=str(model.get("dtype", "fp16")),
        kv_cache=str(model.get("kv_cache", "fp16")),
    )
    pol = Policy(
        target_tps=float(policy.get("target_tps", 18)),
        max_latency_ms=float(policy.get("max_latency_ms", 250)),
        prefer_bandwidth=bool(policy.get("prefer_bandwidth", True)),
        allow_cpu_offload=bool(policy.get("allow_cpu_offload", False)),
    )

    return Manifest(
        cluster_name=str(cluster.get("name", "zombie-cluster")),
        nodes=nodes,
        model=ms,
        policy=pol,
    )
