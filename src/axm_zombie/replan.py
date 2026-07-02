"""Degrade a manifest after hardware loss and replan.

This is the loop that makes salvage fleets durable: a dead GPU or node is a
manifest edit plus a re-plan, never a hand-edited plan. See DURABILITY.md.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

def parse_loss(spec: str) -> Tuple[str, Optional[int]]:
    """Parse a loss spec: "node-id" (whole node) or "node-id:N" (one GPU)."""
    if ":" in spec:
        node_id, idx = spec.rsplit(":", 1)
        try:
            return node_id, int(idx)
        except ValueError:
            raise ValueError(f"Bad loss spec {spec!r}: GPU index must be an integer.")
    return spec, None

def degrade_manifest(manifest: Dict[str, Any], losses: List[str]) -> Dict[str, Any]:
    """Return a copy of the manifest with the lost GPUs/nodes removed.

    Raises ValueError if a loss spec matches nothing (typo safety) or if no
    hardware remains.
    """
    m = copy.deepcopy(manifest)
    nodes = m["cluster"]["nodes"]

    for spec in losses:
        node_id, gpu_idx = parse_loss(spec)
        node = next((n for n in nodes if n["id"] == node_id), None)
        if node is None:
            raise ValueError(f"Loss {spec!r}: no node with id {node_id!r} in manifest.")
        if gpu_idx is None:
            nodes.remove(node)
            continue
        gpu = next((g for g in node["gpus"] if int(g["index"]) == gpu_idx), None)
        if gpu is None:
            raise ValueError(f"Loss {spec!r}: node {node_id!r} has no GPU with index {gpu_idx}.")
        node["gpus"].remove(gpu)
        if not node["gpus"]:
            nodes.remove(node)

    if not nodes:
        raise ValueError("All nodes lost; nothing left to plan.")

    live = {n["id"] for n in nodes}
    for n in nodes:
        n["links"] = [l for l in n.get("links", []) if l.get("to") in live]

    return m
