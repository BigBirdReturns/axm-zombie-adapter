import json
from pathlib import Path

import pytest

from axm_zombie.manifest import load_manifest
from axm_zombie.planner import build_plan
from axm_zombie.replan import degrade_manifest

REPO = Path(__file__).resolve().parent.parent

def _singlebox():
    return load_manifest(str(REPO / "examples" / "cluster_4x3090_singlebox.json"))

def _two_nodes(tmp_path):
    data = {
        "cluster": {
            "name": "c",
            "nodes": [
                {
                    "id": "node-a",
                    "host": "10.0.0.1",
                    "gpus": [
                        {"index": 0, "vram_gb": 24, "mem_bw_gbps": 936},
                        {"index": 1, "vram_gb": 24, "mem_bw_gbps": 936},
                    ],
                    "links": [{"type": "ethernet", "to": "node-b", "gbps": 10}],
                },
                {
                    "id": "node-b",
                    "host": "10.0.0.2",
                    "gpus": [
                        {"index": 0, "vram_gb": 24, "mem_bw_gbps": 936},
                        {"index": 1, "vram_gb": 24, "mem_bw_gbps": 936},
                    ],
                    "links": [{"type": "ethernet", "to": "node-a", "gbps": 10}],
                },
            ],
        },
        "model": {"name": "llama-3-70b", "dtype": "q4", "kv_cache": "fp16"},
        "policy": {},
    }
    p = tmp_path / "two.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return load_manifest(str(p))

def _placed(plan):
    return [(p["node"], p["gpu"]) for s in plan["pipeline_stages"] for p in s["placement"]]

def test_lose_one_gpu():
    degraded = degrade_manifest(_singlebox(), ["node-a:3"])
    plan = build_plan(degraded)
    placed = _placed(plan)
    assert len(placed) == 3
    assert ("node-a", 3) not in placed

def test_lose_whole_node(tmp_path):
    degraded = degrade_manifest(_two_nodes(tmp_path), ["node-b"])
    plan = build_plan(degraded)
    assert plan["cluster"]["nodes"] == ["node-a"]
    # No dangling links to the dead node remain in the manifest.
    assert all(
        link["to"] != "node-b"
        for node in degraded["cluster"]["nodes"]
        for link in node["links"]
    )

def test_lose_last_gpu_removes_node(tmp_path):
    degraded = degrade_manifest(_two_nodes(tmp_path), ["node-b:0", "node-b:1"])
    assert [n["id"] for n in degraded["cluster"]["nodes"]] == ["node-a"]

def test_lose_too_much_is_infeasible():
    degraded = degrade_manifest(_singlebox(), ["node-a:1", "node-a:2", "node-a:3"])
    with pytest.raises(ValueError, match="Insufficient VRAM"):
        build_plan(degraded)

def test_unknown_node_rejected():
    with pytest.raises(ValueError, match="no node"):
        degrade_manifest(_singlebox(), ["node-z"])

def test_unknown_gpu_rejected():
    with pytest.raises(ValueError, match="no GPU"):
        degrade_manifest(_singlebox(), ["node-a:9"])

def test_original_manifest_untouched():
    m = _singlebox()
    degrade_manifest(m, ["node-a:0"])
    assert len(m["cluster"]["nodes"][0]["gpus"]) == 4
