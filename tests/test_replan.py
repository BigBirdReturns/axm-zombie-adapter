import json
from pathlib import Path

import pytest

from axm_zombie.manifest import load_manifest
from axm_zombie.planner import build_plan
from axm_zombie.replan import degrade_manifest

REPO = Path(__file__).resolve().parent.parent

DIGEST = "863a91e7b83b67c2781e40f95fa6416edf737f5e7908d0c4d6b321cfc0df3a3d"

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
                    "runtimes": [{"engine": "llamacpp", "compatibility": "MEASURED"}],
                    "gpus": [
                        {"index": 0, "vram_gb": 24, "mem_bw_gbps": 936},
                        {"index": 1, "vram_gb": 24, "mem_bw_gbps": 936},
                    ],
                    "links": [{"type": "ethernet", "to": "node-b", "gbps": 10}],
                },
                {
                    "id": "node-b",
                    "host": "10.0.0.2",
                    "runtimes": [{"engine": "llamacpp", "compatibility": "MEASURED"}],
                    "gpus": [
                        {"index": 0, "vram_gb": 24, "mem_bw_gbps": 936},
                        {"index": 1, "vram_gb": 24, "mem_bw_gbps": 936},
                    ],
                    "links": [{"type": "ethernet", "to": "node-a", "gbps": 10}],
                },
            ],
        },
        # WL-02: placement consumes the tensor payload, a verified object
        # identity, and verified current residence, and only places on nodes
        # where that object is actually resident.
        "model": {
            "name": "llama-3-70b",
            "dtype": "q4",
            "kv_cache": "fp16",
            "bytes": 35000000000,
            "sha256": DIGEST,
            "weights": {
                "checkpoint_set_bytes": 35008388608,
                "tensor_payload_bytes": 35000000000,
                "container_overhead_bytes": 8388608,
            },
            "identity": {
                "identity_scheme": "example/ascii-label-object@1",
                "identity_digest": DIGEST,
                "identity_source": "examples/cluster_4x3090_2nodes_10gbe.json#_note",
                "identity_state": "VERIFIED",
                "verified_at": "2026-08-16T00:00:00Z",
                "validator": "sha256 over the ASCII label named in _note",
            },
            "engines": ["llamacpp"],
            "residence": [
                {"node": "node-a", "verified": True, "freshness": "CURRENT",
                 "identity_digest": DIGEST},
                {"node": "node-b", "verified": True, "freshness": "CURRENT",
                 "identity_digest": DIGEST},
            ],
        },
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
    from axm_zombie.placement import PlacementRefusal, REFUSAL_STAGE_DOES_NOT_FIT

    degraded = degrade_manifest(_singlebox(), ["node-a:1", "node-a:2", "node-a:3"])
    with pytest.raises(PlacementRefusal) as excinfo:
        build_plan(degraded)
    # The surviving seat is named, with the shortfall against its own memory.
    assert excinfo.value.code == REFUSAL_STAGE_DOES_NOT_FIT
    assert excinfo.value.detail["deficit_bytes"] > 0

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

def test_replanning_after_gpu_loss_does_not_weaken_the_model_requirement():
    """Fewer devices must mean a smaller share each, never a smaller model.

    The tempting failure is to survive hardware loss by quietly asking less of
    the survivors. Every byte still has to land somewhere, so the invariant is
    that the payload placed after the loss is exactly the payload placed
    before it -- redistributed, not reduced.
    """
    before = build_plan(_singlebox())
    after = build_plan(degrade_manifest(_singlebox(), ["node-a:3"]))

    assert before["placement"]["state"] == after["placement"]["state"] == "RUNNABLE"
    before_proof, after_proof = before["placement"]["proof"], after["placement"]["proof"]
    assert after_proof["weights"] == before_proof["weights"]
    assert after_proof["assigned_bytes_total"] == before_proof["assigned_bytes_total"]
    assert after_proof["identity"] == before_proof["identity"]

    # Redistributed across three seats instead of four, and still exact.
    devices = [d for s in after_proof["stages"] for d in s["devices"]]
    assert len(devices) == 3
    assert sum(d["assigned_bytes"] for d in devices) == after_proof["assigned_bytes_total"]
    assert all(d["assigned_bytes"] <= d["usable_bytes"] for d in devices)

def test_replanning_after_node_loss_does_not_weaken_the_model_requirement(tmp_path):
    before = build_plan(_two_nodes(tmp_path))
    after = build_plan(degrade_manifest(_two_nodes(tmp_path), ["node-b"]))

    assert after["cluster"]["nodes"] == ["node-a"]
    before_proof, after_proof = before["placement"]["proof"], after["placement"]["proof"]
    assert after_proof["weights"] == before_proof["weights"]
    assert after_proof["assigned_bytes_total"] == before_proof["assigned_bytes_total"]
    devices = [d for s in after_proof["stages"] for d in s["devices"]]
    assert {d["node"] for d in devices} == {"node-a"}
    assert sum(d["assigned_bytes"] for d in devices) == after_proof["assigned_bytes_total"]
    assert all(d["assigned_bytes"] <= d["usable_bytes"] for d in devices)
