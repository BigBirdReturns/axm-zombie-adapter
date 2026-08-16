import json
from pathlib import Path

import pytest

from axm_zombie.manifest import load_manifest
from axm_zombie.planner import build_plan

import golden_check

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"

def test_resurrection():
    # The stdlib-only front door: JSON example -> plan -> golden byte-compare -> exporters.
    assert golden_check.main() == 0

def test_smoke_plan_json_example():
    plan = build_plan(load_manifest(str(REPO / "examples" / "cluster_4x3090_singlebox.json")))
    assert plan["schema_version"] == 1
    assert len(plan["pipeline_stages"]) >= 1
    placed = [p for s in plan["pipeline_stages"] for p in s["placement"]]
    assert len(placed) == 4

@pytest.mark.parametrize(
    "example,golden",
    [
        ("cluster_4x3090_singlebox.yaml", "cluster_4x3090_singlebox.plan.json"),
        ("cluster_4x3090_2nodes_10gbe.yaml", "cluster_4x3090_2nodes_10gbe.plan.json"),
    ],
)
def test_yaml_examples_match_goldens(example, golden):
    pytest.importorskip("yaml")
    golden_check.check_golden(REPO / "examples" / example, GOLDEN / golden)

def test_yaml_and_json_singlebox_examples_agree():
    pytest.importorskip("yaml")
    assert golden_check.plan_text(REPO / "examples" / "cluster_4x3090_singlebox.yaml") == \
        golden_check.plan_text(REPO / "examples" / "cluster_4x3090_singlebox.json")

def test_yaml_and_json_2node_examples_agree():
    # The JSON twin is what regenerates the 2-node golden, because JSON is the
    # canonical dependency-free form (SPEC.md rule 6) and PyYAML is optional.
    # It is also what docs/planner.js is gated against, since the browser
    # writer has no YAML parser. If the two drift, the browser parity test
    # silently starts checking a different cluster.
    pytest.importorskip("yaml")
    assert golden_check.plan_text(REPO / "examples" / "cluster_4x3090_2nodes_10gbe.yaml") == \
        golden_check.plan_text(REPO / "examples" / "cluster_4x3090_2nodes_10gbe.json")

def test_infeasible_model_rejected(tmp_path):
    # WL-02: the refusal is typed, and it names the stage that cannot hold its
    # own assigned bytes rather than reporting a pooled shortfall. Size comes
    # from declared bytes now, so this enlarges the object rather than changing
    # dtype -- and enlarging it means restating all three weight quantities,
    # because they are one accounting and the planner refuses a partial one.
    from axm_zombie.placement import PlacementRefusal, REFUSAL_STAGE_DOES_NOT_FIT

    manifest = json.loads((REPO / "examples" / "cluster_4x3090_singlebox.json").read_text())
    payload = 1_560_860_324_864          # Kimi-K3 tensor payload, 1453.7 GiB
    overhead = 75_766_584                # safetensors headers, 96 shards
    manifest["model"]["bytes"] = payload
    manifest["model"]["weights"] = {
        "checkpoint_set_bytes": payload + overhead,
        "tensor_payload_bytes": payload,
        "container_overhead_bytes": overhead,
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PlacementRefusal) as excinfo:
        build_plan(load_manifest(str(p)))
    assert excinfo.value.code == REFUSAL_STAGE_DOES_NOT_FIT
    assert excinfo.value.detail["deficit_bytes"] > 0

def test_plan_assigns_every_weight_byte_to_a_named_device():
    plan = build_plan(load_manifest(str(REPO / "examples" / "cluster_4x3090_singlebox.json")))
    assert plan["placement"]["state"] == "RUNNABLE"
    proof = plan["placement"]["proof"]
    assigned = sum(d["assigned_bytes"] for s in proof["stages"] for d in s["devices"])
    # What is placed is the tensor payload, which is also what model.bytes
    # carries. Custody binds the larger checkpoint set, and the proof keeps
    # both so neither can be read as the other.
    assert assigned == proof["weights"]["tensor_payload_bytes"]
    assert assigned == plan["model"]["model_size_bytes"]
    assert proof["weights"]["checkpoint_set_bytes"] > assigned
