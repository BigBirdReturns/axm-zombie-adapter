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

def test_infeasible_model_rejected(tmp_path):
    manifest = json.loads((REPO / "examples" / "cluster_4x3090_singlebox.json").read_text())
    manifest["model"]["dtype"] = "fp32"  # 70B fp32 cannot fit 4x24GB
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Insufficient VRAM"):
        build_plan(load_manifest(str(p)))
