import json
from axm_zombie.manifest import load_manifest
from axm_zombie.planner import build_plan

def test_smoke_plan(tmp_path):
    m = load_manifest("examples/cluster_4x3090_singlebox.yaml")
    plan = build_plan(m)
    assert "pipeline_stages" in plan
    assert len(plan["pipeline_stages"]) >= 1
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
