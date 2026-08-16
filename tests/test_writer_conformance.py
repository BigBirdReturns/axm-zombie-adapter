"""Both plan writers must decide the same way about the same model.

The repository ships two writers of the plan artifact: the normative planner in
`src/axm_zombie/planner.py` and the published browser port in
`docs/planner.js`. They have drifted before -- the browser port kept a silent
30e9 parameter default and a bare "7b" substring match long after the reference
implementation refused both -- and nothing in the suite could detect it,
because the Python tests cannot load JavaScript.

This test closes that gap: one shared case table, run through both writers,
compared. It skips when no `node` is on PATH so the Python-only matrix stays
green; CI runs it in a job that installs both runtimes.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from axm_zombie.planner import build_plan  # noqa: E402

MARKERS = [
    "Insufficient VRAM",
    "Size-claim conflict",
    "mixture-of-experts",
    "Unknown model size",
    "cannot resolve",
    "exact integer range",
    "model.bytes",
    "model.params",
    "model.sha256",
]

CASES = json.loads((REPO / "tests" / "conformance_cases.json").read_text(encoding="utf-8"))


def manifest_for(model):
    return {
        "schema_version": 1,
        "cluster": {
            "name": "t",
            "nodes": [{
                "id": "node-a", "host": "127.0.0.1",
                "gpus": [
                    {"index": i, "name": "RTX_3090", "vram_gb": 24, "mem_bw_gbps": 936}
                    for i in range(4)
                ],
            }],
        },
        "model": dict(model),
        "policy": {"target_tps": 18, "max_latency_ms": 250, "prefer_bandwidth": True,
                   "allow_cpu_offload": False, "reserve_gb_per_gpu": 4.0},
    }


def python_outcome(model):
    """The complete emitted model object, or the refusal that replaced it.

    Deliberately not a projection. Selecting `gb`, `source`, and `sha` left
    `model_size_bytes` -- the exact number the placement was authorized
    against -- outside the comparison, so the two writers could publish
    different byte counts and still agree on every field this table checked.
    Whatever the schema says the model object contains is what gets compared.
    """
    try:
        plan = build_plan(manifest_for(model))
    except ValueError as e:
        return {"kind": "refuse", "markers": sorted(m for m in MARKERS if m in str(e))}
    return {"kind": "plan", "model": plan["model"]}


@pytest.fixture(scope="module")
def browser_outcomes():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; browser writer cannot be loaded")
    proc = subprocess.run(
        [node, str(REPO / "tests" / "conformance_browser.js")],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if proc.returncode != 0:
        pytest.fail(f"browser writer failed to run:\n{proc.stderr}")
    return json.loads(proc.stdout)


# Every key SPEC.md v1 says the plan's model object carries. Pinned here so a
# new field cannot be added to the writers and quietly stay outside the
# comparison; the point of this table is that nothing in the model object is
# exempt from it.
MODEL_OBJECT_KEYS = {
    "name",
    "dtype",
    "kv_cache",
    "estimated_model_gb",
    "model_size_bytes",
    "model_size_source",
    "model_object_sha256",
}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_both_writers_agree(case, browser_outcomes):
    assert case["id"] in browser_outcomes, "browser writer produced no outcome for this case"
    outcome = python_outcome(case["model"])
    if outcome["kind"] == "plan":
        assert set(outcome["model"]) == MODEL_OBJECT_KEYS, (
            "the plan's model object gained or lost a field; update "
            "MODEL_OBJECT_KEYS and SPEC.md deliberately, and keep the new "
            "field inside the two-writer comparison"
        )
    assert outcome == browser_outcomes[case["id"]], (
        f"the two plan writers disagree about {case['id']}"
    )
