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

from axm_zombie.placement import PlacementRefusal  # noqa: E402
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


# Named clusters, so a case can select the hardware its predicate is about
# without every case restating four GPUs. The default one is deliberately bare:
# no board identities and no runtimes, which is what a manifest written before
# placement evidence existed looks like.
#
# The digests in the case table are sha-256 of the ASCII labels
# "witness-model-object:llama-3-70b-q4" and
# "witness-model-object:a-different-object", the same labels the Python and
# browser witnesses use. Nothing here is an invented digest.
DUPLICATE_BOARD = "GPU-493239dc-f76e-bbbb-8e68-ffd34a5e7bbc"


def _gpus(uuids=None):
    gpus = []
    for i in range(4):
        gpu = {"index": i, "name": "RTX_3090", "vram_gb": 24, "mem_bw_gbps": 936}
        if uuids is not None:
            gpu["uuid"] = uuids[i]
        gpus.append(gpu)
    return gpus


_DISTINCT = [f"GPU-aaaa1111-0000-4000-8000-00000000000{i}" for i in range(4)]
_MEASURED = [{"engine": "llamacpp", "compatibility": "MEASURED"}]

CLUSTERS = {
    None: {"gpus": _gpus(), "runtimes": []},
    "evidenced": {"gpus": _gpus(_DISTINCT), "runtimes": _MEASURED},
    "duplicate-board": {
        "gpus": _gpus(_DISTINCT[:2] + [DUPLICATE_BOARD, DUPLICATE_BOARD]),
        "runtimes": _MEASURED,
    },
    "no-runtime": {"gpus": _gpus(_DISTINCT), "runtimes": []},
    "unmeasured-runtime": {"gpus": _gpus(_DISTINCT), "runtimes": ["llamacpp"]},
}


def manifest_for(model, cluster=None):
    shape = CLUSTERS[cluster]
    return {
        "schema_version": 1,
        "cluster": {
            "name": "t",
            "nodes": [{
                "id": "node-a", "host": "127.0.0.1",
                "runtimes": list(shape["runtimes"]),
                "gpus": [dict(g) for g in shape["gpus"]],
            }],
        },
        "model": dict(model),
        "policy": {"target_tps": 18, "max_latency_ms": 250, "prefer_bandwidth": True,
                   "allow_cpu_offload": False, "reserve_gb_per_gpu": 4.0},
    }


def python_outcome(case):
    """The complete emitted placement artifact, or the refusal that replaced it.

    Deliberately not a projection. An earlier version of this table selected
    `gb`, `source`, and `sha`, which left `model_size_bytes` -- the exact
    number the placement was authorized against -- outside the comparison
    entirely, so the two writers could publish different byte counts and still
    agree on every field this table checked. Now the complete model object and
    the complete placement decision are both compared.

    A typed refusal is compared by `code`, which is the machine-readable
    contract; prose cannot be compared across runtimes, so the marker set is
    kept as a secondary check for the untyped size-authority refusals that
    have no code.
    """
    try:
        plan = build_plan(manifest_for(case["model"], case.get("cluster")))
    except PlacementRefusal as refusal:
        return {"kind": "refuse", "code": refusal.code,
                "markers": sorted(m for m in MARKERS if m in str(refusal))}
    except ValueError as e:
        return {"kind": "refuse", "code": None,
                "markers": sorted(m for m in MARKERS if m in str(e))}
    return {"kind": "plan", "model": plan["model"], "placement": plan["placement"]}


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
    outcome = python_outcome(case)
    if outcome["kind"] == "plan":
        assert set(outcome["model"]) == MODEL_OBJECT_KEYS, (
            "the plan's model object gained or lost a field; update "
            "MODEL_OBJECT_KEYS and SPEC.md deliberately, and keep the new "
            "field inside the two-writer comparison"
        )
    assert outcome == browser_outcomes[case["id"]], (
        f"the two plan writers disagree about {case['id']}"
    )
