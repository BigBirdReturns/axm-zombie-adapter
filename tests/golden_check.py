"""Resurrection test: prove the toolchain is alive with no network and no
third-party dependencies.

    python tests/golden_check.py

Parses the JSON example manifest, builds a plan, verifies it is byte-identical
to the committed golden file, and runs every exporter on it. If this passes,
the project works, regardless of what year it is.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"

sys.path.insert(0, str(REPO / "src"))

from axm_zombie.manifest import load_manifest  # noqa: E402
from axm_zombie.planner import build_plan  # noqa: E402
from axm_zombie.export.exo import export_exo  # noqa: E402
from axm_zombie.export.petals import export_petals  # noqa: E402
from axm_zombie.export.torchrun import export_torchrun  # noqa: E402

def plan_text(manifest_path: Path) -> str:
    manifest = load_manifest(str(manifest_path))
    return json.dumps(build_plan(manifest), indent=2) + "\n"

def check_golden(manifest_path: Path, golden_path: Path) -> None:
    got = plan_text(manifest_path)
    want = golden_path.read_text(encoding="utf-8")
    if got != want:
        raise AssertionError(
            f"Plan for {manifest_path.name} is not byte-identical to {golden_path.name}. "
            "If the schema changed intentionally, bump schema_version per SPEC.md and "
            "regenerate goldens with: python tests/golden_check.py --regen"
        )

def check_exporters(manifest_path: Path) -> None:
    plan = json.loads(plan_text(manifest_path))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        export_exo(plan, out)
        export_petals(plan, out)
        export_torchrun(plan, out)
        for f in ["plan.json", "exo_notes.md", "run_node.sh", "petals_notes.md", "run_petals.sh", "run_torchrun.sh"]:
            if not (out / f).exists():
                raise AssertionError(f"Exporter did not write {f}")

def main(regen: bool = False) -> int:
    manifest_path = REPO / "examples" / "cluster_4x3090_singlebox.json"
    golden_path = GOLDEN / "cluster_4x3090_singlebox.plan.json"
    if regen:
        GOLDEN.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(plan_text(manifest_path), encoding="utf-8")
        print(f"regenerated {golden_path}")
        return 0
    check_golden(manifest_path, golden_path)
    check_exporters(manifest_path)
    print("resurrection test: OK (plan is byte-identical to golden; all exporters ran)")
    return 0

if __name__ == "__main__":
    sys.exit(main(regen="--regen" in sys.argv[1:]))
