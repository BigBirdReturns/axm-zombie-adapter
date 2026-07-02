import argparse
import json
from pathlib import Path

from .manifest import load_manifest
from .planner import build_plan
from .export.exo import export_exo
from .export.petals import export_petals
from .export.torchrun import export_torchrun

EXPORTERS = {
    "exo": export_exo,
    "petals": export_petals,
    "torchrun": export_torchrun,
}

def main():
    parser = argparse.ArgumentParser(
        prog="axm-zombie",
        description="AXM Zombie Adapter: manifest to plan to exporter scaffolds",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Generate plan.json from a cluster manifest (YAML or JSON)")
    p_plan.add_argument("manifest", help="Path to cluster manifest (.yaml or .json)")
    p_plan.add_argument("--out", default="plan.json", help="Output plan.json path")
    p_plan.add_argument("--strict", action="store_true", help="Fail on missing optional fields")

    p_exp = sub.add_parser("export", help="Export scaffolds and notes from plan.json")
    p_exp.add_argument("target", choices=sorted(EXPORTERS), help="Exporter target")
    p_exp.add_argument("plan", help="Path to plan.json")
    p_exp.add_argument("--outdir", default="out", help="Output directory")

    args = parser.parse_args()

    if args.cmd == "plan":
        manifest = load_manifest(args.manifest, strict=args.strict)
        plan = build_plan(manifest)
        out_path = Path(args.out)
        out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"Wrote {out_path.resolve()}")
        return

    if args.cmd == "export":
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        EXPORTERS[args.target](plan, outdir)
        print(f"Wrote exporter artifacts to {outdir.resolve()}")
        return
