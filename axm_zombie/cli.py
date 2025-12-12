from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import load_manifest
from .planner import Planner
from .exporter.torchrun import export_torchrun

def cmd_plan(args: argparse.Namespace) -> int:
    m = load_manifest(args.manifest)
    plan = Planner().build(m)
    out = Path(args.out)
    out.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote plan: {out}")
    return 0

def cmd_export(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if args.kind == "torchrun":
        script = export_torchrun(plan, serve_py=args.serve_py)
    else:
        raise ValueError(f"Unsupported exporter kind: {args.kind}")
    out = Path(args.out)
    out.write_text(script, encoding="utf-8")
    out.chmod(0o755)
    print(f"Wrote launcher: {out}")
    return 0

def main() -> int:
    p = argparse.ArgumentParser(prog="axm-zombie")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Compile a cluster manifest into plan.json")
    p_plan.add_argument("manifest", help="Path to cluster manifest yaml")
    p_plan.add_argument("--out", default="plan.json", help="Output plan path")
    p_plan.set_defaults(fn=cmd_plan)

    p_export = sub.add_parser("export", help="Export a plan into a runnable launcher")
    p_export.add_argument("kind", choices=["torchrun"], help="Exporter kind")
    p_export.add_argument("plan", help="Path to plan.json")
    p_export.add_argument("--out", default="run_torchrun.sh", help="Output script path")
    p_export.add_argument("--serve-py", default="serve.py", help="Entry point script on your side")
    p_export.set_defaults(fn=cmd_export)

    args = p.parse_args()
    return int(args.fn(args))

if __name__ == "__main__":
    raise SystemExit(main())
