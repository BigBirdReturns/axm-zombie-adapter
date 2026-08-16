import argparse
import json
import sys
from pathlib import Path

from .manifest import load_manifest
from .placement import PlacementRefusal
from .planner import build_plan
from .replan import degrade_manifest
from .export.exo import export_exo
from .export.llamacpp import export_llamacpp
from .export.petals import export_petals
from .export.torchrun import export_torchrun

EXPORTERS = {
    "exo": export_exo,
    "llamacpp": export_llamacpp,
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

    p_rep = sub.add_parser(
        "replan",
        help="Drop lost GPUs/nodes from a manifest and generate a new plan",
    )
    p_rep.add_argument("manifest", help="Path to cluster manifest (.yaml or .json)")
    p_rep.add_argument(
        "--lose",
        action="append",
        required=True,
        metavar="NODE[:GPU]",
        help="Lost hardware: a node id, or node-id:gpu-index. Repeatable.",
    )
    p_rep.add_argument("--out", default="plan.json", help="Output plan.json path")
    p_rep.add_argument(
        "--manifest-out",
        default=None,
        help="Also write the degraded manifest as JSON (becomes your new cluster manifest)",
    )
    p_rep.add_argument("--strict", action="store_true", help="Fail on missing optional fields")

    p_exp = sub.add_parser("export", help="Export scaffolds and notes from plan.json")
    p_exp.add_argument("target", choices=sorted(EXPORTERS), help="Exporter target")
    p_exp.add_argument("plan", help="Path to plan.json")
    p_exp.add_argument("--outdir", default="out", help="Output directory")

    args = parser.parse_args()

    def emit(manifest, out, suffix=""):
        """Write a plan, or write the refusal that replaced it.

        A refusal is a real artifact, not an absent one. Writing it to the same
        path a plan would have taken means the reason a placement did not
        happen is recorded and diffable, instead of scrolling past in a
        traceback.
        """
        out_path = Path(out)
        try:
            plan = build_plan(manifest)
        except PlacementRefusal as refusal:
            out_path.write_text(
                json.dumps(refusal.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
            print(f"REFUSED [{refusal.code}] {refusal.message}", file=sys.stderr)
            print(f"Wrote refusal {out_path.resolve()}", file=sys.stderr)
            raise SystemExit(2)
        out_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path.resolve()}{suffix}")
        return plan

    if args.cmd == "plan":
        emit(load_manifest(args.manifest, strict=args.strict), args.out)
        return

    if args.cmd == "replan":
        manifest = load_manifest(args.manifest, strict=args.strict)
        degraded = degrade_manifest(manifest, args.lose)
        emit(degraded, args.out, suffix=f" (lost: {', '.join(args.lose)})")
        if args.manifest_out:
            mo = Path(args.manifest_out)
            mo.write_text(json.dumps(degraded, indent=2), encoding="utf-8")
            print(f"Wrote degraded manifest {mo.resolve()}")
        return

    if args.cmd == "export":
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        EXPORTERS[args.target](plan, outdir)
        print(f"Wrote exporter artifacts to {outdir.resolve()}")
        return


# Without this, `python -m axm_zombie.cli ...` exits 0 having done nothing. The
# console script requires an install; this path does not, which is the same
# reason the resurrection test exists: the tool has to work from a checkout
# alone.
if __name__ == "__main__":
    main()
