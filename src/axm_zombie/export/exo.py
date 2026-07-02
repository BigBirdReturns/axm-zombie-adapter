from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json

def export_exo(plan: Dict[str, Any], outdir: Path) -> None:
    (outdir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    lines = []
    lines.append("# EXO placement notes")
    lines.append("")
    lines.append("EXO repo: https://github.com/exo-explore/exo")
    lines.append("")
    lines.append("This exporter writes scaffolds and placement notes only.")
    lines.append("Install and run EXO separately, then map shards to the stage placement below.")
    lines.append("")
    lines.append(f"Model: {plan['model']['name']} ({plan['model']['dtype']})")
    lines.append(f"Stages: {len(plan['pipeline_stages'])}")
    lines.append(f"Target TPS: {plan['policy'].get('target_tps')}")
    lines.append("")
    lines.append("## Stage placement")
    for st in plan["pipeline_stages"]:
        devices = ", ".join([f"{p['node']}:gpu:{p['gpu']}" for p in st["placement"]])
        lines.append(f"Stage {st['stage']}: {devices}")
    lines.append("")
    lines.append("Bandwidth guidance:")
    lines.append("- Prefer intra-node stage boundaries first.")
    lines.append("- Keep cross-node boundaries minimal unless you have 10GbE or better.")
    lines.append("")
    (outdir / "exo_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    run = "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'echo "EXO scaffolds written:"',
        'echo "  - plan.json"',
        'echo "  - exo_notes.md"',
        "echo",
        'echo "Next:"',
        'echo "1) Install EXO on each node (see https://github.com/exo-explore/exo)"',
        'echo "2) Ensure node discovery works"',
        'echo "3) Apply stage placement from exo_notes.md in your EXO shard config"',
        "",
    ])
    p = outdir / "run_node.sh"
    p.write_text(run, encoding="utf-8")
    p.chmod(0o755)
