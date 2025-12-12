from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json

def export_petals(plan: Dict[str, Any], outdir: Path) -> None:
    (outdir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Petals placement notes")
    lines.append("")
    lines.append("Petals repo: https://github.com/bigscience-workshop/petals")
    lines.append("")
    lines.append("This exporter writes scaffolds and placement notes only.")
    lines.append("Install and run Petals separately, then map layer ranges to the stage placement below.")
    lines.append("")
    lines.append(f"Model: {plan['model']['name']} ({plan['model']['dtype']})")
    lines.append(f"Stages: {len(plan['pipeline_stages'])}")
    lines.append(f"Target TPS: {plan['policy'].get('target_tps')}")
    lines.append("")
    lines.append("## Stage placement")
    for st in plan["pipeline_stages"]:
        devices = ", ".join([f"{p['node']}:cuda:{p['gpu']}" for p in st["placement"]])
        lines.append(f"Stage {st['stage']}: {devices}")
    lines.append("")
    lines.append("Bandwidth guidance:")
    lines.append("- Petals performance is bandwidth-sensitive.")
    lines.append("- Prefer intra-node shards and minimize cross-node boundaries.")
    lines.append("")
    (outdir / "petals_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    run = "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "echo "Petals scaffolds written:"",
        "echo "  - plan.json"",
        "echo "  - petals_notes.md"",
        "echo",
        "echo "Example (edit for your model):"",
        "echo "python -m venv .venv && source .venv/bin/activate"",
        "echo "pip install -U petals"",
        "echo",
        "echo "# Serve a shard (choose a layer range):"",
        "echo "# python -m petals.cli.run_server --model <model_name> --device cuda --num_blocks <N> --block_idx <i>"",
        "",
    ])
    p = outdir / "run_petals.sh"
    p.write_text(run, encoding="utf-8")
    p.chmod(0o755)
