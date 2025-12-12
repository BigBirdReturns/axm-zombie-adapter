# axm-zombie-adapter

AXM Zombie Adapter is a small control-plane module that compiles a cluster manifest into an executable inference plan for multi-GPU, multi-node "zombie compute" rigs.

It focuses on the gap that shows up immediately after "we can chain GPUs": arbitration, placement, and repeatable execution under bandwidth and VRAM constraints.

## What it does

- Loads a `cluster.yaml` manifest
- Produces a deterministic `plan.json`
- Emits a `run_torchrun.sh` launcher script (first supported exporter)
- Includes a simple monitor loop scaffold for tokens/sec and error-triggered replans (optional)

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Build a plan

```bash
axm-zombie plan examples/cluster_4x3090.yaml --out plan.json
```

### 3) Export a launcher

```bash
axm-zombie export torchrun plan.json --out run_torchrun.sh
bash run_torchrun.sh
```

## Manifest

See `examples/cluster_4x3090.yaml`.

## Notes

This repo intentionally avoids coupling to any proprietary package. If you later obtain the Zombie Compute Kit dossier, you can map its assumptions (pipeline strategy, preferred stack, benchmarking targets) into:

- `axm_zombie/cost.py`
- `axm_zombie/planner.py`
- `axm_zombie/exporter/*`

## License

MIT
