# axm-zombie-adapter

A lightweight control plane for Zombie Compute style consumer GPU clusters.

**Site & live demo:** https://bigbirdreturns.github.io/axm-zombie-adapter/ — the
planner ported to JavaScript, running entirely in your browser: edit a cluster
manifest, click GPUs to kill them, watch the plan reroute. (Deployed
automatically from `docs/` by the `pages` workflow on every push to main.)

This project is an independent adapter. It is not affiliated with Navigator's Log and it does not reproduce any paid dossier content. It implements the integration surface discussed publicly: pipeline parallel inference across multiple GPUs with bandwidth-first planning.

## What it does

It turns "I have some GPUs" into a repeatable, inspectable plan:

1) Ingest a cluster and model manifest (YAML)
2) Generate a placement plan (plan.json)
3) Export launch scaffolds and placement notes for an execution engine (EXO or Petals)

This repo is intentionally not an execution engine. You still install and run EXO or Petals separately.

## Why it is useful

Zombie Compute as a concept assumes the builder will manually decide:
- how to shard a model into pipeline stages
- where stage boundaries should go given link bandwidth
- how much KV cache headroom to reserve
- what to do when a GPU OOMs or a node drops

This adapter makes those decisions explicit and diffable through artifacts:
- `cluster.yaml` captures topology and constraints
- `plan.json` captures placements and policies
- exporters turn the plan into runnable scaffolds and notes

## Quickstart

### 0) Install

```bash
pip install -e .          # zero dependencies; JSON manifests work out of the box
pip install -e '.[yaml]'  # optional: YAML manifest support
```

### 1) Create a manifest

Start with one of these:
- `examples/cluster_4x3090_singlebox.yaml` (or the equivalent `.json`)
- `examples/cluster_4x3090_2nodes_10gbe.yaml`

Manifests may be YAML or JSON; they are semantically identical and produce
byte-identical plans. The format is defined in `SPEC.md`.

### 2) Generate a plan

```bash
axm-zombie plan examples/cluster_4x3090_singlebox.yaml --out plan.json
```

### 3) Export launch scaffolds

llama.cpp RPC (recommended — actively maintained upstream):
```bash
axm-zombie export llamacpp plan.json --outdir out_llamacpp
```

EXO:
```bash
axm-zombie export exo plan.json --outdir out_exo
```

Petals:
```bash
axm-zombie export petals plan.json --outdir out_petals
```

torchrun:
```bash
axm-zombie export torchrun plan.json --outdir out_torchrun
```

### 4) When hardware dies

A dead GPU or node is a manifest edit plus a re-plan, never a hand-edited
plan:

```bash
# GPU 3 on node-a died:
axm-zombie replan examples/cluster_4x3090_singlebox.yaml \
  --lose node-a:3 --out plan.json --manifest-out cluster_degraded.json

# Whole node gone:
axm-zombie replan cluster.yaml --lose node-b --out plan.json
```

If the model no longer fits the surviving hardware, the planner says so
instead of emitting a plan that will OOM.

### Verify the toolchain

```bash
python tests/golden_check.py
```

This is the resurrection test: no network, no third-party dependencies. It
builds a plan from the JSON example, checks it is byte-identical to the
committed golden file, and runs every exporter.

## What authorizes a placement

A plan is a claim that a model fits particular hardware, so it has to carry the
evidence rather than the assertion.

- **Exact size, or no plan.** `model.bytes` and `model.sha256` are required to
  place. The planner does not infer a size from the model name, because a wrong
  guess is not surfaced as an error — it is surfaced as a confident plan that
  OOMs on contact with hardware.
- **Seats are not boards.** A GPU entry describes a *seat*; `uuid` identifies
  the *physical accelerator* in it. Two seats claiming one board are refused,
  because their memory is not independent and summing it invents VRAM.
- **Independent VRAM is never pooled.** Every weight byte is assigned to a named
  device, and each stage is proved against its own devices. Memory on a node
  that lacks the weights or lacks an engine is not capacity, whatever the
  cluster total says. The arithmetic ships inside the plan as
  `placement_proof`.
- **Refusals are artifacts.** When a placement cannot be proved, the CLI writes
  a typed refusal where the plan would have gone and exits `2`:

```bash
axm-zombie plan examples/estate_octo_w01_kimi_k3.json --out refusal.json
# REFUSED [NO_SUPPORTED_ENGINE] ...
```

`artifacts/estate-octo-w01-kimi-k3.refusal.json` is a committed refusal for a
real two-accelerator workstation asked to hold a 1453.7 GiB model.

## Model and cluster notes

- The planner is heuristic in how it *shapes* stages. It is not heuristic about
  whether they fit.
- The planner stays conservative and it favors bandwidth locality.
- For multi-node rigs, stage boundaries across nodes are expensive unless you have real interconnect (10GbE or better).
- KV cache headroom matters. The planner reserves a fixed default per GPU, and you should tune it per model and batch.

## Longevity

- `SPEC.md` defines the manifest and plan schemas (version 1). The artifacts
  are the durable interface; the code is a reference implementation.
- `DURABILITY.md` is the endstate analysis and 30-year durability plan.
- `recipes/` pins last-known-good driver stacks per GPU architecture before
  vendors drop them.

## References

- llama.cpp: https://github.com/ggml-org/llama.cpp
- EXO: https://github.com/exo-explore/exo
- Petals: https://github.com/bigscience-workshop/petals

Both upstream engines are cited as integration targets, not dependencies;
nothing in this repo requires them to exist.

## License

MIT
