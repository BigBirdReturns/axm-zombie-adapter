# axm-zombie-adapter

A lightweight control plane for Zombie Compute style consumer GPU clusters.

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

### 1) Create a manifest

Start with one of these:
- `examples/cluster_4x3090_singlebox.yaml`
- `examples/cluster_4x3090_2nodes_10gbe.yaml`

### 2) Generate a plan

```bash
axm-zombie plan examples/cluster_4x3090_singlebox.yaml --out plan.json
```

### 3) Export launch scaffolds

EXO:
```bash
axm-zombie export exo plan.json --outdir out_exo
```

Petals:
```bash
axm-zombie export petals plan.json --outdir out_petals
```

## Model and cluster notes

- The planner is heuristic on purpose. It stays conservative and it favors bandwidth locality.
- For multi-node rigs, stage boundaries across nodes are expensive unless you have real interconnect (10GbE or better).
- KV cache headroom matters. The planner reserves a fixed default per GPU, and you should tune it per model and batch.

## References

- EXO: https://github.com/exo-explore/exo
- Petals: https://github.com/bigscience-workshop/petals

## License

MIT
