# SPEC.md — Artifact Schemas, Version 1

This document defines the two artifacts that are the durable interface of this
project: the **cluster manifest** (input) and the **plan** (output). Per
DURABILITY.md, these artifacts are the API; the Python code is a reference
implementation. This spec is written to be sufficient for reimplementing a
reader or writer without consulting the code.

## Compatibility rules

1. Both artifacts carry a top-level integer `schema_version`. This document
   defines version `1`. A manifest without the field is read as version 1.
2. **Readers must ignore unknown fields.** Unknown fields are how the format
   grows without breaking old readers.
3. **Writers must never repurpose a field.** A field's name, type, and unit are
   permanent within a schema version. Changing any of them requires a new
   `schema_version`.
4. Units are fixed: memory in **GB**, bandwidth in **GB/s** for GPU memory
   (`mem_bw_gbps`) and **Gbit/s** for network links (`gbps`), latency in
   **milliseconds**, throughput in **tokens per second**.
5. The core schema describes physics and topology only. It never names a
   vendor, driver, or execution engine. Engine specifics live in exporter
   output, outside this spec.
6. Manifests may be YAML or JSON; the two are semantically identical. JSON is
   the canonical, dependency-free form. Plans are always JSON.

## Cluster manifest (input)

Top-level keys: `schema_version` (optional, default 1), `cluster`, `model`,
`policy` (all three required).

### `cluster`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | no (default `"zombie"`) | Human label for the cluster |
| `nodes` | list, non-empty | yes | Machines in the cluster |

Each node:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Unique node identifier, referenced by links and placements |
| `host` | string | no (default null) | Address, informational |
| `gpus` | list, non-empty | yes | Accelerators on this node |
| `links` | list | no (default `[]`) | Network links from this node |
| `notes` | string | no (default `""`) | Freeform |

Each GPU:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `index` | integer | yes | Device index on the node (vendor-neutral) |
| `vram_gb` | number | yes | Device memory in GB |
| `name` | string | no (default `"GPU"`) | Human label, informational |
| `mem_bw_gbps` | number | no (default null; required in strict mode) | Memory bandwidth in GB/s |

Each link:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `to` | string | yes | Target node `id` |
| `gbps` | number | yes | Link bandwidth in Gbit/s |
| `type` | string | no | e.g. `ethernet`, informational |

### `model`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | Model identifier; used for coarse size estimation |
| `dtype` | string | yes | Weight precision: `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `q8`, `int4`, `q4` |
| `kv_cache` | string | no (default: `dtype`) | KV-cache precision |

### `policy`

All optional, with defaults:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `target_tps` | number | 18 | Target decode throughput, tokens/s |
| `max_latency_ms` | number | 250 | Latency budget |
| `prefer_bandwidth` | bool | true | Bias placement toward bandwidth locality |
| `allow_cpu_offload` | bool | false | Permit KV/weight spill to host memory |
| `reserve_gb_per_gpu` | number | 4.0 | VRAM held back per GPU for KV cache and overhead |

### Strict mode

A strict reader additionally rejects manifests where any GPU lacks
`mem_bw_gbps`, or where multiple nodes exist but no links are defined.

## Plan (output)

A plan is a JSON object. Version 1 keys:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | integer | `1` |
| `model` | object | `name`, `dtype`, `kv_cache` echoed from manifest; `estimated_model_gb` (number) added by the planner |
| `cluster` | object | `name`; `nodes`: sorted list of node ids; `gpus`: flattened list of GPU records (`node`, `host`, `gpu` [the index], `name`, `vram_gb`, `mem_bw_gbps`) |
| `policy` | object | The manifest policy after defaulting, echoed verbatim |
| `pipeline_stages` | list | Ordered stages; each `{stage: int, placement: [{node: string, gpu: int}]}` |
| `kv_cache_policy` | object | `reserve_gb_per_gpu` (number), `spill_allowed` (bool) |
| `health` | object | `replan_on_gpu_oom` (bool), `replan_on_node_loss` (bool) — replan triggers, not engine config |
| `diagnostics` | object | `stage_boundary_costs`: list of numbers, one per adjacent stage pair (dimensionless penalty; higher is worse; ≥1e6 means a boundary the planner considers saturated); `notes`: string |

### Semantics

- Stage order is pipeline order: stage `i` feeds stage `i+1`.
- A placement's `gpu` is the device index on `node` — no vendor runtime is
  implied. Exporters map indices to whatever device namespace their engine
  uses.
- Plans are **deterministic**: the same manifest must produce a byte-identical
  plan (JSON, 2-space indent, insertion order as specified above). This is
  what makes plans diffable across decades and golden-testable in CI.
- Plans are **disposable**: losing a GPU means editing the manifest and
  re-planning, never hand-editing the plan.

## Reference model-size estimation (informative, not normative)

The v1 reference planner estimates parameter count from the model name
(`70b`, `34b`, `13b`, `7b` substrings; otherwise 30B) and multiplies by
bytes-per-parameter for `dtype` (4 for fp32; 2 for fp16/bf16 and unknown
dtypes; 1 for fp8/int8/q8; 0.5 for int4/q4). Feasibility requires usable VRAM
(total minus per-GPU reserve) ≥ 1.05 × estimated model size. Future planners
may estimate differently; the manifest and plan schemas do not change for it.
