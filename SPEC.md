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
| `gpus` | list, non-empty | yes | Accelerator **seats** on this node |
| `links` | list | no (default `[]`) | Network links from this node |
| `notes` | string | no (default `""`) | Freeform |
| `runtimes` | list of string | no (default `[]`) | Execution engines installed on this node. A node offering none cannot execute a model however much VRAM it has |

Each GPU entry describes an **accelerator seat**, not a board:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `index` | integer | yes | Device index on the node (vendor-neutral) |
| `vram_gb` | number | yes | Device memory in GB |
| `name` | string | no (default `"GPU"`) | Human label, informational |
| `mem_bw_gbps` | number | no (default null; required in strict mode) | Memory bandwidth in GB/s |
| `uuid` | string | no (default null) | Identity of the **physical accelerator** currently in this seat |

`index`, `vram_gb`, `name` and `mem_bw_gbps` are *seat capability*. They do not
identify a board: two substantially identical cards match on every one of them.
`uuid` is *board identity*, and when it is declared it must be unique across the
cluster. Two seats declaring the same `uuid` describe one board twice, so their
memory is not independent; a reader must **refuse** rather than sum it.

Each link:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `to` | string | yes | Target node `id` |
| `gbps` | number | yes | Link bandwidth in Gbit/s |
| `type` | string | no | e.g. `ethernet`, informational |

### `model`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | Model identifier, informational |
| `dtype` | string | yes | Weight precision: `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `q8`, `int4`, `q4` |
| `kv_cache` | string | no (default: `dtype`) | KV-cache precision |
| `bytes` | integer | **yes, to place** | Exact on-disk weight size in bytes |
| `sha256` | string | **yes, to place** | Content identity of the measured model object, 64 hex characters |
| `engines` | list of string | no (default `[]`) | Engines able to execute this model. Empty means unconstrained |
| `residence` | list | no (default `[]`) | Where this object is verifiably resident |

Each residence record: `node` (string, required), `verified` (bool, default
false), `sha256` (string, optional), `path` (string, optional).

**Placement consumes exact bytes and an exact identity, or it refuses.** A size
inferred from a model name is a guess, and a guess that authorizes a placement
is indistinguishable in the emitted plan from a measurement that does. `bytes`
must be a finite, positive, whole number; a reader must refuse zero, negative,
fractional, non-finite, boolean, non-numeric, and explicitly-null declarations
rather than coerce them, because a zero or negative size satisfies every
feasibility comparison and so authorizes any placement at all. `sha256` must be
exactly 64 hexadecimal characters: an identity that cannot be checked is not an
identity, and residence is verified against it.

A residence record whose `sha256` names a different object is **refused**, not
ignored — silently skipping it degrades to "no residence declared" and hides a
manifest that is actively wrong.

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
| `model` | object | `name`, `dtype`, `kv_cache` echoed from manifest; `estimated_model_gb` (number), `model_size_bytes` (integer), `model_object_sha256` (string) added by the planner |
| `cluster` | object | `name`; `nodes`: sorted list of node ids; `gpus`: flattened list of seat records (`node`, `host`, `gpu` [the index], `name`, `vram_gb`, `mem_bw_gbps`, `accelerator_uuid`) |
| `policy` | object | The manifest policy after defaulting, echoed verbatim |
| `pipeline_stages` | list | Ordered stages; each `{stage: int, placement: [{node: string, gpu: int}]}` |
| `placement_proof` | object | Why this placement is believed to fit — see below |
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

### `placement_proof`

A plan must carry the evidence that it fits, not merely the assertion.

| Field | Type | Meaning |
|---|---|---|
| `model_object_sha256` | string | The object that was placed |
| `weight_bytes` | integer | Its exact size |
| `assigned_bytes_total` | integer | Sum of all stage assignments; equals `weight_bytes` |
| `stages` | list | Per stage: `stage`, `assigned_bytes`, `usable_bytes`, and `devices` |
| `diagnostics` | object | `aggregate_usable_bytes`, `aggregate_is_not_an_admission_criterion` |

Each device record: `node`, `gpu`, `accelerator_uuid`, `assigned_bytes`,
`usable_bytes`, `engine`, `residence_verified`.

**Independent VRAM is never summed into a pooled address space to admit a
placement.** Aggregate capacity across devices is at best a necessary condition
and never a sufficient one, because independent devices do not share an address
space. A conforming writer must therefore:

1. assign exact weight bytes to every stage, and inside each stage to every
   device, so that the assignments sum to `weight_bytes` with no byte lost or
   invented by rounding;
2. admit a stage only when that stage's **own** seats can hold their **own**
   assigned bytes;
3. count a seat's capacity as zero unless the object is verifiably resident on
   its node **and** that node offers an engine able to execute the model —
   memory that cannot be loaded or executed is not placement capacity;
4. subtract `reserve_gb_per_gpu` **per device, clamped at zero**, never as
   `reserve × device_count` from a pooled total, which lets a small device lend
   its shortfall to a large one.

`aggregate_usable_bytes` is reported for diagnosis only, and the plan says so in
the artifact itself.

## Refusal (output)

When a placement cannot be proved, the writer emits a refusal. A refusal is a
real artifact and is written where the plan would have been, because the reason
a placement did not happen is worth recording and diffing.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | integer | `1` |
| `result` | string | `"refusal"` |
| `code` | string | Machine-readable reason; readers branch on this, never on prose |
| `message` | string | Human-readable explanation |
| `detail` | object | Code-specific facts (stage, node, gpu, byte counts, deficits) |

Codes: `WEIGHT_BYTES_MISSING`, `WEIGHT_BYTES_INVALID`, `MODEL_IDENTITY_MISSING`,
`MODEL_IDENTITY_INVALID`, `SEAT_IDENTITY_CONFLICT`, `RESIDENCE_UNVERIFIED`,
`RESIDENCE_IDENTITY_MISMATCH`, `NO_SUPPORTED_ENGINE`, `STAGE_DOES_NOT_FIT`,
`DEVICE_DOES_NOT_FIT`, `NO_ADMISSIBLE_SEAT`.

The reference CLI writes the refusal to `--out` and exits `2`.

## What replaced model-size estimation, and why (informative)

An earlier reference planner estimated parameter count from the model name
(`70b`, `34b`, `13b`, `7b` substrings; **otherwise 30B**), multiplied by
bytes-per-parameter, and admitted the placement when

    sum(vram_gb) - reserve_gb_per_gpu × gpu_count  ≥  estimated_gb × 1.05

Both halves were wrong, and they failed in the same direction.

The estimate was a guess that was never surfaced as uncertainty. It was
surfaced as a placement plan, byte-identical in shape to a correct one. A
1453.7 GiB model whose name matched no token was sized at 27.94 GB by the 30B
default and placed across four 24 GB boards.

The admission test was a claim about an address space that does not exist.
Summing independent devices is a necessary condition at best, and it admitted
three placements that cannot run: one board seated twice (its memory counted
twice), seats on a node without the weights resident, and seats on a node with
no engine able to execute the model. In each case the pooled arithmetic cleared
and the hardware did not.

Both are now refusals. Size is consumed exactly or not at all, and a placement
is admitted only when every stage is proved against its own devices. Feasibility
is no longer a single comparison; it is the `placement_proof` above, and it
ships inside the plan so a reader can check the arithmetic rather than trust it.

Bytes-per-parameter is retained only as a manifest-authoring aid and is not used
to authorize anything: 4 for fp32; 2 for fp16/bf16; 1 for fp8/int8/q8; 0.5 for
int4/q4.
