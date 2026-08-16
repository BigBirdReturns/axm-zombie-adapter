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
| `name` | string | yes | Model identifier; used for coarse size estimation when no size is declared |
| `dtype` | string | yes | Weight precision: `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `q8`, `int4`, `q4` |
| `kv_cache` | string | no (default: `dtype`) | KV-cache precision |
| `params` | integer | no | Total parameter count. For a mixture-of-experts model this is the **total**, not per-expert |
| `bytes` | integer | no | Exact on-disk weight size in bytes. Highest precedence |
| `sha256` | string | no (default null) | Content identity of the measured model object, echoed into the plan as `model_object_sha256`. Declaring it is what lets a reader tell a measured object from an unverified declaration |

`bytes` and `params` must each be a **finite, positive, whole number** when
present. A reader must refuse zero, negative, fractional, non-finite, boolean,
and non-numeric declarations rather than coerce them. This is not pedantry: a
zero or negative size satisfies every feasibility comparison, so it does not
merely mis-size a plan, it authorizes any placement at all.

`sha256`, when present, must be exactly 64 hexadecimal characters. A reader
must refuse anything else rather than emit it as `model_object_sha256`, since
an identity that cannot be checked is not an identity.

Size precedence is `bytes`, then `params` × bytes-per-parameter, then inference
from `name`. A declared size always wins over the name.

If **both** `bytes` and `params` are declared and they imply sizes differing by
a factor of 10 or more, a reader must **refuse** rather than apply precedence.
Quantization and packaging move on-disk size by small factors, so smaller
disagreements are tolerated; an order of magnitude means one field is stale, and
silently preferring `bytes` would emit a confident plan built on a stale claim.

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
| `model` | object | `name`, `dtype`, `kv_cache` echoed from manifest; `estimated_model_gb` (number), `model_size_bytes` (integer), `model_size_source` (string), `model_object_sha256` (string or null) added by the planner |
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
- A plan discloses **how the size that authorized its placement was obtained**.
  `model_size_source` is one of `manifest_bytes`,
  `manifest_params_and_quantization`, or `legacy_name_heuristic`, and
  `model_size_bytes` is the exact value used before rounding to
  `estimated_model_gb`. A plan sized by `legacy_name_heuristic` remains useful
  for compatibility and previews, but it must not be read as carrying the same
  evidence status as a measured model object. `model_object_sha256` is null
  unless the manifest declared the identity of the object that was measured; a
  declared `bytes` with no `sha256` is an unverified claim, not a measurement.

## Reference model-size estimation (informative, not normative)

The v1 reference planner resolves model size in this order, recording the
matching `model_size_source` in the plan:

1. `model.bytes`, used exactly (`manifest_bytes`);
2. `model.params` × bytes-per-parameter for `dtype`
   (`manifest_params_and_quantization`);
3. inference from `model.name` via the `7b`, `13b`, `34b`, `70b` parameter
   tokens (`legacy_name_heuristic`).

A parameter token is a **complete** number followed by `b`, delimited so that
it is neither preceded by a digit or decimal point nor followed by another
letter or digit. `model-170b` therefore does not match `70b`, and `model-107b`
does not match `7b`; both refuse. A name carrying a token outside the supported
set, or carrying more than one distinct token, also refuses — the planner does
not round a name to the nearest size it happens to know. Because tokens are
matched whole, match ordering no longer affects the result.

Bytes-per-parameter is 4 for fp32; 2 for fp16/bf16 and unknown dtypes; 1 for
fp8/int8/q8; 0.5 for int4/q4. Feasibility requires usable VRAM (total minus
per-GPU reserve) ≥ 1.05 × model size.

If none of the three resolve, the planner **raises rather than assuming a
default**. It also refuses a name carrying a mixture-of-experts multiplier
(`8x7b`, `4x22b`), because a bare parameter substring understates such a model
by roughly its expert count. Both refusals name the model and ask for `params`
or `bytes`.

This refusal is deliberate. A wrong size estimate is not surfaced as an error;
it is surfaced as a confident placement plan that is byte-identical in shape to
a correct one and OOMs on contact with real hardware. Earlier versions defaulted
to 30B, which silently produced such plans — a 1.42 TiB model estimated at
27.94 GB and placed across four 24 GB GPUs.

The same reasoning applies to two contradictory declarations. When `bytes` and
`params` disagree by an order of magnitude the planner refuses instead of
applying precedence, because a stale `bytes` beside honest `params` recreates
the identical false plan through a different input channel. The refusal names
both claims and the size each implies. Refusals that reach the feasibility check
also name the resolved size and its source, so an operator can see whether the
number that refused the placement was measured or guessed.

Future planners may estimate differently; the manifest and plan schemas do not
change for it.
