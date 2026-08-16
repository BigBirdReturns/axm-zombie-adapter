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
| `name` | string | yes | Model identifier; used for coarse size estimation when no size is declared |
| `dtype` | string | yes | Weight precision: `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `q8`, `int4`, `q4` |
| `kv_cache` | string | no (default: `dtype`) | KV-cache precision |
| `params` | integer | no | Total parameter count. For a mixture-of-experts model this is the **total**, not per-expert |
| `bytes` | integer | no (**yes, to place**) | Exact on-disk weight size in bytes. Highest precedence |
| `sha256` | string | no (default null) | Content identity of the measured model object, echoed into the plan as `model_object_sha256`. Declaring it is what lets a reader tell a measured object from an unverified declaration |

`bytes` and `params` must each be a **finite, positive, whole number** when
present. A reader must refuse zero, negative, fractional, non-finite, boolean,
and non-numeric declarations rather than coerce them. This is not pedantry: a
zero or negative size satisfies every feasibility comparison, so it does not
merely mis-size a plan, it authorizes any placement at all.

**Presence and value are separate facts.** An *absent* `bytes` or `params` key
means no size was declared at that level and the reader falls through to the
next source. A *present* key whose value is `null` is a size declaration that
is not a size, and a reader must **refuse** it — it must not be treated as
absent. Conflating the two lets an explicitly empty declaration fall through to
the name heuristic, so a manifest that declared its size is authorized by a
guess and the plan reports `legacy_name_heuristic` as though the field had
never been written. (`sha256` is the deliberate exception: absent and `null`
both mean "no identity was declared" and both emit `model_object_sha256:
null`, so neither can be mistaken for a measurement.)

**Declared sizes must be exactly representable by every conforming reader.**
`bytes` and `params` must lie within ±(2⁵³−1) = ±9007199254740991. A reader
must refuse a larger magnitude rather than round it. Plans are exchanged as
JSON and read by implementations whose only numeric type is an IEEE-754
double — `9007199254740993` is read back as `9007199254740992` — and a rounded
byte count published as `model_size_bytes` is a number nobody asserted
standing as the size that authorized a placement. Refusing is the only option
that keeps the two readable as the same fact.

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

### Placement evidence (`model`, continued)

The fields above establish a model's declared **size and identity syntax**.
None of them establishes that a specific measured object can be loaded and
executed on specific devices. The fields below are that evidence, and they are
what separates a plan that is *runnable* from one that is merely
*representable*.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `weights` | object | **yes, to place** | The three byte quantities, below |
| `identity` | object | **yes, to place** | Verified object identity binding, below |
| `residence` | list | **yes, to place** | Where this object is verifiably resident |
| `engines` | list of string | no (default `[]`) | Engines able to execute this model. Empty means unconstrained |
| `sharding_strategy` | string | no (default `"pipeline"`) | How weights are decomposed across stages |

#### `weights` — three quantities, three meanings

| Field | Type | Meaning |
|---|---|---|
| `checkpoint_set_bytes` | integer | The complete on-disk checkpoint set. **What object custody binds.** |
| `tensor_payload_bytes` | integer | The tensor payload. **What occupies device memory, and what is placed.** |
| `container_overhead_bytes` | integer | Non-tensor container and header bytes; the difference between the two |

All three are required together and must close:
`tensor_payload_bytes + container_overhead_bytes = checkpoint_set_bytes`. A
reader must **refuse** a partial or non-closing accounting rather than infer
the missing quantity.

Each is normalized by the same rules as `bytes` above: finite, positive, whole,
and inside the exact integer range. `model.bytes` is the size that authorizes
the placement, so when `weights` is declared `model.bytes` must equal
`tensor_payload_bytes`; a reader must refuse otherwise.

This is not bookkeeping pedantry. A real measurement of Kimi-K3 is 1560936091448
checkpoint-set bytes, 1560860324864 tensor-payload bytes, and 75766584 bytes of
safetensors headers across 96 shards. The headers are parsed on the host and
never occupy device memory, while custody was computed over the whole set.
Calling either total "model bytes" produces one number standing for two facts,
and the layer below then resolves the ambiguity by guessing.

#### `identity` — a declaration is not custody

| Field | Type | Meaning |
|---|---|---|
| `identity_scheme` | string | Which scheme produced the digest, and therefore what it binds |
| `identity_digest` | string | 64 hexadecimal characters |
| `identity_source` | string | Where the verification came from: a receipt, a committed artifact, a coordinate |
| `identity_state` | string | `VERIFIED` authorizes a placement. Anything else does not. |
| `verified_at` | string | When the verification happened |
| `validator` | string | What performed it |

`model.sha256` is a *syntactically valid declaration*: it says a digest was
written down. It does not say anyone computed one, under what scheme, over what
material, or when. **Placement authority requires an `identity` binding whose
`identity_state` is `VERIFIED`**, with every field above present and non-empty.
A binding in any other state — or absent entirely — must produce a typed
`UNVERIFIED_MODEL_IDENTITY` refusal. A digest that is merely declared cannot
authorize putting bytes on a device.

If both `model.sha256` and `identity.identity_digest` are present they must be
equal; two identities for one object is not a precedence question, it means the
manifest names two objects.

A reader does **not** adjudicate which schemes count as custody — that is a
policy decision belonging to whoever consumes the plan. What a conforming
reader guarantees is that the scheme travels into the plan, so the decision is
possible at all. A digest under `example/ascii-label-object@1` and one under a
real custody scheme are both structurally valid and are not interchangeable,
and the artifact says which is which.

#### `residence` — per node, per object, per moment

Each residence record: `node` (string, required), `verified` (bool, default
false), `identity_digest` (string, optional), `freshness` (string, default
`CURRENT`), `path`, `verified_at`, `validator` (all optional).

A record whose `identity_digest` names a different object is **refused**, not
ignored — silently skipping it degrades to "no residence declared" and hides a
manifest that is actively wrong. A verified record whose `freshness` is
anything but `CURRENT` is refused as stale: where the bytes were when they were
last checked is not a statement that they are there now.

#### `runtimes` — declared is not measured

A node's `runtimes` entries may be objects `{engine, compatibility}` or bare
strings. A bare string normalizes to `compatibility: "UNMEASURED"`, and only
`MEASURED` runtimes can support a placement. Naming software is not observing
it run; treating `UNMEASURED` as compatible authorizes a placement against a
runtime nobody has ever exercised.

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
| `cluster` | object | `name`; `nodes`: sorted list of node ids; `gpus`: flattened list of seat records (`node`, `host`, `gpu` [the index], `name`, `vram_gb`, `mem_bw_gbps`, `accelerator_uuid`) |
| `policy` | object | The manifest policy after defaulting, echoed verbatim |
| `pipeline_stages` | list | Ordered stages; each `{stage: int, placement: [{node: string, gpu: int}]}` |
| `placement` | object | Whether this plan is runnable, and the proof if it is — see below |
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
- `estimated_model_gb` is `model_size_bytes ÷ 1073741824` rounded to two
  decimals, **half away from zero**, and the rounding is defined on the exact
  value rather than on any host's floating-point `round`. Host defaults
  disagree at a tie — Python's `round` is half-to-even and JavaScript's
  `Math.round` is half-up, so 1207959552 B is 1.12 GB in one and 1.13 GB in
  the other — and determinism is a property of the format, not of the writer.
  The exact form is: multiply the byte count by 100, divide by 1073741824 with
  remainder, and increment the quotient when twice the remainder is at least
  the divisor. `model_size_bytes` remains the authoritative size;
  `estimated_model_gb` is the rounded presentation of it.
- A plan discloses **how the size that authorized its placement was obtained**.
  `model_size_source` is one of `manifest_bytes`,
  `manifest_params_and_quantization`, or `legacy_name_heuristic`, and
  `model_size_bytes` is the exact value used before rounding to
  `estimated_model_gb`. A plan sized by `legacy_name_heuristic` remains useful
  for compatibility and previews, but it must not be read as carrying the same
  evidence status as a measured model object. `model_object_sha256` is null
  unless the manifest declared the identity of the object that was measured; a
  declared `bytes` with no `sha256` is an unverified claim, not a measurement.

### `placement` — runnable, or merely representable

A plan says which of two things it is, in a field, because the defect this
schema exists to close is that the two used to be the same artifact.

| Field | Type | Meaning |
|---|---|---|
| `state` | string | `RUNNABLE` or `REPRESENTABLE_NOT_RUNNABLE` |
| `missing_predicates` | list of string | Which placement predicates the manifest never claimed. Empty when runnable. |
| `proof` | object or null | The placement proof, below. Null when not runnable. |

A writer classifies as follows, and the three cases are exhaustive:

- **Contradicted evidence** — a duplicate physical accelerator, an unverified
  identity, stale or mismatched residence, an unmeasured runtime, an
  unsupported sharding strategy, a non-closing weight accounting, or a stage or
  device that cannot hold what it was assigned — is a typed **refusal**. No
  plan is emitted.
- **Complete evidence that proves out** is `RUNNABLE`, with `proof` present.
- **Absent evidence** is `REPRESENTABLE_NOT_RUNNABLE`. The topology is real and
  the plan is a useful description of it; nothing in it establishes that a
  specific measured object can be loaded and executed there, and
  `missing_predicates` names exactly what was never claimed:
  `verified_object_identity`, `declared_weight_accounting`,
  `verified_residence`, `measured_runtime`.

Placement is *attempted* when the manifest declares object-level evidence —
`identity`, `residence`, or `weights`. Topology alone is not a placement claim:
a manifest describing hardware and a model name is describing a cluster, and
seat UUIDs and node runtimes are hardware description.

A writer that cannot observe residence, runtime compatibility, or physical
execution — the published browser planner, for instance, which has no view of a
remote node's disk and no way to run an engine — may evaluate an evidence
packet the manifest supplies, and may display hypothetical topology. It must
not infer any predicate from topology, and must not label a plan `RUNNABLE`
without the same evidence the reference implementation requires.

#### `placement.proof`

A plan must carry the evidence that it fits, not merely the assertion.

| Field | Type | Meaning |
|---|---|---|
| `identity` | object | The verified identity binding that authorized the placement, carried whole |
| `sharding_strategy` | string | The decomposition this proof is about |
| `weights` | object | All three byte quantities, unchanged from the manifest |
| `placed_bytes_are` | string | `"tensor_payload_bytes"` — which quantity was apportioned |
| `custody_binds` | string | `"checkpoint_set_bytes"` — which quantity identity binds |
| `assigned_bytes_total` | integer | Sum of all stage assignments; equals `tensor_payload_bytes` |
| `stages` | list | Per stage: `stage`, `assigned_bytes`, `usable_bytes`, and `devices` |
| `diagnostics` | object | `aggregate_usable_bytes`, `aggregate_is_not_an_admission_criterion` |

Each device record: `node`, `gpu`, `accelerator_uuid`, `assigned_bytes`,
`usable_bytes`, `engine`, `residence_verified`.

`placed_bytes_are` and `custody_binds` are in the artifact rather than in this
document alone, so a reader never has to know which byte total a given field
meant.

**Independent VRAM is never summed into a pooled address space to admit a
placement.** Aggregate capacity across devices is at best a necessary condition
and never a sufficient one, because independent devices do not share an address
space. A conforming writer must therefore:

1. assign exact tensor-payload bytes to every stage, and inside each stage to
   every device, so that the assignments sum to `tensor_payload_bytes` with no
   byte lost or invented by rounding;
2. admit a stage only when that stage's **own** seats can hold their **own**
   assigned bytes;
3. count a seat's capacity as zero unless the object is verifiably and
   currently resident on its node **and** that node offers a measured engine
   able to execute the model — memory that cannot be loaded or executed is not
   placement capacity;
4. subtract `reserve_gb_per_gpu` **per device, clamped at zero**, never as
   `reserve × device_count` from a pooled total, which lets a small device lend
   its shortfall to a large one.

`aggregate_usable_bytes` is reported for diagnosis only, and the plan says so in
the artifact itself.

A writer may apply an aggregate screen *only* to refuse, and only where no
proof is possible: when no placement evidence was supplied, a model larger than
every device put together certainly does not fit, and saying so is more useful
than emitting a sketch. Such a screen may never admit anything, and a plan that
survives it is `REPRESENTABLE_NOT_RUNNABLE`, never `RUNNABLE`.

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

Codes: `WEIGHT_ACCOUNTING_MISSING`, `WEIGHT_ACCOUNTING_INCONSISTENT`,
`WEIGHT_ACCOUNTING_CONFLICT`, `UNVERIFIED_MODEL_IDENTITY`,
`SEAT_IDENTITY_CONFLICT`, `RESIDENCE_UNVERIFIED`, `RESIDENCE_STALE`,
`RESIDENCE_IDENTITY_MISMATCH`, `NO_SUPPORTED_ENGINE`, `RUNTIME_UNMEASURED`,
`UNSUPPORTED_SHARDING_STRATEGY`, `STAGE_DOES_NOT_FIT`, `DEVICE_DOES_NOT_FIT`,
`NO_ADMISSIBLE_SEAT`.

The reference CLI writes the refusal to `--out` and exits `2`.

Refusals raised by the size-and-identity layer — an unresolvable model size, a
contradictory pair of size claims, a malformed `sha256` — carry no code. That
layer answers a different question (*what did the manifest say the object is?*)
than the placement layer (*can this object run here?*), and it predates typed
codes. A reader distinguishes them by the presence of `code`.

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

The multiplier and the parameter token must be delimited by **the same law**.
Both end at any character that is not a letter or digit — `_` included. A
reader that ends the multiplier at a word boundary in the regex-`\b` sense (in
which `_` is a word character) but ends the parameter token at a
letter-or-digit boundary makes `mixtral-8x7b_model` a mixture for the refusal
and a plain 7B model for the heuristic. The heuristic wins that disagreement,
and the result is a 13 GB estimate for a ~46.7B mixture, placed. Where one
pattern sees a boundary the other must see one too.

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
