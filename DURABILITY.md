# DURABILITY.md — Endstates and the 30-Year Plan

This document thinks through where axm-zombie-adapter can end up, what actually
kills salvage-compute software over decades, and the plan for surviving to 2056.

The premise worth stating plainly: consumer and datacenter GPUs almost never die
of physics first. They die of *policy* — driver EOL, CUDA version gates, engine
abandonment, dependency rot. The hardware oligopoly's real moat is not silicon,
it is the depreciation treadmill that declares working silicon obsolete. A
salvage-compute control plane is therefore not primarily a performance problem;
it is a longevity problem. Everything below follows from that.

---

## 1. What this project actually is (and the part that must survive)

The repo has three layers with very different life expectancies:

| Layer | What it is | Life expectancy |
|---|---|---|
| **Artifacts** | `cluster.yaml`, `plan.json` — plain-text descriptions of physics: VRAM in GB, memory bandwidth in GB/s, link speeds, placements | Decades, if versioned |
| **Planner core** | ~200 lines of heuristic Python over those artifacts | Years-to-decades, if dependency-free |
| **Exporters / engines** | Scaffolds for EXO, Petals, torchrun | **Months-to-years. Already dying.** |

The evidence for that last row is already in this repo, at version 0.3.0:

- Both export targets are effectively unmaintained upstream. Petals has been
  largely dormant since ~2023; EXO's public development stalled in 2025. The
  adapter's only two engine integrations went stale faster than the adapter itself.
- The plan producer and plan consumers already disagree about the schema:
  `Plan.to_dict()` emits `"model"` as a string, while `src/axm_zombie/export/exo.py`
  and `petals.py` read `plan["model"]["name"]` and `plan["policy"]` — keys the
  planner never writes. Schema drift arrived in year zero, inside one repo.
- Both package trees (`axm_zombie/` and `src/axm_zombie/`) currently contain
  syntax errors in their shell-script generators (unescaped quotes in
  `export/exo.py`, `export/petals.py`, `exporter/torchrun.py`), and the two
  trees have diverged from each other. `pyproject.toml` installs from `src/`,
  so the root tree is an unbuilt fork living in the same repo.

None of this is fatal — it is the threat model demonstrating itself early, which
is useful. The conclusion it forces: **the artifacts are the project. The code is
a reference implementation. The engines are disposable.** Every durability
decision below protects the layers in that order.

---

## 2. Endstates

Five plausible places this project can be in 2056, from best to worst. The plan
optimizes for E1–E3 and hard-fails only at E5.

### E1 — Reference standard (best case)
The `cluster.yaml` / `plan.json` schemas become the boring, obvious lingua franca
for describing salvage clusters — the thing forks, engines, and rewrites all
import and emit. The Python code matters the way the first HTTP server matters:
historically. Reaching E1 requires a written schema spec that is independent of
the implementation, versioned, and small enough to reimplement in an afternoon.

### E2 — Absorbed
A living execution engine grows its own bandwidth-aware planner and the adapter
dissolves into it. This is a *good* death if the schema survives as that engine's
import format, and a bad one if the knowledge is reabsorbed into engine-internal
config no one can diff. The defense is the same as E1: the spec must be
freestanding so absorption carries the format, not just the ideas.

### E3 — Fork federation
No single successor; instead per-hardware-generation forks (an Ampere-salvage
fork, an MI300-salvage fork, a 2040s-NPU fork) sharing the artifact format.
Acceptable and likely. Requires that vendor-specific anything lives in exporters,
never in the core or the schema, so forks diverge at the edges only.

### E4 — Dormant but resurrectable (the design floor)
No maintainer for a decade, then someone with a pallet of decommissioned
accelerators clones the repo and it *still works*: deps vendored or absent,
tests runnable, schema self-describing, no network required, no dead upstream in
the critical path. This is the minimum acceptable endstate and the one the plan
explicitly engineers for, because every project passes through dormancy whether
or not it plans to.

### E5 — Dead (the failure mode)
Unrunnable and unreadable: broken imports, a schema that only ever existed
implicitly in code, exporters pointing at repos that 404. The repo is currently
*closer to E5 than to E4* — the installed package cannot complete its own
quickstart because the exporters do not parse. Phase 0 exists to fix that.

---

## 3. Threat model — what kills salvage compute over 30 years

**T1. Vendor driver EOL (software kills hardware first).**
The pattern is established and regular: Kepler lost driver support in 2021;
Maxwell, Pascal, and Volta were moved to legacy with the 580-series drivers in
2025; CUDA 13 dropped them entirely. Ampere (the RTX 3090 in `examples/`, whose
GDDR6X is Micron's hottest-running part — the oligopoly names itself in the
BOM) should be assumed legacy by ~2031–2032 on the historical ~11-year cadence.
The GPU still computes; the vendor stack refuses to.
*Countermeasures:* per-architecture **freeze recipes** (last-known-good
driver + CUDA + container image, hashes recorded in-repo); treat open stacks
(Vulkan compute via llama.cpp, NVK/nouveau, tinygrad) as first-class exporter
targets since open drivers have no EOL policy; never let the core or schema
mention CUDA — today the exporters' `cuda:{gpu}` device strings are the single
vendor leak in the artifact chain, and they must become neutral device indices.

**T2. Engine mortality.**
Both current export targets stalled within ~2 years of this repo's creation.
Over 30 years assume *every* engine dies, repeatedly.
*Countermeasures:* exporters are plugins with a stable input (the plan schema)
and zero upstream imports; `plan.json` never names an engine; adding an exporter
for whatever runs in 2040 must require reading only SPEC.md, not the git history.

**T3. Language and dependency rot.**
`requires-python >=3.10` — Python 3.10 EOLs in October 2026, an expiry date in
the metadata already. The sole dependency is PyYAML; over decades even one
dependency is one supply chain.
*Countermeasures:* core must run on stdlib alone — accept JSON manifests
natively so YAML becomes an optional convenience; widen (or better, stop
narrowing) the Python floor; CI against oldest and newest supported Python; the
repo must install and run fully offline from a bare clone.

**T4. Schema drift.**
Already observed (§1). Implicit schemas rot silently; explicit ones rot loudly.
*Countermeasures:* `schema_version` field in both `cluster.yaml` and
`plan.json`, starting at `1`; a SPEC.md defining every field, its unit, and its
compatibility rules ("readers ignore unknown fields; writers never repurpose a
field"); golden-file contract tests — example manifest in, byte-identical plan
out — so producer/consumer drift fails CI instead of failing a user in 2034.

**T5. Hardware physics (the honest mortality).**
What actually wears out on a salvaged 3090-class card: fans (3–7 years),
thermal pads and paste (GDDR6X junction temps make this the known Ampere
weakness), solder fatigue from thermal cycling, PSU capacitors. This is fleet
math, not tragedy: run cards undervolted and cool, hold N+1 spares, and the
bathtub curve is manageable for decades — *if* replacing a dead card is cheap in
the control plane. `plan.json` plus `replan_on_node_loss` is exactly that
cheapness: a dead GPU is a manifest edit and a re-plan, not an architecture
meeting. The durability feature is that plans are disposable and regenerable.
Meanwhile the salvage supply *improves* on a predictable conveyor: each
datacenter refresh cycle pushes the previous generation's accelerators toward
e-waste pricing (Ampere/A100-class through the late 2020s, Hopper/MI300-class in
the early-to-mid 2030s, and so on). The planner's only job is to be ready to
describe whatever falls off the truck: heterogeneous VRAM, mixed vendors, weird
links.

**T6. Model-architecture drift.**
Pipeline stages, tensor-parallel groups, and KV-cache reserves are transformer
assumptions. The models of 2046 may not decompose that way.
*Countermeasures:* keep the schema split along its natural fault line —
**topology** (nodes, GPUs, GB, GB/s, links: physics, durable for 30 years) versus
**strategy** (stages, TP groups, cache policy: fashion, replaceable per era).
Strategy blocks should be versioned sub-objects that a future planner can swap
out wholesale without touching topology.

---

## 4. The 30-year plan

### Phase 0 — Repair the foundation (now, 2026)
The repo must first reach its own design floor (E4). Concretely:

> **Status: executed in 0.4.0.** Root tree deleted (`src/` is canonical, with
> the torchrun exporter ported over and fixed); exporter syntax errors fixed;
> schema frozen as version 1 in SPEC.md with `schema_version` stamped into
> manifests and plans; golden-file contract tests added, including a
> stdlib-only resurrection test (`python tests/golden_check.py`); JSON
> manifests supported natively and PyYAML moved to an optional extra, so the
> core has zero dependencies. One addition beyond the list below: the model
> size estimator learned quantized dtypes (q4/int4/q8/int8/fp8), because the
> flagship 4×3090 examples were infeasible at fp16 — the quickstart now
> completes because the plan is actually placeable, not because the check was
> loosened.

1. Delete the divergent root `axm_zombie/` tree; `src/` layout is canonical
   (fold the root tree's `cost.py`/`types.py` improvements into `src/` first).
2. Fix the quoting syntax errors in `src/axm_zombie/export/exo.py` and
   `petals.py` so the package parses and the quickstart completes.
3. Reconcile the plan schema: make `Plan.to_dict()` and the exporters agree,
   then freeze the result as **schema_version 1** in a new `SPEC.md`.
4. Add golden-file contract tests (`examples/*.yaml` → committed expected
   `plan.json`) and run them in CI on every supported Python version.
5. Accept JSON manifests in `load_manifest` so PyYAML becomes optional; the
   dependency count of the core drops to zero.

### Phase 1 — The Ampere wave (2026–2030)
3090s, P40s, and first A100 cast-offs are the fleet. Priorities:

> **Status: core delivered in 0.5.0.** A llama.cpp RPC exporter (the first
> actively-maintained engine target) generates per-node `rpc-server` worker
> scaffolds and a VRAM-proportional `--tensor-split` host launcher; EXO and
> Petals exporters remain as frozen plugins. The loss loop is automated as
> `axm-zombie replan --lose NODE[:GPU]` — degrade the manifest, re-plan,
> optionally emit the degraded manifest as the new source of truth, and
> refuse honestly when the model no longer fits the survivors. The freeze
> recipe library exists at `recipes/` with the format rules and a Pascal
> template awaiting hardware verification. Still open in this phase: filling
> recipe checksums from a real archive, and wiring a live health signal
> (nvidia-smi/engine logs) to trigger `replan` automatically.
- Replace the dead-engine monoculture: add exporters for whatever is actually
  alive (llama.cpp RPC clusters at minimum), keep EXO/Petals exporters as
  frozen plugins, clearly labeled with their upstreams' status.
- Ship freeze recipes per architecture as drivers go legacy: Pascal now,
  document the pinned driver/CUDA/container triple with hashes.
- Replace the placeholder `Monitor` with the one loop that matters for
  salvage fleets: detect GPU/node loss → edit manifest → re-plan → re-export.
  Automating T5 is the phase's real deliverable.

### Phase 2 — Heterogeneity becomes the norm (2030–2036)
Hopper and MI300-class hardware reaches salvage pricing; Ampere goes
driver-legacy; no fleet is single-vendor anymore.
- The planner's node/GPU model already speaks only GB and GB/s — keep it that
  way and make mixed-vendor, mixed-VRAM plans a tested first-class case
  (uneven stage sizing by VRAM, not by card count).
- Promote open driver stacks from fallback to default recommendation as they
  mature; the freeze-recipe library becomes the compatibility memory of the
  project.
- Schema review #1: if the strategy block no longer describes real workloads,
  version it (`strategy_version: 2`) — topology block stays untouched.

### Phase 3 — The long dormancy (2036–2046)
Plan for zero maintainers. The repo must be self-sufficient:
- Everything needed lives in-tree: SPEC.md, vendored optional deps, freeze
  recipes, examples, golden tests. No load-bearing URL, ever — every upstream
  reference in the repo today (EXO, Petals) should be treated as a citation,
  not a dependency.
- Define the **resurrection test** and make it the repo's front door: one
  command, no network — parse an example manifest, emit a plan, diff against
  the golden file. If a stranger in 2044 runs it and it passes, the project is
  alive regardless of commit dates.
- An annual ritual while anyone is around: run the resurrection test on the
  current Python; fix or re-pin; tag. Ten minutes a year buys a decade.

### Phase 4 — The spec outlives the code (2046–2056)
By now Python-as-we-know-it, transformers, and PCIe are all open questions. The
endstate to have engineered for: SPEC.md plus the golden files are sufficient
for someone to reimplement the planner in the 2050s' language of choice in a
day, against whatever salvaged accelerators the 2040s datacenter refresh
dumped. If the format is being emitted by tools that never heard of this repo,
that is E1 or E2 — victory either way. The code was always just the first
reader and writer of the format.

### Standing invariants (the constitution, all phases)
1. **Artifacts are the API.** Plain text, versioned, self-describing. Code and
   engines serve the artifacts, never the reverse.
2. **The core describes physics, not vendors.** GB, GB/s, links, indices. The
   word "cuda" may appear in exporters only.
3. **Engines are plugins and presumed mortal.** A plan never names one.
4. **Zero required dependencies.** Stdlib-runnable core, offline-installable
   repo.
5. **Determinism.** Same manifest → byte-identical plan, forever. This is what
   makes plans diffable across decades and makes the golden tests possible.
6. **Abandonment-safe by default.** Every release must pass the resurrection
   test from a bare clone with no network and no maintainer.

---

## 5. Why this serves the actual goal

Ending dependence on the Nvidia/Micron-tier oligopoly does not require beating
them at fabrication. It requires refusing the write-off schedule: keeping
hardware productive after the vendor stack abandons it, and keeping the
knowledge of *how to run it* in formats that survive every intermediary's
death — engine, driver, package index, maintainer. Silicon is durable; the
tyranny lives entirely in software expiry. A planner whose artifacts stay
readable and whose core stays runnable for 30 years is a small tool, but it is
aimed at the load-bearing wall.
