/* Executable gate for the second plan writer.
 *
 *     node tests/browser_parity.js
 *
 * docs/planner.js is published via GitHub Pages and writes the same plan
 * artifact as the normative Python planner. It has drifted from that planner
 * before -- it once kept a silent 30e9 parameter default and a bare "7b"
 * substring match long after Python had refused both -- and a later edit left
 * a deleted identifier in the exported API, so the module threw during
 * initialization and the page never loaded. Neither failure was reachable by
 * the Python suite.
 *
 * This test therefore loads the real module, asserts it initializes, compares
 * its plans against the committed goldens, and pins the refusals that must
 * agree with Python. No dependencies, per DURABILITY.md invariant 4.
 */
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "..");
const EXAMPLES = path.join(REPO, "examples");
const GOLDEN = path.join(REPO, "tests", "golden");

let failures = 0;

function check(label, fn) {
  try {
    fn();
    console.log(`  ok   ${label}`);
  } catch (e) {
    failures += 1;
    console.log(`  FAIL ${label}\n         ${e.message}`);
  }
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf-8"));
}

// The module must load before anything else can be asserted about it.
const planner = require(path.join(REPO, "docs", "planner.js"));

console.log("module initialization");
check("docs/planner.js initializes and exports its API", () => {
  for (const name of ["normalizeManifest", "buildPlan", "degradeManifest", "resolveModelSize"]) {
    assert.strictEqual(typeof planner[name], "function", `missing export: ${name}`);
  }
});
check("the removed estimateModelBytes export is really gone", () => {
  assert.strictEqual(planner.estimateModelBytes, undefined);
});
check("the published page consumes the placement classification", () => {
  // docs/index.html is the surface a reader actually sees. If it stops reading
  // placement.state, the page can draw a topology that looks exactly like a
  // proved placement -- which is the original WL-02 defect wearing a UI.
  const page = fs.readFileSync(path.join(REPO, "docs", "index.html"), "utf-8");
  assert.ok(page.includes("placement.state"), "the page ignores placement.state");
  assert.ok(
    page.includes("missing_predicates"),
    "the page does not show which predicates were never claimed"
  );
});

function planFor(manifestFile) {
  return planner.buildPlan(planner.normalizeManifest(readJson(manifestFile)));
}

// Golden parity. Compared as parsed structures rather than bytes: Python
// serializes a float as 24.0 where JavaScript writes 24, which is the same
// number and a different string. Every value must otherwise match exactly.
console.log("golden parity");
for (const [example, golden] of [
  ["cluster_4x3090_singlebox.json", "cluster_4x3090_singlebox.plan.json"],
  ["cluster_4x3090_2nodes_10gbe.json", "cluster_4x3090_2nodes_10gbe.plan.json"],
]) {
  check(`browser plan for ${example} equals ${golden}`, () => {
    assert.deepStrictEqual(planFor(path.join(EXAMPLES, example)), readJson(path.join(GOLDEN, golden)));
  });
}

check("the golden plan discloses that its size was declared, not guessed", () => {
  const plan = planFor(path.join(EXAMPLES, "cluster_4x3090_singlebox.json"));
  assert.strictEqual(plan.model.model_size_source, "manifest_bytes");
  assert.strictEqual(
    plan.model.model_object_sha256,
    "b55515f6bff0e4efbca1fe74c68bf4ec5762f70e787d544586b702a6843caf71"
  );
});
check("a name-guessed size is still disclosed as one", () => {
  const plan = planner.buildPlan(planner.normalizeManifest(manifest({ name: "llama-3-70b", dtype: "q4" })));
  assert.strictEqual(plan.model.model_size_source, "legacy_name_heuristic");
  assert.strictEqual(plan.model.model_object_sha256, null);
});

// Refusals. These are the cases the Python suite pins; the browser writer must
// refuse the same inputs, or the published page authorizes what the reference
// implementation rejects.
function manifest(model, gpus, vramGb) {
  return {
    schema_version: 1,
    cluster: {
      name: "t",
      nodes: [{
        id: "node-a",
        host: "127.0.0.1",
        gpus: Array.from({ length: gpus === undefined ? 4 : gpus }, (_, i) => ({
          index: i, name: "RTX_3090", vram_gb: vramGb === undefined ? 24 : vramGb, mem_bw_gbps: 936,
        })),
      }],
    },
    model: model,
    policy: { target_tps: 18, max_latency_ms: 250, prefer_bandwidth: true, allow_cpu_offload: false, reserve_gb_per_gpu: 4.0 },
  };
}

function refusal(model) {
  try {
    planner.buildPlan(planner.normalizeManifest(manifest(model)));
  } catch (e) {
    return e.message;
  }
  return null;
}

function refuses(label, model, needle) {
  check(label, () => {
    const message = refusal(model);
    assert.ok(message !== null, "a plan was emitted instead of a refusal");
    assert.ok(
      message.includes(needle),
      `refusal did not mention ${JSON.stringify(needle)}: ${message}`
    );
  });
}

console.log("induced refusals");
refuses(
  "Kimi-K3 at its measured size refuses four RTX 3090s, naming the source",
  { name: "Kimi-K3", dtype: "fp8", bytes: 1560860324864 },
  "manifest_bytes"
);
refuses("mixtral-8x7b refuses as unsupported MoE", { name: "mixtral-8x7b", dtype: "fp16" }, "mixture-of-experts");
// `\b` counts "_" as a word character, so the MoE pattern used to stop
// matching at "8x7b_" while the parameter-token pattern happily read the same
// name as a 7B model -- 13 GB for a ~46.7B mixture, and a plan to place it.
refuses(
  "mixtral-8x7b_model refuses as MoE despite the underscore",
  { name: "mixtral-8x7b_model", dtype: "fp16" },
  "mixture-of-experts"
);
refuses(
  "mixtral_8x7b_instruct refuses as MoE",
  { name: "mixtral_8x7b_instruct", dtype: "fp16" },
  "mixture-of-experts"
);
refuses(
  "a spaced multiplier before an underscore refuses as MoE",
  { name: "moe 4 x 22 b_v2", dtype: "fp16" },
  "mixture-of-experts"
);
refuses("an unknown model with no size fields refuses", { name: "Kimi-K3", dtype: "fp8" }, "Unknown model size");
refuses(
  "contradictory bytes and params refuse",
  { name: "some-model", dtype: "fp8", bytes: 34359738368, params: 1000000000000 },
  "Size-claim conflict"
);
refuses(
  "70B fp32 refuses on four 3090s (insufficient VRAM)",
  { name: "llama-3-70b", dtype: "fp32" },
  "Insufficient VRAM"
);

console.log("impossible size declarations");
for (const [label, value] of [
  ["zero", 0],
  ["negative", -1],
  ["fractional", 1.5],
  ["boolean", true],
  ["string", "34359738368"],
  ["null-like NaN", NaN],
  ["Infinity", Infinity],
]) {
  refuses(`model.bytes = ${label} refuses`, { name: "some-model", dtype: "fp8", bytes: value }, "model.bytes");
  refuses(`model.params = ${label} refuses`, { name: "some-model", dtype: "fp8", params: value }, "model.params");
}

check("an excessively large size refuses rather than authorizing a placement", () => {
  // 1e30 is an integer-valued double but not an exact integer, so it refuses
  // as unrepresentable before it can reach the feasibility check.
  const message = refusal({ name: "some-model", dtype: "fp8", bytes: 1e30 });
  assert.ok(message !== null && message.includes("exact integer range"), `got: ${message}`);
});

// A JSON byte count above 2**53-1 does not survive being read as a double.
// 9007199254740993 parses to 9007199254740992 here, so publishing it would
// mean publishing a number the manifest never asserted as the size that
// authorized the placement.
console.log("exact integer authority");
check("the lossy parse this guards against is real", () => {
  assert.strictEqual(JSON.parse("9007199254740993"), 9007199254740992);
});
refuses(
  "model.bytes above the exact integer range refuses instead of rounding",
  { name: "some-model", dtype: "fp8", bytes: 9007199254740993 },
  "exact integer range"
);
refuses(
  "model.params above the exact integer range refuses instead of rounding",
  { name: "some-model", dtype: "fp8", params: 9007199254740993 },
  "exact integer range"
);
check("the largest exactly representable byte count is still read as a size", () => {
  const message = refusal({ name: "some-model", dtype: "fp8", bytes: Number.MAX_SAFE_INTEGER });
  assert.ok(message !== null && message.includes("Insufficient VRAM"), `got: ${message}`);
});

// Presence and value are separate facts: an absent size field falls through to
// the name heuristic, a declared-null one is a size claim that is not a size.
console.log("declared-null size fields");
refuses(
  "model.bytes declared null refuses rather than falling through to the name",
  { name: "llama-3-70b", dtype: "q4", bytes: null },
  "model.bytes"
);
refuses(
  "model.params declared null refuses rather than falling through to the name",
  { name: "llama-3-70b", dtype: "q4", params: null },
  "model.params"
);
check("the same name with the field absent still plans", () => {
  const plan = planner.buildPlan(planner.normalizeManifest(manifest({ name: "llama-3-70b", dtype: "q4" })));
  assert.strictEqual(plan.model.model_size_source, "legacy_name_heuristic");
});

// One rounding law, spelled out in exact integer arithmetic. Math.round(gb *
// 100) / 100 gives 1.13 for a decimal tie where Python's round() gives 1.12.
console.log("decimal rounding law");
check("a decimal tie rounds half away from zero (1207959552 B -> 1.13 GB)", () => {
  const plan = planner.buildPlan(planner.normalizeManifest(manifest({ name: "m", dtype: "fp8", bytes: 1207959552 })));
  assert.strictEqual(plan.model.estimated_model_gb, 1.13);
  assert.strictEqual(plan.model.model_size_bytes, 1207959552);
});
check("one byte below the tie rounds down (1207959551 B -> 1.12 GB)", () => {
  const plan = planner.buildPlan(planner.normalizeManifest(manifest({ name: "m", dtype: "fp8", bytes: 1207959551 })));
  assert.strictEqual(plan.model.estimated_model_gb, 1.12);
});

console.log("model object identity");
for (const [label, value] of [["too short", "abc"], ["non-hex", "z".repeat(64)], ["numeric", 12345]]) {
  refuses(`model.sha256 ${label} refuses`, { name: "llama-3-70b", dtype: "q4", sha256: value }, "model.sha256");
}
check("a valid sha256 is carried into the plan, normalized", () => {
  const digest = "A".repeat(64);
  const plan = planner.buildPlan(planner.normalizeManifest(manifest({ name: "llama-3-70b", dtype: "q4", sha256: digest })));
  assert.strictEqual(plan.model.model_object_sha256, "a".repeat(64));
});

console.log("model-name token boundaries");
refuses("model-170b is not read as a 70B model", { name: "model-170b", dtype: "q4" }, "170b");
refuses("model-107b is not read as a 7B model", { name: "model-107b", dtype: "q4" }, "107b");
check("llama-3-70b still resolves to 32.6 GB", () => {
  const plan = planner.buildPlan(planner.normalizeManifest(manifest({ name: "llama-3-70b", dtype: "q4" })));
  assert.strictEqual(plan.model.estimated_model_gb, 32.6);
});

/* ------------------------------------------------------------------------
 * WL-02: the published writer must reach the same placement decision.
 *
 * These mirror tests/test_placement.py. `crypto` is a Node built-in, so this
 * stays dependency-free per DURABILITY.md invariant 4; no digest below is
 * invented, each is the sha-256 of an ASCII label computed here, under the
 * same scheme and the same labels the Python witnesses use.
 * --------------------------------------------------------------------- */
const crypto = require("crypto");

function labelDigest(label) {
  return crypto.createHash("sha256").update(label, "ascii").digest("hex");
}

const W_DIGEST = labelDigest("witness-model-object:llama-3-70b-q4");
const W_OTHER = labelDigest("witness-model-object:a-different-object");

function wIdentity(digest, state) {
  return {
    identity_scheme: "test/ascii-label-object@1",
    identity_digest: digest === undefined ? W_DIGEST : digest,
    identity_source: "tests/browser_parity.js::labelDigest",
    identity_state: state === undefined ? "VERIFIED" : state,
    verified_at: "2026-08-16T00:00:00Z",
    validator: "crypto.createHash('sha256') over the ASCII label",
  };
}

function wWeights(payload) {
  const p = payload === undefined ? 35000000000 : payload;
  return {
    checkpoint_set_bytes: p + 8388608,
    tensor_payload_bytes: p,
    container_overhead_bytes: 8388608,
  };
}

function wModel(overrides) {
  return Object.assign({
    name: "llama-3-70b",
    dtype: "q4",
    kv_cache: "fp16",
    bytes: 35000000000,
    sha256: W_DIGEST,
    weights: wWeights(),
    identity: wIdentity(),
    engines: ["llamacpp"],
    residence: [{
      node: "node-a", verified: true, freshness: "CURRENT",
      identity_digest: W_DIGEST,
    }],
  }, overrides || {});
}

function wManifest(model, gpus, runtimes) {
  return {
    schema_version: 1,
    cluster: {
      name: "witness",
      nodes: [{
        id: "node-a",
        host: "10.0.0.1",
        runtimes: runtimes === undefined
          ? [{ engine: "llamacpp", compatibility: "MEASURED" }]
          : runtimes,
        gpus: gpus === undefined
          ? [0, 1, 2, 3].map((i) => ({
              index: i, name: "RTX_3090", vram_gb: 24, mem_bw_gbps: 936,
              uuid: `GPU-aaaa1111-0000-4000-8000-00000000000${i}`,
            }))
          : gpus,
      }],
    },
    model: model,
    policy: { target_tps: 18, reserve_gb_per_gpu: 4.0 },
  };
}

function wRefusal(model, gpus, runtimes) {
  try {
    planner.buildPlan(planner.normalizeManifest(wManifest(model, gpus, runtimes)));
  } catch (e) {
    return e;
  }
  return null;
}

function wRefuses(label, code, model, gpus, runtimes) {
  check(label, () => {
    const error = wRefusal(model, gpus, runtimes);
    assert.ok(error !== null, "a plan was emitted instead of a refusal");
    assert.strictEqual(
      error.code, code,
      `expected ${code}, got ${error.code || "(untyped)"}: ${error.message}`
    );
  });
}

console.log("placement: runnable is not representable");
check("a manifest with no object evidence is representable, not runnable", () => {
  const plan = planner.buildPlan(planner.normalizeManifest(
    wManifest({ name: "llama-3-70b", dtype: "q4" }, undefined, [])
  ));
  assert.strictEqual(plan.placement.state, "REPRESENTABLE_NOT_RUNNABLE");
  assert.strictEqual(plan.placement.proof, null);
  assert.deepStrictEqual(plan.placement.missing_predicates, [
    "verified_object_identity", "declared_weight_accounting",
    "verified_residence", "measured_runtime",
  ]);
});
check("this page cannot promote a sketch to runnable by looking at topology", () => {
  // Topology only, but every seat identified and a measured runtime declared.
  // Nothing here names an object, so nothing here can be runnable.
  const plan = planner.buildPlan(planner.normalizeManifest(
    wManifest({ name: "llama-3-70b", dtype: "q4" })
  ));
  assert.strictEqual(plan.placement.state, "REPRESENTABLE_NOT_RUNNABLE");
  assert.deepStrictEqual(plan.placement.missing_predicates, [
    "verified_object_identity", "declared_weight_accounting",
    "verified_residence",
  ]);
});
check("a fully evidenced manifest is runnable and carries its proof", () => {
  const plan = planner.buildPlan(planner.normalizeManifest(wManifest(wModel())));
  assert.strictEqual(plan.placement.state, "RUNNABLE");
  assert.deepStrictEqual(plan.placement.missing_predicates, []);
  assert.strictEqual(plan.placement.proof.identity.identity_digest, W_DIGEST);
  assert.strictEqual(plan.placement.proof.sharding_strategy, "pipeline");
});

console.log("placement: typed refusals");
wRefuses(
  "one board seated twice refuses instead of summing its VRAM",
  planner.REFUSAL.SEAT_IDENTITY_CONFLICT,
  wModel(),
  [0, 1].map((i) => ({
    index: i, name: "RTX_3090", vram_gb: 24, mem_bw_gbps: 936,
    uuid: "GPU-493239dc-f76e-bbbb-8e68-ffd34a5e7bbc",
  }))
);
wRefuses(
  "a bare sha256 is a declaration, not custody",
  planner.REFUSAL.UNVERIFIED_MODEL_IDENTITY,
  wModel({ identity: undefined })
);
wRefuses(
  "an identity declared but not verified refuses",
  planner.REFUSAL.UNVERIFIED_MODEL_IDENTITY,
  wModel({ identity: wIdentity(W_DIGEST, "DECLARED") })
);
wRefuses(
  "two identities for one object refuse",
  planner.REFUSAL.UNVERIFIED_MODEL_IDENTITY,
  wModel({ sha256: W_OTHER })
);
wRefuses(
  "absent residence is not capacity",
  planner.REFUSAL.RESIDENCE_UNVERIFIED,
  wModel({ residence: [] })
);
wRefuses(
  "stale residence is not current residence",
  planner.REFUSAL.RESIDENCE_STALE,
  wModel({ residence: [{ node: "node-a", verified: true, freshness: "STALE", identity_digest: W_DIGEST }] })
);
wRefuses(
  "residence for a different object refuses rather than being ignored",
  planner.REFUSAL.RESIDENCE_IDENTITY_MISMATCH,
  wModel({ residence: [{ node: "node-a", verified: true, freshness: "CURRENT", identity_digest: W_OTHER }] })
);
wRefuses(
  "a node with no runtime cannot execute, whatever its VRAM",
  planner.REFUSAL.NO_SUPPORTED_ENGINE,
  wModel(), undefined, []
);
wRefuses(
  "no engine in common refuses even when runtimes exist",
  planner.REFUSAL.NO_SUPPORTED_ENGINE,
  wModel(), undefined, [{ engine: "vllm", compatibility: "MEASURED" }]
);
wRefuses(
  "a declared but unmeasured runtime is not a runtime",
  planner.REFUSAL.RUNTIME_UNMEASURED,
  wModel(), undefined, ["llamacpp"]
);
wRefuses(
  "an unsupported sharding strategy refuses rather than proving a different one",
  planner.REFUSAL.UNSUPPORTED_SHARDING_STRATEGY,
  wModel({ sharding_strategy: "tensor_parallel" })
);
wRefuses(
  "weight accounting that does not close refuses",
  planner.REFUSAL.WEIGHT_ACCOUNTING_INCONSISTENT,
  wModel({ weights: { checkpoint_set_bytes: 35008388609, tensor_payload_bytes: 35000000000, container_overhead_bytes: 8388608 } })
);
wRefuses(
  "a partial weight accounting refuses",
  planner.REFUSAL.WEIGHT_ACCOUNTING_MISSING,
  wModel({ weights: { tensor_payload_bytes: 35000000000, container_overhead_bytes: 8388608 } })
);
wRefuses(
  "placement evidence without any weight accounting refuses",
  planner.REFUSAL.WEIGHT_ACCOUNTING_MISSING,
  wModel({ weights: undefined })
);
wRefuses(
  "declaring the checkpoint set as model.bytes refuses",
  planner.REFUSAL.WEIGHT_ACCOUNTING_CONFLICT,
  wModel({ bytes: 35008388608 })
);
wRefuses(
  "Kimi-K3's measured payload does not fit four 3090s",
  planner.REFUSAL.STAGE_DOES_NOT_FIT,
  wModel({ bytes: 1560860324864, weights: {
    checkpoint_set_bytes: 1560936091448,
    tensor_payload_bytes: 1560860324864,
    container_overhead_bytes: 75766584,
  } })
);

console.log("placement: every byte is assigned and proved");
check("stage and device assignments are exact and within each device", () => {
  const proof = planner.buildPlan(planner.normalizeManifest(wManifest(wModel()))).placement.proof;
  const devices = proof.stages.reduce((all, s) => all.concat(s.devices), []);
  assert.strictEqual(proof.assigned_bytes_total, proof.weights.tensor_payload_bytes);
  assert.strictEqual(
    devices.reduce((sum, d) => sum + d.assigned_bytes, 0),
    proof.weights.tensor_payload_bytes
  );
  for (const stage of proof.stages) {
    const own = stage.devices.reduce((sum, d) => sum + d.usable_bytes, 0);
    assert.strictEqual(stage.usable_bytes, own);
    assert.ok(stage.assigned_bytes <= own, "a stage exceeded its own seats");
    assert.strictEqual(
      stage.devices.reduce((sum, d) => sum + d.assigned_bytes, 0),
      stage.assigned_bytes
    );
  }
  for (const device of devices) {
    assert.ok(
      device.assigned_bytes <= device.usable_bytes,
      `${device.node}:${device.gpu} was assigned more than it holds`
    );
  }
});
check("the proof keeps the three byte quantities apart", () => {
  const proof = planner.buildPlan(planner.normalizeManifest(wManifest(wModel()))).placement.proof;
  assert.strictEqual(proof.placed_bytes_are, "tensor_payload_bytes");
  assert.strictEqual(proof.custody_binds, "checkpoint_set_bytes");
  assert.ok(proof.weights.checkpoint_set_bytes > proof.weights.tensor_payload_bytes);
  assert.strictEqual(
    proof.weights.tensor_payload_bytes + proof.weights.container_overhead_bytes,
    proof.weights.checkpoint_set_bytes
  );
});
check("apportionment stays exact at terabyte scale", () => {
  // total * weight here is ~3.4e22, far outside the exact integer range, so
  // this is the case that fails silently if the arithmetic is not exact.
  const parts = planner.apportionExact(1560860324864, [21474836480, 21474836480, 4294967296]);
  assert.strictEqual(parts.reduce((a, b) => a + b, 0), 1560860324864);
  assert.deepStrictEqual(
    planner.apportionExact(1000000007, [5, 5, 3]),
    planner.apportionExact(1000000007, [5, 5, 3])
  );
  assert.deepStrictEqual(planner.apportionExact(10, [0, 0]), [0, 0]);
});
check("a device smaller than the reserve contributes nothing", () => {
  assert.strictEqual(planner.usableBytesForSeat({ vram_gb: 2.0 }, 4.0), 0);
  assert.strictEqual(planner.usableBytesForSeat({ vram_gb: 24.0 }, 4.0), 20 * Math.pow(1024, 3));
});

console.log("placement: the real estate refusal");
check("the committed OCTO-W01 refusal is what this writer emits too", () => {
  const manifestPath = path.join(EXAMPLES, "estate_octo_w01_kimi_k3.json");
  const committed = readJson(path.join(REPO, "artifacts", "estate-octo-w01-kimi-k3.refusal.json"));
  let emitted = null;
  try {
    planner.buildPlan(planner.normalizeManifest(readJson(manifestPath)));
  } catch (e) {
    emitted = e.toObject ? e.toObject() : null;
  }
  assert.ok(emitted !== null, "the estate manifest did not refuse");
  assert.deepStrictEqual(emitted, committed);
});
check("the real manifest carries no invented digest", () => {
  const model = readJson(path.join(EXAMPLES, "estate_octo_w01_kimi_k3.json")).model;
  assert.strictEqual(model.sha256, undefined);
  assert.strictEqual(model.identity.identity_digest, "");
  assert.ok(model.identity.identity_source.length > 0);
  assert.strictEqual(
    model.weights.tensor_payload_bytes + model.weights.container_overhead_bytes,
    model.weights.checkpoint_set_bytes
  );
});

console.log("");
if (failures) {
  console.log(`browser parity: ${failures} FAILED`);
  process.exit(1);
}
console.log("browser parity: OK (module loads, plans match goldens, refusals agree with Python)");
