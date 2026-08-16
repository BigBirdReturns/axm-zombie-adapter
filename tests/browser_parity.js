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

check("the golden plan discloses its size came from a name guess", () => {
  const plan = planFor(path.join(EXAMPLES, "cluster_4x3090_singlebox.json"));
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
  const message = refusal({ name: "some-model", dtype: "fp8", bytes: 1e30 });
  assert.ok(message !== null && message.includes("Insufficient VRAM"), `got: ${message}`);
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

console.log("");
if (failures) {
  console.log(`browser parity: ${failures} FAILED`);
  process.exit(1);
}
console.log("browser parity: OK (module loads, plans match goldens, refusals agree with Python)");
