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

console.log("");
if (failures) {
  console.log(`browser parity: ${failures} FAILED`);
  process.exit(1);
}
console.log("browser parity: OK (module loads, plans match goldens, refusals agree with Python)");
