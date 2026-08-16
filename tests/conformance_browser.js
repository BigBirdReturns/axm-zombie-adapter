/* Browser half of the two-writer conformance check.
 *
 *     node tests/conformance_browser.js
 *
 * Runs tests/conformance_cases.json through docs/planner.js and writes one
 * JSON object of outcomes, keyed by case id, to stdout. The Python half
 * (tests/test_writer_conformance.py) runs the same table through the reference
 * planner and asserts the two decide identically.
 *
 * Refusal wording cannot be compared verbatim across runtimes -- Python names
 * a type `str` where JavaScript names it `string` -- so a refusal is reported
 * as the set of defect markers it raises.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "..");
const planner = require(path.join(REPO, "docs", "planner.js"));

const MARKERS = [
  "Insufficient VRAM",
  "Size-claim conflict",
  "mixture-of-experts",
  "Unknown model size",
  "cannot resolve",
  "model.bytes",
  "model.params",
  "model.sha256",
];

const cases = JSON.parse(fs.readFileSync(path.join(__dirname, "conformance_cases.json"), "utf-8"));

function manifestFor(model) {
  return {
    schema_version: 1,
    cluster: {
      name: "t",
      nodes: [{
        id: "node-a",
        host: "127.0.0.1",
        gpus: [0, 1, 2, 3].map((i) => ({ index: i, name: "RTX_3090", vram_gb: 24, mem_bw_gbps: 936 })),
      }],
    },
    model: Object.assign({}, model),
    policy: {
      target_tps: 18, max_latency_ms: 250, prefer_bandwidth: true,
      allow_cpu_offload: false, reserve_gb_per_gpu: 4.0,
    },
  };
}

const out = {};
for (const c of cases) {
  try {
    const plan = planner.buildPlan(planner.normalizeManifest(manifestFor(c.model)));
    out[c.id] = {
      kind: "plan",
      gb: plan.model.estimated_model_gb,
      source: plan.model.model_size_source,
      sha: plan.model.model_object_sha256,
    };
  } catch (e) {
    out[c.id] = { kind: "refuse", markers: MARKERS.filter((m) => e.message.includes(m)).sort() };
  }
}

process.stdout.write(JSON.stringify(out));
