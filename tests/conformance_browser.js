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
  "exact integer range",
  "model.bytes",
  "model.params",
  "model.sha256",
];

const cases = JSON.parse(fs.readFileSync(path.join(__dirname, "conformance_cases.json"), "utf-8"));

// Named clusters, mirroring test_writer_conformance.py. The default is
// deliberately bare -- no board identities, no runtimes -- because that is
// what a manifest written before placement evidence existed looks like.
const DUPLICATE_BOARD = "GPU-493239dc-f76e-bbbb-8e68-ffd34a5e7bbc";
const DISTINCT = [0, 1, 2, 3].map((i) => `GPU-aaaa1111-0000-4000-8000-00000000000${i}`);
const MEASURED = [{ engine: "llamacpp", compatibility: "MEASURED" }];

function gpus(uuids) {
  return [0, 1, 2, 3].map((i) => {
    const gpu = { index: i, name: "RTX_3090", vram_gb: 24, mem_bw_gbps: 936 };
    if (uuids) gpu.uuid = uuids[i];
    return gpu;
  });
}

const CLUSTERS = {
  "": { gpus: gpus(null), runtimes: [] },
  "evidenced": { gpus: gpus(DISTINCT), runtimes: MEASURED },
  "duplicate-board": {
    gpus: gpus(DISTINCT.slice(0, 2).concat([DUPLICATE_BOARD, DUPLICATE_BOARD])),
    runtimes: MEASURED,
  },
  "no-runtime": { gpus: gpus(DISTINCT), runtimes: [] },
  "unmeasured-runtime": { gpus: gpus(DISTINCT), runtimes: ["llamacpp"] },
};

function manifestFor(model, cluster) {
  const shape = CLUSTERS[cluster === undefined ? "" : cluster];
  return {
    schema_version: 1,
    cluster: {
      name: "t",
      nodes: [{
        id: "node-a",
        host: "127.0.0.1",
        runtimes: JSON.parse(JSON.stringify(shape.runtimes)),
        gpus: JSON.parse(JSON.stringify(shape.gpus)),
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
    const plan = planner.buildPlan(planner.normalizeManifest(manifestFor(c.model, c.cluster)));
    // The complete model object and the complete placement decision, not a
    // projection of either. A selected view left model_size_bytes -- the
    // number the placement was authorized against -- outside the comparison
    // entirely.
    out[c.id] = { kind: "plan", model: plan.model, placement: plan.placement };
  } catch (e) {
    out[c.id] = {
      kind: "refuse",
      // The typed code is the machine-readable contract and is what a
      // downstream reader branches on; the marker set stays as the secondary
      // check for the untyped size-authority refusals, which have no code.
      code: e.code === undefined ? null : e.code,
      markers: MARKERS.filter((m) => e.message.includes(m)).sort(),
    };
  }
}

process.stdout.write(JSON.stringify(out));
