/* Faithful JavaScript port of the axm-zombie-adapter reference planner
 * (src/axm_zombie/manifest.py, planner.py, replan.py), schema_version 1.
 * The Python implementation is normative; this port exists so the demo page
 * runs entirely in the browser. Verified against the repo's golden plan files.
 */
(function (root) {
  "use strict";

  const SCHEMA_VERSION = 1;

  function normalizeManifest(data) {
    if (typeof data !== "object" || data === null || Array.isArray(data)) {
      throw new Error("Manifest must be a mapping.");
    }
    if (data.schema_version === undefined) data.schema_version = SCHEMA_VERSION;
    if (Number(data.schema_version) !== SCHEMA_VERSION) {
      throw new Error(
        `Unsupported manifest schema_version ${data.schema_version}; ` +
        `this reader supports version ${SCHEMA_VERSION}. See SPEC.md.`
      );
    }
    for (const k of ["cluster", "model", "policy"]) {
      if (!(k in data)) throw new Error(`Missing top-level key: ${k}`);
    }
    const cluster = data.cluster;
    if (!Array.isArray(cluster.nodes) || cluster.nodes.length === 0) {
      throw new Error("cluster.nodes must be a non-empty list.");
    }
    for (const node of cluster.nodes) {
      if (!("id" in node)) throw new Error("Each node must have an id.");
      if (!Array.isArray(node.gpus) || node.gpus.length === 0) {
        throw new Error("Each node must have a non-empty gpus list.");
      }
      for (const gpu of node.gpus) {
        for (const req of ["index", "vram_gb"]) {
          if (!(req in gpu)) throw new Error(`GPU entry on node ${node.id} missing ${req}.`);
        }
        if (gpu.name === undefined) gpu.name = "GPU";
        if (gpu.mem_bw_gbps === undefined) gpu.mem_bw_gbps = null;
      }
      if (node.links === undefined) node.links = [];
      if (node.notes === undefined) node.notes = "";
      if (node.host === undefined) node.host = null;
    }
    const model = data.model;
    for (const req of ["name", "dtype"]) {
      if (!(req in model)) throw new Error(`model.${req} is required.`);
    }
    if (model.kv_cache === undefined) model.kv_cache = model.dtype;
    const policy = data.policy;
    if (policy.target_tps === undefined) policy.target_tps = 18;
    if (policy.max_latency_ms === undefined) policy.max_latency_ms = 250;
    if (policy.prefer_bandwidth === undefined) policy.prefer_bandwidth = true;
    if (policy.allow_cpu_offload === undefined) policy.allow_cpu_offload = false;
    if (policy.reserve_gb_per_gpu === undefined) policy.reserve_gb_per_gpu = 4.0;
    return data;
  }

  function estimateModelBytes(modelName, dtype) {
    const name = modelName.toLowerCase();
    let params;
    if (name.includes("70b")) params = 70e9;
    else if (name.includes("34b")) params = 34e9;
    else if (name.includes("13b")) params = 13e9;
    else if (name.includes("7b")) params = 7e9;
    else params = 30e9;
    const table = { fp32: 4.0, fp16: 2.0, bf16: 2.0, fp8: 1.0, int8: 1.0, q8: 1.0, int4: 0.5, q4: 0.5 };
    const bpp = table[dtype.toLowerCase()] !== undefined ? table[dtype.toLowerCase()] : 2.0;
    return Math.trunc(params * bpp);
  }

  function clusterGpus(manifest) {
    const gpus = [];
    for (const node of manifest.cluster.nodes) {
      for (const gpu of node.gpus) {
        gpus.push({
          node: node.id,
          host: node.host !== undefined ? node.host : null,
          gpu: Math.trunc(gpu.index),
          name: gpu.name !== undefined ? gpu.name : "GPU",
          vram_gb: Number(gpu.vram_gb),
          mem_bw_gbps: gpu.mem_bw_gbps !== undefined ? gpu.mem_bw_gbps : null,
        });
      }
    }
    return gpus;
  }

  function linkBwBetween(manifest, a, b) {
    if (a === b) return Infinity;
    for (const node of manifest.cluster.nodes) {
      if (node.id !== a) continue;
      for (const link of node.links || []) {
        if (link.to === b && link.gbps !== undefined && link.gbps !== null) {
          return Number(link.gbps);
        }
      }
    }
    return 0.0;
  }

  function scoreStageBoundary(manifest, leftNodes, rightNodes, actGbPerToken, targetTps) {
    if (!leftNodes.length || !rightNodes.length) return 0.0;
    let bw = 0.0;
    for (const a of leftNodes) {
      for (const b of rightNodes) {
        bw = Math.max(bw, linkBwBetween(manifest, a, b));
      }
    }
    if (bw <= 0.0) return 1e9;
    const requiredGbps = actGbPerToken * targetTps * 8.0;
    const util = requiredGbps / bw;
    if (util > 0.7) return 1e6 * util;
    return 1e3 * util;
  }

  function buildPlan(manifest) {
    const model = manifest.model;
    const policy = manifest.policy;
    const gpus = clusterGpus(manifest);

    const totalVram = gpus.reduce((s, g) => s + g.vram_gb, 0);
    const modelBytes = estimateModelBytes(model.name, model.dtype);
    const modelGb = modelBytes / Math.pow(1024, 3);

    const reserve = Number(policy.reserve_gb_per_gpu !== undefined ? policy.reserve_gb_per_gpu : 4.0);
    const usableVram = Math.max(0.0, totalVram - reserve * gpus.length);
    if (usableVram < modelGb * 1.05) {
      throw new Error(
        `Insufficient VRAM for rough model estimate. Usable ~${usableVram.toFixed(1)} GB, model ~${modelGb.toFixed(1)} GB.`
      );
    }

    const nodeIds = Array.from(new Set(gpus.map((g) => g.node))).sort();
    const multiNode = nodeIds.length > 1;
    const stageCount = multiNode ? nodeIds.length : Math.max(1, Math.ceil(gpus.length / 2));

    const stages = [];
    if (multiNode) {
      for (const nid of nodeIds) stages.push(gpus.filter((g) => g.node === nid));
    } else {
      const sorted = gpus.slice().sort((a, b) => {
        const abw = a.mem_bw_gbps !== null ? Number(a.mem_bw_gbps) : 0.0;
        const bbw = b.mem_bw_gbps !== null ? Number(b.mem_bw_gbps) : 0.0;
        if (bbw !== abw) return bbw - abw;
        return b.vram_gb - a.vram_gb;
      });
      const chunk = Math.ceil(sorted.length / stageCount);
      for (let i = 0; i < stageCount; i++) {
        stages.push(sorted.slice(i * chunk, (i + 1) * chunk));
      }
    }

    const actGbPerToken = 0.002;
    const boundaryCosts = [];
    for (let i = 0; i < stages.length - 1; i++) {
      const left = Array.from(new Set(stages[i].map((g) => g.node))).sort();
      const right = Array.from(new Set(stages[i + 1].map((g) => g.node))).sort();
      boundaryCosts.push(
        scoreStageBoundary(manifest, left, right, actGbPerToken, Number(policy.target_tps !== undefined ? policy.target_tps : 18))
      );
    }

    return {
      schema_version: 1,
      model: {
        name: model.name,
        dtype: model.dtype,
        kv_cache: model.kv_cache !== undefined ? model.kv_cache : model.dtype,
        estimated_model_gb: Math.round(modelGb * 100) / 100,
      },
      cluster: {
        name: manifest.cluster.name !== undefined ? manifest.cluster.name : "zombie",
        nodes: nodeIds,
        gpus: gpus,
      },
      policy: policy,
      pipeline_stages: stages.map((stage, i) => ({
        stage: i,
        placement: stage.map((g) => ({ node: g.node, gpu: g.gpu })),
      })),
      kv_cache_policy: {
        reserve_gb_per_gpu: reserve,
        spill_allowed: Boolean(policy.allow_cpu_offload),
      },
      health: { replan_on_gpu_oom: true, replan_on_node_loss: true },
      diagnostics: {
        stage_boundary_costs: boundaryCosts,
        notes: "Planner is heuristic. Tune KV cache and activation sizing per model for accurate TPS estimates.",
      },
    };
  }

  function parseLoss(spec) {
    if (spec.includes(":")) {
      const i = spec.lastIndexOf(":");
      const nodeId = spec.slice(0, i);
      const idx = Number(spec.slice(i + 1));
      if (!Number.isInteger(idx)) {
        throw new Error(`Bad loss spec ${spec}: GPU index must be an integer.`);
      }
      return [nodeId, idx];
    }
    return [spec, null];
  }

  function degradeManifest(manifest, losses) {
    const m = JSON.parse(JSON.stringify(manifest));
    const nodes = m.cluster.nodes;
    for (const spec of losses) {
      const [nodeId, gpuIdx] = parseLoss(spec);
      const node = nodes.find((n) => n.id === nodeId);
      if (!node) throw new Error(`Loss ${spec}: no node with id ${nodeId} in manifest.`);
      if (gpuIdx === null) {
        nodes.splice(nodes.indexOf(node), 1);
        continue;
      }
      const gpu = node.gpus.find((g) => Math.trunc(g.index) === gpuIdx);
      if (!gpu) throw new Error(`Loss ${spec}: node ${nodeId} has no GPU with index ${gpuIdx}.`);
      node.gpus.splice(node.gpus.indexOf(gpu), 1);
      if (node.gpus.length === 0) nodes.splice(nodes.indexOf(node), 1);
    }
    if (nodes.length === 0) throw new Error("All nodes lost; nothing left to plan.");
    const live = new Set(nodes.map((n) => n.id));
    for (const n of nodes) {
      n.links = (n.links || []).filter((l) => live.has(l.to));
    }
    return m;
  }

  const api = { normalizeManifest, buildPlan, degradeManifest, estimateModelBytes, linkBwBetween, SCHEMA_VERSION };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.axmZombie = api;
})(typeof self !== "undefined" ? self : this);
