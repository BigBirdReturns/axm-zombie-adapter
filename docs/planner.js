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

  const SIZE_SOURCE_MANIFEST_BYTES = "manifest_bytes";
  const SIZE_SOURCE_MANIFEST_PARAMS = "manifest_params_and_quantization";
  const SIZE_SOURCE_NAME_HEURISTIC = "legacy_name_heuristic";

  // Two declared size claims disagreeing by this factor or more are a
  // contradiction, not a precedence question. See planner.py.
  const SIZE_CONFLICT_FACTOR = 10.0;

  // Parameter counts the name heuristic recognises, keyed by the token in
  // billions. Tokens are matched whole (see PARAM_TOKEN), so ordering is
  // irrelevant: "170b" is not a 70B model and "107b" is not a 7B model.
  const KNOWN_PARAM_TOKENS = { 7: 7e9, 13: 13e9, 34: 34e9, 70: 70e9 };

  // Identical in form to planner.py's _PARAM_TOKEN, using a leading
  // alternation rather than a lookbehind so both writers recognise exactly the
  // same names on every runtime.
  const PARAM_TOKEN = /(?:^|[^\d.])(\d+)\s*b(?![a-z0-9])/g;

  // A model object identity is a sha-256 digest or it is not an identity.
  const SHA256 = /^[0-9a-f]{64}$/;

  // "8x7b", "4 x 22b": a mixture-of-experts multiplier makes a bare parameter
  // substring a lie. mixtral-8x7b is ~46.7B, not 7B.
  //
  // The trailing delimiter is the same law PARAM_TOKEN uses. `\b` treats "_"
  // as a word character, so "mixtral-8x7b_model" escaped this check while
  // PARAM_TOKEN's `(?![a-z0-9])` accepted the same "_" and read the name as a
  // 7B model. The leading side is deliberately left unanchored: this pattern
  // only ever refuses, so matching more than PARAM_TOKEN can is fail-closed.
  const MOE_MULTIPLIER = /\d+\s*x\s*\d+\s*b(?![a-z0-9])/;

  // A byte count is exchanged as a JSON number and read here as a double.
  // Above 2**53-1 that read is lossy -- 9007199254740993 parses to
  // 9007199254740992 -- so both writers refuse beyond this magnitude rather
  // than publish a rounded number as the size that authorized a placement.
  const MAX_EXACT_INTEGER = Number.MAX_SAFE_INTEGER; // 9007199254740991

  // One gibibyte, as a BigInt, for the exact rounding law in estimatedGb().
  const GIB = 1073741824n;

  // Bytes -> GiB at two decimals, half away from zero, in exact integer
  // arithmetic. Mirrors planner.py's _estimated_gb. `Math.round(gb * 100) /
  // 100` and Python's `round(gb, 2)` disagree at a decimal tie -- 1207959552 B
  // is 1.13 here and 1.12 there -- so neither writer's native rounding is used.
  function estimatedGb(modelBytes) {
    const scaled = BigInt(modelBytes) * 100n;
    let whole = scaled / GIB;
    const remainder = scaled % GIB;
    if (2n * remainder >= GIB) whole += 1n;
    return Number(whole) / 100;
  }

  function bytesPerParam(dtype) {
    const table = { fp32: 4.0, fp16: 2.0, bf16: 2.0, fp8: 1.0, int8: 1.0, q8: 1.0, int4: 0.5, q4: 0.5 };
    const key = String(dtype).toLowerCase();
    return table[key] !== undefined ? table[key] : 2.0;
  }

  // Mirrors planner.py's _validated_size_field. A zero, negative, or NaN size
  // satisfies every feasibility comparison, so it authorizes any placement.
  function validatedSizeField(value, field, name) {
    const unit = field === "bytes" ? "bytes" : "parameters";
    if (typeof value !== "number") {
      throw new Error(
        `model.${field} for '${name}' must be a positive whole number of ${unit}, ` +
        `got ${JSON.stringify(value)} (${typeof value}). A size that is not a ` +
        "number cannot be compared against VRAM, and the planner will not " +
        "coerce it into one."
      );
    }
    if (!Number.isFinite(value)) {
      throw new Error(
        `model.${field} for '${name}' must be finite, got ${String(value)}. ` +
        "A non-finite size never fails the feasibility check."
      );
    }
    if (!Number.isInteger(value)) {
      throw new Error(
        `model.${field} for '${name}' must be a whole number of ${unit}, got ${value}.`
      );
    }
    if (Math.abs(value) > MAX_EXACT_INTEGER) {
      throw new Error(
        `model.${field} for '${name}' must lie within the exact integer range ` +
        `of +/-${MAX_EXACT_INTEGER} ${unit}, got ${value}. Beyond that ` +
        "magnitude the value cannot be represented exactly by both plan " +
        "writers, and rounding an asserted size would publish a number " +
        "nobody asserted as the size that authorized a placement."
      );
    }
    if (value <= 0) {
      throw new Error(
        `model.${field} for '${name}' must be greater than zero, got ${value}. ` +
        "A zero or negative size passes every feasibility check and would " +
        "authorize any placement."
      );
    }
    return value;
  }

  // Mirrors planner.py's _validated_sha256.
  function validatedSha256(value, name) {
    if (value === undefined || value === null) return null;
    if (typeof value === "string" && SHA256.test(value.trim().toLowerCase())) {
      return value.trim().toLowerCase();
    }
    throw new Error(
      `model.sha256 for '${name}' must be 64 hexadecimal characters ` +
      `identifying the measured model object, got ${JSON.stringify(value)}. An ` +
      "unverifiable identity must not be emitted as model_object_sha256, " +
      "because the whole point of the field is that it can be checked."
    );
  }

  function refuseContradictorySizeClaims(name, declared, params, implied, dtype) {
    const lo = Math.min(declared, implied);
    const hi = Math.max(declared, implied);
    const ratio = lo <= 0 ? Infinity : hi / lo;
    if (ratio < SIZE_CONFLICT_FACTOR) return;
    const gap = ratio === Infinity ? "unbounded" : `${ratio.toFixed(1)}x`;
    throw new Error(
      `Size-claim conflict for '${name}': model.bytes declares ${declared} B, ` +
      `but model.params (${params}) at ${bytesPerParam(dtype)} bytes/param for ` +
      `dtype '${dtype}' implies ${implied} B - a ${gap} disagreement. The ` +
      "planner will not pick a winner between two contradictory claims, " +
      "because a stale size field produces a confident plan that OOMs. " +
      "Correct model.bytes or model.params so the two agree within a factor " +
      `of ${SIZE_CONFLICT_FACTOR.toFixed(0)}.`
    );
  }

  // Returns { bytes, source }. Declared size always wins over the name, and an
  // unresolvable size refuses rather than defaulting: a wrong guess is emitted
  // as a confident placement plan that OOMs on contact with real hardware.
  function resolveModelSize(model) {
    const name = String(model.name);
    const dtype = model.dtype;
    // Presence and value are separate facts, exactly as in planner.py. An
    // absent `bytes` key means "no size declared, fall through"; a present
    // `bytes: null` means "a size was declared and it is nothing", which is
    // not a size. Treating the two alike let an explicitly empty declaration
    // fall through to the name heuristic and be published as a guess.
    const declaredBytes = Object.prototype.hasOwnProperty.call(model, "bytes");
    const declaredParams = Object.prototype.hasOwnProperty.call(model, "params");

    if (declaredBytes) {
      const size = validatedSizeField(model.bytes, "bytes", name);
      if (declaredParams) {
        const params = validatedSizeField(model.params, "params", name);
        const implied = Math.trunc(params * bytesPerParam(dtype));
        refuseContradictorySizeClaims(name, size, params, implied, dtype);
      }
      return { bytes: size, source: SIZE_SOURCE_MANIFEST_BYTES };
    }

    if (declaredParams) {
      const params = validatedSizeField(model.params, "params", name);
      return {
        bytes: Math.trunc(params * bytesPerParam(dtype)),
        source: SIZE_SOURCE_MANIFEST_PARAMS,
      };
    }

    const lowered = name.toLowerCase();
    if (MOE_MULTIPLIER.test(lowered)) {
      throw new Error(
        `Model '${name}' carries a mixture-of-experts multiplier, so its ` +
        "parameter count cannot be read from its name. Declare model.params " +
        "(total parameters, not per-expert) or model.bytes in the manifest."
      );
    }
    const tokens = [];
    PARAM_TOKEN.lastIndex = 0;
    let match;
    while ((match = PARAM_TOKEN.exec(lowered)) !== null) {
      const value = parseInt(match[1], 10);
      if (!tokens.includes(value)) tokens.push(value);
    }
    tokens.sort((a, b) => a - b);
    if (tokens.length === 1 && KNOWN_PARAM_TOKENS[tokens[0]] !== undefined) {
      return {
        bytes: Math.trunc(KNOWN_PARAM_TOKENS[tokens[0]] * bytesPerParam(dtype)),
        source: SIZE_SOURCE_NAME_HEURISTIC,
      };
    }
    if (tokens.length > 0) {
      const supported = Object.keys(KNOWN_PARAM_TOKENS).map(Number).sort((a, b) => a - b);
      throw new Error(
        `Model name '${name}' carries parameter token(s) ` +
        `${tokens.map((t) => t + "b").join(", ")} that the size heuristic ` +
        "cannot resolve; it recognises exactly " +
        `${supported.map((t) => t + "b").join(", ")} and will not round a name ` +
        "to the nearest supported size. Declare model.params or model.bytes " +
        "in the manifest."
      );
    }
    throw new Error(
      `Unknown model size for '${name}'. The planner will not assume a ` +
      "default parameter count, because the resulting plan would look " +
      "identical to a correct one. Declare model.params or model.bytes " +
      "in the manifest."
    );
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
    const resolved = resolveModelSize(model);
    const modelSha256 = validatedSha256(model.sha256, model.name);
    const modelBytes = resolved.bytes;
    const modelGb = modelBytes / Math.pow(1024, 3);

    const reserve = Number(policy.reserve_gb_per_gpu !== undefined ? policy.reserve_gb_per_gpu : 4.0);
    const usableVram = Math.max(0.0, totalVram - reserve * gpus.length);
    if (usableVram < modelGb * 1.05) {
      throw new Error(
        `Insufficient VRAM for rough model estimate. Usable ~${usableVram.toFixed(1)} GB, ` +
        `model ~${modelGb.toFixed(1)} GB (size ${modelBytes} B from ${resolved.source}).`
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
        estimated_model_gb: estimatedGb(modelBytes),
        model_size_bytes: modelBytes,
        model_size_source: resolved.source,
        model_object_sha256: modelSha256,
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

  // `estimateModelBytes` is deliberately removed rather than wrapped. It
  // returned a size with no indication of where the size came from, which is
  // precisely the ambiguity this module now exists to prevent; a compatibility
  // shim would keep that accessor alive as a supported way to get an
  // unattributed number. `resolveModelSize` replaces it and returns
  // { bytes, source }. Nothing in docs/ consumed the old export.
  const api = { normalizeManifest, buildPlan, degradeManifest, resolveModelSize, linkBwBetween, SCHEMA_VERSION };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.axmZombie = api;
})(typeof self !== "undefined" ? self : this);
