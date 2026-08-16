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
      // Engines installed on this node. A node offering none cannot execute a
      // model however much VRAM it has, so placement treats its memory as
      // unavailable rather than as capacity. Mirrors manifest.py.
      if (node.runtimes === undefined) node.runtimes = [];
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

  /* ----------------------------------------------------------------------
   * WL-02: prove a placement, or refuse it in a typed way.
   *
   * Faithful port of src/axm_zombie/placement.py. The published planner is a
   * second *writer of the plan artifact*, so it has to reach the same
   * admission decision as the normative planner or the page authorizes what
   * the reference implementation rejects.
   *
   * What this port may and may not do is the whole point. It evaluates an
   * evidence packet that the manifest supplies: a verified object identity, a
   * weight accounting, residence records, measured runtimes. It cannot
   * observe any of those - a browser has no view of a remote node's disk, no
   * way to run an engine, and no way to confirm a board is physically
   * present. So it never infers residence, runtime compatibility, or physical
   * execution from topology. A cluster description with no evidence packet
   * yields REPRESENTABLE_NOT_RUNNABLE: the topology is drawable and the plan
   * is a useful sketch of it, and the page says in a field that nothing here
   * establishes it can run.
   * -------------------------------------------------------------------- */

  const BYTES_PER_GB = Math.pow(1024, 3);
  const MIN_UUID_LENGTH = 8;
  const DIGEST_RE = /^[0-9a-f]{64}$/;

  const IDENTITY_VERIFIED = "VERIFIED";
  const RESIDENCE_CURRENT = "CURRENT";
  const RUNTIME_MEASURED = "MEASURED";
  const SHARDING_PIPELINE = "pipeline";
  const SUPPORTED_SHARDING_STRATEGIES = [SHARDING_PIPELINE];

  const WEIGHT_FIELDS = [
    "checkpoint_set_bytes",
    "tensor_payload_bytes",
    "container_overhead_bytes",
  ];

  const REFUSAL = {
    WEIGHT_ACCOUNTING_MISSING: "WEIGHT_ACCOUNTING_MISSING",
    WEIGHT_ACCOUNTING_INCONSISTENT: "WEIGHT_ACCOUNTING_INCONSISTENT",
    WEIGHT_ACCOUNTING_CONFLICT: "WEIGHT_ACCOUNTING_CONFLICT",
    UNVERIFIED_MODEL_IDENTITY: "UNVERIFIED_MODEL_IDENTITY",
    SEAT_IDENTITY_CONFLICT: "SEAT_IDENTITY_CONFLICT",
    RESIDENCE_UNVERIFIED: "RESIDENCE_UNVERIFIED",
    RESIDENCE_STALE: "RESIDENCE_STALE",
    RESIDENCE_IDENTITY_MISMATCH: "RESIDENCE_IDENTITY_MISMATCH",
    NO_SUPPORTED_ENGINE: "NO_SUPPORTED_ENGINE",
    RUNTIME_UNMEASURED: "RUNTIME_UNMEASURED",
    UNSUPPORTED_SHARDING_STRATEGY: "UNSUPPORTED_SHARDING_STRATEGY",
    STAGE_DOES_NOT_FIT: "STAGE_DOES_NOT_FIT",
    DEVICE_DOES_NOT_FIT: "DEVICE_DOES_NOT_FIT",
    NO_ADMISSIBLE_SEAT: "NO_ADMISSIBLE_SEAT",
  };

  const STATE_RUNNABLE = "RUNNABLE";
  const STATE_REPRESENTABLE_NOT_RUNNABLE = "REPRESENTABLE_NOT_RUNNABLE";

  const PREDICATE_OBJECT_EVIDENCE = "verified_object_identity";
  const PREDICATE_WEIGHT_ACCOUNTING = "declared_weight_accounting";
  const PREDICATE_RESIDENCE = "verified_residence";
  const PREDICATE_RUNTIME = "measured_runtime";

  // Mirrors placement.PlacementRefusal. Extends Error for the same reason the
  // Python class extends ValueError: callers written against the previous
  // planner catch the base type and keep working. New callers branch on
  // `.code` and read `.detail`, never on the message text.
  class PlacementRefusal extends Error {
    constructor(code, message, detail) {
      super(message);
      this.name = "PlacementRefusal";
      this.code = code;
      this.detail = detail || {};
    }
    toObject() {
      return {
        schema_version: 1,
        result: "refusal",
        code: this.code,
        message: this.message,
        detail: this.detail,
      };
    }
  }

  function refuse(code, message, detail) {
    return new PlacementRefusal(code, message, detail);
  }

  function has(object, key) {
    return Object.prototype.hasOwnProperty.call(object, key);
  }

  function text(value) {
    return String(value === undefined || value === null ? "" : value).trim();
  }

  // Mirrors placement.reconcile_weight_accounting. The three quantities are
  // one measurement of one object from three angles; when they do not close
  // the manifest is describing more than one object.
  function reconcileWeightAccounting(fields, name) {
    const payload = fields.tensor_payload_bytes;
    const overhead = fields.container_overhead_bytes;
    const checkpoint = fields.checkpoint_set_bytes;
    if (payload + overhead !== checkpoint) {
      throw refuse(
        REFUSAL.WEIGHT_ACCOUNTING_INCONSISTENT,
        `Weight accounting for '${name}' does not close: tensor payload ` +
        `${payload} B plus container overhead ${overhead} B is ` +
        `${payload + overhead} B, but the checkpoint set is declared as ` +
        `${checkpoint} B - a ${Math.abs(checkpoint - payload - overhead)} B ` +
        "discrepancy. These three quantities describe one object from three " +
        "angles; when they disagree the manifest is describing more than one " +
        "object and the planner will not choose between them.",
        {
          model: name,
          tensor_payload_bytes: payload,
          container_overhead_bytes: overhead,
          checkpoint_set_bytes: checkpoint,
          residual_bytes: checkpoint - payload - overhead,
        }
      );
    }
    return {
      checkpoint_set_bytes: checkpoint,
      tensor_payload_bytes: payload,
      container_overhead_bytes: overhead,
    };
  }

  // Mirrors planner._resolve_weight_accounting: the model-size authority
  // normalizes each field with validatedSizeField, placement decides meaning.
  function resolveWeightAccounting(model, modelBytes) {
    const name = String(model.name);
    if (!has(model, "weights")) return null;
    const declared = model.weights;
    if (typeof declared !== "object" || declared === null || Array.isArray(declared)) {
      throw refuse(
        REFUSAL.WEIGHT_ACCOUNTING_MISSING,
        `model.weights for '${name}' must be an object declaring ` +
        `${WEIGHT_FIELDS.join(", ")}, got ${JSON.stringify(declared)}.`,
        { model: name }
      );
    }
    const missing = WEIGHT_FIELDS.filter((f) => !has(declared, f));
    if (missing.length) {
      throw refuse(
        REFUSAL.WEIGHT_ACCOUNTING_MISSING,
        `model.weights for '${name}' is missing ${missing.join(", ")}. All ` +
        "three quantities are required together: the checkpoint set is what " +
        "custody binds, the tensor payload is what occupies device memory, " +
        "and the container overhead is the difference. Declaring some of " +
        "them leaves the others to be inferred, which is how one number ends " +
        "up standing for two different facts.",
        { model: name, missing: missing }
      );
    }
    const fields = {};
    for (const field of WEIGHT_FIELDS) {
      fields[field] = validatedSizeField(declared[field], `weights.${field}`, name);
    }
    const accounting = reconcileWeightAccounting(fields, name);
    if (accounting.tensor_payload_bytes !== modelBytes) {
      throw refuse(
        REFUSAL.WEIGHT_ACCOUNTING_CONFLICT,
        `model.bytes for '${name}' resolves to ${modelBytes} B but ` +
        `model.weights.tensor_payload_bytes is ` +
        `${accounting.tensor_payload_bytes} B. model.bytes is the size that ` +
        "authorizes the placement, so it must be the quantity that occupies " +
        "device memory - the tensor payload, not the checkpoint set and not " +
        "a third number.",
        {
          model: name,
          model_size_bytes: modelBytes,
          tensor_payload_bytes: accounting.tensor_payload_bytes,
        }
      );
    }
    return accounting;
  }

  // Mirrors placement.resolve_verified_identity. model.sha256 is a declaration;
  // model.identity is the binding that says a verification actually happened,
  // under which scheme, from which source, by whom, and when.
  function resolveVerifiedIdentity(model) {
    const name = String(model.name);
    const binding = model.identity;
    if (typeof binding !== "object" || binding === null || Array.isArray(binding)) {
      throw refuse(
        REFUSAL.UNVERIFIED_MODEL_IDENTITY,
        `model.identity is required to place '${name}', and must be an ` +
        "object binding the digest to its scheme, its source, its validator, " +
        "and the time it was verified. A bare model.sha256 declares a " +
        "digest; it does not establish that anyone computed one.",
        {
          model: name,
          declared_sha256: model.sha256 === undefined ? null : model.sha256,
        }
      );
    }

    const declaredState = text(binding.identity_state).toUpperCase();
    if (declaredState && declaredState !== IDENTITY_VERIFIED) {
      throw refuse(
        REFUSAL.UNVERIFIED_MODEL_IDENTITY,
        `model.identity for '${name}' declares identity_state=` +
        `'${declaredState}'. Placement authority requires ` +
        `'${IDENTITY_VERIFIED}': a digest that has been written down, or ` +
        "pointed at, but not verified in this reader's evidence is a claim " +
        "about an object, and a claim cannot authorize putting bytes on a " +
        "device.",
        {
          model: name,
          identity_state: declaredState,
          identity_scheme: text(binding.identity_scheme),
          identity_source: text(binding.identity_source),
        }
      );
    }

    const missing = [
      "identity_scheme", "identity_digest", "identity_source",
      "identity_state", "verified_at", "validator",
    ].filter((field) => !text(binding[field]));
    if (missing.length) {
      throw refuse(
        REFUSAL.UNVERIFIED_MODEL_IDENTITY,
        `model.identity for '${name}' is incomplete: ${missing.join(", ")} ` +
        "is missing or empty. An identity binding that omits any of scheme, " +
        "digest, source, state, validator, or verification time cannot be " +
        "checked by a reader, and an identity that cannot be checked is not " +
        "custody.",
        { model: name, missing: missing }
      );
    }

    const digest = text(binding.identity_digest).toLowerCase();
    if (!DIGEST_RE.test(digest)) {
      throw refuse(
        REFUSAL.UNVERIFIED_MODEL_IDENTITY,
        `model.identity.identity_digest for '${name}' must be 64 ` +
        `hexadecimal characters, got ${JSON.stringify(binding.identity_digest)}. ` +
        "A digest that cannot be parsed cannot be compared against a " +
        "residence record, so residence could never be verified against it.",
        { model: name, declared: JSON.stringify(binding.identity_digest) }
      );
    }

    if (model.sha256 !== undefined && model.sha256 !== null) {
      const normalized = text(model.sha256).toLowerCase();
      if (normalized !== digest) {
        throw refuse(
          REFUSAL.UNVERIFIED_MODEL_IDENTITY,
          `model.sha256 for '${name}' declares ${normalized} but the ` +
          `verified identity binding is ${digest} under scheme ` +
          `'${text(binding.identity_scheme)}'. Two identities for one object ` +
          "is not a precedence question: the manifest names two objects.",
          { model: name, declared_sha256: normalized, identity_digest: digest }
        );
      }
    }

    return {
      identity_scheme: text(binding.identity_scheme),
      identity_digest: digest,
      identity_source: text(binding.identity_source),
      identity_state: IDENTITY_VERIFIED,
      verified_at: text(binding.verified_at),
      validator: text(binding.validator),
    };
  }

  // Mirrors placement.build_seats. Refuses a manifest that seats one physical
  // accelerator twice, for every manifest and not only for placements: such a
  // manifest is wrong about the hardware before anyone asks it to hold a model.
  function buildSeats(manifest) {
    const seats = [];
    const seenBoards = new Map();
    for (const node of manifest.cluster.nodes) {
      const nodeId = String(node.id);
      for (const gpu of node.gpus) {
        const index = Math.trunc(gpu.index);
        let uuid = gpu.uuid === undefined || gpu.uuid === null ? null : String(gpu.uuid).trim();
        if (uuid !== null && uuid.length < MIN_UUID_LENGTH) uuid = null;
        if (uuid !== null) {
          const previous = seenBoards.get(uuid);
          if (previous !== undefined) {
            throw refuse(
              REFUSAL.SEAT_IDENTITY_CONFLICT,
              `Physical accelerator ${uuid} is declared in two seats: ` +
              `${previous[0]}:${previous[1]} and ${nodeId}:${index}. One ` +
              "board cannot occupy two seats, so their VRAM is not " +
              "independent and must not be summed. Correct the manifest; the " +
              "planner will not guess which seat is real.",
              {
                uuid: uuid,
                seats: [
                  { node: previous[0], gpu: previous[1] },
                  { node: nodeId, gpu: index },
                ],
              }
            );
          }
          seenBoards.set(uuid, [nodeId, index]);
        }
        seats.push({
          node: nodeId,
          host: node.host !== undefined ? node.host : null,
          gpu: index,
          name: gpu.name !== undefined ? gpu.name : "GPU",
          vram_gb: Number(gpu.vram_gb),
          mem_bw_gbps: gpu.mem_bw_gbps !== undefined ? gpu.mem_bw_gbps : null,
          // Seat capability. Never board identity.
          accelerator_uuid: uuid,
        });
      }
    }
    return seats;
  }

  // floor(x + 0.5), not Math.round: Python's round() is half-to-even, so
  // spelling the law out is what keeps the two writers agreeing about a
  // device's capacity. See placement.usable_bytes_for_seat.
  function usableBytesForSeat(seat, reserveGb) {
    const capacity = Math.floor(Number(seat.vram_gb) * BYTES_PER_GB + 0.5);
    const reserve = Math.floor(Number(reserveGb) * BYTES_PER_GB + 0.5);
    return Math.max(0, capacity - reserve);
  }

  function verifiedResidenceNodes(model, identity) {
    const digest = identity.identity_digest;
    const resident = new Map();
    for (const record of model.residence || []) {
      const nodeId = String(record.node === undefined ? "" : record.node);
      const declared = record.identity_digest;
      const verified = Boolean(record.verified);
      if (declared !== undefined && declared !== null &&
          text(declared).toLowerCase() !== digest) {
        throw refuse(
          REFUSAL.RESIDENCE_IDENTITY_MISMATCH,
          `Residence record for node '${nodeId}' names object ` +
          `${text(declared).toLowerCase()} but the model being placed is ` +
          `${digest}. That record describes a different object; it cannot ` +
          "verify residence of this one.",
          {
            node: nodeId,
            declared: text(declared).toLowerCase(),
            identity_digest: digest,
          }
        );
      }
      if (!verified) continue;
      const freshness = (record.freshness === undefined
        ? RESIDENCE_CURRENT
        : text(record.freshness)).toUpperCase();
      if (freshness !== RESIDENCE_CURRENT) {
        throw refuse(
          REFUSAL.RESIDENCE_STALE,
          `Residence of object ${digest} on node '${nodeId}' is declared ` +
          `'${freshness}', not '${RESIDENCE_CURRENT}'. Stale evidence records ` +
          "where the bytes were when they were last checked, which is not a " +
          "statement that they are there now.",
          { node: nodeId, freshness: freshness, identity_digest: digest }
        );
      }
      resident.set(nodeId, {
        node: nodeId,
        verified: true,
        freshness: RESIDENCE_CURRENT,
        path: record.path === undefined ? null : record.path,
        verified_at: record.verified_at === undefined ? null : record.verified_at,
        validator: record.validator === undefined ? null : record.validator,
      });
    }
    return resident;
  }

  // A bare string normalizes to UNMEASURED: naming software is not observing
  // it run. Mirrors placement._runtime_entries.
  function runtimeEntries(node) {
    const entries = [];
    for (const raw of node.runtimes || []) {
      let engine;
      let compatibility;
      if (typeof raw === "object" && raw !== null && !Array.isArray(raw)) {
        engine = text(raw.engine);
        compatibility = (raw.compatibility === undefined
          ? "UNMEASURED"
          : text(raw.compatibility)).toUpperCase();
      } else {
        engine = text(raw);
        compatibility = "UNMEASURED";
      }
      if (engine) entries.push({ engine: engine, compatibility: compatibility });
    }
    return entries;
  }

  function enginesForNode(node) {
    return runtimeEntries(node).map((e) => e.engine);
  }

  function measuredEnginesForNode(node) {
    return runtimeEntries(node)
      .filter((e) => e.compatibility === RUNTIME_MEASURED)
      .map((e) => e.engine);
  }

  function compatibleEngine(model, node) {
    const required = (model.engines || []).map(String);
    const available = measuredEnginesForNode(node);
    if (!available.length) return null;
    if (!required.length) return available.slice().sort()[0];
    const shared = Array.from(new Set(required.filter((e) => available.includes(e)))).sort();
    return shared.length ? shared[0] : null;
  }

  function resolveShardingStrategy(model) {
    const name = String(model.name);
    const strategy = (model.sharding_strategy === undefined
      ? SHARDING_PIPELINE
      : text(model.sharding_strategy)).toLowerCase();
    if (!SUPPORTED_SHARDING_STRATEGIES.includes(strategy)) {
      throw refuse(
        REFUSAL.UNSUPPORTED_SHARDING_STRATEGY,
        `model.sharding_strategy for '${name}' is '${strategy}'; this ` +
        `planner proves ` +
        `${SUPPORTED_SHARDING_STRATEGIES.map((s) => `'${s}'`).join(", ")} and ` +
        "nothing else. A proof constructed for a different decomposition " +
        "would not be a proof about the placement that was requested.",
        {
          model: name,
          requested: strategy,
          supported: SUPPORTED_SHARDING_STRATEGIES.slice(),
        }
      );
    }
    return strategy;
  }

  // BigInt, not Number: `total * weight` for a terabyte-scale model against a
  // 20 GiB device overflows the exact integer range by orders of magnitude,
  // and an apportionment computed in approximate arithmetic is exactly the
  // "no byte is lost or invented" guarantee failing silently.
  function apportionExact(total, weights) {
    if (total < 0) throw new Error("cannot apportion a negative total");
    if (!weights.length) return [];
    const bigWeights = weights.map((w) => BigInt(w));
    const weightSum = bigWeights.reduce((a, b) => a + b, 0n);
    if (weightSum <= 0n) return weights.map(() => 0);
    const bigTotal = BigInt(total);
    const numerators = bigWeights.map((w) => bigTotal * w);
    const parts = numerators.map((n) => n / weightSum);
    let remainder = bigTotal - parts.reduce((a, b) => a + b, 0n);
    const order = weights.map((_, i) => i).sort((a, b) => {
      const ra = numerators[a] % weightSum;
      const rb = numerators[b] % weightSum;
      if (ra !== rb) return ra > rb ? -1 : 1;
      return a - b;
    });
    for (let position = 0; remainder > 0n; position++, remainder--) {
      parts[order[position]] += 1n;
    }
    return parts.map((p) => Number(p));
  }

  function provePlacement(manifest, stages, weights, identity, reserveGb) {
    const nodesById = new Map(manifest.cluster.nodes.map((n) => [String(n.id), n]));
    const model = manifest.model;
    const strategy = resolveShardingStrategy(model);
    const resident = verifiedResidenceNodes(model, identity);
    const weightBytes = weights.tensor_payload_bytes;

    const stageRecords = stages.map((stageSeats, index) => ({
      stage: index,
      seats: stageSeats.map((seat) => {
        const node = nodesById.get(seat.node);
        const engine = compatibleEngine(model, node);
        const isResident = resident.has(seat.node);
        return {
          node: seat.node,
          gpu: seat.gpu,
          accelerator_uuid: seat.accelerator_uuid === undefined ? null : seat.accelerator_uuid,
          usable_bytes: usableBytesForSeat(seat, reserveGb),
          engine: engine,
          residence_verified: isResident,
          admissible: Boolean(engine) && isResident,
        };
      }),
    }));

    for (const record of stageRecords) {
      if (record.seats.some((s) => s.admissible)) continue;
      const withoutEngine = record.seats.filter((s) => !s.engine);
      const withoutResidence = record.seats.filter((s) => !s.residence_verified);
      const stageNodes = Array.from(new Set(record.seats.map((s) => s.node))).sort();
      if (withoutEngine.length) {
        const declared = {};
        const measured = {};
        for (const n of stageNodes) {
          declared[n] = enginesForNode(nodesById.get(n));
          measured[n] = measuredEnginesForNode(nodesById.get(n));
        }
        const unmeasuredOnly = stageNodes.some(
          (n) => declared[n].length && !measured[n].length
        );
        if (unmeasuredOnly) {
          throw refuse(
            REFUSAL.RUNTIME_UNMEASURED,
            `Stage ${record.stage} is placed on node(s) ` +
            `${stageNodes.join(", ")}, whose runtimes are declared but not ` +
            "measured. An engine nobody has observed executing anything is " +
            "not evidence that this model can run there; declare " +
            "compatibility=MEASURED only for a runtime that was actually " +
            "exercised.",
            {
              stage: record.stage, nodes: stageNodes,
              node_runtimes: declared, measured_runtimes: measured,
            }
          );
        }
        throw refuse(
          REFUSAL.NO_SUPPORTED_ENGINE,
          `Stage ${record.stage} is placed on node(s) ` +
          `${stageNodes.join(", ")}, which offer no runtime able to execute ` +
          `'${model.name}'. Device memory on a node that cannot execute the ` +
          "model is not placement capacity.",
          {
            stage: record.stage, nodes: stageNodes,
            model_engines: (model.engines || []).map(String),
            node_runtimes: declared,
          }
        );
      }
      if (withoutResidence.length) {
        throw refuse(
          REFUSAL.RESIDENCE_UNVERIFIED,
          `Stage ${record.stage} is placed on node(s) ` +
          `${stageNodes.join(", ")}, where object ${identity.identity_digest} ` +
          "is not verifiably resident. Weights that are not present cannot " +
          "be loaded, so that VRAM is not placement capacity either.",
          {
            stage: record.stage, nodes: stageNodes,
            identity_digest: identity.identity_digest,
            verified_residence_nodes: Array.from(resident.keys()).sort(),
          }
        );
      }
      throw refuse(
        REFUSAL.NO_ADMISSIBLE_SEAT,
        `Stage ${record.stage} has no admissible seat.`,
        { stage: record.stage, nodes: stageNodes }
      );
    }

    const stageCapacities = stageRecords.map((r) =>
      r.seats.filter((s) => s.admissible).reduce((sum, s) => sum + s.usable_bytes, 0)
    );
    if (stageCapacities.reduce((a, b) => a + b, 0) <= 0) {
      throw refuse(
        REFUSAL.NO_ADMISSIBLE_SEAT,
        "No admissible seat in the cluster has usable memory after the " +
        "per-device reserve.",
        { reserve_gb_per_gpu: reserveGb }
      );
    }

    const stageAssignments = apportionExact(weightBytes, stageCapacities);

    stageRecords.forEach((record, i) => {
      const assigned = stageAssignments[i];
      const capacity = stageCapacities[i];
      record.assigned_bytes = assigned;
      record.usable_bytes = capacity;
      if (assigned > capacity) {
        throw refuse(
          REFUSAL.STAGE_DOES_NOT_FIT,
          `Stage ${record.stage} is assigned ${assigned} B of weights but ` +
          `its own admissible devices hold ${capacity} B - short by ` +
          `${assigned - capacity} B. The cluster total is irrelevant here: ` +
          "these are the only devices this stage runs on.",
          {
            stage: record.stage, assigned_bytes: assigned,
            usable_bytes: capacity, deficit_bytes: assigned - capacity,
          }
        );
      }
      const admissible = record.seats.filter((s) => s.admissible);
      const deviceAssignments = apportionExact(
        assigned, admissible.map((s) => s.usable_bytes)
      );
      admissible.forEach((seat, j) => {
        seat.assigned_bytes = deviceAssignments[j];
        if (seat.assigned_bytes > seat.usable_bytes) {
          throw refuse(
            REFUSAL.DEVICE_DOES_NOT_FIT,
            `Device ${seat.node}:${seat.gpu} is assigned ` +
            `${seat.assigned_bytes} B but holds ${seat.usable_bytes} B after ` +
            `its reserve - short by ` +
            `${seat.assigned_bytes - seat.usable_bytes} B.`,
            {
              stage: record.stage, node: seat.node, gpu: seat.gpu,
              assigned_bytes: seat.assigned_bytes,
              usable_bytes: seat.usable_bytes,
              deficit_bytes: seat.assigned_bytes - seat.usable_bytes,
            }
          );
        }
      });
      for (const seat of record.seats) {
        if (seat.assigned_bytes === undefined) seat.assigned_bytes = 0;
      }
    });

    const assignedTotal = stageRecords.reduce((sum, r) => sum + r.assigned_bytes, 0);
    if (assignedTotal !== weightBytes) {
      throw refuse(
        REFUSAL.STAGE_DOES_NOT_FIT,
        `Internal apportionment error: assigned ${assignedTotal} B for a ` +
        `${weightBytes} B model.`,
        { assigned_bytes: assignedTotal, weight_bytes: weightBytes }
      );
    }

    return {
      identity: Object.assign({}, identity),
      sharding_strategy: strategy,
      weights: {
        checkpoint_set_bytes: weights.checkpoint_set_bytes,
        tensor_payload_bytes: weights.tensor_payload_bytes,
        container_overhead_bytes: weights.container_overhead_bytes,
      },
      placed_bytes_are: "tensor_payload_bytes",
      custody_binds: "checkpoint_set_bytes",
      assigned_bytes_total: assignedTotal,
      stages: stageRecords.map((r) => ({
        stage: r.stage,
        assigned_bytes: r.assigned_bytes,
        usable_bytes: r.usable_bytes,
        devices: r.seats.map((s) => ({
          node: s.node,
          gpu: s.gpu,
          accelerator_uuid: s.accelerator_uuid,
          assigned_bytes: s.assigned_bytes,
          usable_bytes: s.usable_bytes,
          engine: s.engine,
          residence_verified: s.residence_verified,
        })),
      })),
      diagnostics: {
        aggregate_usable_bytes: stageCapacities.reduce((a, b) => a + b, 0),
        aggregate_is_not_an_admission_criterion: true,
      },
    };
  }

  // Object-level evidence is the trigger, exactly as in planner.py. Topology
  // alone is a cluster description, not a claim that a measured object can run
  // on it, and this port has no way to turn one into the other.
  function placementIsAttempted(model) {
    return ["identity", "residence", "weights"].some((key) => has(model, key));
  }

  function missingPlacementPredicates(manifest) {
    const model = manifest.model;
    const missing = [];
    if (!has(model, "identity")) missing.push(PREDICATE_OBJECT_EVIDENCE);
    if (!has(model, "weights")) missing.push(PREDICATE_WEIGHT_ACCOUNTING);
    if (!(model.residence || []).length) missing.push(PREDICATE_RESIDENCE);
    if (!manifest.cluster.nodes.some((n) => measuredEnginesForNode(n).length)) {
      missing.push(PREDICATE_RUNTIME);
    }
    return missing;
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
    // Seats, not boards. Refuses a manifest that seats one accelerator twice
    // before anything is sized.
    const gpus = buildSeats(manifest);

    const resolved = resolveModelSize(model);
    const modelSha256 = validatedSha256(model.sha256, model.name);
    const modelBytes = resolved.bytes;
    const weights = resolveWeightAccounting(model, modelBytes);
    const modelGb = modelBytes / BYTES_PER_GB;

    const reserve = Number(policy.reserve_gb_per_gpu !== undefined ? policy.reserve_gb_per_gpu : 4.0);
    const attempted = placementIsAttempted(model);

    if (!attempted) {
      // The aggregate screen, and the exact limits of what it can do. It is a
      // *necessary* condition: a model larger than every device put together
      // certainly does not fit. It is never a sufficient one, because
      // independent devices do not share an address space - which is why it
      // can only refuse here, and why a plan that survives it is classified
      // representable rather than runnable. When placement evidence *is*
      // supplied this screen is skipped entirely and the proof decides.
      const totalVram = gpus.reduce((s, g) => s + g.vram_gb, 0);
      const usableVram = Math.max(0.0, totalVram - reserve * gpus.length);
      if (usableVram < modelGb * 1.05) {
        throw new Error(
          `Insufficient VRAM for rough model estimate. Usable ~${usableVram.toFixed(1)} GB, ` +
          `model ~${modelGb.toFixed(1)} GB (size ${modelBytes} B from ${resolved.source}).`
        );
      }
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

    let placement;
    if (attempted) {
      if (weights === null) {
        throw refuse(
          REFUSAL.WEIGHT_ACCOUNTING_MISSING,
          `Placement of '${model.name}' was requested but model.weights is ` +
          "absent. Device memory holds the tensor payload while custody " +
          "binds the complete checkpoint set; without both declared there is " +
          "no unambiguous number to place, and renaming either one 'model " +
          "bytes' is how the ambiguity gets rebuilt one layer down.",
          { model: String(model.name) }
        );
      }
      placement = {
        state: STATE_RUNNABLE,
        missing_predicates: [],
        proof: provePlacement(manifest, stages, weights, resolveVerifiedIdentity(model), reserve),
      };
    } else {
      // Representable, not runnable. This page can draw the topology and is
      // useful for doing so; it cannot observe a remote disk or run an engine,
      // so it must not label the result runnable. Saying which predicates are
      // absent is the difference between a sketch and a proof.
      placement = {
        state: STATE_REPRESENTABLE_NOT_RUNNABLE,
        missing_predicates: missingPlacementPredicates(manifest),
        proof: null,
      };
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
      placement: placement,
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
  const api = {
    normalizeManifest, buildPlan, degradeManifest, resolveModelSize,
    linkBwBetween, SCHEMA_VERSION,
    // WL-02 surface, exported so the parity gate can exercise the placement
    // pieces directly rather than only through a whole plan.
    PlacementRefusal, buildSeats, usableBytesForSeat, apportionExact,
    resolveVerifiedIdentity, resolveWeightAccounting, REFUSAL,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.axmZombie = api;
})(typeof self !== "undefined" ? self : this);
