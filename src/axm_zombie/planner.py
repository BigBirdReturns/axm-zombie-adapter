from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import re

from .placement import (
    BYTES_PER_GB,
    PREDICATE_OBJECT_EVIDENCE,
    PREDICATE_RESIDENCE,
    PREDICATE_RUNTIME,
    PREDICATE_WEIGHT_ACCOUNTING,
    REFUSAL_WEIGHT_ACCOUNTING_MISSING,
    STATE_REPRESENTABLE_NOT_RUNNABLE,
    STATE_RUNNABLE,
    WEIGHT_FIELDS,
    PlacementRefusal,
    build_seats,
    measured_engines_for_node,
    prove_placement,
    reconcile_weight_accounting,
    resolve_verified_identity,
)

# Distinguishes "the manifest did not declare this field" from "the manifest
# declared it as null". `None` cannot carry that distinction, and the two mean
# opposite things for a size claim.
_ABSENT = object()

_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8": 1.0,
    "int4": 0.5,
    "q4": 0.5,
}

# Parameter counts the name heuristic recognises, keyed by the token in
# billions. Tokens are matched whole (see _PARAM_TOKEN), so ordering is
# irrelevant: "170b" is not a 70B model and "107b" is not a 7B model.
_KNOWN_PARAM_TOKENS = {
    7: 7_000_000_000,
    13: 13_000_000_000,
    34: 34_000_000_000,
    70: 70_000_000_000,
}

# A parameter token is a complete number followed by "b". The leading
# alternation (rather than a lookbehind) keeps this pattern identical in
# JavaScript, so both plan writers recognise exactly the same names.
_PARAM_TOKEN = re.compile(r"(?:^|[^\d.])(\d+)\s*b(?![a-z0-9])")

# A model object identity is a sha-256 digest or it is not an identity.
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

# "8x7b", "4 x 22b": a mixture-of-experts multiplier makes a bare parameter
# substring a lie. mixtral-8x7b is ~46.7B, not 7B.
#
# The trailing delimiter is the same law _PARAM_TOKEN uses, and it has to be:
# `\b` treats "_" as a word character, so "mixtral-8x7b_model" ended the MoE
# match at a non-boundary and escaped this check -- while _PARAM_TOKEN's
# `(?![a-z0-9])` happily accepted the same "_" and read the name as a 7B model.
# One delimiter law for both patterns is what keeps a name from being MoE for
# the refusal and plain for the heuristic. The leading side is deliberately
# left unanchored: this pattern only ever refuses, so matching more of the
# string than _PARAM_TOKEN can is fail-closed.
_MOE_MULTIPLIER = re.compile(r"\d+\s*x\s*\d+\s*b(?![a-z0-9])")

# Sizes are exchanged as JSON numbers and read by a JavaScript writer whose
# only numeric type is a double. Above 2**53-1 a byte count no longer survives
# that round trip -- 9007199254740993 is read back as 9007199254740992 -- so a
# declaration beyond this magnitude is refused by both writers rather than
# silently rounded and then published as the size that authorized a placement.
MAX_EXACT_INTEGER = 2**53 - 1

# One gibibyte. `estimated_model_gb` is derived from this by exact integer
# arithmetic (see _estimated_gb) rather than by float rounding, because the two
# writers' native round() disagree at a decimal tie.
_GIB = BYTES_PER_GB

# How the size that authorized a placement was obtained. A plan carries this so
# a declared measurement and a name guess are never read as the same evidence.
SIZE_SOURCE_MANIFEST_BYTES = "manifest_bytes"
SIZE_SOURCE_MANIFEST_PARAMS = "manifest_params_and_quantization"
SIZE_SOURCE_NAME_HEURISTIC = "legacy_name_heuristic"

# Two declared size claims that disagree by this factor or more are a
# contradiction, not a precedence question. Quantization and packaging move the
# on-disk size by small factors; an order of magnitude means one field is stale,
# and silently preferring `bytes` would rebuild the false-plan defect through a
# different input channel.
SIZE_CONFLICT_FACTOR = 10.0


def _bytes_per_param(dtype: str) -> float:
    return _BYTES_PER_PARAM.get(str(dtype).lower(), 2.0)


def _estimated_gb(model_bytes: int) -> float:
    """Bytes -> GiB at two decimals under one explicit rounding law.

    Half-way values round up (away from zero). This is spelled out in exact
    integer arithmetic instead of `round(bytes / 1024**3, 2)` because the two
    plan writers disagree about what `round` means at a tie: Python rounds
    half to even and gives 1.12 for 1207959552 B, while JavaScript's
    Math.round rounds half up and gives 1.13. Neither is wrong; having two of
    them is. `scaled` is the exact numerator, so the comparison
    `2 * remainder >= _GIB` is the tie test with no floating point in it.
    """
    scaled = model_bytes * 100
    whole, remainder = divmod(scaled, _GIB)
    if 2 * remainder >= _GIB:
        whole += 1
    return whole / 100


def _validated_size_field(value: Any, field: str, name: str) -> int:
    """Refuse a size declaration that cannot describe a real object.

    A zero or negative size satisfies every feasibility comparison, so it
    authorizes any placement at all. That is the same false-authorization the
    name heuristic used to produce, reached through a declared field instead.

    This is the single normalizer for every declared byte count in the
    manifest, including WL-02's three weight-accounting quantities. Placement
    consumes what this returns; it does not re-parse the same fields with a
    second set of rules.
    """
    unit = "bytes" if field.endswith("bytes") else "parameters"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"model.{field} for {name!r} must be a positive whole number of "
            f"{unit}, got {value!r} ({type(value).__name__}). A size that is "
            "not a number cannot be compared against VRAM, and the planner "
            "will not coerce it into one."
        )
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(
                f"model.{field} for {name!r} must be finite, got {value!r}. "
                "A non-finite size never fails the feasibility check."
            )
        if not value.is_integer():
            raise ValueError(
                f"model.{field} for {name!r} must be a whole number of {unit}, "
                f"got {value!r}."
            )
    resolved = int(value)
    if abs(resolved) > MAX_EXACT_INTEGER:
        raise ValueError(
            f"model.{field} for {name!r} must lie within the exact integer "
            f"range of +/-{MAX_EXACT_INTEGER} {unit}, got {resolved}. Beyond "
            "that magnitude the value cannot be represented exactly by both "
            "plan writers, and rounding an asserted size would publish a "
            "number nobody asserted as the size that authorized a placement."
        )
    if resolved <= 0:
        raise ValueError(
            f"model.{field} for {name!r} must be greater than zero, got "
            f"{resolved}. A zero or negative size passes every feasibility "
            "check and would authorize any placement."
        )
    return resolved


def _validated_sha256(value: Any, name: str) -> Any:
    """Refuse to publish an unparseable string as model object identity."""
    if value is None:
        return None
    if isinstance(value, str) and _SHA256.match(value.strip().lower()):
        return value.strip().lower()
    raise ValueError(
        f"model.sha256 for {name!r} must be 64 hexadecimal characters "
        f"identifying the measured model object, got {value!r}. An "
        "unverifiable identity must not be emitted as model_object_sha256, "
        "because the whole point of the field is that it can be checked."
    )


def _refuse_contradictory_size_claims(
    name: str, declared: int, params: Any, implied: int, dtype: str
) -> None:
    """Fail closed when `bytes` and `params` describe different objects."""
    lo, hi = sorted((declared, implied))
    ratio = float("inf") if lo <= 0 else hi / lo
    if ratio < SIZE_CONFLICT_FACTOR:
        return
    gap = "unbounded" if ratio == float("inf") else f"{ratio:.1f}x"
    raise ValueError(
        f"Size-claim conflict for {name!r}: model.bytes declares {declared} B, "
        f"but model.params ({params}) at {_bytes_per_param(dtype)} bytes/param "
        f"for dtype {dtype!r} implies {implied} B - a {gap} disagreement. The "
        "planner will not pick a winner between two contradictory claims, "
        "because a stale size field produces a confident plan that OOMs. "
        "Correct model.bytes or model.params so the two agree within a factor "
        f"of {SIZE_CONFLICT_FACTOR:.0f}."
    )


def _resolve_model_size(model: Dict[str, Any]) -> tuple[int, str]:
    """Return (size in bytes, how that size was obtained).

    Declared size always wins over inference from the name. When the size can be
    neither declared nor inferred unambiguously, this refuses instead of
    guessing: a wrong guess here is emitted as a confident placement plan that
    OOMs on contact with real hardware. The source travels with the size so a
    downstream reader can tell a measured object from a name guess.
    """
    dtype = model.get("dtype", "")
    name = str(model.get("name", ""))

    # Presence and value are separate facts. An absent `bytes` key means "no
    # size was declared, fall through to params or the name"; a present
    # `bytes: null` means "a size was declared and it is nothing", which is not
    # a size. Reading both as None let an explicitly empty declaration fall
    # through to the name heuristic, so a manifest that declared its size ended
    # up authorized by a guess -- and said so in `model_size_source`.
    declared_bytes = model["bytes"] if "bytes" in model else _ABSENT
    declared_params = model["params"] if "params" in model else _ABSENT

    if declared_bytes is not _ABSENT:
        size = _validated_size_field(declared_bytes, "bytes", name)
        if declared_params is not _ABSENT:
            params = _validated_size_field(declared_params, "params", name)
            implied = int(params * _bytes_per_param(dtype))
            _refuse_contradictory_size_claims(name, size, params, implied, dtype)
        return size, SIZE_SOURCE_MANIFEST_BYTES

    if declared_params is not _ABSENT:
        params = _validated_size_field(declared_params, "params", name)
        return (
            int(params * _bytes_per_param(dtype)),
            SIZE_SOURCE_MANIFEST_PARAMS,
        )

    lowered = name.lower()
    if _MOE_MULTIPLIER.search(lowered):
        raise ValueError(
            f"Model {name!r} carries a mixture-of-experts multiplier, so its "
            "parameter count cannot be read from its name. Declare model.params "
            "(total parameters, not per-expert) or model.bytes in the manifest."
        )
    tokens = sorted({int(m.group(1)) for m in _PARAM_TOKEN.finditer(lowered)})
    if len(tokens) == 1 and tokens[0] in _KNOWN_PARAM_TOKENS:
        return (
            int(_KNOWN_PARAM_TOKENS[tokens[0]] * _bytes_per_param(dtype)),
            SIZE_SOURCE_NAME_HEURISTIC,
        )
    if tokens:
        raise ValueError(
            f"Model name {name!r} carries parameter token(s) "
            f"{', '.join(f'{t}b' for t in tokens)} that the size heuristic "
            f"cannot resolve; it recognises exactly "
            f"{', '.join(f'{t}b' for t in sorted(_KNOWN_PARAM_TOKENS))} and "
            "will not round a name to the nearest supported size. Declare "
            "model.params or model.bytes in the manifest."
        )
    raise ValueError(
        f"Unknown model size for {name!r}. The planner will not assume a "
        "default parameter count, because the resulting plan would look "
        "identical to a correct one. Declare model.params or model.bytes "
        "in the manifest."
    )


# --------------------------------------------------------------------------
# WL-02 intake: weight accounting and the runnable/representable question
# --------------------------------------------------------------------------

# A size authority that disagrees with the weight accounting is the same defect
# class as `bytes` disagreeing with `params`: two numbers, one object, no
# principled way to choose. Named here rather than in placement.py because the
# contradiction is between the two layers, not inside either one.
REFUSAL_WEIGHT_ACCOUNTING_CONFLICT = "WEIGHT_ACCOUNTING_CONFLICT"


def _resolve_weight_accounting(
    model: Dict[str, Any], model_bytes: int
) -> Optional[Dict[str, int]]:
    """The three weight quantities, normalized and reconciled, or None.

    Normalization is the model-size authority's job and is done here with the
    same `_validated_size_field` every other declared byte count goes through.
    Reconciliation - whether the three describe one object - belongs to
    placement, which owns what the quantities *mean*.

    Returns None when `model.weights` is absent, which is a complete answer:
    the manifest simply did not declare the accounting, and the caller decides
    whether that makes the plan not-runnable or a refusal.
    """
    name = str(model.get("name", ""))
    if "weights" not in model:
        return None
    declared = model["weights"]
    if not isinstance(declared, dict):
        raise PlacementRefusal(
            REFUSAL_WEIGHT_ACCOUNTING_MISSING,
            f"model.weights for {name!r} must be an object declaring "
            f"{', '.join(WEIGHT_FIELDS)}, got {declared!r}.",
            model=name,
        )
    missing = [field for field in WEIGHT_FIELDS if field not in declared]
    if missing:
        raise PlacementRefusal(
            REFUSAL_WEIGHT_ACCOUNTING_MISSING,
            f"model.weights for {name!r} is missing {', '.join(missing)}. All "
            "three quantities are required together: the checkpoint set is "
            "what custody binds, the tensor payload is what occupies device "
            "memory, and the container overhead is the difference. Declaring "
            "some of them leaves the others to be inferred, which is how one "
            "number ends up standing for two different facts.",
            model=name, missing=missing,
        )
    fields = {
        field: _validated_size_field(declared[field], f"weights.{field}", name)
        for field in WEIGHT_FIELDS
    }
    accounting = reconcile_weight_accounting(fields, name)

    if accounting["tensor_payload_bytes"] != model_bytes:
        raise PlacementRefusal(
            REFUSAL_WEIGHT_ACCOUNTING_CONFLICT,
            f"model.bytes for {name!r} resolves to {model_bytes} B but "
            f"model.weights.tensor_payload_bytes is "
            f"{accounting['tensor_payload_bytes']} B. model.bytes is the size "
            "that authorizes the placement, so it must be the quantity that "
            "occupies device memory - the tensor payload, not the checkpoint "
            "set and not a third number.",
            model=name, model_size_bytes=model_bytes,
            tensor_payload_bytes=accounting["tensor_payload_bytes"],
        )
    return accounting


def _placement_is_attempted(model: Dict[str, Any]) -> bool:
    """Whether this manifest is asking for a placement to be proved.

    Object-level evidence is the trigger: an identity binding, residence
    records, or weight accounting. Topology alone is not - a manifest that
    describes hardware and a model name is describing a cluster, not asserting
    that a specific measured object can run on it, and refusing it would break
    every manifest written before this evidence existed for no safety gain.

    What is *not* on this list matters as much as what is. Seat UUIDs and node
    runtimes are hardware description; they do not by themselves claim that any
    object is placeable.
    """
    return any(key in model for key in ("identity", "residence", "weights"))


def _missing_placement_predicates(manifest: Dict[str, Any]) -> List[str]:
    """Which placement predicates this manifest never claimed at all."""
    model = manifest["model"]
    missing: List[str] = []
    if "identity" not in model:
        missing.append(PREDICATE_OBJECT_EVIDENCE)
    if "weights" not in model:
        missing.append(PREDICATE_WEIGHT_ACCOUNTING)
    if not (model.get("residence") or []):
        missing.append(PREDICATE_RESIDENCE)
    if not any(
        measured_engines_for_node(node)
        for node in manifest["cluster"]["nodes"]
    ):
        missing.append(PREDICATE_RUNTIME)
    return missing


def _link_bw_between(manifest: Dict[str, Any], a: str, b: str) -> float:
    if a == b:
        return float("inf")
    for node in manifest["cluster"]["nodes"]:
        if node["id"] != a:
            continue
        for link in node.get("links", []):
            if link.get("to") == b and link.get("gbps") is not None:
                return float(link["gbps"])
    return 0.0

def _score_stage_boundary(manifest: Dict[str, Any], left_nodes: List[str], right_nodes: List[str], activation_gb_per_token: float, target_tps: float) -> float:
    if not left_nodes or not right_nodes:
        return 0.0
    bw = 0.0
    for a in left_nodes:
        for b in right_nodes:
            bw = max(bw, _link_bw_between(manifest, a, b))
    if bw <= 0.0:
        return 1e9
    required_gbps = activation_gb_per_token * target_tps * 8.0
    util = required_gbps / bw
    if util > 0.7:
        return 1e6 * util
    return 1e3 * util

def build_plan(manifest: Dict[str, Any]) -> Dict[str, Any]:
    model = manifest["model"]
    policy = manifest["policy"]

    # Seats, not boards. This refuses a manifest that seats one accelerator
    # twice before anything is sized, because such a manifest is wrong about
    # the hardware whether or not a placement is being attempted.
    gpus = build_seats(manifest)

    model_bytes, size_source = _resolve_model_size(model)
    model_sha256 = _validated_sha256(model.get("sha256"), model["name"])
    weights = _resolve_weight_accounting(model, model_bytes)
    model_gb = model_bytes / _GIB

    reserve_gb_per_gpu = float(policy.get("reserve_gb_per_gpu", 4.0))
    attempted = _placement_is_attempted(model)

    if not attempted:
        # The aggregate screen, and the exact limits of what it can do. It is a
        # *necessary* condition: a model larger than every device put together
        # certainly does not fit. It is never a sufficient one, because
        # independent devices do not share an address space - which is why it
        # can only refuse here, and why a plan that survives it is classified
        # representable rather than runnable. When placement evidence *is*
        # supplied this screen is skipped entirely: the proof below decides,
        # and it names the precise reason instead of an arithmetic result.
        total_vram = sum(g["vram_gb"] for g in gpus)
        usable_vram = max(0.0, total_vram - reserve_gb_per_gpu * len(gpus))
        if usable_vram < model_gb * 1.05:
            raise ValueError(
                f"Insufficient VRAM for rough model estimate. Usable ~{usable_vram:.1f} GB, "
                f"model ~{model_gb:.1f} GB (size {model_bytes} B from {size_source})."
            )

    node_ids = sorted(list({g["node"] for g in gpus}))
    multi_node = len(node_ids) > 1

    stage_count = len(node_ids) if multi_node else max(1, math.ceil(len(gpus) / 2))

    stages: List[List[Dict[str, Any]]] = []
    if multi_node:
        for nid in node_ids:
            stages.append([g for g in gpus if g["node"] == nid])
    else:
        def sort_key(g):
            bw = g["mem_bw_gbps"]
            bwv = float(bw) if bw is not None else 0.0
            return (bwv, g["vram_gb"])
        sorted_gpus = sorted(gpus, key=sort_key, reverse=True)
        chunk = math.ceil(len(sorted_gpus) / stage_count)
        for i in range(stage_count):
            stages.append(sorted_gpus[i * chunk : (i + 1) * chunk])

    activation_gb_per_token = 0.002
    boundary_costs: List[float] = []
    for i in range(len(stages) - 1):
        left_nodes = sorted(list({g["node"] for g in stages[i]}))
        right_nodes = sorted(list({g["node"] for g in stages[i + 1]}))
        boundary_costs.append(
            _score_stage_boundary(
                manifest, left_nodes, right_nodes, activation_gb_per_token, float(policy.get("target_tps", 18))
            )
        )

    if attempted:
        if weights is None:
            raise PlacementRefusal(
                REFUSAL_WEIGHT_ACCOUNTING_MISSING,
                f"Placement of {model['name']!r} was requested but "
                "model.weights is absent. Device memory holds the tensor "
                "payload while custody binds the complete checkpoint set; "
                "without both declared there is no unambiguous number to "
                "place, and renaming either one 'model bytes' is how the "
                "ambiguity gets rebuilt one layer down.",
                model=str(model["name"]),
            )
        identity = resolve_verified_identity(model)
        placement = {
            "state": STATE_RUNNABLE,
            "missing_predicates": [],
            "proof": prove_placement(
                manifest, stages, weights, identity, reserve_gb_per_gpu
            ),
        }
    else:
        # Representable, not runnable. The topology is describable and the plan
        # is a useful sketch of it, but nothing here establishes that a
        # specific measured object can be loaded and executed on these
        # devices. Saying so in a field is the whole point: the old head's
        # failure was not that it computed the wrong number, it was that a
        # sketch and a proof came out looking the same.
        placement = {
            "state": STATE_REPRESENTABLE_NOT_RUNNABLE,
            "missing_predicates": _missing_placement_predicates(manifest),
            "proof": None,
        }

    return {
        "schema_version": 1,
        "model": {
            "name": model["name"],
            "dtype": model["dtype"],
            "kv_cache": model.get("kv_cache", model["dtype"]),
            "estimated_model_gb": _estimated_gb(model_bytes),
            "model_size_bytes": model_bytes,
            "model_size_source": size_source,
            "model_object_sha256": model_sha256,
        },
        "cluster": {
            "name": manifest["cluster"].get("name", "zombie"),
            "nodes": node_ids,
            "gpus": gpus,
        },
        "policy": policy,
        "pipeline_stages": [
            {"stage": i, "placement": [{"node": g["node"], "gpu": g["gpu"]} for g in stage]}
            for i, stage in enumerate(stages)
        ],
        "placement": placement,
        "kv_cache_policy": {
            "reserve_gb_per_gpu": reserve_gb_per_gpu,
            "spill_allowed": bool(policy.get("allow_cpu_offload", False)),
        },
        "health": {"replan_on_gpu_oom": True, "replan_on_node_loss": True},
        "diagnostics": {
            "stage_boundary_costs": boundary_costs,
            "notes": "Planner is heuristic. Tune KV cache and activation sizing per model for accurate TPS estimates.",
        },
    }
