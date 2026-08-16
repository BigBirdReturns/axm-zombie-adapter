from __future__ import annotations
from typing import Any, Dict, List
import math
import re

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
_MOE_MULTIPLIER = re.compile(r"\d+\s*x\s*\d+\s*b\b")

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


def _validated_size_field(value: Any, field: str, name: str) -> int:
    """Refuse a size declaration that cannot describe a real object.

    A zero or negative size satisfies every feasibility comparison, so it
    authorizes any placement at all. That is the same false-authorization the
    name heuristic used to produce, reached through a declared field instead.
    """
    unit = "bytes" if field == "bytes" else "parameters"
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

    declared_bytes = model.get("bytes")
    declared_params = model.get("params")

    if declared_bytes is not None:
        size = _validated_size_field(declared_bytes, "bytes", name)
        if declared_params is not None:
            params = _validated_size_field(declared_params, "params", name)
            implied = int(params * _bytes_per_param(dtype))
            _refuse_contradictory_size_claims(name, size, params, implied, dtype)
        return size, SIZE_SOURCE_MANIFEST_BYTES

    if declared_params is not None:
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

def _cluster_gpus(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    for node in manifest["cluster"]["nodes"]:
        for gpu in node["gpus"]:
            gpus.append({
                "node": node["id"],
                "host": node.get("host"),
                "gpu": int(gpu["index"]),
                "name": gpu.get("name", "GPU"),
                "vram_gb": float(gpu["vram_gb"]),
                "mem_bw_gbps": gpu.get("mem_bw_gbps"),
            })
    return gpus

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
    gpus = _cluster_gpus(manifest)

    total_vram = sum(g["vram_gb"] for g in gpus)
    model_bytes, size_source = _resolve_model_size(model)
    model_sha256 = _validated_sha256(model.get("sha256"), model["name"])
    model_gb = model_bytes / (1024**3)

    reserve_gb_per_gpu = float(policy.get("reserve_gb_per_gpu", 4.0))
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

    return {
        "schema_version": 1,
        "model": {
            "name": model["name"],
            "dtype": model["dtype"],
            "kv_cache": model.get("kv_cache", model["dtype"]),
            "estimated_model_gb": round(model_gb, 2),
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
