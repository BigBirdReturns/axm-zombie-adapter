"""WL-02: prove a placement, or refuse it in a typed way.

The defect this module closes is not a bad estimate. It is a *category error*
in what counts as evidence that a model fits:

    usable = sum(every gpu.vram_gb) - reserve * gpu_count
    if usable >= model_gb * 1.05: authorize the placement

That sum is a fictitious pooled address space. Independent devices do not share
one, so aggregate capacity is at best a necessary condition and never a
sufficient one. It admits three placements that cannot run:

  1. Two seats that are actually the same physical accelerator. The board's
     VRAM is counted twice, so the pool reports capacity that does not exist.
  2. Seats on a node where the weights are not verifiably resident. Those bytes
     cannot be loaded there, but the pool counts the VRAM anyway.
  3. Seats on a node with no engine able to execute the model. Same problem:
     capacity that cannot be used is still capacity in the sum.

In every case the old head emits a confident plan, byte-identical in shape to a
correct one, that fails on contact with hardware.

So this module never sums independent VRAM to admit anything. It assigns exact
weight bytes to every stage and to every device inside that stage, and it
admits a stage only when that stage's *own* seats can hold their *own* assigned
bytes. Aggregate capacity is reported as diagnostics and is explicitly not an
admission criterion.

Two objects are kept apart throughout, for the reason the estate's own base
seal records: capability predicates establish what a *seat* can do, never which
*board* is in it.

    AcceleratorSeat      a position in the cluster: (node id, device index).
                         Carries capability - VRAM, bandwidth.
    PhysicalAccelerator  one physical board, identified by its GPU UUID.
                         Globally unique and immutable when declared.

One board cannot occupy two seats. When two seats declare the same UUID the
manifest is describing one accelerator twice, and summing their VRAM invents
memory. That is refused rather than averaged, because there is no correct plan
to fall back to.

## What this module does not own

Declared *size and identity syntax* belongs to the model-size authority in
`planner.py`: field presence versus value, positive whole numbers, the exact
integer range, sha-256 shape, and the name heuristic's disclosure. This module
consumes those normalized results and never re-parses a manifest field that
another layer has already normalized. Two parsers for one field is how the two
layers start disagreeing about what a manifest said.

What this module adds on top of that syntax is *evidence*: a declared digest is
a claim, and a claim is not custody. Placement authority requires an identity
binding that states its scheme, its digest, where the verification came from,
who performed it, and when - and that says in a field, not by implication, that
the verification actually happened.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math
import re

# The existing v1 convention: manifest memory is declared in `_gb` fields and
# compared against binary gigabytes. WL-02 does not change the unit - SPEC.md
# rule 3 makes a field's unit permanent within a schema version - it only stops
# doing arithmetic in floating-point GB once exact bytes are available.
BYTES_PER_GB = 1024 ** 3

# A physical accelerator identity, as reported by the driver. Vendor-neutral in
# shape: a non-empty opaque string. This module never parses meaning out of it;
# it only requires that one board does not appear in two seats.
_MIN_UUID_LENGTH = 8

# An identity digest is a sha-256 digest or it is not a digest. Shape only:
# whether the *scheme* that produced it is one a consumer trusts is the
# consumer's decision, and the scheme travels in the plan so that decision can
# be made downstream instead of being guessed here.
_DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")

# The one identity state that authorizes a placement. Anything else - a bare
# declaration, a pointer to evidence that lives somewhere this reader cannot
# see, an explicit refusal upstream - is not custody.
IDENTITY_VERIFIED = "VERIFIED"

# Residence that is known to be behind the object it describes cannot authorize
# a load. Freshness is a fact about evidence, not about bytes.
RESIDENCE_CURRENT = "CURRENT"

# A runtime that has never been observed executing anything is not a runtime
# you can place against. The estate's own Rung 1 review found this exact
# defect: `UNMEASURED` was read as compatible, so a placement was authorized
# against a runtime nobody had ever run.
RUNTIME_MEASURED = "MEASURED"

# The only weight-distribution strategy this planner knows how to prove. A
# manifest asking for any other strategy is asking for a proof this code cannot
# construct, and saying so is the only honest answer.
SHARDING_PIPELINE = "pipeline"
SUPPORTED_SHARDING_STRATEGIES = (SHARDING_PIPELINE,)


# Typed refusal codes. These are machine-readable and permanent: a downstream
# reader branches on the code, never on the prose.
REFUSAL_WEIGHT_ACCOUNTING_MISSING = "WEIGHT_ACCOUNTING_MISSING"
REFUSAL_WEIGHT_ACCOUNTING_INCONSISTENT = "WEIGHT_ACCOUNTING_INCONSISTENT"
REFUSAL_UNVERIFIED_MODEL_IDENTITY = "UNVERIFIED_MODEL_IDENTITY"
REFUSAL_SEAT_IDENTITY_CONFLICT = "SEAT_IDENTITY_CONFLICT"
REFUSAL_RESIDENCE_UNVERIFIED = "RESIDENCE_UNVERIFIED"
REFUSAL_RESIDENCE_STALE = "RESIDENCE_STALE"
REFUSAL_RESIDENCE_IDENTITY_MISMATCH = "RESIDENCE_IDENTITY_MISMATCH"
REFUSAL_NO_SUPPORTED_ENGINE = "NO_SUPPORTED_ENGINE"
REFUSAL_RUNTIME_UNMEASURED = "RUNTIME_UNMEASURED"
REFUSAL_UNSUPPORTED_SHARDING_STRATEGY = "UNSUPPORTED_SHARDING_STRATEGY"
REFUSAL_STAGE_DOES_NOT_FIT = "STAGE_DOES_NOT_FIT"
REFUSAL_DEVICE_DOES_NOT_FIT = "DEVICE_DOES_NOT_FIT"
REFUSAL_NO_ADMISSIBLE_SEAT = "NO_ADMISSIBLE_SEAT"

# Placement classification. A plan is one of exactly three things, and the third
# is why this module can be strict without breaking every manifest that was
# written before placement evidence existed.
STATE_RUNNABLE = "RUNNABLE"
STATE_REPRESENTABLE_NOT_RUNNABLE = "REPRESENTABLE_NOT_RUNNABLE"

# Predicate names used when a plan is representable but not runnable. Same
# vocabulary as the refusal codes, so an operator reading "what is missing"
# and an operator reading "what was contradicted" are reading one language.
PREDICATE_OBJECT_EVIDENCE = "verified_object_identity"
PREDICATE_WEIGHT_ACCOUNTING = "declared_weight_accounting"
PREDICATE_RESIDENCE = "verified_residence"
PREDICATE_RUNTIME = "measured_runtime"


class PlacementRefusal(ValueError):
    """A refusal that carries a machine-readable reason.

    Subclasses ValueError so that callers written against the v1 planner, which
    raised bare ValueError for infeasible placements, keep working unchanged.
    New callers should branch on `.code` and read `.detail`, never on the
    message text.
    """

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail: Dict[str, Any] = dict(detail)

    def to_dict(self) -> Dict[str, Any]:
        """The refusal as a serializable artifact.

        A refusal is a real output of this planner, not an absence of output,
        so it has to be writable to disk and diffable like a plan.
        """
        return {
            "schema_version": 1,
            "result": "refusal",
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }


def _refuse(code: str, message: str, **detail: Any) -> "PlacementRefusal":
    return PlacementRefusal(code, message, **detail)


# --------------------------------------------------------------------------
# Weight accounting: three quantities, three meanings
# --------------------------------------------------------------------------
#
# The estate's own measurement of Kimi-K3 closes two byte identities that are
# not the same number:
#
#     tensor payload (index total_size)   1560860324864
#     container/header residual              75766584
#     complete checkpoint set             1560936091448
#
# Object custody binds the complete checkpoint set: that is what was hashed and
# what is on disk. VRAM placement consumes the tensor payload: safetensors
# headers are parsed on the host and never occupy device memory. Calling either
# one "model bytes" and moving on is how a 76 MB discrepancy becomes an
# ambiguity that some later layer resolves by guessing.

WEIGHT_FIELDS = (
    "checkpoint_set_bytes",
    "tensor_payload_bytes",
    "container_overhead_bytes",
)


def reconcile_weight_accounting(
    fields: Mapping[str, int], name: str
) -> Dict[str, int]:
    """Check the three declared quantities against each other.

    `fields` arrives already normalized by the model-size authority - each
    value is a positive whole number inside the exact integer range - so the
    only question left here is whether the three describe one object. They do
    when the payload and the container residual sum to the checkpoint set. A
    manifest where they do not is describing two different objects with one
    name, and picking a winner between them is the defect this whole lane
    exists to remove.
    """
    payload = fields["tensor_payload_bytes"]
    overhead = fields["container_overhead_bytes"]
    checkpoint = fields["checkpoint_set_bytes"]
    if payload + overhead != checkpoint:
        raise _refuse(
            REFUSAL_WEIGHT_ACCOUNTING_INCONSISTENT,
            f"Weight accounting for {name!r} does not close: tensor payload "
            f"{payload} B plus container overhead {overhead} B is "
            f"{payload + overhead} B, but the checkpoint set is declared as "
            f"{checkpoint} B - a {abs(checkpoint - payload - overhead)} B "
            "discrepancy. These three quantities describe one object from "
            "three angles; when they disagree the manifest is describing more "
            "than one object and the planner will not choose between them.",
            model=name,
            tensor_payload_bytes=payload,
            container_overhead_bytes=overhead,
            checkpoint_set_bytes=checkpoint,
            residual_bytes=checkpoint - payload - overhead,
        )
    return {
        "checkpoint_set_bytes": checkpoint,
        "tensor_payload_bytes": payload,
        "container_overhead_bytes": overhead,
    }


# --------------------------------------------------------------------------
# Object identity: a declaration is not custody
# --------------------------------------------------------------------------

def resolve_verified_identity(model: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the verified object identity binding, or refuse.

    `model.sha256` is a *syntactically valid declaration* and nothing more. It
    says a digest was written down; it does not say anyone computed it, what
    scheme produced it, what it binds, or when. Placement needs custody, so it
    reads `model.identity` - a binding that has to state all of that - and
    refuses when the binding says the verification did not happen.

    The scheme is carried through into the plan rather than being judged here.
    Which schemes count as custody is a policy decision belonging to whoever
    consumes the plan; what this planner guarantees is that the plan says which
    scheme was used, so that decision is possible at all.
    """
    name = str(model.get("name", ""))
    binding = model.get("identity")
    if not isinstance(binding, Mapping):
        raise _refuse(
            REFUSAL_UNVERIFIED_MODEL_IDENTITY,
            f"model.identity is required to place {name!r}, and must be an "
            "object binding the digest to its scheme, its source, its "
            "validator, and the time it was verified. A bare model.sha256 "
            "declares a digest; it does not establish that anyone computed "
            "one.",
            model=name,
            declared_sha256=model.get("sha256"),
        )

    # State first, when it is declared. A binding that says outright that the
    # verification did not happen here should be refused for *that* reason;
    # reporting its empty digest field instead would describe a symptom of the
    # honest declaration rather than the declaration itself.
    declared_state = str(binding.get("identity_state", "") or "").strip().upper()
    if declared_state and declared_state != IDENTITY_VERIFIED:
        raise _refuse(
            REFUSAL_UNVERIFIED_MODEL_IDENTITY,
            f"model.identity for {name!r} declares identity_state="
            f"{declared_state!r}. Placement authority requires "
            f"{IDENTITY_VERIFIED!r}: a digest that has been written down, or "
            "pointed at, but not verified in this reader's evidence is a "
            "claim about an object, and a claim cannot authorize putting "
            "bytes on a device.",
            model=name,
            identity_state=declared_state,
            identity_scheme=str(binding.get("identity_scheme", "") or "").strip(),
            identity_source=str(binding.get("identity_source", "") or "").strip(),
        )

    missing = [
        field
        for field in (
            "identity_scheme",
            "identity_digest",
            "identity_source",
            "identity_state",
            "verified_at",
            "validator",
        )
        if not str(binding.get(field, "") or "").strip()
    ]
    if missing:
        raise _refuse(
            REFUSAL_UNVERIFIED_MODEL_IDENTITY,
            f"model.identity for {name!r} is incomplete: "
            f"{', '.join(missing)} is missing or empty. An identity binding "
            "that omits any of scheme, digest, source, state, validator, or "
            "verification time cannot be checked by a reader, and an identity "
            "that cannot be checked is not custody.",
            model=name, missing=missing,
        )

    digest = str(binding["identity_digest"]).strip().lower()
    if not _DIGEST.match(digest):
        raise _refuse(
            REFUSAL_UNVERIFIED_MODEL_IDENTITY,
            f"model.identity.identity_digest for {name!r} must be 64 "
            f"hexadecimal characters, got {binding['identity_digest']!r}. A "
            "digest that cannot be parsed cannot be compared against a "
            "residence record, so residence could never be verified against "
            "it.",
            model=name, declared=repr(binding["identity_digest"]),
        )

    declared_sha256 = model.get("sha256")
    if declared_sha256 is not None:
        normalized = str(declared_sha256).strip().lower()
        if normalized != digest:
            raise _refuse(
                REFUSAL_UNVERIFIED_MODEL_IDENTITY,
                f"model.sha256 for {name!r} declares {normalized} but the "
                f"verified identity binding is {digest} under scheme "
                f"{str(binding['identity_scheme']).strip()!r}. Two identities "
                "for one object is not a precedence question: the manifest "
                "names two objects.",
                model=name, declared_sha256=normalized, identity_digest=digest,
            )

    return {
        "identity_scheme": str(binding["identity_scheme"]).strip(),
        "identity_digest": digest,
        "identity_source": str(binding["identity_source"]).strip(),
        "identity_state": IDENTITY_VERIFIED,
        "verified_at": str(binding["verified_at"]).strip(),
        "validator": str(binding["validator"]).strip(),
    }


# --------------------------------------------------------------------------
# Seats and physical accelerators
# --------------------------------------------------------------------------

def build_seats(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One record per accelerator seat, with its declared board identity.

    Refuses when two seats claim the same physical accelerator. That is not a
    tolerable duplicate: the pooled sum would count one board's VRAM twice and
    report memory that does not exist on the machine.

    This check runs for every manifest, not only for manifests attempting a
    placement. A manifest that seats one board twice is wrong about the
    hardware before anyone asks it to hold a model.
    """
    seats: List[Dict[str, Any]] = []
    seen_boards: Dict[str, Tuple[str, int]] = {}

    for node in manifest["cluster"]["nodes"]:
        node_id = str(node["id"])
        for gpu in node["gpus"]:
            index = int(gpu["index"])
            vram_gb = float(gpu["vram_gb"])
            uuid = gpu.get("uuid")
            if uuid is not None:
                uuid = str(uuid).strip()
                if len(uuid) < _MIN_UUID_LENGTH:
                    uuid = None
            if uuid is not None:
                previous = seen_boards.get(uuid)
                if previous is not None:
                    raise _refuse(
                        REFUSAL_SEAT_IDENTITY_CONFLICT,
                        f"Physical accelerator {uuid} is declared in two seats: "
                        f"{previous[0]}:{previous[1]} and {node_id}:{index}. One "
                        "board cannot occupy two seats, so their VRAM is not "
                        "independent and must not be summed. Correct the "
                        "manifest; the planner will not guess which seat is real.",
                        uuid=uuid,
                        seats=[
                            {"node": previous[0], "gpu": previous[1]},
                            {"node": node_id, "gpu": index},
                        ],
                    )
                seen_boards[uuid] = (node_id, index)

            seats.append({
                "node": node_id,
                "host": node.get("host"),
                "gpu": index,
                "name": gpu.get("name", "GPU"),
                "vram_gb": vram_gb,
                "mem_bw_gbps": gpu.get("mem_bw_gbps"),
                # Seat capability. Never board identity - the estate's two
                # substantially identical 3090s satisfy identical predicates.
                "accelerator_uuid": uuid,
            })
    return seats


def usable_bytes_for_seat(seat: Mapping[str, Any], reserve_gb: float) -> int:
    """Bytes this seat can hold for weights, after its own reserve.

    Clamped per seat. The reserve is a property of a device, so a device that
    is smaller than the reserve contributes nothing - it never lends its
    shortfall to a larger sibling, which is what subtracting
    `reserve * gpu_count` from a pooled total quietly does.
    """
    # floor(x + 0.5) rather than round(): Python's round() is half-to-even and
    # JavaScript's Math.round() is half-up, and the two plan writers must agree
    # about a device's capacity even at a value neither of them is likely to
    # see. Spelling the law out costs nothing and removes the question.
    capacity = int(math.floor(float(seat["vram_gb"]) * BYTES_PER_GB + 0.5))
    reserve = int(math.floor(float(reserve_gb) * BYTES_PER_GB + 0.5))
    return max(0, capacity - reserve)


# --------------------------------------------------------------------------
# Residence, runtime compatibility, sharding strategy
# --------------------------------------------------------------------------

def verified_residence_nodes(
    model: Mapping[str, Any], identity: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Nodes where *this* model object is verifiably and currently resident.

    Residence is per node, per object, and per moment. A record naming a
    different digest describes a different object and is refused rather than
    ignored - silently skipping it would degrade to "no residence declared" and
    hide a manifest that is actively wrong. A record whose freshness is
    anything but CURRENT describes where the object *was*, and stale custody
    is the failure mode the estate's own review found laundering an ancient
    residence into a current-looking one.
    """
    digest = identity["identity_digest"]
    resident: Dict[str, Dict[str, Any]] = {}
    for record in model.get("residence", []) or []:
        node_id = str(record.get("node", ""))
        declared = record.get("identity_digest")
        verified = bool(record.get("verified", False))
        if declared is not None and str(declared).strip().lower() != digest:
            raise _refuse(
                REFUSAL_RESIDENCE_IDENTITY_MISMATCH,
                f"Residence record for node {node_id!r} names object "
                f"{str(declared).strip().lower()} but the model being placed "
                f"is {digest}. That record describes a different object; it "
                "cannot verify residence of this one.",
                node=node_id, declared=str(declared).strip().lower(),
                identity_digest=digest,
            )
        if not verified:
            continue
        freshness = str(record.get("freshness", RESIDENCE_CURRENT)).strip().upper()
        if freshness != RESIDENCE_CURRENT:
            raise _refuse(
                REFUSAL_RESIDENCE_STALE,
                f"Residence of object {digest} on node {node_id!r} is "
                f"declared {freshness!r}, not {RESIDENCE_CURRENT!r}. Stale "
                "evidence records where the bytes were when they were last "
                "checked, which is not a statement that they are there now.",
                node=node_id, freshness=freshness, identity_digest=digest,
            )
        resident[node_id] = {
            "node": node_id,
            "verified": True,
            "freshness": RESIDENCE_CURRENT,
            "path": record.get("path"),
            "verified_at": record.get("verified_at"),
            "validator": record.get("validator"),
        }
    return resident


def _runtime_entries(node: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Node runtimes, normalized to {engine, compatibility}.

    A bare string is accepted for compatibility with manifests written before
    runtime measurement existed, and it normalizes to UNMEASURED. That is the
    conservative reading and the one the estate's Rung 1 review demanded: an
    engine nobody has observed running is not evidence that anything can run.
    """
    entries: List[Dict[str, str]] = []
    for raw in node.get("runtimes") or []:
        if isinstance(raw, Mapping):
            engine = str(raw.get("engine", "")).strip()
            compatibility = str(
                raw.get("compatibility", "UNMEASURED")
            ).strip().upper()
        else:
            engine = str(raw).strip()
            compatibility = "UNMEASURED"
        if engine:
            entries.append({"engine": engine, "compatibility": compatibility})
    return entries


def engines_for_node(node: Mapping[str, Any]) -> List[str]:
    return [entry["engine"] for entry in _runtime_entries(node)]


def measured_engines_for_node(node: Mapping[str, Any]) -> List[str]:
    return [
        entry["engine"]
        for entry in _runtime_entries(node)
        if entry["compatibility"] == RUNTIME_MEASURED
    ]


def compatible_engine(
    model: Mapping[str, Any], node: Mapping[str, Any]
) -> Optional[str]:
    """The measured engine this node would use, or None.

    An empty `model.engines` means the model does not constrain the engine, and
    any measured runtime the node offers will do. A node offering no runtime at
    all can never execute, whatever its VRAM; a node offering only unmeasured
    runtimes has not shown that it can.
    """
    required = [str(e) for e in (model.get("engines") or [])]
    available = measured_engines_for_node(node)
    if not available:
        return None
    if not required:
        return sorted(available)[0]
    for engine in sorted(set(required) & set(available)):
        return engine
    return None


def resolve_sharding_strategy(model: Mapping[str, Any]) -> str:
    """The weight-distribution strategy this plan must prove, or refuse.

    A manifest asking for tensor parallelism or expert parallelism is asking
    for a proof about a decomposition this planner does not construct. Emitting
    a pipeline proof under that label would be answering a different question
    than the one asked.
    """
    name = str(model.get("name", ""))
    strategy = str(model.get("sharding_strategy", SHARDING_PIPELINE)).strip().lower()
    if strategy not in SUPPORTED_SHARDING_STRATEGIES:
        raise _refuse(
            REFUSAL_UNSUPPORTED_SHARDING_STRATEGY,
            f"model.sharding_strategy for {name!r} is {strategy!r}; this "
            f"planner proves "
            f"{', '.join(repr(s) for s in SUPPORTED_SHARDING_STRATEGIES)} and "
            "nothing else. A proof constructed for a different decomposition "
            "would not be a proof about the placement that was requested.",
            model=name, requested=strategy,
            supported=list(SUPPORTED_SHARDING_STRATEGIES),
        )
    return strategy


# --------------------------------------------------------------------------
# Exact apportionment
# --------------------------------------------------------------------------

def apportion_exact(total: int, weights: Sequence[int]) -> List[int]:
    """Split `total` into integer parts proportional to `weights`.

    Exact by construction: the parts sum to `total`, so no byte is lost or
    invented by rounding. Remainders go to the largest fractional shares, ties
    broken by position so the result is deterministic and the plan stays
    byte-identical across runs.
    """
    if total < 0:
        raise ValueError("cannot apportion a negative total")
    if not weights:
        return []
    weight_sum = sum(weights)
    if weight_sum <= 0:
        # Callers refuse before reaching here; returning zeros keeps this
        # function total rather than letting a division error surface as if it
        # were a placement decision.
        return [0] * len(weights)

    numerators = [total * weight for weight in weights]
    parts = [numerator // weight_sum for numerator in numerators]
    remainder = total - sum(parts)
    order = sorted(
        range(len(weights)),
        key=lambda i: (-(numerators[i] % weight_sum), i),
    )
    for position in range(remainder):
        parts[order[position]] += 1
    return parts


# --------------------------------------------------------------------------
# The placement proof
# --------------------------------------------------------------------------

def prove_placement(
    manifest: Mapping[str, Any],
    stages: Sequence[Sequence[Mapping[str, Any]]],
    weights: Mapping[str, int],
    identity: Mapping[str, Any],
    reserve_gb: float,
) -> Dict[str, Any]:
    """Assign every weight byte to a device and prove each device can hold it.

    Returns the proof to be embedded in the plan. Raises PlacementRefusal when
    any stage or device cannot hold what it was assigned.

    The bytes apportioned here are the *tensor payload*, not the checkpoint
    set: container headers are parsed on the host and never occupy device
    memory. The checkpoint set is what custody binds, and it travels in the
    proof beside the payload so the two are never read as one number.

    Aggregate capacity is computed here for diagnostics only. It is never
    compared against the model size to admit anything: a sum over independent
    devices describes an address space that does not exist.
    """
    nodes_by_id = {str(n["id"]): n for n in manifest["cluster"]["nodes"]}
    model = manifest["model"]
    strategy = resolve_sharding_strategy(model)
    resident = verified_residence_nodes(model, identity)
    weight_bytes = weights["tensor_payload_bytes"]

    # Which seats can actually hold weights. A seat is admissible only when its
    # node has this object resident AND a measured engine that can execute it.
    # An inadmissible seat contributes zero capacity - the old head counted its
    # VRAM regardless, which is the pooled fiction in its purest form.
    stage_records: List[Dict[str, Any]] = []
    for index, stage_seats in enumerate(stages):
        seat_rows: List[Dict[str, Any]] = []
        for seat in stage_seats:
            node = nodes_by_id[seat["node"]]
            engine = compatible_engine(model, node)
            is_resident = seat["node"] in resident
            admissible = bool(engine) and is_resident
            seat_rows.append({
                "node": seat["node"],
                "gpu": seat["gpu"],
                "accelerator_uuid": seat.get("accelerator_uuid"),
                "usable_bytes": usable_bytes_for_seat(seat, reserve_gb),
                "engine": engine,
                "residence_verified": is_resident,
                "admissible": admissible,
            })
        stage_records.append({"stage": index, "seats": seat_rows})

    # Refuse the specific reason a stage has no capacity, so the operator is
    # told which fact to fix rather than being handed an arithmetic result.
    for record in stage_records:
        admissible = [s for s in record["seats"] if s["admissible"]]
        if admissible:
            continue
        without_engine = [s for s in record["seats"] if not s["engine"]]
        without_residence = [
            s for s in record["seats"] if not s["residence_verified"]
        ]
        stage_nodes = sorted({s["node"] for s in record["seats"]})
        if without_engine:
            declared = {
                n: engines_for_node(nodes_by_id[n]) for n in stage_nodes
            }
            measured = {
                n: measured_engines_for_node(nodes_by_id[n]) for n in stage_nodes
            }
            unmeasured_only = any(
                declared[n] and not measured[n] for n in stage_nodes
            )
            if unmeasured_only:
                raise _refuse(
                    REFUSAL_RUNTIME_UNMEASURED,
                    f"Stage {record['stage']} is placed on node(s) "
                    f"{', '.join(stage_nodes)}, whose runtimes are declared "
                    "but not measured. An engine nobody has observed "
                    "executing anything is not evidence that this model can "
                    "run there; declare compatibility=MEASURED only for a "
                    "runtime that was actually exercised.",
                    stage=record["stage"], nodes=stage_nodes,
                    node_runtimes=declared, measured_runtimes=measured,
                )
            raise _refuse(
                REFUSAL_NO_SUPPORTED_ENGINE,
                f"Stage {record['stage']} is placed on node(s) "
                f"{', '.join(stage_nodes)}, which offer no runtime able to "
                f"execute {model.get('name')!r}. Device memory on a node that "
                "cannot execute the model is not placement capacity.",
                stage=record["stage"], nodes=stage_nodes,
                model_engines=[str(e) for e in (model.get("engines") or [])],
                node_runtimes=declared,
            )
        if without_residence:
            raise _refuse(
                REFUSAL_RESIDENCE_UNVERIFIED,
                f"Stage {record['stage']} is placed on node(s) "
                f"{', '.join(stage_nodes)}, where object "
                f"{identity['identity_digest']} is not verifiably resident. "
                "Weights that are not present cannot be loaded, so that VRAM "
                "is not placement capacity either.",
                stage=record["stage"], nodes=stage_nodes,
                identity_digest=identity["identity_digest"],
                verified_residence_nodes=sorted(resident),
            )
        raise _refuse(
            REFUSAL_NO_ADMISSIBLE_SEAT,
            f"Stage {record['stage']} has no admissible seat.",
            stage=record["stage"], nodes=stage_nodes,
        )

    # Assign exact bytes to stages, then inside each stage to its own devices.
    stage_capacities = [
        sum(s["usable_bytes"] for s in r["seats"] if s["admissible"])
        for r in stage_records
    ]
    if sum(stage_capacities) <= 0:
        raise _refuse(
            REFUSAL_NO_ADMISSIBLE_SEAT,
            "No admissible seat in the cluster has usable memory after the "
            "per-device reserve.",
            reserve_gb_per_gpu=reserve_gb,
        )

    stage_assignments = apportion_exact(weight_bytes, stage_capacities)

    for record, assigned, capacity in zip(
        stage_records, stage_assignments, stage_capacities
    ):
        record["assigned_bytes"] = assigned
        record["usable_bytes"] = capacity
        if assigned > capacity:
            raise _refuse(
                REFUSAL_STAGE_DOES_NOT_FIT,
                f"Stage {record['stage']} is assigned {assigned} B of weights "
                f"but its own admissible devices hold {capacity} B - short by "
                f"{assigned - capacity} B. The cluster total is irrelevant "
                "here: these are the only devices this stage runs on.",
                stage=record["stage"], assigned_bytes=assigned,
                usable_bytes=capacity, deficit_bytes=assigned - capacity,
            )

        admissible = [s for s in record["seats"] if s["admissible"]]
        device_assignments = apportion_exact(
            assigned, [s["usable_bytes"] for s in admissible]
        )
        for seat, device_bytes in zip(admissible, device_assignments):
            seat["assigned_bytes"] = device_bytes
            if device_bytes > seat["usable_bytes"]:
                raise _refuse(
                    REFUSAL_DEVICE_DOES_NOT_FIT,
                    f"Device {seat['node']}:{seat['gpu']} is assigned "
                    f"{device_bytes} B but holds {seat['usable_bytes']} B after "
                    f"its reserve - short by "
                    f"{device_bytes - seat['usable_bytes']} B.",
                    stage=record["stage"], node=seat["node"], gpu=seat["gpu"],
                    assigned_bytes=device_bytes,
                    usable_bytes=seat["usable_bytes"],
                    deficit_bytes=device_bytes - seat["usable_bytes"],
                )
        for seat in record["seats"]:
            seat.setdefault("assigned_bytes", 0)

    assigned_total = sum(r["assigned_bytes"] for r in stage_records)
    if assigned_total != weight_bytes:
        # Not reachable by construction; asserted because the whole guarantee
        # of this module is that every byte is accounted to a named device.
        raise _refuse(
            REFUSAL_STAGE_DOES_NOT_FIT,
            f"Internal apportionment error: assigned {assigned_total} B for a "
            f"{weight_bytes} B model.",
            assigned_bytes=assigned_total, weight_bytes=weight_bytes,
        )

    return {
        "identity": dict(identity),
        "sharding_strategy": strategy,
        "weights": {
            "checkpoint_set_bytes": weights["checkpoint_set_bytes"],
            "tensor_payload_bytes": weights["tensor_payload_bytes"],
            "container_overhead_bytes": weights["container_overhead_bytes"],
        },
        "placed_bytes_are": "tensor_payload_bytes",
        "custody_binds": "checkpoint_set_bytes",
        "assigned_bytes_total": assigned_total,
        "stages": [
            {
                "stage": r["stage"],
                "assigned_bytes": r["assigned_bytes"],
                "usable_bytes": r["usable_bytes"],
                "devices": [
                    {
                        "node": s["node"],
                        "gpu": s["gpu"],
                        "accelerator_uuid": s["accelerator_uuid"],
                        "assigned_bytes": s["assigned_bytes"],
                        "usable_bytes": s["usable_bytes"],
                        "engine": s["engine"],
                        "residence_verified": s["residence_verified"],
                    }
                    for s in r["seats"]
                ],
            }
            for r in stage_records
        ],
        "diagnostics": {
            # Reported, never used to admit. Kept so an operator can see the
            # number the old head would have decided on.
            "aggregate_usable_bytes": sum(stage_capacities),
            "aggregate_is_not_an_admission_criterion": True,
        },
    }
