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
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple
import math
import re

# The existing v1 convention: manifest memory is declared in `_gb` fields and
# compared against binary gigabytes. WL-02 does not change the unit - SPEC.md
# rule 3 makes a field's unit permanent within a schema version - it only stops
# doing arithmetic in floating-point GB once exact bytes are available.
BYTES_PER_GB = 1024 ** 3

# A model object identity is a sha-256 digest or it is not an identity.
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

# A physical accelerator identity, as reported by the driver. Vendor-neutral in
# shape: a non-empty opaque string. This module never parses meaning out of it;
# it only requires that one board does not appear in two seats.
_MIN_UUID_LENGTH = 8


# Typed refusal codes. These are machine-readable and permanent: a downstream
# reader branches on the code, never on the prose.
REFUSAL_WEIGHT_BYTES_MISSING = "WEIGHT_BYTES_MISSING"
REFUSAL_WEIGHT_BYTES_INVALID = "WEIGHT_BYTES_INVALID"
REFUSAL_MODEL_IDENTITY_MISSING = "MODEL_IDENTITY_MISSING"
REFUSAL_MODEL_IDENTITY_INVALID = "MODEL_IDENTITY_INVALID"
REFUSAL_SEAT_IDENTITY_CONFLICT = "SEAT_IDENTITY_CONFLICT"
REFUSAL_RESIDENCE_UNVERIFIED = "RESIDENCE_UNVERIFIED"
REFUSAL_RESIDENCE_IDENTITY_MISMATCH = "RESIDENCE_IDENTITY_MISMATCH"
REFUSAL_NO_SUPPORTED_ENGINE = "NO_SUPPORTED_ENGINE"
REFUSAL_STAGE_DOES_NOT_FIT = "STAGE_DOES_NOT_FIT"
REFUSAL_DEVICE_DOES_NOT_FIT = "DEVICE_DOES_NOT_FIT"
REFUSAL_NO_ADMISSIBLE_SEAT = "NO_ADMISSIBLE_SEAT"


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
# Exact model authority: bytes and identity
# --------------------------------------------------------------------------

def resolve_exact_weight_bytes(model: Mapping[str, Any]) -> int:
    """Return the exact on-disk weight size, or refuse.

    Placement is authorized by measurement or not at all. A size inferred from
    a model name is a guess, and a guess that authorizes a placement is
    indistinguishable in the emitted plan from a measurement that does - which
    is exactly how a 1.42 TiB model was placed on four 24 GB boards.
    """
    name = str(model.get("name", ""))
    if "bytes" not in model:
        raise _refuse(
            REFUSAL_WEIGHT_BYTES_MISSING,
            f"model.bytes is required to place {name!r}. Placement consumes the "
            "exact weight size; it will not infer one from the model name, "
            "because an inferred size produces a confident plan that OOMs.",
            model=name,
        )
    value = model["bytes"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(
            REFUSAL_WEIGHT_BYTES_INVALID,
            f"model.bytes for {name!r} must be a positive whole number of "
            f"bytes, got {value!r} ({type(value).__name__}). A size that is not "
            "a number cannot be compared against device memory.",
            model=name, declared=repr(value),
        )
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise _refuse(
                REFUSAL_WEIGHT_BYTES_INVALID,
                f"model.bytes for {name!r} must be finite, got {value!r}. A "
                "non-finite size never fails a feasibility comparison.",
                model=name, declared=repr(value),
            )
        if not value.is_integer():
            raise _refuse(
                REFUSAL_WEIGHT_BYTES_INVALID,
                f"model.bytes for {name!r} must be a whole number of bytes, "
                f"got {value!r}. Weights are not divisible below a byte.",
                model=name, declared=repr(value),
            )
    exact = int(value)
    if exact <= 0:
        raise _refuse(
            REFUSAL_WEIGHT_BYTES_INVALID,
            f"model.bytes for {name!r} must be greater than zero, got {exact}. "
            "A zero or negative size satisfies every feasibility comparison and "
            "would authorize any placement at all.",
            model=name, declared=exact,
        )
    return exact


def resolve_model_identity(model: Mapping[str, Any]) -> str:
    """Return the sha-256 identity of the measured model object, or refuse.

    A declared byte count with no identity is an unverified claim about an
    unnamed object. Placement needs to know *which* object it sized, because
    residence is checked by digest.
    """
    name = str(model.get("name", ""))
    value = model.get("sha256")
    if value is None:
        raise _refuse(
            REFUSAL_MODEL_IDENTITY_MISSING,
            f"model.sha256 is required to place {name!r}. Exact bytes without "
            "an object identity describe an unnamed object, and residence "
            "cannot then be verified against anything.",
            model=name,
        )
    if not isinstance(value, str) or not _SHA256.match(value.strip().lower()):
        raise _refuse(
            REFUSAL_MODEL_IDENTITY_INVALID,
            f"model.sha256 for {name!r} must be 64 hexadecimal characters, got "
            f"{value!r}. An identity that cannot be checked is not an identity.",
            model=name, declared=repr(value),
        )
    return value.strip().lower()


# --------------------------------------------------------------------------
# Seats and physical accelerators
# --------------------------------------------------------------------------

def build_seats(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One record per accelerator seat, with its declared board identity.

    Refuses when two seats claim the same physical accelerator. That is not a
    tolerable duplicate: the pooled sum would count one board's VRAM twice and
    report memory that does not exist on the machine.
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
    capacity = int(round(float(seat["vram_gb"]) * BYTES_PER_GB))
    reserve = int(round(float(reserve_gb) * BYTES_PER_GB))
    return max(0, capacity - reserve)


# --------------------------------------------------------------------------
# Residence and runtime compatibility
# --------------------------------------------------------------------------

def verified_residence_nodes(
    model: Mapping[str, Any], identity: str
) -> Dict[str, Dict[str, Any]]:
    """Nodes where *this* model object is verifiably resident.

    Residence is per node and per object. A record that names a different
    digest describes a different object, and is refused rather than ignored -
    silently skipping it would degrade to "no residence declared" and hide a
    manifest that is actively wrong.
    """
    resident: Dict[str, Dict[str, Any]] = {}
    for record in model.get("residence", []) or []:
        node_id = str(record.get("node", ""))
        digest = record.get("sha256")
        verified = bool(record.get("verified", False))
        if digest is not None and str(digest).strip().lower() != identity:
            raise _refuse(
                REFUSAL_RESIDENCE_IDENTITY_MISMATCH,
                f"Residence record for node {node_id!r} names object "
                f"{str(digest).strip().lower()} but the model being placed is "
                f"{identity}. That record describes a different object; it "
                "cannot verify residence of this one.",
                node=node_id, declared=str(digest).strip().lower(),
                model_object_sha256=identity,
            )
        if verified:
            resident[node_id] = {
                "node": node_id,
                "verified": True,
                "path": record.get("path"),
            }
    return resident


def engines_for_node(node: Mapping[str, Any]) -> List[str]:
    return [str(e) for e in (node.get("runtimes") or [])]


def compatible_engine(
    model: Mapping[str, Any], node: Mapping[str, Any]
) -> str | None:
    """The engine this node would use, or None when it cannot execute at all.

    An empty `model.engines` means the model does not constrain the engine, and
    any runtime the node offers will do. A node offering no runtime at all can
    never execute, whatever its VRAM.
    """
    required = [str(e) for e in (model.get("engines") or [])]
    available = engines_for_node(node)
    if not available:
        return None
    if not required:
        return sorted(available)[0]
    for engine in sorted(set(required) & set(available)):
        return engine
    return None


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
    weight_bytes: int,
    identity: str,
    reserve_gb: float,
) -> Dict[str, Any]:
    """Assign every weight byte to a device and prove each device can hold it.

    Returns the proof to be embedded in the plan. Raises PlacementRefusal when
    any stage or device cannot hold what it was assigned.

    Aggregate capacity is computed here for diagnostics only. It is never
    compared against the model size to admit anything: a sum over independent
    devices describes an address space that does not exist.
    """
    nodes_by_id = {str(n["id"]): n for n in manifest["cluster"]["nodes"]}
    model = manifest["model"]
    resident = verified_residence_nodes(model, identity)

    # Which seats can actually hold weights. A seat is admissible only when its
    # node has this object resident AND an engine that can execute it. An
    # inadmissible seat contributes zero capacity - the old head counted its
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
            raise _refuse(
                REFUSAL_NO_SUPPORTED_ENGINE,
                f"Stage {record['stage']} is placed on node(s) "
                f"{', '.join(stage_nodes)}, which offer no runtime able to "
                f"execute {model.get('name')!r}. Device memory on a node that "
                "cannot execute the model is not placement capacity.",
                stage=record["stage"], nodes=stage_nodes,
                model_engines=[str(e) for e in (model.get("engines") or [])],
                node_runtimes={
                    n: engines_for_node(nodes_by_id[n]) for n in stage_nodes
                },
            )
        if without_residence:
            raise _refuse(
                REFUSAL_RESIDENCE_UNVERIFIED,
                f"Stage {record['stage']} is placed on node(s) "
                f"{', '.join(stage_nodes)}, where object {identity} is not "
                "verifiably resident. Weights that are not present cannot be "
                "loaded, so that VRAM is not placement capacity either.",
                stage=record["stage"], nodes=stage_nodes,
                model_object_sha256=identity,
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
        "model_object_sha256": identity,
        "weight_bytes": weight_bytes,
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
