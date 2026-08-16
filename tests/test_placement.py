"""WL-02 witnesses: the placement contradiction, and proof that it is closed.

    python -B -m unittest discover -s tests -t . -p "test_placement.py"

The hostile witnesses are the point of this file. Each one builds a cluster the
*old head admits* and the corrected head refuses, and asserts both halves. A
test that only checked the new behaviour would pass just as well against a
planner that refused everything, and would not witness a contradiction at all.

`old_head_admits` is the v1 predicate reproduced exactly, so the tests attack
the real thing rather than a description of it.

These are stdlib unittest cases on purpose: pytest is not installed on the host
this project is developed on, and a validator that cannot run on that host is
not a validator. DURABILITY.md invariant 4 says the same thing about the core.

No digest in this file is invented. Every one is the sha-256 of an ASCII label
computed here, under a scheme named `test/ascii-label-object@1`, so a reader
who wonders what these identities bind can recompute them in one line - and can
never mistake one for custody of real model weights.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from axm_zombie import placement  # noqa: E402
from axm_zombie.manifest import load_manifest  # noqa: E402
from axm_zombie.placement import PlacementRefusal  # noqa: E402
from axm_zombie.planner import build_plan  # noqa: E402

BYTES_PER_GB = 1024 ** 3

# Kimi-K3, as measured by the estate. Three quantities, three meanings; they
# close exactly, and the placement consumes the middle one.
KIMI_CHECKPOINT_SET_BYTES = 1_560_936_091_448
KIMI_TENSOR_PAYLOAD_BYTES = 1_560_860_324_864
KIMI_CONTAINER_OVERHEAD_BYTES = 75_766_584

IDENTITY_SCHEME = "test/ascii-label-object@1"


def label_digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


DIGEST = label_digest("witness-model-object:llama-3-70b-q4")
OTHER_DIGEST = label_digest("witness-model-object:a-different-object")


def old_head_admits(manifest, model_bytes: int) -> bool:
    """The v1 admission predicate, reproduced verbatim.

        usable = sum(vram_gb) - reserve_gb_per_gpu * gpu_count
        admit if usable >= model_gb * 1.05

    This is the fictitious pooled address space. It is kept here, and only
    here, so the hostile witnesses can prove the old head said yes.
    """
    gpus = [g for n in manifest["cluster"]["nodes"] for g in n["gpus"]]
    total_vram = sum(float(g["vram_gb"]) for g in gpus)
    reserve = float(manifest["policy"].get("reserve_gb_per_gpu", 4.0))
    usable = max(0.0, total_vram - reserve * len(gpus))
    return usable >= (model_bytes / BYTES_PER_GB) * 1.05


def gpu(index: int, uuid=None, vram_gb: float = 24.0) -> dict:
    entry = {
        "index": index,
        "name": "RTX_3090",
        "vram_gb": vram_gb,
        "mem_bw_gbps": 936,
    }
    if uuid is not None:
        entry["uuid"] = uuid
    return entry


def measured(*engines) -> list:
    return [{"engine": e, "compatibility": "MEASURED"} for e in engines]


def cluster(nodes) -> dict:
    return {"name": "witness", "nodes": nodes}


def identity(digest: str = DIGEST, state: str = "VERIFIED") -> dict:
    return {
        "identity_scheme": IDENTITY_SCHEME,
        "identity_digest": digest,
        "identity_source": "tests/test_placement.py::label_digest",
        "identity_state": state,
        "verified_at": "2026-08-16T00:00:00Z",
        "validator": "hashlib.sha256 over the ASCII label",
    }


def weights(payload: int = 35_000_000_000, overhead: int = 8_388_608) -> dict:
    return {
        "checkpoint_set_bytes": payload + overhead,
        "tensor_payload_bytes": payload,
        "container_overhead_bytes": overhead,
    }


def model(payload: int = 35_000_000_000, digest: str = DIGEST,
          residence=None, engines=("llamacpp",), **overrides) -> dict:
    body = {
        "name": "llama-3-70b",
        "dtype": "q4",
        "kv_cache": "fp16",
        "bytes": payload,
        "sha256": digest,
        "weights": weights(payload),
        "identity": identity(digest),
        "engines": list(engines),
    }
    if residence is not None:
        body["residence"] = residence
    body.update(overrides)
    return body


def resident(*node_ids, digest: str = DIGEST, freshness: str = "CURRENT") -> list:
    return [
        {"node": n, "verified": True, "freshness": freshness,
         "identity_digest": digest, "verified_at": "2026-08-16T00:00:00Z",
         "validator": "witness"}
        for n in node_ids
    ]


def manifest_of(cluster_body: dict, model_body: dict) -> dict:
    raw = {
        "schema_version": 1,
        "cluster": cluster_body,
        "model": model_body,
        "policy": {"reserve_gb_per_gpu": 4.0, "target_tps": 18},
    }
    # Round-trip through the real reader so defaults are applied exactly as in
    # production; a hand-built dict would test a manifest no user can write.
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "m.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return load_manifest(str(path))


def two_nodes(runtimes_b=("llamacpp",), residence_nodes=("node-a", "node-b"),
              measured_b=True):
    return manifest_of(
        cluster([
            {
                "id": "node-a", "host": "10.0.0.1",
                "runtimes": measured("llamacpp"),
                "gpus": [gpu(0, "GPU-aaaa1111-0000-4000-8000-000000000001"),
                         gpu(1, "GPU-aaaa1111-0000-4000-8000-000000000002")],
                "links": [{"to": "node-b", "gbps": 10}],
            },
            {
                "id": "node-b", "host": "10.0.0.2",
                "runtimes": (measured(*runtimes_b) if measured_b
                             else list(runtimes_b)),
                "gpus": [gpu(0, "GPU-bbbb2222-0000-4000-8000-000000000001"),
                         gpu(1, "GPU-bbbb2222-0000-4000-8000-000000000002")],
                "links": [{"to": "node-a", "gbps": 10}],
            },
        ]),
        model(residence=resident(*residence_nodes)),
    )


def single_node(gpus, model_body):
    return manifest_of(
        cluster([{"id": "node-a", "host": "10.0.0.1",
                  "runtimes": measured("llamacpp"), "gpus": gpus}]),
        model_body,
    )


class HostileOldHead(unittest.TestCase):
    """Clusters the pooled sum admits and the proof refuses."""

    def test_one_board_seated_twice_inflates_the_pool(self):
        """The estate owns two substantially identical 3090s.

        Seat capability cannot tell them apart, so a manifest listing one board
        in two seats looks exactly like one listing two boards. The pooled sum
        then reports memory that does not exist on the machine.
        """
        duplicate = "GPU-493239dc-f76e-bbbb-8e68-ffd34a5e7bbc"
        manifest = single_node(
            [gpu(0, "GPU-aaaa1111-0000-4000-8000-000000000001"),
             gpu(1, "GPU-aaaa1111-0000-4000-8000-000000000002"),
             gpu(2, duplicate),
             gpu(3, duplicate)],
            model(payload=40 * BYTES_PER_GB, residence=resident("node-a")),
        )
        self.assertTrue(
            old_head_admits(manifest, 40 * BYTES_PER_GB),
            "witness is not hostile: the old head must admit this cluster",
        )
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_SEAT_IDENTITY_CONFLICT
        )
        self.assertEqual(caught.exception.detail["uuid"], duplicate)
        self.assertEqual(
            caught.exception.detail["seats"],
            [{"node": "node-a", "gpu": 2}, {"node": "node-a", "gpu": 3}],
        )

    def test_vram_on_a_node_without_the_weights_is_not_capacity(self):
        manifest = two_nodes(residence_nodes=("node-a",))
        self.assertTrue(old_head_admits(manifest, 35_000_000_000))
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_RESIDENCE_UNVERIFIED
        )
        self.assertEqual(caught.exception.detail["nodes"], ["node-b"])
        self.assertEqual(
            caught.exception.detail["verified_residence_nodes"], ["node-a"]
        )

    def test_vram_on_a_node_that_cannot_execute_is_not_capacity(self):
        manifest = two_nodes(runtimes_b=())
        self.assertTrue(old_head_admits(manifest, 35_000_000_000))
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_NO_SUPPORTED_ENGINE
        )
        self.assertEqual(caught.exception.detail["nodes"], ["node-b"])
        self.assertEqual(
            caught.exception.detail["node_runtimes"], {"node-b": []}
        )

    def test_no_engine_in_common_is_refused_even_when_runtimes_exist(self):
        manifest = two_nodes(runtimes_b=("vllm",))
        self.assertTrue(old_head_admits(manifest, 35_000_000_000))
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_NO_SUPPORTED_ENGINE
        )
        self.assertEqual(caught.exception.detail["model_engines"], ["llamacpp"])

    def test_a_declared_but_unmeasured_runtime_is_not_a_runtime(self):
        """The estate's own Rung 1 review found this exact defect.

        `UNMEASURED` was read as compatible, so a placement was authorized
        against a runtime nobody had ever observed executing anything. A bare
        engine name in the manifest normalizes to UNMEASURED for the same
        reason: naming software is not observing it run.
        """
        manifest = two_nodes(runtimes_b=("llamacpp",), measured_b=False)
        self.assertTrue(old_head_admits(manifest, 35_000_000_000))
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_RUNTIME_UNMEASURED
        )
        self.assertEqual(caught.exception.detail["nodes"], ["node-b"])
        self.assertEqual(
            caught.exception.detail["node_runtimes"], {"node-b": ["llamacpp"]}
        )
        self.assertEqual(
            caught.exception.detail["measured_runtimes"], {"node-b": []}
        )

    def test_stale_residence_is_not_current_residence(self):
        """Where the bytes were is not a statement about where they are."""
        manifest = two_nodes()
        manifest["model"]["residence"][1]["freshness"] = "STALE"
        self.assertTrue(old_head_admits(manifest, 35_000_000_000))
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_RESIDENCE_STALE
        )
        self.assertEqual(caught.exception.detail["node"], "node-b")
        self.assertEqual(caught.exception.detail["freshness"], "STALE")

    def test_every_hostile_witness_is_actually_hostile(self):
        """Guards the guard: if these stop being hostile, they stop testing."""
        for manifest in (two_nodes(residence_nodes=("node-a",)),
                         two_nodes(runtimes_b=()),
                         two_nodes(runtimes_b=("vllm",)),
                         two_nodes(runtimes_b=("llamacpp",), measured_b=False)):
            with self.subTest(cluster=manifest["cluster"]["name"]):
                self.assertTrue(old_head_admits(manifest, 35_000_000_000))


class KimiCannotFit(unittest.TestCase):
    def test_kimi_k3_is_refused_with_a_typed_code(self):
        manifest = single_node(
            [gpu(i, f"GPU-cccc3333-0000-4000-8000-00000000000{i}")
             for i in range(4)],
            model(payload=KIMI_TENSOR_PAYLOAD_BYTES,
                  residence=resident("node-a")),
        )
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        refusal = caught.exception
        self.assertEqual(refusal.code, placement.REFUSAL_STAGE_DOES_NOT_FIT)
        self.assertGreater(refusal.detail["deficit_bytes"], 0)
        self.assertGreater(
            refusal.detail["assigned_bytes"], refusal.detail["usable_bytes"]
        )

    def test_the_refusal_is_a_serializable_artifact(self):
        refusal = PlacementRefusal(
            placement.REFUSAL_STAGE_DOES_NOT_FIT, "short",
            stage=0, deficit_bytes=7,
        )
        artifact = refusal.to_dict()
        self.assertEqual(artifact["result"], "refusal")
        self.assertEqual(artifact["code"], placement.REFUSAL_STAGE_DOES_NOT_FIT)
        self.assertEqual(artifact["detail"]["deficit_bytes"], 7)
        json.dumps(artifact)  # must survive being written to disk

    def test_a_refusal_is_still_a_valueerror_for_v1_callers(self):
        """The v1 planner raised ValueError; existing callers must not break."""
        self.assertTrue(issubclass(PlacementRefusal, ValueError))


class ObjectIdentityMustBeCustodyNotDeclaration(unittest.TestCase):
    """A syntactically valid digest is a claim. Placement needs custody."""

    def _one_seat(self, model_body):
        return single_node(
            [gpu(0, "GPU-dddd4444-0000-4000-8000-000000000001")], model_body
        )

    def test_a_bare_sha256_does_not_authorize_a_placement(self):
        body = model(residence=resident("node-a"))
        del body["identity"]
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(self._one_seat(body))
        self.assertEqual(
            caught.exception.code,
            placement.REFUSAL_UNVERIFIED_MODEL_IDENTITY,
        )
        self.assertEqual(caught.exception.detail["declared_sha256"], DIGEST)

    def test_a_declared_but_unverified_identity_is_refused(self):
        body = model(residence=resident("node-a"))
        body["identity"]["identity_state"] = "DECLARED"
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(self._one_seat(body))
        self.assertEqual(
            caught.exception.code,
            placement.REFUSAL_UNVERIFIED_MODEL_IDENTITY,
        )
        self.assertEqual(caught.exception.detail["identity_state"], "DECLARED")

    def test_an_incomplete_binding_is_refused_field_by_field(self):
        for field in ("identity_scheme", "identity_digest", "identity_source",
                      "verified_at", "validator"):
            with self.subTest(field=field):
                body = model(residence=resident("node-a"))
                body["identity"][field] = ""
                with self.assertRaises(PlacementRefusal) as caught:
                    build_plan(self._one_seat(body))
                self.assertEqual(
                    caught.exception.code,
                    placement.REFUSAL_UNVERIFIED_MODEL_IDENTITY,
                )
                self.assertIn(field, caught.exception.detail["missing"])

    def test_an_unparseable_verified_digest_is_refused(self):
        for value in ("", "zz", DIGEST[:63], DIGEST + "0", DIGEST.upper() + "x"):
            with self.subTest(declared=repr(value)):
                body = model(residence=resident("node-a"))
                body["identity"]["identity_digest"] = value
                body["sha256"] = DIGEST
                with self.assertRaises(PlacementRefusal) as caught:
                    build_plan(self._one_seat(body))
                self.assertEqual(
                    caught.exception.code,
                    placement.REFUSAL_UNVERIFIED_MODEL_IDENTITY,
                )

    def test_two_identities_for_one_object_is_refused(self):
        body = model(residence=resident("node-a"))
        body["sha256"] = OTHER_DIGEST
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(self._one_seat(body))
        self.assertEqual(
            caught.exception.code,
            placement.REFUSAL_UNVERIFIED_MODEL_IDENTITY,
        )
        self.assertEqual(caught.exception.detail["identity_digest"], DIGEST)

    def test_the_verified_binding_travels_into_the_plan(self):
        plan = build_plan(two_nodes())
        carried = plan["placement"]["proof"]["identity"]
        self.assertEqual(carried["identity_scheme"], IDENTITY_SCHEME)
        self.assertEqual(carried["identity_digest"], DIGEST)
        self.assertEqual(carried["identity_state"], "VERIFIED")
        self.assertTrue(carried["identity_source"])
        self.assertTrue(carried["validator"])
        self.assertTrue(carried["verified_at"])

    def test_residence_for_a_different_object_is_refused_not_ignored(self):
        """Skipping it silently would degrade to 'no residence declared'."""
        manifest = two_nodes()
        manifest["model"]["residence"][1]["identity_digest"] = OTHER_DIGEST
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code,
            placement.REFUSAL_RESIDENCE_IDENTITY_MISMATCH,
        )
        self.assertEqual(caught.exception.detail["node"], "node-b")


class ThreeByteQuantitiesStayThree(unittest.TestCase):
    """One object, three measurements, and none of them is "model bytes".

    The checkpoint set is what custody binds. The tensor payload is what
    occupies device memory. The container residual is the difference. Kimi-K3's
    residual is 75766584 B of safetensors headers across 96 shards - small
    against 1.42 TiB, and exactly the size of the ambiguity that appears if the
    two totals are ever collapsed into one field.
    """

    def _one_seat(self, model_body):
        return single_node(
            [gpu(0, "GPU-dddd4444-0000-4000-8000-000000000001")], model_body
        )

    def test_the_estate_measurement_closes(self):
        self.assertEqual(
            KIMI_TENSOR_PAYLOAD_BYTES + KIMI_CONTAINER_OVERHEAD_BYTES,
            KIMI_CHECKPOINT_SET_BYTES,
        )

    def test_accounting_that_does_not_close_is_refused(self):
        body = model(residence=resident("node-a"))
        body["weights"]["container_overhead_bytes"] += 1
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(self._one_seat(body))
        self.assertEqual(
            caught.exception.code,
            placement.REFUSAL_WEIGHT_ACCOUNTING_INCONSISTENT,
        )
        self.assertEqual(caught.exception.detail["residual_bytes"], -1)

    def test_a_partial_accounting_is_refused(self):
        for field in placement.WEIGHT_FIELDS:
            with self.subTest(field=field):
                body = model(residence=resident("node-a"))
                del body["weights"][field]
                with self.assertRaises(PlacementRefusal) as caught:
                    build_plan(self._one_seat(body))
                self.assertEqual(
                    caught.exception.code,
                    placement.REFUSAL_WEIGHT_ACCOUNTING_MISSING,
                )
                self.assertEqual(caught.exception.detail["missing"], [field])

    def test_placement_without_any_accounting_is_refused(self):
        body = model(residence=resident("node-a"))
        del body["weights"]
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(self._one_seat(body))
        self.assertEqual(
            caught.exception.code,
            placement.REFUSAL_WEIGHT_ACCOUNTING_MISSING,
        )

    def test_size_authority_and_payload_must_be_the_same_number(self):
        """model.bytes authorizes the placement, so it is the placed quantity.

        Declaring the checkpoint set there instead would put 75 MB of host-side
        headers into a VRAM comparison - small here, and the exact species of
        ambiguity this schema exists to prevent.
        """
        from axm_zombie.planner import REFUSAL_WEIGHT_ACCOUNTING_CONFLICT

        body = model(residence=resident("node-a"))
        body["bytes"] = body["weights"]["checkpoint_set_bytes"]
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(self._one_seat(body))
        self.assertEqual(
            caught.exception.code, REFUSAL_WEIGHT_ACCOUNTING_CONFLICT
        )

    def test_the_proof_carries_all_three_and_labels_which_is_placed(self):
        proof = build_plan(two_nodes())["placement"]["proof"]
        self.assertEqual(proof["placed_bytes_are"], "tensor_payload_bytes")
        self.assertEqual(proof["custody_binds"], "checkpoint_set_bytes")
        self.assertEqual(
            proof["assigned_bytes_total"],
            proof["weights"]["tensor_payload_bytes"],
        )
        self.assertGreater(
            proof["weights"]["checkpoint_set_bytes"],
            proof["weights"]["tensor_payload_bytes"],
        )


class ShardingStrategyMustBeOneThisPlannerProves(unittest.TestCase):
    def test_an_unsupported_strategy_is_refused(self):
        for strategy in ("tensor_parallel", "expert_parallel", "zero3"):
            with self.subTest(strategy=strategy):
                manifest = two_nodes()
                manifest["model"]["sharding_strategy"] = strategy
                with self.assertRaises(PlacementRefusal) as caught:
                    build_plan(manifest)
                self.assertEqual(
                    caught.exception.code,
                    placement.REFUSAL_UNSUPPORTED_SHARDING_STRATEGY,
                )
                self.assertEqual(caught.exception.detail["requested"], strategy)

    def test_the_default_strategy_is_pipeline_and_is_recorded(self):
        manifest = two_nodes()
        self.assertNotIn("sharding_strategy", manifest["model"])
        plan = build_plan(manifest)
        self.assertEqual(
            plan["placement"]["proof"]["sharding_strategy"], "pipeline"
        )


class RunnableIsNotTheSameAsRepresentable(unittest.TestCase):
    """A topology sketch and a placement proof must not look alike.

    The old head's failure was not arithmetic. It was that a plan built from a
    guess and a plan built from a measurement were the same artifact. A
    manifest that never claimed placement evidence still gets a plan - the
    topology is real and useful - but the plan says in a field that nothing
    here establishes it can run.
    """

    def _topology_only(self):
        return manifest_of(
            cluster([{"id": "node-a", "host": "10.0.0.1",
                      "gpus": [gpu(0), gpu(1)]}]),
            {"name": "llama-3-70b", "dtype": "q4"},
        )

    def test_a_manifest_with_no_evidence_is_representable_not_runnable(self):
        plan = build_plan(self._topology_only())
        self.assertEqual(
            plan["placement"]["state"], "REPRESENTABLE_NOT_RUNNABLE"
        )
        self.assertIsNone(plan["placement"]["proof"])

    def test_it_names_every_predicate_it_is_missing(self):
        plan = build_plan(self._topology_only())
        self.assertEqual(
            plan["placement"]["missing_predicates"],
            ["verified_object_identity", "declared_weight_accounting",
             "verified_residence", "measured_runtime"],
        )

    def test_a_fully_evidenced_manifest_is_runnable(self):
        plan = build_plan(two_nodes())
        self.assertEqual(plan["placement"]["state"], "RUNNABLE")
        self.assertEqual(plan["placement"]["missing_predicates"], [])
        self.assertIsNotNone(plan["placement"]["proof"])

    def test_topology_alone_never_triggers_a_placement_refusal(self):
        """Seat UUIDs and runtimes describe hardware, not a placement claim."""
        manifest = manifest_of(
            cluster([{"id": "node-a", "host": "10.0.0.1",
                      "runtimes": measured("llamacpp"),
                      "gpus": [gpu(0, "GPU-eeee5555-0000-4000-8000-000000000001"),
                               gpu(1, "GPU-eeee5555-0000-4000-8000-000000000002")]}]),
            {"name": "llama-3-70b", "dtype": "q4"},
        )
        plan = build_plan(manifest)
        self.assertEqual(
            plan["placement"]["state"], "REPRESENTABLE_NOT_RUNNABLE"
        )
        self.assertEqual(
            plan["placement"]["missing_predicates"],
            ["verified_object_identity", "declared_weight_accounting",
             "verified_residence"],
        )


class EveryByteIsAssignedAndProved(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan(two_nodes())
        self.proof = self.plan["placement"]["proof"]

    def test_stage_assignments_sum_to_the_exact_payload_bytes(self):
        total = sum(s["assigned_bytes"] for s in self.proof["stages"])
        self.assertEqual(total, 35_000_000_000)
        self.assertEqual(
            self.proof["weights"]["tensor_payload_bytes"], 35_000_000_000
        )
        self.assertEqual(self.proof["assigned_bytes_total"], 35_000_000_000)

    def test_device_assignments_sum_to_their_stage(self):
        for stage in self.proof["stages"]:
            with self.subTest(stage=stage["stage"]):
                assigned = sum(d["assigned_bytes"] for d in stage["devices"])
                self.assertEqual(assigned, stage["assigned_bytes"])

    def test_every_device_holds_no_more_than_its_own_memory(self):
        for stage in self.proof["stages"]:
            for device in stage["devices"]:
                with self.subTest(node=device["node"], gpu=device["gpu"]):
                    self.assertLessEqual(
                        device["assigned_bytes"], device["usable_bytes"]
                    )

    def test_every_stage_is_proved_against_its_own_seats_only(self):
        for stage in self.proof["stages"]:
            with self.subTest(stage=stage["stage"]):
                own = sum(d["usable_bytes"] for d in stage["devices"])
                self.assertEqual(stage["usable_bytes"], own)
                self.assertLessEqual(stage["assigned_bytes"], own)

    def test_the_plan_records_which_board_holds_which_bytes(self):
        uuids = [d["accelerator_uuid"]
                 for s in self.proof["stages"] for d in s["devices"]]
        self.assertEqual(len(uuids), 4)
        self.assertEqual(len(set(uuids)), 4)
        self.assertTrue(all(u is not None for u in uuids))

    def test_aggregate_capacity_is_declared_not_an_admission_criterion(self):
        diagnostics = self.proof["diagnostics"]
        self.assertIs(
            diagnostics["aggregate_is_not_an_admission_criterion"], True
        )
        self.assertGreater(diagnostics["aggregate_usable_bytes"], 0)


class Apportionment(unittest.TestCase):
    def test_apportionment_never_loses_or_invents_a_byte(self):
        cases = (
            (35_000_000_000, [1, 1]),
            (1, [1, 1, 1]),
            (7, [3, 1]),
            (KIMI_TENSOR_PAYLOAD_BYTES, [21474836480, 21474836480, 4294967296]),
            (0, [1, 1]),
            (999_999_999_999, [7, 11, 13, 17]),
        )
        for total, weights_ in cases:
            with self.subTest(total=total, weights=weights_):
                parts = placement.apportion_exact(total, weights_)
                self.assertEqual(sum(parts), total)
                self.assertTrue(all(p >= 0 for p in parts))

    def test_apportionment_is_deterministic(self):
        first = placement.apportion_exact(1_000_000_007, [5, 5, 3])
        second = placement.apportion_exact(1_000_000_007, [5, 5, 3])
        self.assertEqual(first, second)

    def test_a_stage_with_no_capacity_cannot_absorb_bytes(self):
        self.assertEqual(placement.apportion_exact(10, [0, 0]), [0, 0])


class SeatIsNotBoard(unittest.TestCase):
    def test_identical_seat_capability_with_distinct_boards_is_admitted(self):
        """Two genuinely different 3090s are two accelerators, and pass."""
        plan = build_plan(two_nodes())
        self.assertEqual(len(plan["placement"]["proof"]["stages"]), 2)

    def test_a_seat_without_a_declared_board_identity_still_places(self):
        """UUID is optional; its absence must not become a silent conflict."""
        manifest = single_node(
            [gpu(0), gpu(1)], model(residence=resident("node-a"))
        )
        plan = build_plan(manifest)
        devices = [d for s in plan["placement"]["proof"]["stages"]
                   for d in s["devices"]]
        self.assertTrue(all(d["accelerator_uuid"] is None for d in devices))

    def test_usable_bytes_are_clamped_per_device_not_pooled(self):
        """A device smaller than the reserve contributes nothing.

        Subtracting `reserve * gpu_count` from a pooled total lets a small
        device lend its shortfall to a large one, which no device can do.
        """
        self.assertEqual(
            placement.usable_bytes_for_seat({"vram_gb": 2.0}, 4.0), 0
        )
        self.assertEqual(
            placement.usable_bytes_for_seat({"vram_gb": 24.0}, 4.0),
            20 * BYTES_PER_GB,
        )


class EstateArtifact(unittest.TestCase):
    """The committed refusal for a real host, bound to the manifest.

    OCTO-W01 has two seats holding two physically distinct boards: an RTX 4060
    and an RTX 3090. Four facts about Kimi-K3 there are all true at once: this
    public repository holds no verified identity for the object, no inference
    engine is installed, residence cannot be verified against an identity this
    repository does not have, and 1453.7 GiB does not fit 8 GB + 24 GB. The
    artifact records the first one the planner reaches, and it reaches identity
    first on purpose - naming the object is prior to placing it.

    The identity itself is deliberately absent rather than invented. An
    accepted Rung 1A binding exists in a private evidence packet; the manifest
    points at it by name. A syntactically valid invented digest would be worse
    than none, because a downstream reader could not tell it from custody.
    """

    MANIFEST = REPO / "examples" / "estate_octo_w01_kimi_k3.json"
    ARTIFACT = REPO / "artifacts" / "estate-octo-w01-kimi-k3.refusal.json"

    def setUp(self):
        self.manifest = load_manifest(str(self.MANIFEST))

    def test_the_committed_artifact_matches_what_the_planner_emits(self):
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(copy.deepcopy(self.manifest))
        emitted = caught.exception.to_dict()
        committed = json.loads(self.ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(emitted, committed)

    def test_the_refusal_is_that_the_object_is_not_verified_here(self):
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(copy.deepcopy(self.manifest))
        self.assertEqual(
            caught.exception.code,
            placement.REFUSAL_UNVERIFIED_MODEL_IDENTITY,
        )
        self.assertEqual(
            caught.exception.detail["identity_state"], "UNVERIFIED_HERE"
        )
        self.assertEqual(
            caught.exception.detail["identity_scheme"],
            "estate/object-residence@1",
        )

    def test_the_manifest_carries_no_invented_digest(self):
        model_body = self.manifest["model"]
        self.assertNotIn("sha256", model_body)
        self.assertEqual(model_body["identity"]["identity_digest"], "")
        self.assertTrue(model_body["identity"]["identity_source"])

    def test_the_three_estate_byte_quantities_are_carried_and_close(self):
        declared = self.manifest["model"]["weights"]
        self.assertEqual(
            declared["checkpoint_set_bytes"], KIMI_CHECKPOINT_SET_BYTES
        )
        self.assertEqual(
            declared["tensor_payload_bytes"], KIMI_TENSOR_PAYLOAD_BYTES
        )
        self.assertEqual(
            declared["container_overhead_bytes"], KIMI_CONTAINER_OVERHEAD_BYTES
        )
        self.assertEqual(
            declared["tensor_payload_bytes"]
            + declared["container_overhead_bytes"],
            declared["checkpoint_set_bytes"],
        )
        # The placement consumes the payload, never the checkpoint set.
        self.assertEqual(
            self.manifest["model"]["bytes"], KIMI_TENSOR_PAYLOAD_BYTES
        )

    def test_kimi_would_still_not_fit_octo_w01_on_capacity_alone(self):
        """Remove every prior objection; capacity still refuses.

        This is the counterfactual, and it is why the artifact is not merely a
        missing-evidence complaint: 1453.7 GiB has nowhere to go on 8 GB +
        24 GB. The identity supplied here is hypothetical and is labelled as
        such - it exists only inside this test, is the sha-256 of an ASCII
        label, and is never written to the repository.
        """
        manifest = copy.deepcopy(self.manifest)
        hypothetical = label_digest("hypothetical-object:octo-w01-kimi-k3")
        manifest["model"]["identity"] = {
            "identity_scheme": "test/ascii-label-object@1",
            "identity_digest": hypothetical,
            "identity_source": "tests/test_placement.py, counterfactual only",
            "identity_state": "VERIFIED",
            "verified_at": "2026-08-16T00:00:00Z",
            "validator": "hashlib.sha256 over the ASCII label",
        }
        for node in manifest["cluster"]["nodes"]:
            node["runtimes"] = measured("llamacpp")
        manifest["model"]["residence"] = [
            {"node": "octo-w01", "verified": True, "freshness": "CURRENT",
             "identity_digest": hypothetical}
        ]
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_STAGE_DOES_NOT_FIT
        )
        self.assertEqual(
            caught.exception.detail["assigned_bytes"],
            KIMI_TENSOR_PAYLOAD_BYTES,
        )
        self.assertGreater(caught.exception.detail["deficit_bytes"], 0)

    def test_the_two_estate_seats_hold_two_distinct_boards(self):
        """Seat capability differs here, but identity is what is asserted."""
        seats = placement.build_seats(self.manifest)
        uuids = [s["accelerator_uuid"] for s in seats]
        self.assertEqual(len(uuids), 2)
        self.assertEqual(len(set(uuids)), 2)
        self.assertIn("GPU-e0b1541d-fc7d-38f5-d4c0-c15a3bd241a0", uuids)
        self.assertIn("GPU-493239dc-f76e-bbbb-8e68-ffd34a5e7bbc", uuids)


class PlanRemainsDeterministic(unittest.TestCase):
    def test_the_same_manifest_produces_a_byte_identical_plan(self):
        manifest = two_nodes()
        first = json.dumps(build_plan(copy.deepcopy(manifest)), indent=2)
        second = json.dumps(build_plan(copy.deepcopy(manifest)), indent=2)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
