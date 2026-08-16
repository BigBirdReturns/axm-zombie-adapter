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
"""
from __future__ import annotations

import copy
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

# Kimi-K3, measured tensor payload: 1453.7 GiB. The model that started this
# lane - a name heuristic sized it at 27.94 GB and placed it on four 24 GB
# boards.
KIMI_K3_BYTES = 1_560_860_324_864

DIGEST = "b55515f6bff0e4efbca1fe74c68bf4ec5762f70e787d544586b702a6843caf71"
OTHER_DIGEST = "0" * 63 + "1"


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


def cluster(nodes) -> dict:
    return {"name": "witness", "nodes": nodes}


def model(bytes_: int = 35_000_000_000, digest: str = DIGEST,
          residence=None, engines=("llamacpp",)) -> dict:
    body = {
        "name": "llama-3-70b",
        "dtype": "q4",
        "kv_cache": "fp16",
        "bytes": bytes_,
        "sha256": digest,
        "engines": list(engines),
    }
    if residence is not None:
        body["residence"] = residence
    return body


def resident(*node_ids, digest: str = DIGEST) -> list:
    return [{"node": n, "verified": True, "sha256": digest} for n in node_ids]


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


def two_nodes(runtimes_b=("llamacpp",), residence_nodes=("node-a", "node-b")):
    return manifest_of(
        cluster([
            {
                "id": "node-a", "host": "10.0.0.1", "runtimes": ["llamacpp"],
                "gpus": [gpu(0, "GPU-aaaa1111-0000-4000-8000-000000000001"),
                         gpu(1, "GPU-aaaa1111-0000-4000-8000-000000000002")],
                "links": [{"to": "node-b", "gbps": 10}],
            },
            {
                "id": "node-b", "host": "10.0.0.2",
                "runtimes": list(runtimes_b),
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
                  "runtimes": ["llamacpp"], "gpus": gpus}]),
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
            model(bytes_=40 * BYTES_PER_GB, residence=resident("node-a")),
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

    def test_every_hostile_witness_is_actually_hostile(self):
        """Guards the guard: if these stop being hostile, they stop testing."""
        for manifest in (two_nodes(residence_nodes=("node-a",)),
                         two_nodes(runtimes_b=()),
                         two_nodes(runtimes_b=("vllm",))):
            with self.subTest(cluster=manifest["cluster"]["name"]):
                self.assertTrue(old_head_admits(manifest, 35_000_000_000))


class KimiCannotFit(unittest.TestCase):
    def test_kimi_k3_is_refused_with_a_typed_code(self):
        manifest = single_node(
            [gpu(i, f"GPU-cccc3333-0000-4000-8000-00000000000{i}")
             for i in range(4)],
            model(bytes_=KIMI_K3_BYTES, residence=resident("node-a")),
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


class ExactAuthority(unittest.TestCase):
    def _one_seat(self, model_body):
        return single_node(
            [gpu(0, "GPU-dddd4444-0000-4000-8000-000000000001")], model_body
        )

    def test_placement_refuses_a_model_with_no_declared_bytes(self):
        body = model(residence=resident("node-a"))
        del body["bytes"]
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(self._one_seat(body))
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_WEIGHT_BYTES_MISSING
        )

    def test_placement_refuses_a_model_with_no_object_identity(self):
        body = model(residence=resident("node-a"))
        del body["sha256"]
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(self._one_seat(body))
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_MODEL_IDENTITY_MISSING
        )

    def test_impossible_byte_declarations_are_refused(self):
        for value in (0, -1, 1.5, float("nan"), float("inf"), True,
                      "35000000000", None, [35]):
            with self.subTest(declared=repr(value)):
                with self.assertRaises(PlacementRefusal) as caught:
                    placement.resolve_exact_weight_bytes(
                        {"name": "m", "bytes": value}
                    )
                self.assertEqual(
                    caught.exception.code,
                    placement.REFUSAL_WEIGHT_BYTES_INVALID,
                )

    def test_an_explicit_null_size_is_not_an_absent_field(self):
        """Present-but-null is an invalid declaration, not a missing one."""
        with self.assertRaises(PlacementRefusal) as caught:
            placement.resolve_exact_weight_bytes({"name": "m", "bytes": None})
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_WEIGHT_BYTES_INVALID
        )
        with self.assertRaises(PlacementRefusal) as missing:
            placement.resolve_exact_weight_bytes({"name": "m"})
        self.assertEqual(
            missing.exception.code, placement.REFUSAL_WEIGHT_BYTES_MISSING
        )

    def test_unusable_identities_are_refused(self):
        for value in ("", "zz", DIGEST.upper() + "x", 7, DIGEST[:63], DIGEST + "0"):
            with self.subTest(declared=repr(value)):
                with self.assertRaises(PlacementRefusal) as caught:
                    placement.resolve_model_identity(
                        {"name": "m", "sha256": value}
                    )
                self.assertEqual(
                    caught.exception.code,
                    placement.REFUSAL_MODEL_IDENTITY_INVALID,
                )

    def test_an_uppercase_identity_is_normalised_not_rejected(self):
        self.assertEqual(
            placement.resolve_model_identity(
                {"name": "m", "sha256": "  " + DIGEST.upper() + " "}
            ),
            DIGEST,
        )

    def test_residence_for_a_different_object_is_refused_not_ignored(self):
        """Skipping it silently would degrade to 'no residence declared'."""
        manifest = two_nodes()
        manifest["model"]["residence"][1]["sha256"] = OTHER_DIGEST
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code,
            placement.REFUSAL_RESIDENCE_IDENTITY_MISMATCH,
        )
        self.assertEqual(caught.exception.detail["node"], "node-b")


class EveryByteIsAssignedAndProved(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan(two_nodes())
        self.proof = self.plan["placement_proof"]

    def test_stage_assignments_sum_to_the_exact_weight_bytes(self):
        total = sum(s["assigned_bytes"] for s in self.proof["stages"])
        self.assertEqual(total, 35_000_000_000)
        self.assertEqual(self.proof["weight_bytes"], 35_000_000_000)
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
            (KIMI_K3_BYTES, [21474836480, 21474836480, 4294967296]),
            (0, [1, 1]),
            (999_999_999_999, [7, 11, 13, 17]),
        )
        for total, weights in cases:
            with self.subTest(total=total, weights=weights):
                parts = placement.apportion_exact(total, weights)
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
        self.assertEqual(len(plan["placement_proof"]["stages"]), 2)

    def test_a_seat_without_a_declared_board_identity_still_places(self):
        """UUID is optional; its absence must not become a silent conflict."""
        manifest = single_node(
            [gpu(0), gpu(1)], model(residence=resident("node-a"))
        )
        plan = build_plan(manifest)
        devices = [d for s in plan["placement_proof"]["stages"]
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
    and an RTX 3090. Kimi-K3 does not fit, cannot be executed there, and is not
    resident there. All three are true at once, and the artifact records the
    first one the planner reaches.
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

    def test_the_refusal_is_that_no_engine_can_execute_there(self):
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(copy.deepcopy(self.manifest))
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_NO_SUPPORTED_ENGINE
        )
        self.assertEqual(
            caught.exception.detail["node_runtimes"], {"octo-w01": []}
        )

    def test_kimi_would_still_not_fit_octo_w01_on_capacity_alone(self):
        """Remove the engine and residence objections; capacity still refuses.

        This is the counterfactual, and it is why the artifact is not merely a
        missing-software complaint: 1453.7 GiB has nowhere to go on 8 GB + 24 GB.
        """
        manifest = copy.deepcopy(self.manifest)
        digest = manifest["model"]["sha256"]
        for node in manifest["cluster"]["nodes"]:
            node["runtimes"] = ["llamacpp"]
        manifest["model"]["residence"] = [
            {"node": "octo-w01", "verified": True, "sha256": digest}
        ]
        with self.assertRaises(PlacementRefusal) as caught:
            build_plan(manifest)
        self.assertEqual(
            caught.exception.code, placement.REFUSAL_STAGE_DOES_NOT_FIT
        )
        self.assertEqual(caught.exception.detail["assigned_bytes"], KIMI_K3_BYTES)
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
