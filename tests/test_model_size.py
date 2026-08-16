"""Model-size determination: declared size wins, and a guess is never a plan.

The planner's feasibility check is only as good as its size estimate. A wrong
estimate is not returned as an error -- it is returned as a confident placement
plan that OOMs on contact with real hardware. These tests pin the boundary
between "known" and "refused".
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from axm_zombie.planner import build_plan  # noqa: E402


def manifest(model, gpus=4, vram_gb=24):
    return {
        "cluster": {
            "name": "t",
            "nodes": [{
                "id": "node-a",
                "host": "127.0.0.1",
                "gpus": [
                    {"index": i, "name": "RTX_3090", "vram_gb": vram_gb,
                     "mem_bw_gbps": 936}
                    for i in range(gpus)
                ],
            }],
        },
        "model": model,
        "policy": {"reserve_gb_per_gpu": 4.0, "target_tps": 18},
    }


class DeclaredSizeWins(unittest.TestCase):
    def test_explicit_bytes_are_authoritative(self):
        plan = build_plan(manifest(
            {"name": "anything-at-all", "dtype": "fp16", "bytes": 8 * 1024**3}))
        self.assertEqual(plan["model"]["estimated_model_gb"], 8.0)

    def test_explicit_params_are_multiplied_by_dtype(self):
        plan = build_plan(manifest(
            {"name": "anything-at-all", "dtype": "fp8", "params": 30_000_000_000}))
        self.assertAlmostEqual(plan["model"]["estimated_model_gb"], 27.94, places=2)

    def test_explicit_bytes_outrank_a_recognised_name(self):
        # The name says 7B; the declaration says otherwise. The declaration wins.
        plan = build_plan(manifest(
            {"name": "some-7b-model", "dtype": "fp16", "bytes": 40 * 1024**3}))
        self.assertEqual(plan["model"]["estimated_model_gb"], 40.0)


class RecognisedNamesStillWork(unittest.TestCase):
    def test_70b_is_not_shadowed_by_7b(self):
        plan = build_plan(manifest({"name": "llama-3-70b", "dtype": "q4"}))
        self.assertEqual(plan["model"]["estimated_model_gb"], 32.6)


class GuessesAreRefused(unittest.TestCase):
    def test_unknown_model_name_is_refused_not_defaulted(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "Kimi-K3", "dtype": "fp8"}))
        self.assertIn("Unknown model size", str(caught.exception))
        self.assertIn("model.params", str(caught.exception))

    def test_mixture_of_experts_multiplier_is_refused(self):
        # "mixtral-8x7b" is ~46.7B. Matching the bare "7b" substring produced a
        # 13 GB estimate and a plan that would OOM on four 3090s.
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "mixtral-8x7b", "dtype": "fp16"}))
        self.assertIn("mixture-of-experts", str(caught.exception))

    def test_an_underscore_does_not_end_the_moe_multiplier(self):
        """The two patterns must agree about where a token ends.

        `\\b` counts "_" as a word character, so the MoE pattern stopped
        matching at "8x7b_" -- while the parameter-token pattern's
        `(?![a-z0-9])` accepted that same "_" and read the name as a 7B model.
        One name was therefore MoE for the refusal and plain for the heuristic,
        and the heuristic won: 13 GB for a ~46.7B mixture, placed on four
        3090s. Any delimiter one pattern treats as a boundary the other must
        too.
        """
        for name in ("mixtral-8x7b_model", "mixtral_8x7b_instruct",
                     "moe 4 x 22 b_v2", "mixtral-8x7b.safetensors"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    build_plan(manifest({"name": name, "dtype": "fp16"}))
                self.assertIn("mixture-of-experts", str(caught.exception))


class SizeProvenanceIsDisclosed(unittest.TestCase):
    """A plan must say how the size that authorized its placement was obtained.

    Without this, a plan sized by a name guess is indistinguishable from one
    sized by a measured checkpoint, and silently acquires its evidence status.
    """

    def test_declared_bytes_are_reported_as_manifest_bytes(self):
        plan = build_plan(manifest(
            {"name": "anything-at-all", "dtype": "fp16", "bytes": 8 * 1024**3}))
        self.assertEqual(plan["model"]["model_size_source"], "manifest_bytes")
        self.assertEqual(plan["model"]["model_size_bytes"], 8 * 1024**3)

    def test_declared_params_are_reported_as_params_and_quantization(self):
        plan = build_plan(manifest(
            {"name": "anything-at-all", "dtype": "fp8", "params": 30_000_000_000}))
        self.assertEqual(
            plan["model"]["model_size_source"], "manifest_params_and_quantization")

    def test_name_inference_is_reported_as_a_legacy_heuristic(self):
        # The validated golden model. Its size is a guess and the plan says so.
        plan = build_plan(manifest({"name": "llama-3-70b", "dtype": "q4"}))
        self.assertEqual(plan["model"]["model_size_source"], "legacy_name_heuristic")

    def test_object_identity_is_null_unless_declared(self):
        plan = build_plan(manifest(
            {"name": "anything-at-all", "dtype": "fp16", "bytes": 8 * 1024**3}))
        self.assertIsNone(plan["model"]["model_object_sha256"])

    def test_declared_object_identity_is_carried_into_the_plan(self):
        digest = "a" * 64
        plan = build_plan(manifest({"name": "anything-at-all", "dtype": "fp16",
                                    "bytes": 8 * 1024**3, "sha256": digest}))
        self.assertEqual(plan["model"]["model_object_sha256"], digest)

    def test_refusal_names_the_size_source(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "Kimi-K3", "dtype": "fp8",
                                 "bytes": 1_560_860_324_864}))
        self.assertIn("manifest_bytes", str(caught.exception))


class ContradictorySizeClaimsFailClosed(unittest.TestCase):
    """A stale `bytes` beside honest `params` is the same defect, other channel.

    Preferring `bytes` by precedence would place a terabyte-scale object using a
    32 GB claim -- a confident plan, and wrong.
    """

    def test_order_of_magnitude_disagreement_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "some-model", "dtype": "fp8",
                                 "bytes": 32 * 1024**3,
                                 "params": 1_000_000_000_000}))
        message = str(caught.exception)
        self.assertIn("Size-claim conflict", message)
        self.assertIn("model.bytes", message)
        self.assertIn("model.params", message)

    def test_quantization_scale_disagreement_is_tolerated(self):
        # 7B declared at fp16 (14 GB) beside a 13 GB byte count: packaging
        # noise, not a contradiction. `bytes` still wins.
        plan = build_plan(manifest({"name": "some-model", "dtype": "fp16",
                                    "bytes": 13 * 1024**3,
                                    "params": 7_000_000_000}))
        self.assertEqual(plan["model"]["estimated_model_gb"], 13.0)
        self.assertEqual(plan["model"]["model_size_source"], "manifest_bytes")


class ImpossibleSizeDeclarationsAreRefused(unittest.TestCase):
    """A declared size still has to be a size.

    Zero and negative values are the sharp cases: they satisfy every
    feasibility comparison, so they do not merely mis-size a plan, they
    authorize any placement at all.
    """

    IMPOSSIBLE = [
        ("zero", 0),
        ("negative", -1),
        ("fractional", 1.5),
        ("boolean", True),
        ("string", "34359738368"),
        ("none-typed", []),
        ("nan", float("nan")),
        ("infinite", float("inf")),
    ]

    def test_impossible_bytes_are_refused(self):
        for label, value in self.IMPOSSIBLE:
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    build_plan(manifest({"name": "some-model", "dtype": "fp8",
                                         "bytes": value}))
                self.assertIn("model.bytes", str(caught.exception))

    def test_impossible_params_are_refused(self):
        for label, value in self.IMPOSSIBLE:
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    build_plan(manifest({"name": "some-model", "dtype": "fp8",
                                         "params": value}))
                self.assertIn("model.params", str(caught.exception))

    def test_a_zero_size_does_not_authorize_a_placement(self):
        # Without validation this returned a plan: 0 GB fits anything.
        with self.assertRaises(ValueError):
            build_plan(manifest({"name": "some-model", "dtype": "fp8", "bytes": 0}))

    def test_an_excessively_large_size_refuses_rather_than_authorizing(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "some-model", "dtype": "fp8",
                                 "bytes": 10**30}))
        self.assertIn("exact integer range", str(caught.exception))


class DeclaredNullIsNotAnAbsentField(unittest.TestCase):
    """Field presence and field value are two different facts.

    An absent `bytes` key means "no size was declared here, fall through to
    params or the name". A present `bytes: null` means "a size was declared and
    it is nothing", which is not a size. Reading both as None let an explicitly
    empty declaration fall through to the name heuristic, so a manifest that
    declared its size was authorized by a guess -- and the plan reported
    `legacy_name_heuristic` as if the field had never been written.
    """

    def test_declared_null_bytes_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "llama-3-70b", "dtype": "q4",
                                 "bytes": None}))
        self.assertIn("model.bytes", str(caught.exception))

    def test_declared_null_params_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "llama-3-70b", "dtype": "q4",
                                 "params": None}))
        self.assertIn("model.params", str(caught.exception))

    def test_declared_null_bytes_do_not_fall_through_to_params(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "llama-3-70b", "dtype": "q4",
                                 "bytes": None, "params": 7_000_000_000}))
        self.assertIn("model.bytes", str(caught.exception))

    def test_the_same_name_with_the_field_absent_still_plans(self):
        plan = build_plan(manifest({"name": "llama-3-70b", "dtype": "q4"}))
        self.assertEqual(plan["model"]["model_size_source"], "legacy_name_heuristic")


class SizesMustSurviveBothWriters(unittest.TestCase):
    """A byte count is exchanged as a JSON number and read as a double.

    Above 2**53-1 that read is lossy: 9007199254740993 comes back as
    9007199254740992. Publishing the rounded value would mean publishing a
    number nobody asserted as the size that authorized a placement, so both
    writers refuse the declaration instead.
    """

    UNSAFE = 9_007_199_254_740_993
    LARGEST_EXACT = 9_007_199_254_740_991

    def test_bytes_beyond_the_exact_integer_range_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "some-model", "dtype": "fp8",
                                 "bytes": self.UNSAFE}))
        self.assertIn("exact integer range", str(caught.exception))
        self.assertIn("model.bytes", str(caught.exception))

    def test_params_beyond_the_exact_integer_range_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "some-model", "dtype": "fp8",
                                 "params": self.UNSAFE}))
        self.assertIn("exact integer range", str(caught.exception))
        self.assertIn("model.params", str(caught.exception))

    def test_the_lossy_round_trip_this_guards_against_is_real(self):
        # What the browser writer would have received for UNSAFE.
        self.assertEqual(
            json.loads(str(self.UNSAFE), parse_int=float), float(self.LARGEST_EXACT + 1))

    def test_the_largest_exact_value_is_still_read_as_a_size(self):
        # Refused for the honest reason -- 8 PB does not fit four 3090s -- not
        # because the value could not be represented.
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "some-model", "dtype": "fp8",
                                 "bytes": self.LARGEST_EXACT}))
        self.assertIn("Insufficient VRAM", str(caught.exception))


class OneDecimalRoundingLaw(unittest.TestCase):
    """`estimated_model_gb` must not depend on which writer produced it.

    Python's round() rounds half to even and JavaScript's Math.round() rounds
    half up, so a decimal tie published 1.12 from one writer and 1.13 from the
    other for the same declared bytes. Both now round half away from zero, in
    exact integer arithmetic, and the plan is a plan either way.
    """

    TIE_BYTES = 1_207_959_552          # exactly 1.125 GiB
    JUST_UNDER_TIE_BYTES = 1_207_959_551

    def test_a_decimal_tie_rounds_half_away_from_zero(self):
        plan = build_plan(manifest({"name": "m", "dtype": "fp8",
                                    "bytes": self.TIE_BYTES}))
        self.assertEqual(plan["model"]["estimated_model_gb"], 1.13)
        self.assertEqual(plan["model"]["model_size_bytes"], self.TIE_BYTES)

    def test_one_byte_below_the_tie_rounds_down(self):
        plan = build_plan(manifest({"name": "m", "dtype": "fp8",
                                    "bytes": self.JUST_UNDER_TIE_BYTES}))
        self.assertEqual(plan["model"]["estimated_model_gb"], 1.12)

    def test_the_law_is_not_the_language_default(self):
        # The regression this pins: the old expression, still evaluated here.
        self.assertEqual(round(self.TIE_BYTES / 1024**3, 2), 1.12)


class ModelObjectIdentityIsValidated(unittest.TestCase):
    """An identity that cannot be checked must not be published as one."""

    def test_malformed_digests_are_refused(self):
        for label, value in [("too short", "abc"), ("non-hex", "z" * 64),
                             ("numeric", 12345), ("too long", "a" * 65)]:
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    build_plan(manifest({"name": "llama-3-70b", "dtype": "q4",
                                         "sha256": value}))
                self.assertIn("model.sha256", str(caught.exception))

    def test_a_valid_digest_is_normalized_into_the_plan(self):
        plan = build_plan(manifest({"name": "llama-3-70b", "dtype": "q4",
                                    "sha256": "A" * 64}))
        self.assertEqual(plan["model"]["model_object_sha256"], "a" * 64)


class NameTokensNeedBoundaries(unittest.TestCase):
    """Substring capture is the same defect as the MoE multiplier.

    "model-170b" containing "70b" produced a confident underestimate for a
    model whose size the heuristic never actually knew.
    """

    def test_170b_is_not_read_as_70b(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "model-170b", "dtype": "q4"}))
        self.assertIn("170b", str(caught.exception))

    def test_107b_is_not_read_as_7b(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest({"name": "model-107b", "dtype": "q4"}))
        self.assertIn("107b", str(caught.exception))

    def test_an_ambiguous_pair_of_tokens_is_refused(self):
        with self.assertRaises(ValueError):
            build_plan(manifest({"name": "frankenmerge-13b-7b", "dtype": "q4"}))

    def test_supported_tokens_still_resolve(self):
        for token, expected_gb in [("llama-3-70b", 32.6), ("model-34b", 15.83),
                                   ("model-13b", 6.05), ("model-7b", 3.26)]:
            with self.subTest(token=token):
                plan = build_plan(manifest({"name": token, "dtype": "q4"}))
                self.assertAlmostEqual(
                    plan["model"]["estimated_model_gb"], expected_gb, places=2)


class RealKimiK3DoesNotFitFourThreeNineties(unittest.TestCase):
    """The measured object, against the measured hardware.

    Kimi-K3 tensor payload measured on OCTO-W01, 2026-08-16: 1560860324864 B
    (1453.7 GiB). Four RTX 3090s aggregate 96 GiB, less a 4 GB per-GPU reserve.
    """

    KIMI_K3_BYTES = 1_560_860_324_864

    def test_declared_real_size_is_refused_on_four_3090s(self):
        with self.assertRaises(ValueError) as caught:
            build_plan(manifest(
                {"name": "Kimi-K3", "dtype": "fp8", "bytes": self.KIMI_K3_BYTES}))
        self.assertIn("Insufficient VRAM", str(caught.exception))

    def test_the_shortfall_is_about_fifteen_fold(self):
        model_gb = self.KIMI_K3_BYTES / 1024**3
        usable_gb = 4 * 24 - 4 * 4.0
        self.assertGreater(model_gb / (4 * 24), 15.0)
        self.assertLess(usable_gb, model_gb)


if __name__ == "__main__":
    unittest.main(verbosity=2)
