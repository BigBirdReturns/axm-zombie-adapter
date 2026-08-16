"""Model-size determination: declared size wins, and a guess is never a plan.

The planner's feasibility check is only as good as its size estimate. A wrong
estimate is not returned as an error -- it is returned as a confident placement
plan that OOMs on contact with real hardware. These tests pin the boundary
between "known" and "refused".
"""
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
