"""Fail-closed classification coverage for repeated-advance stability evidence."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from benchmarks import torch_memory_stability as stability


def _result(**overrides):
    result = {
        "checkout_module_origin_valid": True,
        "storage_stable": True,
        "all_batches_finite": True,
        "field_error_samples": [0.0] * 15,
        "field_error_tolerance": 1e-10,
        "operator_copy_records": [{"full_domain_copy_count": 0}]
        * stability.PROFILE_EXECUTIONS,
        "lowered_copy_materialization_verified": True,
        "compile_policy": "compile",
        "compiled_region_executed": True,
        "compiler_counter_delta": {
            "graph_breaks": 0,
            "unique_graphs": 0,
            "frames_total": 0,
        },
        "rss_assessment": {"adequate": True, "sustained_growth": False},
        "device_type": "cpu",
        "batches": 15,
        "steps_per_batch": 100,
    }
    result.update(overrides)
    return result


class StabilityEvidenceTest(unittest.TestCase):
    def test_growth_detects_monotonic_and_oscillatory_retention(self):
        self.assertTrue(
            stability._growth_assessment([10, 11, 12, 13, 14, 15])["sustained_growth"]
        )
        self.assertTrue(
            stability._growth_assessment([100, 120, 105, 120])["sustained_growth"]
        )
        self.assertFalse(
            stability._growth_assessment([100, 120, 120, 120, 120, 120])[
                "sustained_growth"
            ]
        )

    def test_clone_record_marks_exact_full_domain_only(self):
        events = [
            SimpleNamespace(key="aten::clone", input_shapes=[(2, 3)]),
            SimpleNamespace(key="aten::add", input_shapes=[]),
        ]
        self.assertEqual(
            stability._copy_record(events, ((2, 3),)),
            [
                {
                    "operator": "aten::clone",
                    "input_shapes": ((2, 3),),
                    "full_domain": True,
                }
            ],
        )

    def test_evaluator_rejects_storage_nonfinite_and_nonfinite_error(self):
        for override in (
            {"storage_stable": False},
            {"all_batches_finite": False},
            {"field_error_samples": [float("nan")]},
            {"rss_assessment": {"adequate": True, "sustained_growth": True}},
        ):
            self.assertFalse(stability._evaluate(_result(**override))["diagnostic_ok"])

    def test_evaluator_rejects_recurring_clone_and_compiler_fallback(self):
        copy_records = [
            {"full_domain_copy_count": 0},
            {"full_domain_copy_count": 1},
            {"full_domain_copy_count": 1},
        ]
        self.assertFalse(
            stability._evaluate(_result(operator_copy_records=copy_records))[
                "diagnostic_ok"
            ]
        )
        self.assertFalse(
            stability._evaluate(
                _result(
                    compiler_counter_delta={
                        "graph_breaks": 1,
                        "unique_graphs": 0,
                        "frames_total": 0,
                    }
                )
            )["diagnostic_ok"]
        )

    def test_evaluator_marks_eager_or_short_samples_diagnostic_only(self):
        self.assertFalse(
            stability._evaluate(_result(compile_policy="eager"))["qualified"]
        )
        self.assertFalse(
            stability._evaluate(
                _result(
                    batches=3,
                    steps_per_batch=1,
                    rss_assessment={"adequate": False, "sustained_growth": False},
                )
            )["qualified"]
        )
        self.assertTrue(
            stability._evaluate(
                _result(
                    batches=3,
                    steps_per_batch=1,
                    rss_assessment={"adequate": False, "sustained_growth": True},
                )
            )["diagnostic_ok"]
        )

    def test_evaluator_never_qualifies_observation_only_or_unverified_lowering(self):
        self.assertFalse(
            stability._evaluate(_result(observation_only=True))["qualified"]
        )
        self.assertFalse(
            stability._evaluate(_result(lowered_copy_materialization_verified=False))[
                "qualified"
            ]
        )

    def test_evaluator_rejects_stable_live_increasing_reserved_cuda(self):
        assessment = {"adequate": True, "sustained_growth": False}
        self.assertFalse(
            stability._evaluate(
                _result(
                    device_type="cuda",
                    cuda_assessment={
                        "allocated": assessment,
                        "reserved": {"adequate": True, "sustained_growth": True},
                    },
                )
            )["diagnostic_ok"]
        )

    def test_copy_record_detects_non_clone_materialization(self):
        event = SimpleNamespace(key="aten::copy_", input_shapes=[(2, 3)])
        self.assertTrue(stability._copy_record([event], ((2, 3),))[0]["full_domain"])

    def test_provenance_binds_all_git_calls_to_candidate_root(self):
        root = stability.Path(__file__).resolve().parents[1]
        calls = []

        def run(command, **_kwargs):
            calls.append(command)
            stdout = (
                str(root)
                if command[-2:] == ("rev-parse", "--show-toplevel")
                else "abc\n"
            )
            return SimpleNamespace(returncode=0, stdout=stdout)

        with mock.patch.object(stability.subprocess, "run", side_effect=run):
            provenance = stability._candidate_provenance()
        self.assertTrue(provenance["checkout_module_origin_valid"])
        self.assertTrue(all(command[:2] == ("git", "-C") for command in calls))
        self.assertEqual(len(provenance["runtime_source_sha256"]), 64)
        self.assertEqual(
            set(provenance["runtime_source_files_sha256"]),
            {"gmes/torch_fdtd.py", "gmes/torch_plan.py", "gmes/torch_source.py"},
        )

    def test_reevaluation_binds_raw_and_current_evaluator_without_recertifying(self):
        raw = _result(
            harness_sha256="old-collector", diagnostic_ok=False, qualified=False
        )
        record = stability.reevaluate(raw, json.dumps(raw, sort_keys=True).encode())
        self.assertEqual(record["raw_collector_harness_sha256"], "old-collector")
        self.assertFalse(record["new_collection"])
        self.assertIsInstance(record["reevaluator_harness_sha256"], str)
        self.assertFalse(record["effective_decision"]["qualified"])
        self.assertTrue(record["current_evaluator_diagnostic"]["qualified"])
        self.assertIn(
            "raw collection was not qualified", record["non_promotion_reasons"]
        )

    def test_reevaluation_never_promotes_missing_or_failed_raw_status(self):
        partial = _result(qualified=False)
        del partial["qualified"]
        failed = _result(
            qualified=False, diagnostic_ok=False, hard_failures=["historical failure"]
        )
        for raw, reason in (
            (partial, "raw qualification status is missing"),
            (failed, "raw collection recorded a failure"),
        ):
            record = stability.reevaluate(raw, b"raw")
            self.assertTrue(record["current_evaluator_diagnostic"]["qualified"])
            self.assertFalse(record["effective_decision"]["qualified"])
            self.assertIn(reason, record["non_promotion_reasons"])

    def test_reevaluation_downgrades_an_old_qualified_record_on_new_failure(self):
        raw = _result(
            qualified=True,
            diagnostic_ok=True,
            rss_assessment={"adequate": True, "sustained_growth": True},
        )
        record = stability.reevaluate(raw, b"raw")
        self.assertFalse(record["current_evaluator_diagnostic"]["qualified"])
        self.assertFalse(record["effective_decision"]["qualified"])
        self.assertIn(
            "current evaluator diagnostic is not qualified",
            record["non_promotion_reasons"],
        )

    def test_reevaluation_cli_emits_effective_not_promoted_status(self):
        raw = _result(qualified=False, diagnostic_ok=True)
        with tempfile.TemporaryDirectory() as directory:
            raw_path = stability.Path(directory) / "raw.json"
            output_path = stability.Path(directory) / "result.json"
            raw_path.write_text(json.dumps(raw))
            with mock.patch.object(
                sys,
                "argv",
                [
                    "torch_memory_stability.py",
                    "--reevaluate",
                    str(raw_path),
                    "--output",
                    str(output_path),
                ],
            ):
                stability.main()
            result = json.loads(output_path.read_text())
        self.assertNotIn("qualified", result)
        self.assertTrue(result["current_evaluator_diagnostic"]["qualified"])
        self.assertFalse(result["effective_decision"]["qualified"])

    def test_reevaluation_derives_split_cuda_assessments_from_raw_samples(self):
        raw = _result(
            device_type="cuda",
            cuda_samples=[{"allocated": 10, "reserved": 20}] * 15,
            cuda_assessment={"adequate": True, "sustained_growth": False},
        )
        record = stability.reevaluate(raw, b"raw")
        self.assertEqual(
            record["evaluation_normalizations"],
            ["derived-v3-cuda-assessments-from-raw-samples"],
        )
        self.assertNotIn(
            "CUDA live allocated memory shows sustained retained growth",
            record["current_evaluator_diagnostic"]["hard_failures"],
        )


if __name__ == "__main__":
    unittest.main()
