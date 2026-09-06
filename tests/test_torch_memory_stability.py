"""Fail-closed classification coverage for repeated-advance stability evidence."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from benchmarks import torch_memory_stability as stability


def _result(**overrides):
    result = {
        "checkout_module_origin_valid": True,
        "candidate_commit": "a" * 40,
        "candidate_dirty": False,
        "collector_harness_sha256": "b" * 64,
        "runtime_source_sha256": "c" * 64,
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
        "observer_warmup": {
            "complete": True,
            "compiler_counter_collector_primed": True,
            "field_error_checked": True,
            "rss_samples_bytes": [100],
            "candidate_tensors": [4],
            "reference_tensors": [4],
            "candidate_finite": [True],
            "reference_finite": [True],
            "field_error_samples": [0.0],
            "storage_stable": [True],
            "cuda_samples": [],
        },
    }
    result.update(overrides)
    if result["device_type"] == "cuda" and "observer_warmup" not in overrides:
        result["observer_warmup"] = {
            **result["observer_warmup"],
            "cuda_samples": [{"allocated": 10, "reserved": 20}],
        }
    return result


class StabilityEvidenceTest(unittest.TestCase):
    def test_cuda_memory_snapshot_reads_both_totals_once_without_legacy_flatteners(
        self,
    ):
        snapshot = {
            "allocated_bytes": {"all": {"current": 11}},
            "reserved_bytes": {"all": {"current": 22}},
        }
        device = torch.device("cuda", 0)
        with (
            mock.patch.object(
                stability.torch.cuda,
                "memory_stats_as_nested_dict",
                return_value=snapshot,
            ) as stats,
            mock.patch.object(stability.torch.cuda, "memory_allocated") as allocated,
            mock.patch.object(stability.torch.cuda, "memory_reserved") as reserved,
        ):
            self.assertEqual(stability._cuda_current_memory(device), (11, 22))
        stats.assert_called_once_with(device)
        allocated.assert_not_called()
        reserved.assert_not_called()

    def test_cuda_memory_snapshot_rejects_missing_malformed_bool_and_negative_values(
        self,
    ):
        valid = {
            "allocated_bytes": {"all": {"current": 11}},
            "reserved_bytes": {"all": {"current": 22}},
        }
        invalid_snapshots = (
            {},
            {"allocated_bytes": {"all": {"current": 11}}},
            {
                "allocated_bytes": {"all": {"current": "11"}},
                "reserved_bytes": {"all": {"current": 22}},
            },
            {
                "allocated_bytes": {"all": {"current": True}},
                "reserved_bytes": {"all": {"current": 22}},
            },
            {
                "allocated_bytes": {"all": {"current": 11}},
                "reserved_bytes": {"all": {"current": -1}},
            },
        )
        for snapshot in invalid_snapshots:
            with (
                self.subTest(snapshot=snapshot),
                mock.patch.object(
                    stability.torch.cuda,
                    "memory_stats_as_nested_dict",
                    return_value=snapshot,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "non-negative integer"):
                    stability._cuda_current_memory(torch.device("cuda", 0))
        self.assertEqual(valid["allocated_bytes"]["all"]["current"], 11)

    def test_cuda_observation_uses_one_snapshot_and_not_legacy_flatteners(self):
        telemetry = stability._new_telemetry(1, cuda=True)
        candidate = SimpleNamespace(device=torch.device("cuda", 0))
        reference = SimpleNamespace(device=torch.device("cuda", 0))
        observer = SimpleNamespace(
            slots=(object(),), finite=lambda: True, stable=lambda: True
        )
        snapshot = {
            "allocated_bytes": {"all": {"current": 11}},
            "reserved_bytes": {"all": {"current": 22}},
        }
        with (
            mock.patch.object(stability, "_synchronize"),
            mock.patch.object(
                stability.torch.cuda,
                "memory_stats_as_nested_dict",
                return_value=snapshot,
            ) as stats,
            mock.patch.object(stability.torch.cuda, "memory_allocated") as allocated,
            mock.patch.object(stability.torch.cuda, "memory_reserved") as reserved,
        ):
            stability._observe_into(
                telemetry,
                0,
                candidate=candidate,
                reference=reference,
                read_rss=lambda: 33,
                candidate_live=observer,
                reference_live=observer,
                candidate_storage=observer,
                reference_storage=observer,
                compare_fields=False,
            )
        stats.assert_called_once_with(candidate.device)
        allocated.assert_not_called()
        reserved.assert_not_called()
        self.assertEqual(telemetry.cuda_allocated.tolist(), [11])
        self.assertEqual(telemetry.cuda_reserved.tolist(), [22])

    def test_one_batch_is_diagnostic_only_not_an_invalid_request(self):
        decision = stability._evaluate(
            _result(
                batches=1,
                steps_per_batch=1,
                rss_assessment={"adequate": False, "sustained_growth": None},
            )
        )
        self.assertTrue(decision["diagnostic_ok"])
        self.assertFalse(decision["qualified"])
        self.assertIn(
            "fewer than 15 post-warmup batches were observed",
            decision["qualification_errors"],
        )

    def test_phase_progress_emits_fixed_marker_only_when_enabled(self):
        with mock.patch.object(stability.os, "write") as write:
            stability._phase_progress(False, "warmup-start")
            write.assert_not_called()
            stability._phase_progress(True, "warmup-start")
        write.assert_called_once_with(2, b"torch-memory-stability phase=warmup-start\n")

    def test_prebound_storage_observer_detects_replacement_and_nonfinite(self):
        for name in ("field", "material", "source"):
            with self.subTest(name=name):
                owner = torch.nn.Module()
                owner.register_buffer(name, torch.ones(2))
                observer = stability._buffer_observer((owner,))
                self.assertTrue(observer.stable())
                self.assertTrue(observer.finite())
                owner.register_buffer(name, torch.full((2,), float("nan")))
                self.assertFalse(observer.stable())
                self.assertFalse(observer.finite())

    def test_prebound_storage_observer_rejects_added_or_removed_buffer_keys(self):
        added = torch.nn.Module()
        added.register_buffer("field", torch.ones(2))
        observer = stability._buffer_observer((added,))
        added.register_buffer("unexpected", torch.ones(2))
        self.assertFalse(observer.stable())

        removed = torch.nn.Module()
        removed.register_buffer("field", torch.ones(2))
        observer = stability._buffer_observer((removed,))
        del removed._buffers["field"]
        self.assertFalse(observer.stable())
        self.assertFalse(observer.finite())

    def test_preallocated_telemetry_keeps_observer_warmup_separate(self):
        observer = stability._new_telemetry(1, cuda=False)
        measured = stability._new_telemetry(3, cuda=False)
        observer.rss[:] = [100]
        measured.rss[:] = [120, 120, 120]
        for telemetry in (observer, measured):
            telemetry.candidate_tensors[:] = 4
            telemetry.reference_tensors[:] = 4
            telemetry.candidate_finite[:] = True
            telemetry.reference_finite[:] = True
            telemetry.field_error[:] = 0.0
            telemetry.storage_stable[:] = True
        self.assertEqual(
            stability._telemetry_report(observer)["rss_samples_bytes"], [100]
        )
        samples = stability._telemetry_report(measured)["rss_samples_bytes"]
        self.assertEqual(samples, [120, 120, 120])
        self.assertFalse(stability._growth_assessment(samples)["sustained_growth"])

    def test_missing_or_malformed_warmup_blocks_qualification(self):
        for warmup in (None, {}, {"complete": True}):
            decision = stability._evaluate(_result(observer_warmup=warmup))
            self.assertFalse(decision["qualified"])
            self.assertIn(
                "complete healthy primed observer warmup evidence is missing",
                decision["hard_failures"],
            )

    def test_unhealthy_warmup_cannot_qualify_a_both_runtime_collection(self):
        for name in (
            "candidate_finite",
            "reference_finite",
            "storage_stable",
        ):
            with self.subTest(name=name):
                warmup = dict(_result()["observer_warmup"])
                warmup[name] = [False]
                decision = stability._evaluate(_result(observer_warmup=warmup))
                self.assertFalse(decision["qualified"])
                self.assertIn(
                    "complete healthy primed observer warmup evidence is missing",
                    decision["hard_failures"],
                )
        warmup = dict(_result()["observer_warmup"])
        warmup["field_error_checked"] = False
        decision = stability._evaluate(_result(observer_warmup=warmup))
        self.assertFalse(decision["qualified"])
        self.assertIn(
            "complete healthy primed observer warmup evidence is missing",
            decision["hard_failures"],
        )

    def test_legacy_qualified_record_without_warmup_cannot_be_promoted(self):
        raw = _result(qualified=True, diagnostic_ok=True)
        del raw["observer_warmup"]
        record = stability.reevaluate(raw, b"legacy")
        self.assertFalse(record["current_evaluator_diagnostic"]["qualified"])
        self.assertFalse(record["effective_decision"]["qualified"])
        self.assertIn(
            "complete healthy primed observer warmup evidence is missing",
            record["current_evaluator_diagnostic"]["hard_failures"],
        )

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

    def test_single_runtime_control_remains_diagnostic_without_field_comparison(self):
        decision = stability._evaluate(
            _result(advance_role="reference", field_error_checked=False)
        )
        self.assertTrue(decision["diagnostic_ok"])
        self.assertFalse(decision["qualified"])
        self.assertIn(
            "single-runtime or observation control cannot qualify advance stability",
            decision["qualification_errors"],
        )

    def test_sustained_growth_fails_after_observer_warmup(self):
        decision = stability._evaluate(
            _result(
                advance_role="both",
                rss_assessment={"adequate": True, "sustained_growth": True},
            )
        )
        self.assertFalse(decision["diagnostic_ok"])
        self.assertIn(
            "RSS shows sustained post-warmup retained growth", decision["hard_failures"]
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

    def test_postprime_rss_growth_fails_with_flat_cuda_accounting(self):
        stable_cuda = {"adequate": True, "sustained_growth": False}
        decision = stability._evaluate(
            _result(
                device_type="cuda",
                cuda_assessment={
                    "allocated": stable_cuda,
                    "reserved": stable_cuda,
                },
                rss_assessment={"adequate": True, "sustained_growth": True},
            )
        )
        self.assertFalse(decision["diagnostic_ok"])
        self.assertIn(
            "RSS shows sustained post-warmup retained growth", decision["hard_failures"]
        )

    def test_all_nonboth_advance_roles_remain_diagnostic(self):
        for role in ("candidate", "reference", "none"):
            with self.subTest(role=role):
                decision = stability._evaluate(
                    _result(
                        advance_role=role,
                        observation_only=role == "none",
                        field_error_checked=role == "none",
                    )
                )
                self.assertTrue(decision["diagnostic_ok"])
                self.assertFalse(decision["qualified"])

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
            collector_harness_sha256="d" * 64,
            diagnostic_ok=False,
            qualified=False,
        )
        record = stability.reevaluate(raw, json.dumps(raw, sort_keys=True).encode())
        self.assertEqual(record["raw_collector_harness_sha256"], "d" * 64)
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
