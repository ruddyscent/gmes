"""Negative controls and independent CPU equations for #114 qualification."""

import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmarks import full_state_qualification as qualification


class FullArrayComparatorTest(unittest.TestCase):
    def setUp(self):
        self.expected = {}
        for step in qualification.CAPTURES:
            for name in qualification.COMPONENTS:
                self.expected[f"step/{step}/field/{name}"] = np.full((3, 4, 5), 0.2)
            self.expected[f"step/{step}/time"] = np.asarray(
                [step + 2, (step + 2) * 0.1, 0.1]
            )
            self.expected[f"step/{step}/state/Ex/0-Drude/values"] = np.full(120, 0.3)
            self.expected[f"step/{step}/source/Ex/0-PointSourceEx/values"] = np.full(
                4, 0.4
            )
            self.expected[f"step/{step}/source_aux/0-GaussianBeam/field/Ex"] = np.full(
                (3, 4, 5), 0.5
            )
        self.tolerances = dict.fromkeys(self.expected, {"rtol": 1e-12, "atol": 1e-14})

    def test_all_elements_are_compared(self):
        result = qualification.compare_arrays(
            self.expected, copy.deepcopy(self.expected), self.tolerances
        )
        self.assertTrue(result["passed"])
        self.assertEqual(
            sum(r["elements"] for r in result["arrays"]),
            sum(a.size for a in self.expected.values()),
        )

    def test_single_interior_field_mutation_is_detected(self):
        actual = copy.deepcopy(self.expected)
        actual["step/20/field/Ey"][1, 2, 3] += 1e-3
        result = qualification.compare_arrays(self.expected, actual, self.tolerances)
        self.assertEqual(
            result["failures"], [{"key": "step/20/field/Ey", "reason": "values"}]
        )

    def test_single_material_state_mutation_is_detected(self):
        actual = copy.deepcopy(self.expected)
        key = "step/100/state/Ex/0-Drude/values"
        actual[key][37] += 1e-3
        self.assertEqual(
            qualification.compare_arrays(self.expected, actual, self.tolerances)[
                "failures"
            ],
            [{"key": key, "reason": "values"}],
        )

    def test_single_source_state_mutation_is_detected(self):
        actual = copy.deepcopy(self.expected)
        key = "step/100/source/Ex/0-PointSourceEx/values"
        actual[key][2] += 1e-3
        self.assertEqual(
            qualification.compare_arrays(self.expected, actual, self.tolerances)[
                "failures"
            ],
            [{"key": key, "reason": "values"}],
        )

    def test_single_source_auxiliary_state_mutation_is_detected(self):
        actual = copy.deepcopy(self.expected)
        key = "step/100/source_aux/0-GaussianBeam/field/Ex"
        actual[key][1, 2, 3] += 1e-3
        self.assertEqual(
            qualification.compare_arrays(self.expected, actual, self.tolerances)[
                "failures"
            ],
            [{"key": key, "reason": "values"}],
        )

    def test_missing_unexpected_and_broadcastable_shape_are_rejected(self):
        key = "step/5/state/Ex/0-Drude/values"
        for mutation, reason in (
            (lambda a: a.pop(key), "missing"),
            (lambda a: a.update({key: a[key][None, :]}), "shape"),
            (lambda a: a.update({"unexpected": np.zeros(1)}), "unexpected"),
        ):
            with self.subTest(reason=reason):
                actual = copy.deepcopy(self.expected)
                mutation(actual)
                result = qualification.compare_arrays(
                    self.expected, actual, self.tolerances
                )
                self.assertFalse(result["passed"])
                self.assertIn(reason, [r["reason"] for r in result["failures"]])

    def test_nonfinite_and_unpinned_tolerance_are_rejected(self):
        key = "step/2/field/Ex"
        for value in (float("nan"), float("inf")):
            actual = copy.deepcopy(self.expected)
            actual[key][1, 1, 1] = value
            self.assertFalse(
                qualification.compare_arrays(self.expected, actual, self.tolerances)[
                    "passed"
                ]
            )
        self.assertFalse(
            qualification.compare_arrays(self.expected, self.expected, {})["passed"]
        )

    def test_missing_capture_and_wrong_clock_are_rejected(self):
        shapes = dict.fromkeys(qualification.COMPONENTS, (3, 4, 5))
        qualification.validate_capture_contract(
            self.expected, qualification.CAPTURES, shapes
        )
        actual = {
            k: v for k, v in self.expected.items() if not k.startswith("step/20/")
        }
        with self.assertRaisesRegex(ValueError, "capture provenance"):
            qualification.validate_capture_contract(
                actual, qualification.CAPTURES, shapes
            )
        actual = copy.deepcopy(self.expected)
        actual["step/100/time"][0] += 1
        with self.assertRaisesRegex(ValueError, "capture clock"):
            qualification.validate_capture_contract(
                actual, qualification.CAPTURES, shapes
            )

    def test_one_ulp_float32_clock_mutation_is_rejected(self):
        shapes = dict.fromkeys(qualification.COMPONENTS, (3, 4, 5))
        actual = copy.deepcopy(self.expected)
        for step in qualification.CAPTURES:
            actual[f"step/{step}/time"] = np.asarray(
                [
                    step + 2,
                    float(
                        np.multiply(
                            np.float32(step + 2), np.float32(0.1), dtype=np.float32
                        )
                    ),
                    float(np.float32(0.1)),
                ]
            )
        key = "step/100/time"
        actual[key][1] = float(
            np.nextafter(np.float32(actual[key][1]), np.float32(np.inf))
        )
        with self.assertRaisesRegex(ValueError, "capture clock"):
            qualification.validate_capture_contract(
                actual, qualification.CAPTURES, shapes, precision="float32"
            )

    def test_distributed_row_canonicalization_rejects_duplicate_ownership(self):
        rows = qualification._canonical_rows(
            (
                (
                    "source/Ex/overwrite",
                    np.asarray([[2, 0, 0]], dtype=np.int64),
                    np.asarray([[1.0, 2.0]]),
                ),
                (
                    "source/Ex/overwrite",
                    np.asarray([[1, 0, 0]], dtype=np.int64),
                    np.asarray([[3.0, 4.0]]),
                ),
            )
        )
        np.testing.assert_array_equal(
            rows["source/Ex/overwrite/indices"], [[1, 0, 0], [2, 0, 0]]
        )
        with self.assertRaisesRegex(ValueError, "duplicate global ownership"):
            qualification._canonical_rows(
                (
                    (
                        "source/Ex/overwrite",
                        np.asarray([[1, 0, 0]], dtype=np.int64),
                        np.asarray([[1.0, 2.0]]),
                    ),
                    (
                        "source/Ex/overwrite",
                        np.asarray([[1, 0, 0]], dtype=np.int64),
                        np.asarray([[3.0, 4.0]]),
                    ),
                )
            )

    def test_material_ownership_keeps_only_the_explicit_yee_owner(self):
        indices = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int64)
        np.testing.assert_array_equal(
            qualification._owned_material_rows(
                indices, "Ey", rank=0, split_axis=0, local_shape=(3, 4, 4)
            ),
            [True, True, False],
        )
        np.testing.assert_array_equal(
            qualification._owned_material_rows(
                indices, "Hy", rank=1, split_axis=0, local_shape=(3, 4, 4)
            ),
            [False, True, True],
        )
        np.testing.assert_array_equal(
            qualification._owned_material_rows(
                indices, "Ex", rank=0, split_axis=0, local_shape=(3, 4, 4)
            ),
            [True, True, True],
        )

    def test_source_clock_is_read_from_the_live_candidate_state(self):
        import gmes

        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((1, 1, 0), 2),
            geometry=[gmes.DefaultMedium(gmes.Dielectric())],
            dt=0.125,
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
        )
        simulation.advance(4)
        clock = qualification._source_clock(simulation)
        np.testing.assert_array_equal(clock["source/step_count"], [4])
        np.testing.assert_array_equal(clock["source/time"], [0.5, 0.125])

    def test_two_gpu_report_metadata_excludes_stable_device_identifiers(self):
        metadata = qualification._public_device_metadata(
            1,
            SimpleNamespace(
                name="Public GPU model", uuid="GPU-private-stable-identifier"
            ),
        )
        self.assertEqual(metadata, {"logical": 1, "name": "Public GPU model"})
        self.assertNotIn("uuid", metadata)
        self.assertNotIn("GPU-private-stable-identifier", repr(metadata))

    def test_two_gpu_report_failures_propagate_to_a_nonzero_exit_code(self):
        passing_capture = {
            "field_comparison": {"passed": True},
            "material_comparison": {"passed": True},
        }
        passing_source = {"passed": True}
        passing_clocks = [{"comparison": {"passed": True}}]
        passing_checkpoint = {"passed": True}
        arguments = (
            [passing_capture],
            passing_source,
            passing_clocks,
            passing_checkpoint,
            [1, 1],
        )
        self.assertTrue(qualification._two_gpu_report_passed(*arguments))
        self.assertEqual(qualification._two_gpu_report_exit_code({"passed": True}), 0)
        failures = (
            (
                [
                    {
                        "field_comparison": {"passed": False},
                        "material_comparison": {"passed": True},
                    }
                ],
                passing_source,
                passing_clocks,
                passing_checkpoint,
                [1, 1],
            ),
            (
                [
                    {
                        "field_comparison": {"passed": True},
                        "material_comparison": {"passed": False},
                    }
                ],
                passing_source,
                passing_clocks,
                passing_checkpoint,
                [1, 1],
            ),
            (
                [passing_capture],
                {"passed": False},
                passing_clocks,
                passing_checkpoint,
                [1, 1],
            ),
            (
                [passing_capture],
                passing_source,
                [{"comparison": {"passed": False}}],
                passing_checkpoint,
                [1, 1],
            ),
            (
                [passing_capture],
                passing_source,
                passing_clocks,
                {"passed": False},
                [1, 1],
            ),
            (
                [passing_capture],
                passing_source,
                passing_clocks,
                passing_checkpoint,
                [1, 0],
            ),
        )
        for values in failures:
            with self.subTest(values=values):
                self.assertFalse(qualification._two_gpu_report_passed(*values))
                self.assertEqual(
                    qualification._two_gpu_report_exit_code({"passed": False}), 1
                )
        with self.assertRaisesRegex(ValueError, "boolean passed"):
            qualification._two_gpu_report_exit_code({})

    def test_two_gpu_cli_returns_the_rank_zero_report_exit_code(self):
        for exit_code in (0, 1):
            with (
                self.subTest(exit_code=exit_code),
                patch.object(
                    qualification, "run_two_gpu_partition_case", return_value=exit_code
                ),
                patch.object(
                    sys,
                    "argv",
                    [
                        "full_state_qualification.py",
                        "--mode",
                        "two-gpu",
                        "--output-dir",
                        "unused-private-output",
                    ],
                ),
            ):
                self.assertEqual(qualification.main(), exit_code)


class NativeCaptureCompatibilityTest(unittest.TestCase):
    def test_eager_execution_record_is_not_compiled_scope(self):
        import gmes

        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((1, 1, 0), 2),
            geometry=[gmes.DefaultMedium(gmes.Dielectric())],
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
        )
        self.assertEqual(
            qualification._native_execution_record(simulation),
            {
                "scope": "eager-full-state-native-correctness",
                "compile_policy": "eager",
                "execution_policy": "auto",
            },
        )

    def test_point_projection_contains_candidate_material_parameters(self):
        manifest = qualification.native_oracle.load_manifest()
        specification = next(
            case for case in manifest["correctness"] if case["name"] == "dielectric-1d"
        )
        simulation = qualification.torch_correctness._build_torch_simulation(
            specification,
            dt=0.025,
            threads=1,
            device="cpu",
            precision="float64",
            graph_mode="eager",
            compile_mode="default",
        )
        arrays = {"step/0/time": np.asarray([0, 0.0, 0.025])}
        qualification._point_observables(simulation, 0, arrays)
        row = arrays["step/0/source/Ex/0-PointSourceEx/values"].reshape(-1, 4)
        self.assertEqual(row.shape, (1, 4))
        np.testing.assert_allclose(row[0, :3], (0.001, 1.7, 1.05), rtol=0, atol=0)
        self.assertTrue(np.isfinite(row[0, 3]))

    def test_compatibility_retains_current_scientific_inputs(self):
        current = qualification.native_oracle.load_manifest()
        legacy = copy.deepcopy(current)
        legacy["reference"]["observer_tag"] = "native-oracle-observer-v4"
        legacy["reference"]["observer_commit"] = "1" * 40
        legacy["performance_gates"]["cpu_acceptance"] = {
            "historical": "legacy-schema-only"
        }
        compatibility = qualification._compatibility_payload(current, legacy)
        self.assertEqual(
            qualification._non_performance_manifest(compatibility),
            qualification._non_performance_manifest(current),
        )
        self.assertEqual(
            compatibility["reference"]["observer_tag"],
            current["reference"]["observer_tag"],
        )
        self.assertEqual(
            compatibility["reference"]["observer_commit"],
            current["reference"]["observer_commit"],
        )
        self.assertNotIn("performance_observer_tag", compatibility["reference"])
        self.assertNotIn("performance_observer_commit", compatibility["reference"])
        self.assertEqual(
            compatibility["performance_gates"]["cpu_acceptance"],
            legacy["performance_gates"]["cpu_acceptance"],
        )

    def test_compatibility_rejects_legacy_scientific_delta(self):
        current = qualification.native_oracle.load_manifest()
        legacy = copy.deepcopy(current)
        legacy["correctness"][0]["size"][0] += 1
        with self.assertRaisesRegex(ValueError, "non-performance capture input"):
            qualification._compatibility_payload(current, legacy)


class IndependentCpuQualificationTest(unittest.TestCase):
    def test_dm2_nonzero_state_closed_form_through_100_steps(self):
        for precision in ("float64", "float32"):
            with self.subTest(precision=precision):
                result = qualification.run_dm2_invariant(
                    precision=precision, output=None, final_step=100
                )
                self.assertEqual(
                    result["status"], "passed", result["comparison"]["failures"]
                )
                self.assertFalse(result["native_qualification"])
                self.assertEqual(result["capture_steps"][-1], 100)

    def test_numpy_yee_full_fields_at_five_capture_steps(self):
        for precision in ("float64", "float32"):
            for size in ((2, 0, 0), (2, 2, 0), (2, 2, 2)):
                for paired in (False, True):
                    with self.subTest(precision=precision, size=size, paired=paired):
                        result = qualification.run_analytic_case(
                            size=size,
                            precision=precision,
                            paired_real=paired,
                            output=None,
                        )
                        self.assertEqual(
                            result["status"], "passed", result["comparison"]["failures"]
                        )
                        self.assertFalse(result["native_qualification"])
                        self.assertEqual(
                            result["reference"]["kind"], "independent-numpy-equations"
                        )


if __name__ == "__main__":
    unittest.main()
