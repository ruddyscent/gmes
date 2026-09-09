"""Negative controls and independent CPU equations for #114 qualification."""

import copy
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks import full_state_qualification as qualification


class TestFullArrayComparator:
    def setup_method(self):
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
        assert result["passed"]
        assert (sum(r["elements"] for r in result["arrays"])) == (
            sum(a.size for a in self.expected.values())
        )

    def test_single_interior_field_mutation_is_detected(self):
        actual = copy.deepcopy(self.expected)
        actual["step/20/field/Ey"][1, 2, 3] += 1e-3
        result = qualification.compare_arrays(self.expected, actual, self.tolerances)
        assert (result["failures"]) == (
            [{"key": "step/20/field/Ey", "reason": "values"}]
        )

    def test_single_material_state_mutation_is_detected(self):
        actual = copy.deepcopy(self.expected)
        key = "step/100/state/Ex/0-Drude/values"
        actual[key][37] += 1e-3
        assert (
            qualification.compare_arrays(self.expected, actual, self.tolerances)[
                "failures"
            ]
        ) == ([{"key": key, "reason": "values"}])

    def test_single_source_state_mutation_is_detected(self):
        actual = copy.deepcopy(self.expected)
        key = "step/100/source/Ex/0-PointSourceEx/values"
        actual[key][2] += 1e-3
        assert (
            qualification.compare_arrays(self.expected, actual, self.tolerances)[
                "failures"
            ]
        ) == ([{"key": key, "reason": "values"}])

    def test_single_source_auxiliary_state_mutation_is_detected(self):
        actual = copy.deepcopy(self.expected)
        key = "step/100/source_aux/0-GaussianBeam/field/Ex"
        actual[key][1, 2, 3] += 1e-3
        assert (
            qualification.compare_arrays(self.expected, actual, self.tolerances)[
                "failures"
            ]
        ) == ([{"key": key, "reason": "values"}])

    def test_replay_aggregation_rejects_nested_source_auxiliary_clock_mutation(self):
        state = {
            "fields": {"Ex": np.zeros((1, 1, 1))},
            "material": {"material/Ex/values": np.zeros(1)},
            "sources": {"source/Ex/values": np.zeros(1)},
            "auxiliaries": [
                {
                    "source_aux/0-TFSF/live_clock/time": np.asarray([0.1, 0.025]),
                    "source_aux/0-TFSF/field/Ex": np.zeros((1, 1, 1)),
                }
            ],
            "clocks": [{"source/time": np.asarray([0.1, 0.025])}],
        }
        expected = qualification._persistent_replay_arrays(state)
        actual = copy.deepcopy(expected)
        key = "source_auxiliary/rank-0/source_aux/0-TFSF/live_clock/time"
        actual[key][0] += 0.125
        assert not (
            qualification.compare_arrays(
                expected,
                actual,
                qualification._comparison_tolerances(
                    expected, {"rtol": 1e-12, "atol": 1e-14}
                ),
            )["passed"]
        )
        actual = copy.deepcopy(expected)
        actual[key][0] = np.nextafter(actual[key][0], np.inf)
        assert not (
            qualification.compare_arrays(
                expected,
                actual,
                qualification._live_clock_tolerances(
                    expected, {"rtol": 1e-12, "atol": 1e-14}
                ),
            )["passed"]
        )

    @pytest.mark.parametrize(
        "mutation_reason_case", range(3), ids=("missing", "shape", "unexpected")
    )
    def test_missing_unexpected_and_broadcastable_shape_are_rejected(
        self, mutation_reason_case
    ):
        key = "step/5/state/Ex/0-Drude/values"
        mutation_reason_case_values = tuple(
            (
                (lambda a: a.pop(key), "missing"),
                (lambda a: a.update({key: a[key][None, :]}), "shape"),
                (lambda a: a.update({"unexpected": np.zeros(1)}), "unexpected"),
            )
        )
        assert len(mutation_reason_case_values) == 3
        mutation, reason = mutation_reason_case_values[mutation_reason_case]
        actual = copy.deepcopy(self.expected)
        mutation(actual)
        result = qualification.compare_arrays(self.expected, actual, self.tolerances)
        assert not (result["passed"])
        assert (reason) in ([r["reason"] for r in result["failures"]])

    def test_nonfinite_and_unpinned_tolerance_are_rejected(self):
        key = "step/2/field/Ex"
        for value in (float("nan"), float("inf")):
            actual = copy.deepcopy(self.expected)
            actual[key][1, 1, 1] = value
            assert not (
                qualification.compare_arrays(self.expected, actual, self.tolerances)[
                    "passed"
                ]
            )
        assert not (
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
        with pytest.raises(ValueError, match="capture provenance"):
            qualification.validate_capture_contract(
                actual, qualification.CAPTURES, shapes
            )
        actual = copy.deepcopy(self.expected)
        actual["step/100/time"][0] += 1
        with pytest.raises(ValueError, match="capture clock"):
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
        with pytest.raises(ValueError, match="capture clock"):
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
        with pytest.raises(ValueError, match="duplicate global ownership"):
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
        assert (metadata) == ({"logical": 1, "name": "Public GPU model"})
        assert ("uuid") not in (metadata)
        assert ("GPU-private-stable-identifier") not in (repr(metadata))

    @pytest.mark.parametrize(
        "values_case",
        range(13),
        ids=(
            "capture-field",
            "capture-material",
            "capture-source",
            "capture-auxiliary",
            "capture-clock",
            "source",
            "auxiliary",
            "clock",
            "checkpoint",
            "state-count",
            "replay-count",
            "inventory-count",
            "rank-status",
        ),
    )
    def test_two_gpu_report_failures_propagate_to_a_nonzero_exit_code(
        self, values_case
    ):
        passing_capture = {
            "field_comparison": {"passed": True},
            "material_comparison": {"passed": True},
            "source_comparison": {"passed": True},
            "source_auxiliary_comparisons": [{"comparison": {"passed": True}}],
            "source_clock_comparisons": [{"comparison": {"passed": True}}],
        }
        passing_source = {"passed": True}
        passing_auxiliaries = [{"comparison": {"passed": True}}]
        passing_clocks = [{"comparison": {"passed": True}}]
        passing_checkpoint = {"passed": True}
        arguments = (
            [passing_capture],
            passing_source,
            passing_auxiliaries,
            passing_clocks,
            passing_checkpoint,
            [1, 1],
            [1, 1],
            [1, 1],
            1,
        )
        assert qualification._two_gpu_report_passed(*arguments)
        assert (qualification._two_gpu_report_exit_code({"passed": True})) == (0)
        failures = []
        for name in (
            "field_comparison",
            "material_comparison",
            "source_comparison",
        ):
            capture = dict(passing_capture)
            capture[name] = {"passed": False}
            failures.append(([capture], *arguments[1:]))
        for name in (
            "source_auxiliary_comparisons",
            "source_clock_comparisons",
        ):
            capture = dict(passing_capture)
            capture[name] = [{"comparison": {"passed": False}}]
            failures.append(([capture], *arguments[1:]))
        for index, value in (
            (1, {"passed": False}),
            (2, [{"comparison": {"passed": False}}]),
            (3, [{"comparison": {"passed": False}}]),
            (4, {"passed": False}),
            (5, [1, 0]),
            (6, [1, 0]),
            (7, [1, 0]),
            (8, 0),
        ):
            values = list(arguments)
            values[index] = value
            failures.append(tuple(values))
        values_case_values = tuple(failures)
        assert len(values_case_values) == 13
        values = values_case_values[values_case]
        assert not (qualification._two_gpu_report_passed(*values))
        assert (qualification._two_gpu_report_exit_code({"passed": False})) == (1)
        with pytest.raises(ValueError, match="boolean passed"):
            qualification._two_gpu_report_exit_code({})

    @pytest.mark.parametrize("exit_code_case", range(2), ids=("passed", "failed"))
    def test_two_gpu_cli_returns_the_rank_zero_report_exit_code(self, exit_code_case):
        exit_code_case_values = tuple((0, 1))
        assert len(exit_code_case_values) == 2
        exit_code = exit_code_case_values[exit_code_case]
        with (
            patch.object(
                qualification, "run_two_gpu_partition_case", return_value=exit_code
            ) as runner,
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
            assert (qualification.main()) == (exit_code)
            assert (runner.call_args.kwargs["compile_policy"]) == ("eager")

    def test_two_gpu_compile_policy_records_and_propagates_to_both_runtimes(self):
        launch = SimpleNamespace(rank=0)
        distributed_options = qualification._two_gpu_runtime_options(
            "compile", launch=launch
        )
        serial_options = qualification._two_gpu_runtime_options("compile")
        assert (distributed_options["compile_policy"]) == ("compile")
        assert (serial_options["compile_policy"]) == ("compile")
        assert (distributed_options["launch"]) is (launch)
        assert ("launch") not in (serial_options)
        assert (qualification._two_gpu_execution_record("compile")) == (
            {
                "scope": "compiled-two-gpu-full-state-serial-torch-comparison",
                "compile_policy": "compile",
                "execution_mode": "graph",
            }
        )

    def test_two_gpu_capture_is_disabled_by_default_and_ordered_for_compile(self):
        calls = []

        class Runtime:
            def __init__(self, name):
                self.name = name

            def capture_cuda_graphs(self):
                calls.append(self.name)

        distributed = Runtime("distributed")
        serial = Runtime("serial")
        qualification._capture_two_gpu_compute_regions(
            distributed, serial, rank=0, compile_policy="eager"
        )
        assert (calls) == ([])
        qualification._capture_two_gpu_compute_regions(
            distributed, serial, rank=0, compile_policy="compile"
        )
        assert (calls) == (["distributed", "serial"])

    def test_two_gpu_cli_accepts_compile_and_rejects_other_modes(self):
        with (
            patch.object(
                qualification, "run_two_gpu_partition_case", return_value=0
            ) as runner,
            patch.object(
                sys,
                "argv",
                [
                    "full_state_qualification.py",
                    "--mode",
                    "two-gpu",
                    "--compile-policy",
                    "compile",
                    "--output-dir",
                    "unused-private-output",
                ],
            ),
        ):
            assert (qualification.main()) == (0)
            assert (runner.call_args.kwargs["compile_policy"]) == ("compile")
        with (
            patch.object(
                sys,
                "argv",
                [
                    "full_state_qualification.py",
                    "--mode",
                    "analytic",
                    "--compile-policy",
                    "compile",
                    "--output-dir",
                    "unused-private-output",
                ],
            ),
            pytest.raises(SystemExit, match="2") as error,
        ):
            qualification.main()
        assert (error.value.code) == (2)

    def test_two_gpu_rejects_unknown_compile_policy_before_hardware_import(self):
        with pytest.raises(ValueError, match="compile policy"):
            qualification.run_two_gpu_partition_case(
                "unused-private-output", compile_policy="automatic"
            )


class TestDistributedSourceCrossing:
    def test_rank_local_tfsf_crossing_matches_serial_source_and_auxiliary_state(self):
        import gmes
        from gmes.torch_distributed import TwoGpuDecomposition, rank_local_space
        from gmes.torch_fdtd import DistributedLaunch, TorchSimulation

        space = gmes.Cartesian((5, 4, 4), 1)
        decomposition = TwoGpuDecomposition(
            (5, 4, 4), 0, 2, (1.0, 1.0), (0.5, 0.5), 16, 1
        )

        def auxiliary_factory(**kwargs):
            auxiliary_space = kwargs.pop("space")
            auxiliary_runtime = kwargs.pop("runtime")
            return TorchSimulation(
                space=auxiliary_space,
                runtime=replace(auxiliary_runtime, launch=DistributedLaunch()),
                **kwargs,
            )

        def make_local(rank):
            return TorchSimulation(
                space=rank_local_space(space, decomposition, rank),
                geometry=qualification._partition_geometry(gmes),
                sources=qualification._partition_sources(gmes),
                runtime=gmes.TorchRuntimeConfig(
                    device="cpu",
                    precision="float64",
                    compile_policy="eager",
                    cpu_threads=1,
                    cpu_interop_threads=1,
                    launch=DistributedLaunch(
                        rank=rank,
                        world_size=2,
                        local_rank=rank,
                        local_world_size=2,
                    ),
                ),
                dt=0.025,
                _distributed_partition=decomposition,
                _auxiliary_factory=auxiliary_factory,
            )

        serial = TorchSimulation(
            space=space,
            geometry=qualification._partition_geometry(gmes),
            sources=qualification._partition_sources(gmes),
            runtime=gmes.TorchRuntimeConfig(
                device="cpu", precision="float64", cpu_threads=1, cpu_interop_threads=1
            ),
            dt=0.025,
        )
        locals_ = tuple(make_local(rank) for rank in (0, 1))
        local_rows = tuple(
            qualification._source_rows(simulation, offset=decomposition.offset(rank))
            for rank, simulation in enumerate(locals_)
        )
        assert all(
            qualification._transparent_source_ownership(rows) for rows in local_rows
        )
        for rows in local_rows:
            total = sum(
                len(value) for key, value in rows.items() if key.endswith("/indices")
            )
            assert (total) == (
                qualification._point_source_ownership(rows)
                + qualification._transparent_source_ownership(rows)
            )
        distributed_rows = qualification._canonical_rows(
            (
                (
                    key.removesuffix("/indices"),
                    values,
                    rows[key.replace("/indices", "/values")],
                )
                for rows in local_rows
                for key, values in rows.items()
                if key.endswith("/indices")
            )
        )
        serial_rows = qualification._source_rows(serial)
        assert qualification.compare_arrays(
            serial_rows,
            distributed_rows,
            dict.fromkeys(serial_rows, {"rtol": 0.0, "atol": 0.0}),
        )["passed"]
        for simulation in (serial, *locals_):
            simulation.sources.auxiliaries[0].advance(3)
        serial_auxiliary = qualification._source_auxiliary_arrays(serial)
        for simulation in locals_:
            assert qualification.compare_arrays(
                serial_auxiliary,
                qualification._source_auxiliary_arrays(simulation),
                qualification._comparison_tolerances(
                    serial_auxiliary, {"rtol": 1e-12, "atol": 1e-14}
                ),
            )["passed"]
        baseline = qualification._source_auxiliary_arrays(serial)
        auxiliary = serial.sources.auxiliaries[0]
        assert not (np.any(auxiliary.host_snapshot()["Ey"]))
        checkpoint_key = next(
            key for key in baseline if key.endswith("/checkpoint/state/pml_ey_0_state")
        )
        pml_ey_before = auxiliary.state.pml_ey_0_state.clone()
        auxiliary.state.pml_ey_0_state[0, 0].add_(0.125)
        try:
            assert not (
                qualification.compare_arrays(
                    baseline,
                    qualification._source_auxiliary_arrays(serial),
                    qualification._comparison_tolerances(
                        baseline, {"rtol": 1e-12, "atol": 1e-14}
                    ),
                )["passed"]
            )
        finally:
            auxiliary.state.pml_ey_0_state.copy_(pml_ey_before)
        assert (checkpoint_key) in (baseline)
        auxiliary.state.source_time.add_(0.125)
        auxiliary.state.time_step.mul_(2)
        assert not (
            qualification.compare_arrays(
                baseline,
                qualification._source_auxiliary_arrays(serial),
                qualification._comparison_tolerances(
                    baseline, {"rtol": 1e-12, "atol": 1e-14}
                ),
            )["passed"]
        )
        live_clock = qualification._source_auxiliary_arrays(serial)
        one_ulp_clock = copy.deepcopy(live_clock)
        clock_key = next(key for key in live_clock if key.endswith("/live_clock/time"))
        one_ulp_clock[clock_key][0] = np.nextafter(one_ulp_clock[clock_key][0], np.inf)
        assert not (
            qualification.compare_arrays(
                live_clock,
                one_ulp_clock,
                qualification._live_clock_tolerances(
                    live_clock,
                    {"rtol": 1e-12, "atol": 1e-14},
                ),
            )["passed"]
        )


class TestNativeCaptureCompatibility:
    def test_eager_execution_record_is_not_compiled_scope(self):
        import gmes

        simulation = gmes.TorchSimulation(
            space=gmes.Cartesian((1, 1, 0), 2),
            geometry=[gmes.DefaultMedium(gmes.Dielectric())],
            runtime=gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
        )
        assert (qualification._native_execution_record(simulation)) == (
            {
                "scope": "eager-full-state-native-correctness",
                "compile_policy": "eager",
                "execution_policy": "auto",
            }
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
        assert (row.shape) == ((1, 4))
        np.testing.assert_allclose(row[0, :3], (0.001, 1.7, 1.05), rtol=0, atol=0)
        assert np.isfinite(row[0, 3])

    def test_tfsf_capture_validation_rejects_batch_mutations(self):
        import torch

        from gmes.torch_source import TorchPointSourceBatch, TorchTransparentBatch

        manifest = qualification.native_oracle.load_manifest()
        spec = next(
            case
            for case in manifest["correctness"]
            if case["name"] == "tfsf-transparent"
        )
        simulation = qualification.torch_correctness._build_torch_simulation(
            spec,
            dt=0.025,
            threads=1,
            device="cpu",
            precision="float64",
            graph_mode="eager",
            compile_mode="default",
        )
        simulation.advance(1)
        captured = {}
        qualification.torch_correctness._independent_snapshot(simulation, 1, captured)
        baseline = qualification._native_tfsf_capture_state(simulation, captured, 1)
        assert baseline
        duplicate_targets = copy.deepcopy(captured)
        duplicate_key = "torch/step/1/sources/batches/0/targets"
        duplicate_targets[duplicate_key][1] = duplicate_targets[duplicate_key][0]
        with pytest.raises(ValueError, match="batch layout"):
            qualification._native_tfsf_capture_state(simulation, duplicate_targets, 1)
        batch = next(
            batch
            for batch in simulation.sources.batches
            if isinstance(batch, TorchTransparentBatch) and batch.component == "Ex"
        )
        original_samples = batch.samples
        batch.samples = batch.samples.to(torch.float64) + 0.25
        invalid_indices = {}
        qualification.torch_correctness._independent_snapshot(
            simulation, 1, invalid_indices
        )
        batch.samples = original_samples
        with pytest.raises(ValueError, match="batch layout"):
            qualification._native_tfsf_capture_state(simulation, invalid_indices, 1)
        auxiliary = next(
            item
            for item in simulation.sources.auxiliaries[0].sources.batches
            if isinstance(item, TorchPointSourceBatch)
        )
        original_amplitude = auxiliary.overwrite_amplitudes[0].detach().clone()
        with torch.no_grad():
            auxiliary.overwrite_amplitudes[0] = 1.125
        invalid_auxiliary = {}
        qualification.torch_correctness._independent_snapshot(
            simulation, 1, invalid_auxiliary
        )
        with torch.no_grad():
            auxiliary.overwrite_amplitudes[0] = original_amplitude
        with pytest.raises(ValueError, match="auxiliary drive"):
            qualification._native_tfsf_capture_state(simulation, invalid_auxiliary, 1)
        original_weight = batch.weights[0, 0].detach().clone()
        with torch.no_grad():
            batch.weights[0, 0] = torch.nextafter(
                batch.weights[0, 0],
                torch.tensor(float("inf"), dtype=batch.weights.dtype),
            )
        before_capture = {}
        qualification.torch_correctness._independent_snapshot(
            simulation, 1, before_capture
        )
        with torch.no_grad():
            batch.weights[0, 0] = original_weight
        with pytest.raises(ValueError, match="source state changed"):
            qualification._validate_native_tfsf_capture_stability(
                baseline,
                qualification._native_tfsf_capture_state(simulation, before_capture, 1),
            )
        with torch.no_grad():
            batch.weights[0, 0] = torch.nextafter(
                batch.weights[0, 0],
                torch.tensor(float("inf"), dtype=batch.weights.dtype),
            )
        after_capture = qualification._native_tfsf_capture_state(
            simulation, captured, 1
        )
        qualification._validate_native_tfsf_capture_stability(baseline, after_capture)

    def test_compatibility_retains_current_scientific_inputs(self):
        current = qualification.native_oracle.load_manifest()
        legacy = copy.deepcopy(current)
        legacy["reference"]["observer_tag"] = "native-oracle-observer-v4"
        legacy["reference"]["observer_commit"] = "1" * 40
        legacy["performance_gates"]["cpu_acceptance"] = {
            "historical": "legacy-schema-only"
        }
        compatibility = qualification._compatibility_payload(current, legacy)
        assert (qualification._non_performance_manifest(compatibility)) == (
            qualification._non_performance_manifest(current)
        )
        assert (compatibility["reference"]["observer_tag"]) == (
            current["reference"]["observer_tag"]
        )
        assert (compatibility["reference"]["observer_commit"]) == (
            current["reference"]["observer_commit"]
        )
        assert ("performance_observer_tag") not in (compatibility["reference"])
        assert ("performance_observer_commit") not in (compatibility["reference"])
        assert (compatibility["performance_gates"]["cpu_acceptance"]) == (
            legacy["performance_gates"]["cpu_acceptance"]
        )

    def test_compatibility_rejects_legacy_scientific_delta(self):
        current = qualification.native_oracle.load_manifest()
        legacy = copy.deepcopy(current)
        legacy["correctness"][0]["size"][0] += 1
        with pytest.raises(ValueError, match="non-performance capture input"):
            qualification._compatibility_payload(current, legacy)


class TestIndependentCpuQualification:
    @pytest.mark.parametrize("precision_case", range(2), ids=("float64", "float32"))
    def test_dm2_nonzero_state_closed_form_through_100_steps(self, precision_case):
        precision_case_values = tuple(("float64", "float32"))
        assert len(precision_case_values) == 2
        precision = precision_case_values[precision_case]
        result = qualification.run_dm2_invariant(
            precision=precision, output=None, final_step=100
        )
        assert (result["status"]) == ("passed")
        assert not (result["native_qualification"])
        assert (result["capture_steps"][-1]) == (100)

    @pytest.mark.parametrize("paired_case", range(2), ids=("real", "paired-real"))
    @pytest.mark.parametrize("size_case", range(3), ids=("1d", "2d", "3d"))
    @pytest.mark.parametrize("precision_case", range(2), ids=("float64", "float32"))
    def test_numpy_yee_full_fields_at_five_capture_steps(
        self, paired_case, size_case, precision_case
    ):
        precision_case_values = tuple(("float64", "float32"))
        assert len(precision_case_values) == 2
        precision = precision_case_values[precision_case]
        size_case_values = tuple(((2, 0, 0), (2, 2, 0), (2, 2, 2)))
        assert len(size_case_values) == 3
        size = size_case_values[size_case]
        paired_case_values = tuple((False, True))
        assert len(paired_case_values) == 2
        paired = paired_case_values[paired_case]
        result = qualification.run_analytic_case(
            size=size,
            precision=precision,
            paired_real=paired,
            output=None,
        )
        assert (result["status"]) == ("passed")
        assert not (result["native_qualification"])
        assert (result["reference"]["kind"]) == ("independent-numpy-equations")
