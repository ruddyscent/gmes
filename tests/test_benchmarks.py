import importlib.util
import hashlib
import json
import platform
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_benchmark(name):
    benchmark_path = Path(__file__).resolve().parents[1] / "benchmarks" / name
    spec = importlib.util.spec_from_file_location(
        name.removesuffix(".py"), benchmark_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_field_updates():
    return _load_benchmark("field_updates.py")


def load_torch_material_planner():
    return _load_benchmark("torch_material_planner.py")


def load_torch_dm2():
    return _load_benchmark("torch_dm2.py")


def load_torch_tuning():
    return _load_benchmark("torch_tuning.py")


class FieldUpdateBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gmes

        cls.gmes = gmes
        cls.benchmark = load_field_updates()

    def test_reports_repeated_initialization_and_step_metrics(self):
        result = self.benchmark.run_case(
            "small", warmup=0, steps=1, repeats=2, gmes=self.gmes
        )

        self.assertEqual(len(result["seconds_per_construction"]), 2)
        self.assertEqual(len(result["seconds_per_initialization"]), 2)
        self.assertEqual(len(result["seconds_per_step"]), 2)
        self.assertGreater(result["median_seconds_per_construction"], 0)
        self.assertGreater(result["median_seconds_per_initialization"], 0)
        self.assertGreater(result["median_seconds_per_step"], 0)
        self.assertIn("Ez", result["field_shapes"])
        self.assertEqual(result["field_shapes"]["Ez"]["shape"], [41, 41, 1])
        self.assertTrue(result["material_update_sizes"])
        self.assertGreater(result["native_update_plan_bytes"], 0)
        self.assertGreater(result["peak_rss_bytes"], 0)
        self.assertTrue(
            all(
                item["plan_bytes"] > 0 and item["plan_runs"] > 0
                for item in result["material_update_sizes"]
            )
        )

    def test_additional_initialization_workloads(self):
        heterogeneous = self.benchmark.build_simulation("heterogeneous", self.gmes)
        heterogeneous.init()
        complex_field = self.benchmark.build_simulation("complex", self.gmes)
        complex_field.init()

        self.assertEqual(len(heterogeneous.geom_list), 18)
        heterogeneous_materials = {
            item["material"]
            for item in self.benchmark.material_update_sizes(heterogeneous)
        }
        self.assertIn("DielectricEzReal", heterogeneous_materials)
        self.assertIn("CpmlEzReal", heterogeneous_materials)
        self.assertEqual(
            self.benchmark.field_shapes(complex_field)["Ez"]["dtype"],
            "complex128",
        )

    def test_state_heavy_workloads_cover_all_native_families(self):
        expected_materials = {
            "pml": "UpmlEzReal",
            "dispersive": "DrudeEzReal",
            "lorentz": "LorentzEzReal",
            "dcp": "DcpAdeEzReal",
            "dm2": "Dm2EzReal",
        }
        for case, material_name in expected_materials.items():
            with self.subTest(case=case):
                simulation = self.benchmark.build_simulation(case, self.gmes)
                simulation.init()
                materials = {
                    item["material"]
                    for item in self.benchmark.material_update_sizes(simulation)
                }
                self.assertIn(material_name, materials)


class TorchMaterialPlannerBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gmes

        cls.gmes = gmes
        cls.benchmark = load_torch_material_planner()

    def test_policy_matrix_fails_closed_until_runtime_paths_are_distinct(self):
        sample = {"median_seconds_per_step": 1.0}
        with patch.object(self.benchmark, "run_case", return_value=sample):
            result = self.benchmark.run_policy_matrix("homogeneous")
        self.assertFalse(result["comparison_valid"])
        self.assertIsNone(result["auto_to_fastest_forced_ratio"])
        self.assertIsNone(result["within_ten_percent"])
        self.assertFalse(result["passed"])

    def test_reports_plan_execution_and_policy_metrics(self):
        result = self.benchmark.run_case(
            "homogeneous",
            policy="auto",
            device="cpu",
            precision="float64",
            compile_policy="eager",
            threads=1,
            warmup=0,
            steps=1,
            repeats=1,
            tile_size=64,
            profile=False,
            gmes=self.gmes,
        )
        self.assertGreater(result["plan_creation_seconds"], 0)
        self.assertGreater(result["estimated_plan_bytes"], 0)
        self.assertGreater(result["bytes_per_active_component_cell"], 0)
        self.assertEqual(result["material_launches_per_step"], 6)
        self.assertGreater(result["cells_per_second"], 0)
        self.assertEqual(
            {
                bucket["selected_policy"]
                for component in result["decisions"]
                for bucket in component["buckets"]
            },
            {"dense"},
        )

    def test_pml_benchmark_reports_execution_and_storage_metrics(self):
        result = self.benchmark.run_case(
            "pml-thin",
            policy="auto",
            device="cpu",
            precision="float64",
            compile_policy="eager",
            threads=1,
            warmup=0,
            steps=1,
            repeats=1,
            tile_size=64,
            profile=True,
            gmes=self.gmes,
        )
        self.assertGreater(result["cells_per_second"], 0)
        self.assertGreater(result["pml"]["active_cells"], 0)
        self.assertGreater(result["pml"]["state_bytes"], 0)
        self.assertGreater(result["pml"]["gather_scatter_bytes_per_step"], 0)
        self.assertEqual(result["pml"]["launches_per_step"], 6)
        self.assertGreater(result["profile"]["gather_count"], 0)
        self.assertGreater(result["profile"]["scatter_count"], 0)
        self.assertEqual(
            result["timing_scope"]["included"], ("advance", "device_synchronize")
        )
        self.assertIn(
            "flush_probes",
            result["timing_scope"]["excluded_host_boundaries"],
        )

    def test_fragmented_coverage_matches_contiguous_target_counts(self):
        for coverage in (1, 10, 50, 90):
            counts = {}
            for layout in ("contiguous", "fragmented"):
                _, _, _, plans, _ = self.benchmark.build_host_plan(
                    f"coverage-{layout}-{coverage}",
                    policy="auto",
                    precision="float64",
                    device_type="cpu",
                    tile_size=64,
                    gmes=self.gmes,
                )
                counts[layout] = sum(
                    bucket.target_count
                    for plan in plans
                    for bucket in plan.buckets
                    if bucket.signature.model == "drude"
                )
            self.assertAlmostEqual(
                counts["fragmented"] / counts["contiguous"], 1.0, delta=0.1
            )

    def test_stateful_matrix_executes_with_exact_width_normalization(self):
        result = self.benchmark.run_case(
            "state-widths",
            policy="auto",
            device="cpu",
            precision="float64",
            compile_policy="eager",
            threads=1,
            warmup=0,
            steps=1,
            repeats=1,
            tile_size=64,
            profile=False,
            gmes=self.gmes,
        )
        self.assertGreater(result["cells_per_second"], 0)
        self.assertGreater(result["dispersive_state_bytes"], 0)
        self.assertEqual(result["state_width_policy"], "exact")
        self.assertEqual(result["state_padding_elements"], 0)
        self.assertGreater(result["state_padding_elements_avoided"], 0)
        self.assertTrue(result["state_width_decisions"])
        electric_widths = {
            tuple(item["state_shape"])
            for item in result["signatures"]
            if item["component"] == "Ex" and item["model"] == "drude"
        }
        self.assertEqual(electric_widths, {(1,), (2,), (4,), (8,)})
        magnetic_models = {
            item["model"] for item in result["signatures"] if item["component"] == "Hx"
        }
        self.assertEqual(magnetic_models, {"dielectric"})

    def test_thousand_dispersive_regions_share_signatures_and_launches(self):
        _, geometry, _, plans, _ = self.benchmark.build_host_plan(
            "many-dispersive-regions",
            policy="compact",
            precision="float64",
            device_type="cpu",
            tile_size=64,
            gmes=self.gmes,
        )
        self.assertEqual(len(geometry), 1001)
        drude_buckets = [
            bucket
            for plan in plans
            for bucket in plan.buckets
            if bucket.signature.model == "drude"
        ]
        self.assertEqual(len(drude_buckets), 3)
        self.assertTrue(
            all(len(bucket.coefficient_table) == 1 for bucket in drude_buckets)
        )
        self.assertEqual(sum(plan.launch_count for plan in plans), 9)
        self.assertEqual(sum(bucket.launch_count for bucket in drude_buckets), 3)


class TorchDm2BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gmes

        cls.gmes = gmes
        cls.benchmark = load_torch_dm2()

    def test_reports_state_convergence_and_fixed_storage(self):
        args = self.benchmark.parse_args(
            [
                "--case",
                "width-1",
                "--compile-policy",
                "eager",
                "--warmup",
                "1",
                "--steps",
                "1",
                "--repeats",
                "1",
            ]
        )
        result = self.benchmark.run_case(args, self.gmes)

        self.assertTrue(result["fixed_storage"])
        self.assertGreater(result["dm2_cells_per_second"], 0)
        self.assertGreater(result["state"]["persistent_state_bytes"], 0)
        self.assertGreater(result["state"]["scratch_bytes"], 0)
        self.assertEqual(result["state"]["transition_widths"], [1])
        self.assertTrue(result["iteration_distributions"])

    def test_mixed_widths_report_exact_and_padding_tradeoff(self):
        space, geometry = self.benchmark.build_case("mixed-widths", self.gmes)
        simulation = self.gmes.TorchSimulation(
            space=space,
            geometry=geometry,
            runtime=self.gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
        )
        metrics = self.benchmark._state_metrics(simulation)

        self.assertEqual(metrics["transition_widths"], [1, 2, 4, 8])
        self.assertGreater(metrics["bounded_padding_overhead"], 1)
        self.assertLess(
            metrics["exact_transition_elements"],
            metrics["bounded_padding_elements"],
        )

    def test_all_material_cases_execute_in_2d_and_3d(self):
        expected_models = {
            "cpml",
            "dcp-ade",
            "dcp-plrc",
            "dcp-rc",
            "dielectric",
            "dm2",
            "drude",
            "lorentz",
        }
        for case in ("all-material-2d", "all-material-3d"):
            with self.subTest(case=case):
                space, geometry = self.benchmark.build_case(case, self.gmes)
                simulation = self.gmes.TorchSimulation(
                    space=space,
                    geometry=geometry,
                    runtime=self.gmes.TorchRuntimeConfig(device="cpu", cpu_threads=1),
                )
                simulation.step()
                models = {
                    bucket.signature.model
                    for component in simulation.plan.components.values()
                    for bucket in component.buckets
                }
                self.assertTrue(expected_models.issubset(models))
                diagnostics = simulation.diagnostics()
                self.assertTrue(diagnostics["dm2"])
                self.assertGreater(diagnostics["pml"]["active_cells"], 0)
                self.assertEqual(
                    set(diagnostics["dispersive"]["models"]),
                    {"dcp-ade", "dcp-plrc", "dcp-rc", "drude", "lorentz"},
                )


class TorchTuningBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = load_torch_tuning()
        cls.manifest = cls.benchmark.load_manifest(cls.benchmark.MANIFEST)

    def test_timer_restores_the_post_warmup_checkpoint_for_hidden_runs(self):
        import torch

        class Simulation:
            device = torch.device("cpu")

            def __init__(self):
                self.step_count = 0
                self.sample_starts = []

            def load_checkpoint(self, checkpoint):
                self.step_count = checkpoint

            def advance(self, steps):
                self.sample_starts.append(self.step_count)
                self.step_count += steps

        simulation = Simulation()
        samples = self.benchmark._timer_samples(
            simulation,
            steps=2,
            repeats=3,
            threads=1,
            checkpoint=5,
        )
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(value > 0 for value in samples))
        self.assertEqual(simulation.step_count, 7)
        self.assertEqual(simulation.sample_starts, [5, 7, 5] * 3)

    def test_authoritative_timer_replays_real_warmup_for_each_repeat(self):
        import torch

        class Simulation:
            device = torch.device("cpu")

            def __init__(self):
                self.step_count = 0
                self.sample_starts = []

            def load_checkpoint(self, checkpoint):
                self.step_count = checkpoint

            def advance(self, steps):
                self.sample_starts.append(self.step_count)
                self.step_count += steps

        simulation = Simulation()
        with patch.object(
            self.benchmark.time,
            "perf_counter",
            side_effect=[0.0, 1.0] * 3,
        ):
            samples = self.benchmark._perf_counter_samples(
                simulation,
                steps=2,
                repeats=3,
                initial_checkpoint=0,
                warmup=5,
            )
        self.assertEqual(samples, [1.0, 1.0, 1.0])
        self.assertEqual(simulation.sample_starts, [0, 5] * 3)

    def test_trace_summary_counts_raw_allocator_events(self):
        trace = {
            "traceEvents": [
                {"name": "[memory]", "args": {"Bytes": 32}},
                {"name": "[memory]", "args": {"Bytes": -32}},
                {
                    "name": "Torch-Compiled Region: 0/0",
                    "ph": "X",
                    "ts": 0,
                    "dur": 10,
                    "pid": 1,
                    "tid": 1,
                },
                {"name": "Torch-Compiled Region: 1/0", "ph": "X"},
                {
                    "name": "aten::index_copy_",
                    "ph": "X",
                    "ts": 5,
                    "dur": 1,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "aten::index_add_",
                    "ph": "X",
                    "ts": 12,
                    "dur": 1,
                    "pid": 1,
                    "tid": 1,
                },
                {"name": "cudaGraphLaunch", "ph": "X"},
                {"name": "kernel", "cat": "kernel", "ph": "X"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(trace))
            result = self.benchmark._trace_summary(path)
        self.assertEqual(result["positive_allocation_events"], 1)
        self.assertEqual(result["allocated_bytes"], 32)
        self.assertEqual(result["freed_bytes"], 32)
        self.assertEqual(result["allocation_net_bytes"], 0)
        self.assertEqual(result["max_allocation_bytes"], 32)
        self.assertEqual(result["compiled_region_events"], 2)
        self.assertEqual(
            result["compiled_region_names"],
            {
                "Torch-Compiled Region: 0/0": 1,
                "Torch-Compiled Region: 1/0": 1,
            },
        )
        self.assertEqual(result["cuda_graph_launches"], 1)
        self.assertEqual(
            result["indexed_write_operations_outside_compiled_regions"], 1
        )
        self.assertEqual(
            result["indexed_write_names_outside_compiled_regions"],
            {"aten::index_add_": 1},
        )

    def test_trace_filename_separates_every_execution_contract(self):
        common = {
            "name": "cpu-crossover-2d",
            "device": "cpu",
            "precision": "float64",
            "compile_mode": "default",
            "capture_graphs": False,
            "execution_policy": "auto",
            "threads": 1,
            "interop_threads": 1,
        }
        baseline = self.benchmark._trace_filename(**common)
        variants = []
        for name, value in (
            ("precision", "float32"),
            ("execution_policy", "dense"),
            ("threads", 4),
            ("interop_threads", 2),
            ("capture_graphs", True),
        ):
            arguments = {**common, name: value}
            variants.append(self.benchmark._trace_filename(**arguments))
        self.assertEqual(len({baseline, *variants}), 1 + len(variants))

    def test_native_gate_rejects_a_mismatched_measurement_contract(self):
        spec = self.benchmark.find_case(self.manifest, "cpu-crossover-2d")
        reference = self.manifest["reference"]
        cpu_contract = self.benchmark._cpu_contract_environment()
        summary = {
            "observer_tag": reference["observer_tag"],
            "observer_commit": reference["observer_commit"],
            "physics_reference": reference["tag"],
            "environment": {
                "git_commit": reference["observer_commit"],
                "git_status": "",
                "hostname": platform.node(),
                "platform": platform.platform(),
                "cpu_count_physical": cpu_contract[
                    "cpu_count_physical_affinity"
                ],
                "cpu_topology": cpu_contract["cpu_topology"],
                "openmp_enabled": True,
            },
            "samples": [
                {
                    "workload": spec,
                    "threads": "1",
                    "openmp_threads": 1,
                    "measurements": {
                        "advance": {
                            "median_seconds": 0.1,
                            "raw_seconds": [0.1] * 15,
                            "steps_per_repeat": 100,
                            "repetitions": 15,
                        }
                    },
                }
            ],
        }
        candidate = {
            "workload": spec,
            "benchmark_contract": {
                "initializer": reference["field_initializer"],
                "seed": reference["seed"],
                "field_scale": reference["field_scale"],
                "warmup_steps": reference["performance_warmup_steps"],
                "steps_per_repeat": reference["performance_steps_per_repeat"],
                "repetitions": reference["performance_repetitions"],
                "profile_steps": reference["performance_profile_steps"],
                "timer": "time.perf_counter",
                "sample_start": "independently-restored-pre-warmup-state",
            },
            "runtime": {
                "device": "cpu",
                "precision": "float64",
                "threads": 1,
                "interop_threads": 1,
                **cpu_contract,
            },
            "measurements": {
                "advance": {
                    "raw_seconds": [0.1] * 15,
                    "median_seconds": 0.1,
                    "seconds_per_step": 0.001,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.json"
            path.write_text(json.dumps(summary))
            pinned_manifest = json.loads(json.dumps(self.manifest))
            pinned_manifest["reference"]["performance_summary_sha256"] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
            valid = self.benchmark._native_gate(
                path, "cpu-crossover-2d", 1, candidate, pinned_manifest
            )
            candidate["benchmark_contract"]["warmup_steps"] = 1
            invalid = self.benchmark._native_gate(
                path, "cpu-crossover-2d", 1, candidate, pinned_manifest
            )
            candidate["benchmark_contract"]["warmup_steps"] = reference[
                "performance_warmup_steps"
            ]
            candidate["measurements"]["advance"]["median_seconds"] = 0.2
            inconsistent = self.benchmark._native_gate(
                path, "cpu-crossover-2d", 1, candidate, pinned_manifest
            )
        self.assertTrue(valid["comparison_valid"])
        self.assertFalse(invalid["comparison_valid"])
        self.assertIn("contract", " ".join(invalid["contract_errors"]))
        self.assertFalse(inconsistent["comparison_valid"])
        self.assertIn("raw samples", " ".join(inconsistent["contract_errors"]))

    def test_native_gate_rejects_unpinned_embedded_contract(self):
        spec = self.benchmark.find_case(self.manifest, "cpu-crossover-2d")
        reference = self.manifest["reference"]
        cpu_contract = self.benchmark._cpu_contract_environment()
        summary = {
            "observer_tag": reference["observer_tag"],
            "observer_commit": reference["observer_commit"],
            "physics_reference": reference["tag"],
            "environment": {
                "git_commit": reference["observer_commit"],
                "git_status": "",
                "hostname": platform.node(),
                "platform": platform.platform(),
                "cpu_count_physical": cpu_contract[
                    "cpu_count_physical_affinity"
                ],
                "cpu_topology": cpu_contract["cpu_topology"],
                "openmp_enabled": True,
            },
            "benchmark_contract": {
                "initializer": reference["field_initializer"],
                "seed": reference["seed"],
                "field_scale": reference["field_scale"],
                "warmup_steps": reference["performance_warmup_steps"],
                "steps_per_repeat": reference["performance_steps_per_repeat"],
                "repetitions": reference["performance_repetitions"],
                "timer": "time.perf_counter",
                "sample_start": "independently-rebuilt-post-warmup-state",
            },
            "samples": [
                {
                    "workload": spec,
                    "threads": "1",
                    "openmp_threads": 1,
                    "measurements": {
                        "advance": {
                            "median_seconds": 0.1,
                            "raw_seconds": [0.1] * 15,
                            "steps_per_repeat": 100,
                            "repetitions": 15,
                        }
                    },
                }
            ],
        }
        candidate = {
            "workload": spec,
            "benchmark_contract": {
                "initializer": reference["field_initializer"],
                "seed": reference["seed"],
                "field_scale": reference["field_scale"],
                "warmup_steps": reference["performance_warmup_steps"],
                "steps_per_repeat": reference["performance_steps_per_repeat"],
                "repetitions": reference["performance_repetitions"],
                "profile_steps": reference["performance_profile_steps"],
                "timer": "time.perf_counter",
                "sample_start": "independently-restored-pre-warmup-state",
            },
            "runtime": {
                "device": "cpu",
                "precision": "float64",
                "threads": 1,
                "interop_threads": 1,
                **cpu_contract,
            },
            "measurements": {
                "advance": {
                    "raw_seconds": [0.1] * 15,
                    "median_seconds": 0.1,
                    "seconds_per_step": 0.001,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(summary))
            result = self.benchmark._native_gate(
                path, "cpu-crossover-2d", 1, candidate, self.manifest
            )
        self.assertFalse(result["comparison_valid"])
        self.assertIn("SHA-256", " ".join(result["contract_errors"]))

    def test_bootstrap_geomean_detects_significant_regression(self):
        statistics = dict(
            self.manifest["performance_gates"]["cpu_acceptance"]["statistics"]
        )
        statistics["resamples"] = 1000

        def gate(candidate):
            return {
                "comparison_valid": True,
                "reference_raw_seconds_per_step": [1.0] * 15,
                "candidate_raw_seconds_per_step": [candidate] * 15,
                "torch_to_native_ratio": candidate,
            }

        equal = self.benchmark._bootstrap_geomean_regression(
            [gate(1.0), gate(1.0)], statistics
        )
        slower = self.benchmark._bootstrap_geomean_regression(
            [gate(1.02), gate(1.02)], statistics
        )
        self.assertTrue(equal["evaluated"])
        self.assertFalse(equal["significant_regression"])
        self.assertTrue(equal["passed"])
        self.assertTrue(slower["significant_regression"])
        self.assertFalse(slower["passed"])

    def test_bootstrap_geomean_fails_closed_for_invalid_evidence(self):
        statistics = self.manifest["performance_gates"]["cpu_acceptance"][
            "statistics"
        ]
        result = self.benchmark._bootstrap_geomean_regression(
            [{"comparison_valid": False}], statistics
        )
        self.assertFalse(result["evaluated"])
        self.assertIsNone(result["geometric_mean_ratio"])
        self.assertFalse(result["passed"])

    def test_bootstrap_geomean_rejects_nonfinite_or_inconsistent_samples(self):
        statistics = dict(
            self.manifest["performance_gates"]["cpu_acceptance"]["statistics"]
        )
        statistics["resamples"] = 10

        def gate(candidate, ratio=1.0):
            return {
                "comparison_valid": True,
                "reference_raw_seconds_per_step": [1.0] * 15,
                "candidate_raw_seconds_per_step": [candidate] * 15,
                "torch_to_native_ratio": ratio,
            }

        for malformed in (
            gate(float("nan")),
            gate(0.0),
            gate(1.0, ratio=2.0),
            gate(True),
            gate(1.0, ratio=True),
        ):
            with self.subTest(malformed=malformed):
                result = self.benchmark._bootstrap_geomean_regression(
                    [malformed], statistics
                )
                self.assertFalse(result["evaluated"])
                self.assertFalse(result["passed"])

    def test_state_progress_requires_every_field_and_material_state(self):
        import torch

        names = ("ex", "ey", "ez", "hx", "hy", "hz")
        first = {name: torch.zeros(1) for name in names}
        first.update(
            {
                "pml_ex_0_state": torch.zeros(1),
                "bucket_ex_0_state": torch.zeros(1),
                "dm2_buckets.0.u": torch.zeros(1),
            }
        )
        source_only = {name: value.clone() for name, value in first.items()}
        source_only["ex"].fill_(1)
        incomplete = self.benchmark._state_change_summary(first, source_only)
        self.assertFalse(incomplete["all_fields_changed"])
        self.assertFalse(incomplete["pml_state_changed"])

        complete = {name: value.clone().fill_(1) for name, value in first.items()}
        result = self.benchmark._state_change_summary(first, complete)
        self.assertTrue(result["all_fields_changed"])
        self.assertTrue(result["pml_state_changed"])
        self.assertTrue(result["dispersive_state_changed"])
        self.assertTrue(result["dm2_state_changed"])

    def test_cpu_slice_aggregator_recomputes_and_binds_evidence(self):
        manifest = json.loads(json.dumps(self.manifest))
        manifest["performance_gates"]["cpu_acceptance"]["statistics"][
            "resamples"
        ] = 100
        cases = manifest["performance_gates"]["cpu_acceptance"]["cases"]
        evidence = {
            "evidence_contract_id": "torch-cpu-acceptance-v3",
            "cpu_contract_id": manifest["performance_gates"]["cpu_acceptance"][
                "contract_id"
            ],
            "manifest_sha256": "manifest",
            "runner_sha256": "runner",
            "solver_sha256": "solver",
            "solver_abi": "abi",
            "candidate_git_commit": "commit",
            "candidate_git_status": "",
        }
        gate = {
            "comparison_valid": True,
            "within_five_percent": True,
            "reference_raw_seconds_per_step": [1.0] * 15,
            "candidate_raw_seconds_per_step": [1.0] * 15,
            "torch_to_native_ratio": 1.0,
        }

        def artifact(threads):
            return {
                "schema_version": 3,
                "kind": "cpu-acceptance-thread-slice",
                "evidence": dict(evidence),
                "environment": {
                    "hostname": "host",
                    "platform": "platform",
                    "python": "3.14",
                    "torch": "2.13",
                    "cpu_affinity": [0, 1, 2, 3],
                    "cpu_count_physical_affinity": 4,
                    "cpu_topology": "topology",
                    "cpu_model": "model",
                },
                "cases": [
                    {
                        "workload": self.benchmark.find_case(manifest, name),
                        "runtime": {
                            "device": "cpu",
                            "precision": "float64",
                            "compile_policy": "compile",
                            "compile_mode": "default",
                            "explicit_cuda_graphs": False,
                            "execution_policy": "auto",
                            "threads": threads,
                            "interop_threads": 1,
                            "cpu_affinity": [0, 1, 2, 3],
                            "cpu_count_physical_affinity": 4,
                            "cpu_topology": "topology",
                        },
                        "acceptance": {
                            name: True
                            for name in self.benchmark.RUNTIME_ACCEPTANCE_KEYS
                        },
                        "native_gate": dict(gate),
                    }
                    for name in cases
                ],
            }

        one = artifact(1)
        physical = artifact(4)
        with patch.object(self.benchmark, "_native_gate", return_value=gate):
            result = self.benchmark._aggregate_cpu_slice_outputs(
                [one, physical],
                manifest,
                Path("native.json"),
                evidence,
            )
            self.assertTrue(result["suite_acceptance"]["passed"])
            del physical["cases"][0]["acceptance"]["state_progressed"]
            incomplete = self.benchmark._aggregate_cpu_slice_outputs(
                [one, physical],
                manifest,
                Path("native.json"),
                evidence,
            )
            self.assertFalse(incomplete["suite_acceptance"]["passed"])
            physical = artifact(4)
            physical["cases"][0]["acceptance"]["passed"] = "true"
            invalid = self.benchmark._aggregate_cpu_slice_outputs(
                [one, physical],
                manifest,
                Path("native.json"),
                evidence,
            )
        self.assertFalse(invalid["suite_acceptance"]["passed"])

    def test_cpu_slice_aggregator_rejects_mixed_candidate_provenance(self):
        output = {"schema_version": 3, "kind": "cpu-acceptance-thread-slice"}
        result = self.benchmark._evaluate_cpu_slice(
            output,
            self.manifest,
            expected_evidence={"candidate_git_status": ""},
        )
        self.assertFalse(result["passed"])
        self.assertIn("provenance", " ".join(result["errors"]))

    def test_policy_matrix_fails_closed_until_runtime_paths_are_distinct(self):
        args = SimpleNamespace(
            device="cpu",
            precision="float64",
            compile_mode="default",
            capture_graphs=False,
            threads=1,
            interop_threads=1,
            warmup=5,
            steps=100,
            repeats=15,
            profile_steps=1,
            trace_directory=Path("/tmp"),
        )
        sample = {
            "acceptance": {"passed": True},
            "measurements": {"advance": {"seconds_per_step": 1.0}},
        }
        with patch.object(self.benchmark, "run_case", return_value=sample):
            result = self.benchmark._policy_matrix(
                args, "cpu-crossover-2d", self.manifest
            )
        self.assertFalse(result["comparison_valid"])
        self.assertIsNone(result["auto_to_fastest_forced_ratio"])
        self.assertIsNone(result["within_ten_percent"])
        self.assertFalse(result["passed"])

    def test_enforced_cpu_gate_requires_a_native_comparison(self):
        import torch

        args = SimpleNamespace(
            case="cpu-crossover-2d",
            device="cpu",
            precision="float64",
            compile_mode="default",
            policy="auto",
            capture_graphs=False,
            threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            warmup=5,
            steps=100,
            repeats=15,
            profile_steps=1,
            trace_directory=Path("/tmp"),
            native_summary=None,
            output=None,
            enforce=True,
        )
        sample = {"acceptance": {"passed": True}}
        with (
            patch.object(
                self.benchmark,
                "_arguments",
                return_value=(args, self.manifest),
            ),
            patch.object(self.benchmark, "run_case", return_value=sample),
            patch.object(self.benchmark, "_environment", return_value={}),
            patch("builtins.print"),
        ):
            status = self.benchmark.main()
        self.assertEqual(status, 2)

    def test_single_cpu_case_cannot_claim_suite_acceptance(self):
        args = SimpleNamespace(
            case="cpu-crossover-2d",
            device="cpu",
            precision="float64",
            compile_mode="default",
            policy="auto",
            capture_graphs=False,
            threads=1,
            interop_threads=1,
            warmup=5,
            steps=100,
            repeats=15,
            profile_steps=5,
            trace_directory=Path("/tmp"),
            native_summary=Path("native.json"),
            cpu_slice_artifacts=None,
            output=None,
            enforce=True,
        )
        sample = {"acceptance": {"passed": True}}
        native_gate = {
            "comparison_valid": True,
            "within_five_percent": True,
            "torch_to_native_ratio": 1.0,
        }
        environment = {
            "cpu_count_physical_affinity": 4,
        }
        with (
            patch.object(
                self.benchmark,
                "_arguments",
                return_value=(args, self.manifest),
            ),
            patch.object(self.benchmark, "run_case", return_value=sample),
            patch.object(self.benchmark, "_native_gate", return_value=native_gate),
            patch.object(self.benchmark, "_environment", return_value=environment),
            patch("builtins.print") as rendered,
        ):
            status = self.benchmark.main()
        output = json.loads(rendered.call_args.args[0])
        self.assertTrue(output["diagnostic_acceptance"]["passed"])
        self.assertFalse(output["suite_acceptance"]["passed"])
        self.assertEqual(output["suite_acceptance"]["cpu_suite_status"], "diagnostic-only")
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
