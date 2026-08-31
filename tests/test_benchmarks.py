import hashlib
import importlib.util
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
        self.assertEqual(
            result["pml"]["traffic_representation"], "compact-full-curl-v1"
        )
        self.assertEqual(
            result["pml"]["indexed_target_cells"], result["pml"]["active_cells"]
        )
        self.assertEqual(result["pml"]["sparse_residual_axis_targets"], 0)
        self.assertEqual(
            result["pml"]["gather_scatter_scalar_values_per_step"],
            6 * result["pml"]["active_cells"],
        )
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

    def test_compiled_cpu_pml_plan_reports_sparse_residual_traffic(self):
        _, _, _, plans, _ = self.benchmark.build_host_plan(
            "pml-thin",
            policy="auto",
            precision="float64",
            device_type="cpu",
            tile_size=64,
            gmes=self.gmes,
            compile_policy="compile",
        )
        summary = self.benchmark.plan_summary(plans)
        self.assertTrue(summary["cpml_sparse_residual"])
        cpml_buckets = [
            bucket
            for plan in plans
            for bucket in plan.buckets
            if bucket.signature.model == "cpml"
        ]
        logical_targets = sum(bucket.target_count for bucket in cpml_buckets)
        active_axes = sum(
            bool(len(axis.targets))
            for bucket in cpml_buckets
            for axis in bucket.cpml_residual_axes
        )
        traffic = self.benchmark._pml_traffic_summary(
            plans,
            scalar_width=1,
            element_size=8,
        )
        self.assertEqual(traffic["traffic_representation"], "axis-sparse-residual-v1")
        self.assertEqual(traffic["indexed_target_cells"], 0)
        self.assertGreater(traffic["sparse_residual_axis_targets"], 0)
        self.assertLess(traffic["sparse_residual_axis_targets"], 2 * logical_targets)
        self.assertEqual(
            traffic["gather_scatter_scalar_values_per_step"],
            4 * traffic["sparse_residual_axis_targets"],
        )
        self.assertLess(
            traffic["gather_scatter_bytes_per_step"],
            6 * logical_targets * 8,
        )
        self.assertEqual(summary["material_launches_per_step"], 6 + active_axes)
        self.assertEqual(
            summary["material_launches_per_step"],
            sum(item["launches"] for item in summary["decisions"]),
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

    def test_current_rss_reads_linux_proc_resident_pages(self):
        with (
            patch.object(self.benchmark.platform, "system", return_value="Linux"),
            patch.object(
                self.benchmark.Path,
                "read_text",
                return_value="1000 42 7 0 0 0 0\n",
            ),
            patch.object(self.benchmark.os, "sysconf", return_value=4096),
        ):
            self.assertEqual(self.benchmark._current_rss_bytes(), 42 * 4096)

    def test_current_rss_reads_macos_direct_provider(self):
        with (
            patch.object(self.benchmark.platform, "system", return_value="Darwin"),
            patch.object(
                self.benchmark,
                "_darwin_proc_pid_rusage_bytes",
                return_value=12_641_280,
            ) as direct,
        ):
            self.assertEqual(self.benchmark._current_rss_bytes(), 12_641_280)
        direct.assert_called_once_with()

    def test_ps_rss_reference_converts_kibibytes_to_bytes(self):
        completed = SimpleNamespace(returncode=0, stdout=" 12345\n")
        with (
            patch.object(self.benchmark.os, "getpid", return_value=321),
            patch.object(
                self.benchmark.subprocess, "run", return_value=completed
            ) as run,
        ):
            self.assertEqual(self.benchmark._ps_current_rss_bytes(), 12345 * 1024)
        run.assert_called_once_with(
            ("ps", "-o", "rss=", "-p", "321"),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_current_rss_fails_closed_when_measurement_is_unavailable(self):
        with patch.object(self.benchmark.platform, "system", return_value="Windows"):
            self.assertIsNone(self.benchmark._current_rss_bytes())
        with (
            patch.object(self.benchmark.platform, "system", return_value="Linux"),
            patch.object(self.benchmark.Path, "read_text", side_effect=OSError),
        ):
            self.assertIsNone(self.benchmark._current_rss_bytes())

    def test_cpu_memory_probe_measures_growth_and_restores_warm_checkpoint(self):
        import torch

        class Simulation:
            device = torch.device("cpu")

            def __init__(self):
                self.checkpoints = []
                self.advances = []

            def load_checkpoint(self, checkpoint):
                self.checkpoints.append(checkpoint)

            def advance(self, steps):
                self.advances.append(steps)

        simulation = Simulation()
        with patch.object(
            self.benchmark,
            "_current_rss_bytes",
            side_effect=[10_000, 10_512],
        ):
            result = self.benchmark._cpu_memory_probe(simulation, "warm", 0)
        self.assertEqual(
            result,
            {
                "probe_steps": 1,
                "before_bytes": 10_000,
                "after_bytes": 10_512,
                "growth_bytes": 512,
            },
        )
        self.assertEqual(simulation.advances, [1])
        self.assertEqual(simulation.checkpoints, ["warm", "warm"])

    def test_cpu_memory_probe_records_missing_measurement(self):
        import torch

        simulation = SimpleNamespace(
            device=torch.device("cpu"),
            load_checkpoint=lambda checkpoint: None,
            advance=lambda steps: None,
        )
        with patch.object(
            self.benchmark,
            "_current_rss_bytes",
            side_effect=[None, 10_512],
        ):
            result = self.benchmark._cpu_memory_probe(simulation, {}, 2)
        self.assertIsNone(result["growth_bytes"])

    def test_cpu_memory_gate_requires_a_real_bounded_measurement(self):
        cpu = self.benchmark.torch.device("cpu")
        self.assertFalse(self.benchmark._memory_growth_bounded(cpu, None, None))
        self.assertTrue(self.benchmark._memory_growth_bounded(cpu, None, 1024**2))
        self.assertFalse(self.benchmark._memory_growth_bounded(cpu, None, 1024**2 + 1))

    def test_cpu_rss_plateau_accepts_stabilization_and_bounded_oscillation(self):
        samples = [
            {"before_bytes": value, "after_bytes": value}
            for value in (
                tuple(range(0, 16_000, 1_000))
                + (20_000, 20_100, 20_000, 20_200, 20_050, 20_100)
                + (20_100, 20_000, 20_150, 20_050, 20_100, 20_000)
            )
        ]
        result = self.benchmark._evaluate_cpu_rss_plateau(samples)
        self.assertTrue(result["bounded"])
        self.assertFalse(result["persistent_positive_order_trend"])
        self.assertLessEqual(result["stable_start_upward_excursion_bytes"], 1024**2)
        self.assertEqual(result["stabilization_boundary_index"], 16)
        self.assertEqual(len(result["evaluation_block_slopes_bytes_per_window"]), 2)
        self.assertEqual(result["peak_rss_bytes"], 20_200)

    def test_cpu_rss_plateau_accepts_falling_evaluation_windows(self):
        stable = tuple(range(20_000_000, 18_800_000, -100_000))
        result = self.benchmark._evaluate_cpu_rss_plateau(
            [
                {"before_bytes": value, "after_bytes": value}
                for value in tuple(range(16)) + stable
            ]
        )
        self.assertTrue(result["bounded"])
        self.assertFalse(result["persistent_positive_order_trend"])
        self.assertEqual(result["stable_start_upward_excursion_bytes"], 0)
        self.assertTrue(
            all(
                slope < 0
                for slope in result["evaluation_block_slopes_bytes_per_window"]
            )
        )

    def test_cpu_rss_plateau_rejects_persistent_growth(self):
        values = tuple(range(16)) + tuple(
            10_000 + 16_384 * index for index in range(12)
        )
        result = self.benchmark._evaluate_cpu_rss_plateau(
            [{"before_bytes": value, "after_bytes": value} for value in values]
        )
        self.assertFalse(result["bounded"])
        self.assertTrue(result["persistent_positive_order_trend"])
        self.assertLessEqual(result["stable_start_upward_excursion_bytes"], 1024**2)

    def test_cpu_rss_plateau_rejects_large_excursion_without_trend(self):
        stable = (
            10_000,
            10_000 + 1024**2 + 1,
            10_000,
            10_000,
            10_000,
            10_000,
        ) * 2
        result = self.benchmark._evaluate_cpu_rss_plateau(
            [
                {"before_bytes": value, "after_bytes": value}
                for value in tuple(range(16)) + stable
            ]
        )
        self.assertFalse(result["bounded"])
        self.assertFalse(result["persistent_positive_order_trend"])

    def test_cpu_rss_plateau_fails_closed_for_unavailable_evidence(self):
        self.assertFalse(self.benchmark._evaluate_cpu_rss_plateau([])["bounded"])
        malformed = [{"before_bytes": 1, "after_bytes": 1} for _ in range(28)]
        malformed[-1]["after_bytes"] = None
        result = self.benchmark._evaluate_cpu_rss_plateau(malformed)
        self.assertFalse(result["bounded"])
        self.assertIn("unavailable", result["error"])

    def test_darwin_rss_provider_validates_direct_bytes_against_ps(self):
        with (
            patch.object(self.benchmark.platform, "system", return_value="Darwin"),
            patch.object(
                self.benchmark,
                "_darwin_proc_pid_rusage_bytes",
                side_effect=[100_000_000, 100_200_000, 100_300_000],
            ),
            patch.object(
                self.benchmark,
                "_ps_current_rss_bytes",
                return_value=100_100_000,
            ),
        ):
            reader, provider = self.benchmark._current_rss_provider()
            self.assertEqual(reader(), 100_300_000)
        self.assertEqual(provider["name"], "proc-pid-rusage-v0")
        self.assertEqual(provider["units"], "bytes")
        self.assertTrue(provider["validated"])

    def test_darwin_rss_provider_fails_closed_on_unit_mismatch(self):
        with (
            patch.object(self.benchmark.platform, "system", return_value="Darwin"),
            patch.object(
                self.benchmark,
                "_darwin_proc_pid_rusage_bytes",
                side_effect=[100_000_000, 100_200_000],
            ),
            patch.object(
                self.benchmark,
                "_ps_current_rss_bytes",
                return_value=100_000,
            ),
        ):
            reader, provider = self.benchmark._current_rss_provider()
            self.assertIsNone(reader())
        self.assertFalse(provider["validated"])
        self.assertEqual(
            provider["validation"]["reference_bytes"],
            100_000,
        )

    def test_fresh_cpu_rss_probe_binds_request_and_child_pid(self):
        request = self.benchmark._cpu_rss_request(
            "cpu-crossover-2d",
            precision="float64",
            compile_mode="default",
            execution_policy="auto",
            experimental_dispersive_grouping=True,
            experimental_dispersive_grouping_scope="two-level",
            threads=1,
            interop_threads=1,
            warmup=5,
            profile_steps=5,
        )
        payload = {
            "schema_version": 1,
            "kind": "cpu-rss-fresh-process",
            "pid": 101,
            "parent_pid": 100,
            "request": request,
            "evidence": {"candidate": "same"},
            "plateau": {"schema_version": 2, "bounded": True},
        }
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )
        with (
            patch.object(self.benchmark.os, "getpid", return_value=100),
            patch.object(
                self.benchmark,
                "_current_evidence",
                return_value={"candidate": "same"},
            ),
            patch.object(
                self.benchmark.subprocess, "run", return_value=completed
            ) as run,
        ):
            result = self.benchmark._fresh_cpu_memory_probe(request, self.manifest)
        self.assertTrue(result["plateau"]["bounded"])
        command = run.call_args.args[0]
        self.assertIn("--cpu-rss-child", command)
        self.assertIn("two-level", command)

        payload["parent_pid"] = 99
        completed.stdout = json.dumps(payload)
        with (
            patch.object(self.benchmark.os, "getpid", return_value=100),
            patch.object(
                self.benchmark,
                "_current_evidence",
                return_value={"candidate": "same"},
            ),
            patch.object(self.benchmark.subprocess, "run", return_value=completed),
        ):
            invalid = self.benchmark._fresh_cpu_memory_probe(request, self.manifest)
        self.assertFalse(invalid["plateau"]["bounded"])
        self.assertIn("binding", invalid["binding_error"])

    def test_cuda_memory_gate_preserves_allocated_growth_behavior(self):
        cuda = self.benchmark.torch.device("cuda")
        self.assertTrue(self.benchmark._memory_growth_bounded(cuda, None, None))
        self.assertTrue(self.benchmark._memory_growth_bounded(cuda, 1024**2, None))
        self.assertFalse(self.benchmark._memory_growth_bounded(cuda, 1024**2 + 1, None))

    def test_trace_summary_counts_raw_allocator_events(self):
        trace = {
            "traceEvents": [
                {"name": "[memory]", "args": {"Bytes": 32, "Total Allocated": 1032}},
                {"name": "[memory]", "args": {"Bytes": 16, "Total Allocated": 1048}},
                {"name": "[memory]", "args": {"Bytes": 32, "Total Allocated": 1080}},
                {"name": "[memory]", "args": {"Bytes": -80, "Total Allocated": 1000}},
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
                {
                    "name": "aten::index_put_",
                    "ph": "X",
                    "ts": 13,
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
            trace_bytes = path.read_bytes()
            result = self.benchmark._trace_summary(path)
        self.assertEqual(result["chrome_trace_size_bytes"], len(trace_bytes))
        self.assertEqual(
            result["chrome_trace_sha256"], hashlib.sha256(trace_bytes).hexdigest()
        )
        self.assertEqual(result["positive_allocation_events"], 3)
        self.assertEqual(result["allocated_bytes"], 80)
        self.assertEqual(result["freed_bytes"], 80)
        self.assertEqual(result["allocation_net_bytes"], 0)
        self.assertEqual(result["max_allocation_bytes"], 32)
        self.assertEqual(result["allocation_size_histogram"], {"16": 1, "32": 2})
        self.assertEqual(result["live_allocation_baseline_bytes"], 1000)
        self.assertEqual(result["peak_live_allocated_bytes"], 1080)
        self.assertEqual(result["final_live_allocated_bytes"], 1000)
        self.assertEqual(result["live_allocation_growth_bytes"], 0)
        self.assertTrue(result["live_allocation_metrics_complete"])
        self.assertEqual(result["compiled_region_events"], 2)
        self.assertEqual(
            result["compiled_region_names"],
            {
                "Torch-Compiled Region: 0/0": 1,
                "Torch-Compiled Region: 1/0": 1,
            },
        )
        self.assertEqual(result["cuda_graph_launches"], 1)
        self.assertEqual(result["indexed_write_operations_outside_compiled_regions"], 2)
        self.assertEqual(
            result["indexed_write_names_outside_compiled_regions"],
            {"aten::index_add_": 1, "aten::index_put_": 1},
        )
        self.assertEqual(
            result["policy_write_operations"],
            {
                "aten::masked_scatter_": 0,
                "aten::index_copy_": 1,
                "aten::scatter_": 0,
            },
        )

    def test_trace_summary_reports_complete_zero_live_metrics_without_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": []}))
            result = self.benchmark._trace_summary(path)

        self.assertEqual(result["live_allocation_baseline_bytes"], 0)
        self.assertEqual(result["peak_live_allocated_bytes"], 0)
        self.assertEqual(result["final_live_allocated_bytes"], 0)
        self.assertEqual(result["live_allocation_growth_bytes"], 0)
        self.assertTrue(result["live_allocation_metrics_complete"])

    def test_trace_summary_marks_missing_total_allocated_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(
                json.dumps(
                    {"traceEvents": [{"name": "[memory]", "args": {"Bytes": 16}}]}
                )
            )
            result = self.benchmark._trace_summary(path)

        self.assertFalse(result["live_allocation_metrics_complete"])
        self.assertIsNone(result["live_allocation_growth_bytes"])

    def test_trace_summary_rejects_discontinuous_live_allocation_totals(self):
        trace = {
            "traceEvents": [
                {
                    "name": "[memory]",
                    "args": {"Bytes": 16, "Total Allocated": 1016},
                },
                {
                    "name": "[memory]",
                    "args": {"Bytes": 16, "Total Allocated": 1020},
                },
                {
                    "name": "[memory]",
                    "args": {"Bytes": -20, "Total Allocated": 1000},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(trace))
            result = self.benchmark._trace_summary(path)

        self.assertFalse(result["live_allocation_metrics_complete"])
        self.assertIsNone(result["live_allocation_baseline_bytes"])
        self.assertIsNone(result["final_live_allocated_bytes"])

    def test_trace_summary_rejects_negative_live_allocation_values(self):
        trace = {
            "traceEvents": [
                {
                    "name": "[memory]",
                    "args": {"Bytes": 16, "Total Allocated": 8},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(trace))
            result = self.benchmark._trace_summary(path)

        self.assertFalse(result["live_allocation_metrics_complete"])
        self.assertIsNone(result["peak_live_allocated_bytes"])

    def _allocation_profiler(self):
        return {
            "positive_allocation_events": 15,
            "allocated_bytes": 320,
            "freed_bytes": 320,
            "allocation_net_bytes": 0,
            "allocation_size_histogram": {"16": 10, "32": 5},
            "profile_steps": 5,
            "positive_allocation_operations": 3,
            "live_allocation_baseline_bytes": 4096,
            "peak_live_allocated_bytes": 4128,
            "final_live_allocated_bytes": 4096,
            "live_allocation_growth_bytes": 0,
            "live_allocation_metrics_complete": True,
            "chrome_trace_sha256": "trace-sha256",
            "field_buffer_sizes_bytes": {
                "state.Ex": 6400,
                "state.Ey": 6528,
                "state.Ez": 6656,
                "state.Hx": 6784,
                "state.Hy": 6912,
                "state.Hz": 7040,
            },
            "fixed_boundary_buffer_sizes_bytes": {},
        }

    def _allocation_provenance(self, source_path):
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        return {
            "method": self.benchmark.ALLOCATION_PROVENANCE_METHOD,
            "reviewed": True,
            "trace_sha256": "trace-sha256",
            "compile_cache_key": "compile-key",
            "profile_steps": 5,
            "allocation_size_histogram": {"16": 10, "32": 5},
            "fixed_boundary_buffer_sizes_bytes": {},
            "full_field_or_domain_clone_events": 0,
            "upstream_issue_urls": ["https://github.com/pytorch/pytorch/issues/195330"],
            "allocations": [
                {
                    "size_bytes": 16,
                    "events_per_step": 2,
                    "classification": "allowed-plan-bounded-temporary",
                    "generated_operation": "allocate buf0 for indexed update",
                },
                {
                    "size_bytes": 32,
                    "events_per_step": 1,
                    "classification": "allowed-plan-bounded-temporary",
                    "generated_operation": "allocate buf1 for coefficient gather",
                },
            ],
            "generated_sources": [{"path": str(source_path), "sha256": source_sha256}],
        }

    def _allocation_contract(self, profiler, provenance=None):
        return self.benchmark._fixed_temporary_allocation_contract(
            self.benchmark.torch.device("cpu"),
            profiler,
            compile_cache_key="compile-key",
            allocation_provenance=provenance,
        )

    def test_cpu_zero_allocation_passes_without_provenance(self):
        profiler = {
            "positive_allocation_events": 0,
            "allocated_bytes": 0,
            "freed_bytes": 0,
            "allocation_net_bytes": 0,
            "allocation_size_histogram": {},
            "profile_steps": 5,
            "positive_allocation_operations": 0,
            "live_allocation_baseline_bytes": 0,
            "peak_live_allocated_bytes": 0,
            "final_live_allocated_bytes": 0,
            "live_allocation_growth_bytes": 0,
            "live_allocation_metrics_complete": True,
        }
        contract = self._allocation_contract(profiler)
        self.assertTrue(contract["satisfied"])
        self.assertEqual(contract["status"], "zero-allocation")

    def test_cpu_nonzero_allocation_passes_with_bound_reviewed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "generated.cpp"
            source_path.write_text("// generated indexed-update kernel\n")
            contract = self._allocation_contract(
                self._allocation_profiler(),
                self._allocation_provenance(source_path),
            )
        self.assertTrue(contract["satisfied"])
        self.assertEqual(contract["status"], "reviewed-fixed-temporary")
        self.assertTrue(contract["checks"]["generated_sources_verified"])
        self.assertEqual(len(contract["verified_generated_sources"]), 1)

    def test_cpu_compiled_allocation_does_not_require_op_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "generated.cpp"
            source_path.write_text("// generated indexed-update kernel\n")
            profiler = self._allocation_profiler()
            profiler["positive_allocation_operations"] = 0
            contract = self._allocation_contract(
                profiler, self._allocation_provenance(source_path)
            )
        self.assertTrue(contract["checks"]["trace_allocation_integrity"])
        self.assertTrue(contract["satisfied"])

    def test_cpu_nonzero_allocation_requires_canonical_public_issue_urls(self):
        valid_url = "https://github.com/pytorch/pytorch/issues/195330"
        invalid_values = (
            None,
            [],
            [valid_url, valid_url],
            ["https://github.com/pytorch/pytorch/issues/195330?query=1"],
            ["https://github.com/pytorch/pytorch/pull/195330"],
            ["https://github.com/pytorch/pytorch/issues/0"],
            [{"issue": 195330}],
        )
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "generated.cpp"
            source_path.write_text("// generated indexed-update kernel\n")
            for value in invalid_values:
                with self.subTest(value=value):
                    provenance = self._allocation_provenance(source_path)
                    if value is None:
                        del provenance["upstream_issue_urls"]
                    else:
                        provenance["upstream_issue_urls"] = value
                    contract = self._allocation_contract(
                        self._allocation_profiler(), provenance
                    )
                    self.assertFalse(contract["satisfied"])
                    self.assertFalse(contract["checks"]["public_upstream_issues_valid"])

    def test_cpu_allocation_rejects_full_field_or_domain_clones(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "generated.cpp"
            source_path.write_text("// generated indexed-update kernel\n")
            profiler = self._allocation_profiler()
            profiler["field_buffer_sizes_bytes"]["state.Ex"] = 32
            field_sized = self._allocation_contract(
                profiler, self._allocation_provenance(source_path)
            )
            self.assertFalse(field_sized["satisfied"])
            self.assertFalse(field_sized["checks"]["no_field_buffer_sized_allocations"])
            provenance = self._allocation_provenance(source_path)
            provenance["full_field_or_domain_clone_events"] = 1
            clone = self._allocation_contract(self._allocation_profiler(), provenance)
            self.assertFalse(clone["satisfied"])
            self.assertFalse(clone["checks"]["full_field_or_domain_clone_events_zero"])

    def test_cpu_allocation_rejects_field_size_despite_fixed_scratch_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "generated.cpp"
            source_path.write_text("// generated indexed-update kernel\n")
            profiler = self._allocation_profiler()
            profiler["field_buffer_sizes_bytes"]["state.Ex"] = 32
            fixed_buffers = {"state._boundary_ey_2": 32}
            profiler["fixed_boundary_buffer_sizes_bytes"] = fixed_buffers
            provenance = self._allocation_provenance(source_path)
            provenance["fixed_boundary_buffer_sizes_bytes"] = fixed_buffers
            provenance["allocations"][1][
                "fixed_temporary_buffer"
            ] = "state._boundary_ey_2"
            contract = self._allocation_contract(profiler, provenance)
            self.assertFalse(contract["satisfied"])
            self.assertFalse(contract["checks"]["no_field_buffer_sized_allocations"])
            self.assertFalse(contract["checks"]["provenance_allocations_valid"])

    def test_fixed_temporary_buffer_inventory_recurses_auxiliaries(self):
        torch = self.benchmark.torch
        state = torch.nn.Module()
        state.register_buffer("ex", torch.zeros(4, dtype=torch.float64))
        state.register_buffer(
            "_boundary_ey_2", torch.zeros(3, dtype=torch.float64), persistent=False
        )
        state.register_buffer("_scratch_ex", torch.zeros(11), persistent=False)
        sources = torch.nn.Module()
        sources.register_buffer("_values", torch.zeros(5), persistent=False)
        probes = torch.nn.Module()
        probes.register_buffer("samples", torch.zeros(2))
        auxiliary_state = torch.nn.Module()
        auxiliary_state.register_buffer("_scratch_ex", torch.zeros(7), persistent=False)
        auxiliary_state.register_buffer(
            "_boundary_hx_1", torch.zeros(9), persistent=False
        )
        auxiliary_sources = torch.nn.Module()
        auxiliary_sources.auxiliaries = ()
        auxiliary = SimpleNamespace(
            state=auxiliary_state,
            sources=auxiliary_sources,
            probes=torch.nn.Module(),
        )
        sources.auxiliaries = (auxiliary,)
        simulation = SimpleNamespace(state=state, sources=sources, probes=probes)

        result = self.benchmark._fixed_boundary_buffer_sizes_bytes(simulation)

        self.assertEqual(
            result,
            {
                "sources.auxiliaries[0].state._boundary_hx_1": 36,
                "state._boundary_ey_2": 24,
            },
        )

    def test_cpu_allocation_rejects_imbalance_and_final_live_growth(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "generated.cpp"
            source_path.write_text("// generated indexed-update kernel\n")
            profiler = self._allocation_profiler()
            profiler["freed_bytes"] = 319
            profiler["allocation_net_bytes"] = 1
            imbalance = self._allocation_contract(
                profiler, self._allocation_provenance(source_path)
            )
            self.assertFalse(imbalance["satisfied"])
            self.assertFalse(imbalance["checks"]["allocation_bytes_balanced"])
            profiler = self._allocation_profiler()
            profiler["final_live_allocated_bytes"] = 4097
            profiler["live_allocation_growth_bytes"] = 1
            growth = self._allocation_contract(
                profiler, self._allocation_provenance(source_path)
            )
            self.assertFalse(growth["satisfied"])
            self.assertFalse(growth["checks"]["live_allocation_growth_zero"])

    def test_cpu_allocation_missing_or_unaccounted_provenance_fails_closed(self):
        missing = self._allocation_contract(self._allocation_profiler())
        self.assertFalse(missing["satisfied"])
        self.assertFalse(missing["checks"]["reviewed_provenance_present"])
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "generated.cpp"
            source_path.write_text("// generated indexed-update kernel\n")
            provenance = self._allocation_provenance(source_path)
            provenance["allocations"][0]["events_per_step"] = 1
            unaccounted = self._allocation_contract(
                self._allocation_profiler(), provenance
            )
        self.assertFalse(unaccounted["satisfied"])
        self.assertFalse(
            unaccounted["checks"]["provenance_allocations_fully_accounted"]
        )

    def test_cpu_allocation_rejects_malformed_field_size_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "generated.cpp"
            source_path.write_text("// generated indexed-update kernel\n")
            profiler = self._allocation_profiler()
            profiler["field_buffer_sizes_bytes"] = None
            contract = self._allocation_contract(
                profiler, self._allocation_provenance(source_path)
            )
        self.assertFalse(contract["satisfied"])
        self.assertFalse(contract["checks"]["field_buffer_sizes_present"])

    def test_cpu_allocation_requires_generated_source_hash_match(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "generated.cpp"
            source_path.write_text("// generated indexed-update kernel\n")
            provenance = self._allocation_provenance(source_path)
            provenance["generated_sources"][0]["sha256"] = "0" * 64
            contract = self._allocation_contract(
                self._allocation_profiler(), provenance
            )
        self.assertFalse(contract["satisfied"])
        self.assertFalse(contract["checks"]["generated_sources_verified"])
        self.assertFalse(
            contract["verified_generated_sources"][0]["matches_provenance"]
        )

    def test_non_cpu_allocation_contract_is_not_applied(self):
        contract = self.benchmark._fixed_temporary_allocation_contract(
            self.benchmark.torch.device("cuda"),
            {},
            compile_cache_key="compile-key",
        )
        self.assertTrue(contract["satisfied"])
        self.assertFalse(contract["applied"])
        self.assertEqual(contract["status"], "not-applied")

    def test_allocation_provenance_loader_selects_one_exact_record(self):
        record = {
            "workload": "cpu-crossover-2d",
            "device": "cpu",
            "precision": "float64",
            "compile_mode": "default",
            "execution_policy": "auto",
            "threads": 1,
            "method": self.benchmark.ALLOCATION_PROVENANCE_METHOD,
        }
        document = {
            "schema_version": 1,
            "kind": self.benchmark.ALLOCATION_PROVENANCE_KIND,
            "method": self.benchmark.ALLOCATION_PROVENANCE_METHOD,
            "records": [record],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocation.json"
            path.write_text(json.dumps(document))
            loaded = self.benchmark._load_allocation_provenance(path)
            selected = self.benchmark._select_allocation_provenance(
                loaded,
                workload="cpu-crossover-2d",
                device="cpu",
                precision="float64",
                compile_mode="default",
                execution_policy="auto",
                threads=1,
            )
            self.assertEqual(
                loaded["source_artifact"]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            document["records"].append(dict(record))
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "duplicate selector"):
                self.benchmark._load_allocation_provenance(path)
        self.assertEqual(selected, record)

    def test_saved_slice_allocation_can_be_reviewed_without_rerunning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            live = 1000
            trace_events = []
            for _step in range(5):
                live += 16
                trace_events.append(
                    {
                        "name": "[memory]",
                        "args": {"Bytes": 16, "Total Allocated": live},
                    }
                )
                live -= 16
                trace_events.append(
                    {
                        "name": "[memory]",
                        "args": {"Bytes": -16, "Total Allocated": live},
                    }
                )
            trace_path.write_text(json.dumps({"traceEvents": trace_events}))
            profiler = self.benchmark._trace_summary(trace_path)
            profiler.update(
                {
                    "profile_steps": 5,
                    "positive_allocation_operations": 1,
                    "field_buffer_sizes_bytes": {"state.Ex": 6400},
                    "fixed_boundary_buffer_sizes_bytes": {},
                }
            )
            generated_source = root / "generated.cpp"
            generated_source.write_text("// generated bounded temporary\n")
            record = {
                "workload": "cpu-crossover-2d",
                "device": "cpu",
                "precision": "float64",
                "compile_mode": "default",
                "execution_policy": "auto",
                "threads": 1,
                "method": self.benchmark.ALLOCATION_PROVENANCE_METHOD,
                "reviewed": True,
                "trace_sha256": profiler["chrome_trace_sha256"],
                "compile_cache_key": "compile-key",
                "profile_steps": 5,
                "allocation_size_histogram": {"16": 5},
                "fixed_boundary_buffer_sizes_bytes": {},
                "full_field_or_domain_clone_events": 0,
                "upstream_issue_urls": [
                    "https://github.com/pytorch/pytorch/issues/195330"
                ],
                "allocations": [
                    {
                        "size_bytes": 16,
                        "events_per_step": 1,
                        "classification": "allowed-plan-bounded-temporary",
                        "generated_operation": "allocate bounded update temporary",
                    }
                ],
                "generated_sources": [
                    {
                        "path": str(generated_source),
                        "sha256": hashlib.sha256(
                            generated_source.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
            sidecar_path = root / "allocation.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": self.benchmark.ALLOCATION_PROVENANCE_KIND,
                        "method": self.benchmark.ALLOCATION_PROVENANCE_METHOD,
                        "records": [record],
                    }
                )
            )
            document = self.benchmark._load_allocation_provenance(sidecar_path)
            draft = self.benchmark._fixed_temporary_allocation_contract(
                self.benchmark.torch.device("cpu"),
                profiler,
                compile_cache_key="compile-key",
            )
            acceptance = {name: True for name in self.benchmark.RUNTIME_ACCEPTANCE_KEYS}
            acceptance["fixed_temporary_contract_satisfied"] = False
            acceptance["passed"] = False
            candidate = {
                "workload": {"name": "cpu-crossover-2d"},
                "runtime": {
                    "device": "cpu",
                    "precision": "float64",
                    "compile_mode": "default",
                    "execution_policy": "auto",
                    "threads": 1,
                    "compile_cache_key": "compile-key",
                },
                "profiler": profiler,
                "allocation_contract": draft,
                "acceptance": acceptance,
            }
            recomputed, errors = self.benchmark._recompute_cpu_allocation_contract(
                candidate,
                document,
                self.benchmark.ALLOCATION_PROVENANCE_METHOD,
            )
            self.assertEqual(errors, [])
            self.assertTrue(recomputed["satisfied"])
            self.assertEqual(recomputed["status"], "reviewed-fixed-temporary")
            missing, missing_errors = self.benchmark._recompute_cpu_allocation_contract(
                candidate,
                None,
                self.benchmark.ALLOCATION_PROVENANCE_METHOD,
            )
            self.assertFalse(missing["satisfied"])
            self.assertTrue(missing_errors)
            other_failure = json.loads(json.dumps(candidate))
            other_failure["acceptance"]["compiler_clean"] = False
            _result, other_errors = self.benchmark._recompute_cpu_allocation_contract(
                other_failure,
                document,
                self.benchmark.ALLOCATION_PROVENANCE_METHOD,
            )
            self.assertIn("non-allocation gate failure", " ".join(other_errors))
            trace_path.write_text(json.dumps({"traceEvents": []}))
            _result, trace_errors = self.benchmark._recompute_cpu_allocation_contract(
                candidate,
                document,
                self.benchmark.ALLOCATION_PROVENANCE_METHOD,
            )
            self.assertIn("does not match", " ".join(trace_errors))

    def test_source_and_compiled_region_contracts_recurse_auxiliaries(self):
        target = SimpleNamespace(numel=lambda: 1)
        batch = SimpleNamespace(additive_targets=target)
        auxiliary = SimpleNamespace(
            _electric_half=object(),
            _magnetic_half=object(),
            _fused_source_updates=False,
            sources=SimpleNamespace(batches=[batch], auxiliaries=[]),
        )
        simulation = SimpleNamespace(
            _electric_half=object(),
            _magnetic_half=object(),
            _fused_source_updates=True,
            sources=SimpleNamespace(batches=[batch], auxiliaries=[auxiliary]),
        )
        self.assertEqual(self.benchmark._compiled_local_simulation_count(simulation), 2)
        self.assertEqual(
            self.benchmark._source_index_operations_per_step(simulation),
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
            ("experimental_dispersive_grouping", True),
        ):
            arguments = {**common, name: value}
            variants.append(self.benchmark._trace_filename(**arguments))
        variants.append(
            self.benchmark._trace_filename(
                **common,
                experimental_dispersive_grouping=True,
                experimental_dispersive_grouping_scope="two-level",
            )
        )
        self.assertEqual(len({baseline, *variants}), 1 + len(variants))

    def test_native_gate_rejects_a_mismatched_measurement_contract(self):
        spec = self.benchmark.find_case(self.manifest, "cpu-crossover-2d")
        reference = self.manifest["reference"]
        cpu_contract = self.benchmark._cpu_contract_environment()
        summary = {
            "observer_tag": reference["performance_observer_tag"],
            "observer_commit": reference["performance_observer_commit"],
            "physics_reference": reference["tag"],
            "environment": {
                "git_commit": reference["performance_observer_commit"],
                "git_status": "",
                "hostname": platform.node(),
                "platform": platform.platform(),
                "cpu_count_physical": cpu_contract["cpu_count_physical_affinity"],
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
            pinned_manifest["reference"]["performance_summary_sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
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
            "observer_tag": reference["performance_observer_tag"],
            "observer_commit": reference["performance_observer_commit"],
            "physics_reference": reference["tag"],
            "environment": {
                "git_commit": reference["performance_observer_commit"],
                "git_status": "",
                "hostname": platform.node(),
                "platform": platform.platform(),
                "cpu_count_physical": cpu_contract["cpu_count_physical_affinity"],
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
                "candidate_to_torch_baseline_ratio": candidate,
            }

        equal = self.benchmark._bootstrap_geomean_regression(
            [gate(1.0), gate(1.0)],
            statistics,
            ratio_key="candidate_to_torch_baseline_ratio",
        )
        slower = self.benchmark._bootstrap_geomean_regression(
            [gate(1.02), gate(1.02)],
            statistics,
            ratio_key="candidate_to_torch_baseline_ratio",
        )
        self.assertTrue(equal["evaluated"])
        self.assertFalse(equal["significant_regression"])
        self.assertTrue(equal["passed"])
        self.assertTrue(slower["significant_regression"])
        self.assertFalse(slower["passed"])

    def test_bootstrap_geomean_fails_closed_for_invalid_evidence(self):
        statistics = self.manifest["performance_gates"]["cpu_acceptance"]["statistics"]
        result = self.benchmark._bootstrap_geomean_regression(
            [{"comparison_valid": False}],
            statistics,
            ratio_key="candidate_to_torch_baseline_ratio",
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
                "candidate_to_torch_baseline_ratio": ratio,
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
                    [malformed],
                    statistics,
                    ratio_key="candidate_to_torch_baseline_ratio",
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
        manifest["performance_gates"]["cpu_acceptance"]["statistics"]["resamples"] = 100
        cases = manifest["performance_gates"]["cpu_acceptance"]["cases"]
        evidence = {
            "evidence_contract_id": "torch-cpu-acceptance-v8",
            "cpu_contract_id": manifest["performance_gates"]["cpu_acceptance"][
                "contract_id"
            ],
            "manifest_sha256": "f" * 64,
            "runner_sha256": "runner",
            "solver_sha256": "solver",
            "solver_abi": "abi",
            "candidate_git_commit": "b" * 40,
            "candidate_git_status": "",
        }
        native_gate = {
            "comparison_role": "informational",
            "comparison_valid": True,
            "contract_errors": [],
            "reference_raw_seconds_per_step": [1.0] * 15,
            "candidate_raw_seconds_per_step": [50.0] * 15,
            "torch_to_native_ratio": 50.0,
        }
        baseline_gate = {
            "comparison_valid": True,
            "contract_errors": [],
            "reference_raw_seconds_per_step": [1.0] * 15,
            "candidate_raw_seconds_per_step": [1.0] * 15,
            "candidate_to_torch_baseline_ratio": 1.0,
            "within_five_percent": True,
        }

        def environment(threads=None):
            result = {
                "hostname": "host",
                "platform": "platform",
                "python": "3.14",
                "torch": "2.13",
                "cuda_runtime": None,
                "devices": [],
                "cpu_count": 4,
                "cpu_affinity": [0, 1, 2, 3],
                "cpu_count_physical_affinity": 4,
                "cpu_topology": "topology",
                "cpu_model": "model",
                "gpu_topology": None,
            }
            if threads is not None:
                result["thread_environment"] = {
                    name: str(threads)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                }
            return result

        zero_profiler = {
            "chrome_trace": "/tmp/trace.json",
            "chrome_trace_size_bytes": 1,
            "chrome_trace_sha256": "trace",
            "positive_allocation_events": 0,
            "allocated_bytes": 0,
            "freed_bytes": 0,
            "allocation_net_bytes": 0,
            "max_allocation_bytes": 0,
            "allocation_size_histogram": {},
            "profile_steps": 5,
            "positive_allocation_operations": 0,
            "live_allocation_baseline_bytes": 0,
            "peak_live_allocated_bytes": 0,
            "final_live_allocated_bytes": 0,
            "live_allocation_growth_bytes": 0,
            "live_allocation_metrics_complete": True,
            "compiled_region_events": 10,
            "compiled_region_names": {
                "torch-compiled region:electric": 5,
                "torch-compiled region:magnetic": 5,
            },
            "host_to_device_events": 0,
            "device_to_host_events": 0,
            "indexed_write_operations_outside_compiled_regions": 0,
            "indexed_write_names_outside_compiled_regions": {},
            "expected_source_indexed_write_names_outside_compiled_regions": {},
        }
        zero_contract = self.benchmark._fixed_temporary_allocation_contract(
            self.benchmark.torch.device("cpu"),
            zero_profiler,
            compile_cache_key="compile-key",
        )

        baseline_salt = "a" * 64
        torch_baseline = {
            "kind": "torch-cpu-baseline",
            "cpu_acceptance_contract_id": "cpu-acceptance-v2",
            "timing_reference": {"root_commit": "baseline"},
            "environment": {
                "hostname": "redacted",
                "host_identity": self.benchmark.privacy_preserving_host_identity(
                    environment(), salt=baseline_salt
                ),
            },
            "source_artifacts": [
                {
                    "publication_url": (
                        "https://github.com/ruddyscent/gmes/releases/download/"
                        f"issue-123-torch-cpu-baseline-v2/{threads}.json"
                    ),
                    "size_bytes": threads,
                    "sha256": str(threads) * 64,
                    "thread_mode": "one" if threads == 1 else "physical",
                    "threads": threads,
                    "thread_environment": environment(threads)["thread_environment"],
                    "root_commit": "baseline",
                }
                for threads in (1, 4)
            ],
        }

        def artifact(
            threads,
            embedded_native=native_gate,
            embedded_baseline=baseline_gate,
        ):
            benchmark_contract = {
                "initializer": self.benchmark.FIELD_INITIALIZER,
                "seed": manifest["reference"]["seed"],
                "field_scale": manifest["reference"]["field_scale"],
                "warmup_steps": 5,
                "steps_per_repeat": 100,
                "repetitions": 15,
                "profile_steps": 5,
                "timer": "time.perf_counter",
                "sample_start": "independently-restored-pre-warmup-state",
            }
            counters = {name: 0 for name in self.benchmark.COUNTER_FIELDS}
            addresses = {"state.ex": 4096, "state.ey": 8192}
            rss_samples = [
                {"before_bytes": 1000, "after_bytes": 1000}
                for _ in range(
                    self.benchmark.CPU_RSS_STABILIZATION_WINDOWS
                    + self.benchmark.CPU_RSS_EVALUATION_BLOCK_WINDOWS
                    * self.benchmark.CPU_RSS_EVALUATION_BLOCKS
                )
            ]
            plateau = self.benchmark._evaluate_cpu_rss_plateau(rss_samples)
            plateau["probe_steps_per_window"] = 5
            plateau["measurement_provider"] = {
                "name": "proc-self-statm",
                "units": "bytes",
                "validated": True,
            }

            def raw_case(name):
                workload = self.benchmark.find_case(manifest, name)
                requirements = self.benchmark._cpu_case_material_state_requirements(
                    workload
                )
                changed = set(self.benchmark.STATE_FIELD_NAMES)
                if requirements["pml"]:
                    changed.add("pml_ex_0_state")
                if requirements["dispersive"]:
                    changed.add("bucket_ex_0_state")
                if requirements["dm2"]:
                    changed.add("dm2_buckets.0.u")
                runtime = {
                    "device": "cpu",
                    "precision": "float64",
                    "compile_policy": "compile",
                    "compile_mode": "default",
                    "explicit_cuda_graphs": False,
                    "execution_policy": "auto",
                    "experimental_dispersive_grouping": False,
                    "experimental_dispersive_grouping_scope": "combined",
                    "threads": threads,
                    "interop_threads": 1,
                    "compile_cache_key": "compile-key",
                    "cpu_affinity": [0, 1, 2, 3],
                    "cpu_count_physical_affinity": 4,
                    "cpu_topology": "topology",
                }
                request = self.benchmark._cpu_rss_request(
                    name,
                    precision="float64",
                    compile_mode="default",
                    execution_policy="auto",
                    experimental_dispersive_grouping=False,
                    experimental_dispersive_grouping_scope="combined",
                    threads=threads,
                    interop_threads=1,
                    warmup=5,
                    profile_steps=5,
                )
                fresh_process = {
                    "schema_version": 1,
                    "kind": "cpu-rss-fresh-process",
                    "pid": 101,
                    "parent_pid": 100,
                    "request": request,
                    "evidence": dict(evidence),
                    "compile_cache_key": "compile-key",
                    "counter_growth": dict(counters),
                    "compiler_clean": True,
                    "storage_addresses_before": dict(addresses),
                    "storage_addresses_after": dict(addresses),
                    "storage_addresses_stable": True,
                    "plateau": json.loads(json.dumps(plateau)),
                }
                return {
                    "workload": workload,
                    "benchmark_contract": dict(benchmark_contract),
                    "runtime": runtime,
                    "compiler": {
                        "after_cold": dict(counters),
                        "after_warmup": dict(counters),
                        "after_steady": dict(counters),
                        "steady_state_delta": dict(counters),
                        "fullgraph_clean": True,
                    },
                    "memory": {
                        "peak_rss_bytes": 1000,
                        "cpu_rss_probe_steps": 5,
                        "cpu_rss_before_bytes": 1000,
                        "cpu_rss_after_bytes": 1000,
                        "cpu_rss_growth_bytes": 0,
                        "cpu_rss_fresh_process": fresh_process,
                        "cuda_allocated_before_bytes": None,
                        "cuda_allocated_after_bytes": None,
                        "cuda_allocated_growth_bytes": None,
                        "cuda_peak_allocated_bytes": None,
                        "cuda_peak_reserved_bytes": None,
                        "storage_addresses_before": dict(addresses),
                        "storage_addresses_after": dict(addresses),
                        "storage_addresses_stable": True,
                        "bounded": True,
                    },
                    "profiler": dict(zero_profiler),
                    "diagnostics": {
                        "sources": {
                            "execution_representation": (
                                self.benchmark.gmes.torch_fdtd.FUSED_SOURCE_REPRESENTATION
                            )
                        },
                        "pml": {"active_cells": 1 if requirements["pml"] else 0},
                        "dispersive": {
                            "active_cells": 1 if requirements["dispersive"] else 0
                        },
                        "dm2": [{}] if requirements["dm2"] else [],
                    },
                    "state_progress": {
                        "initial_checksum": 1.0,
                        "post_warmup_checksum": 2.0,
                        "post_one_step_checksum": 3.0,
                        "final_checksum": 4.0,
                        "changed_after_first_timed_step": True,
                        "one_step_count": 6,
                        "expected_one_step_count": 6,
                        "timed_step_count": 105,
                        "expected_timed_step_count": 105,
                        "profiler_step_count": 10,
                        "expected_profiler_step_count": 10,
                        "changed_buffers": sorted(changed),
                        "fields_changed": sorted(
                            changed & self.benchmark.STATE_FIELD_NAMES
                        ),
                        "all_fields_changed": True,
                        "pml_state_changed": requirements["pml"],
                        "dispersive_state_changed": requirements["dispersive"],
                        "dm2_state_changed": requirements["dm2"],
                    },
                    "acceptance": {
                        key: True for key in self.benchmark.RUNTIME_ACCEPTANCE_KEYS
                    },
                    "allocation_contract": json.loads(json.dumps(zero_contract)),
                    "native_gate": dict(embedded_native),
                    "torch_baseline_gate": dict(embedded_baseline),
                }

            return {
                "schema_version": 4,
                "kind": "cpu-acceptance-thread-slice",
                "evidence": dict(evidence),
                "environment": environment(threads),
                "torch_baseline": self.benchmark._torch_baseline_provenance(
                    torch_baseline
                ),
                "cases": [raw_case(name) for name in cases],
            }

        one = artifact(1)
        physical = artifact(4)
        physical["environment"]["cpu_model"] = "model\nCPU(s) scaling MHz: 87%"
        with (
            patch.object(self.benchmark, "_native_gate", return_value=native_gate),
            patch.object(
                self.benchmark,
                "compare_candidate_to_baseline",
                return_value=baseline_gate,
            ),
            patch.object(self.benchmark, "_profiler_trace_matches", return_value=True),
        ):
            result = self.benchmark._aggregate_cpu_slice_outputs(
                [one, physical],
                manifest,
                Path("native.json"),
                torch_baseline,
                None,
                evidence,
            )
            self.assertTrue(result["suite_acceptance"]["passed"])
            self.assertTrue(
                all(
                    "path" not in source
                    for source in result["torch_baseline"]["source_artifacts"]
                )
            )
            self.assertEqual(result["acceptance_scope"], "cpu-performance-only")
            self.assertFalse(result["issue_completion_satisfied"])
            self.assertEqual(
                result["issue_completion_blockers"],
                ["complete-field-and-persistent-state-correctness-not-bound"],
            )
            required_correctness = [
                case["name"]
                for group in ("correctness", "physical_checks")
                for case in manifest[group]
            ]
            descriptor_candidate = {
                key: evidence[key]
                for key in (
                    "candidate_git_commit",
                    "candidate_git_status",
                    "manifest_sha256",
                )
            }
            correctness_evidence = {
                "schema_version": 1,
                "kind": "torch-correctness-evidence-index",
                "contract_id": "complete-field-and-persistent-state-v1",
                "manifest_contract_sha256": hashlib.sha256(
                    json.dumps(
                        manifest,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
                "candidate_evidence": evidence,
                "runtime_mode": {
                    "device": "cpu",
                    "precision": "float64",
                    "graph_mode": "eager",
                    "compile_policy": "eager",
                    "compile_mode": "default",
                },
                "required_cases": required_correctness,
                "artifacts": [
                    {
                        "case": name,
                        "group": (
                            "correctness"
                            if any(
                                case["name"] == name for case in manifest["correctness"]
                            )
                            else "physical_checks"
                        ),
                        "reference": {
                            "path": f"references/{name}.npz",
                            "sha256": "a" * 64,
                            "size_bytes": 1,
                            "media_type": "application/x-npz",
                            "candidate_evidence": descriptor_candidate,
                        },
                        "reference_observer_commit": manifest["reference"][
                            "observer_commit"
                        ],
                        "candidate": {
                            "path": f"candidates/{name}.npz",
                            "sha256": "b" * 64,
                            "size_bytes": 1,
                            "media_type": "application/x-npz",
                            "candidate_evidence": descriptor_candidate,
                        },
                        "candidate_provenance": {
                            "commit": evidence["candidate_git_commit"],
                            "source_sha256": "c" * 64,
                            "controller_sha256": "d" * 64,
                        },
                        "comparison": {"passed": True, "failures": []},
                        "tolerance_results": [
                            {
                                "key": "step/0/time",
                                "dtype": "float64",
                                "scope": "exact/test",
                                "rtol": 0.0,
                                "atol": 0.0,
                                "max_abs_error": 0.0,
                            }
                        ],
                    }
                    for name in required_correctness
                ],
                "source_artifact": {
                    "path": "correctness-index.json",
                    "sha256": "e" * 64,
                    "size_bytes": 1,
                    "media_type": "application/json",
                    "candidate_evidence": descriptor_candidate,
                },
                "suite_acceptance": {
                    "correctness_case_count": len(manifest["correctness"]),
                    "physical_check_case_count": len(manifest["physical_checks"]),
                    "evaluated_case_count": len(required_correctness),
                    "complete_fields": True,
                    "persistent_state": True,
                    "source_and_auxiliary_state": True,
                    "physical_observables": True,
                    "passed": True,
                },
            }
            bound = self.benchmark._aggregate_cpu_slice_outputs(
                [artifact(1), artifact(4)],
                manifest,
                Path("native.json"),
                torch_baseline,
                None,
                evidence,
                correctness_evidence,
            )
            self.assertEqual(
                bound["acceptance_scope"], "cpu-performance-and-correctness"
            )
            self.assertTrue(bound["cpu_correctness_satisfied"])
            self.assertTrue(bound["suite_acceptance"]["correctness_evidence_bound"])
            self.assertFalse(bound["issue_completion_satisfied"])
            self.assertEqual(
                bound["issue_completion_blockers"],
                ["gpu-policy-macos-evidence-not-bound"],
            )
            tampered_correctness = json.loads(json.dumps(correctness_evidence))
            tampered_correctness["candidate_evidence"]["solver_abi"] = "tampered"
            unbound = self.benchmark._aggregate_cpu_slice_outputs(
                [artifact(1), artifact(4)],
                manifest,
                Path("native.json"),
                torch_baseline,
                None,
                evidence,
                tampered_correctness,
            )
            self.assertFalse(unbound["cpu_correctness_satisfied"])
            self.assertEqual(result["suite_acceptance"]["cpu_evaluated_cell_count"], 12)
            self.assertTrue(
                result["suite_acceptance"]["torch_baseline_geomean_statistics"][
                    "evaluated"
                ]
            )
            self.assertEqual(
                result["suite_acceptance"]["native_comparison_role"],
                "informational",
            )
            self.assertFalse(
                result["suite_acceptance"]["native_geomean_statistics"]["passed"]
            )
            tampered_baseline = artifact(4)
            tampered_baseline["torch_baseline"]["kind"] = "tampered"
            tampered_baseline_result = self.benchmark._aggregate_cpu_slice_outputs(
                [artifact(1), tampered_baseline],
                manifest,
                Path("native.json"),
                torch_baseline,
                None,
                evidence,
            )
            invalid_allocation = artifact(4)
            invalid_allocation["cases"][0]["allocation_contract"][
                "status"
            ] = "reviewed-fixed-temporary"
            invalid_allocation_result = self.benchmark._aggregate_cpu_slice_outputs(
                [artifact(1), invalid_allocation],
                manifest,
                Path("native.json"),
                torch_baseline,
                None,
                evidence,
            )
            del physical["cases"][0]["acceptance"]["state_progressed"]
            incomplete = self.benchmark._aggregate_cpu_slice_outputs(
                [one, physical],
                manifest,
                Path("native.json"),
                torch_baseline,
                None,
                evidence,
            )
            self.assertFalse(incomplete["suite_acceptance"]["passed"])
            physical = artifact(4)
            physical["cases"][0]["acceptance"]["passed"] = "true"
            invalid = self.benchmark._aggregate_cpu_slice_outputs(
                [one, physical],
                manifest,
                Path("native.json"),
                torch_baseline,
                None,
                evidence,
            )
            experimental = artifact(4)
            for case in experimental["cases"]:
                case["runtime"]["experimental_dispersive_grouping"] = True
            experimental_result = self.benchmark._aggregate_cpu_slice_outputs(
                [one, experimental],
                manifest,
                Path("native.json"),
                torch_baseline,
                None,
                evidence,
            )
            slow_baseline = {
                **baseline_gate,
                "candidate_raw_seconds_per_step": [1.06] * 15,
                "candidate_to_torch_baseline_ratio": 1.06,
                "within_five_percent": False,
            }
            slow_one = artifact(1, embedded_baseline=slow_baseline)
            slow_physical = artifact(4, embedded_baseline=slow_baseline)
            with patch.object(
                self.benchmark,
                "compare_candidate_to_baseline",
                return_value=slow_baseline,
            ):
                slow = self.benchmark._aggregate_cpu_slice_outputs(
                    [slow_one, slow_physical],
                    manifest,
                    Path("native.json"),
                    torch_baseline,
                    None,
                    evidence,
                )
            bootstrap_regression = {
                **baseline_gate,
                "candidate_raw_seconds_per_step": [1.02] * 15,
                "candidate_to_torch_baseline_ratio": 1.02,
            }
            with patch.object(
                self.benchmark,
                "compare_candidate_to_baseline",
                return_value=bootstrap_regression,
            ):
                bootstrap_failure = self.benchmark._aggregate_cpu_slice_outputs(
                    [
                        artifact(1, embedded_baseline=bootstrap_regression),
                        artifact(4, embedded_baseline=bootstrap_regression),
                    ],
                    manifest,
                    Path("native.json"),
                    torch_baseline,
                    None,
                    evidence,
                )
            invalid_native = {**native_gate, "comparison_valid": False}
            with patch.object(
                self.benchmark,
                "_native_gate",
                return_value=invalid_native,
            ):
                invalid_native_result = self.benchmark._aggregate_cpu_slice_outputs(
                    [
                        artifact(1, embedded_native=invalid_native),
                        artifact(4, embedded_native=invalid_native),
                    ],
                    manifest,
                    Path("native.json"),
                    torch_baseline,
                    None,
                    evidence,
                )
            thread_environment_mismatch = artifact(4)
            thread_environment_mismatch["environment"]["thread_environment"][
                "OPENBLAS_NUM_THREADS"
            ] = "1"
            thread_environment_result = self.benchmark._aggregate_cpu_slice_outputs(
                [artifact(1), thread_environment_mismatch],
                manifest,
                Path("native.json"),
                torch_baseline,
                None,
                evidence,
            )
            raw_evidence_mutations = {
                "compiler graph break": lambda case: case["compiler"][
                    "after_steady"
                ].__setitem__("graph_breaks", 1),
                "compiled region count": lambda case: case["profiler"].__setitem__(
                    "compiled_region_events", 9
                ),
                "steady transfer": lambda case: case["profiler"].__setitem__(
                    "host_to_device_events", 1
                ),
                "external indexed write": lambda case: (
                    case["profiler"].__setitem__(
                        "indexed_write_operations_outside_compiled_regions", 1
                    ),
                    case["profiler"].__setitem__(
                        "indexed_write_names_outside_compiled_regions",
                        {"aten::index_add_": 1},
                    ),
                ),
                "parent storage address": lambda case: case["memory"][
                    "storage_addresses_after"
                ].__setitem__("state.ex", 4097),
                "child RSS raw window": lambda case: case["memory"][
                    "cpu_rss_fresh_process"
                ]["plateau"]["after_bytes"].__setitem__(-1, 2 * 1024 * 1024),
                "child compile cache": lambda case: case["memory"][
                    "cpu_rss_fresh_process"
                ].__setitem__("compile_cache_key", "tampered"),
                "child RSS request": lambda case: case["memory"][
                    "cpu_rss_fresh_process"
                ]["request"].__setitem__("threads", 2),
                "child RSS checkout evidence": lambda case: case["memory"][
                    "cpu_rss_fresh_process"
                ]["evidence"].__setitem__("candidate_git_commit", "tampered"),
                "child compiler counter": lambda case: case["memory"][
                    "cpu_rss_fresh_process"
                ]["counter_growth"].__setitem__("unique_graphs", 1),
                "child storage address": lambda case: case["memory"][
                    "cpu_rss_fresh_process"
                ]["storage_addresses_after"].__setitem__("state.ex", 4097),
                "measurement contract": lambda case: case[
                    "benchmark_contract"
                ].__setitem__("warmup_steps", 6),
                "changed field": lambda case: case["state_progress"][
                    "changed_buffers"
                ].remove("ex"),
                "state step count": lambda case: case["state_progress"].__setitem__(
                    "timed_step_count", 106
                ),
                "material diagnostics": lambda case: case["diagnostics"][
                    "pml"
                ].__setitem__("active_cells", 0),
            }
            raw_evidence_results = {}
            for label, mutate in raw_evidence_mutations.items():
                tampered = artifact(4)
                mutate(tampered["cases"][0])
                raw_evidence_results[label] = (
                    self.benchmark._aggregate_cpu_slice_outputs(
                        [artifact(1), tampered],
                        manifest,
                        Path("native.json"),
                        torch_baseline,
                        None,
                        evidence,
                    )
                )
        self.assertFalse(invalid["suite_acceptance"]["passed"])
        self.assertFalse(experimental_result["suite_acceptance"]["passed"])
        self.assertFalse(slow["suite_acceptance"]["passed"])
        self.assertFalse(tampered_baseline_result["suite_acceptance"]["passed"])
        self.assertFalse(invalid_allocation_result["suite_acceptance"]["passed"])
        self.assertTrue(
            bootstrap_failure["suite_acceptance"][
                "torch_baseline_individual_within_five_percent"
            ]
        )
        self.assertFalse(
            bootstrap_failure["suite_acceptance"]["torch_baseline_geomean_statistics"][
                "passed"
            ]
        )
        self.assertFalse(bootstrap_failure["suite_acceptance"]["passed"])
        self.assertFalse(invalid_native_result["suite_acceptance"]["passed"])
        self.assertFalse(thread_environment_result["suite_acceptance"]["passed"])
        for label, raw_evidence_result in raw_evidence_results.items():
            with self.subTest(raw_evidence=label):
                self.assertFalse(raw_evidence_result["suite_acceptance"]["passed"])
                self.assertIn(
                    "CPU runtime evidence",
                    " ".join(raw_evidence_result["suite_acceptance"]["errors"]),
                )

    def test_cpu_slice_aggregator_rejects_mixed_candidate_provenance(self):
        output = {"schema_version": 4, "kind": "cpu-acceptance-thread-slice"}
        result = self.benchmark._evaluate_cpu_slice(
            output,
            self.manifest,
            expected_evidence={"candidate_git_status": ""},
        )
        self.assertFalse(result["passed"])
        self.assertIn("provenance", " ".join(result["errors"]))

    def test_policy_matrix_binds_distinct_runtime_paths_and_timing(self):
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
            descriptor_root=Path("/tmp"),
            candidate_evidence={
                "candidate_git_commit": "a" * 40,
                "candidate_git_status": "",
                "manifest_sha256": "b" * 64,
            },
        )

        representations = self.benchmark.POLICY_EXECUTION_REPRESENTATIONS

        def sample(policy, seconds):
            resolved = "compact" if policy == "auto" else policy
            expected_representation = (
                "policy-dispatched-bucket-io-v2[" f"{representations[resolved]}]"
            )
            runtime_preimage = [None] * 31
            runtime_preimage[3] = "torch.float64"
            runtime_preimage[4] = "compile"
            runtime_preimage[5] = "default"
            runtime_preimage[6] = (
                "local-two-static-half-step-regions+external-boundary-sync-v1"
            )
            runtime_preimage[8] = expected_representation
            runtime_preimage[18] = False
            runtime_preimage[19] = True
            runtime_preimage[20] = None
            value = {
                "runtime": {
                    "device": "cpu",
                    "precision": "float64",
                    "field_storage_dtype": "torch.float64",
                    "compile_policy": "compile",
                    "compile_mode": "default",
                    "explicit_cuda_graphs": False,
                    "paired_real": False,
                    "execution_policy": policy,
                    "compile_cache_key": hashlib.sha256(
                        repr(tuple(runtime_preimage)).encode()
                    ).hexdigest(),
                },
                "diagnostics": {
                    "dispersive": {
                        "execution_representation": expected_representation,
                        "policy_executions": [
                            {
                                "component": "Ex",
                                "model": "dcp-plrc",
                                "policy": resolved,
                                "execution_representation": representations[resolved],
                                "targets": 12,
                            }
                        ],
                    }
                },
                "acceptance": {"passed": True},
                "measurements": {"advance": {"seconds_per_step": seconds}},
            }
            config = self.benchmark._policy_config_preimage(
                value, "cpu-crossover-2d", policy
            )
            value["compile_cache_key_evidence"] = {
                "schema_version": 1,
                "algorithm": self.benchmark.COMPILE_CACHE_PREIMAGE_ALGORITHM,
                "runtime_preimage": runtime_preimage,
                "policy_config": config,
                "policy_config_sha256": self.benchmark._canonical_json_sha256(config),
            }
            return value

        samples = {
            "auto": sample("auto", 1.05),
            "dense": sample("dense", 1.0),
            "compact": sample("compact", 1.2),
            "tiled": sample("tiled", 1.3),
        }

        def run(_name, **kwargs):
            return json.loads(json.dumps(samples[kwargs["execution_policy"]]))

        with (
            patch.object(self.benchmark, "run_case", side_effect=run),
            patch.object(
                self.benchmark,
                "_policy_execution_diagnostic",
                side_effect=lambda _args, _name, _manifest, policy: {
                    "execution_policy": policy
                },
            ),
        ):
            result = self.benchmark._policy_matrix(
                args, "cpu-crossover-2d", self.manifest
            )
            self.assertTrue(result["comparison_valid"])
            self.assertAlmostEqual(result["auto_to_fastest_forced_ratio"], 1.05)
            self.assertTrue(result["within_ten_percent"])
            self.assertTrue(result["all_acceptance_passed"])
            self.assertTrue(result["passed"])

            samples["compact"]["diagnostics"]["dispersive"]["policy_executions"][0][
                "execution_representation"
            ] = representations["dense"]
            tampered_representation = self.benchmark._policy_matrix(
                args, "cpu-crossover-2d", self.manifest
            )
            self.assertFalse(tampered_representation["comparison_valid"])
            self.assertFalse(tampered_representation["passed"])

            samples["compact"] = sample("compact", 1.2)
            samples["compact"]["runtime"]["compile_cache_key"] = samples["dense"][
                "runtime"
            ]["compile_cache_key"]
            duplicate_key = self.benchmark._policy_matrix(
                args, "cpu-crossover-2d", self.manifest
            )
            self.assertFalse(duplicate_key["comparison_valid"])
            self.assertFalse(duplicate_key["passed"])

            samples["compact"] = sample("compact", 1.2)
            samples["auto"] = sample("auto", 1.11)
            slow_auto = self.benchmark._policy_matrix(
                args, "cpu-crossover-2d", self.manifest
            )
            self.assertTrue(slow_auto["comparison_valid"])
            self.assertFalse(slow_auto["within_ten_percent"])
            self.assertFalse(slow_auto["passed"])

            samples["auto"] = sample("auto", 1.05)
            samples["tiled"]["acceptance"]["passed"] = False
            rejected = self.benchmark._policy_matrix(
                args, "cpu-crossover-2d", self.manifest
            )
            self.assertTrue(rejected["comparison_valid"])
            self.assertFalse(rejected["all_acceptance_passed"])
            self.assertFalse(rejected["passed"])

    def _cuda_evidence_case(self, name, *, complex_case, region_count=None):
        raw_plan = [
            {
                "component": "Ex",
                "shape": [2, 2, 1],
                "dense_inverse": [[[1.0], [1.0]], [[1.0], [1.0]]],
                "constant_targets": [],
                "constant_values": [],
                "buckets": [
                    {
                        "signature": {
                            "component": "Ex",
                            "model": "drude",
                            "precision": "float32",
                            "state_shape": [1],
                        },
                        "coefficient_names": ["eps_inf"],
                        "targets": [0, 1],
                        "target_coefficients": [[1.2], [1.2]],
                        "cell_coefficient_names": [],
                        "cell_coefficients": [],
                    }
                ],
            }
        ]
        result = {
            "workload": {"name": name, "complex": complex_case},
            "benchmark_contract": {
                "repetitions": 3,
                "steps_per_repeat": 10,
                "profile_steps": 2,
            },
            "runtime": {
                "device": "cuda:0",
                "precision": "float32",
                "paired_real": complex_case,
                "field_storage_representation": (
                    "paired-real-v1" if complex_case else "real-v1"
                ),
                "field_storage_channels": 2 if complex_case else 1,
                "field_storage_dtype": "torch.float32",
            },
            "measurements": {
                "advance": {
                    "raw_seconds": [1.0, 1.0, 1.0],
                    "median_seconds": 1.0,
                    "relative_mad": 0.0,
                    "repetitions": 3,
                    "steps_per_repeat": 10,
                    "seconds_per_step": 0.1,
                }
            },
            "memory": {
                "cuda_allocated_before_bytes": 1024,
                "cuda_allocated_after_bytes": 1024,
                "cuda_allocated_growth_bytes": 0,
                "cuda_peak_allocated_bytes": 4096,
                "cuda_peak_reserved_bytes": 8192,
                "bounded": True,
            },
            "profiler": {
                "profile_steps": 2,
                "kernel_launches": 20,
                "host_to_device_events": 0,
                "device_to_host_events": 0,
            },
            "acceptance": {"passed": True},
        }
        if region_count is not None:
            result["workload"].update(
                {
                    "geometry_region_count": region_count,
                    "geometry_object_count": region_count + 1,
                }
            )
            result["diagnostics"] = {
                "material_plan": [{"launches": 3}, {"launches": 3}]
            }
            result["region_equivalence"] = {
                "contract_id": "material-region-launch-invariance-v1",
                "equivalence_group": "overlapping-identical-drude-block-v1",
                "geometry_region_count": region_count,
                "geometry_object_count": region_count + 1,
                "material_compute_launches_per_step": 6,
                "effective_material_plan": raw_plan,
                "effective_material_plan_sha256": self.benchmark._canonical_sha256(
                    raw_plan
                ),
            }
        return result

    def test_paired_real_gate_recomputes_timing_memory_and_trace_contracts(self):
        results = [
            self._cuda_evidence_case(name, complex_case=True)
            for name in self.benchmark.PAIRED_REAL_GATES
        ]
        with patch.object(self.benchmark, "_profiler_trace_matches", return_value=True):
            gate = self.benchmark._paired_real_cuda_gate(results)
            self.assertTrue(gate["passed"])

            timing_tamper = json.loads(json.dumps(results))
            timing_tamper[0]["measurements"]["advance"]["relative_mad"] = 0.5
            self.assertFalse(
                self.benchmark._paired_real_cuda_gate(timing_tamper)["passed"]
            )

            transfer_tamper = json.loads(json.dumps(results))
            transfer_tamper[0]["profiler"]["host_to_device_events"] = 1
            self.assertFalse(
                self.benchmark._paired_real_cuda_gate(transfer_tamper)["passed"]
            )

            representation_tamper = json.loads(json.dumps(results))
            representation_tamper[0]["runtime"]["paired_real"] = False
            self.assertFalse(
                self.benchmark._paired_real_cuda_gate(representation_tamper)["passed"]
            )

    def test_region_invariance_gate_binds_raw_plans_and_profiled_launches(self):
        results = [
            self._cuda_evidence_case(name, complex_case=False, region_count=count)
            for name, count in zip(
                self.benchmark.REGION_INVARIANCE_GATES, (1, 32), strict=True
            )
        ]
        with patch.object(self.benchmark, "_profiler_trace_matches", return_value=True):
            gate = self.benchmark._region_invariance_gate(results)
            self.assertTrue(gate["passed"])

            plan_tamper = json.loads(json.dumps(results))
            plan_tamper[1]["region_equivalence"]["effective_material_plan"][0][
                "buckets"
            ][0]["targets"].append(2)
            plan_tamper[1]["region_equivalence"]["effective_material_plan_sha256"] = (
                self.benchmark._canonical_sha256(
                    plan_tamper[1]["region_equivalence"]["effective_material_plan"]
                )
            )
            self.assertFalse(
                self.benchmark._region_invariance_gate(plan_tamper)["passed"]
            )

            launch_tamper = json.loads(json.dumps(results))
            launch_tamper[1]["profiler"]["kernel_launches"] = 21
            self.assertFalse(
                self.benchmark._region_invariance_gate(launch_tamper)["passed"]
            )

    def test_equivalent_region_cases_increase_only_input_region_count(self):
        baseline = self.benchmark._build_case(
            self.benchmark.REGION_INVARIANCE_GATES[0], self.manifest
        )
        expanded = self.benchmark._build_case(
            self.benchmark.REGION_INVARIANCE_GATES[1], self.manifest
        )
        baseline_spec, baseline_space, baseline_geometry, _, _ = baseline
        expanded_spec, expanded_space, expanded_geometry, _, _ = expanded
        self.assertEqual(baseline_spec["size"], expanded_spec["size"])
        self.assertEqual(
            baseline_space.whole_field_size.tolist(),
            expanded_space.whole_field_size.tolist(),
        )
        self.assertEqual(baseline_spec["geometry_region_count"], 1)
        self.assertEqual(expanded_spec["geometry_region_count"], 32)
        self.assertEqual(len(baseline_geometry), 2)
        self.assertEqual(len(expanded_geometry), 33)
        self.assertEqual(
            baseline_spec["equivalence_group"], expanded_spec["equivalence_group"]
        )

    def test_policy_gate_enforces_the_complete_matrix(self):
        import torch

        args = SimpleNamespace(
            case="policy-gates",
            device="cpu",
            precision="float64",
            compile_mode="default",
            policy="matrix",
            capture_graphs=False,
            cpu_rss_child=False,
            experimental_dispersive_grouping=False,
            experimental_dispersive_grouping_scope="combined",
            threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            warmup=5,
            steps=100,
            repeats=15,
            profile_steps=1,
            trace_directory=Path("/tmp"),
            descriptor_root=Path("/tmp"),
            native_summary=None,
            torch_baseline_slice_artifacts=None,
            allocation_provenance=None,
            cpu_slice_artifacts=None,
            correctness_evidence_index=None,
            output=None,
            enforce=True,
        )

        def result(_args, name, _manifest, _allocation):
            return {"case": name, "passed": True}

        with (
            patch.object(
                self.benchmark,
                "_arguments",
                return_value=(args, self.manifest),
            ),
            patch.object(self.benchmark, "_policy_matrix", side_effect=result),
            patch.object(self.benchmark, "_environment", return_value={}),
            patch.object(
                self.benchmark,
                "_current_evidence",
                return_value={"candidate_git_status": ""},
            ),
            patch("builtins.print"),
        ):
            self.assertEqual(self.benchmark.main(), 0)

        def rejected(_args, name, _manifest, _allocation):
            return {
                "case": name,
                "passed": name != self.benchmark.POLICY_GATES[-1],
            }

        with (
            patch.object(
                self.benchmark,
                "_arguments",
                return_value=(args, self.manifest),
            ),
            patch.object(self.benchmark, "_policy_matrix", side_effect=rejected),
            patch.object(self.benchmark, "_environment", return_value={}),
            patch.object(
                self.benchmark,
                "_current_evidence",
                return_value={"candidate_git_status": ""},
            ),
            patch("builtins.print"),
        ):
            self.assertEqual(self.benchmark.main(), 2)

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
            torch_baseline_slice_artifacts=None,
            allocation_provenance=None,
            cpu_slice_artifacts=None,
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

    def test_cpu_slice_aggregation_forwards_allocation_document(self):
        args = SimpleNamespace(
            case="cpu-gates",
            native_summary=Path("native.json"),
            torch_baseline_slice_artifacts=(
                Path("baseline-one.json"),
                Path("baseline-physical.json"),
            ),
            allocation_provenance=Path("allocation.json"),
            cpu_slice_artifacts=(
                Path("candidate-one.json"),
                Path("candidate-physical.json"),
            ),
            output=None,
            enforce=True,
        )
        document = {
            "source_artifact": {
                "path": "/allocation.json",
                "sha256": "a" * 64,
            },
            "records": [],
        }
        aggregate = {
            "suite_acceptance": {"passed": True},
            "issue_completion_satisfied": False,
        }
        with (
            patch.object(
                self.benchmark,
                "_arguments",
                return_value=(args, self.manifest),
            ),
            patch.object(
                self.benchmark,
                "_load_allocation_provenance",
                return_value=document,
            ),
            patch.object(
                self.benchmark,
                "_aggregate_cpu_slice_files",
                return_value=aggregate,
            ) as aggregate_mock,
            patch("builtins.print"),
        ):
            status = self.benchmark.main()
        aggregate_mock.assert_called_once_with(
            args.cpu_slice_artifacts,
            self.manifest,
            args.native_summary,
            args.torch_baseline_slice_artifacts,
            document,
            None,
        )
        self.assertEqual(status, 2)

    def test_enforced_cpu_gate_requires_a_torch_baseline_comparison(self):
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
            torch_baseline_slice_artifacts=None,
            allocation_provenance=None,
            cpu_slice_artifacts=None,
            output=None,
            enforce=True,
        )
        sample = {"acceptance": {"passed": True}}
        native_gate = {
            "comparison_role": "informational",
            "comparison_valid": True,
            "torch_to_native_ratio": 100.0,
        }
        with (
            patch.object(
                self.benchmark,
                "_arguments",
                return_value=(args, self.manifest),
            ),
            patch.object(self.benchmark, "run_case", return_value=sample),
            patch.object(self.benchmark, "_native_gate", return_value=native_gate),
            patch.object(
                self.benchmark,
                "_environment",
                return_value={"cpu_count_physical_affinity": 4},
            ),
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
            torch_baseline_slice_artifacts=(Path("one.json"), Path("four.json")),
            allocation_provenance=None,
            cpu_slice_artifacts=None,
            output=None,
            enforce=True,
        )
        sample = {"acceptance": {"passed": True}}
        native_gate = {
            "comparison_role": "informational",
            "comparison_valid": True,
            "torch_to_native_ratio": 100.0,
        }
        baseline_gate = {
            "comparison_valid": True,
            "candidate_to_torch_baseline_ratio": 1.0,
            "within_five_percent": True,
        }
        environment = {
            "hostname": "host",
            "platform": "platform",
            "python": "3.14",
            "torch": "2.13",
            "cuda_runtime": None,
            "devices": [],
            "cpu_count": 4,
            "cpu_affinity": [0, 1, 2, 3],
            "cpu_count_physical_affinity": 4,
            "cpu_topology": "topology",
            "cpu_model": "model",
            "gpu_topology": None,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        }
        torch_baseline = {
            "kind": "torch-cpu-baseline",
            "cpu_acceptance_contract_id": "cpu-acceptance-v2",
            "timing_reference": {"root_commit": "baseline"},
            "environment": {
                "hostname": "redacted",
                "host_identity": self.benchmark.privacy_preserving_host_identity(
                    environment, salt="b" * 64
                ),
            },
            "source_artifacts": [
                {
                    "publication_url": (
                        "https://github.com/ruddyscent/gmes/releases/download/"
                        "issue-123-torch-cpu-baseline-v2/one.json"
                    ),
                    "size_bytes": 1,
                    "sha256": "1" * 64,
                    "thread_mode": "one",
                    "threads": 1,
                    "thread_environment": dict(environment["thread_environment"]),
                    "root_commit": "baseline",
                }
            ],
        }
        with (
            patch.object(
                self.benchmark,
                "_arguments",
                return_value=(args, self.manifest),
            ),
            patch.object(self.benchmark, "run_case", return_value=sample),
            patch.object(self.benchmark, "_native_gate", return_value=native_gate),
            patch.object(
                self.benchmark,
                "load_torch_cpu_baseline",
                return_value=torch_baseline,
            ),
            patch.object(
                self.benchmark,
                "compare_candidate_to_baseline",
                return_value=baseline_gate,
            ),
            patch.object(self.benchmark, "_environment", return_value=environment),
            patch("builtins.print") as rendered,
        ):
            status = self.benchmark.main()
        output = json.loads(rendered.call_args.args[0])
        self.assertTrue(output["diagnostic_acceptance"]["passed"])
        self.assertFalse(output["suite_acceptance"]["passed"])
        self.assertEqual(
            output["suite_acceptance"]["cpu_suite_status"], "diagnostic-only"
        )
        self.assertEqual(
            output["suite_acceptance"]["native_comparison_role"],
            "informational",
        )
        self.assertTrue(
            output["suite_acceptance"]["torch_baseline_individual_within_five_percent"]
        )
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
