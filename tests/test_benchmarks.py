import importlib.util
import unittest
from pathlib import Path


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

    def test_stateful_matrix_is_plan_only_and_width_normalized(self):
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
        self.assertIn("plan-only", result["execution"])
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


if __name__ == "__main__":
    unittest.main()
