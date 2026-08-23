import importlib.util
import unittest
from pathlib import Path


def load_field_updates():
    benchmark_path = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "field_updates.py"
    )
    spec = importlib.util.spec_from_file_location("field_updates", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


if __name__ == "__main__":
    unittest.main()
