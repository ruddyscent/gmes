import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "benchmarks" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gmes

        cls.gmes = gmes
        cls.oracle = load_script("native_oracle.py")
        cls.isolated = load_script("run_isolated_oracle.py")
        cls.manifest = cls.oracle.load_manifest()

    def test_manifest_freezes_reference_matrix_and_gates(self):
        reference = self.manifest["reference"]
        self.assertEqual(
            reference["commit"], "d87d25afd160d96b1fa0890cacecd90802448d57"
        )
        self.assertEqual(reference["tag"], "native-oracle-d87d25a")
        self.assertEqual(reference["observer_tag"], "native-oracle-observer-v2")
        self.assertEqual(reference["field_scale"], 1e-3)
        self.assertEqual(reference["capture_steps"], [1, 2, 5, 20, 100])
        names = {case["name"] for case in self.manifest["correctness"]}
        self.assertTrue(
            {
                "dielectric-1d",
                "dielectric-2d",
                "dielectric-3d",
                "singleton-3d",
                "bloch-2d",
                "bloch-3d",
                "upml",
                "cpml",
                "dcp-ade",
                "dcp-plrc",
                "dcp-rc",
                "upml-bloch",
                "cpml-bloch",
                "lorentz-bloch",
                "dcp-ade-bloch",
                "dcp-plrc-bloch",
                "dcp-rc-bloch",
                "dm2-4",
                "mixed-2d",
                "mixed-3d",
            }.issubset(names)
        )
        gates = self.manifest["performance_gates"]
        self.assertEqual(gates["coverage_percent"], [1, 10, 50, 90])
        self.assertEqual(gates["single_gpu"]["devices"], [0])
        self.assertEqual(gates["two_gpu"]["devices"], [0, 1])
        self.assertGreaterEqual(gates["cpu_large"]["repeats"], 11)
        self.assertEqual(gates["cpu_large"]["cases"], ["cpu-large-2d", "cpu-large-3d"])
        benchmark_names = {case["name"] for case in self.manifest["benchmarks"]}
        self.assertTrue(
            {
                "heterogeneous-16-cylinder",
                "pml-thin",
                "pml-thick",
                "coverage-1-contiguous",
                "coverage-90-fragmented",
                "single-gpu-3d",
                "two-gpu-3d",
            }.issubset(benchmark_names)
        )

    def test_dcp_rc_remains_a_distinct_strategy(self):
        rc = self.oracle.material_from_name("dcp-rc", self.gmes)
        plrc = self.oracle.material_from_name("dcp-plrc", self.gmes)
        self.assertIs(type(rc), self.gmes.DcpRc)
        self.assertIs(type(plrc), self.gmes.DcpPlrc)
        self.assertIsNot(type(rc), type(plrc))

    def test_mixed_workload_initializes_all_stateful_families(self):
        spec = dict(self.oracle.find_case(self.manifest, "mixed-3d"))
        spec.update(size=[12, 6, 4], resolution=2)
        simulation = self.oracle.build_simulation(spec, self.gmes)
        simulation.init()
        strategies = {
            type(geometry.material).__name__ for geometry in simulation.geom_list
        }
        self.assertTrue(
            {
                "Upml",
                "Cpml",
                "Drude",
                "Lorentz",
                "DcpAde",
                "DcpPlrc",
                "DcpRc",
                "Dm2",
            }.issubset(strategies)
        )
        self.oracle.initialize_fields(simulation, 115)
        simulation.step()
        arrays = {}
        snapshot = self.oracle._snapshot(simulation, 1, arrays)
        captured = {
            strategy
            for record in snapshot["materials"]
            for strategy in record["strategies"]
        }
        self.assertTrue(
            {
                "Upml",
                "Cpml",
                "Drude",
                "Lorentz",
                "DcpAde",
                "DcpPlrc",
                "DcpRc",
                "Dm2",
            }.issubset(captured)
        )
        self.assertTrue(
            all(
                record["state_nonzero_values"] > 0
                for record in snapshot["materials"]
                if record["state_values"]
            )
        )

    def test_tfsf_snapshot_contains_auxiliary_fields_and_state(self):
        spec = self.oracle.find_case(self.manifest, "tfsf-transparent")
        simulation = self.oracle.build_simulation(spec, self.gmes)
        simulation.init()
        self.oracle.initialize_fields(simulation, 115)
        simulation.step()
        arrays = {}
        snapshot = self.oracle._snapshot(simulation, 1, arrays)
        self.assertTrue(snapshot["sources"]["auxiliary"])
        self.assertTrue(
            any("/source_aux/" in key and "/field/" in key for key in arrays)
        )
        self.assertTrue(
            any("/source/" in key and key.endswith("/indices") for key in arrays)
        )

    def test_capture_contains_complete_fields_maps_and_state(self):
        spec = dict(self.oracle.find_case(self.manifest, "drude-1"))
        spec.update(size=[2, 2, 2], resolution=2)
        manifest = json.loads(json.dumps(self.manifest))
        manifest["reference"]["capture_steps"] = [1, 2]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "reference.npz"
            metadata = self.oracle.capture_case(spec, manifest, artifact)
            self.assertGreater(metadata["active_cells"], 0)
            self.assertGreater(metadata["state_bytes"], 0)
            self.assertGreater(metadata["index_bytes"], 0)
            self.assertGreater(metadata["parameter_bytes"], metadata["state_bytes"])
            self.assertEqual(
                metadata["live_updater_bytes"],
                metadata["plan_bytes"]
                + metadata["index_bytes"]
                + metadata["parameter_bytes"],
            )
            self.assertTrue(metadata["nonzero_persistent_state"])
            material = metadata["steps"]["1"]["materials"][0]
            self.assertGreater(material["coverage"], 0)
            self.assertGreater(material["fragmentation_runs"], 0)
            self.assertGreater(material["state_nonzero_values"], 0)
            with np.load(artifact, allow_pickle=False) as archive:
                for component in self.oracle.COMPONENT_NAMES:
                    self.assertIn(f"step/1/field/{component}", archive.files)
                    self.assertIn(f"step/2/field/{component}", archive.files)
                    self.assertIn(f"map/{component}/material_ids", archive.files)
                    self.assertIn(f"map/{component}/underlying_ids", archive.files)
                state_keys = [key for key in archive.files if "/state/" in key]
                self.assertTrue(state_keys)
                self.assertIn("step/1/time", archive.files)
                self.assertIn("step/0/time", archive.files)
                self.assertEqual(metadata["input_state"]["archive_prefix"], "step/0")
                self.assertIn("step/1/physical/summary", archive.files)
                self.assertTrue(any("/source/" in key for key in archive.files))
                result = self.oracle.compare_archives(artifact, artifact, manifest)
                self.assertTrue(result["passed"])

    def test_benchmark_schema_characterizes_noise_and_memory_growth(self):
        spec = dict(self.oracle.find_case(self.manifest, "dielectric-2d"))
        spec.update(size=[2, 2, 0], resolution=2)
        simulations = []
        original_build = self.oracle.build_simulation

        def tracked_build(*args, **kwargs):
            simulation = original_build(*args, **kwargs)
            original_step = simulation.step
            simulation.oracle_step_count = 0

            def tracked_step():
                simulation.oracle_step_count += 1
                return original_step()

            simulation.step = tracked_step
            simulations.append(simulation)
            return simulation

        with patch.object(self.oracle, "build_simulation", side_effect=tracked_build):
            result = self.oracle.benchmark_case(
                spec, self.manifest, repeats=2, warmup=1, steps=1
            )
        measurements = result["measurements"]
        self.assertEqual(measurements["advance"]["repetitions"], 2)
        self.assertIn("relative_mad", measurements["advance"])
        self.assertIn("geometry_mapping", measurements)
        self.assertIn("one_step", measurements)
        self.assertEqual(len(result["memory"]["rss_samples_bytes"]), 3)
        self.assertGreater(result["memory"]["live_field_bytes"], 0)
        self.assertGreater(result["memory"]["live_index_bytes"], 0)
        self.assertGreater(result["memory"]["live_parameter_bytes"], 0)
        self.assertIn("git_commit", result["environment"])
        self.assertIn("cpu_count_physical", result["environment"])
        self.assertEqual(
            [simulation.oracle_step_count for simulation in simulations],
            [0, 3, 2, 2],
        )

    def test_isolated_runner_uses_controller_with_checkout_bound_import(self):
        manifest = json.loads(json.dumps(self.manifest))
        manifest["reference"]["capture_steps"] = [1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            output = directory / "isolated.npz"
            result = self.isolated.run_capture(
                ROOT, Path(sys.executable), manifest_path, "dielectric-1d", output
            )
            self.assertEqual(Path(result["checkout"]), ROOT)
            self.assertEqual(result["capture"]["capture_steps"], [1])
            self.assertTrue(output.is_file())

    def test_capture_rejects_import_outside_requested_checkout(self):
        spec = dict(self.oracle.find_case(self.manifest, "dielectric-1d"))
        manifest = json.loads(json.dumps(self.manifest))
        manifest["reference"]["capture_steps"] = [1]
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"GMES_ORACLE_EXPECTED_CHECKOUT": str(Path(directory) / "wrong")},
            ):
                with self.assertRaisesRegex(RuntimeError, "outside the requested"):
                    self.oracle.capture_case(
                        spec, manifest, Path(directory) / "rejected.npz"
                    )

    def test_isolated_environment_removes_import_leaks(self):
        environment = self.isolated.sanitized_environment()
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("VIRTUAL_ENV", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")


if __name__ == "__main__":
    unittest.main()
