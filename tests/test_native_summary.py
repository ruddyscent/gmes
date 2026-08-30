import copy
import hashlib
import importlib.util
import json
import math
import platform
import tempfile
import unittest
from pathlib import Path
from statistics import median, pstdev

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "native_oracle_workloads.json"


def load_script(name):
    path = ROOT / "benchmarks" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_script("native_summary.py")
        cls.tuning = load_script("torch_tuning.py")
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.reference = cls.manifest["reference"]
        cls.cases = {
            case["name"]: case
            for group in ("correctness", "benchmarks", "physical_checks")
            for case in cls.manifest.get(group, ())
        }
        cls.contract = {
            "initializer": cls.reference["field_initializer"],
            "seed": cls.reference["seed"],
            "field_scale": cls.reference["field_scale"],
            "warmup_steps": cls.reference["performance_warmup_steps"],
            "steps_per_repeat": cls.reference["performance_steps_per_repeat"],
            "repetitions": cls.reference["performance_repetitions"],
            "timer": "time.perf_counter",
            "sample_start": "independently-rebuilt-post-warmup-state",
        }

    def _timing(self, base):
        repetitions = self.contract["repetitions"]
        raw = [base + base * index / 1000 for index in range(repetitions)]
        middle = median(raw)
        return {
            "raw_seconds": raw,
            "median_seconds": middle,
            "p95_seconds": self.summary._percentile95(raw),
            "population_stdev_seconds": pstdev(raw),
            "relative_mad": median(abs(value - middle) for value in raw) / middle,
            "repetitions": repetitions,
        }

    def _environment(self, threads, scaling):
        return {
            "platform": "Linux-test",
            "hostname": "oracle-host",
            "os": {
                "system": "Linux",
                "node": "oracle-host",
                "release": "test",
                "version": "test",
                "machine": "x86_64",
                "processor": "x86_64",
            },
            "python": "3.14.0",
            "python_executable": "/observer/.venv/bin/python",
            "numpy": "2.5.0",
            "gmes_version": "0.10.0",
            "gmes_source": "/observer/gmes/__init__.py",
            "native_extension": "/observer/gmes/_pw_material.so",
            "git_commit": self.reference["observer_commit"],
            "git_status": "",
            "uv_lock_sha256": "a" * 64,
            "python_compiler": "Clang test",
            "python_build_cflags": "-O3",
            "cxx_version": "c++ test",
            "swig_version": "SWIG test",
            "extension_compile_standard": "c++23",
            "build_environment": {
                "CC": None,
                "CXX": None,
                "CFLAGS": None,
                "CXXFLAGS": None,
                "LDFLAGS": None,
                "GMES_ENABLE_OPENMP": None,
                "GMES_OPENMP_PREFIX": None,
                "MACOSX_DEPLOYMENT_TARGET": None,
            },
            "openmp_enabled": True,
            "openmp_threads": threads,
            "omp_num_threads": str(threads),
            "cpu_count_logical": 8,
            "cpu_count_physical": 4,
            "cpu_topology": "# Core,Socket\n0,0\n1,0\n2,0\n3,0",
            "cpu_model": (
                "Architecture: x86_64\n"
                f"CPU(s) scaling MHz: {scaling}%\n"
                "Model name: Test CPU"
            ),
            "memory_bytes": 16_000_000_000,
            "gpu": [],
            "gpu_topology": None,
            "torch": None,
        }

    def _cell(self, name, threads, index):
        advance = self._timing(1.0 + index / 10)
        cells = 10 + index
        state_values = cells * 2
        state_bytes = state_values * 16
        plan_bytes = 64 + index
        index_bytes = cells * 12
        parameter_bytes = state_bytes + 32
        live_updater_bytes = plan_bytes + index_bytes + parameter_bytes
        updater = {
            "component": "Ex",
            "strategy": "Drude",
            "strategies": ["Drude"],
            "native_type": "ExDrudeReal",
            "cells": cells,
            "coverage": 0.5,
            "fragmentation_runs": 1,
            "fragmentation_ratio": 1 / cells,
            "state_values": state_values,
            "state_nonzero_values": state_values,
            "state_width": 2.0,
            "state_key": "step/benchmark/state/Ex/0-Drude/values",
            "state_bytes": state_bytes,
            "plan_bytes": plan_bytes,
            "index_bytes": index_bytes,
            "parameter_bytes": parameter_bytes,
            "live_updater_bytes": live_updater_bytes,
            "plan_runs": 1,
            "bucket_signature": [
                "Ex",
                "Drude",
                "ExDrudeReal",
                cells,
                state_values,
            ],
        }
        advance.update(
            {
                "steps_per_repeat": self.contract["steps_per_repeat"],
                "steps_per_second": self.contract["steps_per_repeat"]
                / advance["median_seconds"],
                "cells_per_second": (
                    cells
                    * self.contract["steps_per_repeat"]
                    / advance["median_seconds"]
                ),
            }
        )
        repetitions = self.contract["repetitions"]
        rss = [1_000_000 + index * 1000 + sample for sample in range(repetitions + 1)]
        return {
            "schema_version": 2,
            "backend": "native",
            "workload": copy.deepcopy(self.cases[name]),
            "benchmark_contract": copy.deepcopy(self.contract),
            "environment": self._environment(threads, 80 + index),
            "measurements": {
                "construction": self._timing(0.01 + index / 10_000),
                "geometry_mapping": self._timing(0.02 + index / 10_000),
                "native_initialization_and_plan_lowering": self._timing(
                    0.03 + index / 10_000
                ),
                "host_to_device_transfer": {
                    "raw_seconds": [0.0] * repetitions,
                    "median_seconds": 0.0,
                    "p95_seconds": 0.0,
                },
                "eager_warmup_seconds": 0.1 + index / 1000,
                "cold_compile": None,
                "cached_compile": None,
                "one_step": self._timing(0.001 + index / 100_000),
                "advance": advance,
            },
            "memory": {
                "peak_rss_bytes": 2_000_000 + index,
                "rss_samples_bytes": rss,
                "rss_growth_bytes": rss[-1] - rss[0],
                "live_field_bytes": 1000 + index,
                "live_plan_bytes": plan_bytes,
                "live_index_bytes": index_bytes,
                "live_parameter_bytes": parameter_bytes,
                "live_updater_bytes": live_updater_bytes,
                "live_state_bytes": state_bytes,
                "cuda_allocated_peak_bytes": None,
                "cuda_reserved_peak_bytes": None,
            },
            "updaters": [updater],
            "profiler": None,
        }

    def _write_inputs(self, directory):
        paths = []
        index = 0
        for name in self.summary.CASE_NAMES:
            for threads in (1, 4):
                path = Path(directory) / f"{name}-t{threads}.json"
                path.write_text(
                    json.dumps(
                        self._cell(name, threads, index), indent=2, sort_keys=True
                    )
                    + "\n"
                )
                paths.append(path)
                index += 1
        return paths

    def _mutated_input(self, directory, source, mutation, suffix):
        cell = json.loads(Path(source).read_text())
        mutation(cell)
        path = Path(directory) / f"mutated-{suffix}.json"
        path.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
        return path

    def test_assembles_complete_matrix_in_deterministic_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_inputs(directory)
            expected_sha = {
                (
                    json.loads(path.read_text())["workload"]["name"],
                    json.loads(path.read_text())["environment"]["openmp_threads"],
                ): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            }
            forward = self.summary.assemble_summary(paths, MANIFEST)
            reverse = self.summary.assemble_summary(reversed(paths), MANIFEST)

        self.assertEqual(forward, reverse)
        self.assertEqual(
            self.summary.render_summary(forward),
            self.summary.render_summary(reverse),
        )
        self.assertEqual(forward["schema_version"], 2)
        self.assertEqual(forward["kind"], "native-cpu-acceptance-summary")
        self.assertEqual(forward["observer_commit"], self.reference["observer_commit"])
        self.assertEqual(forward["benchmark_contract"], self.contract)
        self.assertEqual(len(forward["samples"]), 12)
        self.assertEqual(
            [
                (sample["workload"]["name"], sample["openmp_threads"])
                for sample in forward["samples"]
            ],
            [(name, threads) for name in self.summary.CASE_NAMES for threads in (1, 4)],
        )
        self.assertNotIn("openmp_threads", forward["environment"])
        self.assertNotIn("omp_num_threads", forward["environment"])
        self.assertNotIn("CPU(s) scaling MHz:", forward["environment"]["cpu_model"])
        for source in forward["source_artifacts"]:
            key = (source["workload"], source["threads"])
            self.assertEqual(source["sha256"], expected_sha[key])
            self.assertIn("CPU(s) scaling MHz:", source["raw_environment"]["cpu_model"])
            self.assertEqual(
                source["raw_environment"]["omp_num_threads"],
                str(source["threads"]),
            )

    def test_cli_writes_canonical_output(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_inputs(directory)
            output = Path(directory) / "summary.json"
            self.summary.main(
                [
                    "--manifest",
                    str(MANIFEST),
                    "--output",
                    str(output),
                    *(str(path) for path in reversed(paths)),
                ]
            )
            expected = self.summary.render_summary(
                self.summary.assemble_summary(paths, MANIFEST)
            )
            self.assertEqual(output.read_text(), expected)
            self.assertTrue(output.read_bytes().endswith(b"\n"))

    def test_assembled_summary_is_consumed_by_native_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = self._write_inputs(directory)
            summary = self.summary.assemble_summary(paths, MANIFEST)
            summary["environment"]["hostname"] = platform.node()
            summary["environment"]["platform"] = platform.platform()
            output = directory / "native-summary.json"
            output.write_text(self.summary.render_summary(summary))
            manifest = copy.deepcopy(self.manifest)
            manifest["reference"]["performance_summary_sha256"] = hashlib.sha256(
                output.read_bytes()
            ).hexdigest()
            spec = copy.deepcopy(self.cases["cpu-crossover-2d"])
            raw = [0.1] * self.contract["repetitions"]
            candidate = {
                "workload": spec,
                "benchmark_contract": {
                    **self.contract,
                    "profile_steps": self.reference["performance_profile_steps"],
                    "sample_start": "independently-restored-pre-warmup-state",
                },
                "runtime": {
                    "device": "cpu",
                    "precision": "float64",
                    "threads": 1,
                    "interop_threads": 1,
                    "cpu_count_physical_affinity": summary["environment"][
                        "cpu_count_physical"
                    ],
                    "cpu_topology": summary["environment"]["cpu_topology"],
                },
                "measurements": {
                    "advance": {
                        "raw_seconds": raw,
                        "median_seconds": median(raw),
                        "seconds_per_step": (
                            median(raw) / self.contract["steps_per_repeat"]
                        ),
                    }
                },
            }
            result = self.tuning._native_gate(
                output, spec["name"], 1, candidate, manifest
            )
        self.assertTrue(result["comparison_valid"], result["contract_errors"])
        self.assertEqual(result["comparison_role"], "informational")

    def test_rejects_missing_and_duplicate_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_inputs(directory)
            with self.assertRaisesRegex(ValueError, "exactly twelve"):
                self.summary.assemble_summary(paths[:-1], MANIFEST)
            duplicate = [*paths[:-1], paths[0]]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.summary.assemble_summary(duplicate, MANIFEST)

    def test_rejects_cell_schema_workload_and_contract_changes(self):
        mutations = {
            "schema": lambda cell: cell.__setitem__("schema_version", 1),
            "workload": lambda cell: cell["workload"].__setitem__("size", [1]),
            "contract": lambda cell: cell["benchmark_contract"].__setitem__(
                "warmup_steps", 1
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            for suffix, mutation in mutations.items():
                with self.subTest(suffix=suffix):
                    paths = list(original)
                    paths[0] = self._mutated_input(
                        directory, original[0], mutation, suffix
                    )
                    with self.assertRaises(ValueError):
                        self.summary.assemble_summary(paths, MANIFEST)

    def test_rejects_wrong_or_dirty_observer_checkout(self):
        mutations = {
            "wrong-commit": lambda cell: cell["environment"].__setitem__(
                "git_commit", "0" * 40
            ),
            "dirty": lambda cell: cell["environment"].__setitem__(
                "git_status", " M gmes/fdtd.py"
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            for suffix, mutation in mutations.items():
                with self.subTest(suffix=suffix):
                    paths = list(original)
                    paths[0] = self._mutated_input(
                        directory, original[0], mutation, suffix
                    )
                    with self.assertRaisesRegex(ValueError, "observer commit|dirty"):
                        self.summary.assemble_summary(paths, MANIFEST)

    def test_rejects_invalid_threads_and_mixed_environments(self):
        mutations = {
            "thread": lambda cell: (
                cell["environment"].__setitem__("openmp_threads", 2),
                cell["environment"].__setitem__("omp_num_threads", "2"),
            ),
            "host": lambda cell: cell["environment"].__setitem__(
                "hostname", "other-host"
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            for suffix, mutation in mutations.items():
                with self.subTest(suffix=suffix):
                    paths = list(original)
                    paths[0] = self._mutated_input(
                        directory, original[0], mutation, suffix
                    )
                    with self.assertRaisesRegex(
                        ValueError, "thread count|one environment"
                    ):
                        self.summary.assemble_summary(paths, MANIFEST)

    def test_rejects_physical_threads_that_differ_from_baseline_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            paths = []
            for index, source in enumerate(original):
                cell = json.loads(source.read_text())
                environment = cell["environment"]
                environment["cpu_count_physical"] = 8
                if environment["openmp_threads"] != 1:
                    environment["openmp_threads"] = 8
                    environment["omp_num_threads"] = "8"
                path = Path(directory) / f"eight-core-{index}.json"
                path.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "frozen artifact pin"):
                self.summary.assemble_summary(paths, MANIFEST)

    def test_rejects_unstable_native_advance_samples(self):
        def make_unstable(cell):
            advance = cell["measurements"]["advance"]
            raw = [float(value) for value in range(1, 16)]
            middle = median(raw)
            advance.update(
                raw_seconds=raw,
                median_seconds=middle,
                p95_seconds=self.summary._percentile95(raw),
                population_stdev_seconds=pstdev(raw),
                relative_mad=(median(abs(value - middle) for value in raw) / middle),
                steps_per_second=(self.contract["steps_per_repeat"] / middle),
            )

        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            paths = list(original)
            paths[0] = self._mutated_input(
                directory, original[0], make_unstable, "unstable"
            )
            with self.assertRaisesRegex(ValueError, "relative-MAD limit"):
                self.summary.assemble_summary(paths, MANIFEST)

    def test_rejects_unfrozen_relative_mad_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = self._write_inputs(directory)
            manifest = copy.deepcopy(self.manifest)
            manifest["performance_gates"]["cpu_acceptance"]["statistics"][
                "max_relative_mad"
            ] = 0.99
            manifest_path = directory / "weakened-manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "not frozen"):
                self.summary.assemble_summary(paths, manifest_path)

    def test_rejects_inconsistent_native_accounting(self):
        mutations = {
            "rss-peak": lambda cell: cell["memory"].__setitem__("peak_rss_bytes", 1),
            "updater-bytes": lambda cell: cell["updaters"][0].__setitem__(
                "live_updater_bytes", 1
            ),
            "throughput": lambda cell: cell["measurements"]["advance"].__setitem__(
                "cells_per_second", 1e99
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            for suffix, mutation in mutations.items():
                with self.subTest(suffix=suffix):
                    paths = list(original)
                    paths[0] = self._mutated_input(
                        directory, original[0], mutation, suffix
                    )
                    with self.assertRaisesRegex(
                        ValueError, "RSS|updater|timing contract"
                    ):
                        self.summary.assemble_summary(paths, MANIFEST)

    def test_rejects_signed_zero_environment_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            paths = list(original)
            paths[0] = self._mutated_input(
                directory,
                original[0],
                lambda cell: cell["environment"].__setitem__("memory_bytes", -0.0),
                "negative-zero",
            )
            with self.assertRaisesRegex(ValueError, "environment identity"):
                self.summary.assemble_summary(paths, MANIFEST)

    def test_rejects_ansi_instead_of_sanitizing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            paths = list(original)
            paths[0] = self._mutated_input(
                directory,
                original[0],
                lambda cell: cell["environment"].__setitem__(
                    "cxx_version", "\x1b[31mc++ test\x1b[0m"
                ),
                "ansi",
            )
            with self.assertRaisesRegex(ValueError, "ANSI"):
                self.summary.assemble_summary(paths, MANIFEST)

    def test_rejects_duplicate_json_keys_and_inconsistent_raw_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            duplicate_key = Path(directory) / "duplicate-key.json"
            duplicate_key.write_text('{"schema_version": 2, "schema_version": 2}\n')
            paths = list(original)
            paths[0] = duplicate_key
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.summary.assemble_summary(paths, MANIFEST)

            paths = list(original)
            paths[0] = self._mutated_input(
                directory,
                original[0],
                lambda cell: cell["measurements"]["advance"].__setitem__(
                    "median_seconds", math.inf
                ),
                "statistics",
            )
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                self.summary.assemble_summary(paths, MANIFEST)

    def test_rejects_ansi_keys_and_boolean_transfer_samples(self):
        mutations = {
            "ansi-key": lambda cell: cell["updaters"][0].__setitem__(
                "\x1b[31mcells", 10
            ),
            "boolean-transfer": lambda cell: cell["measurements"][
                "host_to_device_transfer"
            ]["raw_seconds"].__setitem__(0, False),
        }
        with tempfile.TemporaryDirectory() as directory:
            original = self._write_inputs(directory)
            for suffix, mutation in mutations.items():
                with self.subTest(suffix=suffix):
                    paths = list(original)
                    paths[0] = self._mutated_input(
                        directory, original[0], mutation, suffix
                    )
                    with self.assertRaisesRegex(ValueError, "ANSI|transfer"):
                        self.summary.assemble_summary(paths, MANIFEST)


if __name__ == "__main__":
    unittest.main()
