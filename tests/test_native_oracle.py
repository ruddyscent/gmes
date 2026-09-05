import copy
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class _UnitTestArchive(dict):
    """In-memory array archive for validator-unit tests only."""

    @property
    def files(self):
        return list(self)


def load_script(name):
    path = ROOT / "benchmarks" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = load_script("native_oracle.py")
        cls.isolated = load_script("run_isolated_oracle.py")
        cls.manifest = cls.oracle.load_manifest()

    def _write_manifest(self, directory, manifest):
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def _metadata_archive(self, directory, metadata, *, name="metadata.npz"):
        path = Path(directory) / name
        np.savez_compressed(path, **{"metadata.json": np.asarray(metadata)})
        return path

    @staticmethod
    def _arrays(**arrays):
        return _UnitTestArchive(arrays)

    def _source_record(self):
        return {
            "updaters": [
                {
                    "component": "Ex",
                    "native_type": "unit-test-source",
                    "cells": 1,
                    "state_values": 1,
                }
            ],
            "auxiliary": [],
        }

    def _material_record(self):
        return {
            "component": "Ex",
            "strategy": "dielectric",
            "strategies": ["dielectric"],
            "native_type": "unit-test-material",
            "cells": 1,
            "coverage": 1.0,
            "fragmentation_runs": 1,
            "fragmentation_ratio": 1.0,
            "state_values": 1,
            "state_nonzero_values": 1,
            "state_width": 1.0,
            "state_key": "step/1/state/Ex/0-dielectric/values",
            "state_bytes": 8,
            "plan_bytes": 1,
            "index_bytes": 1,
            "parameter_bytes": 1,
            "live_updater_bytes": 3,
            "plan_runs": 1,
            "bucket_signature": ["Ex", "dielectric", "unit-test-material", 1, 1],
        }

    def _write_synthetic_candidate(self, directory):
        """Write a parser-contract candidate, never a historical numeric authority."""
        manifest = copy.deepcopy(self.manifest)
        manifest["reference"]["capture_steps"] = [1]
        workload = copy.deepcopy(self.oracle.find_case(manifest, "drude-1"))
        workload["capture_steps"] = [1]
        manifest["correctness"] = [
            workload if case["name"] == workload["name"] else case
            for case in manifest["correctness"]
        ]
        arrays = {}
        maps = {}
        for component in self.oracle.COMPONENT_NAMES:
            arrays[f"map/{component}/material_ids"] = np.asarray([0], dtype=np.int64)
            arrays[f"map/{component}/underlying_ids"] = np.asarray([-1], dtype=np.int64)
            maps[component] = {
                "shape": [1, 1, 1],
                "dtype": "float64",
                "active_cells": 1,
                "material_regions": 1,
                "underlying_regions": 0,
            }
        steps = {}
        for step in (0, 1):
            fields = {}
            for component in self.oracle.COMPONENT_NAMES:
                arrays[f"step/{step}/field/{component}"] = np.ones((1, 1, 1))
                arrays[f"step/{step}/physical/spectrum/{component}"] = np.asarray(
                    [1.0 + 0.0j]
                )
            arrays[f"step/{step}/time"] = np.asarray(
                [
                    manifest["reference"]["precondition_steps"] + step,
                    (manifest["reference"]["precondition_steps"] + step) * 0.5,
                    0.5,
                ]
            )
            arrays[f"step/{step}/physical/summary"] = np.asarray(
                [6.0, 1.0, 6.0, 6.0, 1.0]
            )
            material = self._material_record()
            material["state_key"] = f"step/{step}/state/Ex/0-dielectric/values"
            material_prefix = f"step/{step}/state/Ex/0-dielectric"
            arrays[f"{material_prefix}/indices"] = np.zeros((1, 3), dtype=np.int64)
            arrays[f"{material_prefix}/values"] = np.asarray([1.0])
            source = self._source_record()
            source_prefix = f"step/{step}/source/Ex/0-unit-test-source"
            arrays[f"{source_prefix}/indices"] = np.zeros((1, 3), dtype=np.int64)
            arrays[f"{source_prefix}/values"] = np.asarray([0.25])
            steps[str(step)] = {
                "materials": [material],
                "sources": source,
                "physical": {
                    "energy": 6.0,
                    "maximum_abs_field": 1.0,
                    "boundary_low_energy": 6.0,
                    "boundary_high_energy": 6.0,
                    "finite": True,
                },
            }
        source = {
            "checkout": "/tmp/unit-test-candidate",
            "commit": "a" * 40,
            "git_status": "",
            "clean": True,
            "source": "/tmp/unit-test-candidate/observer.py",
            "source_sha256": "b" * 64,
        }
        metadata = {
            "schema_version": 2,
            "backend": "torch",
            "workload": workload,
            "reference": self.oracle._correctness_reference_contract(
                manifest["reference"]
            ),
            "capture_steps": [1],
            "input_state": {
                "archive_prefix": "step/0",
                "precondition_steps": manifest["reference"]["precondition_steps"],
                "relative_capture_steps": True,
            },
            "maps": maps,
            "steps": steps,
            "geometry_and_coefficients": [
                {"geometry": "unit-test-only", "material": {"kind": "synthetic"}}
            ],
            "active_cells": 1,
            "state_bytes": 8,
            "plan_bytes": 1,
            "index_bytes": 1,
            "parameter_bytes": 1,
            "live_updater_bytes": 3,
            "archive_array_bytes": sum(array.nbytes for array in arrays.values()),
            "nonzero_seed": True,
            "nonzero_persistent_state": True,
            "provenance": {"source": source, "controller": dict(source)},
            "reference_source": source["source"],
        }
        path = Path(directory) / "unit-test-candidate.npz"
        np.savez_compressed(
            path, **arrays, **{"metadata.json": np.asarray(json.dumps(metadata))}
        )
        return manifest, path

    @staticmethod
    def _rewrite_archive(
        source, destination, *, drop=(), mutate_arrays=None, mutate_metadata=None
    ):
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        metadata = json.loads(str(arrays.pop("metadata.json")))
        for key in drop:
            arrays.pop(key)
        if mutate_arrays is not None:
            mutate_arrays(arrays)
        if mutate_metadata is not None:
            mutate_metadata(metadata)
        arrays["metadata.json"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(destination, **arrays)

    def test_manifest_pins_historical_reference_and_performance_records(self):
        reference = self.manifest["reference"]
        self.assertEqual(self.manifest["schema_version"], 2)
        self.assertEqual(reference["tag"], "native-oracle-d87d25a")
        self.assertEqual(
            reference["commit"], "d87d25afd160d96b1fa0890cacecd90802448d57"
        )
        self.assertEqual(reference["observer_tag"], "native-oracle-observer-v6")
        self.assertEqual(
            reference["observer_commit"], "2d5810cebf610fa6384235d9771f4ac699c23fc5"
        )
        self.assertEqual(
            reference["performance_observer_tag"], "native-oracle-observer-v5"
        )
        self.assertEqual(
            reference["performance_summary_sha256"],
            "1c9bdce2717ba858fd03b2e40302a5b2d19a29920496f969e33aee36e34e1baa",
        )

    def test_manifest_rejects_weakened_cpu_allocation_contract(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["performance_gates"]["cpu_acceptance"]["allocation_contract"][
            "max_full_field_or_domain_clones"
        ] = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "max_full_field_or_domain_clones must be zero"
            ):
                self.oracle.load_manifest(self._write_manifest(directory, manifest))

    def test_manifest_rejects_redirected_or_weakened_cpu_timing_contract(self):
        mutations = (
            ("timing_reference", "root_commit", "0" * 40),
            (None, "max_individual_ratio", 1.10),
            ("statistics", "resamples", 1),
            ("statistics", "regression_ratio", 2.0),
        )
        for group, name, value in mutations:
            with (
                self.subTest(group=group, name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                manifest = copy.deepcopy(self.manifest)
                acceptance = manifest["performance_gates"]["cpu_acceptance"]
                (acceptance if group is None else acceptance[group])[name] = value
                with self.assertRaises(ValueError):
                    self.oracle.load_manifest(self._write_manifest(directory, manifest))

    def test_manifest_rejects_timing_runtime_identity_tampering(self):
        for identity in (
            None,
            {"schema_version": 1, "torch": "2.13.0+cu126", "cuda_runtime": None},
        ):
            with (
                self.subTest(identity=identity),
                tempfile.TemporaryDirectory() as directory,
            ):
                manifest = copy.deepcopy(self.manifest)
                manifest["performance_gates"]["cpu_acceptance"]["timing_reference"][
                    "timing_runtime_identity"
                ] = identity
                with self.assertRaisesRegex(ValueError, "frozen baseline"):
                    self.oracle.load_manifest(self._write_manifest(directory, manifest))

    def test_manifest_rejects_noncanonical_cpu_artifact_release_url(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["performance_gates"]["cpu_acceptance"]["timing_reference"][
            "slice_artifacts"
        ][0]["publication_url"] = (
            "https://github.com/ruddyscent/gmes/releases/download/latest/"
            "torch-cpu-baseline-one.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "artifact modes are invalid"):
                self.oracle.load_manifest(self._write_manifest(directory, manifest))

    def test_metadata_reader_rejects_duplicate_members_and_noncanonical_json(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            valid = self._metadata_archive(directory, '{"schema_version": 2}')
            with np.load(valid, allow_pickle=False) as archive:
                self.assertEqual(
                    self.oracle.read_metadata(archive), {"schema_version": 2}
                )
            duplicate = directory / "duplicate.npz"
            duplicate.write_bytes(valid.read_bytes())
            with zipfile.ZipFile(duplicate, "a") as archive:
                with self.assertWarns(UserWarning):
                    archive.writestr(
                        "metadata.json.npy", archive.read("metadata.json.npy")
                    )
            for name, encoded in (
                ("duplicate-key", '{"schema_version": 1, "schema_version": 2}'),
                ("nonfinite", '{"schema_version": NaN}'),
            ):
                path = self._metadata_archive(directory, encoded, name=f"{name}.npz")
                with np.load(path, allow_pickle=False) as archive:
                    with self.assertRaises(ValueError):
                        self.oracle.read_metadata(archive)
            with np.load(duplicate, allow_pickle=False) as archive:
                with self.assertRaisesRegex(ValueError, "exactly one metadata.json"):
                    self.oracle.read_metadata(archive)

    def test_archive_contract_rejects_duplicate_members_and_schema_tampering(self):
        metadata = {
            key: None
            for key in (
                "schema_version",
                "backend",
                "workload",
                "reference",
                "capture_steps",
                "input_state",
                "maps",
                "steps",
                "geometry_and_coefficients",
                "active_cells",
                "state_bytes",
                "plan_bytes",
                "index_bytes",
                "parameter_bytes",
                "live_updater_bytes",
                "archive_array_bytes",
                "nonzero_seed",
                "nonzero_persistent_state",
                "provenance",
                "reference_source",
            )
        }
        metadata["schema_version"] = 1
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = self._metadata_archive(directory, json.dumps(metadata))
            with np.load(path, allow_pickle=False) as archive:
                with self.assertRaisesRegex(
                    ValueError, "unsupported correctness archive schema"
                ):
                    self.oracle._validate_archive(archive, self.manifest, "candidate")
            duplicate = directory / "duplicate-member.npz"
            duplicate.write_bytes(path.read_bytes())
            with zipfile.ZipFile(duplicate, "a") as archive:
                archive.writestr("unit-test-only.npy", b"not an authority artifact")
                with self.assertWarns(UserWarning):
                    archive.writestr("unit-test-only.npy", b"not an authority artifact")
            with np.load(duplicate, allow_pickle=False) as archive:
                with self.assertRaisesRegex(ValueError, "array names must be unique"):
                    self.oracle._validate_archive(archive, self.manifest, "candidate")

    def test_synthetic_candidate_rejects_truncation_dimension_and_bytecount_tampering(
        self,
    ):
        """Exercise complete-archive checks with a temporary Torch candidate only."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, baseline = self._write_synthetic_candidate(directory)
            with np.load(baseline, allow_pickle=False) as archive:
                self.oracle._validate_archive(archive, manifest, "candidate")
            mutations = (
                (
                    "truncated-field",
                    {"drop": {"step/1/field/Ex"}},
                    "missing required array",
                ),
                (
                    "reshaped-map",
                    {
                        "mutate_arrays": lambda arrays: arrays.__setitem__(
                            "map/Ex/material_ids",
                            arrays["map/Ex/material_ids"].reshape(1, 1),
                        )
                    },
                    "map arrays are invalid",
                ),
                (
                    "bytecount",
                    {
                        "mutate_metadata": lambda metadata: metadata.__setitem__(
                            "archive_array_bytes", metadata["archive_array_bytes"] - 1
                        )
                    },
                    "archive_array_bytes is inaccurate",
                ),
            )
            for label, options, error in mutations:
                with self.subTest(label=label):
                    candidate = directory / f"{label}.npz"
                    self._rewrite_archive(baseline, candidate, **options)
                    with np.load(candidate, allow_pickle=False) as archive:
                        with self.assertRaisesRegex(ValueError, error):
                            self.oracle._validate_archive(
                                archive, manifest, "candidate"
                            )

    def test_source_and_material_state_contracts_reject_nonfinite_and_reshaped_arrays(
        self,
    ):
        source_prefix = "step/1/source/Ex/0-unit-test-source"
        material_prefix = "step/1/state/Ex/0-dielectric"
        source_arrays = {
            f"{source_prefix}/indices": np.zeros((1, 3), dtype=np.int64),
            f"{source_prefix}/values": np.asarray([0.25]),
        }
        material_arrays = {
            f"{material_prefix}/indices": np.zeros((1, 3), dtype=np.int64),
            f"{material_prefix}/values": np.asarray([1.0]),
        }
        for label, arrays, validator, arguments in (
            (
                "source-nonfinite",
                dict(
                    source_arrays, **{f"{source_prefix}/values": np.asarray([np.nan])}
                ),
                self.oracle._validate_source_arrays,
                (self._source_record(), 1),
            ),
            (
                "source-reshaped",
                dict(source_arrays, **{f"{source_prefix}/values": np.ones((1, 1))}),
                self.oracle._validate_source_arrays,
                (self._source_record(), 1),
            ),
            (
                "material-nonfinite",
                dict(
                    material_arrays,
                    **{f"{material_prefix}/values": np.asarray([np.inf])},
                ),
                self.oracle._validate_material_records,
                ([self._material_record()], 1, "state", {"Ex": (1,)}),
            ),
            (
                "material-reshaped",
                dict(material_arrays, **{f"{material_prefix}/values": np.ones((1, 1))}),
                self.oracle._validate_material_records,
                ([self._material_record()], 1, "state", {"Ex": (1,)}),
            ),
        ):
            with self.subTest(label=label):
                required = set()
                with self.assertRaisesRegex(
                    ValueError, "(source updater state|material state)"
                ):
                    validator(self._arrays(**arrays), *arguments, required)

    def test_provenance_contract_rejects_source_and_clean_state_mismatches(self):
        source = {
            "checkout": "/tmp/unit-test-source",
            "commit": "a" * 40,
            "git_status": "",
            "clean": True,
            "source": "/tmp/unit-test-source/benchmarks/observer.py",
            "source_sha256": "b" * 64,
        }
        metadata = {
            "provenance": {"source": source, "controller": dict(source)},
            "reference_source": source["source"],
        }
        for label, mutate in (
            (
                "source-outside-checkout",
                lambda value: value.update(source="/tmp/outside.py"),
            ),
            (
                "dirty-source",
                lambda value: value.update(git_status=" M observer.py", clean=False),
            ),
            ("reference-source", lambda value: None),
        ):
            with self.subTest(label=label):
                candidate = copy.deepcopy(metadata)
                if label == "reference-source":
                    candidate["reference_source"] = "/tmp/different.py"
                else:
                    mutate(candidate["provenance"]["source"])
                with self.assertRaises(ValueError):
                    self.oracle._validate_archive_provenance(
                        candidate, self.manifest, "candidate"
                    )

    def test_retained_readers_and_validators_remain_available(self):
        case = self.oracle.find_case(self.manifest, "drude-1")
        self.assertEqual(case["name"], "drude-1")
        self.assertEqual(
            self.oracle.tolerance_for_key(
                self.manifest, "torch", "step/1/material/drude/state", "float32"
            ),
            self.manifest["tolerances"]["torch"]["drude"]["float32"],
        )
        with self.assertRaises(ValueError):
            self.oracle.find_case(self.manifest, "not-a-workload")

    def test_candidate_checkout_rejects_native_execution(self):
        case = self.oracle.find_case(self.manifest, "drude-1")
        for operation, arguments in (
            (self.oracle.build_simulation, (case, object())),
            (self.oracle.capture_case, (case, self.manifest, Path("reference.npz"))),
            (self.oracle.benchmark_case, (case, self.manifest, 1, 0, 1)),
        ):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(RuntimeError, "retired"):
                    operation(*arguments)

    def test_cli_only_exposes_reader_and_comparator_operations(self):
        with patch.object(sys, "argv", ["native_oracle.py", "--help"]):
            with self.assertRaises(SystemExit) as result:
                self.oracle.main()
        self.assertEqual(result.exception.code, 0)

    def test_isolated_capture_requires_the_pinned_observer_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "observer"
            (checkout / "benchmarks").mkdir(parents=True)
            (checkout / "benchmarks" / "native_oracle.py").touch()
            output = Path(directory) / "reference.npz"
            with patch.object(
                self.isolated.subprocess,
                "run",
                return_value=SimpleNamespace(stdout="wrong-commit\n"),
            ):
                with self.assertRaisesRegex(ValueError, "pinned observer commit"):
                    self.isolated.run_capture(
                        checkout,
                        Path(sys.executable),
                        self.oracle.DEFAULT_MANIFEST,
                        "drude-1",
                        output,
                    )

    def test_isolated_capture_runs_the_pinned_checkout_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "observer"
            (checkout / "benchmarks").mkdir(parents=True)
            runner = checkout / "benchmarks" / "native_oracle.py"
            runner.touch()
            expected = self.manifest["reference"]["observer_commit"]
            with patch.object(
                self.isolated.subprocess,
                "run",
                side_effect=(
                    SimpleNamespace(stdout=f"{expected}\n"),
                    SimpleNamespace(stdout='{\n"schema_version": 2\n}\n'),
                ),
            ) as run:
                result = self.isolated.run_capture(
                    checkout,
                    Path(sys.executable),
                    self.oracle.DEFAULT_MANIFEST,
                    "drude-1",
                    Path(directory) / "reference.npz",
                )
            self.assertEqual(result["historical_observer_commit"], expected)
            self.assertEqual(result["command"][2], str(runner))
            self.assertEqual(run.call_count, 2)

    def test_isolated_environment_excludes_controller_import_paths(self):
        with patch.dict(
            self.isolated.os.environ,
            {"PYTHONPATH": "controller", "PYTHONHOME": "home", "VIRTUAL_ENV": "venv"},
            clear=False,
        ):
            environment = self.isolated.sanitized_environment()
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("VIRTUAL_ENV", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")


if __name__ == "__main__":
    unittest.main()
