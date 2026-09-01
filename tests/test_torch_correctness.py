import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import gmes
import gmes.torch_fdtd
from benchmarks import native_oracle, torch_correctness


class TorchCorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = native_oracle.load_manifest()

    @staticmethod
    def _provenance(commit):
        def clean(checkout, source):
            return {
                "checkout": str(Path(checkout).resolve()),
                "commit": commit,
                "git_status": "",
                "clean": True,
                "source": str(Path(source).resolve()),
                "source_sha256": "a" * 64,
            }

        return clean

    def _small_manifest(self, correctness, physical=()):
        manifest = copy.deepcopy(self.manifest)
        manifest["reference"]["capture_steps"] = [1]

        def small(name):
            spec = copy.deepcopy(native_oracle.find_case(manifest, name))
            spec["capture_steps"] = [1]
            if spec["recipe"] != "mixed":
                spec.update(size=[2, 2, 2], resolution=2)
            return spec

        manifest["correctness"] = [small(name) for name in correctness]
        manifest["physical_checks"] = [small(name) for name in physical]
        return manifest

    def _capture_pair(self, directory, manifest, name, **runtime):
        spec = native_oracle.find_case(manifest, name)
        reference = Path(directory) / f"{name}-native.npz"
        candidate = Path(directory) / f"{name}-torch.npz"
        observer_commit = manifest["reference"]["observer_commit"]
        with patch.object(
            native_oracle,
            "_checkout_provenance",
            side_effect=self._provenance(observer_commit),
        ):
            native_oracle.capture_case(spec, manifest, reference)
        with patch.object(
            native_oracle,
            "_checkout_provenance",
            side_effect=self._provenance("b" * 40),
        ):
            torch_correctness.capture_torch_candidate(
                reference, manifest, candidate, **runtime
            )
        return reference, candidate

    @staticmethod
    def _corrupt_archive(directory, candidate, key):
        output = Path(directory) / "corrupted-torch.npz"
        with np.load(candidate, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        value = arrays[key].copy()
        if not value.size:
            raise AssertionError(f"cannot corrupt empty archive array {key}")
        if np.issubdtype(value.dtype, np.integer):
            value.flat[0] ^= 1
        else:
            value.flat[0] += 1
        arrays[key] = value
        np.savez_compressed(output, **arrays)
        return output

    def test_native_step_zero_produces_complete_torch_archives(self):
        cases = (
            "dummy",
            "dcp-plrc-bloch",
            "dm2-1",
            "cpml-bloch",
            "mixed-2d",
            "tfsf-transparent",
            "gaussian-auxiliary",
        )
        manifest = self._small_manifest(cases)
        with tempfile.TemporaryDirectory() as directory:
            for name in cases:
                with self.subTest(case=name):
                    reference, candidate = self._capture_pair(directory, manifest, name)
                    result = torch_correctness.compare_torch_archives(
                        reference, candidate, manifest
                    )
                    self.assertEqual(result, {"passed": True, "failures": []})
                    with np.load(candidate, allow_pickle=False) as archive:
                        metadata = native_oracle.read_metadata(archive)
                    self.assertEqual(metadata["backend"], "torch")
                    self.assertEqual(
                        metadata["backend_metadata"]["input_archive"]["prefix"],
                        "step/0",
                    )

    def test_compiled_dummy_long_capture_covers_topology_and_tolerance(self):
        capture_steps = [1, 2, 5, 20, 100]
        manifest = self._small_manifest(("dummy",))
        manifest["reference"]["capture_steps"] = capture_steps
        manifest["correctness"][0]["capture_steps"] = capture_steps
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory,
                manifest,
                "dummy",
                graph_mode="graph",
                compile_mode="default",
                precision="float64",
            )
            result = torch_correctness.compare_torch_archives(
                reference,
                candidate,
                manifest,
                include_tolerances=True,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["failures"], [])
            with np.load(candidate, allow_pickle=False) as archive:
                metadata = native_oracle.read_metadata(archive)
            records = {
                record["component"]: record
                for record in metadata["steps"]["0"]["materials"]
            }
            self.assertEqual(set(records), set(native_oracle.COMPONENT_NAMES))
            for component in native_oracle.COMPONENT_NAMES:
                with self.subTest(component=component):
                    self.assertEqual(records[component]["strategies"], ["Dummy"])
                    self.assertEqual(
                        records[component]["cells"],
                        int(np.prod(metadata["maps"][component]["shape"])),
                    )
            tolerances = {
                record["key"]: record for record in result["tolerance_results"]
            }
            expected = manifest["tolerances"]["torch"]["dielectric"]["float64"]
            for key in (
                "step/100/field/Ex",
                "step/100/physical/spectrum/Ex",
            ):
                with self.subTest(key=key):
                    self.assertEqual(
                        {
                            name: tolerances[key][name]
                            for name in ("rtol", "atol", "scope")
                        },
                        {
                            **expected,
                            "scope": "dummy-source-numerics/dielectric/float64",
                        },
                    )

    def test_candidate_capture_is_independent_of_legacy_native_state(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )
            with (
                patch.object(
                    native_oracle,
                    "_checkout_provenance",
                    side_effect=self._provenance("b" * 40),
                ),
                patch.object(
                    native_oracle,
                    "build_simulation",
                    side_effect=AssertionError("legacy native simulation used"),
                ),
                patch.object(
                    native_oracle,
                    "_source_records",
                    side_effect=AssertionError("legacy source adapter used"),
                ),
                patch.object(
                    gmes,
                    "FDTD",
                    side_effect=AssertionError("legacy FDTD used"),
                ),
            ):
                torch_correctness.capture_torch_candidate(
                    reference, manifest, candidate
                )
            self.assertEqual(
                torch_correctness.compare_torch_archives(
                    reference, candidate, manifest
                ),
                {"passed": True, "failures": []},
            )
            with np.load(candidate, allow_pickle=False) as archive:
                metadata = native_oracle.read_metadata(archive)
            backend = metadata["backend_metadata"]
            self.assertEqual(backend["logical_map_source"], "live-torch-plan")
            self.assertNotIn("path", backend["input_archive"])
            self.assertTrue(backend["torch_arrays"])

    def test_copied_reference_is_not_a_torch_candidate(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, _candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )
            result = torch_correctness.compare_torch_archives(
                reference, reference, manifest
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["key"], "candidate/archive-contract")

    def test_raw_planner_corruption_fails_closed(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )
            corrupted = self._corrupt_archive(
                directory, candidate, "torch/planner/Ex/material_ids"
            )
            result = torch_correctness.compare_torch_archives(
                reference, corrupted, manifest
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["key"], "candidate/archive-contract")

    def test_source_plan_corruption_fails_closed(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )
            source_key = "step/0/source/Ex/0-PointSourceEx/values"
            corrupted = self._corrupt_archive(directory, candidate, source_key)
            result = torch_correctness.compare_torch_archives(
                reference, corrupted, manifest
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["key"], "candidate/archive-contract")

    def test_live_source_clock_corruption_fails_closed(self):
        manifest = self._small_manifest(("gaussian-auxiliary",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "gaussian-auxiliary"
            )
            keys = (
                "torch/step/1/state/source_time",
                "torch/step/1/auxiliary/0/state/source_time",
            )
            for ordinal, key in enumerate(keys):
                with self.subTest(key=key):
                    output = Path(directory) / f"corrupted-clock-{ordinal}.npz"
                    with np.load(candidate, allow_pickle=False) as archive:
                        arrays = {name: archive[name].copy() for name in archive.files}
                        metadata = native_oracle.read_metadata(archive)
                    arrays[key] = arrays[key] + np.asarray(1, dtype=arrays[key].dtype)
                    metadata["backend_metadata"]["torch_arrays"][key] = (
                        torch_correctness._array_descriptor(arrays[key])
                    )
                    arrays["metadata.json"] = np.asarray(
                        json.dumps(metadata, sort_keys=True)
                    )
                    np.savez_compressed(output, **arrays)
                    result = torch_correctness.compare_torch_archives(
                        reference, output, manifest
                    )
                    self.assertFalse(result["passed"])
                    self.assertEqual(
                        result["failures"][0]["key"],
                        "candidate/archive-contract",
                    )

            output = Path(directory) / "corrupted-consistent-clock.npz"
            with np.load(candidate, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
                metadata = native_oracle.read_metadata(archive)
            prefix = "torch/step/1/state"
            count_key = f"{prefix}/step_count"
            source_time_key = f"{prefix}/source_time"
            time_step_key = f"{prefix}/time_step"
            arrays[count_key] += 1
            arrays[source_time_key] = np.multiply(
                arrays[count_key].astype(arrays[source_time_key].dtype),
                arrays[time_step_key],
                dtype=arrays[source_time_key].dtype,
            )
            for key in (count_key, source_time_key):
                metadata["backend_metadata"]["torch_arrays"][key] = (
                    torch_correctness._array_descriptor(arrays[key])
                )
            arrays["metadata.json"] = np.asarray(json.dumps(metadata, sort_keys=True))
            np.savez_compressed(output, **arrays)
            result = torch_correctness.compare_torch_archives(
                reference, output, manifest
            )
            self.assertFalse(result["passed"])
            self.assertEqual(result["failures"][0]["key"], "candidate/archive-contract")

    def test_native_step_zero_descriptor_corruption_fails_closed(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )
            corrupted = Path(directory) / "corrupted-input-contract.npz"
            with np.load(candidate, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
                metadata = native_oracle.read_metadata(archive)
            metadata["backend_metadata"]["input_step_zero_contract"]["arrays"][
                "step/0/time"
            ]["sha256"] = ("0" * 64)
            arrays["metadata.json"] = np.asarray(json.dumps(metadata, sort_keys=True))
            np.savez_compressed(corrupted, **arrays)
            result = torch_correctness.compare_torch_archives(
                reference, corrupted, manifest
            )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["failures"][0]["key"],
            "candidate/input_step_zero_contract",
        )

    def test_invalid_npz_fails_closed(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, _candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )
            invalid = Path(directory) / "invalid.npz"
            invalid.write_bytes(b"not an NPZ archive")
            result = torch_correctness.compare_torch_archives(
                reference, invalid, manifest
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["key"], "archive/container")

    def test_runtime_modes_bind_precision_and_graph_execution(self):
        manifest = self._small_manifest(("stability-energy-dielectric",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory,
                manifest,
                "stability-energy-dielectric",
                precision="float32",
            )
            self.assertEqual(
                torch_correctness.compare_torch_archives(
                    reference, candidate, manifest
                ),
                {"passed": True, "failures": []},
            )
            with np.load(candidate, allow_pickle=False) as archive:
                metadata = native_oracle.read_metadata(archive)
            self.assertEqual(
                torch_correctness._runtime_mode(metadata),
                {
                    "device": "cpu",
                    "precision": "float32",
                    "graph_mode": "eager",
                    "compile_policy": "eager",
                    "compile_mode": "default",
                },
            )
        self.assertEqual(
            torch_correctness._runtime_contract(
                "cuda:0", "float32", "graph", "reduce-overhead"
            ),
            {
                "device": "cuda:0",
                "precision": "float32",
                "graph_mode": "graph",
                "compile_policy": "compile",
                "compile_mode": "reduce-overhead",
            },
        )
        with self.assertRaisesRegex(ValueError, "non-default compile mode"):
            torch_correctness._runtime_contract(
                "cuda:0", "float32", "eager", "reduce-overhead"
            )

    def test_index_revalidates_full_correctness_and_physical_matrix(self):
        manifest = self._small_manifest(
            ("dcp-plrc-bloch",), ("stability-energy-dielectric",)
        )
        evidence = {
            "candidate_git_commit": "b" * 40,
            "candidate_git_status": "",
            "manifest_sha256": "c" * 64,
            "solver_abi": gmes.torch_fdtd.TORCH_SOLVER_ABI,
        }
        with tempfile.TemporaryDirectory() as directory:
            pairs = [
                self._capture_pair(directory, manifest, name)
                for name in (
                    "dcp-plrc-bloch",
                    "stability-energy-dielectric",
                )
            ]
            index = torch_correctness.build_correctness_evidence_index(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                manifest,
                evidence,
                descriptor_root=directory,
            )
            self.assertTrue(
                torch_correctness.correctness_binding_complete(
                    index, manifest, evidence
                )
            )
            index_path = Path(directory) / "correctness-index.json"
            index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
            loaded = torch_correctness.load_correctness_evidence_index(
                index_path, manifest, evidence, descriptor_root=directory
            )
            self.assertTrue(
                torch_correctness.correctness_binding_complete(
                    loaded, manifest, evidence
                )
            )
            escaped = copy.deepcopy(index)
            escaped["artifacts"][0]["reference"]["path"] = "../outside.npz"
            index_path.write_text(json.dumps(escaped))
            with self.assertRaisesRegex(ValueError, "differs from recomputed"):
                torch_correctness.load_correctness_evidence_index(
                    index_path,
                    manifest,
                    evidence,
                    descriptor_root=directory,
                )
            index["suite_acceptance"]["passed"] = False
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "differs from recomputed"):
                torch_correctness.load_correctness_evidence_index(
                    index_path,
                    manifest,
                    evidence,
                    descriptor_root=directory,
                )

    def test_dm2_complex_container_uses_float64_state_tolerance(self):
        actual = native_oracle.tolerance_for_key(
            self.manifest,
            "torch",
            "step/1/state/Ex/0-Dm2/values",
            "complex128",
        )
        self.assertEqual(actual, self.manifest["tolerances"]["torch"]["dm2"]["float64"])

    def test_dummy_source_numerics_reuse_nondispersive_tolerance(self):
        for dtype, expected in self.manifest["tolerances"]["torch"][
            "dielectric"
        ].items():
            for key in (
                "step/100/field/Ex",
                "step/100/physical/spectrum/Ex",
            ):
                with self.subTest(dtype=dtype, key=key):
                    actual = torch_correctness._manifest_tolerance(
                        self.manifest,
                        key,
                        dtype,
                        {"Dummy"},
                        "dummy",
                    )
                    self.assertEqual(
                        actual,
                        {
                            **expected,
                            "scope": (f"dummy-source-numerics/dielectric/{dtype}"),
                        },
                    )
        exact = {"rtol": 0.0, "atol": 0.0, "scope": "exact/dummy"}
        for key in (
            "step/100/time",
            "step/100/state/Ex/0-Dummy/values",
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    torch_correctness._manifest_tolerance(
                        self.manifest,
                        key,
                        "float64",
                        {"Dummy"},
                        "dummy",
                    ),
                    exact,
                )
        self.assertEqual(
            torch_correctness._manifest_tolerance(
                self.manifest,
                "step/100/field/Ex",
                "float64",
                set(),
                "dummy",
            ),
            exact,
        )


if __name__ == "__main__":
    unittest.main()
