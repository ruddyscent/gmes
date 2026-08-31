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


if __name__ == "__main__":
    unittest.main()
