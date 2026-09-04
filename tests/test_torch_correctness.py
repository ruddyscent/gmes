import copy
import json
import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import gmes
import gmes.torch_fdtd
from benchmarks import native_oracle, torch_correctness, torch_tuning


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

    @staticmethod
    def _rewrite_candidate(directory, candidate, label, mutate):
        output = Path(directory) / f"{label}.npz"
        with np.load(candidate, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
            metadata = native_oracle.read_metadata(archive)
        mutate(arrays, metadata)
        backend = metadata["backend_metadata"]
        torch_keys = sorted(name for name in arrays if name.startswith("torch/"))
        source_keys = sorted(
            name
            for name in arrays
            if len(name.split("/")) > 2 and name.split("/")[2] == "source"
        )
        backend["torch_arrays"] = {
            name: torch_correctness._array_descriptor(arrays[name])
            for name in torch_keys
        }
        backend["torch_array_bytes"] = sum(
            descriptor["size_bytes"] for descriptor in backend["torch_arrays"].values()
        )
        backend["source_arrays"] = {
            name: torch_correctness._array_descriptor(arrays[name])
            for name in source_keys
        }
        metadata["archive_array_bytes"] = sum(
            value.nbytes
            for name, value in arrays.items()
            if name != "metadata.json" and not name.startswith("torch/")
        )
        arrays["metadata.json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(output, **arrays)
        return output

    @staticmethod
    def _write_runtime_receipt(
        directory,
        manifest,
        evidence,
        candidate_paths,
        *,
        label="runtime",
    ):
        candidates = {}
        modes = []
        for path in candidate_paths:
            with torch_correctness._open_bounded_npz(path) as archive:
                metadata = native_oracle.read_metadata(archive)
            name = metadata["workload"]["name"]
            candidates[name] = {
                "case": name,
                "sha256": torch_correctness._sha256(path),
                "size_bytes": Path(path).stat().st_size,
            }
            modes.append(torch_correctness._runtime_mode(metadata))
        if not modes or any(mode != modes[0] for mode in modes[1:]):
            raise AssertionError("test receipt candidates must have one runtime mode")
        ordered = [
            candidates[case["name"]]
            for group in ("correctness", "physical_checks")
            for case in manifest[group]
        ]
        receipt = {
            "schema_version": 1,
            "kind": torch_correctness.RUNTIME_RECEIPT_KIND,
            "final_sha": evidence["candidate_git_commit"],
            "manifest_sha256": torch_correctness.TRUSTED_MANIFEST_SHA256,
            "workflow": {
                "repository": "ruddyscent/gmes",
                "run_id": 123,
                "run_attempt": 1,
                "job_id": 456,
                "job_name": f"correctness-{modes[0]['device']}-{modes[0]['graph_mode']}",
            },
            "profiler_witness": {
                "name": f"{label}-profiler.json",
                "sha256": "d" * 64,
                "size_bytes": 1,
                "media_type": "application/json",
            },
            "runtime_mode": modes[0],
            "candidate_archives": ordered,
        }
        path = Path(directory) / f"{label}-runtime-receipt.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return path

    @staticmethod
    def _insert_unindexed_local_record(path):
        payload = b"unindexed"
        name = b"unindexed-private.npy"
        crc = zlib.crc32(payload)
        local_record = (
            struct.pack(
                "<4s5H3L2H",
                b"PK\x03\x04",
                20,
                0,
                0,
                0,
                0,
                crc,
                len(payload),
                len(payload),
                len(name),
                0,
            )
            + name
            + payload
        )
        raw = path.read_bytes()
        eocd = raw.rfind(b"PK\x05\x06")
        central_offset = struct.unpack_from("<L", raw, eocd + 16)[0]
        rewritten = bytearray(
            raw[:central_offset] + local_record + raw[central_offset:]
        )
        struct.pack_into(
            "<L",
            rewritten,
            eocd + len(local_record) + 16,
            central_offset + len(local_record),
        )
        path.write_bytes(rewritten)

    def test_native_step_zero_produces_complete_torch_archives(self):
        cases = (
            "dummy",
            "upml",
            "drude-4",
            "lorentz-4",
            "dcp-ade",
            "dcp-plrc-bloch",
            "dcp-rc",
            "dm2-1",
            "cpml-bloch",
            "mixed-2d",
            "overlapping-sources",
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

    def test_float32_tfsf_long_capture_uses_strict_float64_auxiliary(self):
        capture_steps = [1, 2, 5, 20, 100]
        manifest = self._small_manifest(("tfsf-transparent",))
        manifest["reference"]["capture_steps"] = capture_steps
        manifest["correctness"][0]["capture_steps"] = capture_steps
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory,
                manifest,
                "tfsf-transparent",
                precision="float32",
            )
            result = torch_correctness.compare_torch_archives(
                reference,
                candidate,
                manifest,
                include_tolerances=True,
            )
            self.assertTrue(result["passed"], result["failures"])
            with np.load(candidate, allow_pickle=False) as archive:
                metadata = native_oracle.read_metadata(archive)
                self.assertEqual(
                    metadata["backend_metadata"]["auxiliary_precisions"],
                    ["float64"],
                )
                for step in ("0", *(str(value) for value in capture_steps)):
                    with self.subTest(step=step):
                        auxiliary = metadata["steps"][step]["sources"]["auxiliary"][0]
                        self.assertEqual(
                            auxiliary["backend_metadata"]["precision"], "float64"
                        )
                        main_prefix = f"torch/step/{step}/state"
                        auxiliary_prefix = f"torch/step/{step}/auxiliary/0/state"
                        self.assertEqual(
                            archive[f"{main_prefix}/source_time"].dtype,
                            np.dtype("float32"),
                        )
                        self.assertEqual(
                            archive[f"{auxiliary_prefix}/source_time"].dtype,
                            np.dtype("float64"),
                        )
                        for prefix in (main_prefix, auxiliary_prefix):
                            count = archive[f"{prefix}/step_count"]
                            source_time = archive[f"{prefix}/source_time"]
                            time_step = archive[f"{prefix}/time_step"]
                            expected_time = np.multiply(
                                count.astype(source_time.dtype),
                                time_step,
                                dtype=source_time.dtype,
                            )
                            self.assertTrue(
                                np.array_equal(source_time, expected_time), prefix
                            )
                        for component in ("Ex", "Hy"):
                            key = (
                                f"step/{step}/source_aux/"
                                f"0-TotalFieldScatteredField/field/{component}"
                            )
                            self.assertEqual(archive[key].dtype, np.dtype("float64"))

            tolerances = {
                record["key"]: record for record in result["tolerance_results"]
            }
            self.assertEqual(
                {
                    name: tolerances[
                        "step/100/source_aux/" "0-TotalFieldScatteredField/field/Hy"
                    ][name]
                    for name in ("rtol", "atol", "scope")
                },
                {
                    "rtol": 2e-12,
                    "atol": 2e-13,
                    "scope": "strategies/dielectric,pml/float64",
                },
            )
            self.assertEqual(
                {
                    name: tolerances["step/100/field/Hy"][name]
                    for name in ("rtol", "atol", "scope")
                },
                {
                    "rtol": 5e-5,
                    "atol": 5e-6,
                    "scope": "strategies/dielectric,pml/float32",
                },
            )

    def test_auxiliary_precision_metadata_corruption_fails_closed(self):
        manifest = self._small_manifest(("tfsf-transparent",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory,
                manifest,
                "tfsf-transparent",
                precision="float32",
            )
            with np.load(candidate, allow_pickle=False) as archive:
                base_arrays = {name: archive[name].copy() for name in archive.files}
                base_metadata = native_oracle.read_metadata(archive)

            def assert_rejected(label, arrays, metadata, expected_error=None):
                corrupted = Path(directory) / f"corrupted-{label}.npz"
                torch_arrays = metadata["backend_metadata"]["torch_arrays"]
                metadata["backend_metadata"]["torch_array_bytes"] = sum(
                    descriptor["size_bytes"] for descriptor in torch_arrays.values()
                )
                arrays["metadata.json"] = np.asarray(
                    json.dumps(metadata, sort_keys=True)
                )
                np.savez_compressed(corrupted, **arrays)
                result = torch_correctness.compare_torch_archives(
                    reference, corrupted, manifest
                )
                self.assertFalse(result["passed"])
                self.assertEqual(
                    result["failures"][0]["key"], "candidate/archive-contract"
                )
                if expected_error is not None:
                    self.assertIn(expected_error, result["failures"][0]["error"])

            arrays = {name: value.copy() for name, value in base_arrays.items()}
            metadata = copy.deepcopy(base_metadata)
            metadata["backend_metadata"]["auxiliary_precisions"] = ["float32"]
            for step in ("0", "1"):
                metadata["steps"][step]["sources"]["auxiliary"][0]["backend_metadata"][
                    "precision"
                ] = "float32"
            assert_rejected("auxiliary-precision", arrays, metadata)

            arrays = {name: value.copy() for name, value in base_arrays.items()}
            key = "step/1/source_aux/0-TotalFieldScatteredField/field/Hy"
            arrays[key] = arrays[key].astype(np.float32)
            assert_rejected(
                "auxiliary-canonical-field",
                arrays,
                copy.deepcopy(base_metadata),
            )

            arrays = {name: value.copy() for name, value in base_arrays.items()}
            metadata = copy.deepcopy(base_metadata)
            key = "torch/step/1/auxiliary/0/state/hy"
            arrays[key] = arrays[key].astype(np.float32)
            metadata["backend_metadata"]["torch_arrays"][key] = (
                torch_correctness._array_descriptor(arrays[key])
            )
            assert_rejected(
                "auxiliary-raw-field",
                arrays,
                metadata,
                "Torch auxiliary raw field contract differs",
            )

            for dtype in (np.complex128, np.int64):
                arrays = {name: value.copy() for name, value in base_arrays.items()}
                metadata = copy.deepcopy(base_metadata)
                key = "torch/step/1/auxiliary/0/state/hy"
                arrays[key] = arrays[key].astype(dtype)
                metadata["backend_metadata"]["torch_arrays"][key] = (
                    torch_correctness._array_descriptor(arrays[key])
                )
                assert_rejected(
                    f"auxiliary-raw-field-{np.dtype(dtype).name}",
                    arrays,
                    metadata,
                    "Torch auxiliary raw field contract differs",
                )

            arrays = {name: value.copy() for name, value in base_arrays.items()}
            metadata = copy.deepcopy(base_metadata)
            prefix = "torch/step/1/auxiliary/0/state"
            for suffix in ("source_time", "time_step"):
                key = f"{prefix}/{suffix}"
                arrays[key] = arrays[key].astype(np.float32)
                metadata["backend_metadata"]["torch_arrays"][key] = (
                    torch_correctness._array_descriptor(arrays[key])
                )
            assert_rejected(
                "auxiliary-raw-clock",
                arrays,
                metadata,
                "Torch auxiliary raw precision differs",
            )

            arrays = {name: value.copy() for name, value in base_arrays.items()}
            metadata = copy.deepcopy(base_metadata)
            source_batch_root = "torch/step/1/sources/batches/"
            key = next(
                name
                for name in arrays
                if name.startswith(source_batch_root)
                and len(name.removeprefix(source_batch_root).split("/")) == 2
                and name.endswith("/weights")
            )
            arrays[key] = arrays[key].astype(np.float32)
            metadata["backend_metadata"]["torch_arrays"][key] = (
                torch_correctness._array_descriptor(arrays[key])
            )
            assert_rejected(
                "transparent-raw-weights",
                arrays,
                metadata,
                "Torch transparent raw precision differs",
            )

    def test_float32_gaussian_long_capture_uses_strict_float64_auxiliary(self):
        capture_steps = [1, 2, 5, 20, 100]
        manifest = self._small_manifest(("gaussian-auxiliary",))
        manifest["reference"]["capture_steps"] = capture_steps
        manifest["correctness"][0]["capture_steps"] = capture_steps
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory,
                manifest,
                "gaussian-auxiliary",
                precision="float32",
            )
            result = torch_correctness.compare_torch_archives(
                reference,
                candidate,
                manifest,
                include_tolerances=True,
            )
            self.assertTrue(result["passed"], result["failures"])
            with np.load(candidate, allow_pickle=False) as archive:
                metadata = native_oracle.read_metadata(archive)
                self.assertEqual(
                    metadata["backend_metadata"]["auxiliary_precisions"],
                    ["float64"],
                )
                for step in ("0", *(str(value) for value in capture_steps)):
                    main_time = archive[f"torch/step/{step}/state/source_time"]
                    auxiliary_prefix = f"torch/step/{step}/auxiliary/0/state"
                    auxiliary_time = archive[f"{auxiliary_prefix}/source_time"]
                    self.assertEqual(main_time.dtype, np.dtype("float32"))
                    self.assertEqual(auxiliary_time.dtype, np.dtype("float64"))
                    expected_auxiliary_time = np.multiply(
                        archive[f"{auxiliary_prefix}/step_count"].astype(
                            auxiliary_time.dtype
                        ),
                        archive[f"{auxiliary_prefix}/time_step"],
                        dtype=auxiliary_time.dtype,
                    )
                    self.assertTrue(
                        np.array_equal(auxiliary_time, expected_auxiliary_time)
                    )
            tolerances = {
                record["key"]: record for record in result["tolerance_results"]
            }
            self.assertEqual(
                {
                    name: tolerances["step/100/source_aux/0-GaussianBeam/field/Hy"][
                        name
                    ]
                    for name in ("rtol", "atol", "scope")
                },
                {
                    "rtol": 2e-12,
                    "atol": 1e-12,
                    "scope": "source_auxiliary/gaussian-auxiliary/float64",
                },
            )

    def test_gaussian_envelope_raw_state_removal_fails_closed(self):
        manifest = self._small_manifest(("gaussian-auxiliary",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory,
                manifest,
                "gaussian-auxiliary",
                precision="float32",
            )
            with np.load(candidate, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
                metadata = native_oracle.read_metadata(archive)
            source_batch_root = "torch/step/1/sources/batches/"
            envelope_key = next(
                name
                for name in arrays
                if name.startswith(source_batch_root)
                and len(name.removeprefix(source_batch_root).split("/")) == 2
                and name.endswith("/_envelope")
            )
            batch_prefix = envelope_key.removesuffix("/_envelope")
            torch_arrays = metadata["backend_metadata"]["torch_arrays"]
            for name in (
                "_envelope_step",
                "_envelope_step_offset",
                "_envelope",
            ):
                key = f"{batch_prefix}/{name}"
                arrays.pop(key)
                torch_arrays.pop(key)
            metadata["backend_metadata"]["torch_array_bytes"] = sum(
                descriptor["size_bytes"] for descriptor in torch_arrays.values()
            )
            arrays["metadata.json"] = np.asarray(json.dumps(metadata, sort_keys=True))
            corrupted = Path(directory) / "gaussian-envelope-removed.npz"
            np.savez_compressed(corrupted, **arrays)
            result = torch_correctness.compare_torch_archives(
                reference, corrupted, manifest
            )
            self.assertFalse(result["passed"])
            self.assertEqual(result["failures"][0]["key"], "candidate/archive-contract")
            self.assertIn(
                "Torch transparent live buffer closure differs",
                result["failures"][0]["error"],
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

    def test_cuda_graph_execution_representation_corruption_fails_closed(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )
            corrupted = Path(directory) / "corrupted-cuda-graph-representation.npz"
            with np.load(candidate, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
                metadata = native_oracle.read_metadata(archive)
            self.assertEqual(
                metadata["backend_metadata"]["cuda_graph_execution_representation"],
                gmes.torch_fdtd.CUDA_GRAPH_EXECUTION_REPRESENTATION,
            )
            metadata["backend_metadata"][
                "cuda_graph_execution_representation"
            ] = "external-standard-regions+tampered"
            arrays["metadata.json"] = np.asarray(json.dumps(metadata, sort_keys=True))
            np.savez_compressed(corrupted, **arrays)
            result = torch_correctness.compare_torch_archives(
                reference, corrupted, manifest
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["key"], "candidate/archive-contract")
        self.assertIn("backend identity is invalid", result["failures"][0]["error"])

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

    def test_rehashed_logical_map_and_planner_tamper_fails_native_authority(self):
        manifest = self._small_manifest(("mixed-2d",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(directory, manifest, "mixed-2d")
            for suffix in ("material_ids", "underlying_ids"):
                with self.subTest(map=suffix):

                    def mutate(arrays, _metadata, suffix=suffix):
                        canonical_key = f"map/Ex/{suffix}"
                        planner_key = f"torch/planner/Ex/{suffix}"
                        canonical = arrays[canonical_key].copy()
                        planner = arrays[planner_key].copy()
                        canonical_flat = canonical.reshape(-1)
                        planner_flat = planner.reshape(-1)
                        values = sorted(set(int(value) for value in planner_flat))
                        location = next(
                            (
                                index
                                for index, value in enumerate(planner_flat)
                                if any(other != int(value) for other in values)
                            ),
                            None,
                        )
                        if location is None:
                            raise AssertionError(
                                f"mixed map has no mutable {suffix} entry"
                            )
                        replacement = next(
                            value
                            for value in values
                            if value != int(planner_flat[location])
                        )
                        canonical_flat[location] = replacement
                        planner_flat[location] = replacement
                        arrays[canonical_key] = canonical
                        arrays[planner_key] = planner
                        mirror_keys = {
                            f"torch/plan/{suffix}_ex",
                            "torch/step/0/state/plan/" f"{suffix}_ex",
                            "torch/step/1/state/plan/" f"{suffix}_ex",
                        }
                        for mirror_key in mirror_keys:
                            mirror = arrays[mirror_key].copy()
                            mirror.reshape(-1)[location] = replacement
                            arrays[mirror_key] = mirror

                    rewritten = self._rewrite_candidate(
                        directory,
                        candidate,
                        f"rehashed-map-{suffix}",
                        mutate,
                    )
                    result = torch_correctness.compare_torch_archives(
                        reference, rewritten, manifest
                    )
                    self.assertFalse(result["passed"])
                    self.assertIn(
                        "candidate/archive-contract",
                        [failure["key"] for failure in result["failures"]],
                    )
                    self.assertIn(
                        "immutable workload plan",
                        " ".join(
                            failure.get("error", "") for failure in result["failures"]
                        ),
                    )

    def test_rehashed_raw_planner_maps_fail_region_indirection(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )

            for suffix in ("material_ids", "underlying_ids"):
                with self.subTest(planner_map=suffix):

                    def mutate(arrays, metadata, suffix=suffix):
                        planner_key = f"torch/planner/Ex/{suffix}"
                        planner = arrays[planner_key].copy()
                        ownership = arrays["torch/planner/Ex/ownership"].reshape(-1)
                        location = int(np.flatnonzero(ownership >= 0)[0])
                        flat = planner.reshape(-1)
                        geometry_count = len(
                            metadata["backend_metadata"][
                                "actual_geometry_and_coefficients"
                            ]
                        )
                        candidates = (
                            range(geometry_count)
                            if suffix == "material_ids"
                            else (-1, *range(geometry_count))
                        )
                        replacement = next(
                            value
                            for value in candidates
                            if value != int(flat[location])
                        )
                        flat[location] = replacement
                        arrays[planner_key] = planner
                        for step in ("0", "1"):
                            mirror_key = f"torch/step/{step}/state/plan/{suffix}_ex"
                            mirror = arrays[mirror_key].copy()
                            mirror.reshape(-1)[location] = replacement
                            arrays[mirror_key] = mirror
                        mirror_key = f"torch/plan/{suffix}_ex"
                        mirror = arrays[mirror_key].copy()
                        mirror.reshape(-1)[location] = replacement
                        arrays[mirror_key] = mirror

                    rewritten = self._rewrite_candidate(
                        directory,
                        candidate,
                        f"rehashed-raw-planner-{suffix}",
                        mutate,
                    )
                    result = torch_correctness.compare_torch_archives(
                        reference, rewritten, manifest
                    )
                    self.assertFalse(result["passed"])
                    self.assertIn(
                        "candidate/archive-contract",
                        [failure["key"] for failure in result["failures"]],
                    )
                    self.assertIn(
                        "immutable workload plan",
                        " ".join(
                            failure.get("error", "") for failure in result["failures"]
                        ),
                    )

    def test_rehashed_planner_material_identity_and_coefficients_are_derived(self):
        manifest = self._small_manifest(("drude-1",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(directory, manifest, "drude-1")
            baseline = torch_correctness.compare_torch_archives(
                reference, candidate, manifest
            )
            self.assertTrue(baseline["passed"], baseline["failures"])
            with np.load(candidate, allow_pickle=False) as archive:
                coefficient_key = next(
                    key
                    for key in archive.files
                    if key.startswith("torch/planner/")
                    and key.endswith("-drude/coefficient_table")
                    and archive[key].size
                )

            def mutate_coefficient(arrays, _metadata):
                values = arrays[coefficient_key].copy()
                values.flat[0] += 1.0
                arrays[coefficient_key] = values

            def coherently_relabel_material(arrays, metadata):
                steps = (
                    "0",
                    *(str(value) for value in metadata["capture_steps"]),
                )
                for component in native_oracle.COMPONENT_NAMES:
                    component_key = component.lower()
                    planner_key = f"torch/planner/{component}/material_ids"
                    replacement = np.zeros_like(arrays[planner_key])
                    arrays[planner_key] = replacement
                    arrays[f"torch/plan/material_ids_{component_key}"] = (
                        replacement.copy()
                    )
                    for step in steps:
                        arrays[
                            f"torch/step/{step}/state/plan/"
                            f"material_ids_{component_key}"
                        ] = replacement.copy()
                    prefix = f"torch/planner/{component}/bucket/"
                    for key in tuple(arrays):
                        if key.startswith(prefix) and key.endswith("/region_keys"):
                            region_keys = arrays[key].copy()
                            region_keys[:, 0] = 0
                            arrays[key] = region_keys

            def mutate_live_coefficient(arrays, _metadata):
                key = next(
                    name
                    for name in arrays
                    if name.startswith("torch/plan/bucket_")
                    and name.endswith("_coefficients")
                    and arrays[name].size
                )
                values = arrays[key].copy()
                values.flat[0] += 1.0
                arrays[key] = values

            for label, mutate in (
                ("coefficient-table", mutate_coefficient),
                ("coherent-material-relabel", coherently_relabel_material),
                ("live-plan-coefficient", mutate_live_coefficient),
            ):
                with self.subTest(attack=label):
                    rewritten = self._rewrite_candidate(
                        directory, candidate, f"rehashed-{label}", mutate
                    )
                    result = torch_correctness.compare_torch_archives(
                        reference, rewritten, manifest
                    )
                    self.assertFalse(result["passed"])
                    self.assertIn(
                        "immutable workload plan",
                        " ".join(
                            failure.get("error", "") for failure in result["failures"]
                        ),
                    )

    def test_transparent_source_values_shapes_and_finiteness_are_derived(self):
        manifest = self._small_manifest(("tfsf-transparent",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "tfsf-transparent"
            )
            baseline = torch_correctness.compare_torch_archives(
                reference, candidate, manifest
            )
            self.assertTrue(baseline["passed"], baseline["failures"])
            with np.load(candidate, allow_pickle=False) as archive:
                batch_root = next(
                    key.removesuffix("/weights")
                    for key in archive.files
                    if key.startswith("torch/step/1/sources/batches/")
                    and key.endswith("/weights")
                )
                canonical_values = next(
                    key
                    for key in archive.files
                    if key.startswith("step/1/source/")
                    and key.endswith("/values")
                    and "Transparent" in key
                )

            def mutate_packed_values(arrays, _metadata):
                values = arrays[canonical_values].copy()
                values.flat[0] ^= np.uint64(1)
                arrays[canonical_values] = values

            def mutate_weight(arrays, _metadata):
                key = f"{batch_root}/weights"
                values = arrays[key].copy()
                values.flat[0] += 1.0
                arrays[key] = values

            def mutate_live_values(arrays, _metadata):
                key = f"{batch_root}/_values"
                values = arrays[key].copy()
                values.flat[0] = np.nextafter(values.flat[0], np.inf)
                arrays[key] = values

            def mutate_outer_values(arrays, _metadata):
                key = f"{batch_root}/_outer_values"
                values = arrays[key].copy()
                values.flat[0] += 1.0
                arrays[key] = values

            def reshape_weight(arrays, _metadata):
                key = f"{batch_root}/weights"
                arrays[key] = arrays[key].reshape(-1).copy()

            def nonfinite_weight(arrays, _metadata):
                key = f"{batch_root}/weights"
                values = arrays[key].copy()
                values.flat[0] = np.nan
                arrays[key] = values

            def nonfinite_sample_values(arrays, _metadata):
                key = f"{batch_root}/_sample_values"
                values = arrays[key].copy()
                values.flat[0] = np.inf
                arrays[key] = values

            for label, mutate in (
                ("packed-values", mutate_packed_values),
                ("weight", mutate_weight),
                ("live-values", mutate_live_values),
                ("outer-values", mutate_outer_values),
                ("wrong-weight-shape", reshape_weight),
                ("nan-weight", nonfinite_weight),
                ("inf-sample-values", nonfinite_sample_values),
            ):
                with self.subTest(attack=label):
                    rewritten = self._rewrite_candidate(
                        directory, candidate, f"rehashed-transparent-{label}", mutate
                    )
                    result = torch_correctness.compare_torch_archives(
                        reference, rewritten, manifest
                    )
                    self.assertFalse(result["passed"])
                    self.assertEqual(
                        result["failures"][0]["key"],
                        "candidate/archive-contract",
                    )

    def test_native_transparent_values_are_rederived_from_workload(self):
        manifest = self._small_manifest(("tfsf-transparent",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "tfsf-transparent"
            )
            with np.load(reference, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            source_key = next(
                key
                for key in arrays
                if key.startswith("step/1/source/")
                and key.endswith("/values")
                and "Transparent" in key
            )
            values = arrays[source_key].copy()
            values.flat[0] += 1.0
            arrays[source_key] = values
            rewritten_reference = Path(directory) / "rehashed-native-source.npz"
            np.savez_compressed(rewritten_reference, **arrays)

            def bind_rewritten_reference(_arrays, metadata):
                metadata["backend_metadata"]["input_archive"] = {
                    "sha256": torch_correctness._sha256(rewritten_reference),
                    "size_bytes": rewritten_reference.stat().st_size,
                    "media_type": "application/x-npz",
                    "prefix": "step/0",
                }

            rebound_candidate = self._rewrite_candidate(
                directory,
                candidate,
                "rebound-native-source",
                bind_rewritten_reference,
            )
            result = torch_correctness.compare_torch_archives(
                rewritten_reference, rebound_candidate, manifest
            )
            self.assertFalse(result["passed"])
            self.assertIn(
                "reference/source-contract",
                [failure["key"] for failure in result["failures"]],
            )

    def test_rehashed_point_source_arrays_require_values_and_exact_closure(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )
            source_key = "step/1/source/Ex/0-PointSourceEx/values"

            def flip_values(arrays, _metadata):
                values = arrays[source_key].copy()
                values.flat[-1] ^= np.uint64(1)
                arrays[source_key] = values

            def remove_values(arrays, _metadata):
                arrays.pop(source_key)

            def add_unexpected_key(arrays, _metadata):
                arrays[f"{source_key}/unexpected"] = np.asarray([0], dtype=np.uint8)

            for label, mutate in (
                ("packed-values", flip_values),
                ("missing-values", remove_values),
                ("unexpected-key", add_unexpected_key),
            ):
                with self.subTest(source_tamper=label):
                    rewritten = self._rewrite_candidate(
                        directory, candidate, f"rehashed-{label}", mutate
                    )
                    result = torch_correctness.compare_torch_archives(
                        reference, rewritten, manifest
                    )
                    self.assertFalse(result["passed"])

    def test_rehashed_point_source_live_buffers_are_closed_and_semantic(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )
            root = "torch/step/1/sources/batches/0"
            canonical_root = "step/1/source/Ex/0-PointSourceEx"

            def delete_complete_batch(arrays, _metadata):
                for step in ("0", "1"):
                    batch_root = f"torch/step/{step}/sources/batches/0"
                    for name in torch_correctness._POINT_SOURCE_LIVE_ARRAYS:
                        arrays.pop(f"{batch_root}/{name}")

            def mutate_target(arrays, _metadata):
                key = f"{root}/overwrite_targets"
                targets = arrays[key].copy()
                targets[0] = (int(targets[0]) + 1) % arrays["step/1/field/Ex"].size
                arrays[key] = targets

            def mutate_model(arrays, _metadata):
                key = f"{root}/overwrite_models"
                models = arrays[key].copy()
                models[0] = 3
                arrays[key] = models

            def mutate_parameters(arrays, _metadata):
                key = f"{root}/overwrite_parameters"
                parameters = arrays[key].copy()
                parameters[0, 0] += 0.125
                arrays[key] = parameters

            def mutate_amplitude(arrays, _metadata):
                key = f"{root}/overwrite_amplitudes"
                amplitudes = arrays[key].copy()
                amplitudes[0] *= 2
                arrays[key] = amplitudes

            def mutate_live_values(arrays, _metadata):
                key = f"{root}/_overwrite_values"
                values = arrays[key].copy()
                values.flat[0] += 1
                arrays[key] = values

            def mutate_update_mode(arrays, _metadata):
                for suffix in ("targets", "models", "parameters", "amplitudes"):
                    overwrite = f"{root}/overwrite_{suffix}"
                    additive = f"{root}/additive_{suffix}"
                    arrays[additive] = arrays[overwrite].copy()
                    arrays[overwrite] = arrays[overwrite][:0].copy()
                arrays[f"{root}/_additive_values"] = arrays[
                    f"{root}/_overwrite_values"
                ].copy()
                arrays[f"{root}/_overwrite_values"] = arrays[
                    f"{root}/_overwrite_values"
                ][:0].copy()
                packed = arrays[f"{canonical_root}/values"].view("<f8").copy()
                packed.reshape(-1, 9)[:, 0] = 1.0
                arrays[f"{canonical_root}/values"] = packed.view("<u8")

            def mutate_time(arrays, _metadata):
                canonical = arrays["step/1/time"].copy()
                canonical[0] += 1
                canonical[1] = canonical[0] * canonical[2]
                arrays["step/1/time"] = canonical
                count_key = "torch/step/1/state/step_count"
                source_time_key = "torch/step/1/state/source_time"
                arrays[count_key] += 1
                arrays[source_time_key] = np.asarray(
                    arrays[count_key] * arrays["torch/step/1/state/time_step"],
                    dtype=arrays[source_time_key].dtype,
                )

            mutations = (
                ("deleted-ten-buffer-batch", delete_complete_batch),
                ("target", mutate_target),
                ("model", mutate_model),
                ("parameters", mutate_parameters),
                ("amplitude", mutate_amplitude),
                ("live-values", mutate_live_values),
                ("update-mode", mutate_update_mode),
                ("time", mutate_time),
            )
            for label, mutate in mutations:
                with self.subTest(live_source_tamper=label):
                    rewritten = self._rewrite_candidate(
                        directory, candidate, f"rehashed-live-source-{label}", mutate
                    )
                    result = torch_correctness.compare_torch_archives(
                        reference, rewritten, manifest
                    )
                    self.assertFalse(result["passed"])

    def test_rehashed_coherent_point_source_rewrite_fails_workload_binding(self):
        manifest = self._small_manifest(("dcp-plrc-bloch",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "dcp-plrc-bloch"
            )

            def mutate(arrays, metadata):
                frequency = 0.45
                parameters = np.asarray(
                    [frequency, 0.0, 0.0, np.inf, 5.0 / frequency, 0.0],
                    dtype=np.float64,
                )
                for step in ("0", "1"):
                    live_root = f"torch/step/{step}/sources/batches/0"
                    parameters_key = f"{live_root}/overwrite_parameters"
                    arrays[parameters_key] = parameters.reshape(1, 6).copy()
                    time = float(arrays[f"step/{step}/time"][1])
                    time -= 0.5 * float(arrays[f"step/{step}/time"][2])
                    values = torch_correctness._point_source_model_values(
                        arrays[f"{live_root}/overwrite_models"],
                        arrays[parameters_key],
                        time,
                        metadata["backend_metadata"]["paired_real"],
                        np.dtype(metadata["backend_metadata"]["precision"]),
                    )
                    values *= arrays[f"{live_root}/overwrite_amplitudes"][:, None]
                    arrays[f"{live_root}/_overwrite_values"] = values
                    canonical_key = f"step/{step}/source/Ex/0-PointSourceEx/values"
                    packed = arrays[canonical_key].view("<f8").copy().reshape(-1, 9)
                    packed[:, 2:8] = parameters
                    arrays[canonical_key] = packed.reshape(-1).view("<u8")

            rewritten = self._rewrite_candidate(
                directory,
                candidate,
                "rehashed-coherent-point-source",
                mutate,
            )
            result = torch_correctness.compare_torch_archives(
                reference, rewritten, manifest
            )
            self.assertFalse(result["passed"])
            self.assertIn(
                "semantics differ from expected source",
                " ".join(failure.get("error", "") for failure in result["failures"]),
            )

    def test_rehashed_auxiliary_live_state_has_exact_closure_and_shape(self):
        manifest = self._small_manifest(("tfsf-transparent",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "tfsf-transparent"
            )

            def delete_nested_point_source(arrays, _metadata):
                for step in ("0", "1"):
                    root = f"torch/step/{step}/auxiliary/0/sources/batches/0"
                    for name in torch_correctness._POINT_SOURCE_LIVE_ARRAYS:
                        arrays.pop(f"{root}/{name}")

            def reshape_inactive_field(arrays, metadata):
                active = set(
                    metadata["steps"]["1"]["sources"]["auxiliary"][0][
                        "backend_metadata"
                    ]["canonical_components"]
                )
                component = next(
                    name for name in native_oracle.COMPONENT_NAMES if name not in active
                )
                key = f"torch/step/1/auxiliary/0/state/{component.lower()}"
                arrays[key] = np.zeros((), dtype=arrays[key].dtype)

            def mutate_inactive_material_state(arrays, metadata):
                active = set(
                    metadata["steps"]["1"]["sources"]["auxiliary"][0][
                        "backend_metadata"
                    ]["canonical_components"]
                )
                component = next(
                    name for name in native_oracle.COMPONENT_NAMES if name not in active
                )
                key = f"torch/step/1/auxiliary/0/state/pml_{component.lower()}_0_state"
                values = arrays[key].copy()
                values.flat[0] = 1.0
                arrays[key] = values

            mutations = (
                ("nested-point-source-deleted", delete_nested_point_source),
                ("inactive-field-shape", reshape_inactive_field),
                ("inactive-material-state", mutate_inactive_material_state),
            )
            for label, mutate in mutations:
                with self.subTest(auxiliary_tamper=label):
                    rewritten = self._rewrite_candidate(
                        directory,
                        candidate,
                        f"rehashed-{label}",
                        mutate,
                    )
                    result = torch_correctness.compare_torch_archives(
                        reference, rewritten, manifest
                    )
                    self.assertFalse(result["passed"])

    def test_rehashed_live_main_fields_bind_real_and_paired_representations(self):
        cases = (
            ("stability-energy-dielectric", 0),
            ("dcp-plrc-bloch", 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            for ordinal, (name, channel) in enumerate(cases):
                with self.subTest(case=name, channel=channel):
                    case_directory = Path(directory) / str(ordinal)
                    case_directory.mkdir()
                    manifest = self._small_manifest((name,))
                    reference, candidate = self._capture_pair(
                        case_directory, manifest, name
                    )

                    def mutate(arrays, _metadata, channel=channel):
                        key = "torch/step/1/state/ex"
                        values = arrays[key].copy()
                        if values.ndim == arrays["step/1/field/Ex"].ndim:
                            values.flat[0] += 1
                        else:
                            values[..., channel].flat[0] += 1
                        arrays[key] = values

                    rewritten = self._rewrite_candidate(
                        case_directory,
                        candidate,
                        f"rehashed-live-field-{ordinal}",
                        mutate,
                    )
                    result = torch_correctness.compare_torch_archives(
                        reference, rewritten, manifest
                    )
                    self.assertFalse(result["passed"])
                    self.assertIn(
                        "live field differs from canonical field",
                        " ".join(
                            failure.get("error", "") for failure in result["failures"]
                        ),
                    )

    def test_rehashed_live_main_and_auxiliary_material_state_fail_closed(self):
        cases = (
            (
                "upml",
                "torch/step/1/state/",
                "pml_",
            ),
            (
                "drude-4",
                "torch/step/1/state/",
                "_current",
            ),
            (
                "lorentz-4",
                "torch/step/1/state/",
                "_current",
            ),
            (
                "dcp-ade",
                "torch/step/1/state/",
                "_pole_now",
            ),
            (
                "dcp-plrc-bloch",
                "torch/step/1/state/",
                "pole_state",
            ),
            (
                "dcp-rc",
                "torch/step/1/state/",
                "_pole_state",
            ),
            (
                "dm2-1",
                "torch/step/1/state/dm2_buckets/",
                "/u",
            ),
            (
                "tfsf-transparent",
                "torch/step/1/auxiliary/0/state/",
                "pml_",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for ordinal, (name, prefix, token) in enumerate(cases):
                with self.subTest(case=name):
                    case_directory = Path(directory) / str(ordinal)
                    case_directory.mkdir()
                    manifest = self._small_manifest((name,))
                    reference, candidate = self._capture_pair(
                        case_directory, manifest, name
                    )

                    def mutate(arrays, _metadata, prefix=prefix, token=token):
                        key = next(
                            key
                            for key in arrays
                            if key.startswith(prefix)
                            and token in key
                            and not any(
                                scratch in key
                                for scratch in ("scratch", "work", "previous")
                            )
                        )
                        values = arrays[key].copy()
                        if not values.size:
                            raise AssertionError(f"material state is empty: {key}")
                        values.flat[0] += 1
                        arrays[key] = values

                    rewritten = self._rewrite_candidate(
                        case_directory,
                        candidate,
                        f"rehashed-material-state-{ordinal}",
                        mutate,
                    )
                    result = torch_correctness.compare_torch_archives(
                        reference, rewritten, manifest
                    )
                    self.assertFalse(result["passed"])
                    self.assertIn(
                        "live material state differs from canonical state",
                        " ".join(
                            failure.get("error", "") for failure in result["failures"]
                        ),
                    )

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

    def test_runtime_receipt_schema_accepts_an_exact_singleton_binding(self):
        evidence = {
            "candidate_git_commit": "b" * 40,
            "candidate_git_status": "",
            "manifest_sha256": torch_correctness.TRUSTED_MANIFEST_SHA256,
        }
        runtime_mode = {
            "device": "cuda:0",
            "precision": "float64",
            "graph_mode": "eager",
            "compile_policy": "eager",
            "compile_mode": "default",
        }
        candidate_archives = [
            {
                "case": "single-gpu-3d",
                "sha256": "c" * 64,
                "size_bytes": 123,
            }
        ]
        receipt = {
            "schema_version": 1,
            "kind": torch_correctness.RUNTIME_RECEIPT_KIND,
            "final_sha": evidence["candidate_git_commit"],
            "manifest_sha256": evidence["manifest_sha256"],
            "workflow": {
                "repository": "ruddyscent/gmes",
                "run_id": 123,
                "run_attempt": 1,
                "job_id": 456,
                "job_name": "single-gpu-3d-runtime",
            },
            "profiler_witness": {
                "name": "single-gpu-3d-profiler.json",
                "sha256": "d" * 64,
                "size_bytes": 321,
                "media_type": "application/json",
            },
            "runtime_mode": runtime_mode,
            "candidate_archives": candidate_archives,
        }
        self.assertTrue(
            torch_correctness.runtime_publication_receipt_complete(
                receipt,
                self.manifest,
                evidence,
                runtime_mode,
                candidate_archives,
            )
        )
        mismatched_archives = copy.deepcopy(candidate_archives)
        mismatched_archives[0]["sha256"] = "e" * 64
        self.assertFalse(
            torch_correctness.runtime_publication_receipt_complete(
                receipt,
                self.manifest,
                evidence,
                runtime_mode,
                mismatched_archives,
            )
        )

    def test_cpu_binding_rejects_rehashed_noncontract_runtime_archives(self):
        manifest = self._small_manifest(("stability-energy-dielectric",))
        evidence = {
            "candidate_git_commit": "b" * 40,
            "candidate_git_status": "",
            "manifest_sha256": torch_correctness.TRUSTED_MANIFEST_SHA256,
            "solver_abi": gmes.torch_fdtd.TORCH_SOLVER_ABI,
        }

        def load_rebuilt_index(
            directory, trusted_directory, reference, candidate, label
        ):
            receipt = self._write_runtime_receipt(
                directory,
                manifest,
                evidence,
                [candidate],
                label=label,
            )
            index = torch_correctness.build_correctness_evidence_index(
                [reference],
                [candidate],
                manifest,
                evidence,
                descriptor_root=directory,
                runtime_receipt=receipt,
            )
            index_path = Path(directory) / f"{label}-index.json"
            index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "outside descriptor root"):
                torch_correctness.load_correctness_evidence_index(
                    index_path,
                    manifest,
                    evidence,
                    descriptor_root=directory,
                    runtime_receipt=receipt,
                )
            hardlink_receipt = Path(trusted_directory) / f"{label}-hardlink.json"
            os.link(receipt, hardlink_receipt)
            with self.assertRaisesRegex(ValueError, "distinct external file"):
                torch_correctness.load_correctness_evidence_index(
                    index_path,
                    manifest,
                    evidence,
                    descriptor_root=directory,
                    runtime_receipt=hardlink_receipt,
                )
            external_receipt = Path(trusted_directory) / receipt.name
            external_receipt.write_bytes(receipt.read_bytes())
            loaded = torch_correctness.load_correctness_evidence_index(
                index_path,
                manifest,
                evidence,
                descriptor_root=directory,
                runtime_receipt=external_receipt,
            )
            receipt_document = torch_correctness._load_bounded_json(
                receipt, "test runtime receipt", require_canonical=True
            )
            self.assertEqual(
                loaded["artifacts"][0]["candidate"]["sha256"],
                torch_correctness._sha256(candidate),
            )
            self.assertEqual(
                loaded["source_artifact"]["sha256"],
                torch_correctness._sha256(index_path),
            )
            self.assertTrue(
                torch_correctness.correctness_binding_complete(
                    loaded,
                    manifest,
                    evidence,
                    runtime_receipt=receipt_document,
                    require_source_artifact=True,
                )
            )
            self.assertFalse(
                torch_correctness.correctness_binding_complete(
                    loaded,
                    manifest,
                    evidence,
                    runtime_receipt=None,
                )
            )
            externally_loaded = torch_correctness.load_correctness_evidence_index(
                index_path,
                manifest,
                evidence,
                descriptor_root=directory,
                runtime_receipt=external_receipt,
            )
            substituted_receipt = json.loads(receipt.read_text())
            substituted_receipt["workflow"]["job_id"] += 1
            receipt.write_text(
                json.dumps(substituted_receipt, indent=2, sort_keys=True) + "\n"
            )
            refreshed_index = copy.deepcopy(index)
            refreshed_index["runtime_receipt"] = (
                torch_correctness._runtime_receipt_descriptor(receipt, directory)
            )
            index_path.write_text(
                json.dumps(refreshed_index, indent=2, sort_keys=True) + "\n"
            )
            with self.assertRaisesRegex(
                ValueError, "trusted runtime publication receipt bytes differ"
            ):
                torch_correctness.load_correctness_evidence_index(
                    index_path,
                    manifest,
                    evidence,
                    descriptor_root=directory,
                    runtime_receipt=external_receipt,
                )
            self.assertEqual(externally_loaded, loaded)
            receipt.write_text(
                json.dumps(receipt_document, indent=2, sort_keys=True) + "\n"
            )
            index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
            return loaded, receipt_document

        with tempfile.TemporaryDirectory() as workspace:
            directory = Path(workspace) / "bundle"
            trusted_directory = Path(workspace) / "trusted"
            directory.mkdir()
            trusted_directory.mkdir()
            reference, candidate = self._capture_pair(
                directory, manifest, "stability-energy-dielectric"
            )
            loaded, loaded_receipt = load_rebuilt_index(
                directory,
                trusted_directory,
                reference,
                candidate,
                "cpu-float64-eager",
            )
            self.assertTrue(
                torch_tuning._cpu_correctness_binding_complete(
                    loaded, manifest, evidence, loaded_receipt
                )
            )

            runtime_mutations = {
                "wrong-device": {
                    "device": "cuda:0",
                    "resolved_device": "cuda:0",
                },
                "wrong-graph": {
                    "graph_mode": "graph",
                    "compile_policy": "compile",
                },
                "wrong-compile-mode": {
                    "device": "cuda:0",
                    "resolved_device": "cuda:0",
                    "graph_mode": "graph",
                    "compile_policy": "compile",
                    "compile_mode": "reduce-overhead",
                    "cuda_graph_regions": ["electric_half", "magnetic_half"],
                },
            }
            for label, replacement in runtime_mutations.items():
                with self.subTest(runtime=label):
                    rewritten = self._rewrite_candidate(
                        directory,
                        candidate,
                        label,
                        lambda _arrays, metadata, values=replacement: metadata[
                            "backend_metadata"
                        ].update(values),
                    )
                    wrong_mode, wrong_receipt = load_rebuilt_index(
                        directory,
                        trusted_directory,
                        reference,
                        rewritten,
                        label,
                    )
                    self.assertFalse(
                        torch_tuning._cpu_correctness_binding_complete(
                            wrong_mode, manifest, evidence, wrong_receipt
                        )
                    )

            float32_directory = Path(directory) / "float32"
            float32_directory.mkdir()
            float32_reference, float32_candidate = self._capture_pair(
                float32_directory,
                manifest,
                "stability-energy-dielectric",
                precision="float32",
            )
            float32_index, float32_receipt = load_rebuilt_index(
                directory,
                trusted_directory,
                float32_reference,
                float32_candidate,
                "cpu-float32-eager",
            )
            self.assertFalse(
                torch_tuning._cpu_correctness_binding_complete(
                    float32_index, manifest, evidence, float32_receipt
                )
            )

    def test_index_revalidates_full_correctness_and_physical_matrix(self):
        manifest = self._small_manifest(
            ("dcp-plrc-bloch",), ("stability-energy-dielectric",)
        )
        evidence = {
            "candidate_git_commit": "b" * 40,
            "candidate_git_status": "",
            "manifest_sha256": torch_correctness.TRUSTED_MANIFEST_SHA256,
            "solver_abi": gmes.torch_fdtd.TORCH_SOLVER_ABI,
        }
        with tempfile.TemporaryDirectory() as workspace:
            directory = Path(workspace) / "bundle"
            trusted_directory = Path(workspace) / "trusted"
            directory.mkdir()
            trusted_directory.mkdir()
            pairs = [
                self._capture_pair(directory, manifest, name)
                for name in (
                    "dcp-plrc-bloch",
                    "stability-energy-dielectric",
                )
            ]
            receipt = self._write_runtime_receipt(
                directory,
                manifest,
                evidence,
                [pair[1] for pair in pairs],
            )
            index = torch_correctness.build_correctness_evidence_index(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                manifest,
                evidence,
                descriptor_root=directory,
                runtime_receipt=receipt,
            )
            receipt_document = torch_correctness._load_bounded_json(
                receipt, "test runtime receipt", require_canonical=True
            )
            self.assertTrue(
                torch_correctness.correctness_binding_complete(
                    index,
                    manifest,
                    evidence,
                    runtime_receipt=receipt_document,
                )
            )
            index_path = Path(directory) / "correctness-index.json"
            index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
            external_receipt = trusted_directory / receipt.name
            external_receipt.write_bytes(receipt.read_bytes())
            loaded = torch_correctness.load_correctness_evidence_index(
                index_path,
                manifest,
                evidence,
                descriptor_root=directory,
                runtime_receipt=external_receipt,
            )
            self.assertTrue(
                torch_correctness.correctness_binding_complete(
                    loaded,
                    manifest,
                    evidence,
                    runtime_receipt=receipt_document,
                )
            )
            escaped = copy.deepcopy(index)
            escaped["artifacts"][0]["reference"]["path"] = "../outside.npz"
            index_path.write_text(json.dumps(escaped))
            with self.assertRaisesRegex(
                ValueError, "differs from recomputed|dot or empty segment"
            ):
                torch_correctness.load_correctness_evidence_index(
                    index_path,
                    manifest,
                    evidence,
                    descriptor_root=directory,
                    runtime_receipt=external_receipt,
                )
            index["suite_acceptance"]["passed"] = False
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "differs from recomputed"):
                torch_correctness.load_correctness_evidence_index(
                    index_path,
                    manifest,
                    evidence,
                    descriptor_root=directory,
                    runtime_receipt=external_receipt,
                )

    def test_runtime_receipt_prevents_coherent_runtime_relabel(self):
        manifest = self._small_manifest(("stability-energy-dielectric",))
        evidence = {
            "candidate_git_commit": "b" * 40,
            "candidate_git_status": "",
            "manifest_sha256": torch_correctness.TRUSTED_MANIFEST_SHA256,
            "solver_abi": gmes.torch_fdtd.TORCH_SOLVER_ABI,
        }
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "stability-energy-dielectric"
            )
            receipt = self._write_runtime_receipt(
                directory, manifest, evidence, [candidate]
            )
            replacements = {
                "coherent-cuda-eager-relabel": {
                    "device": "cuda:0",
                    "resolved_device": "cuda:0",
                },
                "coherent-cuda-graph-relabel": {
                    "device": "cuda:0",
                    "resolved_device": "cuda:0",
                    "graph_mode": "graph",
                    "compile_policy": "compile",
                    "compile_mode": "reduce-overhead",
                    "cuda_graph_regions": ["electric_half", "magnetic_half"],
                },
            }
            for label, replacement in replacements.items():
                with self.subTest(label=label):
                    relabeled = self._rewrite_candidate(
                        directory,
                        candidate,
                        label,
                        lambda _arrays, metadata, value=replacement: metadata[
                            "backend_metadata"
                        ].update(value),
                    )
                    with self.assertRaisesRegex(
                        ValueError, "runtime publication receipt differs"
                    ):
                        torch_correctness.build_correctness_evidence_index(
                            [reference],
                            [relabeled],
                            manifest,
                            evidence,
                            descriptor_root=directory,
                            runtime_receipt=receipt,
                        )

    def test_npz_preflight_rejects_hidden_records_and_allocation_failures(self):
        manifest = self._small_manifest(("stability-energy-dielectric",))
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = self._capture_pair(
                directory, manifest, "stability-energy-dielectric"
            )
            original_digest = torch_correctness._sha256(candidate)
            self._insert_unindexed_local_record(candidate)
            refreshed_digest = torch_correctness._sha256(candidate)
            self.assertNotEqual(refreshed_digest, original_digest)
            hidden = torch_correctness.compare_torch_archives(
                reference, candidate, manifest
            )
            self.assertFalse(hidden["passed"])
            self.assertIn("indexed local records", hidden["failures"][0]["error"])

            clean_directory = Path(directory) / "clean"
            clean_directory.mkdir()
            clean_reference, clean_candidate = self._capture_pair(
                clean_directory,
                manifest,
                "stability-energy-dielectric",
            )
            real_load = np.load

            def fail_candidate(handle, *args, **kwargs):
                if getattr(handle, "name", None) == str(clean_candidate):
                    raise MemoryError("synthetic allocation failure")
                return real_load(handle, *args, **kwargs)

            with patch.object(torch_correctness.np, "load", side_effect=fail_candidate):
                allocation = torch_correctness.compare_torch_archives(
                    clean_reference, clean_candidate, manifest
                )
            self.assertFalse(allocation["passed"])
            self.assertEqual(allocation["failures"][0]["key"], "archive/container")

    def test_npz_preflight_enforces_each_resource_bound_before_numpy_load(self):
        manifest = self._small_manifest(("stability-energy-dielectric",))
        with tempfile.TemporaryDirectory() as directory:
            reference, _candidate = self._capture_pair(
                directory, manifest, "stability-energy-dielectric"
            )
            limits = (
                "MAX_CORRECTNESS_NPZ_BYTES",
                "MAX_CORRECTNESS_NPZ_MEMBERS",
                "MAX_CORRECTNESS_NPY_HEADER_BYTES",
                "MAX_CORRECTNESS_NPY_PAYLOAD_BYTES",
                "MAX_CORRECTNESS_TOTAL_ARRAY_BYTES",
            )
            for name in limits:
                with (
                    self.subTest(limit=name),
                    patch.object(torch_correctness, name, 1),
                    patch.object(
                        torch_correctness.np,
                        "load",
                        side_effect=AssertionError("np.load must not be reached"),
                    ) as loader,
                    self.assertRaises(ValueError),
                ):
                    with torch_correctness._open_bounded_npz(reference):
                        pass
                loader.assert_not_called()

    def test_standalone_json_loaders_enforce_byte_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text('{"candidate_git_commit":"' + "a" * 40 + '"}')
            with patch.object(torch_correctness, "MAX_CORRECTNESS_JSON_BYTES", 8):
                with self.assertRaisesRegex(ValueError, "JSON byte bound"):
                    torch_correctness._load_candidate_evidence(path)
                with self.assertRaisesRegex(ValueError, "JSON byte bound"):
                    torch_correctness.load_correctness_evidence_index(
                        path,
                        {},
                        {},
                        descriptor_root=directory,
                        runtime_receipt=path,
                    )
                with self.assertRaisesRegex(ValueError, "JSON byte bound"):
                    torch_correctness._load_trusted_manifest(
                        torch_correctness.DEFAULT_MANIFEST
                    )

    def test_trusted_manifest_loader_requires_exact_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            trusted_copy = Path(directory) / "native_oracle_workloads.json"
            trusted_copy.write_bytes(torch_correctness.DEFAULT_MANIFEST.read_bytes())
            manifest, digest = torch_correctness._load_trusted_manifest(trusted_copy)
            self.assertEqual(manifest, self.manifest)
            self.assertEqual(digest, torch_correctness.TRUSTED_MANIFEST_SHA256)

            altered = json.loads(trusted_copy.read_text())
            altered["schema_version"] += 1
            trusted_copy.write_text(json.dumps(altered, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                torch_correctness._load_trusted_manifest(trusted_copy)

    def test_dm2_complex_container_uses_float64_state_tolerance(self):
        actual = native_oracle.tolerance_for_key(
            self.manifest,
            "torch",
            "step/1/state/Ex/0-Dm2/values",
            "complex128",
        )
        self.assertEqual(actual, self.manifest["tolerances"]["torch"]["dm2"]["float64"])

    def test_dm2_float32_500_step_tolerance_is_exact_and_fails_outside_bound(self):
        key = "step/500/state/Ex/0-Dm2/values"
        tolerance = torch_correctness._manifest_tolerance(
            self.manifest,
            key,
            "float32",
            {"Dm2"},
            "ziolkowski-dm2",
        )
        self.assertEqual(
            tolerance,
            {
                "rtol": 6e-4,
                "atol": 3e-6,
                "scope": "strategies/dm2/float32",
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.npz"
            candidate = Path(directory) / "candidate.npz"
            np.savez_compressed(reference, **{key: np.asarray([10.0])})
            np.savez_compressed(
                candidate,
                **{key: np.asarray([10.0 * (1.0 + 6.1e-4)])},
            )
            reference_metadata = {
                "workload": {"name": "ziolkowski-dm2"},
                "geometry_and_coefficients": {},
            }
            candidate_metadata = {
                "backend_metadata": {
                    "precision": "float32",
                    "input_archive": {
                        "sha256": torch_correctness._sha256(reference),
                        "size_bytes": reference.stat().st_size,
                    },
                    "input_step_zero_contract": {
                        "arrays": {},
                        "array_bytes": 0,
                    },
                },
                "geometry_and_coefficients": {},
            }
            with (
                patch.object(
                    native_oracle,
                    "_validate_archive",
                    return_value=reference_metadata,
                ),
                patch.object(
                    torch_correctness,
                    "_validate_torch_candidate_archive",
                    return_value=candidate_metadata,
                ),
                patch.object(
                    torch_correctness,
                    "_source_topology_matches",
                    return_value=True,
                ),
                patch.object(
                    torch_correctness,
                    "_material_topology_matches",
                    return_value=True,
                ),
                patch.object(
                    torch_correctness,
                    "_reference_strategies",
                    return_value={"Dm2"},
                ),
                patch.object(torch_correctness, "COMPONENT_NAMES", ()),
            ):
                result = torch_correctness.compare_torch_archives(
                    reference,
                    candidate,
                    self.manifest,
                    include_tolerances=True,
                )

        self.assertFalse(result["passed"])
        failure = next(item for item in result["failures"] if item["key"] == key)
        self.assertEqual(
            {name: failure[name] for name in ("rtol", "atol", "scope")},
            tolerance,
        )
        tolerance_result = next(
            item for item in result["tolerance_results"] if item["key"] == key
        )
        self.assertEqual(
            {name: tolerance_result[name] for name in ("rtol", "atol", "scope")},
            tolerance,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_float32_ziolkowski_500_step_eager_and_graph_archives(self):
        spec = native_oracle.find_case(self.manifest, "ziolkowski-dm2")
        key = "step/500/state/Ex/0-Dm2/values"
        expected_tolerance = {
            "rtol": 6e-4,
            "atol": 3e-6,
            "scope": "strategies/dm2/float32",
        }
        observer_commit = self.manifest["reference"]["observer_commit"]
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "ziolkowski-dm2-native.npz"
            with patch.object(
                native_oracle,
                "_checkout_provenance",
                side_effect=self._provenance(observer_commit),
            ):
                native_oracle.capture_case(spec, self.manifest, reference)

            with np.load(reference, allow_pickle=False) as archive:
                reference_metadata = native_oracle.read_metadata(archive)
                self.assertIn(key, archive.files)
            self.assertEqual(reference_metadata["capture_steps"], [100, 500])

            modes = (("eager", "default"), ("graph", "reduce-overhead"))
            for graph_mode, compile_mode in modes:
                with self.subTest(
                    graph_mode=graph_mode,
                    compile_mode=compile_mode,
                ):
                    torch._dynamo.reset()
                    candidate = (
                        Path(directory)
                        / f"ziolkowski-dm2-cuda-float32-{graph_mode}.npz"
                    )
                    with patch.object(
                        native_oracle,
                        "_checkout_provenance",
                        side_effect=self._provenance("b" * 40),
                    ):
                        torch_correctness.capture_torch_candidate(
                            reference,
                            self.manifest,
                            candidate,
                            device="cuda:0",
                            precision="float32",
                            graph_mode=graph_mode,
                            compile_mode=compile_mode,
                        )
                    result = torch_correctness.compare_torch_archives(
                        reference,
                        candidate,
                        self.manifest,
                        include_tolerances=True,
                    )
                    self.assertTrue(result["passed"], result["failures"])
                    self.assertEqual(result["failures"], [])
                    records = {
                        item["key"]: item for item in result["tolerance_results"]
                    }
                    self.assertEqual(
                        {
                            name: records[key][name]
                            for name in ("rtol", "atol", "scope")
                        },
                        expected_tolerance,
                    )
                    with np.load(candidate, allow_pickle=False) as archive:
                        metadata = native_oracle.read_metadata(archive)
                        self.assertIn(key, archive.files)
                    backend = metadata["backend_metadata"]
                    self.assertEqual(metadata["capture_steps"], [100, 500])
                    self.assertEqual(backend["device"], "cuda:0")
                    self.assertEqual(backend["precision"], "float32")
                    self.assertEqual(backend["graph_mode"], graph_mode)
                    self.assertEqual(backend["compile_mode"], compile_mode)

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
