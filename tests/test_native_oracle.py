import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
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

    @classmethod
    def clean_archive_provenance(cls, checkout, source):
        return {
            "checkout": str(Path(checkout).resolve()),
            "commit": cls.manifest["reference"]["observer_commit"],
            "git_status": "",
            "clean": True,
            "source": str(Path(source).resolve()),
            "source_sha256": "a" * 64,
        }

    @staticmethod
    def rewrite_archive(
        source, output, *, drop=(), mutate_arrays=None, mutate_metadata=None
    ):
        with np.load(source, allow_pickle=False) as archive:
            arrays = {
                key: np.array(archive[key], copy=True)
                for key in archive.files
                if key not in drop
            }
        if mutate_arrays is not None:
            mutate_arrays(arrays)
        metadata = json.loads(str(arrays["metadata.json"]))
        if mutate_metadata is not None:
            mutate_metadata(metadata)
        arrays["metadata.json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(output, **arrays)

    def capture_small_archive(self, directory):
        spec = dict(self.oracle.find_case(self.manifest, "drude-1"))
        spec.update(size=[2, 2, 2], resolution=2)
        manifest = json.loads(json.dumps(self.manifest))
        manifest["reference"]["capture_steps"] = [1]
        manifest["correctness"] = [
            spec if case["name"] == spec["name"] else case
            for case in manifest["correctness"]
        ]
        artifact = Path(directory) / "valid.npz"
        with patch.object(
            self.oracle,
            "_checkout_provenance",
            side_effect=self.clean_archive_provenance,
        ):
            self.oracle.capture_case(spec, manifest, artifact)
        return manifest, artifact

    def test_manifest_freezes_reference_matrix_and_gates(self):
        self.assertEqual(self.manifest["schema_version"], 2)
        reference = self.manifest["reference"]
        self.assertEqual(
            reference["commit"], "d87d25afd160d96b1fa0890cacecd90802448d57"
        )
        self.assertEqual(reference["tag"], "native-oracle-d87d25a")
        self.assertEqual(reference["observer_tag"], "native-oracle-observer-v4")
        self.assertEqual(
            reference["observer_commit"],
            "11ac12d4992ad85b19b414b837383b392904a131",
        )
        self.assertEqual(
            reference["performance_summary_sha256"],
            "a81b87e8cb2870d2fbbf1bbb7409c98d10780da30eb23cb5965fce39b9109fb6",
        )
        self.assertEqual(reference["field_initializer"], "native-affine-ramp-v1")
        self.assertEqual(reference["field_scale"], 1e-3)
        self.assertEqual(reference["performance_warmup_steps"], 5)
        self.assertEqual(reference["performance_steps_per_repeat"], 100)
        self.assertEqual(reference["performance_repetitions"], 15)
        self.assertEqual(reference["performance_profile_steps"], 5)
        self.assertEqual(reference["capture_steps"], [1, 2, 5, 20, 100])
        dielectric_tolerances = self.manifest["tolerances"]["torch"]["dielectric"]
        self.assertEqual(
            set(dielectric_tolerances),
            {"float32", "float64", "complex64", "complex128"},
        )
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
        self.assertEqual(
            gates["cpu_acceptance"]["cases"],
            [
                "cpu-crossover-2d",
                "cpu-crossover-3d",
                "cpu-large-2d",
                "cpu-large-3d",
                "bloch-2d",
                "bloch-3d",
            ],
        )
        acceptance = gates["cpu_acceptance"]
        self.assertEqual(acceptance["contract_id"], "cpu-acceptance-v2")
        timing_reference = acceptance["timing_reference"]
        self.assertEqual(timing_reference["backend"], "torch")
        self.assertEqual(
            timing_reference["root_commit"],
            "821c075b9328e02c3f3e5d16488a44b64ff08c04",
        )
        self.assertEqual(
            timing_reference["slice_artifacts"],
            [
                {
                    "thread_mode": "one",
                    "threads": 1,
                    "repository_path": (
                        "benchmarks/evidence/issue-123/" "torch-cpu-baseline-one.json"
                    ),
                    "size_bytes": 314181,
                    "sha256": (
                        "e6e765fcd0b0ff1fff1919ff06f95c155beed6ce2c51c3c58cf8dccfcca3387f"
                    ),
                },
                {
                    "thread_mode": "physical",
                    "threads": 4,
                    "repository_path": (
                        "benchmarks/evidence/issue-123/"
                        "torch-cpu-baseline-physical.json"
                    ),
                    "size_bytes": 314460,
                    "sha256": (
                        "27bc2f3f0a880b0faf25480d926f8b3885c33b7571f14bb47130880f2105fa9a"
                    ),
                },
            ],
        )
        self.assertEqual(
            timing_reference["legacy_evidence"],
            {
                "evidence_contract_id": "torch-cpu-acceptance-v7",
                "cpu_contract_id": "cpu-acceptance-v1",
                "manifest_sha256": (
                    "6d7fe084c558cf69771f0c3928bc9be96fc6bb5b55ba777d674151fbbe6cbe19"
                ),
                "runner_sha256": (
                    "fee6d418bb50729ddb26ff14e931a4f51bb8d2a92cb0ad537c2757846247a770"
                ),
                "solver_sha256": (
                    "9cd8decc801a6f9d93551c6e6f427afeff1c65e3092e54b03e5abe0a3e9192d5"
                ),
                "solver_abi": "torch-fdtd-regions-v8",
            },
        )
        self.assertEqual(acceptance["max_individual_ratio"], 1.05)
        self.assertEqual(acceptance["native_comparison"], "informational")
        self.assertEqual(
            acceptance["allocation_contract"],
            {
                "method": "reviewed-fixed-temporary-provenance-v1",
                "fixed_temporaries": {
                    "allowed": True,
                    "reviewed_provenance_required": True,
                },
                "max_net_live_growth_bytes": 0,
                "max_final_live_growth_bytes": 0,
                "rss_growth": "bounded",
                "max_full_field_or_domain_clones": 0,
                "public_upstream_issue_required": True,
            },
        )
        self.assertEqual(gates["cpu_acceptance"]["thread_modes"], ["one", "physical"])
        self.assertEqual(
            gates["cpu_acceptance"]["statistics"]["method"],
            "independent-stratified-bootstrap-log-geomean-v1",
        )
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
        mixed_3d = self.oracle.find_case(self.manifest, "mixed-3d")
        self.assertEqual(mixed_3d["size"], [96, 6, 4])
        self.assertEqual(mixed_3d["source"], "none")
        self.assertEqual(self.oracle.find_case(self.manifest, "dm2-4")["size"][2], 0)

    def test_manifest_rejects_weakened_cpu_allocation_contract(self):
        manifest = json.loads(json.dumps(self.manifest))
        allocation = manifest["performance_gates"]["cpu_acceptance"][
            "allocation_contract"
        ]
        allocation["max_full_field_or_domain_clones"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                ValueError, "max_full_field_or_domain_clones must be zero"
            ):
                self.oracle.load_manifest(path)

    def test_manifest_rejects_redirected_or_weakened_cpu_timing_contract(self):
        mutations = (
            ("timing_reference", "root_commit", "0" * 40),
            (None, "max_individual_ratio", 1.10),
            ("statistics", "resamples", 1),
            ("statistics", "regression_ratio", 2.0),
        )
        for group, name, value in mutations:
            manifest = json.loads(json.dumps(self.manifest))
            acceptance = manifest["performance_gates"]["cpu_acceptance"]
            target = acceptance if group is None else acceptance[group]
            target[name] = value
            with (
                self.subTest(group=group, name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "manifest.json"
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    self.oracle.load_manifest(path)

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

    def test_field_initializer_is_backend_neutral_and_canonical(self):
        shapes = {
            name: (2 + index % 2, 3, 1)
            for index, name in enumerate(reversed(self.oracle.COMPONENT_NAMES))
        }
        first = self.oracle.initial_field_values(shapes, 115, 1e-3)
        second = self.oracle.initial_field_values(shapes, 115, 1e-3)
        complex_values = self.oracle.initial_field_values(
            shapes,
            115,
            1e-3,
            complex_fields=True,
        )
        self.assertEqual(tuple(first), self.oracle.COMPONENT_NAMES)
        for name in self.oracle.COMPONENT_NAMES:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(first[name].shape, shapes[name])
            self.assertEqual(first[name].dtype, np.float64)
            self.assertEqual(complex_values[name].dtype, np.complex128)
            self.assertTrue(np.all(first[name] != 0))
            self.assertTrue(np.all(complex_values[name] != 0))

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
        manifest["correctness"] = [
            spec if case["name"] == spec["name"] else case
            for case in manifest["correctness"]
        ]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "reference.npz"
            with patch.object(
                self.oracle,
                "_checkout_provenance",
                side_effect=self.clean_archive_provenance,
            ):
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

    def test_compare_rejects_matching_truncated_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, artifact = self.capture_small_archive(directory)
            key = "step/1/field/Ex"
            reference = directory / "truncated-reference.npz"
            candidate = directory / "truncated-candidate.npz"
            self.rewrite_archive(artifact, reference, drop={key})
            self.rewrite_archive(artifact, candidate, drop={key})
            result = self.oracle.compare_archives(reference, candidate, manifest)
            self.assertFalse(result["passed"])
            self.assertTrue(
                all(
                    "archive-contract" in failure["key"]
                    for failure in result["failures"]
                )
            )

    def test_compare_rejects_matching_reshaped_maps(self):
        def flatten_map(arrays):
            for suffix in ("material_ids", "underlying_ids"):
                key = f"map/Ex/{suffix}"
                arrays[key] = arrays[key].reshape(-1, 1)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, artifact = self.capture_small_archive(directory)
            reference = directory / "reshaped-reference.npz"
            candidate = directory / "reshaped-candidate.npz"
            self.rewrite_archive(artifact, reference, mutate_arrays=flatten_map)
            self.rewrite_archive(artifact, candidate, mutate_arrays=flatten_map)
            result = self.oracle.compare_archives(reference, candidate, manifest)
            self.assertFalse(result["passed"])
            self.assertTrue(
                all(
                    "archive-contract" in failure["key"]
                    for failure in result["failures"]
                )
            )

    def test_compare_rejects_matching_invalid_state_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, artifact = self.capture_small_archive(directory)
            with np.load(artifact, allow_pickle=False) as archive:
                state_key = next(
                    key
                    for key in archive.files
                    if "/state/" in key
                    and key.endswith("/values")
                    and archive[key].size
                )

            def make_nonfinite(arrays):
                arrays[state_key].flat[0] = np.nan

            def reshape_state(arrays):
                arrays[state_key] = arrays[state_key].reshape(1, -1)

            mutations = (
                ("nonfinite", make_nonfinite),
                ("reshaped", reshape_state),
            )
            for name, mutation in mutations:
                with self.subTest(name=name):
                    reference = directory / f"{name}-reference.npz"
                    candidate = directory / f"{name}-candidate.npz"
                    self.rewrite_archive(artifact, reference, mutate_arrays=mutation)
                    self.rewrite_archive(artifact, candidate, mutate_arrays=mutation)
                    result = self.oracle.compare_archives(
                        reference, candidate, manifest
                    )
                    self.assertFalse(result["passed"])
                    self.assertEqual(
                        [failure["key"] for failure in result["failures"]],
                        [
                            "reference/archive-contract",
                            "candidate/archive-contract",
                        ],
                    )

    def test_compare_rejects_nonfinite_source_state(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, artifact = self.capture_small_archive(directory)
            with np.load(artifact, allow_pickle=False) as archive:
                source_key = next(
                    key
                    for key in archive.files
                    if "/source/" in key
                    and key.endswith("/values")
                    and archive[key].size
                )

            def make_nonfinite(arrays):
                arrays[source_key].flat[0] = np.nan

            candidate = directory / "nonfinite-source.npz"
            self.rewrite_archive(artifact, candidate, mutate_arrays=make_nonfinite)
            result = self.oracle.compare_archives(artifact, candidate, manifest)
            self.assertFalse(result["passed"])
            self.assertEqual(
                [failure["key"] for failure in result["failures"]],
                ["candidate/archive-contract"],
            )

    def test_compare_binds_geometry_and_coefficients(self):
        def mutate_geometry(metadata):
            metadata["geometry_and_coefficients"][0]["geometry"] += "Tampered"

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, artifact = self.capture_small_archive(directory)
            candidate = directory / "geometry-tampered.npz"
            self.rewrite_archive(artifact, candidate, mutate_metadata=mutate_geometry)
            result = self.oracle.compare_archives(artifact, candidate, manifest)
            self.assertFalse(result["passed"])
            self.assertEqual(
                [failure["key"] for failure in result["failures"]],
                ["geometry_and_coefficients"],
            )

    def test_compare_rejects_duplicate_archive_members(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, artifact = self.capture_small_archive(directory)
            candidate = directory / "duplicate-member.npz"
            shutil.copyfile(artifact, candidate)
            with zipfile.ZipFile(candidate, "a") as archive:
                member = next(
                    name
                    for name in archive.namelist()
                    if "/state/" in name and name.endswith("/values.npy")
                )
                content = archive.read(member)
                with self.assertWarns(UserWarning):
                    archive.writestr(member, content)
            with np.load(candidate, allow_pickle=False) as archive:
                self.assertEqual(archive.files.count(member.removesuffix(".npy")), 2)
            result = self.oracle.compare_archives(artifact, candidate, manifest)
            self.assertFalse(result["passed"])
            self.assertEqual(
                [failure["key"] for failure in result["failures"]],
                ["candidate/archive-contract"],
            )

    def test_compare_rejects_noncanonical_metadata_json(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, artifact = self.capture_small_archive(directory)
            with np.load(artifact, allow_pickle=False) as archive:
                original_arrays = {
                    key: np.array(archive[key], copy=True) for key in archive.files
                }
            raw_metadata = str(original_arrays["metadata.json"])
            metadata = json.loads(raw_metadata)
            duplicate_key = raw_metadata.replace(
                '"backend": "native"',
                '"backend": "tampered", "backend": "native"',
                1,
            )
            nonfinite_constant = raw_metadata.replace(
                f'"active_cells": {metadata["active_cells"]}',
                '"active_cells": NaN',
                1,
            )
            for name, tampered_metadata in (
                ("duplicate-key", duplicate_key),
                ("nonfinite-constant", nonfinite_constant),
            ):
                with self.subTest(name=name):
                    self.assertNotEqual(tampered_metadata, raw_metadata)
                    arrays = {
                        key: np.array(value, copy=True)
                        for key, value in original_arrays.items()
                    }
                    arrays["metadata.json"] = np.asarray(tampered_metadata)
                    candidate = directory / f"{name}.npz"
                    np.savez_compressed(candidate, **arrays)
                    result = self.oracle.compare_archives(artifact, candidate, manifest)
                    self.assertFalse(result["passed"])
                    self.assertEqual(
                        [failure["key"] for failure in result["failures"]],
                        ["candidate/archive-contract"],
                    )

    def test_compare_rejects_archive_contract_tampering(self):
        def mutate_reference_commit(metadata):
            metadata["provenance"]["source"]["commit"] = "b" * 40

        def mutate_candidate_status(metadata):
            metadata["provenance"]["source"].update(
                git_status=" M gmes/torch_fdtd.py", clean=False
            )

        def mutate_candidate_commit(metadata):
            metadata["provenance"]["source"]["commit"] = "deadbeef"

        def mutate_backend(metadata):
            metadata["backend"] = "unknown"

        def mutate_workload(metadata):
            metadata["workload"]["resolution"] += 1

        def mutate_schema(metadata):
            metadata["schema_version"] = 1

        def mutate_physical(metadata):
            metadata["steps"]["1"]["physical"]["energy"] += 1.0

        def mutate_bytes(metadata):
            metadata["archive_array_bytes"] -= 1

        mutations = (
            ("reference-commit", mutate_reference_commit, None),
            ("candidate-status", None, mutate_candidate_status),
            ("candidate-commit", None, mutate_candidate_commit),
            ("backend", None, mutate_backend),
            ("workload", mutate_workload, mutate_workload),
            ("schema", mutate_schema, mutate_schema),
            ("physical", mutate_physical, mutate_physical),
            ("bytes", mutate_bytes, mutate_bytes),
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, artifact = self.capture_small_archive(directory)
            for name, reference_mutation, candidate_mutation in mutations:
                with self.subTest(name=name):
                    reference = directory / f"{name}-reference.npz"
                    candidate = directory / f"{name}-candidate.npz"
                    self.rewrite_archive(
                        artifact,
                        reference,
                        mutate_metadata=reference_mutation,
                    )
                    self.rewrite_archive(
                        artifact,
                        candidate,
                        mutate_metadata=candidate_mutation,
                    )
                    result = self.oracle.compare_archives(
                        reference, candidate, manifest
                    )
                    self.assertFalse(result["passed"])
                    self.assertTrue(
                        any(
                            "archive-contract" in failure["key"]
                            for failure in result["failures"]
                        )
                    )

    def test_compare_accepts_declared_torch_candidate_metadata(self):
        def mark_torch_candidate(metadata):
            metadata["backend"] = "torch"
            metadata["backend_metadata"] = {"producer": "future-torch-oracle"}
            metadata["provenance"]["source"]["commit"] = "b" * 40

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest, reference = self.capture_small_archive(directory)
            candidate = directory / "torch-candidate.npz"
            self.rewrite_archive(
                reference,
                candidate,
                mutate_metadata=mark_torch_candidate,
            )
            result = self.oracle.compare_archives(reference, candidate, manifest)
            self.assertTrue(result["passed"], result["failures"])

    def test_archive_contract_covers_dimensions_sources_and_state(self):
        for name in (
            "dielectric-1d",
            "tfsf-transparent",
            "gaussian-auxiliary",
            "dm2-4",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                spec = self.oracle.find_case(self.manifest, name)
                manifest = json.loads(json.dumps(self.manifest))
                manifest["reference"]["capture_steps"] = [1]
                artifact = Path(directory) / f"{name}.npz"
                with patch.object(
                    self.oracle,
                    "_checkout_provenance",
                    side_effect=self.clean_archive_provenance,
                ):
                    self.oracle.capture_case(spec, manifest, artifact)
                result = self.oracle.compare_archives(artifact, artifact, manifest)
                self.assertTrue(result["passed"], result["failures"])

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
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(
            result["benchmark_contract"],
            {
                "initializer": "native-affine-ramp-v1",
                "seed": 115,
                "field_scale": 1e-3,
                "warmup_steps": 1,
                "steps_per_repeat": 1,
                "repetitions": 2,
                "timer": "time.perf_counter",
                "sample_start": "independently-rebuilt-post-warmup-state",
            },
        )
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
            [0, 1, 2, 2, 2, 2],
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
            self.assertEqual(result["capture"]["schema_version"], 2)
            source = result["capture"]["provenance"]["source"]
            self.assertEqual(Path(source["checkout"]), ROOT)
            self.assertEqual(
                source["commit"],
                self.oracle._git_output(ROOT, "rev-parse", "HEAD"),
            )
            self.assertEqual(
                source["git_status"],
                self.oracle._git_output(
                    ROOT, "status", "--short", "--untracked-files=all"
                ),
            )
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
