import copy
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from benchmarks import issue123_completion as completion
from benchmarks import issue123_differential as differential
from benchmarks import native_oracle


class _Archive(dict):
    @property
    def files(self):
        return list(self)


class Issue123DifferentialEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = native_oracle.load_manifest(differential.DEFAULT_MANIFEST)
        for workload in self.manifest["correctness"]:
            if workload.get("complex") is not True:
                continue
            workload["size"] = [4, 4, 4 if float(workload["size"][2]) > 0 else 0]
            workload["resolution"] = 1
        for workload in self.manifest["benchmarks"]:
            if workload["name"] == "single-gpu-2d":
                workload["size"] = [4, 4, 0]
                workload["resolution"] = 1
            elif workload["name"] == "single-gpu-3d":
                workload["size"] = [4, 4, 4]
                workload["resolution"] = 1
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": hashlib.sha256(
                differential.DEFAULT_MANIFEST.read_bytes()
            ).hexdigest(),
        }
        self.frozen_geometry = dict(
            differential.FROZEN_PERSISTENT_GEOMETRY_SHA256_BY_CASE
        )
        geometry_patch = mock.patch.object(
            differential,
            "FROZEN_PERSISTENT_GEOMETRY_SHA256_BY_CASE",
            self.frozen_geometry,
        )
        geometry_patch.start()
        self.addCleanup(geometry_patch.stop)
        completion_geometry_patch = mock.patch.object(
            completion,
            "FROZEN_DIFFERENTIAL_PERSISTENT_GEOMETRY_SHA256_BY_CASE",
            self.frozen_geometry,
        )
        completion_geometry_patch.start()
        self.addCleanup(completion_geometry_patch.stop)

    def descriptor(self, path):
        raw = path.read_bytes()
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "media_type": differential.MEDIA_TYPE_NPZ,
            "candidate_evidence": self.candidate,
        }

    def projection_contract(
        self,
        scope="single-gpu-cuda",
        case="single-gpu-2d",
        device="cuda:0",
    ):
        groups = differential.frozen_projection_groups(
            self.manifest, scope, case, device
        )
        steps = differential.frozen_projection_steps(self.manifest, scope, case, device)
        fields = [
            f"step/{step}/field/{name}"
            for step in steps
            for name in differential.FIELD_ARRAYS
        ]
        comparison = differential.frozen_comparison_contract(
            self.manifest, scope, case, device
        )
        physical = (
            [
                f"step/{step}/{suffix}"
                for step in steps
                for suffix in differential.PHYSICAL_ARRAY_SUFFIXES
            ]
            if comparison["mode"] == differential.NORMALIZED_COMPARISON_MODE
            else []
        )
        persistent = differential._persistent_projection_names(
            steps, differential._expected_persistent_suffixes(case)
        )
        contracts = [
            differential.SOURCE_CONTRACT_ARRAY,
            differential.SOURCE_PROOF_ARRAY,
        ]
        return groups, steps, fields, physical, persistent, contracts

    def arrays(
        self,
        dtype,
        case="single-gpu-2d",
        *,
        scope="single-gpu-cuda",
        device="cuda:0",
    ):
        groups, steps, fields, physical, persistent, contracts = (
            self.projection_contract(scope, case, device)
        )
        workload = differential._workload(self.manifest, case)
        field_shapes = differential._expected_field_shapes(workload)
        values = {
            name: np.full(
                field_shapes[name.rsplit("/", 1)[-1]],
                0.25,
                dtype=dtype,
            )
            for name in fields
        }
        if (scope, case, device) == differential.NORMALIZED_RESIDUAL_CASE and np.dtype(
            dtype
        ) == np.dtype("float64"):
            values[f"step/{steps[-1]}/field/Ex"].flat[0] = (
                float(np.finfo(np.float32).max) * 2.0
            )
        if physical:
            for step in steps:
                summary = [0.0, 0.0, 0.0, 0.0, 1.0]
                for component in differential.FIELD_ARRAYS:
                    field = values[f"step/{step}/field/{component}"]
                    magnitude = np.abs(field)
                    summary[0] += float(np.sum(magnitude * magnitude))
                    summary[1] = max(summary[1], float(np.max(magnitude)))
                    summary[2] += float(np.sum(magnitude[0] * magnitude[0]))
                    summary[3] += float(np.sum(magnitude[-1] * magnitude[-1]))
                    summary[4] = float(bool(summary[4]) and np.isfinite(field).all())
                    axes = tuple(range(1, field.ndim))
                    line = np.mean(field, axis=axes) if axes else field
                    values[f"step/{step}/physical/spectrum/{component}"] = np.abs(
                        np.fft.fft(line)
                    )
                values[f"step/{step}/physical/summary"] = np.asarray(
                    summary, dtype=np.float64
                )
        updater_labels = differential.CASE_UPDATER_LABELS[case]
        for step in steps:
            for component, shape in field_shapes.items():
                targets = np.arange(np.prod(shape), dtype=np.int64)
                for updater_label, chunk in zip(
                    updater_labels,
                    np.array_split(targets, len(updater_labels)),
                    strict=True,
                ):
                    prefix = f"step/{step}/state/{component}/{updater_label}"
                    values[f"{prefix}/indices"] = np.column_stack(
                        np.unravel_index(chunk, shape)
                    ).astype(np.int64, copy=False)
                    width = differential._persistent_state_width(
                        workload, component, updater_label
                    )
                    values[f"{prefix}/values"] = np.full(
                        len(chunk) * width,
                        complex(0.25, -0.5),
                        dtype=np.complex128,
                    )
        self.frozen_geometry[case] = differential._persistent_geometry_sha256(
            values, workload, case, steps[0]
        )
        source = differential._expected_point_source_contract(workload)
        values[differential.SOURCE_CONTRACT_ARRAY] = np.frombuffer(
            differential._canonical_source_bytes(source), dtype=np.uint8
        ).copy()
        native, native_metadata, _workload, _expected = self.native_point_source_inputs(
            case
        )
        real_dtype = np.empty((), dtype=dtype).real.dtype
        candidate, candidate_metadata, _workload, _expected = (
            self.candidate_point_source_inputs(case, precision=real_dtype)
        )
        proof = differential._build_point_source_raw_proof(
            native,
            candidate,
            native_metadata,
            candidate_metadata,
            workload,
        )
        values[differential.SOURCE_PROOF_ARRAY] = np.frombuffer(
            differential._canonical_source_bytes(proof), dtype=np.uint8
        ).copy()
        return {
            name: values[name] for name in (*fields, *physical, *persistent, *contracts)
        }

    def group_arrays(self, arrays, group):
        contracts = {
            differential.SOURCE_CONTRACT_ARRAY,
            differential.SOURCE_PROOF_ARRAY,
        }
        return {
            name: value
            for name, value in arrays.items()
            if name in contracts or differential._projection_array_step(name) in group
        }

    def refresh_physical(self, arrays, step):
        spectra, summary = differential._recomputed_physical_arrays(arrays, step)
        for component, spectrum in spectra.items():
            arrays[f"step/{step}/physical/spectrum/{component}"] = spectrum
        arrays[f"step/{step}/physical/summary"] = summary

    def write_group_npz(self, name, arrays, scope, case, device, role, ordinal, group):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        differential._write_npz(
            path,
            self.group_arrays(arrays, group),
            differential._group_archive_comment(
                scope,
                case,
                device,
                role,
                ordinal,
                group,
                self.candidate,
            ),
        )
        return path

    def write_source_npz(self, name, marker):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, marker=np.asarray([marker], dtype=np.int64))
        differential._preflight_source_npz(path, "test differential source")
        return path

    @staticmethod
    def insert_unindexed_local_record(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            comment_size = len(archive.comment)
        eocd = len(raw) - 22 - comment_size
        central_offset = int.from_bytes(raw[eocd + 16 : eocd + 20], "little")
        hidden = io.BytesIO()
        with zipfile.ZipFile(hidden, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("unindexed.npy", b"not-an-indexed-member")
        hidden_raw = hidden.getvalue()
        hidden_eocd = len(hidden_raw) - 22
        hidden_central = int.from_bytes(
            hidden_raw[hidden_eocd + 16 : hidden_eocd + 20], "little"
        )
        local_record = hidden_raw[:hidden_central]
        changed = bytearray(raw[:central_offset] + local_record + raw[central_offset:])
        changed_eocd = eocd + len(local_record)
        changed[changed_eocd + 16 : changed_eocd + 20] = (
            central_offset + len(local_record)
        ).to_bytes(4, "little")
        return bytes(changed)

    def document(self, scope="single-gpu-cuda"):
        records = []
        reference_sources = {}
        for index, expected in enumerate(
            differential.expected_records(self.manifest, scope)
        ):
            dtype = (
                np.complex128
                if scope == "paired-real" and expected["device"] == "cpu"
                else (
                    np.complex64
                    if scope == "paired-real"
                    else np.dtype(expected["precision"])
                )
            )
            (
                projection_groups,
                projection_steps,
                fields,
                physical,
                persistent,
                contracts,
            ) = self.projection_contract(scope, expected["case"], expected["device"])
            reference = self.arrays(
                dtype,
                expected["case"],
                scope=scope,
                device=expected["device"],
            )
            candidate = {name: value.copy() for name, value in reference.items()}
            comparison = differential.frozen_comparison_contract(
                self.manifest, scope, expected["case"], expected["device"]
            )
            reference_descriptors = []
            candidate_descriptors = []
            for ordinal, group in enumerate(projection_groups):
                reference_path = self.write_group_npz(
                    f"artifacts/{index}-{ordinal}-reference.npz",
                    reference,
                    scope,
                    expected["case"],
                    expected["device"],
                    "reference",
                    ordinal,
                    group,
                )
                candidate_path = self.write_group_npz(
                    f"artifacts/{index}-{ordinal}-candidate.npz",
                    candidate,
                    scope,
                    expected["case"],
                    expected["device"],
                    "candidate",
                    ordinal,
                    group,
                )
                reference_descriptors.append(self.descriptor(reference_path))
                candidate_descriptors.append(self.descriptor(candidate_path))
            metrics, passed = differential._compare_arrays(
                reference,
                candidate,
                comparison,
                array_comparisons=differential._frozen_array_comparisons(
                    self.manifest,
                    scope,
                    expected["case"],
                    expected["device"],
                    reference,
                ),
            )
            if expected["case"] not in reference_sources:
                reference_sources[expected["case"]] = self.write_source_npz(
                    f"sources/{expected['case']}-reference.npz",
                    1000 + len(reference_sources),
                )
            candidate_source = self.write_source_npz(
                f"sources/{index}-{expected['case']}-candidate.npz",
                2000 + index,
            )
            records.append(
                {
                    **expected,
                    "projection_steps": projection_steps,
                    "projection_groups": projection_groups,
                    "reference": reference_descriptors,
                    "candidate": candidate_descriptors,
                    "reference_source": self.descriptor(
                        reference_sources[expected["case"]]
                    ),
                    "candidate_source": self.descriptor(candidate_source),
                    "field_arrays": fields,
                    "physical_arrays": physical,
                    "persistent_arrays": persistent,
                    "contract_arrays": contracts,
                    "comparison": comparison,
                    "precision_limitation": differential._precision_limitation(
                        reference,
                        scope,
                        expected["case"],
                        expected["device"],
                        projection_steps,
                    ),
                    "metrics": metrics,
                    "passed": passed,
                }
            )
        return {
            "schema_version": differential.INDEX_SCHEMA_VERSION,
            "kind": differential.INDEX_KIND,
            "scope": scope,
            "candidate_evidence": self.candidate,
            "required_cases": differential.expected_records(self.manifest, scope),
            "cases": records,
            "passed": True,
        }

    def write_index(self, document, name="index.json"):
        path = self.root / name
        path.write_text(
            json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def load(self, document):
        with mock.patch.object(
            differential,
            "_regenerate_record_projections",
            side_effect=self.regenerate_fixture_projections,
        ):
            return differential.load_differential_evidence_index(
                self.write_index(document),
                self.manifest,
                self.candidate,
                descriptor_root=self.root,
                expected_scope=document["scope"],
            )

    def regenerate_fixture_projections(
        self,
        root,
        record,
        _manifest,
        candidate,
        _scope,
        expected,
        projection_groups,
        artifact_loader,
        used_paths,
        used_digests,
        reference_sources,
    ):
        sources = []
        for role in ("reference", "candidate"):
            descriptor = record[f"{role}_source"]
            path, raw = differential._load_bound_descriptor(
                root,
                descriptor,
                candidate,
                f"fixture {role} source",
                artifact_loader,
                source=True,
            )
            self.assertEqual(path.read_bytes(), raw)
            differential._preflight_source_npz(path, f"fixture {role} source")
            sources.append((path, descriptor))
        self.assertNotEqual(sources[0][0], sources[1][0])
        reference_identity = tuple(
            sources[0][1][name] for name in ("path", "sha256", "size_bytes")
        )
        prior = reference_sources.setdefault(expected["case"], reference_identity)
        self.assertEqual(prior, reference_identity)
        if sources[0][1]["path"] not in used_paths:
            self.assertNotIn(sources[0][1]["sha256"], used_digests)
            used_paths.add(sources[0][1]["path"])
            used_digests.add(sources[0][1]["sha256"])
        candidate_descriptor = sources[1][1]
        self.assertNotIn(candidate_descriptor["path"], used_paths)
        self.assertNotIn(candidate_descriptor["sha256"], used_digests)
        used_paths.add(candidate_descriptor["path"])
        used_digests.add(candidate_descriptor["sha256"])

        artifacts = {}
        for ordinal, _group in enumerate(projection_groups):
            loaded = []
            for role in ("reference", "candidate"):
                descriptor = record[role][ordinal]
                path, raw = differential._load_bound_descriptor(
                    root,
                    descriptor,
                    candidate,
                    f"fixture group {ordinal} {role}",
                    artifact_loader,
                    source=False,
                )
                differential._require(
                    descriptor["path"] not in used_paths
                    and descriptor["sha256"] not in used_digests,
                    f"fixture group {ordinal} reuses artifact path or bytes",
                )
                used_paths.add(descriptor["path"])
                used_digests.add(descriptor["sha256"])
                loaded.append((path, raw))
            artifacts[ordinal] = tuple(loaded)
        return artifacts, (
            record["field_arrays"],
            record["physical_arrays"],
            record["persistent_arrays"],
            record["contract_arrays"],
        )

    def rewrite_group(self, document, index, role, ordinal, arrays):
        record = document["cases"][index]
        path = self.root / record[role][ordinal]["path"]
        path.unlink()
        differential._write_npz(
            path,
            self.group_arrays(arrays, record["projection_groups"][ordinal]),
            differential._group_archive_comment(
                document["scope"],
                record["case"],
                record["device"],
                role,
                ordinal,
                record["projection_groups"][ordinal],
                self.candidate,
            ),
        )
        record[role][ordinal] = self.descriptor(path)

    def rewrite_candidate(self, document, index, arrays):
        for ordinal in range(len(document["cases"][index]["projection_groups"])):
            self.rewrite_group(document, index, "candidate", ordinal, arrays)

    def rewrite_reference(self, document, index, arrays):
        for ordinal in range(len(document["cases"][index]["projection_groups"])):
            self.rewrite_group(document, index, "reference", ordinal, arrays)

    def validate_completion(self, document):
        artifact = completion.LoadedArtifact(
            descriptor={"sha256": "0" * 64},
            path=self.root / "index.json",
            raw=b"",
            document=document,
        )
        with mock.patch.object(
            differential,
            "_regenerate_record_projections",
            side_effect=self.regenerate_fixture_projections,
        ):
            return completion._validate_differential(
                artifact,
                completion.ArtifactReader(self.root, self.candidate),
                self.manifest,
                self.candidate,
                scope=document["scope"],
            )

    def validate_completion_mirror(self, document):
        with mock.patch.object(
            differential,
            "validate_differential_document",
            return_value={"passed": True},
        ):
            return self.validate_completion(document)

    def native_point_source_inputs(self, case="single-gpu-2d"):
        workload = differential._workload(self.manifest, case)
        expected = differential._expected_point_source_contract(workload)
        capture_steps = workload.get(
            "capture_steps", self.manifest["reference"]["capture_steps"]
        )
        arrays = _Archive()
        steps = {}
        amplitude = float(workload.get("source_amp", 1e-3))
        eps_inf, mu_inf = differential._expected_point_source_medium(workload)
        for value in (0, *capture_steps):
            step = str(value)
            dt = differential._expected_point_source_time_step(workload)
            time = (value + 2) * dt
            prefix = f"step/{step}/source/Ex/0-PointSourceEx"
            arrays[f"step/{step}/time"] = np.asarray(
                [value + 2, time, dt], dtype=np.float64
            )
            arrays[f"{prefix}/indices"] = np.asarray(
                [expected["sources"][0]["target_index"]], dtype=np.intc
            )
            arrays[f"{prefix}/values"] = np.asarray(
                [
                    amplitude,
                    eps_inf,
                    mu_inf,
                    differential._continuous_point_source_value(
                        time, bool(workload.get("complex"))
                    ),
                ],
                dtype=np.complex128,
            )
            steps[step] = {
                "sources": {
                    "updaters": [
                        {
                            "component": "Ex",
                            "native_type": "PointSourceEx",
                            "cells": 1,
                            "state_values": 4,
                        }
                    ]
                }
            }
        metadata = {"capture_steps": capture_steps, "steps": steps}
        return arrays, metadata, workload, expected

    def candidate_point_source_inputs(self, case="single-gpu-2d", precision=None):
        workload = differential._workload(self.manifest, case)
        expected = differential._expected_point_source_contract(workload)
        capture_steps = workload.get(
            "capture_steps", self.manifest["reference"]["capture_steps"]
        )
        precision = np.dtype("float32" if precision is None else precision)
        parameters = np.asarray(
            [
                differential.POINT_SOURCE_FREQUENCY,
                differential.POINT_SOURCE_PHASE,
                differential.POINT_SOURCE_START,
                differential.POINT_SOURCE_END,
                differential.POINT_SOURCE_WIDTH,
                0.0,
            ],
            dtype=precision,
        )
        amplitude = np.asarray(workload.get("source_amp", 1e-3), dtype=precision)
        target = expected["sources"][0]["target_index"]
        field_shape = differential._point_source_field_shape(workload)
        flat_target = np.ravel_multi_index(tuple(target), field_shape)
        channels = 2 if workload.get("complex") else 1
        arrays = _Archive()
        steps = {}
        for value in (0, *capture_steps):
            step = str(value)
            dt = differential._expected_point_source_time_step(workload)
            time = (value + 2) * dt
            evaluated_time = time - 0.5 * dt
            oscillator = differential._continuous_point_source_value(
                evaluated_time, bool(workload.get("complex"))
            )
            live_root = f"torch/step/{step}/sources/batches/0"
            arrays[f"step/{step}/time"] = np.asarray(
                [value + 2, time, dt], dtype=np.float64
            )
            arrays[f"step/{step}/field/Ex"] = np.zeros(field_shape, dtype=precision)
            arrays[f"{live_root}/overwrite_targets"] = np.asarray(
                [flat_target], dtype=np.int64
            )
            arrays[f"{live_root}/overwrite_models"] = np.asarray(
                [expected["sources"][0]["model"]["id"]], dtype=np.int8
            )
            arrays[f"{live_root}/overwrite_parameters"] = parameters.reshape(1, 6)
            arrays[f"{live_root}/overwrite_amplitudes"] = amplitude.reshape(1)
            evaluated = (
                [complex(oscillator).real, complex(oscillator).imag]
                if channels == 2
                else [float(oscillator)]
            )
            arrays[f"{live_root}/_overwrite_values"] = np.asarray(
                [amplitude.item() * np.asarray(evaluated)], dtype=precision
            )
            arrays[f"{live_root}/additive_targets"] = np.empty(0, dtype=np.int64)
            arrays[f"{live_root}/additive_models"] = np.empty(0, dtype=np.int8)
            arrays[f"{live_root}/additive_parameters"] = np.empty(
                (0, 6), dtype=precision
            )
            arrays[f"{live_root}/additive_amplitudes"] = np.empty(0, dtype=precision)
            arrays[f"{live_root}/_additive_values"] = np.empty(
                (0, channels), dtype=precision
            )
            packed_root = f"step/{step}/source/Ex/0-PointSourceEx"
            arrays[f"{packed_root}/indices"] = np.asarray([target], dtype=np.int64)
            arrays[f"{packed_root}/values"] = (
                np.concatenate(
                    (
                        np.asarray(
                            [0.0, expected["sources"][0]["model"]["id"]],
                            dtype=np.float64,
                        ),
                        parameters.astype(np.float64),
                        amplitude.reshape(1).astype(np.float64),
                    )
                )
                .astype("<f8", copy=False)
                .view("<u8")
            )
            steps[step] = {
                "sources": {
                    "updaters": [
                        {
                            "component": "Ex",
                            "native_type": "PointSourceEx",
                            "cells": 1,
                            "state_values": 9,
                        }
                    ]
                }
            }
        metadata = {
            "capture_steps": capture_steps,
            "backend_metadata": {"precision": precision.name},
            "steps": steps,
        }
        return arrays, metadata, workload, expected

    def test_valid_index_recomputes_every_raw_projection(self):
        document = self.document()
        result = self.load(document)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["cases"]), 2)

        self.validate_completion(document)
        self.validate_completion_mirror(document)

    def test_manifest_record_capture_and_tolerance_contracts_are_literal(self):
        self.assertEqual(
            self.candidate["manifest_sha256"],
            differential.TRUSTED_MANIFEST_SHA256,
        )
        paired = differential.expected_records(self.manifest, "paired-real")
        self.assertEqual(len(paired), 16)
        self.assertEqual(
            [(record["case"], record["device"]) for record in paired],
            [
                (case, device)
                for case in differential.PAIRED_CASES
                for device in ("cpu", "cuda:0")
            ],
        )
        self.assertEqual(
            len(
                paired + differential.expected_records(self.manifest, "single-gpu-cuda")
            ),
            18,
        )

        changed = copy.deepcopy(self.manifest)
        changed["correctness"][0]["complex"] = not changed["correctness"][0].get(
            "complex", False
        )
        with self.assertRaisesRegex(ValueError, "frozen case closure"):
            differential.expected_records(changed, "paired-real")

        changed = copy.deepcopy(self.manifest)
        differential._workload(changed, "single-gpu-2d")["capture_steps"] = [100]
        with self.assertRaisesRegex(ValueError, "frozen contract"):
            differential.frozen_projection_steps(
                changed, "single-gpu-cuda", "single-gpu-2d", "cuda:0"
            )

        changed = copy.deepcopy(self.manifest)
        changed["tolerances"]["torch"]["dm2"]["float32"]["rtol"] = 1.0
        with self.assertRaisesRegex(ValueError, "frozen values"):
            differential.expected_records(changed, "single-gpu-cuda")

        changed_candidate = dict(self.candidate, manifest_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "trusted repository manifest"):
            differential._candidate_projection(changed_candidate)

    def test_full_sources_reproduce_every_compact_group_exactly(self):
        document = self.document()
        record = document["cases"][0]
        expected = document["required_cases"][0]
        baseline_reference = self.arrays(
            np.float32,
            expected["case"],
            scope=document["scope"],
            device=expected["device"],
        )
        baseline_candidate = {
            name: value.copy() for name, value in baseline_reference.items()
        }

        def regenerate(
            _reference_path,
            candidate_path,
            _manifest,
            scope,
            expected_record,
            *,
            group_consumer=None,
            source_bindings=None,
        ):
            self.assertIsNotNone(source_bindings)
            reference = {
                name: value.copy() for name, value in baseline_reference.items()
            }
            candidate = {
                name: value.copy() for name, value in baseline_candidate.items()
            }
            with np.load(candidate_path, allow_pickle=False) as archive:
                marker = int(archive["marker"][0])
            if marker != 2000:
                candidate[f"step/100/field/Ex"].flat[0] += 0.5
            groups = differential.frozen_projection_groups(
                self.manifest,
                scope,
                expected_record["case"],
                expected_record["device"],
            )
            for ordinal, group in enumerate(groups):
                group_consumer(
                    ordinal,
                    group,
                    self.group_arrays(reference, group),
                    self.group_arrays(candidate, group),
                )
            _, steps, fields, physical, persistent, contracts = (
                self.projection_contract(
                    scope,
                    expected_record["case"],
                    expected_record["device"],
                )
            )
            return (
                None,
                None,
                self.candidate,
                steps,
                groups,
                fields,
                physical,
                persistent,
                contracts,
            )

        def validate_record(value):
            return differential._regenerate_record_projections(
                self.root,
                value,
                self.manifest,
                self.candidate,
                document["scope"],
                expected,
                value["projection_groups"],
                None,
                set(),
                set(),
                {},
            )

        with mock.patch.object(
            differential, "_projection_arrays", side_effect=regenerate
        ):
            artifacts, _names = validate_record(record)
            self.assertEqual(len(artifacts), 1)

            changed = copy.deepcopy(document)["cases"][0]
            substituted = {
                name: value.copy() for name, value in baseline_reference.items()
            }
            substituted["step/100/field/Ex"].fill(123.0)
            self.rewrite_group(
                {"scope": document["scope"], "cases": [changed]},
                0,
                "reference",
                0,
                substituted,
            )
            self.rewrite_group(
                {"scope": document["scope"], "cases": [changed]},
                0,
                "candidate",
                0,
                substituted,
            )
            with self.assertRaisesRegex(ValueError, "complete source archive"):
                validate_record(changed)

            fresh_document = self.document()
            changed = fresh_document["cases"][0]
            source_path = self.root / changed["candidate_source"]["path"]
            source_path.unlink()
            np.savez_compressed(source_path, marker=np.asarray([9999], dtype=np.int64))
            changed["candidate_source"] = self.descriptor(source_path)
            with self.assertRaisesRegex(ValueError, "complete source archive"):
                validate_record(changed)

    def test_unindexed_local_records_fail_after_descriptor_refresh(self):
        for target in ("projection", "source"):
            with self.subTest(target=target):
                document = self.document()
                record = document["cases"][0]
                descriptor = (
                    record["candidate"][0]
                    if target == "projection"
                    else record["candidate_source"]
                )
                path = self.root / descriptor["path"]
                path.write_bytes(self.insert_unindexed_local_record(path.read_bytes()))
                refreshed = self.descriptor(path)
                if target == "projection":
                    record["candidate"][0] = refreshed
                else:
                    record["candidate_source"] = refreshed
                with self.assertRaisesRegex(ValueError, "unindexed|gap|canonical"):
                    self.load(document)

    def test_standalone_json_reads_are_bounded_before_parse(self):
        document = self.document()
        path = self.write_index(document)
        with (
            mock.patch.object(differential, "MAX_INDEX_BYTES", path.stat().st_size - 1),
            self.assertRaisesRegex(ValueError, "size limit"),
        ):
            differential.load_differential_evidence_index(
                path,
                self.manifest,
                self.candidate,
                descriptor_root=self.root,
            )

        candidate_path = self.root / "candidate.json"
        candidate_path.write_text(json.dumps(self.candidate), encoding="utf-8")
        with (
            mock.patch.object(
                differential, "MAX_INDEX_BYTES", candidate_path.stat().st_size - 1
            ),
            self.assertRaisesRegex(ValueError, "size limit"),
        ):
            differential._load_candidate_evidence(candidate_path)

        manifest_size = differential.DEFAULT_MANIFEST.stat().st_size
        with (
            mock.patch.object(differential, "MAX_INDEX_BYTES", manifest_size - 1),
            self.assertRaisesRegex(ValueError, "size limit"),
        ):
            differential._load_trusted_manifest(differential.DEFAULT_MANIFEST)

    def test_single_gpu_group_steps_and_all_archive_bindings_are_exact(self):
        capture_steps = differential._frozen_capture_steps(
            self.manifest, "single-gpu-3d"
        )
        self.assertTrue(all(type(step) is int and step > 0 for step in capture_steps))
        self.assertNotIn(0, capture_steps)
        self.assertEqual(
            differential.frozen_projection_groups(
                self.manifest,
                "single-gpu-cuda",
                "single-gpu-3d",
                "cuda:0",
            ),
            [[0, 1], [2, 5], [20, 100]],
        )
        document = self.document()
        record = next(
            item for item in document["cases"] if item["case"] == "single-gpu-3d"
        )
        self.assertEqual(record["projection_steps"], [0, 1, 2, 5, 20, 100])
        descriptors = [
            descriptor
            for item in document["cases"]
            for role in ("reference", "candidate")
            for descriptor in item[role]
        ]
        self.assertEqual(len(descriptors), 8)
        self.assertEqual(len({item["path"] for item in descriptors}), 8)
        self.assertEqual(len({item["sha256"] for item in descriptors}), 8)
        comments = []
        for item in document["cases"]:
            for role in ("reference", "candidate"):
                for ordinal, (group, descriptor) in enumerate(
                    zip(item["projection_groups"], item[role], strict=True)
                ):
                    raw = (self.root / descriptor["path"]).read_bytes()
                    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                        comment = archive.comment
                    self.assertEqual(
                        comment,
                        differential._group_archive_comment(
                            document["scope"],
                            item["case"],
                            item["device"],
                            role,
                            ordinal,
                            group,
                            self.candidate,
                        ),
                    )
                    comments.append(comment)
        self.assertEqual(len(set(comments)), 8)

    def test_comparison_is_frozen_not_artifact_selected(self):
        document = self.document()
        document["cases"][0]["comparison"]["rtol"] = 1.0
        document["cases"][0]["comparison"]["atol"] = 1.0
        with self.assertRaisesRegex(ValueError, "comparison contract differs"):
            self.load(document)

        artifact = completion.LoadedArtifact(
            descriptor={"sha256": "0" * 64},
            path=self.root / "index.json",
            raw=b"",
            document=document,
        )
        with self.assertRaisesRegex(
            completion.EvidenceError,
            "could not be independently recomputed",
        ):
            completion._validate_differential(
                artifact,
                completion.ArtifactReader(self.root, self.candidate),
                self.manifest,
                self.candidate,
                scope="single-gpu-cuda",
            )

    def test_single_gpu_precision_and_projection_steps_are_frozen(self):
        required = differential.expected_records(self.manifest, "single-gpu-cuda")
        self.assertEqual(
            required,
            [
                {
                    "case": "single-gpu-2d",
                    "device": "cuda:0",
                    "precision": "float32",
                },
                {
                    "case": "single-gpu-3d",
                    "device": "cuda:0",
                    "precision": "float64",
                },
            ],
        )
        document = self.document()
        self.assertEqual(document["cases"][0]["projection_steps"], [100])
        self.assertEqual(
            document["cases"][1]["projection_steps"], [0, 1, 2, 5, 20, 100]
        )
        self.assertEqual(
            document["cases"][1]["projection_groups"], [[0, 1], [2, 5], [20, 100]]
        )
        for field in ("precision", "projection_steps"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(document)
                invalid["cases"][1][field] = (
                    "float32" if field == "precision" else [100]
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "case closure differs|identity differs|projection steps",
                ):
                    self.load(invalid)

    def test_normalized_projection_steps_reject_reordering_and_relabelling(self):
        for steps in ([100, 20], [20, 99]):
            with self.subTest(steps=steps):
                document = self.document()
                document["cases"][1]["projection_steps"] = steps
                with self.assertRaisesRegex(ValueError, "projection steps"):
                    self.load(document)
                with self.assertRaises(completion.EvidenceError):
                    self.validate_completion_mirror(document)

    def test_normalized_projection_groups_reject_missing_reordered_and_duplicate(self):
        for mutation in ("missing", "reordered", "duplicate"):
            with self.subTest(mutation=mutation):
                document = self.document()
                groups = document["cases"][1]["projection_groups"]
                if mutation == "missing":
                    groups.pop()
                elif mutation == "reordered":
                    groups[0], groups[1] = groups[1], groups[0]
                else:
                    groups[1] = copy.deepcopy(groups[0])
                with self.assertRaisesRegex(ValueError, "projection groups differ"):
                    self.load(document)

    def test_group_descriptor_lists_reject_missing_reordered_and_duplicate(self):
        for mutation in ("missing", "reordered", "duplicate"):
            with self.subTest(mutation=mutation):
                document = self.document()
                record = document["cases"][1]
                for role in ("reference", "candidate"):
                    descriptors = record[role]
                    if mutation == "missing":
                        descriptors.pop()
                    elif mutation == "reordered":
                        descriptors[0], descriptors[1] = descriptors[1], descriptors[0]
                    else:
                        descriptors[1] = copy.deepcopy(descriptors[0])
                message = (
                    "descriptor groups differ"
                    if mutation == "missing"
                    else "NPZ group identity differs|reuses artifact path or bytes"
                )
                with self.assertRaisesRegex(ValueError, message):
                    self.load(document)

    def test_group_npz_identity_is_exact_deterministic_and_has_no_extra_members(self):
        document = self.document()
        record = document["cases"][1]
        ordinal = 0
        group = record["projection_groups"][ordinal]
        path = self.root / record["candidate"][ordinal]["path"]
        expected_comment = json.dumps(
            {
                "schema": differential.GROUP_ARCHIVE_SCHEMA,
                "scope": document["scope"],
                "case": record["case"],
                "device": record["device"],
                "role": "candidate",
                "ordinal": ordinal,
                "steps": group,
                "candidate_evidence": self.candidate,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        names = differential._projection_group_names(
            group, document["scope"], record["case"], record["device"]
        )
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(archive.comment, expected_comment)
            self.assertEqual(
                [member.filename for member in archive.infolist()],
                [f"{name}.npy" for name in names],
            )
            self.assertTrue(
                all(
                    member.compress_type == zipfile.ZIP_DEFLATED
                    for member in archive.infolist()
                )
            )

        arrays = self.arrays(np.float64, "single-gpu-3d")
        first = self.root / "deterministic-first.npz"
        second = self.root / "deterministic-second.npz"
        for output in (first, second):
            differential._write_npz(
                output, self.group_arrays(arrays, group), expected_comment
            )
            self.assertGreater(output.stat().st_size, 0)
            self.assertLessEqual(
                output.stat().st_size, differential.MAX_NPZ_ARCHIVE_BYTES
            )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(path.read_bytes(), first.read_bytes())

        bounded = self.root / "bounded-output.npz"
        with (
            mock.patch.object(
                differential,
                "MAX_NPZ_ARCHIVE_BYTES",
                len(first.read_bytes()) - 1,
            ),
            self.assertRaisesRegex(ValueError, "archive size exceeds"),
        ):
            differential._write_npz(
                bounded, self.group_arrays(arrays, group), expected_comment
            )
        self.assertFalse(bounded.exists())

        with zipfile.ZipFile(path, "a") as archive:
            archive.comment = b"{}"
        record["candidate"][ordinal] = self.descriptor(path)
        with self.assertRaisesRegex(ValueError, "NPZ group identity differs"):
            self.load(document)

    def test_npz_preflight_rejects_noncanonical_members_before_numpy_load(self):
        document = self.document()
        record = document["cases"][1]
        ordinal = 0
        group = record["projection_groups"][ordinal]
        path = self.root / record["candidate"][ordinal]["path"]
        raw = path.read_bytes()
        names = differential._projection_group_names(
            group, document["scope"], record["case"], record["device"]
        )
        comment = differential._group_archive_comment(
            document["scope"],
            record["case"],
            record["device"],
            "candidate",
            ordinal,
            group,
            self.candidate,
        )

        def rewrite_first(transform):
            output = io.BytesIO()
            with (
                zipfile.ZipFile(io.BytesIO(raw)) as source,
                zipfile.ZipFile(output, "w") as target,
            ):
                target.comment = source.comment
                for index, member in enumerate(source.infolist()):
                    rewritten = copy.copy(member)
                    payload = source.read(member)
                    if index == 0:
                        rewritten, payload = transform(rewritten, payload)
                    target.writestr(rewritten, payload)
            return output.getvalue()

        def npy_payload(
            descr,
            *,
            shape=(1,),
            fortran_order=False,
            payload_bytes=None,
        ):
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header,
                {
                    "descr": descr,
                    "fortran_order": fortran_order,
                    "shape": shape,
                },
            )
            if payload_bytes is None:
                payload_bytes = math.prod(shape) * np.dtype(descr).itemsize
            return header.getvalue() + bytes(payload_bytes)

        def stored(member, payload):
            member.compress_type = zipfile.ZIP_STORED
            return member, payload

        def symlink(member, payload):
            member.create_system = 3
            member.external_attr = 0o120777 << 16
            return member, payload

        def directory_mode(member, payload):
            member.create_system = 3
            member.external_attr = 0o040755 << 16
            return member, payload

        def version_two(member, payload):
            return member, payload[:6] + b"\x02\x00" + payload[8:]

        def huge_truncated(member, _payload):
            return member, npy_payload("<f8", shape=(2**31 - 1,), payload_bytes=0)

        def replace(payload):
            return lambda member, _original: (member, payload)

        malformed = {
            "tiny-huge-shape": rewrite_first(huge_truncated),
            "truncated-payload": rewrite_first(
                replace(npy_payload("<f8", shape=(2,), payload_bytes=8))
            ),
            "padded-payload": rewrite_first(
                replace(npy_payload("<f8", payload_bytes=16))
            ),
            "unsupported-version": rewrite_first(version_two),
            "object": rewrite_first(replace(npy_payload("|O", payload_bytes=8))),
            "structured": rewrite_first(
                replace(
                    npy_payload(
                        np.dtype([("value", "<f8")]).descr,
                        payload_bytes=8,
                    )
                )
            ),
            "subarray": rewrite_first(
                replace(
                    npy_payload(
                        ("<f8", (2,)),
                        payload_bytes=16,
                    )
                )
            ),
            "fortran": rewrite_first(replace(npy_payload("<f8", fortran_order=True))),
            "stored": rewrite_first(stored),
            "symlink": rewrite_first(symlink),
            "directory-mode": rewrite_first(directory_mode),
            "prepended": b"stub" + raw,
            "trailing": raw + b"padding",
        }

        encrypted = bytearray(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            first_member = archive.infolist()[0]
        encrypted[first_member.header_offset + 6] |= 0x01
        central = encrypted.find(b"PK\x01\x02")
        self.assertGreaterEqual(central, 0)
        encrypted[central + 8] |= 0x01
        malformed["encrypted"] = bytes(encrypted)

        local_encrypted = bytearray(raw)
        local_encrypted[first_member.header_offset + 6] |= 0x01
        malformed["local-only-encrypted"] = bytes(local_encrypted)
        local_stored = bytearray(raw)
        local_stored[
            first_member.header_offset + 8 : first_member.header_offset + 10
        ] = zipfile.ZIP_STORED.to_bytes(2, "little")
        malformed["local-only-stored"] = bytes(local_stored)

        corrupted = bytearray(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            member = archive.infolist()[0]
        local = member.header_offset
        filename_size = int.from_bytes(corrupted[local + 26 : local + 28], "little")
        extra_size = int.from_bytes(corrupted[local + 28 : local + 30], "little")
        compressed_start = local + 30 + filename_size + extra_size
        corrupted[compressed_start + member.compress_size // 2] ^= 0x01
        malformed["corrupt-deflate"] = bytes(corrupted)
        invalid_deflate = bytearray(raw)
        invalid_deflate[compressed_start] = (
            invalid_deflate[compressed_start] & ~0x07
        ) | 0x07
        malformed["invalid-deflate"] = bytes(invalid_deflate)

        for mutation, malicious_raw in malformed.items():
            with (
                self.subTest(mutation=mutation),
                mock.patch.object(
                    differential.np,
                    "load",
                    side_effect=AssertionError("np.load must not run before preflight"),
                ) as load,
            ):
                with self.assertRaises(ValueError):
                    differential._load_projection(
                        malicious_raw, names, comment, "malicious group"
                    )
                load.assert_not_called()

    def test_generic_source_npz_preflight_accepts_stored_versions_and_metadata(self):
        def npy_payload(array, version):
            array = np.asarray(array)
            self.assertTrue(array.flags.c_contiguous)
            header = io.BytesIO()
            writer = (
                np.lib.format.write_array_header_1_0
                if version == (1, 0)
                else np.lib.format.write_array_header_2_0
            )
            writer(
                header,
                {
                    "descr": np.lib.format.dtype_to_descr(array.dtype),
                    "fortran_order": False,
                    "shape": array.shape,
                },
            )
            payload = header.getvalue()
            if version == (3, 0):
                payload = payload[:6] + b"\x03\x00" + payload[8:]
            return payload + array.tobytes(order="C")

        for version in ((1, 0), (2, 0), (3, 0)):
            with self.subTest(version=version):
                path = self.root / f"source-{version[0]}.npz"
                with zipfile.ZipFile(
                    path, "w", compression=zipfile.ZIP_STORED
                ) as archive:
                    archive.writestr(
                        "value.npy",
                        npy_payload(np.arange(4, dtype=np.float64), version),
                    )
                    archive.writestr(
                        "metadata.json.npy",
                        npy_payload(np.asarray('{"schema":1}'), version),
                    )
                resolved, binding = differential._preflight_source_npz(
                    path, "generic source"
                )
                self.assertEqual(resolved, path.resolve())
                self.assertEqual(
                    binding, differential._source_archive_identity(resolved)
                )

    def test_generic_source_npz_preflight_rejects_unsafe_dtypes(self):
        def payload(dtype, *, name="value.npy"):
            dtype = np.dtype(dtype)
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header,
                {
                    "descr": np.lib.format.dtype_to_descr(dtype),
                    "fortran_order": False,
                    "shape": (1,),
                },
            )
            path = self.root / f"unsafe-{hashlib.sha256(name.encode()).hexdigest()}.npz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(name, header.getvalue() + bytes(dtype.itemsize))
            return path

        for name, dtype in (
            ("text.npy", "<U1"),
            ("object.npy", "O"),
            ("structured.npy", [("value", "<f8")]),
            ("subarray.npy", ("<f8", (2,))),
            ("metadata.json.npy", "O"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, "shape or dtype|payload contract"
                ):
                    differential._preflight_source_npz(
                        payload(dtype, name=name), "unsafe source"
                    )
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {"descr": "<f8", "fortran_order": True, "shape": (2, 2)},
        )
        fortran_path = self.root / "unsafe-fortran.npz"
        with zipfile.ZipFile(fortran_path, "w") as archive:
            archive.writestr("value.npy", header.getvalue() + bytes(32))
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            differential._preflight_source_npz(fortran_path, "unsafe source")

    def test_builder_preflights_huge_and_truncated_source_before_loaders(self):
        def malformed(shape, payload_bytes):
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header,
                {
                    "descr": "<f8",
                    "fortran_order": False,
                    "shape": shape,
                },
            )
            path = self.root / f"malformed-{shape[0]}-{payload_bytes}.npz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("value.npy", header.getvalue() + bytes(payload_bytes))
            return path

        for name, path in (
            ("huge-shape", malformed((2**31 - 1,), 0)),
            ("truncated", malformed((2,), 8)),
        ):
            with (
                self.subTest(name=name),
                mock.patch.object(
                    differential.torch_correctness,
                    "_archive_record",
                    side_effect=AssertionError("archive loader must not run"),
                ) as archive_record,
                mock.patch.object(
                    differential.torch_correctness,
                    "compare_torch_archives",
                    side_effect=AssertionError("comparison must not run"),
                ) as compare,
                mock.patch.object(
                    differential.np,
                    "load",
                    side_effect=AssertionError("np.load must not run"),
                ) as load,
                self.assertRaises(ValueError),
            ):
                differential.build_differential_evidence(
                    [path],
                    [],
                    self.manifest,
                    self.candidate,
                    scope="single-gpu-cuda",
                    descriptor_root=self.root,
                    output_directory=self.root / f"output-{name}",
                )
            archive_record.assert_not_called()
            compare.assert_not_called()
            load.assert_not_called()

    def test_early_field_and_state_tamper_are_recomputed(self):
        for name in (
            "step/1/field/Ex",
            "step/5/state/Ex/0-Cpml/values",
        ):
            with self.subTest(name=name):
                document = self.document()
                arrays = self.arrays(np.float64, "single-gpu-3d")
                arrays[name].flat[0] += 1e-3
                if name == "step/1/field/Ex":
                    self.refresh_physical(arrays, 1)
                self.rewrite_candidate(document, 1, arrays)
                with self.assertRaisesRegex(
                    ValueError, "recomputed differential failed"
                ):
                    self.load(document)

    def test_early_and_late_tolerance_modes_are_step_derived(self):
        early_name = "step/1/field/Ex"
        early_state_name = "step/1/state/Ex/0-Cpml/values"
        late_name = "step/20/field/Ex"
        early = differential.frozen_array_comparison_contract(
            self.manifest,
            "single-gpu-cuda",
            "single-gpu-3d",
            "cuda:0",
            early_name,
        )
        early_state = differential.frozen_array_comparison_contract(
            self.manifest,
            "single-gpu-cuda",
            "single-gpu-3d",
            "cuda:0",
            early_state_name,
        )
        late = differential.frozen_array_comparison_contract(
            self.manifest,
            "single-gpu-cuda",
            "single-gpu-3d",
            "cuda:0",
            late_name,
        )
        self.assertEqual(early["mode"], differential.ELEMENTWISE_COMPARISON_MODE)
        self.assertEqual(
            differential._active_tolerance_models("single-gpu-3d"),
            (
                "pml",
                "dcp-ade",
                "dcp-plrc",
                "dcp-rc",
                "dielectric",
                "drude",
                "lorentz",
            ),
        )
        self.assertNotIn("dm2", differential._active_tolerance_models("single-gpu-3d"))
        self.assertEqual(
            early,
            {
                "mode": differential.ELEMENTWISE_COMPARISON_MODE,
                "rtol": 5e-12,
                "atol": 5e-13,
            },
        )
        self.assertEqual(
            early_state,
            {
                "mode": differential.ELEMENTWISE_COMPARISON_MODE,
                **self.manifest["tolerances"]["torch"]["pml"]["float64"],
            },
        )
        self.assertEqual(late["mode"], differential.NORMALIZED_COMPARISON_MODE)
        self.assertEqual(
            late["absolute_scale_floor"],
            differential.NORMALIZED_ABSOLUTE_SCALE_FLOOR,
        )

        reference = np.full(16, 0.25, dtype=np.float64)
        candidate = reference.copy()
        candidate[0] += 1e-7
        suite = differential.frozen_comparison_contract(
            self.manifest, "single-gpu-cuda", "single-gpu-3d", "cuda:0"
        )
        _metrics, artifact_selected_passed = differential._compare_arrays(
            {early_name: reference}, {early_name: candidate}, suite
        )
        self.assertTrue(artifact_selected_passed)
        _metrics, early_passed = differential._compare_arrays(
            {early_name: reference},
            {early_name: candidate},
            suite,
            array_comparisons={early_name: early},
        )
        _metrics, late_passed = differential._compare_arrays(
            {late_name: reference},
            {late_name: candidate},
            suite,
            array_comparisons={late_name: late},
        )
        self.assertFalse(early_passed)
        self.assertTrue(late_passed)

    def test_active_early_field_tolerance_rejects_dm2_sized_drift(self):
        document = self.document()
        arrays = self.arrays(np.float64, "single-gpu-3d")
        arrays["step/1/field/Ex"].flat[0] += 1e-11
        self.refresh_physical(arrays, 1)
        self.rewrite_candidate(document, 1, arrays)
        with self.assertRaisesRegex(ValueError, "recomputed differential failed"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_cross_group_persistent_topology_mutation_is_rejected(self):
        document = self.document()
        arrays = self.arrays(np.float64, "single-gpu-3d")
        indices = arrays["step/2/state/Ex/0-Cpml/indices"]
        indices[[0, 1]] = indices[[1, 0]]
        self.rewrite_reference(document, 1, arrays)
        self.rewrite_candidate(document, 1, arrays)
        with self.assertRaisesRegex(ValueError, "change across capture steps"):
            self.load(document)

    def test_frozen_geometry_rejects_count_preserving_strategy_swap(self):
        document = self.document()
        arrays = self.arrays(np.float64, "single-gpu-3d")
        for step in differential.NORMALIZED_PROJECTION_STEPS:
            cpml = arrays[f"step/{step}/state/Ex/0-Cpml/indices"]
            dielectric = arrays[f"step/{step}/state/Ex/3-Dielectric/indices"]
            cpml_coordinate = cpml[0].copy()
            cpml[0] = dielectric[0]
            dielectric[0] = cpml_coordinate
        self.rewrite_reference(document, 1, arrays)
        self.rewrite_candidate(document, 1, arrays)
        with self.assertRaisesRegex(ValueError, "persistent geometry differs"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_frozen_geometry_digest_inventory_covers_every_differential_case(self):
        frozen = differential.FROZEN_PERSISTENT_GEOMETRY_SHA256_BY_CASE
        self.assertEqual(set(frozen), set(differential.CASE_UPDATER_LABELS))
        self.assertTrue(
            all(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
                for digest in frozen.values()
            )
        )

    def test_schema_v4_is_rejected(self):
        document = self.document()
        document["schema_version"] = 4
        with self.assertRaisesRegex(ValueError, "index identity differs"):
            self.load(document)

    def test_single_gpu_3d_float32_projection_is_rejected(self):
        document = self.document()
        self.rewrite_candidate(
            document,
            1,
            self.arrays(np.float32, "single-gpu-3d"),
        )
        with self.assertRaisesRegex(ValueError, "field dtype differs"):
            self.load(document)

    def test_descriptor_path_digest_and_candidate_are_exact(self):
        for mutation in ("path", "sha256", "candidate_evidence"):
            with self.subTest(mutation=mutation):
                document = self.document()
                descriptor = document["cases"][0]["candidate"][0]
                if mutation == "path":
                    descriptor[mutation] = "../escape.npz"
                elif mutation == "sha256":
                    descriptor[mutation] = "0" * 64
                else:
                    descriptor[mutation] = {
                        **self.candidate,
                        "candidate_git_commit": "b" * 40,
                    }
                with self.assertRaises(ValueError):
                    self.load(document)

    def test_missing_and_extra_npz_arrays_fail_closed(self):
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                document = self.document()
                arrays = self.arrays(np.float32, "single-gpu-2d")
                if mutation == "missing":
                    del arrays["step/100/field/Hz"]
                else:
                    arrays["step/100/state/unexpected"] = np.asarray(
                        [1.0], dtype=np.float32
                    )
                self.rewrite_candidate(document, 0, arrays)
                with self.assertRaisesRegex(ValueError, "NPZ array closure differs"):
                    self.load(document)

    def test_projected_field_shape_is_independent_of_artifact_bytes(self):
        for mutation in ("empty", "reshape"):
            with self.subTest(mutation=mutation):
                document = self.document()
                for rewrite in (self.rewrite_reference, self.rewrite_candidate):
                    arrays = self.arrays(np.float32, "single-gpu-2d")
                    name = "step/100/field/Ex"
                    if mutation == "empty":
                        arrays[name] = np.empty(0, dtype=np.float32)
                    else:
                        arrays[name] = arrays[name].reshape(-1)
                    rewrite(document, 0, arrays)
                with self.assertRaisesRegex(ValueError, "field shape or dtype differs"):
                    self.load(document)

    def test_projected_physical_shape_is_independent_of_artifact_bytes(self):
        document = self.document()
        for rewrite in (self.rewrite_reference, self.rewrite_candidate):
            arrays = self.arrays(np.float64, "single-gpu-3d")
            arrays["step/20/physical/spectrum/Ex"] = np.empty(0, dtype=np.float64)
            rewrite(document, 1, arrays)
        with self.assertRaisesRegex(ValueError, "physical spectrum differs"):
            self.load(document)

    def test_rehashed_equal_role_physical_tampering_is_recomputed(self):
        for name in (
            "step/20/physical/spectrum/Ex",
            "step/20/physical/summary",
        ):
            with self.subTest(name=name):
                document = self.document()
                arrays = self.arrays(np.float64, "single-gpu-3d")
                arrays[name].flat[0] += 0.125
                self.rewrite_reference(document, 1, arrays)
                self.rewrite_candidate(document, 1, arrays)
                with self.assertRaisesRegex(
                    ValueError, "physical (spectrum|summary) differs from"
                ):
                    self.load(document)
                with self.assertRaises(completion.EvidenceError):
                    self.validate_completion_mirror(document)

    def test_projected_persistent_count_and_width_are_manifest_derived(self):
        for mutation in ("empty-values", "missing-index"):
            with self.subTest(mutation=mutation):
                document = self.document()
                for rewrite in (self.rewrite_reference, self.rewrite_candidate):
                    arrays = self.arrays(np.float32, "single-gpu-2d")
                    root = "step/100/state/Ex/0-Cpml"
                    if mutation == "empty-values":
                        arrays[f"{root}/values"] = np.empty(0, dtype=np.complex128)
                    else:
                        arrays[f"{root}/indices"] = arrays[f"{root}/indices"][1:]
                        arrays[f"{root}/values"] = arrays[f"{root}/values"][2:]
                    rewrite(document, 0, arrays)
                message = (
                    "persistent value shape or dtype"
                    if mutation == "empty-values"
                    else "do not cover the complete field"
                )
                with self.assertRaisesRegex(ValueError, message):
                    self.load(document)

    def test_mixed_case_uses_model_scoped_persistent_tolerance(self):
        document = self.document()
        arrays = self.arrays(np.float32, "single-gpu-2d")
        name = "step/100/state/Ex/0-Cpml/values"
        candidate = {key: value.copy() for key, value in arrays.items()}
        candidate[name][0] += np.complex128(1e-4)
        suite = differential.frozen_comparison_contract(
            self.manifest, "single-gpu-cuda", "single-gpu-2d", "cuda:0"
        )
        _metrics, suite_passed = differential._compare_arrays(
            {name: arrays[name]}, {name: candidate[name]}, suite
        )
        self.assertTrue(suite_passed)
        scoped = differential._frozen_array_comparisons(
            self.manifest,
            "single-gpu-cuda",
            "single-gpu-2d",
            "cuda:0",
            arrays,
        )
        _metrics, scoped_passed = differential._compare_arrays(
            arrays, candidate, suite, array_comparisons=scoped
        )
        self.assertFalse(scoped_passed)
        self.rewrite_candidate(document, 0, candidate)
        with self.assertRaisesRegex(
            ValueError, "persistent indices overlap|recomputed differential failed"
        ):
            self.load(document)

    def test_raw_numeric_failure_cannot_be_hidden_by_pass_flags(self):
        document = self.document()
        arrays = self.arrays(np.float32, "single-gpu-2d")
        arrays["step/100/field/Ex"] = arrays["step/100/field/Ex"] + np.float32(1.0)
        self.rewrite_candidate(document, 0, arrays)
        document["cases"][0]["metrics"] = {
            name: 1.0 for name in document["cases"][0]["metrics"]
        }
        document["cases"][0]["passed"] = True
        document["passed"] = True
        with self.assertRaisesRegex(
            ValueError, "persistent indices overlap|recomputed differential failed"
        ):
            self.load(document)

    def test_integer_arrays_are_exact_even_under_float_tolerance(self):
        document = self.document()
        arrays = self.arrays(np.float32, "single-gpu-2d")
        indices = document["cases"][0]["persistent_arrays"][0]
        self.assertTrue(indices.endswith("/indices"))
        arrays[indices][0] += 1
        self.rewrite_candidate(document, 0, arrays)
        with self.assertRaisesRegex(
            ValueError, "persistent indices overlap|recomputed differential failed"
        ):
            self.load(document)

    def test_normalized_step20_field_failure_is_recomputed(self):
        document = self.document()
        arrays = self.arrays(np.float64, "single-gpu-3d")
        arrays["step/20/field/Ex"] = arrays["step/20/field/Ex"] + 1.0
        self.refresh_physical(arrays, 20)
        self.rewrite_candidate(document, 1, arrays)
        with self.assertRaisesRegex(ValueError, "recomputed differential failed"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_normalized_step20_physical_array_cannot_be_self_omitted(self):
        document = self.document()
        omitted = "step/20/physical/summary"
        for rewrite in (self.rewrite_reference, self.rewrite_candidate):
            arrays = self.arrays(np.float64, "single-gpu-3d")
            del arrays[omitted]
            rewrite(document, 1, arrays)
        document["cases"][1]["physical_arrays"].remove(omitted)
        with self.assertRaisesRegex(ValueError, "physical array closure"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_normalized_persistent_suffix_closure_cannot_change_by_step(self):
        document = self.document()
        omitted = "step/20/state/Ex/0-Cpml/values"
        for rewrite in (self.rewrite_reference, self.rewrite_candidate):
            arrays = self.arrays(np.float64, "single-gpu-3d")
            del arrays[omitted]
            rewrite(document, 1, arrays)
        document["cases"][1]["persistent_arrays"].remove(omitted)
        with self.assertRaisesRegex(ValueError, "persistent array closure"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_common_persistent_suffix_cannot_be_self_omitted(self):
        document = self.document()
        suffix = "state/Ex/0-Cpml/values"
        omitted = {f"step/{step}/{suffix}" for step in (20, 100)}
        for rewrite in (self.rewrite_reference, self.rewrite_candidate):
            arrays = self.arrays(np.float64, "single-gpu-3d")
            for name in omitted:
                del arrays[name]
            rewrite(document, 1, arrays)
        document["cases"][1]["persistent_arrays"] = [
            name
            for name in document["cases"][1]["persistent_arrays"]
            if name not in omitted
        ]
        with self.assertRaisesRegex(ValueError, "persistent array closure"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_source_contract_array_cannot_be_self_omitted(self):
        document = self.document()
        for rewrite in (self.rewrite_reference, self.rewrite_candidate):
            arrays = self.arrays(np.float32, "single-gpu-2d")
            del arrays[differential.SOURCE_CONTRACT_ARRAY]
            rewrite(document, 0, arrays)
        document["cases"][0]["contract_arrays"].remove(
            differential.SOURCE_CONTRACT_ARRAY
        )
        with self.assertRaisesRegex(ValueError, "contract array closure"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_source_raw_proof_cannot_be_self_omitted(self):
        document = self.document()
        for rewrite in (self.rewrite_reference, self.rewrite_candidate):
            arrays = self.arrays(np.float32, "single-gpu-2d")
            del arrays[differential.SOURCE_PROOF_ARRAY]
            rewrite(document, 0, arrays)
        document["cases"][0]["contract_arrays"].remove(differential.SOURCE_PROOF_ARRAY)
        with self.assertRaisesRegex(ValueError, "contract array closure"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_source_raw_proof_mutation_is_semantically_revalidated(self):
        document = self.document()
        arrays = self.arrays(np.float32, "single-gpu-2d")
        proof = json.loads(arrays[differential.SOURCE_PROOF_ARRAY].tobytes())
        amplitude = proof["captures"][0]["candidate"]["live"]["overwrite_amplitudes"]
        changed = bytearray.fromhex(amplitude["data_hex"])
        changed[0] ^= 1
        amplitude["data_hex"] = changed.hex()
        workload = differential._workload(self.manifest, "single-gpu-2d")
        proof["candidate_preimage_sha256"] = differential._source_role_preimage_sha256(
            workload, "candidate", proof["captures"]
        )
        proof_array = np.frombuffer(
            differential._canonical_source_bytes(proof), dtype=np.uint8
        ).copy()
        for rewrite in (self.rewrite_reference, self.rewrite_candidate):
            changed_arrays = {name: value.copy() for name, value in arrays.items()}
            changed_arrays[differential.SOURCE_PROOF_ARRAY] = proof_array.copy()
            rewrite(document, 0, changed_arrays)
        with self.assertRaisesRegex(ValueError, "overwrite semantics"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_rehashed_native_source_index_widening_is_rejected(self):
        document = self.document()
        arrays = self.arrays(np.float32, "single-gpu-2d")
        proof = json.loads(arrays[differential.SOURCE_PROOF_ARRAY].tobytes())
        native_indices = differential._source_array_from_record(
            proof["captures"][0]["native"]["indices"],
            "native PointSource indices",
        )
        self.assertEqual(native_indices.dtype, np.dtype(np.intc))
        proof["captures"][0]["native"]["indices"] = differential._source_array_record(
            native_indices.astype(np.int64), "widened native indices"
        )
        workload = differential._workload(self.manifest, "single-gpu-2d")
        proof["reference_preimage_sha256"] = differential._source_role_preimage_sha256(
            workload, "reference", proof["captures"]
        )
        arrays[differential.SOURCE_PROOF_ARRAY] = np.frombuffer(
            differential._canonical_source_bytes(proof), dtype=np.uint8
        ).copy()
        self.rewrite_reference(document, 0, arrays)
        self.rewrite_candidate(document, 0, arrays)
        with self.assertRaisesRegex(ValueError, "native PointSource target differs"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_source_raw_proof_parser_rejects_malformed_preimages(self):
        workload = differential._workload(self.manifest, "single-gpu-2d")
        capture_steps = workload.get(
            "capture_steps", self.manifest["reference"]["capture_steps"]
        )
        source = self.arrays(np.float32, "single-gpu-2d")[
            differential.SOURCE_PROOF_ARRAY
        ]
        original = json.loads(source.tobytes())
        for mutation in (
            "dtype",
            "hex-length",
            "extra-key",
            "same-digest",
            "arbitrary-digests",
        ):
            with self.subTest(mutation=mutation):
                proof = copy.deepcopy(original)
                record = proof["captures"][0]["candidate"]["live"][
                    "overwrite_amplitudes"
                ]
                if mutation == "dtype":
                    record["dtype"] = "object"
                elif mutation == "hex-length":
                    record["data_hex"] = record["data_hex"][:-1]
                elif mutation == "extra-key":
                    record["unexpected"] = True
                elif mutation == "same-digest":
                    proof["candidate_preimage_sha256"] = proof[
                        "reference_preimage_sha256"
                    ]
                else:
                    proof["reference_preimage_sha256"] = "1" * 64
                    proof["candidate_preimage_sha256"] = "2" * 64
                raw = differential._canonical_source_bytes(proof)
                message = (
                    "preimage digest differs"
                    if mutation == "arbitrary-digests"
                    else None
                )
                context = (
                    self.assertRaisesRegex(ValueError, message)
                    if message is not None
                    else self.assertRaises(ValueError)
                )
                with context:
                    differential._validate_source_raw_proof_bytes(
                        raw,
                        workload,
                        capture_steps,
                        "float32",
                        self.manifest["reference"]["precondition_steps"],
                    )

    def test_source_raw_proof_clock_is_workload_derived_after_rehash(self):
        workload = differential._workload(self.manifest, "single-gpu-2d")
        capture_steps = workload.get(
            "capture_steps", self.manifest["reference"]["capture_steps"]
        )
        source = self.arrays(np.float32, "single-gpu-2d")[
            differential.SOURCE_PROOF_ARRAY
        ]
        proof = json.loads(source.tobytes())
        changed_dt = differential._expected_point_source_time_step(workload) * 0.75
        amplitude = float(workload.get("source_amp", 1e-3))
        eps_inf, mu_inf = differential._expected_point_source_medium(workload)
        precondition_steps = self.manifest["reference"]["precondition_steps"]
        for capture in proof["captures"]:
            clock_step = precondition_steps + capture["step"]
            time = clock_step * changed_dt
            time_array = np.asarray([clock_step, time, changed_dt], dtype=np.float64)
            capture["native"]["time"] = differential._source_array_record(
                time_array, "changed native time"
            )
            capture["candidate"]["time"] = differential._source_array_record(
                time_array, "changed candidate time"
            )
            native_values = np.asarray(
                [
                    amplitude,
                    eps_inf,
                    mu_inf,
                    differential._continuous_point_source_value(
                        time, bool(workload.get("complex"))
                    ),
                ],
                dtype=np.complex128,
            )
            capture["native"]["values"] = differential._source_array_record(
                native_values, "changed native values"
            )
            evaluated = differential._continuous_point_source_value(
                time - 0.5 * changed_dt, bool(workload.get("complex"))
            )
            capture["candidate"]["live"]["_overwrite_values"] = (
                differential._source_array_record(
                    np.asarray([[amplitude * float(evaluated)]], dtype=np.float32),
                    "changed candidate values",
                )
            )
        proof["reference_preimage_sha256"] = differential._source_role_preimage_sha256(
            workload, "reference", proof["captures"]
        )
        proof["candidate_preimage_sha256"] = differential._source_role_preimage_sha256(
            workload, "candidate", proof["captures"]
        )
        with self.assertRaisesRegex(ValueError, "time differs from the capture clock"):
            differential._validate_source_raw_proof_bytes(
                differential._canonical_source_bytes(proof),
                workload,
                capture_steps,
                "float32",
                precondition_steps,
            )

    def test_source_raw_proof_rejects_rehashed_subtolerance_clock_drift(self):
        workload = differential._workload(self.manifest, "single-gpu-2d")
        capture_steps = workload.get(
            "capture_steps", self.manifest["reference"]["capture_steps"]
        )
        source = self.arrays(np.float32, "single-gpu-2d")[
            differential.SOURCE_PROOF_ARRAY
        ]
        proof = json.loads(source.tobytes())
        capture = next(item for item in proof["captures"] if item["step"] == 20)
        precondition_steps = self.manifest["reference"]["precondition_steps"]
        clock_step = precondition_steps + capture["step"]
        dt = differential._expected_point_source_time_step(workload)
        changed_time = np.nextafter(clock_step * dt, np.inf)
        changed_clock = np.asarray([clock_step, changed_time, dt], dtype=np.float64)
        capture["native"]["time"] = differential._source_array_record(
            changed_clock, "drifted native time"
        )
        capture["candidate"]["time"] = differential._source_array_record(
            changed_clock, "drifted candidate time"
        )
        native_values = differential._source_array_from_record(
            capture["native"]["values"], "native values"
        )
        native_values[3] = differential._continuous_point_source_value(
            changed_time, bool(workload.get("complex"))
        )
        capture["native"]["values"] = differential._source_array_record(
            native_values, "drifted native values"
        )
        candidate_values = differential._source_array_from_record(
            capture["candidate"]["live"]["_overwrite_values"],
            "candidate overwrite values",
        )
        oscillator = differential._continuous_point_source_value(
            changed_time - 0.5 * dt, bool(workload.get("complex"))
        )
        amplitude = float(workload.get("source_amp", 1e-3))
        candidate_values[0, 0] = amplitude * float(oscillator)
        capture["candidate"]["live"]["_overwrite_values"] = (
            differential._source_array_record(
                candidate_values, "drifted candidate values"
            )
        )
        proof["reference_preimage_sha256"] = differential._source_role_preimage_sha256(
            workload, "reference", proof["captures"]
        )
        proof["candidate_preimage_sha256"] = differential._source_role_preimage_sha256(
            workload, "candidate", proof["captures"]
        )
        with self.assertRaisesRegex(ValueError, "time differs from the capture clock"):
            differential._validate_source_raw_proof_bytes(
                differential._canonical_source_bytes(proof),
                workload,
                capture_steps,
                "float32",
                precondition_steps,
            )

    def test_normalized_contract_requires_each_linf_and_l2_limit(self):
        contract = {
            "mode": differential.NORMALIZED_COMPARISON_MODE,
            "linf_limit": 1e-6,
            "l2_limit": 1e-6,
            "absolute_scale_floor": 1e-12,
            "all_zero_reference": "exact",
        }
        reference = {"value": np.ones(4, dtype=np.float64)}
        linf_failure = {"value": reference["value"].copy()}
        linf_failure["value"][0] += 1.5e-6
        metrics, passed = differential._compare_arrays(
            reference, linf_failure, contract
        )
        self.assertFalse(passed)
        self.assertGreater(metrics["maximum_normalized_linf_error"], 1e-6)
        self.assertLess(metrics["maximum_normalized_l2_error"], 1e-6)

        l2_reference = {"value": np.asarray([1.0, 0.0, 0.0, 0.0])}
        l2_failure = {
            "value": l2_reference["value"]
            + np.asarray([0.9e-6, 0.9e-6, 0.9e-6, 0.9e-6])
        }
        metrics, passed = differential._compare_arrays(
            l2_reference, l2_failure, contract
        )
        self.assertFalse(passed)
        self.assertLess(metrics["maximum_normalized_linf_error"], 1e-6)
        self.assertGreater(metrics["maximum_normalized_l2_error"], 1e-6)

    def test_normalized_contract_keeps_zero_and_integer_arrays_exact(self):
        contract = {
            "mode": differential.NORMALIZED_COMPARISON_MODE,
            "linf_limit": 1e-6,
            "l2_limit": 1e-6,
            "absolute_scale_floor": 2e-12,
            "all_zero_reference": "exact",
        }
        zero = {"value": np.zeros(3, dtype=np.float64)}
        tiny = {"value": np.asarray([0.0, 0.0, 1e-30])}
        _metrics, passed = differential._compare_arrays(zero, tiny, contract)
        self.assertFalse(passed)

        elementwise = {
            "mode": differential.ELEMENTWISE_COMPARISON_MODE,
            "rtol": 1.0,
            "atol": 1.0,
        }
        _metrics, passed = differential._compare_arrays(zero, tiny, elementwise)
        self.assertFalse(passed)

        integers = {"value": np.asarray([1, 2], dtype=np.uint64)}
        changed = {"value": np.asarray([1, 3], dtype=np.uint64)}
        _metrics, passed = differential._compare_arrays(integers, changed, contract)
        self.assertFalse(passed)

    def test_canonical_source_contract_is_exact_and_workload_bound(self):
        document = self.document()
        arrays = self.arrays(np.float32, "single-gpu-2d")
        payload = json.loads(arrays[differential.SOURCE_CONTRACT_ARRAY].tobytes())
        payload["sources"][0]["model"]["id"] = 1
        arrays[differential.SOURCE_CONTRACT_ARRAY] = np.frombuffer(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(),
            dtype=np.uint8,
        ).copy()
        self.rewrite_candidate(document, 0, arrays)
        with self.assertRaisesRegex(ValueError, "PointSource contract"):
            self.load(document)
        with self.assertRaises(completion.EvidenceError):
            self.validate_completion_mirror(document)

    def test_native_source_raw_amplitude_and_waveform_are_bound(self):
        archive, metadata, workload, expected = self.native_point_source_inputs()
        differential._validate_native_point_source(
            archive, metadata, workload, expected
        )

        key = "step/20/source/Ex/0-PointSourceEx/values"
        for index in (0, 1, 2, 3):
            with self.subTest(index=index):
                changed = _Archive(
                    {name: value.copy() for name, value in archive.items()}
                )
                changed[key][index] += 0.125
                with self.assertRaisesRegex(
                    ValueError, "amplitude or waveform differs"
                ):
                    differential._validate_native_point_source(
                        changed, metadata, workload, expected
                    )

    def test_candidate_source_packed_and_live_state_are_cross_checked(self):
        archive, metadata, workload, expected = self.candidate_point_source_inputs()
        differential._validate_candidate_point_source(
            archive, metadata, workload, expected
        )

        step = "20"
        packed_root = f"step/{step}/source/Ex/0-PointSourceEx"
        live_root = f"torch/step/{step}/sources/batches/0"
        mutations = {
            "packed-index": (f"{packed_root}/indices", (0, 0), 1),
            "packed-word": (f"{packed_root}/values", (8,), np.uint64(1)),
            "live-amplitude": (
                f"{live_root}/overwrite_amplitudes",
                (0,),
                np.float32(0.125),
            ),
        }
        for name, (key, index, delta) in mutations.items():
            with self.subTest(name=name):
                changed = _Archive(
                    {array_name: value.copy() for array_name, value in archive.items()}
                )
                if name == "packed-word":
                    changed[key][index] ^= delta
                else:
                    changed[key][index] += delta
                with self.assertRaisesRegex(
                    ValueError,
                    "packed/live semantics|overwrite semantics",
                ):
                    differential._validate_candidate_point_source(
                        changed, metadata, workload, expected
                    )

    def test_precision_limitation_is_recomputed_from_native_step100_fields(self):
        document = self.document()
        limitation = document["cases"][1]["precision_limitation"]
        self.assertEqual(limitation["reference_step"], 100)
        self.assertGreater(
            limitation["reference_field_max_abs"], float(np.finfo(np.float32).max)
        )

        reference = self.arrays(np.float64, "single-gpu-3d")
        reference["step/100/field/Ex"].fill(0.25)
        self.refresh_physical(reference, 100)
        candidate = {name: value.copy() for name, value in reference.items()}
        self.rewrite_reference(document, 1, reference)
        self.rewrite_candidate(document, 1, candidate)
        with self.assertRaisesRegex(ValueError, "dynamic-range limitation"):
            self.load(document)
        with self.assertRaisesRegex(
            completion.EvidenceError,
            "range limitation|independently recomputed",
        ):
            self.validate_completion_mirror(document)

    def test_raw_source_projection_name_is_rejected(self):
        document = self.document()
        document["cases"][0]["persistent_arrays"].append(
            "step/100/source/Ex/0-PointSourceEx/values"
        )
        with self.assertRaisesRegex(ValueError, "persistent array (category|closure)"):
            self.load(document)

    def test_projection_paths_cannot_be_reused_across_cases(self):
        document = self.document()
        document["cases"][1]["reference"][0] = copy.deepcopy(
            document["cases"][0]["reference"][0]
        )
        with self.assertRaisesRegex(ValueError, "reuses artifact path or bytes"):
            self.load(document)

    def test_paired_cpu_tolerance_uses_complex128_manifest_entry(self):
        tolerance = differential.frozen_tolerance(
            self.manifest,
            "paired-real",
            "dcp-ade-bloch",
            "cpu",
        )
        self.assertEqual(tolerance, {"rtol": 1e-11, "atol": 1e-12})

    def test_builder_output_round_trips_through_strict_loader(self):
        required = differential.expected_records(self.manifest, "single-gpu-cuda")
        source_directory = self.root / "sources"
        source_directory.mkdir()
        references = []
        candidates = []
        metadata = {}
        provenance = {
            "commit": self.candidate["candidate_git_commit"],
            "git_status": "",
            "clean": True,
        }
        for expected in required:
            case = expected["case"]
            reference = source_directory / f"{case}-reference.npz"
            candidate = source_directory / f"{case}-candidate.npz"
            np.savez_compressed(
                reference,
                marker=np.asarray([len(references)], dtype=np.int64),
            )
            np.savez_compressed(
                candidate,
                marker=np.asarray([100 + len(candidates)], dtype=np.int64),
            )
            references.append(reference)
            candidates.append(candidate)
            workload = differential._workload(self.manifest, case)
            metadata[reference] = {"workload": workload}
            metadata[candidate] = {
                "workload": workload,
                "backend_metadata": {"device": "cuda:0"},
                "provenance": {
                    "source": provenance,
                    "controller": provenance,
                },
            }

        def archive_record(path, _manifest, _role):
            path = Path(path)
            return path, metadata[path]

        def source_preflight(path, _label):
            path = Path(path).resolve()
            return path, differential._source_archive_identity(path)

        def projection(
            _reference,
            _candidate,
            _manifest,
            _scope,
            _expected,
            *,
            group_consumer=None,
            source_bindings=None,
        ):
            (
                projection_groups,
                projection_steps,
                fields,
                physical,
                persistent,
                contracts,
            ) = self.projection_contract(_scope, _expected["case"], _expected["device"])
            left = self.arrays(
                np.dtype(_expected["precision"]),
                _expected["case"],
                scope=_scope,
                device=_expected["device"],
            )
            right = {name: value.copy() for name, value in left.items()}
            if group_consumer is not None:
                for ordinal, group in enumerate(projection_groups):
                    group_consumer(
                        ordinal,
                        group,
                        self.group_arrays(left, group),
                        self.group_arrays(right, group),
                    )
            return (
                None if group_consumer is not None else left,
                None if group_consumer is not None else right,
                self.candidate,
                projection_steps,
                projection_groups,
                fields,
                physical,
                persistent,
                contracts,
            )

        with (
            mock.patch.object(
                differential,
                "_preflight_source_npz",
                side_effect=source_preflight,
            ),
            mock.patch.object(
                differential.torch_correctness,
                "_archive_record",
                side_effect=archive_record,
            ),
            mock.patch.object(
                differential,
                "_projection_arrays",
                side_effect=projection,
            ),
        ):
            document = differential.build_differential_evidence(
                references,
                candidates,
                self.manifest,
                self.candidate,
                scope="single-gpu-cuda",
                descriptor_root=self.root,
                output_directory=self.root / "artifacts",
            )

        for record, reference, candidate in zip(
            document["cases"], references, candidates, strict=True
        ):
            self.assertEqual(record["reference_source"], self.descriptor(reference))
            self.assertEqual(record["candidate_source"], self.descriptor(candidate))
        result = self.load(document)
        self.assertTrue(result["passed"])
        self.assertEqual(
            [record["case"] for record in result["cases"]],
            ["single-gpu-2d", "single-gpu-3d"],
        )

    def test_candidate_cli_writes_exact_three_key_binding(self):
        output = self.root / "candidate.json"
        argv = [
            "issue123_differential",
            "candidate",
            "--output",
            str(output),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                differential.host_contract,
                "_command_text",
                side_effect=(
                    self.candidate["candidate_git_commit"],
                    self.candidate["candidate_git_status"],
                ),
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(differential.main(), 0)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), self.candidate)


if __name__ == "__main__":
    unittest.main()
