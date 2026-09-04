from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import io
import json
import math
import os
import stat
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import gmes
from benchmarks import issue123_completion as completion
from benchmarks import issue123_operations as operations
from benchmarks import issue123_privacy as privacy
from benchmarks import torch_tuning
from benchmarks import two_gpu_failure_evidence as failure_evidence


def _synthetic_common_host_fixture():
    """Return a reserved test-only host contract, never a captured identity."""

    return {
        "hostname": "fixture-host.invalid",
        "platform": "Linux-fixture.invalid",
        "os": {
            "system": "Linux",
            "release": "fixture-release.invalid",
            "machine": "fixture-machine.invalid",
        },
        "python": "3.14.0",
        "cxx_version": "fixture-cxx.invalid",
        "swig_version": "fixture-swig.invalid",
        "uv_lock_sha256": "f" * 64,
    }


def _synthetic_baseline_identity_fixture():
    """Return minimal reserved raw material for private commitment tests."""

    thread_environment = {"OMP_NUM_THREADS": "1"}
    environment = {
        "hostname": "fixture-candidate-host.invalid",
        "platform": "Linux-baseline-fixture.invalid",
        "python": "3.14.0",
        "torch": "2.13.0+cpu",
        "cuda_runtime": None,
        "devices": [],
        "cpu_count": 8,
        "cpu_affinity": list(range(8)),
        "cpu_count_physical_affinity": 4,
        "cpu_topology": "fixture-cpu-topology.invalid",
        "cpu_model": "fixture-cpu-model.invalid",
        "gpu_topology": None,
        "thread_environment": thread_environment,
    }
    return environment, thread_environment


class Issue123CompletionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.external_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.external_temporary.cleanup)
        self.external_directory = Path(self.external_temporary.name)
        manifest_raw = completion.DEFAULT_MANIFEST.read_bytes()
        self.manifest = json.loads(manifest_raw)
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
        self.common_host = _synthetic_common_host_fixture()
        self.cpu_runtime = {"torch": "2.9.0+cpu", "cuda_runtime": None}
        self.cuda_runtime = {"torch": "2.9.0+cu130", "cuda_runtime": "13.0"}
        self.host_contract = {
            "schema_version": 2,
            "common_identity": copy.deepcopy(self.common_host),
            "runtime_identity": copy.deepcopy(self.cpu_runtime),
        }
        self.cuda_host_contract = {
            "schema_version": 2,
            "common_identity": copy.deepcopy(self.common_host),
            "runtime_identity": copy.deepcopy(self.cuda_runtime),
        }

    def write_bytes(self, name, raw, *, media_type=completion.MEDIA_TYPE_BINARY):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "media_type": media_type,
            "candidate_evidence": self.candidate,
        }

    def write_json(self, name, value):
        return self.write_bytes(
            name,
            (json.dumps(value, sort_keys=True) + "\n").encode(),
            media_type=completion.MEDIA_TYPE_JSON,
        )

    def write_private_sdist(
        self,
        name="packages/gmes-0.10.0.tar.gz",
        *,
        payload=b"Metadata-Version: 2.4\nName: gmes\n",
        uid=1001,
        gid=1002,
        uname="builder",
        gname="builders",
        symlink=False,
    ):
        buffer = io.BytesIO()
        with tarfile.open(
            fileobj=buffer, mode="w:gz", format=tarfile.USTAR_FORMAT
        ) as archive:
            info = tarfile.TarInfo("gmes-0.10.0/PKG-INFO")
            info.uid = uid
            info.gid = gid
            info.uname = uname
            info.gname = gname
            info.mtime = 0
            if symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                info.size = 0
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return self.write_bytes(
            name, buffer.getvalue(), media_type=completion.MEDIA_TYPE_GZIP
        )

    @staticmethod
    def private_sdist_result(raw, members):
        return privacy._ValidatedPrivateSdist(
            archive_size=len(raw),
            archive_sha256=hashlib.sha256(raw).hexdigest(),
            members=tuple(members),
            total_member_bytes=sum(member.size for member in members),
            physical_ordinary_count=len(members),
            logical_member_count=len(members),
        )

    def runtime_receipt(self, name, runtime_mode, artifacts):
        value = {
            "schema_version": 1,
            "kind": completion.RUNTIME_RECEIPT_KIND,
            "final_sha": self.candidate["candidate_git_commit"],
            "manifest_sha256": self.candidate["manifest_sha256"],
            "workflow": {
                "repository": "ruddyscent/gmes",
                "run_id": 100,
                "run_attempt": 1,
                "job_id": 200,
                "job_name": "issue123-runtime-evidence",
            },
            "profiler_witness": {
                "name": f"{name}-profiler.json",
                "sha256": "f" * 64,
                "size_bytes": 1,
                "media_type": "application/json",
            },
            "runtime_mode": copy.deepcopy(runtime_mode),
            "candidate_archives": [
                {
                    "case": artifact["case"],
                    "sha256": artifact["candidate"]["sha256"],
                    "size_bytes": artifact["candidate"]["size_bytes"],
                }
                for artifact in artifacts
            ],
        }
        raw = completion._canonical_json_bytes(value)
        descriptor = self.write_bytes(
            f"correctness/{name}-runtime-receipt.json",
            raw,
            media_type=completion.MEDIA_TYPE_JSON,
        )
        path_independent = {
            key: descriptor[key]
            for key in ("path", "sha256", "size_bytes", "media_type")
        }
        external_path = self.external_directory / f"{name}-runtime-receipt.json"
        external_path.write_bytes(raw)
        return path_independent, completion.LoadedArtifact(
            path_independent,
            external_path,
            raw,
            value,
        )

    def frozen_material_plan(self, workload, precision):
        topology, target_counts = completion._frozen_cuda_material_contract(
            workload, "test"
        )
        records = []
        for component, signatures, counts in zip(
            completion.FIELD_ARRAYS, topology, target_counts, strict=True
        ):
            buckets = []
            for (model, state_shape), targets in zip(signatures, counts, strict=True):
                if model == "dielectric":
                    state_width = 0
                elif model in {"cpml", "dm2"}:
                    state_width = sum(state_shape)
                elif model in {"drude", "lorentz"}:
                    state_width = 2 * state_shape[0]
                elif model == "dcp-ade":
                    state_width = 1 + 2 * state_shape[0] + 2 * state_shape[1]
                else:
                    state_width = state_shape[0] + 2 * state_shape[1]
                buckets.append(
                    {
                        "signature": {
                            "model": model,
                            "component": component,
                            "precision": precision,
                            "state_shape": list(state_shape),
                        },
                        "targets": targets,
                        "state_width": state_width,
                    }
                )
            records.append(
                {
                    "component": component,
                    "launches": len(buckets),
                    "buckets": buckets,
                }
            )
        return records

    def write_top_index(self):
        manifest_raw = completion.DEFAULT_MANIFEST.read_bytes()
        manifest_descriptor = self.write_bytes(
            "manifest/native_oracle_workloads.json",
            manifest_raw,
            media_type=completion.MEDIA_TYPE_JSON,
        )
        value = {
            "schema_version": completion.INDEX_SCHEMA_VERSION,
            "kind": completion.INDEX_KIND,
            "issue": 123,
            "bundle": {
                "format": completion.BUNDLE_FORMAT,
                "path_contract": completion.PATH_CONTRACT,
                "artifact_count": 1,
                "artifact_bytes": len(manifest_raw),
            },
            "candidate_evidence": self.candidate,
            "manifest": manifest_descriptor,
            "payloads": [manifest_descriptor],
            "artifacts": {
                "cpu": {},
                "policy_paired_real": {},
                "single_gpu": {},
                "two_gpu": {},
                "macos": {},
                "operations": {},
            },
        }
        path = self.directory / "index.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def differential_fixture(self):
        from tests.test_issue123_differential import (
            Issue123DifferentialEvidenceTest,
        )

        fixture = Issue123DifferentialEvidenceTest(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        geometry_patch = mock.patch.object(
            completion,
            "FROZEN_DIFFERENTIAL_PERSISTENT_GEOMETRY_SHA256_BY_CASE",
            fixture.frozen_geometry,
        )
        geometry_patch.start()
        self.addCleanup(geometry_patch.stop)
        return fixture

    @staticmethod
    def differential_group_arrays(fixture, record, role, ordinal=0):
        path = fixture.root / record[role][ordinal]["path"]
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}

    def differential_record_arrays(self, fixture, record, role):
        arrays = {}
        for ordinal in range(len(record["projection_groups"])):
            group = self.differential_group_arrays(fixture, record, role, ordinal)
            for name, value in group.items():
                previous = arrays.get(name)
                if previous is not None:
                    self.assertTrue(np.array_equal(previous, value), name)
                else:
                    arrays[name] = value
        return arrays

    def patched_validators(self, *, different_gpu_environment=False):
        gpu_environment = {
            "host_contract": copy.deepcopy(self.cuda_host_contract),
            "common_host_identity": copy.deepcopy(self.common_host),
            "platform": self.common_host["platform"],
            "python": self.common_host["python"],
            "torch": self.cuda_runtime["torch"],
            "cuda_runtime": self.cuda_runtime["cuda_runtime"],
            "devices": [
                {
                    "index": 0,
                    "name": "fixture-device-zero.invalid",
                    "memory_bytes": 1,
                },
                {
                    "index": 1,
                    "name": "fixture-device-one.invalid",
                    "memory_bytes": 1,
                },
            ],
            "topology": "fixture-device-topology.invalid",
        }
        gpu_environment["runtime_identity"] = {
            **copy.deepcopy(self.cuda_runtime),
            "kind": "cuda",
            "devices": copy.deepcopy(gpu_environment["devices"]),
            "topology": gpu_environment["topology"],
        }
        two_environment = copy.deepcopy(gpu_environment)
        if different_gpu_environment:
            two_environment["cuda_runtime"] = "different"
            two_environment["runtime_identity"]["cuda_runtime"] = "different"
        cpu_raw = {
            mode: {
                case: ([1.0] * 15 if case.startswith("cpu-crossover") else [2.0] * 15)
                for case in completion.CPU_CASES
            }
            for mode in ("one", "physical")
        }
        native_raw = {
            "one": {},
            "physical": {"cpu-large-3d": [4.0] * 15},
        }
        cuda_raw = {case: [1.0] * 15 for case in completion.CUDA_CASES}
        required_cases = [
            case["name"]
            for group in ("correctness", "physical_checks")
            for case in self.manifest[group]
        ]

        def archive_bindings(role):
            return [
                {
                    "case": case,
                    "sha256": hashlib.sha256(f"{role}:{case}".encode()).hexdigest(),
                    "size_bytes": index + 1,
                }
                for index, case in enumerate(required_cases)
            ]

        def complete_archive_bindings(role):
            role_ordinal = {
                "reference": 1,
                "cpu": 2,
                "cuda-eager": 3,
                "cuda-graph": 4,
            }[role]
            return [
                {
                    "case": case,
                    "path": f"correctness/{role}/{index}.npz",
                    "sha256": hashlib.sha256(
                        f"complete:{role}:{case}".encode()
                    ).hexdigest(),
                    "size_bytes": index + 1,
                    "media_type": completion.MEDIA_TYPE_NPZ,
                    "payload_identity": (
                        1,
                        role_ordinal * 1000 + index,
                        index + 1,
                        1,
                    ),
                }
                for index, case in enumerate(required_cases)
            ]

        cpu_archives = archive_bindings("cpu")
        eager_archives = archive_bindings("cuda-eager")
        graph_archives = archive_bindings("cuda-graph")
        reference_archive_bindings = complete_archive_bindings("reference")
        cpu_archive_bindings = complete_archive_bindings("cpu")
        eager_archive_bindings = complete_archive_bindings("cuda-eager")
        graph_archive_bindings = complete_archive_bindings("cuda-graph")
        cpu_by_case = {record["case"]: record for record in cpu_archives}
        eager_by_case = {record["case"]: record for record in eager_archives}

        def source_descriptor(name, digest, size_bytes):
            return {
                "path": f"differential/{name}.npz",
                "sha256": digest,
                "size_bytes": size_bytes,
                "media_type": completion.MEDIA_TYPE_NPZ,
                "candidate_evidence": self.candidate,
            }

        def differential_sources(scope):
            sources = []
            for index, expected in enumerate(
                completion._expected_completion_differential_records(
                    self.manifest, scope
                )
            ):
                case = expected["case"]
                if expected["device"] == "cpu":
                    archive = cpu_by_case[case]
                elif case in eager_by_case:
                    archive = eager_by_case[case]
                else:
                    archive = {
                        "sha256": hashlib.sha256(
                            f"cuda-float64:{case}".encode()
                        ).hexdigest(),
                        "size_bytes": 999,
                    }
                sources.append(
                    {
                        **copy.deepcopy(expected),
                        "reference_source": source_descriptor(
                            f"reference-{case}",
                            hashlib.sha256(f"reference:{case}".encode()).hexdigest(),
                            1000 + index,
                        ),
                        "candidate_source": source_descriptor(
                            f"candidate-{scope}-{index}",
                            archive["sha256"],
                            archive["size_bytes"],
                        ),
                    }
                )
            return sources

        return {
            "_validate_cpu_scope": {
                "candidate_evidence": self.candidate,
                "torch_raw_seconds_per_step": cpu_raw,
                "native_raw_seconds_per_step": native_raw,
                "host_contract": copy.deepcopy(self.host_contract),
                "common_host_identity": copy.deepcopy(self.common_host),
                "runtime_identity": copy.deepcopy(self.cpu_runtime),
                "_correctness_archive_bindings": {
                    "reference": copy.deepcopy(reference_archive_bindings),
                    "candidate": cpu_archive_bindings,
                },
                "correctness_candidate_archives": cpu_archives,
            },
            "_validate_policy_scope": {
                "candidate_evidence": self.candidate,
                "environment": copy.deepcopy(gpu_environment),
                "differential_source_bindings": differential_sources("paired-real"),
            },
            "_validate_single_gpu_scope": {
                "candidate_evidence": self.candidate,
                "environment": copy.deepcopy(gpu_environment),
                "cuda_raw_seconds_per_step": cuda_raw,
                "_correctness_archive_bindings_by_mode": {
                    "eager": {
                        "reference": copy.deepcopy(reference_archive_bindings),
                        "candidate": eager_archive_bindings,
                    },
                    "graph": {
                        "reference": copy.deepcopy(reference_archive_bindings),
                        "candidate": graph_archive_bindings,
                    },
                },
                "correctness_candidate_archives_by_mode": {
                    "eager": eager_archives,
                    "graph": graph_archives,
                },
                "differential_source_bindings": differential_sources("single-gpu-cuda"),
            },
            "_validate_two_gpu_scope": {
                "candidate_evidence": self.candidate,
                "environment": two_environment,
            },
            "_validate_macos_scope": {
                "candidate_evidence": self.candidate,
                "actions_artifact": {
                    "artifact_id": 2,
                    "run_id": 1,
                    "created_at": "2026-08-31T00:01:00Z",
                    "updated_at": "2026-08-31T00:02:00Z",
                },
            },
            "_validate_operations_scope": {
                "candidate_evidence": self.candidate,
                "final_acceptance": False,
                "final_acceptance_authority": completion.OFFLINE_AUTHORITY,
                "macos_job": {
                    "run_id": 1,
                    "started_at": "2026-08-31T00:00:00Z",
                    "completed_at": "2026-08-31T00:03:00Z",
                },
            },
        }

    def patched_runtime_receipts(self, values):
        receipts = [object(), object(), object()]
        sources = values["_validate_single_gpu_scope"]["differential_source_bindings"]
        for source, mode in zip(
            sources,
            completion.SINGLE_GPU_DIFFERENTIAL_RUNTIME_MODES,
            strict=True,
        ):
            _descriptor, receipt = self.runtime_receipt(
                f"evaluator-{source['case']}",
                mode,
                [
                    {
                        "case": source["case"],
                        "candidate": source["candidate_source"],
                    }
                ],
            )
            receipts.append(receipt)
        return tuple(receipts)

    def evaluate_with_patches(self, values):
        index = self.write_top_index()
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    completion,
                    "_load_trusted_runtime_receipts",
                    return_value=self.patched_runtime_receipts(values),
                )
            )
            for name, value in values.items():
                stack.enter_context(
                    mock.patch.object(completion, name, return_value=value)
                )
            return completion.evaluate_completion(index)

    def live_test_inputs(self, suffix="live"):
        root = self.external_directory / suffix
        root.mkdir()
        runtime_receipts = []
        for ordinal, role in enumerate(completion.RUNTIME_RECEIPT_ROLES):
            path = root / f"runtime-{ordinal}-{role}.json"
            path.write_bytes(
                completion._compact_canonical_json_bytes(
                    {"ordinal": ordinal, "role": role, "suffix": suffix}
                )
            )
            runtime_receipts.append(path)
        policy = root / "trusted-publication-policy.json"
        policy.write_bytes(
            completion._compact_canonical_json_bytes(
                {"kind": "fixture-publication-policy", "suffix": suffix}
            )
        )
        assets = {}
        for ordinal, (role, name) in enumerate(
            operations.TECHNICAL_RELEASE_ASSETS.items()
        ):
            path = root / name
            path.write_bytes(f"{suffix}:{ordinal}:{role}".encode())
            assets[role] = path
        reopened_root = self.external_directory / f"{suffix}-reopened-b1"
        reopened_root.mkdir()
        reopened_index = reopened_root / "index.json"
        reopened_index.write_bytes((self.directory / "index.json").read_bytes())
        authority_root = self.external_directory / f"{suffix}-authority"
        authority_root.mkdir(mode=0o700)
        authority_files = {}
        for name, raw in (
            ("protected-openings.json", f"{suffix}:openings".encode()),
            ("b0-reopen.json", f"{suffix}:b0".encode()),
            ("b1-reopen.json", f"{suffix}:b1".encode()),
        ):
            path = authority_root / name
            path.write_bytes(raw)
            path.chmod(0o600)
            authority_files[name] = path
        return {
            "reopened_index": reopened_index,
            "protected_openings": authority_files["protected-openings.json"],
            "pre_ack_bundle_reopen_receipt": authority_files["b0-reopen.json"],
            "final_bundle_reopen_receipt": authority_files["b1-reopen.json"],
            "runtime_receipt_paths": runtime_receipts,
            "publication_policy": policy,
            "publication_policy_sha256": hashlib.sha256(
                policy.read_bytes()
            ).hexdigest(),
            "publication_assets": assets,
            "output_directory": self.external_directory / f"{suffix}-output",
        }

    @staticmethod
    def authority_gate_fixture():
        post_bundle_expectation = operations.AuthenticatedPostBundleExpectation(
            checked_lines=operations.FINAL_CHECKLIST_CHECKED,
            o0_canonical_response_sha256="0" * 64,
            o1_canonical_response_sha256="1" * 64,
            o1_body_sha256="2" * 64,
            o1_updated_at="2026-09-03T01:31:00Z",
            b0_inventory_root="3" * 64,
            b0_reopen_receipt_sha256="4" * 64,
            b0_reopened_at="2026-09-03T01:30:00Z",
            checklist_transition_sha256="7" * 64,
        )
        return {
            "final_bundle_inventory_root": "5" * 64,
            "post_bundle_expectation": post_bundle_expectation,
            "post_bundle_result": {
                "pre_acknowledgment_receipt": {
                    "schema_version": completion.BUNDLE_REOPEN_RECEIPT_VERSION,
                    "kind": completion.BUNDLE_REOPEN_RECEIPT_KIND,
                    "size_bytes": 101,
                    "sha256": "4" * 64,
                    "observed_at": "2026-09-03T01:30:00Z",
                    "bundle_inventory_root": "3" * 64,
                },
                "final_reopen_receipt": {
                    "schema_version": completion.BUNDLE_REOPEN_RECEIPT_VERSION,
                    "kind": completion.BUNDLE_REOPEN_RECEIPT_KIND,
                    "size_bytes": 102,
                    "sha256": "6" * 64,
                    "observed_at": "2026-09-03T01:32:00Z",
                    "bundle_inventory_root": "5" * 64,
                },
            },
        }

    @staticmethod
    def baseline_descriptors_fixture():
        return operations.PRODUCTION_BASELINE_AUTHORITY_SET

    @classmethod
    def baseline_validation_fixture(cls):
        observed_at = (
            dt.datetime.now(dt.UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        release_id = 42
        api_root = f"https://api.github.com/repos/{operations.REPOSITORY}"
        web_root = f"https://github.com/{operations.REPOSITORY}"
        ledger = []
        for ordinal, descriptor in enumerate(
            cls.baseline_descriptors_fixture().assets, start=101
        ):
            ledger.append(
                {
                    "thread_mode": descriptor.thread_mode,
                    "name": descriptor.name,
                    "size_bytes": descriptor.size_bytes,
                    "sha256": descriptor.sha256,
                    "asset_id": ordinal,
                    "release_id": release_id,
                    "api_url": f"{api_root}/releases/assets/{ordinal}",
                    "browser_download_url": (
                        f"{web_root}/releases/download/"
                        f"{operations.BASELINE_RELEASE_TAG}/{descriptor.name}"
                    ),
                }
            )
        body = {
            "release_identity": {
                "repository": operations.REPOSITORY,
                "release_id": release_id,
                "tag_name": operations.BASELINE_RELEASE_TAG,
                "api_url": f"{api_root}/releases/{release_id}",
                "html_url": (
                    f"{web_root}/releases/tag/{operations.BASELINE_RELEASE_TAG}"
                ),
                "tag_ref": {
                    "ref": f"refs/tags/{operations.BASELINE_RELEASE_TAG}",
                    "object_type": "commit",
                    "object_sha": operations.BASELINE_V3_ROOT_COMMIT,
                    "object_url": (
                        f"{api_root}/git/commits/"
                        f"{operations.BASELINE_V3_ROOT_COMMIT}"
                    ),
                },
            },
            "asset_ledger": ledger,
            "observed_at": observed_at,
            "api_observations": [
                {
                    "endpoint": endpoint,
                    "canonical_response_sha256": "7" * 64,
                    "canonical_response_size_bytes": 1,
                    "page_ledger_sha256": "8" * 64,
                }
                for endpoint in (
                    f"repos/{operations.REPOSITORY}/releases/tags/"
                    f"{operations.BASELINE_RELEASE_TAG}",
                    f"repos/{operations.REPOSITORY}/git/ref/tags/"
                    f"{operations.BASELINE_RELEASE_TAG}",
                )
            ],
        }
        return {
            **body,
            "authority_sha256": privacy.tagged_canonical_sha256(
                operations.BASELINE_AUTHORITY_DOMAIN,
                body,
            ),
        }

    @staticmethod
    def stage_operations_fixture(
        index_snapshot,
        _offline_result,
        destination,
        _forbidden_roots,
    ):
        staging = completion._create_private_subdirectory(
            destination, completion.LIVE_OPERATIONS_DIRECTORY
        )
        response_snapshots = []
        for ordinal, role in enumerate(operations.RESPONSE_ROLE_ORDER):
            response_raw = completion._compact_canonical_json_bytes(
                {"ordinal": ordinal, "role": role}
            )
            response_path = staging / f"response-{ordinal}.json"
            completion._write_exclusive_private_file(
                response_path,
                response_raw,
                f"fixture operations response {role}",
            )
            response_snapshot, _raw = completion._snapshot_regular_file(
                response_path,
                f"fixture operations response {role}",
                max_bytes=completion.MAX_ARTIFACT_BYTES,
            )
            response_snapshots.append((role, response_snapshot))
        raw = completion._compact_canonical_json_bytes(
            {"kind": "fixture-operations-index"}
        )
        path = staging / "operations-index.json"
        completion._write_exclusive_private_file(path, raw, "fixture operations index")
        index_staging_snapshot, _raw = completion._snapshot_regular_file(
            path,
            "fixture operations index",
            max_bytes=completion.MAX_JSON_BYTES,
        )
        return (
            path,
            {
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            completion.StagedOperationsSnapshots(
                index=index_staging_snapshot,
                responses=tuple(response_snapshots),
            ),
        )

    def live_receipt_for_call(self, kwargs):
        index_raw = kwargs["index_path"].read_bytes()
        asset_ledger = [
            {
                "role": role,
                "name": name,
                "size_bytes": kwargs["publication_assets"][role].stat().st_size,
                "sha256": hashlib.sha256(
                    kwargs["publication_assets"][role].read_bytes()
                ).hexdigest(),
            }
            for role, name in operations.TECHNICAL_RELEASE_ASSETS.items()
        ]
        now = (
            dt.datetime.now(dt.UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        return {
            "schema_version": operations.LIVE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
            "kind": operations.LIVE_VERIFICATION_RECEIPT_KIND,
            "authority": completion.LIVE_AUTHORITY,
            "receipt_replay_authority": False,
            "verified_at": now,
            "candidate_evidence": copy.deepcopy(self.candidate),
            "repository": operations.REPOSITORY,
            "pull_request_number": operations.PULL_REQUEST_NUMBER,
            "operations_index": {
                "size_bytes": len(index_raw),
                "sha256": hashlib.sha256(index_raw).hexdigest(),
            },
            "publication_validation": {
                "strict_four_byte_validator": "same-process-invoked",
                "receipt_sha256": "f" * 64,
                "trusted_policy_sha256": kwargs["publication_policy_sha256"],
                "asset_ledger": asset_ledger,
                "release_identity_anchor": {},
                "bindings": {},
                "execution_claims": [],
                "event_profiler": {},
            },
            "post_bundle_acknowledgment": {
                "checked_lines": list(kwargs["post_bundle_expectation"].checked_lines),
                "o0_canonical_response_sha256": (
                    kwargs["post_bundle_expectation"].o0_canonical_response_sha256
                ),
                "o1_canonical_response_sha256": (
                    kwargs["post_bundle_expectation"].o1_canonical_response_sha256
                ),
                "o1_body_sha256": (kwargs["post_bundle_expectation"].o1_body_sha256),
                "o1_updated_at": (kwargs["post_bundle_expectation"].o1_updated_at),
                "b0_inventory_root": (
                    kwargs["post_bundle_expectation"].b0_inventory_root
                ),
                "b0_reopen_receipt_sha256": (
                    kwargs["post_bundle_expectation"].b0_reopen_receipt_sha256
                ),
                "b0_reopened_at": (kwargs["post_bundle_expectation"].b0_reopened_at),
                "fresh_response_equal": True,
            },
            "baseline_validation": self.baseline_validation_fixture(),
            "queries": [
                {
                    "role": role,
                    "canonical_response_sha256": "a" * 64,
                    "canonical_response_size_bytes": 2,
                    "page_count": 1,
                    "page_ledger_sha256": "b" * 64,
                }
                for role in operations.RESPONSE_ROLE_ORDER
            ],
            "same_process_live_accepted": True,
        }

    def successful_live_verifier(self, **kwargs):
        receipt = self.live_receipt_for_call(kwargs)
        operations._write_private_receipt(
            kwargs["receipt_output"],
            operations._canonical_json_bytes(receipt),
            forbidden_roots=(
                kwargs["index_path"].parent,
                *(path.parent for path in kwargs["publication_assets"].values()),
            ),
        )
        return receipt

    def fake_authenticated_lease(self, index, inputs):
        def snapshot(path, label, maximum):
            return completion._snapshot_regular_file(path, label, max_bytes=maximum)[0]

        source_index = snapshot(
            index, "source fixture index", completion.MAX_INDEX_BYTES
        )
        reopened_index = snapshot(
            inputs["reopened_index"],
            "reopened fixture index",
            completion.MAX_INDEX_BYTES,
        )
        manifest = snapshot(
            completion.DEFAULT_MANIFEST,
            "fixture manifest",
            completion.MAX_MANIFEST_BYTES,
        )
        runtime_receipts = tuple(
            snapshot(path, "fixture runtime receipt", completion.MAX_JSON_BYTES)
            for path in inputs["runtime_receipt_paths"]
        )
        protected_openings = snapshot(
            inputs["protected_openings"],
            "fixture protected openings",
            completion.MAX_JSON_BYTES,
        )
        pre_ack = snapshot(
            inputs["pre_ack_bundle_reopen_receipt"],
            "fixture B0 receipt",
            completion.MAX_JSON_BYTES,
        )
        final = snapshot(
            inputs["final_bundle_reopen_receipt"],
            "fixture B1 receipt",
            completion.MAX_JSON_BYTES,
        )
        snapshots = SimpleNamespace(
            source_bundle=SimpleNamespace(root=source_index.path.parent),
            reopened_bundle=SimpleNamespace(root=reopened_index.path.parent),
            source_index=source_index,
            reopened_index=reopened_index,
            manifest=manifest,
            runtime_receipts=runtime_receipts,
            protected_openings=protected_openings,
            pre_acknowledgment_receipt=pre_ack,
            final_reopen_receipt=final,
        )

        def require_unchanged():
            for item, label, maximum in (
                (source_index, "source fixture index", completion.MAX_INDEX_BYTES),
                (
                    reopened_index,
                    "reopened fixture index",
                    completion.MAX_INDEX_BYTES,
                ),
                (manifest, "fixture manifest", completion.MAX_MANIFEST_BYTES),
                (
                    protected_openings,
                    "fixture protected openings",
                    completion.MAX_JSON_BYTES,
                ),
                (pre_ack, "fixture B0 receipt", completion.MAX_JSON_BYTES),
                (final, "fixture B1 receipt", completion.MAX_JSON_BYTES),
            ):
                completion._require_snapshot_unchanged(item, label, max_bytes=maximum)
            for item in runtime_receipts:
                completion._require_snapshot_unchanged(
                    item,
                    "fixture runtime receipt",
                    max_bytes=completion.MAX_JSON_BYTES,
                )

        chain = self.authority_gate_fixture()
        return SimpleNamespace(
            _snapshots=snapshots,
            _chain=chain,
            expectation=chain["post_bundle_expectation"],
            require_unchanged=require_unchanged,
            _private_writer_roots=lambda: (
                source_index.path.parent,
                reopened_index.path.parent,
            ),
            _baseline_authority_set=lambda _operations: (
                self.baseline_descriptors_fixture()
            ),
        )

    def invoke_live_fixture(self, suffix, verifier, *, manager_exit_error=None):
        offline = self.evaluate_with_patches(self.patched_validators())
        index = self.directory / "index.json"
        inputs = self.live_test_inputs(suffix)
        lease = self.fake_authenticated_lease(index, inputs)
        manager = mock.MagicMock()
        manager.__enter__.return_value = lease
        manager.__exit__.return_value = False
        manager.__exit__.side_effect = manager_exit_error

        def open_live_authority(**kwargs):
            legacy_kwargs = {
                **kwargs,
                "post_bundle_expectation": kwargs["post_bundle_lease"].expectation,
            }
            legacy_kwargs.pop("post_bundle_lease")
            legacy_kwargs.pop("baseline_authority")
            receipt = verifier(**legacy_kwargs)
            authority = SimpleNamespace(
                receipt=receipt,
                require_unchanged=lambda: None,
            )
            live_manager = mock.MagicMock()
            live_manager.__enter__.return_value = authority
            live_manager.__exit__.return_value = False
            return live_manager

        with (
            mock.patch.object(
                completion,
                "open_authenticated_post_bundle_transition",
                return_value=manager,
            ),
            mock.patch.object(
                completion,
                "evaluate_completion",
                return_value=offline,
            ) as evaluate,
            mock.patch.object(
                completion,
                "_prepare_operations_live_input",
                side_effect=self.stage_operations_fixture,
            ) as prepare,
            mock.patch.object(
                privacy,
                "verify_publication_bundle_binding",
                return_value={
                    "technical_input_root": "9" * 64,
                    "public_projection_sha256": "a" * 64,
                    "public_asset_ledger_sha256": "b" * 64,
                    "source_count": 5,
                    "runtime_receipt_count": 5,
                    "first_five_scopes_validated": True,
                },
            ),
            mock.patch.object(
                operations,
                "open_verified_operations_live",
                side_effect=open_live_authority,
            ) as verify,
        ):
            result = completion.verify_completion_live(
                index_path=index,
                manifest_path=completion.DEFAULT_MANIFEST,
                **inputs,
            )
        return result, offline, index, inputs, evaluate, prepare, verify

    def test_all_offline_scopes_are_structural_and_never_complete_the_issue(self):
        result = self.evaluate_with_patches(self.patched_validators())
        self.assertTrue(result["structural_validation_satisfied"])
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertFalse(result["final_acceptance"])
        self.assertFalse(result["receipt_replay_authority"])
        self.assertEqual(result["evaluation_mode"], completion.OFFLINE_EVALUATION_MODE)
        self.assertEqual(
            result["final_acceptance_authority"], completion.OFFLINE_AUTHORITY
        )
        self.assertFalse(result["live_verification"]["invocation_attempted"])
        self.assertTrue(all(scope["satisfied"] for scope in result["scopes"].values()))

        values = self.patched_validators()
        index = self.write_top_index()
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    completion,
                    "_load_trusted_runtime_receipts",
                    return_value=self.patched_runtime_receipts(values),
                )
            )
            for name, value in values.items():
                patched = stack.enter_context(mock.patch.object(completion, name))
                if name == "_validate_macos_scope":
                    patched.side_effect = completion.EvidenceError("missing job")
                else:
                    patched.return_value = value
            failed = completion.evaluate_completion(index)
        self.assertFalse(failed["issue_completion_satisfied"])
        self.assertEqual(
            failed["scopes"]["macos"]["errors"],
            [
                {
                    "code": "invalid-evidence",
                    "phase": "scope-validation",
                    "scope": "macos",
                    "message": "evidence validation failed closed",
                }
            ],
        )

    def test_global_correctness_archives_require_exact_shared_136_topology(self):
        valid = self.evaluate_with_patches(self.patched_validators())
        self.assertTrue(valid["structural_validation_satisfied"])
        self.assertEqual(
            valid["cross_scope_details"]["correctness_archive_topology"],
            {
                "case_count": 34,
                "shared_reference_archive_count": 34,
                "candidate_archive_count": 102,
                "unique_archive_count": 136,
            },
        )

        attacks = (
            "reference-path",
            "reference-sha256",
            "reference-size",
            "reference-payload-identity",
            "candidate-path-overlap",
            "candidate-sha256-overlap",
            "candidate-payload-identity-overlap",
        )
        for attack in attacks:
            with self.subTest(attack=attack):
                values = self.patched_validators()
                cpu = values["_validate_cpu_scope"]
                single = values["_validate_single_gpu_scope"]
                cpu_reference = cpu["_correctness_archive_bindings"]["reference"][0]
                graph_candidate = single["_correctness_archive_bindings_by_mode"][
                    "graph"
                ]["candidate"][0]
                cpu_candidate = cpu["_correctness_archive_bindings"]["candidate"][0]
                if attack == "reference-path":
                    cpu_reference["path"] = "correctness/separate-reference/0.npz"
                elif attack == "reference-sha256":
                    cpu_reference["sha256"] = hashlib.sha256(
                        b"separate-reference"
                    ).hexdigest()
                elif attack == "reference-size":
                    cpu_reference["size_bytes"] += 1
                elif attack == "reference-payload-identity":
                    cpu_reference["payload_identity"] = (9, 9000, 1, 1)
                else:
                    field = {
                        "candidate-path-overlap": "path",
                        "candidate-sha256-overlap": "sha256",
                        "candidate-payload-identity-overlap": "payload_identity",
                    }[attack]
                    graph_candidate[field] = copy.deepcopy(cpu_candidate[field])
                result = self.evaluate_with_patches(values)
                self.assertFalse(result["structural_validation_satisfied"])
                self.assertFalse(result["scopes"]["cpu"]["satisfied"])
                self.assertFalse(result["scopes"]["single_gpu"]["satisfied"])
                self.assertIn(
                    "correctness-archive-topology",
                    [error["phase"] for error in result["cross_scope_errors"]],
                )

    def test_offline_cli_distinguishes_structural_from_final_enforcement(self):
        result = {
            "structural_validation_satisfied": True,
            "final_acceptance": False,
            "issue_completion_satisfied": False,
        }
        base = {
            "command": "evaluate",
            "index": Path("completion-index.json"),
            "manifest": completion.DEFAULT_MANIFEST,
            "runtime_receipts": [Path(f"receipt-{index}.json") for index in range(5)],
            "output": None,
        }
        statuses = []
        for enforce, enforce_structural in ((True, False), (False, True)):
            args = SimpleNamespace(
                **base,
                enforce=enforce,
                enforce_structural=enforce_structural,
            )
            with (
                mock.patch.object(completion, "_arguments", return_value=args),
                mock.patch.object(
                    completion,
                    "evaluate_completion",
                    return_value=result,
                ),
                mock.patch("builtins.print"),
            ):
                statuses.append(completion.main())
        self.assertEqual(statuses, [2, 0])

    def test_operations_scope_requires_an_explicit_offline_authority_marker(self):
        artifact = completion.LoadedArtifact({}, Path("operations.json"), b"{}", {})
        responses = {
            "fixture": completion.LoadedArtifact({}, Path("raw.json"), b"{}", {})
        }
        attacks = (
            {
                "final_acceptance": True,
                "final_acceptance_authority": completion.OFFLINE_AUTHORITY,
            },
            {"final_acceptance": False},
            {
                "final_acceptance": False,
                "final_acceptance_authority": completion.LIVE_AUTHORITY,
            },
        )
        for attack in attacks:
            with (
                self.subTest(attack=attack),
                mock.patch.object(
                    completion,
                    "_load_operations_scope_artifacts",
                    return_value=(artifact, responses),
                ),
                mock.patch.object(
                    operations,
                    "evaluate_operations",
                    return_value=attack,
                ),
                self.assertRaisesRegex(
                    completion.EvidenceError,
                    "non-authoritative structural result",
                ),
            ):
                completion._validate_operations_scope({}, object(), self.candidate)

    def test_nested_operations_capture_is_staged_with_exact_relative_bytes(self):
        manifest_raw = completion.DEFAULT_MANIFEST.read_bytes()
        manifest_descriptor = self.write_bytes(
            "staging/manifest.json",
            manifest_raw,
            media_type=completion.MEDIA_TYPE_JSON,
        )
        response_raw_by_role = {}
        response_descriptors = {}
        for ordinal, role in enumerate(operations.RESPONSE_ROLE_ORDER):
            response_raw = completion._compact_canonical_json_bytes(
                {"fixture": True, "ordinal": ordinal, "role": role}
            )
            response_raw_by_role[role] = response_raw
            response_descriptors[role] = self.write_bytes(
                f"raw/{ordinal}.json",
                response_raw,
                media_type=completion.MEDIA_TYPE_JSON,
            )
        operations_document = {
            "candidate_evidence": self.candidate,
            "responses": {
                role: {
                    "request": {},
                    "artifact": response_descriptors[role],
                }
                for role in operations.RESPONSE_ROLE_ORDER
            },
        }
        operations_raw = completion._canonical_json_bytes(operations_document)
        operations_descriptor = self.write_bytes(
            "nested/source/operations-index.json",
            operations_raw,
            media_type=completion.MEDIA_TYPE_JSON,
        )
        payloads = sorted(
            (
                manifest_descriptor,
                operations_descriptor,
                *response_descriptors.values(),
            ),
            key=lambda item: item["path"],
        )
        top = {
            "schema_version": completion.INDEX_SCHEMA_VERSION,
            "kind": completion.INDEX_KIND,
            "issue": 123,
            "bundle": {},
            "candidate_evidence": self.candidate,
            "manifest": manifest_descriptor,
            "payloads": payloads,
            "artifacts": {"operations": {"index": operations_descriptor}},
        }
        top_path = self.directory / "nested-completion-index.json"
        top_path.write_bytes(completion._canonical_json_bytes(top))
        snapshot, _raw = completion._snapshot_regular_file(
            top_path,
            "completion evidence index",
            max_bytes=completion.MAX_INDEX_BYTES,
        )
        structural_operations = {
            "final_acceptance": False,
            "final_acceptance_authority": completion.OFFLINE_AUTHORITY,
        }
        offline = {
            "evidence_index": {
                "path": top_path.name,
                "size_bytes": snapshot.size_bytes,
                "sha256": snapshot.sha256,
            },
            "manifest": {
                "path": manifest_descriptor["path"],
                "size_bytes": manifest_descriptor["size_bytes"],
                "sha256": manifest_descriptor["sha256"],
            },
            "candidate_evidence": self.candidate,
            "scopes": {"operations": {"details": structural_operations}},
        }
        destination = completion._create_private_live_directory(
            self.external_directory / "nested-staging-output",
            (self.directory.resolve(),),
        )
        with mock.patch.object(
            completion,
            "_validate_operations_scope",
            return_value=structural_operations,
        ):
            staged, descriptor, snapshots = completion._prepare_operations_live_input(
                snapshot,
                offline,
                destination,
                (self.directory.resolve(),),
            )
        self.assertEqual(staged.read_bytes(), operations_raw)
        for role, response_descriptor in response_descriptors.items():
            self.assertEqual(
                (staged.parent / response_descriptor["path"]).read_bytes(),
                response_raw_by_role[role],
            )
        self.assertFalse((staged.parent / "nested").exists())
        self.assertEqual(
            [role for role, _snapshot in snapshots.responses],
            sorted(operations.RESPONSE_ROLE_ORDER),
        )
        self.assertEqual(len(snapshots.responses), 22)
        self.assertEqual(snapshots.index.path, staged)
        self.assertEqual(
            descriptor,
            {
                "size_bytes": len(operations_raw),
                "sha256": hashlib.sha256(operations_raw).hexdigest(),
            },
        )
        self.assertEqual(stat.S_IMODE(staged.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o600)

    def test_live_output_preflight_rejects_both_b1_roots_and_aliases(self):
        source = self.directory / "protected-source-b1"
        reopened = self.directory / "protected-reopened-b1"
        source.mkdir()
        reopened.mkdir()
        roots = (source.resolve(), reopened.resolve())
        candidates = (
            source,
            source / "nested-output",
            reopened,
            reopened / "nested-output",
            source / ".." / source.name / "dot-output",
        )
        for ordinal, candidate in enumerate(candidates):
            with (
                self.subTest(candidate=ordinal),
                self.assertRaises(completion.EvidenceError),
            ):
                completion._create_private_live_directory(candidate, roots)
            if candidate not in roots:
                self.assertFalse(candidate.exists())
        alias = self.directory / "protected-source-alias"
        alias.symlink_to(source, target_is_directory=True)
        with self.assertRaises(completion.EvidenceError):
            completion._create_private_live_directory(
                alias / "symlink-output",
                roots,
            )
        self.assertFalse((source / "symlink-output").exists())

    def test_live_completion_passes_exact_inputs_and_is_the_only_final_authority(self):
        result, offline, index, inputs, evaluate, prepare, verify = (
            self.invoke_live_fixture("success", self.successful_live_verifier)
        )
        self.assertTrue(offline["structural_validation_satisfied"])
        self.assertFalse(offline["issue_completion_satisfied"])
        self.assertTrue(result["structural_validation_satisfied"])
        self.assertTrue(result["issue_completion_satisfied"])
        self.assertTrue(result["final_acceptance"])
        self.assertFalse(result["receipt_replay_authority"])
        self.assertEqual(result["evaluation_mode"], completion.LIVE_EVALUATION_MODE)
        self.assertEqual(
            result["final_acceptance_authority"], completion.LIVE_AUTHORITY
        )
        self.assertTrue(result["live_verification"]["invocation_attempted"])
        self.assertTrue(result["live_verification"]["invocation_succeeded"])
        evaluate.assert_called_once_with(
            index,
            completion.DEFAULT_MANIFEST,
            inputs["runtime_receipt_paths"],
        )
        prepare.assert_called_once()
        call = verify.call_args.kwargs
        self.assertEqual(
            set(call),
            {
                "index_path",
                "manifest",
                "publication_policy",
                "publication_policy_sha256",
                "publication_assets",
                "receipt_output",
                "post_bundle_lease",
                "baseline_authority",
            },
        )
        output = inputs["output_directory"].resolve()
        self.assertEqual(
            call["index_path"],
            output / completion.LIVE_OPERATIONS_DIRECTORY / "operations-index.json",
        )
        self.assertEqual(call["manifest"], completion.DEFAULT_MANIFEST.resolve())
        self.assertEqual(
            call["publication_policy"], inputs["publication_policy"].resolve()
        )
        self.assertEqual(
            call["publication_policy_sha256"],
            inputs["publication_policy_sha256"],
        )
        self.assertEqual(
            call["publication_assets"],
            {
                role: path.resolve()
                for role, path in inputs["publication_assets"].items()
            },
        )
        self.assertEqual(call["receipt_output"], output / completion.LIVE_RECEIPT_NAME)
        self.assertEqual(
            call["post_bundle_lease"].expectation,
            self.authority_gate_fixture()["post_bundle_expectation"],
        )
        self.assertEqual(call["baseline_authority"], "live-release")
        self.assertTrue(result["candidate_bundle_binding"]["satisfied"])
        self.assertTrue(result["post_bundle"]["satisfied"])
        self.assertTrue(result["baseline_authority"]["satisfied"])
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        result_path = output / completion.LIVE_RESULT_NAME
        self.assertEqual(stat.S_IMODE(result_path.stat().st_mode), 0o600)
        self.assertEqual(
            result_path.read_bytes(), completion._canonical_json_bytes(result)
        )

    def test_late_b1_mutation_cannot_publish_authoritative_v3_result(self):
        real_writer = privacy.write_private_authority_file
        source_index = self.directory / "index.json"

        def mutate_at_result_commit(*args, **kwargs):
            barrier = kwargs.get("before_commit")
            if barrier is not None:

                def mutation_barrier():
                    source_index.write_bytes(source_index.read_bytes() + b"x")
                    barrier()

                kwargs["before_commit"] = mutation_barrier
            return real_writer(*args, **kwargs)

        with (
            mock.patch.object(
                privacy,
                "write_private_authority_file",
                side_effect=mutate_at_result_commit,
            ),
            self.assertRaisesRegex(
                completion.EvidenceError,
                "completion live result could not be emitted",
            ),
        ):
            self.invoke_live_fixture(
                "late-b1-mutation",
                self.successful_live_verifier,
            )
        destination = self.external_directory / "late-b1-mutation-output"
        self.assertFalse((destination / completion.LIVE_RESULT_NAME).exists())
        self.assertEqual(
            list(destination.glob(f".{completion.LIVE_RESULT_NAME}.tmp-*")),
            [],
        )

    def test_post_link_outer_lease_failure_reports_committed_authority(self):
        with self.assertRaises(completion.CommittedAuthorityError) as caught:
            self.invoke_live_fixture(
                "post-link-lease-close",
                self.successful_live_verifier,
                manager_exit_error=completion.EvidenceError(
                    "synthetic-retained-close-canary"
                ),
            )
        self.assertTrue(caught.exception.committed)
        self.assertEqual(
            str(caught.exception),
            "completion authority was committed but custody cleanup failed",
        )
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        result_path = (
            self.external_directory
            / "post-link-lease-close-output"
            / completion.LIVE_RESULT_NAME
        )
        result = json.loads(result_path.read_bytes())
        self.assertTrue(result["final_acceptance"])
        self.assertTrue(result["issue_completion_satisfied"])

    def test_reopen_binding_and_baseline_gates_block_before_live_authority(self):
        for gate in ("reopen-chain", "private-a-public-b", "baseline-descriptors"):
            suffix = f"authority-gate-{gate}"
            offline = self.evaluate_with_patches(self.patched_validators())
            index = self.directory / "index.json"
            inputs = self.live_test_inputs(suffix)
            chain = mock.Mock(return_value=self.authority_gate_fixture())
            binding = mock.Mock(
                return_value={
                    "technical_input_root": "9" * 64,
                    "public_projection_sha256": "a" * 64,
                    "public_asset_ledger_sha256": "b" * 64,
                    "source_count": 5,
                    "runtime_receipt_count": 5,
                    "first_five_scopes_validated": True,
                }
            )
            baseline = mock.Mock(return_value=self.baseline_descriptors_fixture())
            if gate == "reopen-chain":
                chain.side_effect = completion.EvidenceError(
                    "protected opening or reopen receipt differs"
                )
            elif gate == "private-a-public-b":
                binding.side_effect = privacy.PrivacyError(
                    "published public bytes differ from final B1 projection"
                )
            else:
                baseline.side_effect = completion.EvidenceError(
                    "final B1 baseline descriptor differs"
                )
            with (
                self.subTest(gate=gate),
                mock.patch.object(
                    completion, "evaluate_completion", return_value=offline
                ),
                mock.patch.object(
                    completion,
                    "_validate_final_bundle_reopen_chain",
                    side_effect=chain,
                ),
                mock.patch.object(
                    privacy,
                    "verify_publication_bundle_binding",
                    side_effect=binding,
                ),
                mock.patch.object(
                    completion,
                    "_validate_final_b1_baseline_descriptors",
                    side_effect=baseline,
                ),
                mock.patch.object(
                    completion, "_prepare_operations_live_input"
                ) as prepare,
                mock.patch.object(
                    operations, "open_verified_operations_live"
                ) as verify,
            ):
                result = completion.verify_completion_live(
                    index_path=index,
                    manifest_path=completion.DEFAULT_MANIFEST,
                    **inputs,
                )
            self.assertFalse(result["final_acceptance"])
            self.assertFalse(result["issue_completion_satisfied"])
            self.assertFalse(result["candidate_bundle_binding"]["satisfied"])
            self.assertFalse(result["post_bundle"]["satisfied"])
            self.assertFalse(result["baseline_authority"]["satisfied"])
            prepare.assert_not_called()
            verify.assert_not_called()

    def test_legacy_missing_or_mutated_live_authority_receipt_is_rejected(self):
        for attack in ("legacy-schema", "missing-baseline", "mutated-baseline"):

            def emit_attacked_receipt(*, _attack=attack, **kwargs):
                receipt = self.live_receipt_for_call(kwargs)
                if _attack == "legacy-schema":
                    receipt["schema_version"] = 2
                elif _attack == "missing-baseline":
                    del receipt["baseline_validation"]
                else:
                    receipt["baseline_validation"]["asset_ledger"][0]["sha256"] = (
                        "0" * 64
                    )
                operations._write_private_receipt(
                    kwargs["receipt_output"],
                    operations._canonical_json_bytes(receipt),
                )
                return receipt

            with self.subTest(attack=attack):
                result, _offline, _index, _inputs, _evaluate, _prepare, verify = (
                    self.invoke_live_fixture(attack, emit_attacked_receipt)
                )
                verify.assert_called_once()
                self.assertFalse(result["final_acceptance"])
                self.assertFalse(result["issue_completion_satisfied"])
                self.assertFalse(result["live_verification"]["invocation_succeeded"])
                self.assertFalse(result["baseline_authority"]["satisfied"])

    def test_completion_cli_requires_and_forwards_all_private_authority_inputs(self):
        runtime = [
            str(self.directory / f"runtime-{ordinal}.json") for ordinal in range(5)
        ]
        arguments = [
            "verify-live",
            "--index",
            str(self.directory / "b1-source" / "completion-index.json"),
            "--reopened-index",
            str(self.directory / "b1-reopened" / "completion-index.json"),
            "--private-openings",
            str(self.directory / "private" / "openings.json"),
            "--pre-ack-bundle-reopen-receipt",
            str(self.directory / "private" / "b0-reopen.json"),
            "--final-bundle-reopen-receipt",
            str(self.directory / "private" / "b1-reopen.json"),
            "--manifest",
            str(self.directory / "manifest.json"),
            "--runtime-receipts",
            *runtime,
            "--publication-policy",
            str(self.directory / "policy.json"),
            "--publication-policy-sha256",
            "e" * 64,
            "--technical-evidence-asset",
            str(self.directory / "technical.zip"),
            "--technical-summary-asset",
            str(self.directory / "summary.json"),
            "--raw-timing-asset",
            str(self.directory / "timing.json"),
            "--event-profiler-asset",
            str(self.directory / "profiler.json"),
            "--output-directory",
            str(self.directory / "output"),
            "--enforce",
        ]
        for required in (
            "--reopened-index",
            "--private-openings",
            "--pre-ack-bundle-reopen-receipt",
            "--final-bundle-reopen-receipt",
        ):
            changed = list(arguments)
            location = changed.index(required)
            del changed[location : location + 2]
            with (
                self.subTest(missing=required),
                mock.patch("sys.stderr", new=io.StringIO()),
                self.assertRaises(completion._CliUsageError),
            ):
                completion._arguments(changed)

        output = io.StringIO()
        with (
            mock.patch.object(
                completion,
                "verify_completion_live",
                return_value={"final_acceptance": True},
            ) as verify,
            mock.patch("sys.stdout", new=output),
        ):
            status = completion.main(arguments)
        self.assertEqual(status, 0)
        call = verify.call_args.kwargs
        self.assertEqual(
            call["reopened_index"],
            Path(arguments[arguments.index("--reopened-index") + 1]),
        )
        self.assertEqual(
            call["protected_openings"],
            Path(arguments[arguments.index("--private-openings") + 1]),
        )
        self.assertEqual(
            call["pre_ack_bundle_reopen_receipt"],
            Path(arguments[arguments.index("--pre-ack-bundle-reopen-receipt") + 1]),
        )
        self.assertEqual(
            call["final_bundle_reopen_receipt"],
            Path(arguments[arguments.index("--final-bundle-reopen-receipt") + 1]),
        )

    def test_structured_errors_never_serialize_exception_text(self):
        hostile = (
            "/Users/fixture-person-invalid/private "
            + "github_pat_"
            + "syntheticinvalid" * 2
        )
        for error in (
            completion.EvidenceError(hostile),
            OSError(hostile),
            RuntimeError(hostile),
        ):
            rendered = json.dumps(
                completion._structured_evidence_error(
                    error,
                    phase="synthetic-phase",
                    scope="synthetic-scope",
                )
            )
            self.assertNotIn("fixture-person-invalid", rendered)
            self.assertNotIn("github_pat_", rendered)
            self.assertIn("evidence", rendered)

    def test_completion_cli_failure_tokens_never_render_private_text(self):
        marker = (
            "/tmp/synthetic-private.invalid/identity "
            + "salt="
            + "ab" * 32
            + " hmac="
            + "cd" * 32
            + " raw-body=fixture-private-value"
        )
        for command in ("record-reopen", "verify-live"):
            for boundary in (completion.main, completion._cli):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    self.subTest(command=command, boundary=boundary.__name__),
                    mock.patch.object(
                        completion,
                        "_main",
                        side_effect=completion.EvidenceError(marker),
                    ),
                    mock.patch("sys.stdout", new=stdout),
                    mock.patch("sys.stderr", new=stderr),
                ):
                    status = boundary([command])
                self.assertEqual(status, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(
                    stderr.getvalue(),
                    f"issue123-completion-{command}-failed\n",
                )
                rendered = stdout.getvalue() + stderr.getvalue()
                self.assertNotIn("Traceback", rendered)
                self.assertNotIn(marker, rendered)

    def test_live_verifier_failure_stays_structural_only(self):
        def fail(**_kwargs):
            raise operations.EvidenceError("fresh GitHub response is stale")

        result, _offline, _index, inputs, _evaluate, _prepare, verify = (
            self.invoke_live_fixture("failure", fail)
        )
        verify.assert_called_once()
        self.assertTrue(result["structural_validation_satisfied"])
        self.assertFalse(result["final_acceptance"])
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertTrue(result["live_verification"]["invocation_attempted"])
        self.assertFalse(result["live_verification"]["invocation_succeeded"])
        self.assertEqual(
            result["live_verification"]["errors"][0]["message"],
            "evidence validation failed closed",
        )
        result_path = inputs["output_directory"] / completion.LIVE_RESULT_NAME
        self.assertEqual(
            result_path.read_bytes(), completion._canonical_json_bytes(result)
        )

    def test_returned_or_replayed_receipt_cannot_authorize_completion(self):
        def return_without_emitting(**kwargs):
            return self.live_receipt_for_call(kwargs)

        result, offline, index, inputs, _evaluate, _prepare, verify = (
            self.invoke_live_fixture("offline-receipt", return_without_emitting)
        )
        verify.assert_called_once()
        self.assertFalse(result["final_acceptance"])
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertFalse(result["live_verification"]["invocation_succeeded"])
        self.assertEqual(
            result["live_verification"]["errors"][0]["message"],
            "evidence validation failed closed",
        )

        previous = inputs["output_directory"]
        with (
            mock.patch.object(
                completion,
                "evaluate_completion",
                return_value=offline,
            ),
            mock.patch.object(completion, "_prepare_operations_live_input") as prepare,
            mock.patch.object(operations, "open_verified_operations_live") as replay,
        ):
            replayed = completion.verify_completion_live(
                index_path=index,
                manifest_path=completion.DEFAULT_MANIFEST,
                **inputs,
            )
        self.assertFalse(replayed["final_acceptance"])
        self.assertFalse(replayed["issue_completion_satisfied"])
        self.assertEqual(
            replayed["live_verification"]["errors"][0]["message"],
            "evidence validation failed closed",
        )
        prepare.assert_not_called()
        replay.assert_not_called()
        self.assertEqual(previous, inputs["output_directory"])

    def test_mocked_offline_receipt_cannot_grant_live_authority(self):
        def emit_offline_receipt(**kwargs):
            receipt = {
                "schema_version": 2,
                "kind": completion.OUTPUT_KIND,
                "structural_validation_satisfied": True,
                "final_acceptance": False,
                "issue_completion_satisfied": False,
                "receipt_replay_authority": False,
            }
            kwargs["receipt_output"].write_bytes(
                completion._compact_canonical_json_bytes(receipt)
            )
            return receipt

        result, _offline, _index, _inputs, _evaluate, _prepare, verify = (
            self.invoke_live_fixture("mocked-offline", emit_offline_receipt)
        )
        verify.assert_called_once()
        self.assertTrue(result["structural_validation_satisfied"])
        self.assertFalse(result["final_acceptance"])
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertFalse(result["live_verification"]["invocation_succeeded"])
        self.assertEqual(
            result["live_verification"]["errors"][0]["message"],
            "evidence validation failed closed",
        )

    def test_index_and_runtime_receipt_substitution_during_live_check_fail_closed(self):
        for target in ("index", "runtime"):
            suffix = f"substitute-{target}"

            def substitute_then_succeed(**kwargs):
                if target == "index":
                    path = self.directory / "index.json"
                else:
                    path = self.external_directory / suffix / "runtime-0-cpu.json"
                path.write_bytes(path.read_bytes() + b" ")
                return self.successful_live_verifier(**kwargs)

            with self.subTest(target=target):
                result, _offline, _index, _inputs, _evaluate, _prepare, verify = (
                    self.invoke_live_fixture(suffix, substitute_then_succeed)
                )
                verify.assert_called_once()
                self.assertTrue(result["structural_validation_satisfied"])
                self.assertFalse(result["final_acceptance"])
                self.assertFalse(result["issue_completion_satisfied"])
                self.assertEqual(
                    result["live_verification"]["errors"][0]["message"],
                    "evidence validation failed closed",
                )

    def test_policy_and_asset_substitution_during_live_check_fail_closed(self):
        for target in ("policy", "asset"):
            suffix = f"substitute-{target}"

            def substitute_then_succeed(**kwargs):
                if target == "policy":
                    path = kwargs["publication_policy"]
                else:
                    path = kwargs["publication_assets"]["technical_summary"]
                path.write_bytes(path.read_bytes() + b"substituted")
                return self.successful_live_verifier(**kwargs)

            with self.subTest(target=target):
                result, _offline, _index, _inputs, _evaluate, _prepare, verify = (
                    self.invoke_live_fixture(suffix, substitute_then_succeed)
                )
                verify.assert_called_once()
                self.assertTrue(result["structural_validation_satisfied"])
                self.assertFalse(result["final_acceptance"])
                self.assertFalse(result["issue_completion_satisfied"])
                self.assertEqual(
                    result["live_verification"]["errors"][0]["message"],
                    "evidence validation failed closed",
                )

    def test_derived_operations_index_and_responses_are_immutable_through_live_call(
        self,
    ):
        attacks = (
            (target, mutation)
            for target in ("index", "response")
            for mutation in ("append", "same-size", "replace")
        )
        for target, mutation in attacks:
            suffix = f"substitute-staged-{target}-{mutation}"

            def mutate_staged_then_succeed(**kwargs):
                receipt = self.live_receipt_for_call(kwargs)
                if target == "index":
                    path = kwargs["index_path"]
                else:
                    path = next(
                        candidate
                        for candidate in kwargs["index_path"].parent.iterdir()
                        if candidate.name.startswith("response-")
                    )
                raw = path.read_bytes()
                if mutation == "append":
                    path.write_bytes(raw + b"post-read-substitution")
                elif mutation == "same-size":
                    changed = bytearray(raw)
                    changed[0] ^= 1
                    path.write_bytes(changed)
                else:
                    replacement = path.with_name(f"{path.name}.replacement")
                    replacement.write_bytes(raw)
                    replacement.replace(path)
                kwargs["receipt_output"].write_bytes(
                    operations._canonical_json_bytes(receipt)
                )
                return receipt

            with self.subTest(target=target, mutation=mutation):
                result, _offline, _index, _inputs, _evaluate, _prepare, verify = (
                    self.invoke_live_fixture(suffix, mutate_staged_then_succeed)
                )
                verify.assert_called_once()
                self.assertTrue(result["structural_validation_satisfied"])
                self.assertFalse(result["final_acceptance"])
                self.assertFalse(result["issue_completion_satisfied"])
                self.assertFalse(result["live_verification"]["invocation_succeeded"])
                self.assertEqual(
                    result["live_verification"]["errors"][0]["message"],
                    "evidence validation failed closed",
                )

    def test_stale_policy_and_missing_or_swapped_assets_never_reach_live_verifier(self):
        for attack in ("stale-policy", "noncanonical-policy", "missing", "swapped"):
            suffix = f"preflight-{attack}"
            offline = self.evaluate_with_patches(self.patched_validators())
            index = self.directory / "index.json"
            inputs = self.live_test_inputs(suffix)
            if attack == "stale-policy":
                inputs["publication_policy_sha256"] = "0" * 64
            elif attack == "noncanonical-policy":
                document = json.loads(inputs["publication_policy"].read_bytes())
                inputs["publication_policy"].write_text(
                    json.dumps(document, indent=2) + "\n",
                    encoding="utf-8",
                )
                inputs["publication_policy_sha256"] = hashlib.sha256(
                    inputs["publication_policy"].read_bytes()
                ).hexdigest()
            elif attack == "missing":
                del inputs["publication_assets"]["event_profiler"]
            else:
                assets = inputs["publication_assets"]
                assets["technical_evidence"], assets["technical_summary"] = (
                    assets["technical_summary"],
                    assets["technical_evidence"],
                )
            with (
                self.subTest(attack=attack),
                mock.patch.object(
                    completion,
                    "evaluate_completion",
                    return_value=offline,
                ),
                mock.patch.object(
                    completion,
                    "_prepare_operations_live_input",
                ) as prepare,
                mock.patch.object(
                    operations, "open_verified_operations_live"
                ) as verify,
            ):
                result = completion.verify_completion_live(
                    index_path=index,
                    manifest_path=completion.DEFAULT_MANIFEST,
                    **inputs,
                )
            self.assertFalse(result["final_acceptance"])
            self.assertFalse(result["issue_completion_satisfied"])
            prepare.assert_not_called()
            verify.assert_not_called()

    def test_cross_gpu_environment_mismatch_fails_closed(self):
        result = self.evaluate_with_patches(
            self.patched_validators(different_gpu_environment=True)
        )
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertFalse(result["scopes"]["single_gpu"]["satisfied"])
        self.assertFalse(result["scopes"]["two_gpu"]["satisfied"])
        self.assertTrue(result["cross_scope_errors"])

    def test_all_differential_sources_bind_external_runtime_receipts(self):
        values = self.patched_validators()
        trusted_single_receipts = self.patched_runtime_receipts(values)[3:]
        valid = completion._validate_differential_correctness_source_bindings(
            values["_validate_cpu_scope"],
            values["_validate_policy_scope"],
            values["_validate_single_gpu_scope"],
            self.manifest,
            trusted_single_receipts,
        )
        self.assertEqual(valid["bound_source_count"], 18)

        refreshed = copy.deepcopy(values)
        source = refreshed["_validate_policy_scope"]["differential_source_bindings"][0][
            "candidate_source"
        ]
        source["sha256"] = hashlib.sha256(b"coherently-refreshed").hexdigest()
        source["size_bytes"] += 1
        with self.assertRaisesRegex(
            completion.EvidenceError, "externally attested runtime archive"
        ):
            completion._validate_differential_correctness_source_bindings(
                refreshed["_validate_cpu_scope"],
                refreshed["_validate_policy_scope"],
                refreshed["_validate_single_gpu_scope"],
                self.manifest,
                trusted_single_receipts,
            )

        refreshed_single = copy.deepcopy(values)
        single_source = refreshed_single["_validate_single_gpu_scope"][
            "differential_source_bindings"
        ][0]["candidate_source"]
        single_source["sha256"] = hashlib.sha256(
            b"coherently-refreshed-single"
        ).hexdigest()
        single_source["size_bytes"] += 1
        with self.assertRaisesRegex(
            completion.EvidenceError, "externally attested runtime archive"
        ):
            completion._validate_differential_correctness_source_bindings(
                refreshed_single["_validate_cpu_scope"],
                refreshed_single["_validate_policy_scope"],
                refreshed_single["_validate_single_gpu_scope"],
                self.manifest,
                trusted_single_receipts,
            )

    def test_bundle_receipts_cannot_be_supplied_as_external_trust_roots(self):
        required_cases = [
            case["name"]
            for group in ("correctness", "physical_checks")
            for case in self.manifest[group]
        ]
        artifacts = [
            {
                "case": case,
                "candidate": {
                    "sha256": hashlib.sha256(f"{case}:{index}".encode()).hexdigest(),
                    "size_bytes": index + 1,
                },
            }
            for index, case in enumerate(required_cases)
        ]
        bundled_paths = []
        external_paths = []
        modes = (
            completion.CPU_CORRECTNESS_RUNTIME_MODE,
            *completion.CUDA_CORRECTNESS_RUNTIME_MODES,
            *completion.SINGLE_GPU_DIFFERENTIAL_RUNTIME_MODES,
        )
        archive_closures = (
            artifacts,
            artifacts,
            artifacts,
            [
                {
                    "case": "single-gpu-2d",
                    "candidate": {
                        "sha256": "d" * 64,
                        "size_bytes": 1,
                    },
                }
            ],
            [
                {
                    "case": "single-gpu-3d",
                    "candidate": {
                        "sha256": "e" * 64,
                        "size_bytes": 1,
                    },
                }
            ],
        )
        for name, mode, archive_closure in zip(
            completion.RUNTIME_RECEIPT_ROLES,
            modes,
            archive_closures,
            strict=True,
        ):
            descriptor, receipt = self.runtime_receipt(name, mode, archive_closure)
            bundled_paths.append(self.directory / descriptor["path"])
            external_paths.append(receipt.path)
        loaded = completion._load_trusted_runtime_receipts(
            external_paths,
            self.manifest,
            self.candidate,
            self.directory.resolve(),
        )
        self.assertEqual(len(loaded), 5)
        swapped = list(external_paths)
        swapped[3], swapped[4] = swapped[4], swapped[3]
        with self.assertRaisesRegex(
            completion.EvidenceError, "identity or runtime binding differs"
        ):
            completion._load_trusted_runtime_receipts(
                swapped,
                self.manifest,
                self.candidate,
                self.directory.resolve(),
            )
        with self.assertRaisesRegex(completion.EvidenceError, "outside the evidence"):
            completion._load_trusted_runtime_receipts(
                bundled_paths,
                self.manifest,
                self.candidate,
                self.directory.resolve(),
            )

    def test_missing_scope_fields_return_false_instead_of_raising(self):
        with mock.patch.object(
            completion,
            "_load_trusted_runtime_receipts",
            return_value=(object(), object(), object(), object(), object()),
        ):
            first = completion.evaluate_completion(self.write_top_index())
            second = completion.evaluate_completion(self.write_top_index())
        self.assertFalse(first["issue_completion_satisfied"])
        self.assertEqual(first, second)
        self.assertTrue(all(scope["errors"] for scope in first["scopes"].values()))

    def test_strict_json_rejects_duplicate_keys_and_nan(self):
        with self.assertRaises(completion.EvidenceError):
            completion._strict_json_bytes(b'{"x": 1, "x": 2}', "duplicate")
        with self.assertRaises(completion.EvidenceError):
            completion._strict_json_bytes(b'{"x": NaN}', "nan")
        with self.assertRaises(completion.EvidenceError):
            completion._strict_json_bytes(b'{"x": 1e999}', "overflow")

    def test_exact_integer_contracts_reject_json_booleans(self):
        self.assertFalse(completion._is_exact_int(True, 1))
        host = copy.deepcopy(self.host_contract)
        host["schema_version"] = True
        with self.assertRaises(completion.EvidenceError):
            completion._validate_host_contract(host, "host")

    def test_frozen_baseline_host_is_recomputed_from_raw_environments(self):
        from benchmarks.torch_cpu_baseline import (
            _host_identity_token_from_raw_environment,
        )

        environment, thread = _synthetic_baseline_identity_fixture()
        salt = "0" * 64
        baseline = {
            "hostname": "redacted",
            "host_identity": _host_identity_token_from_raw_environment(
                environment, salt
            ),
            "timing_runtime_identity": {
                "schema_version": 1,
                "torch": "2.13.0+cpu",
                "cuda_runtime": None,
            },
        }
        completion._require_frozen_baseline_host(
            environment, baseline, thread, "CPU one"
        )
        self.assertEqual(environment["hostname"], "fixture-candidate-host.invalid")
        environment["torch"] = "2.13.0+cu130"
        environment["cuda_runtime"] = "13.0"
        environment["devices"] = [
            {"index": 0, "name": "fixture-device-zero.invalid"},
            {"index": 1, "name": "fixture-device-one.invalid"},
        ]
        environment["gpu_topology"] = "fixture-device-topology.invalid"
        with self.assertRaises(completion.EvidenceError):
            completion._require_frozen_baseline_host(
                environment, baseline, thread, "CPU one"
            )
        environment["torch"] = "2.13.0+cpu"
        environment["cuda_runtime"] = None
        completion._require_frozen_baseline_host(
            environment, baseline, thread, "CPU one"
        )
        environment["torch"] = "2.13.1+cu130"
        with self.assertRaises(completion.EvidenceError):
            completion._require_frozen_baseline_host(
                environment, baseline, thread, "CPU one"
            )
        environment["torch"] = "2.13.0+cpu"
        environment["cpu_model"] = "different"
        with self.assertRaises(completion.EvidenceError):
            completion._require_frozen_baseline_host(
                environment, baseline, thread, "CPU one"
            )
        environment["cpu_model"] = "fixture-cpu-model.invalid"
        environment["thread_environment"] = {"OMP_NUM_THREADS": True}
        with self.assertRaises(completion.EvidenceError):
            completion._require_frozen_baseline_host(
                environment, baseline, thread, "CPU one"
            )
        environment["thread_environment"] = thread
        malformed = copy.deepcopy(baseline)
        malformed["timing_runtime_identity"]["schema_version"] = True
        with self.assertRaises(completion.EvidenceError):
            completion._require_frozen_baseline_host(
                environment, malformed, thread, "CPU one"
            )

    def test_native_summary_torch_comparison_uses_strict_public_version(self):
        self.assertTrue(
            completion._public_torch_versions_match("2.13.0+cu130", "2.13.0+cpu")
        )
        self.assertFalse(
            completion._public_torch_versions_match("2.13.1+cu130", "2.13.0+cpu")
        )
        for value in (None, "", "2.13.0+", "2.13.0+cu130+other"):
            with self.subTest(value=value):
                self.assertFalse(
                    completion._public_torch_versions_match(value, "2.13.0+cpu")
                )

    def test_pinned_release_baseline_artifacts_are_bundle_bound(self):
        manifest = copy.deepcopy(self.manifest)
        pins = manifest["performance_gates"]["cpu_acceptance"]["timing_reference"][
            "slice_artifacts"
        ]
        descriptors = [
            self.write_bytes(
                "torch-cpu-baseline-one.json",
                b'{"slice":"one"}',
                media_type=completion.MEDIA_TYPE_JSON,
            ),
            self.write_bytes(
                "torch-cpu-baseline-physical.json",
                b'{"slice":"physical"}',
                media_type=completion.MEDIA_TYPE_JSON,
            ),
        ]
        for pin, descriptor in zip(pins, descriptors, strict=True):
            pin["size_bytes"] = descriptor["size_bytes"]
            pin["sha256"] = descriptor["sha256"]

        def source(mode, threads, descriptor, pin):
            return {
                "thread_mode": mode,
                "threads": threads,
                "sha256": descriptor["sha256"],
                "size_bytes": descriptor["size_bytes"],
                "publication_url": pin["publication_url"],
                "thread_environment": {"OMP_NUM_THREADS": str(threads)},
                "cases": [
                    {
                        "name": case,
                        "measurements": {"advance": {"raw_seconds_per_step": [1.0]}},
                    }
                    for case in completion.CPU_CASES
                ],
            }

        baseline = {
            "timing_reference": {
                "root_commit": manifest["performance_gates"]["cpu_acceptance"][
                    "timing_reference"
                ]["root_commit"]
            },
            "environment": {"hostname": "redacted"},
            "source_artifacts": [
                source("one", 1, descriptors[0], pins[0]),
                source("physical", 4, descriptors[1], pins[1]),
            ],
        }
        reader = completion.ArtifactReader(self.directory, self.candidate)
        with mock.patch(
            "benchmarks.torch_cpu_baseline.load_torch_cpu_baseline",
            return_value=baseline,
        ) as load:
            loaded = completion._load_pinned_torch_baseline(
                manifest,
                reader,
                descriptors,
            )
        self.assertEqual(set(loaded[0]), {"one", "physical"})
        self.assertEqual(
            loaded[-1]["one"]["publication_url"], pins[0]["publication_url"]
        )
        load.assert_called_once()

        pins[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(completion.EvidenceError, "artifact 0 differs"):
            completion._load_pinned_torch_baseline(manifest, reader, descriptors)

    def test_cpu_scope_enforce_accepts_bound_correctness_without_global_completion(
        self,
    ):
        from benchmarks import torch_tuning

        args = SimpleNamespace(
            case="cpu-gates",
            cpu_slice_artifacts=(Path("one.json"), Path("physical.json")),
            torch_baseline_slice_artifacts=(
                Path("baseline-one.json"),
                Path("baseline-physical.json"),
            ),
            allocation_provenance=None,
            native_summary=Path("native.json"),
            correctness_evidence_index=Path("correctness.json"),
            output=None,
            enforce=True,
        )
        aggregate = {
            "suite_acceptance": {"passed": True},
            "acceptance_scope": "cpu-performance-and-correctness",
            "cpu_correctness_satisfied": True,
            "issue_completion_satisfied": False,
        }
        with (
            mock.patch.object(
                torch_tuning, "_arguments", return_value=(args, self.manifest)
            ),
            mock.patch.object(
                torch_tuning, "_aggregate_cpu_slice_files", return_value=aggregate
            ),
            mock.patch("builtins.print"),
        ):
            status = torch_tuning.main()
        self.assertEqual(status, 0)

    def test_region_launches_are_recomputed_from_the_raw_effective_plan(self):
        plan = [
            {
                "component": component,
                "shape": [2, 1, 1],
                "dense_inverse": [[[1.0]], [[0.0]]],
                "constant_targets": [],
                "constant_values": [],
                "buckets": [
                    {
                        "signature": {
                            "component": component,
                            "model": "drude",
                            "precision": "float32",
                            "state_shape": [1],
                        },
                        "coefficient_names": ["inv_eps"],
                        "targets": [1],
                        "target_coefficients": [[1.0]],
                        "cell_coefficient_names": [],
                        "cell_coefficients": [],
                    }
                ],
            }
            for component in completion.FIELD_ARRAYS
        ]
        self.assertEqual(
            completion._validate_effective_material_plan(plan, "plan"), (6, 12)
        )
        plan[0]["buckets"][0]["targets"] = [2]
        with self.assertRaises(completion.EvidenceError):
            completion._validate_effective_material_plan(plan, "plan")

    def test_descriptor_binds_exact_bytes_and_candidate(self):
        descriptor = self.write_json(
            "payload.json",
            {"candidate_evidence": self.candidate, "value": 1},
        )
        reader = completion.ArtifactReader(self.directory, self.candidate)
        loaded = reader.load(descriptor, "payload")
        self.assertEqual(loaded.descriptor["sha256"], descriptor["sha256"])

        for key in (
            "path",
            "sha256",
            "size_bytes",
            "media_type",
            "candidate_evidence",
        ):
            with self.subTest(missing=key):
                mutated = dict(descriptor)
                del mutated[key]
                with self.assertRaises(completion.EvidenceError):
                    reader.load(mutated, "payload")
        mutated = dict(descriptor)
        mutated["extra"] = True
        with self.assertRaises(completion.EvidenceError):
            reader.load(mutated, "payload")
        mutated = copy.deepcopy(descriptor)
        mutated["candidate_evidence"]["candidate_git_commit"] = "b" * 40
        with self.assertRaises(completion.EvidenceError):
            reader.load(mutated, "payload")

        (self.directory / "payload.json").write_bytes(loaded.raw + b" ")
        with self.assertRaises(completion.EvidenceError):
            reader.load(descriptor, "payload")

    def test_macos_sdist_uses_one_retained_raw_first_snapshot(self):
        descriptor = self.write_private_sdist()
        path = self.directory / descriptor["path"]
        raw = path.read_bytes()
        member = privacy._PrivateSdistMember(
            "gmes-0.10.0/PKG-INFO",
            "file",
            len(b"Metadata-Version: 2.4\nName: gmes\n"),
            0,
            hashlib.sha256(b"Metadata-Version: 2.4\nName: gmes\n").hexdigest(),
        )
        observed = {}
        original_retain = privacy._retain_private_sdist_fd

        @contextmanager
        def retain(fd):
            observed["completion_identity"] = os.fstat(fd)
            with original_retain(fd) as source:
                observed["privacy_identity"] = os.fstat(source.fd)
                observed["completion_fd"] = fd
                observed["privacy_fd"] = source.fd
                yield source

        reader = completion.ArtifactReader(self.directory, self.candidate)
        result = self.private_sdist_result(raw, (member,))
        with (
            mock.patch.object(privacy, "_retain_private_sdist_fd", side_effect=retain) as retain_mock,
            mock.patch.object(
                privacy,
                "_validate_private_sdist_raw_first",
                return_value=result,
            ) as validate_mock,
        ):
            artifact, inventory = reader.load_private_sdist(descriptor, "sdist")
        retain_mock.assert_called_once()
        validate_mock.assert_called_once()
        self.assertEqual(validate_mock.call_args.args[1], ())
        self.assertEqual(
            validate_mock.call_args.kwargs["limits"],
            privacy._default_private_sdist_validation_limits(),
        )
        self.assertEqual(artifact.raw, raw)
        self.assertEqual(artifact.descriptor, descriptor)
        self.assertEqual(
            inventory,
            completion._ValidatedSdistInventory(
                hashlib.sha256(raw).hexdigest(),
                len(raw),
                1,
                member.size,
                (member.name,),
            ),
        )
        self.assertEqual(
            (
                observed["completion_identity"].st_dev,
                observed["completion_identity"].st_ino,
            ),
            (observed["privacy_identity"].st_dev, observed["privacy_identity"].st_ino),
        )
        with self.assertRaises(OSError):
            os.fstat(observed["completion_fd"])
        with self.assertRaises(OSError):
            os.fstat(observed["privacy_fd"])

        tree = ast.parse(Path(completion.__file__).read_text(encoding="utf-8"))
        attributes = [
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        ]
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn("tarfile", names)
        self.assertNotIn("getmembers", attributes)
        self.assertNotIn("extract", attributes)

    def test_macos_sdist_positive_unmocked_private_validation(self):
        descriptor = self.write_private_sdist()
        reader = completion.ArtifactReader(self.directory, self.candidate)
        artifact, inventory = reader.load_private_sdist(descriptor, "unmocked sdist")
        self.assertEqual(artifact.raw, (self.directory / descriptor["path"]).read_bytes())
        self.assertEqual(inventory.archive_sha256, descriptor["sha256"])
        self.assertEqual(inventory.archive_size, descriptor["size_bytes"])
        self.assertGreaterEqual(inventory.member_count, 1)
        self.assertEqual(inventory.member_count, len(inventory.member_names))

    def test_macos_sdist_rejects_symlink_and_path_replacement(self):
        symlink_descriptor = self.write_private_sdist(
            "packages/unsafe-symlink.tar.gz", symlink=True
        )
        reader = completion.ArtifactReader(self.directory, self.candidate)
        with mock.patch.object(
            privacy,
            "_validate_private_sdist_raw_first",
            side_effect=privacy.PrivacyError("synthetic-symlink-rejection"),
        ):
            with self.assertRaisesRegex(
                completion.EvidenceError, "macOS sdist privacy validation failed"
            ):
                reader.load_private_sdist(symlink_descriptor, "unsafe sdist")

        descriptor = self.write_private_sdist("packages/replaced.tar.gz")
        original = (self.directory / descriptor["path"]).read_bytes()
        replacement = self.external_directory / "benign.tar.gz"
        benign = self.write_private_sdist(
            "packages/benign.tar.gz", payload=b"benign replacement\n"
        )
        replacement.write_bytes((self.directory / benign["path"]).read_bytes())
        target = self.directory / descriptor["path"]
        observed = {}

        def reject_retained_source(source, _forbidden_values, *, limits):
            observed["raw"] = os.pread(source.fd, descriptor["size_bytes"], 0)
            os.replace(replacement, target)
            raise privacy.PrivacyError("synthetic-private-sdist-marker")

        with mock.patch.object(
            privacy,
            "_validate_private_sdist_raw_first",
            side_effect=reject_retained_source,
        ):
            with self.assertRaisesRegex(
                completion.EvidenceError, "macOS sdist privacy validation failed"
            ) as caught:
                completion.ArtifactReader(
                    self.directory, self.candidate
                ).load_private_sdist(descriptor, "replaced sdist")
        self.assertEqual(observed["raw"], original)
        self.assertNotEqual(target.read_bytes(), original)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn("synthetic-private-sdist-marker", repr(caught.exception))

    def test_macos_sdist_retained_mutation_and_descriptor_mismatch_fail_closed(self):
        descriptor = self.write_private_sdist("packages/mutable.tar.gz")
        raw = (self.directory / descriptor["path"]).read_bytes()
        member = privacy._PrivateSdistMember(
            "gmes-0.10.0/PKG-INFO",
            "file",
            len(b"Metadata-Version: 2.4\nName: gmes\n"),
            0,
            hashlib.sha256(b"Metadata-Version: 2.4\nName: gmes\n").hexdigest(),
        )
        result = self.private_sdist_result(raw, (member,))

        def truncate_retained_source(source, _forbidden_values, *, limits):
            os.truncate(self.directory / descriptor["path"], 0)
            return result

        with mock.patch.object(
            privacy,
            "_validate_private_sdist_raw_first",
            side_effect=truncate_retained_source,
        ):
            with self.assertRaisesRegex(
                completion.EvidenceError, "macOS sdist source changed during validation"
            ):
                completion.ArtifactReader(
                    self.directory, self.candidate
                ).load_private_sdist(descriptor, "mutable sdist")
        with self.assertRaisesRegex(
            completion.EvidenceError, "macOS sdist bytes differ from their descriptor"
        ):
            completion.ArtifactReader(
                self.directory, self.candidate
            ).load_private_sdist(descriptor, "truncated sdist")

        mismatch = self.write_private_sdist("packages/digest-mismatch.tar.gz")
        mismatch["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            completion.EvidenceError, "macOS sdist bytes differ from their descriptor"
        ):
            completion.ArtifactReader(
                self.directory, self.candidate
            ).load_private_sdist(mismatch, "digest mismatch")
        mismatch = self.write_private_sdist("packages/size-mismatch.tar.gz")
        mismatch["size_bytes"] += 1
        with self.assertRaisesRegex(
            completion.EvidenceError, "macOS sdist bytes differ from their descriptor"
        ):
            completion.ArtifactReader(
                self.directory, self.candidate
            ).load_private_sdist(mismatch, "size mismatch")

    def test_macos_sdist_inventory_and_descriptor_close_failures_are_sanitized(self):
        descriptor = self.write_private_sdist("packages/inventory.tar.gz")
        raw = (self.directory / descriptor["path"]).read_bytes()
        invalid_member = privacy._PrivateSdistMember(
            "gmes-0.10.0/link",
            "symlink",
            0,
            0,
            hashlib.sha256(b"").hexdigest(),
        )
        result = self.private_sdist_result(raw, (invalid_member,))
        with mock.patch.object(
            privacy,
            "_validate_private_sdist_raw_first",
            return_value=result,
        ):
            with self.assertRaisesRegex(
                completion.EvidenceError, "macOS sdist structural inventory differs"
            ) as caught:
                completion.ArtifactReader(
                    self.directory, self.candidate
                ).load_private_sdist(descriptor, "invalid inventory")
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

        path = self.directory / descriptor["path"]
        metadata = path.stat()
        lease = completion._RetainedArtifactLease(
            path,
            completion._artifact_descriptor_identity(metadata),
            metadata.st_size,
            descriptor["sha256"],
        )
        captured = {}
        original_close = completion.os.close

        def fail_close(fd):
            captured["fd"] = fd
            raise OSError("synthetic-close-marker")

        with mock.patch.object(completion.os, "close", side_effect=fail_close):
            with self.assertRaisesRegex(
                completion.EvidenceError, "macOS sdist descriptor could not be closed"
            ) as caught:
                with lease:
                    pass
        original_close(captured["fd"])
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn("synthetic-close-marker", repr(caught.exception))

    def test_macos_scope_has_one_authoritative_v2_definition(self):
        source_path = Path(__file__).resolve().parents[1] / "benchmarks" / "issue123_completion.py"
        self.assertEqual(source_path.resolve(), Path(completion.__file__).resolve())
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        definitions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_macos_scope"
        ]
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0].name, completion._validate_macos_scope.__name__)

        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("placeholder", b"safe")
        archive = self.write_bytes(
            "macos/actions.zip",
            archive_buffer.getvalue(),
            media_type=completion.MEDIA_TYPE_ZIP,
        )
        index = self.write_json(
            "macos/index.json",
            {
                "schema_version": 1,
                "kind": completion.MACOS_INDEX_KIND,
                "candidate_evidence": self.candidate,
                "actions_artifact": {},
                "packages": [],
                "runtime_checks": [],
                "passed": True,
            },
        )
        with self.assertRaises(completion.EvidenceError):
            completion._validate_macos_scope(
                {"index": index, "actions_archive": archive},
                completion.ArtifactReader(self.directory, self.candidate),
                self.candidate,
            )

    def test_npz_npy_headers_are_bounded_before_numpy_load(self):
        def truncated_npz(shape):
            member = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                member,
                {"descr": "<f8", "fortran_order": False, "shape": shape},
            )
            payload = io.BytesIO()
            with zipfile.ZipFile(
                payload, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("value.npy", member.getvalue())
            return payload.getvalue()

        cases = (
            ((2**31 - 1,), "declared payload exceeds"),
            ((4,), "header and payload size differ"),
        )
        for shape, message in cases:
            with self.subTest(shape=shape):
                raw = truncated_npz(shape)
                artifact = completion.LoadedArtifact(
                    descriptor={},
                    path=self.directory / "truncated.npz",
                    raw=raw,
                    document=None,
                )
                with (
                    mock.patch.object(
                        np, "load", side_effect=AssertionError("unsafe allocation")
                    ) as load,
                    self.assertRaisesRegex(completion.EvidenceError, message),
                ):
                    completion._npz_arrays(artifact, ["value"], "truncated")
                load.assert_not_called()

    def test_differential_npz_requires_deflate_and_npy_v1_before_load(self):
        comment = b"differential-group-binding"

        def group_npz(
            *,
            version=1,
            compression=zipfile.ZIP_DEFLATED,
            descr="<f8",
            shape=(1,),
            fortran_order=False,
            payload_bytes=None,
            external_attr=None,
        ):
            member = io.BytesIO()
            writer = (
                np.lib.format.write_array_header_1_0
                if version == 1
                else np.lib.format.write_array_header_2_0
            )
            writer(
                member,
                {
                    "descr": descr,
                    "fortran_order": fortran_order,
                    "shape": shape,
                },
            )
            if payload_bytes is None:
                payload_bytes = math.prod(shape) * np.dtype(descr).itemsize
            member.write(bytes(payload_bytes))
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w") as archive:
                info = zipfile.ZipInfo("value.npy")
                info.compress_type = compression
                if external_attr is not None:
                    info.create_system = 3
                    info.external_attr = external_attr
                archive.writestr(info, member.getvalue())
                archive.comment = comment
            return payload.getvalue()

        valid = group_npz()
        malformed = {
            "tiny-huge-shape": group_npz(shape=(2**31 - 1,), payload_bytes=0),
            "truncated": group_npz(shape=(2,), payload_bytes=8),
            "padded": group_npz(payload_bytes=16),
            "unsupported-version": group_npz(version=2),
            "object": group_npz(descr="|O", payload_bytes=8),
            "structured": group_npz(
                descr=np.dtype([("value", "<f8")]).descr,
                payload_bytes=8,
            ),
            "subarray": group_npz(
                descr=("<f8", (2,)),
                payload_bytes=16,
            ),
            "fortran": group_npz(fortran_order=True),
            "stored": group_npz(compression=zipfile.ZIP_STORED),
            "symlink": group_npz(external_attr=0o120777 << 16),
            "directory-mode": group_npz(external_attr=0o040755 << 16),
            "prepended": b"stub" + valid,
            "trailing": valid + b"padding",
        }
        encrypted = bytearray(valid)
        with zipfile.ZipFile(io.BytesIO(valid)) as archive:
            first_member = archive.infolist()[0]
        encrypted[first_member.header_offset + 6] |= 0x01
        central = encrypted.find(b"PK\x01\x02")
        self.assertGreaterEqual(central, 0)
        encrypted[central + 8] |= 0x01
        malformed["encrypted"] = bytes(encrypted)
        local_encrypted = bytearray(valid)
        local_encrypted[first_member.header_offset + 6] |= 0x01
        malformed["local-only-encrypted"] = bytes(local_encrypted)
        local_stored = bytearray(valid)
        local_stored[
            first_member.header_offset + 8 : first_member.header_offset + 10
        ] = zipfile.ZIP_STORED.to_bytes(2, "little")
        malformed["local-only-stored"] = bytes(local_stored)
        corrupted = bytearray(valid)
        with zipfile.ZipFile(io.BytesIO(valid)) as archive:
            info = archive.infolist()[0]
        local = info.header_offset
        filename_size = int.from_bytes(corrupted[local + 26 : local + 28], "little")
        extra_size = int.from_bytes(corrupted[local + 28 : local + 30], "little")
        compressed_start = local + 30 + filename_size + extra_size
        corrupted[compressed_start + info.compress_size // 2] ^= 0x01
        malformed["corrupt-deflate"] = bytes(corrupted)
        invalid_deflate = bytearray(valid)
        invalid_deflate[compressed_start] = (
            invalid_deflate[compressed_start] & ~0x07
        ) | 0x07
        malformed["invalid-deflate"] = bytes(invalid_deflate)

        for mutation, raw in malformed.items():
            with self.subTest(mutation=mutation):
                artifact = completion.LoadedArtifact(
                    descriptor={},
                    path=self.directory / "group.npz",
                    raw=raw,
                    document=None,
                )
                with (
                    mock.patch.object(
                        np, "load", side_effect=AssertionError("unsafe allocation")
                    ) as load,
                    self.assertRaises(completion.EvidenceError),
                ):
                    completion._npz_arrays(
                        artifact,
                        ["value"],
                        "group",
                        expected_comment=comment,
                    )
                load.assert_not_called()

        artifact = completion.LoadedArtifact(
            descriptor={},
            path=self.directory / "stored.npz",
            raw=group_npz(compression=zipfile.ZIP_STORED),
            document=None,
        )
        arrays = completion._npz_arrays(artifact, ["value"], "generic stored")
        np.testing.assert_array_equal(arrays["value"], np.zeros(1, dtype="<f8"))

        bounded = completion.LoadedArtifact(
            descriptor={},
            path=self.directory / "bounded.npz",
            raw=valid,
            document=None,
        )
        with (
            mock.patch.object(completion, "MAX_NPZ_ARCHIVE_BYTES", len(valid) - 1),
            mock.patch.object(
                np, "load", side_effect=AssertionError("unsafe allocation")
            ) as load,
            self.assertRaisesRegex(completion.EvidenceError, "archive exceeds"),
        ):
            completion._npz_arrays(
                bounded,
                ["value"],
                "bounded group",
                expected_comment=comment,
            )
        load.assert_not_called()

    def test_generic_correctness_npz_preflight_preserves_bounded_compatibility(self):
        def npy_member(version, array):
            stream = io.BytesIO()
            header = {
                "descr": np.lib.format.dtype_to_descr(array.dtype),
                "fortran_order": False,
                "shape": array.shape,
            }
            if version == (1, 0):
                np.lib.format.write_array_header_1_0(stream, header)
            elif version == (2, 0):
                np.lib.format.write_array_header_2_0(stream, header)
            else:
                encoded = repr(header).encode("utf-8")
                padding = (-(12 + len(encoded) + 1)) % 64
                encoded_header = encoded + b" " * padding + b"\n"
                stream.write(b"\x93NUMPY\x03\x00")
                stream.write(len(encoded_header).to_bytes(4, "little"))
                stream.write(encoded_header)
            stream.write(array.tobytes(order="C"))
            return stream.getvalue()

        def generic_npz(version, *, string_name="metadata.json", string_shape=()):
            output = io.BytesIO()
            with zipfile.ZipFile(
                output, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                archive.writestr(
                    "value.npy",
                    npy_member(version, np.asarray([1.0], dtype=np.float64)),
                )
                metadata = np.asarray("{}", dtype="<U2")
                if string_shape:
                    metadata = np.full(string_shape, "{}", dtype="<U2")
                archive.writestr(f"{string_name}.npy", npy_member(version, metadata))
            return output.getvalue()

        for version in ((1, 0), (2, 0), (3, 0)):
            with self.subTest(version=version):
                raw = generic_npz(version)
                with mock.patch.object(
                    np, "load", side_effect=AssertionError("unexpected NumPy load")
                ) as load:
                    completion._validate_media_payload(
                        raw, completion.MEDIA_TYPE_NPZ, "generic correctness"
                    )
                load.assert_not_called()

        for mutation, raw in (
            ("renamed", generic_npz((1, 0), string_name="invented")),
            ("shaped", generic_npz((1, 0), string_shape=(1,))),
        ):
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(completion.EvidenceError, "plain numeric type"),
            ):
                completion._validate_media_payload(
                    raw, completion.MEDIA_TYPE_NPZ, "generic correctness"
                )
        with (
            mock.patch.object(completion, "MAX_CORRECTNESS_NPZ_METADATA_BYTES", 4),
            self.assertRaisesRegex(completion.EvidenceError, "plain numeric type"),
        ):
            completion._validate_media_payload(
                generic_npz((1, 0)),
                completion.MEDIA_TYPE_NPZ,
                "oversized correctness metadata",
            )

    def test_zero_cpu_allocation_uses_empty_provenance_closure(self):
        traces = {
            "a"
            * 64: (
                SimpleNamespace(),
                {
                    "positive_allocation_events": 0,
                    "allocated_bytes": 0,
                    "freed_bytes": 0,
                    "allocation_net_bytes": 0,
                },
            )
        }
        self.assertTrue(completion._all_traces_have_zero_allocations(traces))
        scope = {"allocation_sidecars": [], "generated_sources": []}
        aggregate = {"allocation_provenance_artifact": None}
        reader = completion.ArtifactReader(self.directory, self.candidate)
        self.assertEqual(
            completion._load_cpu_allocation_evidence(
                scope,
                reader,
                aggregate,
                all_zero=True,
            ),
            (None, {}),
        )

        traces["a" * 64][1]["allocated_bytes"] = 8
        self.assertFalse(completion._all_traces_have_zero_allocations(traces))
        with self.assertRaisesRegex(completion.EvidenceError, "sidecars"):
            completion._load_cpu_allocation_evidence(
                scope,
                reader,
                aggregate,
                all_zero=False,
            )
        aggregate["allocation_provenance_artifact"] = {"path": "sidecar.json"}
        with self.assertRaisesRegex(completion.EvidenceError, "must not carry"):
            completion._load_cpu_allocation_evidence(
                scope,
                reader,
                aggregate,
                all_zero=True,
            )

    def test_cuda_memory_peak_covers_both_allocated_samples(self):
        single_gpu = {
            "memory": {
                "bounded": True,
                "cuda_allocated_before_bytes": 100,
                "cuda_allocated_after_bytes": 120,
                "cuda_allocated_growth_bytes": 20,
                "cuda_peak_allocated_bytes": 120,
                "cuda_peak_reserved_bytes": 140,
            }
        }
        completion._validate_cuda_memory(single_gpu, "single GPU")
        for field, value in (
            ("cuda_allocated_before_bytes", 121),
            ("cuda_allocated_after_bytes", 121),
        ):
            with self.subTest(scope="single", field=field):
                malformed = copy.deepcopy(single_gpu)
                malformed["memory"][field] = value
                malformed["memory"]["cuda_allocated_growth_bytes"] = (
                    malformed["memory"]["cuda_allocated_after_bytes"]
                    - malformed["memory"]["cuda_allocated_before_bytes"]
                )
                with self.assertRaisesRegex(
                    completion.EvidenceError, "CUDA memory gate failed"
                ):
                    completion._validate_cuda_memory(malformed, "single GPU")

        two_gpu = {
            "allocated_before_bytes": 100,
            "allocated_after_bytes": 120,
            "allocated_growth_bytes": 20,
            "peak_allocated_bytes": 120,
            "peak_reserved_bytes": 140,
            "bounded": True,
        }
        completion._validate_two_gpu_memory(two_gpu, "rank memory")
        for field, value in (
            ("allocated_before_bytes", 121),
            ("allocated_after_bytes", 121),
        ):
            with self.subTest(scope="two", field=field):
                malformed = copy.deepcopy(two_gpu)
                malformed[field] = value
                malformed["allocated_growth_bytes"] = (
                    malformed["allocated_after_bytes"]
                    - malformed["allocated_before_bytes"]
                )
                with self.assertRaisesRegex(
                    completion.EvidenceError, "independently bounded"
                ):
                    completion._validate_two_gpu_memory(malformed, "rank memory")

    def test_primary_artifact_requires_embedded_candidate(self):
        with self.assertRaisesRegex(completion.EvidenceError, "no embedded candidate"):
            completion._document_candidate_matches(
                {},
                self.candidate,
                required=True,
            )

    def test_cpu_embedded_pass_cannot_hide_raw_regression(self):
        reference = [1.0] * 15
        candidate = [1.06] * 15
        gate = {
            "comparison_valid": True,
            "contract_errors": [],
            "reference_source_artifact_sha256": "c" * 64,
            "reference_root_commit": "d" * 40,
            "reference_raw_seconds_per_step": reference,
            "candidate_raw_seconds_per_step": candidate,
            "reference_seconds_per_step": 1.0,
            "candidate_seconds_per_step": 1.06,
            "candidate_to_torch_baseline_ratio": 1.06,
            "individual_ratio_limit": 1.05,
            "within_five_percent": True,
        }
        with self.assertRaisesRegex(
            completion.EvidenceError,
            "exceeds the individual ratio",
        ):
            completion._validate_cpu_gate(
                gate,
                "CPU gate",
                max_ratio=1.05,
                max_relative_mad=0.05,
                expected_reference=reference,
                expected_candidate=candidate,
                expected_source_sha256="c" * 64,
                expected_root_commit="d" * 40,
            )

    def test_native_summary_must_match_the_manifest_pinned_digest(self):
        reference = self.manifest["reference"]
        document = {key: None for key in completion.NATIVE_SUMMARY_KEYS}
        document.update(
            {
                "schema_version": 3,
                "kind": "native-cpu-acceptance-summary",
                "physics_reference": reference["tag"],
                "observer_tag": reference["performance_observer_tag"],
                "observer_commit": reference["performance_observer_commit"],
            }
        )
        artifact = completion.LoadedArtifact(
            {"sha256": "0" * 64},
            Path("native-summary.json"),
            b"",
            document,
        )
        with self.assertRaisesRegex(completion.EvidenceError, "exact bytes"):
            completion._validate_native_summary(
                artifact,
                self.manifest,
                [],
                completion.DEFAULT_MANIFEST,
            )

    def test_single_gpu_speedup_is_recomputed_from_cross_scope_raw_samples(self):
        values = self.patched_validators()
        cpu = values["_validate_cpu_scope"]
        single = values["_validate_single_gpu_scope"]
        cpu["native_raw_seconds_per_step"]["physical"]["cpu-large-3d"] = [1.9] * 15
        with self.assertRaisesRegex(completion.EvidenceError, "speedup gate failed"):
            completion._validate_cpu_gpu_contract(
                cpu,
                single,
                self.manifest,
            )

    def test_above_crossover_cuda_must_beat_best_cpu_raw_samples(self):
        values = self.patched_validators()
        cpu = values["_validate_cpu_scope"]
        single = values["_validate_single_gpu_scope"]
        single["cuda_raw_seconds_per_step"]["cpu-large-2d"] = [2.1] * 15
        with self.assertRaisesRegex(
            completion.EvidenceError,
            "does not beat the best same-host CPU",
        ):
            completion._validate_cpu_gpu_contract(
                cpu,
                single,
                self.manifest,
            )

    def test_trace_metrics_are_recomputed_from_exact_bytes(self):
        trace = {
            "traceEvents": [
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 0,
                    "dur": 2,
                },
                {
                    "name": "Memcpy HtoD (Pageable -> Device)",
                    "cat": "gpu_memcpy",
                    "ph": "X",
                    "ts": 2,
                    "dur": 1,
                },
                {
                    "name": "Memcpy DtoH (Device -> Pageable)",
                    "cat": "gpu_memcpy",
                    "ph": "X",
                    "ts": 3,
                    "dur": 1,
                },
            ]
        }
        descriptor = self.write_json("trace.json", trace)
        reader = completion.ArtifactReader(self.directory, self.candidate)
        artifact = reader.load(descriptor, "trace", json_document=False)
        summary = completion._trace_summary(artifact.raw, "trace")
        self.assertEqual(summary["host_to_device_events"], 1)
        self.assertEqual(summary["device_to_host_events"], 1)
        traces = {descriptor["sha256"]: (artifact, summary)}
        profiler = {
            "chrome_trace_sha256": descriptor["sha256"],
            "chrome_trace_size_bytes": descriptor["size_bytes"],
            "kernel_launches": 1,
        }
        completion._bind_tuning_traces({"profiler": profiler}, traces, "test")
        profiler["kernel_launches"] = 2
        with self.assertRaisesRegex(
            completion.EvidenceError,
            "differs from trace bytes",
        ):
            completion._bind_tuning_traces(
                {"profiler": profiler},
                traces,
                "test",
            )

    def test_tuning_acceptance_binds_boundary_execution_representation(self):
        counters = {name: 0 for name in completion.COMPILER_COUNTER_FIELDS}
        result = {
            "diagnostics": {
                "boundaries": {
                    "scheduling": "external",
                    "execution_representation": (
                        completion.BOUNDARY_SYNC_REPRESENTATION
                    ),
                    "paired_real_scratch_bytes": 0,
                }
            },
            "acceptance": {name: True for name in completion.TUNING_ACCEPTANCE_FIELDS},
            "compiler": {
                "after_cold": dict(counters),
                "after_warmup": dict(counters),
                "after_steady": dict(counters),
                "steady_state_delta": dict(counters),
                "fullgraph_clean": True,
            },
            "memory": {
                "storage_addresses_before": {"state.ex": 1},
                "storage_addresses_after": {"state.ex": 1},
                "storage_addresses_stable": True,
                "bounded": True,
            },
            "allocation_contract": {"satisfied": True},
        }
        completion._validate_tuning_acceptance(result, "test")
        for malformed in (None, []):
            with (
                self.subTest(diagnostics=malformed),
                self.assertRaisesRegex(
                    completion.EvidenceError, "boundary execution diagnostics"
                ),
            ):
                invalid = copy.deepcopy(result)
                invalid["diagnostics"] = malformed
                completion._validate_tuning_acceptance(invalid, "test")
        extra = copy.deepcopy(result)
        extra["diagnostics"]["boundaries"]["extra"] = True
        with self.assertRaisesRegex(
            completion.EvidenceError, "boundary execution diagnostics"
        ):
            completion._validate_tuning_acceptance(extra, "test")
        result["diagnostics"]["boundaries"]["execution_representation"] = "tampered"
        with self.assertRaisesRegex(completion.EvidenceError, "boundary execution"):
            completion._validate_tuning_acceptance(result, "test")

    def test_cuda_state_finiteness_is_recomputed_fail_closed(self):
        reference = self.manifest["reference"]
        workload = copy.deepcopy(
            next(
                item
                for item in self.manifest["benchmarks"]
                if item["name"] == "single-gpu-3d"
            )
        )
        fields = [name.lower() for name in completion.FIELD_ARRAYS]
        material_inventory = completion._expected_cuda_persistent_material_inventory(
            workload, "test"
        )
        material_names = sorted(material_inventory)
        changed = sorted(fields + material_names + ["source_time", "step_count"])
        warmup = reference["performance_warmup_steps"]
        state = {
            "initial_checksum": 1.0,
            "post_warmup_checksum": 2.0,
            "post_one_step_checksum": 3.0,
            "final_checksum": 4.0,
            "changed_after_first_timed_step": True,
            "one_step_count": warmup + 1,
            "expected_one_step_count": warmup + 1,
            "timed_step_count": warmup + reference["performance_steps_per_repeat"],
            "expected_timed_step_count": warmup
            + reference["performance_steps_per_repeat"],
            "profiler_step_count": warmup + reference["performance_profile_steps"],
            "expected_profiler_step_count": warmup
            + reference["performance_profile_steps"],
            "changed_buffers": changed,
            "fields_changed": sorted(fields),
            "all_fields_changed": True,
            "pml_state_changed": True,
            "dispersive_state_changed": True,
            "dm2_state_changed": False,
        }
        result = {
            "workload": workload,
            "runtime": {
                "precision": "float64",
                "field_storage_channels": 1,
            },
            "profiler": {},
            "diagnostics": {
                "material_plan": self.frozen_material_plan(workload, "float64")
            },
            "state_progress": state,
            "state_finiteness": {
                "contract_id": completion.STATE_FINITENESS_CONTRACT_ID,
                "tracked_buffers": sorted(changed + ["time_step"]),
                "stages": {
                    stage: {
                        "floating_or_complex_buffer_count": (
                            len(fields) + len(material_inventory) + 2
                        ),
                        "floating_or_complex_element_count": 0,
                        "nonfinite_element_count": 0,
                        "finite": True,
                    }
                    for stage in (
                        "initial",
                        "post_warmup",
                        "post_one_step",
                        "post_timed",
                        "post_profile",
                    )
                },
                "passed": True,
            },
        }
        field_sizes, element_size = completion._expected_cuda_field_buffer_sizes(
            result, "test"
        )
        result["profiler"]["field_buffer_sizes_bytes"] = {
            **field_sizes,
            "aggregate.all-fields": sum(field_sizes.values()),
        }
        expected_elements = (
            sum(field_sizes.values()) // element_size
            + sum(
                width * targets
                for _family, width, targets in material_inventory.values()
            )
            + 2
        )
        for record in result["state_finiteness"]["stages"].values():
            record["floating_or_complex_element_count"] = expected_elements
        self.assertEqual(len(material_inventory), 45)
        self.assertEqual(material_inventory["pml_ex_0_state"], ("pml", 2, 179168))
        self.assertEqual(
            material_inventory["bucket_ex_1_field_old"],
            ("dispersive", 1, 18240),
        )
        self.assertEqual(
            material_inventory["bucket_ex_2_point_state"],
            ("dispersive", 4, 16416),
        )
        self.assertEqual(
            material_inventory["bucket_ex_5_previous"],
            ("dispersive", 1, 18240),
        )
        completion._validate_cuda_state_finiteness(result, reference, "test")

        missing_changed = copy.deepcopy(result)
        missing_changed["state_finiteness"]["tracked_buffers"].remove(material_names[0])
        with self.assertRaisesRegex(completion.EvidenceError, "inventory closure"):
            completion._validate_cuda_state_finiteness(
                missing_changed, reference, "test"
            )

        invalid_checksum = copy.deepcopy(result)
        invalid_checksum["state_progress"]["final_checksum"] = float("nan")
        with self.assertRaisesRegex(completion.EvidenceError, "non-finite"):
            completion._validate_cuda_state_finiteness(
                invalid_checksum, reference, "test"
            )

        invalid_state = copy.deepcopy(result)
        invalid_state["state_finiteness"]["stages"]["post_timed"][
            "nonfinite_element_count"
        ] = 1
        with self.assertRaisesRegex(completion.EvidenceError, "finiteness"):
            completion._validate_cuda_state_finiteness(invalid_state, reference, "test")

        undersized = copy.deepcopy(result)
        for record in undersized["state_finiteness"]["stages"].values():
            record["floating_or_complex_buffer_count"] = 1
            record["floating_or_complex_element_count"] = 1
        with self.assertRaisesRegex(completion.EvidenceError, "finiteness"):
            completion._validate_cuda_state_finiteness(undersized, reference, "test")

        oversized = copy.deepcopy(result)
        for record in oversized["state_finiteness"]["stages"].values():
            record["floating_or_complex_element_count"] += 1
        with self.assertRaisesRegex(completion.EvidenceError, "finiteness"):
            completion._validate_cuda_state_finiteness(oversized, reference, "test")

        contradictory_flag = copy.deepcopy(result)
        contradictory_flag["state_progress"]["dispersive_state_changed"] = False
        with self.assertRaisesRegex(completion.EvidenceError, "dynamic-state"):
            completion._validate_cuda_state_finiteness(
                contradictory_flag, reference, "test"
            )

        missing_material_progress = copy.deepcopy(result)
        missing_material_progress["state_progress"]["changed_buffers"] = [
            name
            for name in missing_material_progress["state_progress"]["changed_buffers"]
            if not name.startswith(("pml_", "bucket_", "dm2_buckets."))
        ]
        for name in (
            "pml_state_changed",
            "dispersive_state_changed",
            "dm2_state_changed",
        ):
            missing_material_progress["state_progress"][name] = False
        with self.assertRaisesRegex(completion.EvidenceError, "dynamic-state"):
            completion._validate_cuda_state_finiteness(
                missing_material_progress, reference, "test"
            )

        fabricated_prefixes = copy.deepcopy(result)
        fabricated = ["bucket_spoof", "dm2_buckets.spoof", "pml_spoof"]
        fabricated_prefixes["state_progress"]["changed_buffers"] = sorted(
            [
                name
                for name in fabricated_prefixes["state_progress"]["changed_buffers"]
                if name not in material_inventory
            ]
            + fabricated
        )
        fabricated_prefixes["state_finiteness"]["tracked_buffers"] = sorted(
            [
                name
                for name in fabricated_prefixes["state_finiteness"]["tracked_buffers"]
                if name not in material_inventory
            ]
            + fabricated
        )
        for name in (
            "pml_state_changed",
            "dispersive_state_changed",
            "dm2_state_changed",
        ):
            fabricated_prefixes["state_progress"][name] = True
        with self.assertRaisesRegex(completion.EvidenceError, "dynamic-state"):
            completion._validate_cuda_state_finiteness(
                fabricated_prefixes, reference, "test"
            )

        fabricated_targets = copy.deepcopy(result)
        fabricated_targets["diagnostics"]["material_plan"][0]["buckets"][0][
            "targets"
        ] += 1
        with self.assertRaisesRegex(completion.EvidenceError, "material-plan"):
            completion._validate_cuda_state_finiteness(
                fabricated_targets, reference, "test"
            )

        tiny_field_sizes = copy.deepcopy(result)
        tiny_field_sizes["profiler"]["field_buffer_sizes_bytes"] = {
            **{f"state.{component}": 8 for component in completion.FIELD_ARRAYS},
            "aggregate.all-fields": 48,
        }
        with self.assertRaisesRegex(completion.EvidenceError, "field-size inventory"):
            completion._validate_cuda_state_finiteness(
                tiny_field_sizes, reference, "test"
            )

        mixed_names = copy.deepcopy(result)
        mixed_names["state_progress"]["changed_buffers"] = ["ex", 1]
        with self.assertRaisesRegex(completion.EvidenceError, "dynamic-state"):
            completion._validate_cuda_state_finiteness(mixed_names, reference, "test")

    def test_frozen_material_inventory_covers_every_required_lowered_case(self):
        required = set(completion.POLICY_CASES) | set(completion.PAIRED_REAL_CASES)
        required |= set(completion.CUDA_CASES) | set(completion.REGION_INVARIANCE_CASES)
        self.assertEqual(
            set(completion._FROZEN_CUDA_MATERIAL_TOPOLOGY_BY_CASE), required
        )
        self.assertEqual(
            set(completion._FROZEN_CUDA_MATERIAL_TARGETS_BY_CASE), required
        )
        self.assertEqual(
            set(completion._FROZEN_CUDA_MATERIAL_PLAN_SHA256_BY_CASE), required
        )
        expected_tracked_counts = {
            "coverage-1-fragmented": 15,
            "cpu-crossover-3d": 44,
            "cpu-crossover-2d": 57,
            "bloch-2d": 9,
            "equivalent-region-1": 15,
        }
        for name in sorted(required):
            with self.subTest(case=name):
                workload = {"name": name}
                inventory = completion._expected_cuda_persistent_material_inventory(
                    workload, "test"
                )
                if name in expected_tracked_counts:
                    self.assertEqual(
                        len(completion.FIELD_ARRAYS) + len(inventory) + 3,
                        expected_tracked_counts[name],
                    )
                completion._validate_frozen_cuda_material_plan(
                    {
                        "workload": workload,
                        "runtime": {"precision": "float32"},
                        "diagnostics": {
                            "material_plan": self.frozen_material_plan(
                                workload, "float32"
                            )
                        },
                    },
                    "test",
                )

    def test_bloch_3d_real_cpu_eager_inventory_matches_frozen_contract(self):
        workload, space, geometry, sources, bloch = torch_tuning._build_case(
            "bloch-3d", self.manifest
        )
        simulation = gmes.TorchSimulation(
            space=space,
            geometry=geometry,
            sources=sources,
            bloch=bloch,
            runtime=gmes.TorchRuntimeConfig(
                device="cpu",
                precision="float32",
                compile_policy="eager",
                compile_mode="default",
                cpu_threads=1,
                cpu_interop_threads=1,
                execution_policy="auto",
                experimental_dispersive_grouping=False,
            ),
        )
        inventory = completion._expected_cuda_persistent_material_inventory(
            workload, "test"
        )
        expected_names = {name.lower() for name in completion.FIELD_ARRAYS} | {
            "source_time",
            "time_step",
            "step_count",
            *inventory,
        }
        self.assertEqual(
            torch_tuning._simulation_dynamic_tensor_names(simulation),
            sorted(expected_names),
        )
        self.assertIn("bucket_ex_0_previous", inventory)
        self.assertIn("bucket_ex_0_current", inventory)
        self.assertFalse(any(name.startswith("bucket_ex_1_") for name in inventory))
        state = simulation.state.state_dict()
        for name, (_family, width, targets) in inventory.items():
            with self.subTest(buffer=name):
                self.assertEqual(state[name].numel(), width * targets * 2)
        diagnostics = json.loads(json.dumps(simulation.diagnostics()))
        completion._validate_frozen_cuda_material_plan(
            {
                "workload": workload,
                "runtime": {"precision": "float32"},
                "diagnostics": diagnostics,
            },
            "test",
        )

    def test_cuda_correctness_indexes_are_loaded_and_exactly_bound(self):
        records = []
        mode_by_path = {}
        bindings_by_path = {}
        reference_bindings = [
            {
                "path": f"correctness/native/{index}.npz",
                "sha256": f"{index + 1:064x}",
            }
            for index in range(
                len(self.manifest["correctness"])
                + len(self.manifest["physical_checks"])
            )
        ]
        for mode in completion.CUDA_CORRECTNESS_RUNTIME_MODES:
            graph_mode = mode["graph_mode"]
            descriptor = self.write_json(
                f"correctness/{graph_mode}.json",
                {
                    "candidate_evidence": self.candidate,
                    "runtime_mode": copy.deepcopy(mode),
                    "graph_mode_marker": graph_mode,
                },
            )
            records.append(
                {
                    "runtime_mode": copy.deepcopy(mode),
                    "source_artifact": descriptor,
                }
            )
            mode_by_path[descriptor["path"]] = copy.deepcopy(mode)
            bindings_by_path[descriptor["path"]] = {
                "reference": copy.deepcopy(reference_bindings),
                "candidate": [
                    {
                        "path": f"correctness/{graph_mode}/{index}.npz",
                        "sha256": f"{offset + index:064x}",
                    }
                    for index in range(len(reference_bindings))
                    for offset in (1000 if graph_mode == "eager" else 2000,)
                ],
            }

        def rebuilt(
            artifact,
            _manifest,
            _candidate,
            _reader,
            *,
            include_archive_bindings=False,
        ):
            result = {
                "runtime_mode": copy.deepcopy(
                    mode_by_path[artifact.descriptor["path"]]
                ),
                "source_artifact": copy.deepcopy(artifact.descriptor),
            }
            if include_archive_bindings:
                return result, copy.deepcopy(
                    bindings_by_path[artifact.descriptor["path"]]
                )
            return result

        with mock.patch.object(
            completion, "_validate_correctness_index", side_effect=rebuilt
        ):
            completion._validate_cuda_correctness_indexes(
                records,
                completion.ArtifactReader(self.directory, self.candidate),
                self.manifest,
                self.candidate,
                "test",
            )

            missing = copy.deepcopy(records)
            missing[0].pop("source_artifact")
            with self.assertRaisesRegex(completion.EvidenceError, "invalid schema"):
                completion._validate_cuda_correctness_indexes(
                    missing,
                    completion.ArtifactReader(self.directory, self.candidate),
                    self.manifest,
                    self.candidate,
                    "test",
                )

            reused = copy.deepcopy(records)
            reused[1]["source_artifact"] = copy.deepcopy(reused[0]["source_artifact"])
            with self.assertRaisesRegex(completion.EvidenceError, "reuse"):
                completion._validate_cuda_correctness_indexes(
                    reused,
                    completion.ArtifactReader(self.directory, self.candidate),
                    self.manifest,
                    self.candidate,
                    "test",
                )

            reordered = list(reversed(copy.deepcopy(records)))
            with self.assertRaisesRegex(
                completion.EvidenceError, "runtime mode differs"
            ):
                completion._validate_cuda_correctness_indexes(
                    reordered,
                    completion.ArtifactReader(self.directory, self.candidate),
                    self.manifest,
                    self.candidate,
                    "test",
                )

            swapped_descriptors = copy.deepcopy(records)
            (
                swapped_descriptors[0]["source_artifact"],
                swapped_descriptors[1]["source_artifact"],
            ) = (
                swapped_descriptors[1]["source_artifact"],
                swapped_descriptors[0]["source_artifact"],
            )
            with self.assertRaisesRegex(
                completion.EvidenceError, "loaded runtime mode differs"
            ):
                completion._validate_cuda_correctness_indexes(
                    swapped_descriptors,
                    completion.ArtifactReader(self.directory, self.candidate),
                    self.manifest,
                    self.candidate,
                    "test",
                )

            fake_digest = copy.deepcopy(records)
            fake_digest[0]["source_artifact"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(completion.EvidenceError, "digest differs"):
                completion._validate_cuda_correctness_indexes(
                    fake_digest,
                    completion.ArtifactReader(self.directory, self.candidate),
                    self.manifest,
                    self.candidate,
                    "test",
                )

            graph_path = records[1]["source_artifact"]["path"]
            original_graph_reference = copy.deepcopy(
                bindings_by_path[graph_path]["reference"]
            )
            bindings_by_path[graph_path]["reference"][0]["sha256"] = "f" * 64
            with self.assertRaisesRegex(
                completion.EvidenceError, "native reference archives differ"
            ):
                completion._validate_cuda_correctness_indexes(
                    records,
                    completion.ArtifactReader(self.directory, self.candidate),
                    self.manifest,
                    self.candidate,
                    "test",
                )
            bindings_by_path[graph_path]["reference"] = original_graph_reference

            original_graph_candidate = copy.deepcopy(
                bindings_by_path[graph_path]["candidate"]
            )
            bindings_by_path[graph_path]["candidate"][0] = copy.deepcopy(
                bindings_by_path[records[0]["source_artifact"]["path"]]["candidate"][0]
            )
            with self.assertRaisesRegex(
                completion.EvidenceError, "candidate archives overlap"
            ):
                completion._validate_cuda_correctness_indexes(
                    records,
                    completion.ArtifactReader(self.directory, self.candidate),
                    self.manifest,
                    self.candidate,
                    "test",
                )
            bindings_by_path[graph_path]["candidate"] = original_graph_candidate

        def fake_rebuilt(
            artifact,
            _manifest,
            _candidate,
            _reader,
            *,
            include_archive_bindings=False,
        ):
            source = copy.deepcopy(artifact.descriptor)
            source["sha256"] = "0" * 64
            result = {
                "runtime_mode": copy.deepcopy(
                    mode_by_path[artifact.descriptor["path"]]
                ),
                "source_artifact": source,
            }
            if include_archive_bindings:
                return result, copy.deepcopy(
                    bindings_by_path[artifact.descriptor["path"]]
                )
            return result

        with (
            mock.patch.object(
                completion,
                "_validate_correctness_index",
                side_effect=fake_rebuilt,
            ),
            self.assertRaisesRegex(
                completion.EvidenceError, "recomputed source artifact"
            ),
        ):
            completion._validate_cuda_correctness_indexes(
                records,
                completion.ArtifactReader(self.directory, self.candidate),
                self.manifest,
                self.candidate,
                "test",
            )

    def test_correctness_archive_records_are_exact_and_unique_without_producer(
        self,
    ):
        expected_evidence = completion._expected_correctness_candidate_evidence(
            self.manifest, self.candidate
        )
        required_pairs = [
            (group, case["name"])
            for group in ("correctness", "physical_checks")
            for case in self.manifest[group]
        ]
        descriptors = []
        for index in range(2 * len(required_pairs)):
            output = io.BytesIO()
            np.savez_compressed(output, marker=np.asarray([index], dtype=np.int64))
            descriptors.append(
                self.write_bytes(
                    f"correctness/archive-{index}.npz",
                    output.getvalue(),
                    media_type=completion.MEDIA_TYPE_NPZ,
                )
            )
        artifacts = []
        for index, (group, case) in enumerate(required_pairs):
            artifacts.append(
                {
                    "case": case,
                    "group": group,
                    "reference": copy.deepcopy(descriptors[2 * index]),
                    "reference_observer_commit": self.manifest["reference"][
                        "observer_commit"
                    ],
                    "candidate": copy.deepcopy(descriptors[2 * index + 1]),
                    "candidate_provenance": {
                        "commit": self.candidate["candidate_git_commit"],
                        "source_sha256": "1" * 64,
                        "controller_sha256": "2" * 64,
                    },
                    "comparison": {"passed": True, "failures": []},
                    "tolerance_results": [],
                }
            )
        receipt_descriptor, trusted_receipt = self.runtime_receipt(
            "cpu", completion.CPU_CORRECTNESS_RUNTIME_MODE, artifacts
        )
        document = {
            "schema_version": completion.CORRECTNESS_INDEX_SCHEMA_VERSION,
            "kind": completion.CORRECTNESS_INDEX_KIND,
            "contract_id": completion.CORRECTNESS_INDEX_CONTRACT_ID,
            "candidate_evidence": expected_evidence,
            "manifest_contract_sha256": completion._canonical_sha256(self.manifest),
            "runtime_mode": {
                "device": "cpu",
                "precision": "float64",
                "graph_mode": "eager",
                "compile_policy": "eager",
                "compile_mode": "default",
            },
            "runtime_receipt": receipt_descriptor,
            "required_cases": [case for _group, case in required_pairs],
            "artifacts": artifacts,
            "suite_acceptance": {
                "correctness_case_count": len(self.manifest["correctness"]),
                "physical_check_case_count": len(self.manifest["physical_checks"]),
                "evaluated_case_count": len(required_pairs),
                "complete_fields": True,
                "persistent_state": True,
                "source_and_auxiliary_state": True,
                "physical_observables": True,
                "passed": True,
            },
        }

        def index_artifact(value):
            return completion.LoadedArtifact(
                descriptor={"sha256": "a" * 64},
                path=self.directory / "correctness-index.json",
                raw=b"",
                document=value,
            )

        def rebuilt(*_args, **_kwargs):
            return {"source_artifact": {"sha256": "a" * 64}}

        with mock.patch(
            "benchmarks.torch_correctness.load_correctness_evidence_index",
            side_effect=rebuilt,
        ):
            completion._validate_correctness_index(
                index_artifact(document),
                self.manifest,
                self.candidate,
                completion.ArtifactReader(self.directory, self.candidate),
                expected_runtime_mode=completion.CPU_CORRECTNESS_RUNTIME_MODE,
                trusted_runtime_receipt=trusted_receipt,
            )

            substituted_receipt = copy.deepcopy(trusted_receipt.document)
            substituted_receipt["workflow"]["job_id"] += 1
            substituted_raw = completion._canonical_json_bytes(substituted_receipt)
            substituted_path = self.external_directory / "substituted-receipt.json"
            substituted_path.write_bytes(substituted_raw)
            substituted_trusted = completion.LoadedArtifact(
                {
                    "path": substituted_path.name,
                    "sha256": hashlib.sha256(substituted_raw).hexdigest(),
                    "size_bytes": len(substituted_raw),
                    "media_type": completion.MEDIA_TYPE_JSON,
                },
                substituted_path,
                substituted_raw,
                substituted_receipt,
            )
            with self.assertRaisesRegex(
                completion.EvidenceError, "bytes differ from the external receipt"
            ):
                completion._validate_correctness_index(
                    index_artifact(document),
                    self.manifest,
                    self.candidate,
                    completion.ArtifactReader(self.directory, self.candidate),
                    expected_runtime_mode=completion.CPU_CORRECTNESS_RUNTIME_MODE,
                    trusted_runtime_receipt=substituted_trusted,
                )

            wrong_scope_runtime = copy.deepcopy(document)
            wrong_scope_runtime["runtime_mode"]["device"] = "cuda:0"
            with self.assertRaisesRegex(
                completion.EvidenceError, "runtime mode differs from the required scope"
            ):
                completion._validate_correctness_index(
                    index_artifact(wrong_scope_runtime),
                    self.manifest,
                    self.candidate,
                    completion.ArtifactReader(self.directory, self.candidate),
                    expected_runtime_mode=completion.CPU_CORRECTNESS_RUNTIME_MODE,
                    trusted_runtime_receipt=trusted_receipt,
                )

            duplicate = copy.deepcopy(document)
            duplicate["artifacts"][1]["reference"] = copy.deepcopy(
                duplicate["artifacts"][0]["reference"]
            )
            with self.assertRaisesRegex(completion.EvidenceError, "reuse"):
                completion._validate_correctness_index(
                    index_artifact(duplicate),
                    self.manifest,
                    self.candidate,
                    completion.ArtifactReader(self.directory, self.candidate),
                    trusted_runtime_receipt=trusted_receipt,
                )

            relabeled = copy.deepcopy(document)
            relabeled["artifacts"][0]["case"] = required_pairs[1][1]
            with self.assertRaisesRegex(completion.EvidenceError, "identity differs"):
                completion._validate_correctness_index(
                    index_artifact(relabeled),
                    self.manifest,
                    self.candidate,
                    completion.ArtifactReader(self.directory, self.candidate),
                    trusted_runtime_receipt=trusted_receipt,
                )

            extra = copy.deepcopy(document)
            extra["artifacts"][0]["unexpected"] = True
            with self.assertRaisesRegex(completion.EvidenceError, "invalid schema"):
                completion._validate_correctness_index(
                    index_artifact(extra),
                    self.manifest,
                    self.candidate,
                    completion.ArtifactReader(self.directory, self.candidate),
                    trusted_runtime_receipt=trusted_receipt,
                )

            extra_top = copy.deepcopy(document)
            extra_top["unexpected"] = True
            with self.assertRaisesRegex(completion.EvidenceError, "invalid schema"):
                completion._validate_correctness_index(
                    index_artifact(extra_top),
                    self.manifest,
                    self.candidate,
                    completion.ArtifactReader(self.directory, self.candidate),
                    trusted_runtime_receipt=trusted_receipt,
                )

            wrong_runtime = copy.deepcopy(document)
            wrong_runtime["runtime_mode"]["compile_policy"] = "compile"
            with self.assertRaisesRegex(
                completion.EvidenceError, "frozen execution contract"
            ):
                completion._validate_correctness_index(
                    index_artifact(wrong_runtime),
                    self.manifest,
                    self.candidate,
                    completion.ArtifactReader(self.directory, self.candidate),
                    trusted_runtime_receipt=trusted_receipt,
                )

    def test_nested_correctness_npz_headers_fail_before_any_numpy_loader(self):
        expected_evidence = completion._expected_correctness_candidate_evidence(
            self.manifest, self.candidate
        )
        required_pairs = [
            (group, case["name"])
            for group in ("correctness", "physical_checks")
            for case in self.manifest[group]
        ]
        descriptors = []
        for index in range(2 * len(required_pairs)):
            output = io.BytesIO()
            np.savez(output, marker=np.asarray([index], dtype=np.int64))
            descriptors.append(
                self.write_bytes(
                    f"nested/valid-{index}.npz",
                    output.getvalue(),
                    media_type=completion.MEDIA_TYPE_NPZ,
                )
            )
        artifacts = [
            {
                "case": case,
                "group": group,
                "reference": copy.deepcopy(descriptors[2 * index]),
                "reference_observer_commit": self.manifest["reference"][
                    "observer_commit"
                ],
                "candidate": copy.deepcopy(descriptors[2 * index + 1]),
                "candidate_provenance": {
                    "commit": self.candidate["candidate_git_commit"],
                    "source_sha256": "1" * 64,
                    "controller_sha256": "2" * 64,
                },
                "comparison": {"passed": True, "failures": []},
                "tolerance_results": [],
            }
            for index, (group, case) in enumerate(required_pairs)
        ]
        receipt_descriptor, trusted_receipt = self.runtime_receipt(
            "bounded-cpu", completion.CPU_CORRECTNESS_RUNTIME_MODE, artifacts
        )
        base_document = {
            "schema_version": completion.CORRECTNESS_INDEX_SCHEMA_VERSION,
            "kind": completion.CORRECTNESS_INDEX_KIND,
            "contract_id": completion.CORRECTNESS_INDEX_CONTRACT_ID,
            "candidate_evidence": expected_evidence,
            "manifest_contract_sha256": completion._canonical_sha256(self.manifest),
            "runtime_mode": copy.deepcopy(completion.CPU_CORRECTNESS_RUNTIME_MODE),
            "runtime_receipt": receipt_descriptor,
            "required_cases": [case for _group, case in required_pairs],
            "artifacts": artifacts,
            "suite_acceptance": {
                "correctness_case_count": len(self.manifest["correctness"]),
                "physical_check_case_count": len(self.manifest["physical_checks"]),
                "evaluated_case_count": len(required_pairs),
                "complete_fields": True,
                "persistent_state": True,
                "source_and_auxiliary_state": True,
                "physical_observables": True,
                "passed": True,
            },
        }

        def malformed_npz(shape, payload_bytes):
            member = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                member,
                {"descr": "<f8", "fortran_order": False, "shape": shape},
            )
            member.write(bytes(payload_bytes))
            output = io.BytesIO()
            with zipfile.ZipFile(
                output, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                archive.writestr("value.npy", member.getvalue())
            return output.getvalue()

        malformed = {
            "huge": malformed_npz((2**31 - 1,), 0),
            "truncated": malformed_npz((2,), 8),
        }
        for role in ("reference", "candidate"):
            for mutation, raw in malformed.items():
                with self.subTest(role=role, mutation=mutation):
                    document = copy.deepcopy(base_document)
                    document["artifacts"][0][role] = self.write_bytes(
                        f"nested/{role}-{mutation}.npz",
                        raw,
                        media_type=completion.MEDIA_TYPE_NPZ,
                    )
                    artifact = completion.LoadedArtifact(
                        descriptor={"sha256": "a" * 64},
                        path=self.directory / "correctness-index.json",
                        raw=b"",
                        document=document,
                    )
                    with (
                        mock.patch(
                            "benchmarks.torch_correctness.load_correctness_evidence_index",
                            side_effect=AssertionError("producer loader reached"),
                        ) as correctness_load,
                        mock.patch.object(
                            np, "load", side_effect=AssertionError("NumPy load reached")
                        ) as numpy_load,
                        self.assertRaises(completion.EvidenceError),
                    ):
                        completion._validate_correctness_index(
                            artifact,
                            self.manifest,
                            self.candidate,
                            completion.ArtifactReader(self.directory, self.candidate),
                            trusted_runtime_receipt=trusted_receipt,
                        )
                    correctness_load.assert_not_called()
                    numpy_load.assert_not_called()

    def test_correctness_full_candidate_evidence_is_independently_frozen(self):
        expected = completion._expected_correctness_candidate_evidence(
            self.manifest, self.candidate
        )
        self.assertEqual(set(expected), completion.CORRECTNESS_EVIDENCE_KEYS)
        artifact = completion.LoadedArtifact(
            descriptor={"sha256": "f" * 64},
            path=self.directory / "correctness.json",
            raw=b"",
            document=None,
        )
        protected = (
            "evidence_contract_id",
            "cpu_contract_id",
            "runner_sha256",
            "solver_sha256",
            "solver_abi",
        )
        for field in protected:
            with self.subTest(field=field, mutation="omitted"):
                evidence = copy.deepcopy(expected)
                evidence.pop(field)
                malformed = completion.LoadedArtifact(
                    descriptor=artifact.descriptor,
                    path=artifact.path,
                    raw=artifact.raw,
                    document={"candidate_evidence": evidence},
                )
                with self.assertRaisesRegex(
                    completion.EvidenceError, "candidate evidence"
                ):
                    completion._validate_correctness_index(
                        malformed,
                        self.manifest,
                        self.candidate,
                        mock.Mock(),
                    )
            with self.subTest(field=field, mutation="changed"):
                evidence = copy.deepcopy(expected)
                evidence[field] = "0" * 64
                malformed = completion.LoadedArtifact(
                    descriptor=artifact.descriptor,
                    path=artifact.path,
                    raw=artifact.raw,
                    document={"candidate_evidence": evidence},
                )
                with self.assertRaisesRegex(
                    completion.EvidenceError, "candidate evidence"
                ):
                    completion._validate_correctness_index(
                        malformed,
                        self.manifest,
                        self.candidate,
                        mock.Mock(),
                    )

    def test_differential_comparison_is_manifest_derived_in_completion(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        document["cases"][0]["comparison"]["rtol"] = 1.0
        document["cases"][0]["comparison"]["atol"] = 1.0
        with self.assertRaisesRegex(
            completion.EvidenceError, "comparison contract differs from the manifest"
        ):
            fixture.validate_completion_mirror(document)

    def test_differential_closure_is_independently_frozen(self):
        paired = completion._expected_completion_differential_records(
            self.manifest, "paired-real"
        )
        single = completion._expected_completion_differential_records(
            self.manifest, "single-gpu-cuda"
        )
        self.assertEqual(len(paired) + len(single), 18)

        relabeled = copy.deepcopy(self.manifest)
        next(case for case in relabeled["correctness"] if case.get("complex") is True)[
            "complex"
        ] = False
        with self.assertRaisesRegex(completion.EvidenceError, "frozen closure"):
            completion._expected_completion_differential_records(
                relabeled, "paired-real"
            )

        shortened = copy.deepcopy(self.manifest)
        shortened["reference"]["capture_steps"] = [100]
        with self.assertRaisesRegex(completion.EvidenceError, "frozen contract"):
            completion._expected_differential_projection_steps(
                shortened, "single-gpu-cuda", single[0]
            )

        relaxed = copy.deepcopy(self.manifest)
        relaxed["tolerances"]["torch"]["dielectric"]["complex128"]["rtol"] = 1.0
        with self.assertRaisesRegex(completion.EvidenceError, "frozen contract"):
            completion._expected_differential_comparison(
                relaxed, "paired-real", paired[0]
            )

    def test_differential_group_descriptor_lists_are_exact_and_unique(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        descriptors = [
            descriptor
            for record in document["cases"]
            for role in ("reference", "candidate")
            for descriptor in record[role]
        ]
        self.assertEqual(len(descriptors), 8)
        self.assertEqual(len({item["path"] for item in descriptors}), 8)
        self.assertEqual(len({item["sha256"] for item in descriptors}), 8)
        mutations = (
            (
                "missing",
                lambda record: record["reference"].pop(),
                "descriptor group closure",
            ),
            (
                "reordered",
                lambda record: record["reference"].__setitem__(
                    slice(0, 2), reversed(record["reference"][:2])
                ),
                "ZIP comment binding",
            ),
            (
                "duplicate",
                lambda record: record["candidate"].__setitem__(
                    1, copy.deepcopy(record["candidate"][0])
                ),
                "reused",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                fixture = self.differential_fixture()
                document = fixture.document()
                mutate(document["cases"][1])
                with self.assertRaisesRegex(completion.EvidenceError, message):
                    fixture.validate_completion_mirror(document)

    def test_differential_early_steps_cannot_use_normalized_acceptance(self):
        for target_kind in ("field", "persistent-state"):
            with self.subTest(target_kind=target_kind):
                fixture = self.differential_fixture()
                document = fixture.document()
                record = document["cases"][1]
                reference = self.differential_group_arrays(
                    fixture, record, "reference", 0
                )
                candidate = self.differential_group_arrays(
                    fixture, record, "candidate", 0
                )
                if target_kind == "field":
                    target = "step/0/field/Ex"
                else:
                    target = next(
                        name
                        for name in candidate
                        if name.startswith("step/0/state/Ex/")
                        and name.endswith("/0-Cpml/values")
                        and candidate[name].size
                    )
                comparisons = completion._expected_differential_array_comparisons(
                    fixture.manifest,
                    document["scope"],
                    {
                        "case": record["case"],
                        "device": record["device"],
                        "precision": record["precision"],
                    },
                    list(reference),
                )
                self.assertEqual(
                    comparisons[target]["mode"],
                    completion.DIFFERENTIAL_ELEMENTWISE_MODE,
                )
                if target_kind == "persistent-state":
                    self.assertEqual(
                        comparisons[target],
                        {
                            "mode": completion.DIFFERENTIAL_ELEMENTWISE_MODE,
                            **fixture.manifest["tolerances"]["torch"]["pml"]["float64"],
                        },
                    )
                candidate[target].flat[0] += 1.0e-7
                metrics, normalized_passed = completion._recompute_differential_metrics(
                    reference,
                    candidate,
                    record["comparison"],
                    "record-level normalized comparison",
                )
                self.assertTrue(normalized_passed)
                record["metrics"] = metrics
                fixture.rewrite_group(document, 1, "candidate", 0, candidate)
                with self.assertRaisesRegex(
                    completion.EvidenceError,
                    "differs from the projected fields|differential failed",
                ):
                    fixture.validate_completion_mirror(document)

        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][1]
        late = self.differential_group_arrays(fixture, record, "reference", 2)
        comparisons = completion._expected_differential_array_comparisons(
            fixture.manifest,
            document["scope"],
            {
                "case": record["case"],
                "device": record["device"],
                "precision": record["precision"],
            },
            list(late),
        )
        self.assertEqual(
            comparisons["step/20/field/Ex"]["mode"],
            completion.DIFFERENTIAL_NORMALIZED_MODE,
        )

    def test_single_gpu_3d_early_tolerance_uses_only_active_updaters(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][1]
        expected_record = {
            "case": record["case"],
            "device": record["device"],
            "precision": record["precision"],
        }
        active_models = completion._differential_active_model_names(
            record["case"], "test"
        )
        self.assertNotIn("dm2", active_models)
        self.assertEqual(
            completion._differential_model_tolerance(
                fixture.manifest, active_models, "float64"
            ),
            {"rtol": 5e-12, "atol": 5e-13},
        )
        self.assertEqual(
            completion._expected_differential_comparison(
                fixture.manifest, document["scope"], expected_record
            )["absolute_scale_floor"],
            completion.DIFFERENTIAL_NORMALIZED_ABSOLUTE_SCALE_FLOOR,
        )

        reference = self.differential_group_arrays(fixture, record, "reference", 0)
        candidate = self.differential_group_arrays(fixture, record, "candidate", 0)
        target = "step/0/field/Ex"
        comparisons = completion._expected_differential_array_comparisons(
            fixture.manifest,
            document["scope"],
            expected_record,
            list(reference),
        )
        self.assertEqual(
            comparisons[target],
            {
                "mode": completion.DIFFERENTIAL_ELEMENTWISE_MODE,
                "rtol": 5e-12,
                "atol": 5e-13,
            },
        )
        candidate[target].flat[0] += 1e-11
        candidate.update(
            completion._recompute_differential_physical_arrays(candidate, 0)
        )
        _normalized_metrics, normalized_passed = (
            completion._recompute_differential_metrics(
                reference,
                candidate,
                record["comparison"],
                "suite normalized comparison",
            )
        )
        self.assertTrue(normalized_passed)
        metrics, active_passed = completion._recompute_differential_metrics(
            reference,
            candidate,
            record["comparison"],
            "active updater comparison",
            array_comparisons=comparisons,
        )
        self.assertFalse(active_passed)
        record["metrics"] = metrics
        fixture.rewrite_group(document, 1, "candidate", 0, candidate)
        with self.assertRaisesRegex(completion.EvidenceError, "differential failed"):
            fixture.validate_completion_mirror(document)

    def test_differential_physical_arrays_are_recomputed_from_each_role_fields(self):
        for suffix in ("physical/spectrum/Ex", "physical/summary"):
            with self.subTest(suffix=suffix):
                fixture = self.differential_fixture()
                document = fixture.document()
                record = document["cases"][1]
                step = record["projection_groups"][0][0]
                name = f"step/{step}/{suffix}"
                for role in ("reference", "candidate"):
                    arrays = self.differential_group_arrays(fixture, record, role, 0)
                    arrays[name].flat[0] += 1.0
                    fixture.rewrite_group(document, 1, role, 0, arrays)
                with self.assertRaisesRegex(
                    completion.EvidenceError, "differs from the projected fields"
                ):
                    fixture.validate_completion_mirror(document)

    def test_differential_zero_and_integer_arrays_remain_exact(self):
        comparisons = (
            {
                "mode": completion.DIFFERENTIAL_ELEMENTWISE_MODE,
                "rtol": 1.0,
                "atol": 1.0,
            },
            {
                "mode": completion.DIFFERENTIAL_NORMALIZED_MODE,
                "linf_limit": 1.0,
                "l2_limit": 1.0,
                "absolute_scale_floor": 1.0,
                "all_zero_reference": "exact",
            },
        )
        mutations = (
            (
                np.zeros(2, dtype=np.float64),
                np.asarray([0.0, 1.0e-12], dtype=np.float64),
            ),
            (
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([1, 3], dtype=np.int64),
            ),
        )
        for comparison in comparisons:
            for reference, candidate in mutations:
                with self.subTest(mode=comparison["mode"], dtype=reference.dtype.str):
                    _metrics, passed = completion._recompute_differential_metrics(
                        {"array": reference},
                        {"array": candidate},
                        comparison,
                        "exact array",
                    )
                    self.assertFalse(passed)

    def test_differential_persistent_indices_are_bound_across_groups(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][1]
        group = record["projection_groups"][1]
        rewritten = {
            role: self.differential_group_arrays(fixture, record, role, 1)
            for role in ("reference", "candidate")
        }
        first = next(
            name
            for name, value in rewritten["reference"].items()
            if name.startswith(f"step/{group[0]}/state/Ex/")
            and name.endswith("/indices")
            and value.shape[0] >= 2
        )
        suffix = first.removeprefix(f"step/{group[0]}/")
        for arrays in rewritten.values():
            for step in group:
                indices = arrays[f"step/{step}/{suffix}"]
                indices[[0, 1]] = indices[[1, 0]]
        for role, arrays in rewritten.items():
            fixture.rewrite_group(document, 1, role, 1, arrays)
        with self.assertRaisesRegex(
            completion.EvidenceError, "persistent indices change across capture groups"
        ):
            fixture.validate_completion_mirror(document)

    def test_differential_persistent_geometry_is_bound_to_updater_coordinates(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][1]
        for ordinal, group in enumerate(record["projection_groups"]):
            for role in ("reference", "candidate"):
                arrays = self.differential_group_arrays(fixture, record, role, ordinal)
                for step in group:
                    cpml = f"step/{step}/state/Ex/0-Cpml/indices"
                    dielectric = f"step/{step}/state/Ex/3-Dielectric/indices"
                    cpml_coordinate = arrays[cpml][0].copy()
                    arrays[cpml][0] = arrays[dielectric][0]
                    arrays[dielectric][0] = cpml_coordinate
                fixture.rewrite_group(document, 1, role, ordinal, arrays)
        with self.assertRaisesRegex(
            completion.EvidenceError, "persistent geometry differs from the frozen case"
        ):
            fixture.validate_completion_mirror(document)

    def test_differential_common_persistent_omission_is_rejected(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][1]
        first_step = record["projection_steps"][0]
        omitted_suffix = record["persistent_arrays"][0].removeprefix(
            f"step/{first_step}/"
        )
        omitted = {
            f"step/{step}/{omitted_suffix}" for step in record["projection_steps"]
        }
        record["persistent_arrays"] = [
            name for name in record["persistent_arrays"] if name not in omitted
        ]
        for role, rewrite in (
            ("reference", fixture.rewrite_reference),
            ("candidate", fixture.rewrite_candidate),
        ):
            arrays = {
                name: value
                for name, value in self.differential_record_arrays(
                    fixture, record, role
                ).items()
                if name not in omitted
            }
            rewrite(document, 1, arrays)
        with self.assertRaisesRegex(
            completion.EvidenceError, "persistent array closure differs"
        ):
            fixture.validate_completion_mirror(document)

    def test_differential_empty_persistent_partition_is_rejected(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][0]
        index_name = next(
            name for name in record["persistent_arrays"] if name.endswith("/indices")
        )
        values_name = index_name.removesuffix("/indices") + "/values"
        rewritten = {}
        for role in ("reference", "candidate"):
            arrays = self.differential_record_arrays(fixture, record, role)
            arrays[index_name] = np.empty((0, 3), dtype=np.int64)
            arrays[values_name] = np.empty((0,), dtype=np.complex128)
            rewritten[role] = arrays
        fixture.rewrite_reference(document, 0, rewritten["reference"])
        fixture.rewrite_candidate(document, 0, rewritten["candidate"])
        with self.assertRaisesRegex(
            completion.EvidenceError, "persistent index shape or dtype differs"
        ):
            fixture.validate_completion_mirror(document)

    def test_differential_state_values_use_model_scoped_tolerance(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][0]
        reference_arrays = self.differential_record_arrays(fixture, record, "reference")
        candidate_arrays = self.differential_record_arrays(fixture, record, "candidate")
        name = next(
            projected
            for projected in record["persistent_arrays"]
            if projected.endswith("/0-Cpml/values") and candidate_arrays[projected].size
        )
        candidate_arrays[name][0] += 1.0e-4
        metrics, suite_passed = completion._recompute_differential_metrics(
            reference_arrays,
            candidate_arrays,
            record["comparison"],
            "legacy suite-wide comparison",
        )
        self.assertTrue(suite_passed)
        record["metrics"] = metrics
        fixture.rewrite_candidate(document, 0, candidate_arrays)
        with self.assertRaisesRegex(completion.EvidenceError, "differential failed"):
            fixture.validate_completion_mirror(document)

    def test_differential_source_proof_is_required_and_semantically_checked(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][0]
        omitted = completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY
        record["contract_arrays"].remove(omitted)
        for role, rewrite in (
            ("reference", fixture.rewrite_reference),
            ("candidate", fixture.rewrite_candidate),
        ):
            arrays = {
                name: value
                for name, value in self.differential_record_arrays(
                    fixture, record, role
                ).items()
                if name != omitted
            }
            rewrite(document, 0, arrays)
        with self.assertRaisesRegex(
            completion.EvidenceError, "source contract array closure differs"
        ):
            fixture.validate_completion_mirror(document)

        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][0]
        reference_arrays = self.differential_record_arrays(fixture, record, "reference")
        candidate_arrays = self.differential_record_arrays(fixture, record, "candidate")
        proof = json.loads(
            reference_arrays[completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY]
            .tobytes()
            .decode()
        )
        target_record = proof["captures"][0]["candidate"]["live"]["overwrite_targets"]
        target_dtype = np.dtype(target_record["dtype"])
        target = np.frombuffer(
            bytes.fromhex(target_record["data_hex"]), dtype=target_dtype
        ).copy()
        target[0] += 1
        target_record["data_hex"] = target.tobytes(order="C").hex()
        proof["candidate_preimage_sha256"] = (
            completion._differential_source_role_preimage_sha256(
                completion._manifest_case(self.manifest, record["case"]),
                "candidate",
                proof["captures"],
            )
        )
        proof_array = np.frombuffer(
            json.dumps(
                proof,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            dtype=np.uint8,
        ).copy()
        reference_arrays[completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY] = proof_array
        candidate_arrays[completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY] = (
            proof_array.copy()
        )
        fixture.rewrite_reference(document, 0, reference_arrays)
        fixture.rewrite_candidate(document, 0, candidate_arrays)
        with self.assertRaisesRegex(
            completion.EvidenceError, "candidate overwrite semantics differ"
        ):
            fixture.validate_completion_mirror(document)

    def test_differential_source_proof_rejects_reused_preimage_digest(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][0]
        rewritten = {}
        for role in ("reference", "candidate"):
            rewritten[role] = self.differential_record_arrays(fixture, record, role)
        proof = json.loads(
            rewritten["reference"][completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY]
            .tobytes()
            .decode()
        )
        proof["candidate_preimage_sha256"] = proof["reference_preimage_sha256"]
        proof_array = np.frombuffer(
            json.dumps(
                proof,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            dtype=np.uint8,
        ).copy()
        for arrays in rewritten.values():
            arrays[completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY] = proof_array.copy()
        fixture.rewrite_reference(document, 0, rewritten["reference"])
        fixture.rewrite_candidate(document, 0, rewritten["candidate"])
        with self.assertRaisesRegex(
            completion.EvidenceError, "raw proof identity differs"
        ):
            fixture.validate_completion_mirror(document)

    def test_differential_source_proof_rejects_arbitrary_distinct_digests(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][0]
        rewritten = {}
        for role in ("reference", "candidate"):
            rewritten[role] = self.differential_record_arrays(fixture, record, role)
        proof = json.loads(
            rewritten["reference"][completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY]
            .tobytes()
            .decode()
        )
        proof["reference_preimage_sha256"] = "3" * 64
        proof["candidate_preimage_sha256"] = "4" * 64
        proof_array = np.frombuffer(
            json.dumps(
                proof,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            dtype=np.uint8,
        ).copy()
        for arrays in rewritten.values():
            arrays[completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY] = proof_array.copy()
        fixture.rewrite_reference(document, 0, rewritten["reference"])
        fixture.rewrite_candidate(document, 0, rewritten["candidate"])
        with self.assertRaisesRegex(
            completion.EvidenceError, "raw preimage digest differs"
        ):
            fixture.validate_completion_mirror(document)

    def test_differential_source_proof_rejects_rehashed_subtolerance_clock_drift(
        self,
    ):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][0]
        rewritten = {}
        for role in ("reference", "candidate"):
            rewritten[role] = self.differential_record_arrays(fixture, record, role)
        proof = json.loads(
            rewritten["reference"][completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY]
            .tobytes()
            .decode()
        )
        capture = next(item for item in proof["captures"] if item["step"] == 20)
        clock = completion._differential_source_array_from_record(
            capture["native"]["time"], "native clock"
        )
        clock[1] = np.nextafter(clock[1], np.inf)
        clock_record = {
            "dtype": clock.dtype.str,
            "shape": list(clock.shape),
            "data_hex": clock.tobytes(order="C").hex(),
        }
        capture["native"]["time"] = copy.deepcopy(clock_record)
        capture["candidate"]["time"] = copy.deepcopy(clock_record)
        workload = completion._manifest_case(self.manifest, record["case"])
        proof["reference_preimage_sha256"] = (
            completion._differential_source_role_preimage_sha256(
                workload, "reference", proof["captures"]
            )
        )
        proof["candidate_preimage_sha256"] = (
            completion._differential_source_role_preimage_sha256(
                workload, "candidate", proof["captures"]
            )
        )
        proof_array = np.frombuffer(
            json.dumps(
                proof,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            dtype=np.uint8,
        ).copy()
        for arrays in rewritten.values():
            arrays[completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY] = proof_array.copy()
        fixture.rewrite_reference(document, 0, rewritten["reference"])
        fixture.rewrite_candidate(document, 0, rewritten["candidate"])
        with self.assertRaisesRegex(
            completion.EvidenceError, "differs from the relative capture clock"
        ):
            fixture.validate_completion_mirror(document)

    def test_differential_source_proof_rejects_rehashed_native_index_widening(self):
        fixture = self.differential_fixture()
        document = fixture.document()
        record = document["cases"][0]
        rewritten = {
            role: self.differential_record_arrays(fixture, record, role)
            for role in ("reference", "candidate")
        }
        proof = json.loads(
            rewritten["reference"][completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY]
            .tobytes()
            .decode()
        )
        native_indices = proof["captures"][0]["native"]["indices"]
        indices = completion._differential_source_array_from_record(
            native_indices, "native indices"
        ).astype(np.int64)
        native_indices.update(
            {
                "dtype": indices.dtype.str,
                "shape": list(indices.shape),
                "data_hex": indices.tobytes(order="C").hex(),
            }
        )
        workload = completion._manifest_case(self.manifest, record["case"])
        proof["reference_preimage_sha256"] = (
            completion._differential_source_role_preimage_sha256(
                workload, "reference", proof["captures"]
            )
        )
        proof_array = np.frombuffer(
            json.dumps(
                proof,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            dtype=np.uint8,
        ).copy()
        for arrays in rewritten.values():
            arrays[completion.DIFFERENTIAL_SOURCE_PROOF_ARRAY] = proof_array.copy()
        fixture.rewrite_reference(document, 0, rewritten["reference"])
        fixture.rewrite_candidate(document, 0, rewritten["candidate"])
        with self.assertRaisesRegex(
            completion.EvidenceError, "native semantics differ from the workload"
        ):
            fixture.validate_completion_mirror(document)

    def policy_run(self, policy="dense", *, trace_policy=None, extra_trace_op=None):
        trace_policy = policy if trace_policy is None else trace_policy
        case_name = completion.POLICY_CASES[0]
        workload = next(
            case for case in self.manifest["benchmarks"] if case["name"] == case_name
        )
        representation = completion.FORCED_REPRESENTATIONS[policy]
        top_representation = f"policy-dispatched-bucket-io-v2[{representation}]"
        executions = [
            {
                "component": "Ex",
                "model": "dcp-plrc",
                "targets": 12,
                "policy": policy,
                "execution_representation": representation,
            }
        ]
        runtime_preimage = [None] * 31
        runtime_preimage[0] = completion.TORCH_SOLVER_ABI
        runtime_preimage[3] = "torch.float32"
        runtime_preimage[4] = "compile"
        runtime_preimage[5] = "default"
        runtime_preimage[6] = completion.LOCAL_COMPILED_REGION_TOPOLOGY
        runtime_preimage[8] = top_representation
        runtime_preimage[18] = False
        runtime_preimage[19] = True
        runtime_preimage[20] = None
        runtime_preimage[21] = [
            False,
            100,
            10,
            3,
            False,
            False,
            False,
            completion.CUDA_GRAPH_EXECUTION_REPRESENTATION,
        ]
        result = {
            "workload": copy.deepcopy(workload),
            "runtime": {
                "device": "cuda:0",
                "precision": "float32",
                "field_storage_dtype": "torch.float32",
                "compile_policy": "compile",
                "compile_mode": "default",
                "explicit_cuda_graphs": False,
                "execution_policy": policy,
                "paired_real": False,
                "compile_cache_key": hashlib.sha256(
                    repr(
                        completion._tuple_cache_preimage(
                            runtime_preimage, "test cache preimage"
                        )
                    ).encode()
                ).hexdigest(),
            },
            "diagnostics": {
                "compile_solver_abi": completion.TORCH_SOLVER_ABI,
                "compiled_region_topology": completion.LOCAL_COMPILED_REGION_TOPOLOGY,
                "cuda_graph_execution_representation": (
                    completion.CUDA_GRAPH_EXECUTION_REPRESENTATION
                ),
                "boundaries": {
                    "scheduling": "external",
                    "execution_representation": (
                        completion.BOUNDARY_SYNC_REPRESENTATION
                    ),
                    "paired_real_scratch_bytes": 0,
                },
                "dispersive": {
                    "execution_representation": top_representation,
                    "policy_executions": executions,
                },
            },
        }
        config = completion._policy_config_preimage(
            result,
            case_name,
            policy,
            executions,
        )
        result["compile_cache_key_evidence"] = {
            "schema_version": 1,
            "algorithm": completion.COMPILE_CACHE_PREIMAGE_ALGORITHM,
            "runtime_preimage": runtime_preimage,
            "policy_config": config,
            "policy_config_sha256": completion._canonical_sha256(config),
        }
        profile_steps = 2
        trace_events = [
            {
                "name": completion.POLICY_WRITE_OPERATIONS[trace_policy],
                "cat": "cpu_op",
                "ph": "X",
                "ts": index,
                "dur": 1,
            }
            for index in range(profile_steps)
        ]
        if extra_trace_op is not None:
            trace_events.append(
                {
                    "name": completion.POLICY_WRITE_OPERATIONS[extra_trace_op],
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": profile_steps,
                    "dur": 1,
                }
            )
        suffix = extra_trace_op or "exact"
        descriptor = self.write_json(
            f"policy/{policy}-{trace_policy}-{suffix}.json",
            {"traceEvents": trace_events},
        )
        expected_operation = completion.POLICY_WRITE_OPERATIONS[policy]
        expected_counts = {
            operation: profile_steps if operation == expected_operation else 0
            for operation in completion.POLICY_WRITE_OPERATIONS.values()
        }
        result["policy_execution_diagnostic"] = {
            "schema_version": 1,
            "kind": completion.POLICY_DIAGNOSTIC_KIND,
            "contract_id": completion.POLICY_DIAGNOSTIC_CONTRACT,
            "execution_policy": policy,
            "compile_policy": "eager",
            "profile_steps": profile_steps,
            "execution_records_per_step": 1,
            "expected_operation": expected_operation,
            "expected_operation_count": profile_steps,
            "observed_operation_counts": expected_counts,
            "trace": descriptor,
        }
        return result

    def validate_policy_run(self, result, used_digests=None):
        policy = result["runtime"]["execution_policy"]
        with mock.patch.object(
            completion,
            "_validate_cuda_tuning_case",
            return_value={"seconds_per_step": 1.0},
        ):
            return completion._validate_policy_run(
                result,
                f"policy {policy}",
                policy=policy,
                case_name=completion.POLICY_CASES[0],
                traces={},
                manifest=self.manifest,
                reader=completion.ArtifactReader(self.directory, self.candidate),
                diagnostic_trace_digests=(
                    set() if used_digests is None else used_digests
                ),
            )

    def test_policy_raw_ops_and_cache_preimages_fail_closed(self):
        valid = self.policy_run()
        self.validate_policy_run(valid)

        metadata_swap = self.policy_run("compact", trace_policy="dense")
        with self.assertRaisesRegex(completion.EvidenceError, "raw trace"):
            self.validate_policy_run(metadata_swap)

        op_tamper = self.policy_run("dense", extra_trace_op="tiled")
        with self.assertRaisesRegex(completion.EvidenceError, "raw trace"):
            self.validate_policy_run(op_tamper)

        used_digests = set()
        reused = self.policy_run("dense")
        self.validate_policy_run(reused, used_digests)
        with self.assertRaisesRegex(completion.EvidenceError, "reuses"):
            self.validate_policy_run(reused, used_digests)

        preimage_tamper = self.policy_run("dense")
        preimage_tamper["compile_cache_key_evidence"]["runtime_preimage"][
            5
        ] = "max-autotune"
        with self.assertRaisesRegex(completion.EvidenceError, "cache key"):
            self.validate_policy_run(preimage_tamper)

        rehashed_preimage_tamper = self.policy_run("dense")
        runtime_preimage = rehashed_preimage_tamper["compile_cache_key_evidence"][
            "runtime_preimage"
        ]
        runtime_preimage[6] = "local-eager-stencil-and-material-phases"
        rehashed_preimage_tamper["runtime"]["compile_cache_key"] = hashlib.sha256(
            repr(
                completion._tuple_cache_preimage(
                    runtime_preimage, "test cache preimage"
                )
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(completion.EvidenceError, "configuration"):
            self.validate_policy_run(rehashed_preimage_tamper)

        abi_tamper = self.policy_run("dense")
        runtime_preimage = abi_tamper["compile_cache_key_evidence"]["runtime_preimage"]
        runtime_preimage[0] = "torch-fdtd-regions-v8"
        abi_tamper["diagnostics"]["compile_solver_abi"] = "torch-fdtd-regions-v8"
        abi_tamper["runtime"]["compile_cache_key"] = hashlib.sha256(
            repr(
                completion._tuple_cache_preimage(
                    runtime_preimage, "test cache preimage"
                )
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(completion.EvidenceError, "configuration"):
            self.validate_policy_run(abi_tamper)

        graph_preimage_tamper = self.policy_run("dense")
        runtime_preimage = graph_preimage_tamper["compile_cache_key_evidence"][
            "runtime_preimage"
        ]
        runtime_preimage[21][-1] = "external-standard-regions+tampered"
        graph_preimage_tamper["runtime"]["compile_cache_key"] = hashlib.sha256(
            repr(
                completion._tuple_cache_preimage(
                    runtime_preimage, "test cache preimage"
                )
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(completion.EvidenceError, "CUDA graph"):
            self.validate_policy_run(graph_preimage_tamper)

        graph_diagnostic_tamper = self.policy_run("dense")
        graph_diagnostic_tamper["diagnostics"][
            "cuda_graph_execution_representation"
        ] = "external-standard-regions+tampered"
        with self.assertRaisesRegex(completion.EvidenceError, "configuration"):
            self.validate_policy_run(graph_diagnostic_tamper)

    def test_failure_reason_contract_matches_producer(self):
        self.assertEqual(
            completion.FAILURE_REASON_CONTRACTS,
            failure_evidence.FAILURE_REASON_CONTRACTS,
        )
        mode = "checkpoint-mismatch"
        record = {
            "mode": mode,
            "passed": True,
            "rank0_error": (
                "TorchDistributedError: distributed checkpoint metadata "
                "does not match every rank"
            ),
        }
        stdout = (json.dumps(record, sort_keys=True) + "\n").encode()
        self.assertTrue(failure_evidence._observed_failure(mode, 0, stdout, b""))
        record["rank0_error"] = "different failure"
        tampered = (json.dumps(record, sort_keys=True) + "\n").encode()
        self.assertFalse(failure_evidence._observed_failure(mode, 0, tampered, b""))

    def test_failure_wrapper_binds_host_and_raw_reason(self):
        mode = "dtype-mismatch"
        error = (
            "TorchConfigurationError: both ranks must use the same "
            "floating-point precision"
        )
        stdout = (
            json.dumps(
                {"mode": mode, "passed": True, "rank0_error": error},
                sort_keys=True,
            )
            + "\n"
        ).encode()
        stdout_descriptor = self.write_bytes(
            f"{mode}.stdout",
            stdout,
            media_type=completion.MEDIA_TYPE_TEXT,
        )
        stderr_descriptor = self.write_bytes(
            f"{mode}.stderr",
            b"",
            media_type=completion.MEDIA_TYPE_TEXT,
        )
        document = {
            "schema_version": 1,
            "kind": completion.FAILURE_RUN_KIND,
            "mode": mode,
            "candidate_evidence": self.candidate,
            "host_contract": copy.deepcopy(self.host_contract),
            "command": [
                "uv",
                "run",
                "--no-sync",
                "torchrun",
                "--standalone",
                "--nproc-per-node=2",
                "--module",
                "benchmarks.torch_two_gpu_failures",
                mode,
            ],
            "exit_code": 0,
            "stdout": stdout_descriptor,
            "stderr": stderr_descriptor,
            "expected_failure": copy.deepcopy(
                completion.FAILURE_REASON_CONTRACTS[mode]
            ),
            "passed": True,
        }
        reader = completion.ArtifactReader(self.directory, self.candidate)
        artifact = completion.LoadedArtifact({}, Path("failure.json"), b"", document)
        self.assertEqual(
            completion._validate_failure_run(artifact, reader, self.candidate),
            (mode, self.host_contract),
        )

        document["expected_failure"]["reason_id"] = "tampered"
        with self.assertRaisesRegex(
            completion.EvidenceError, "reason contract differs"
        ):
            completion._validate_failure_run(artifact, reader, self.candidate)

    def two_gpu_document(self, distributed_seconds):
        repeats = self.manifest["reference"]["performance_repetitions"]
        steps = self.manifest["reference"]["performance_steps_per_repeat"]
        serial_seconds = 1.0
        size = [128, 96, 96]
        serial_cells = distributed_cells = 128 * 96 * 96
        ratio = serial_seconds / distributed_seconds
        environment = {
            "host_contract": copy.deepcopy(self.cuda_host_contract),
            "platform": self.common_host["platform"],
            "python": self.common_host["python"],
            "torch": self.cuda_runtime["torch"],
            "cuda_runtime": self.cuda_runtime["cuda_runtime"],
            "nccl": [2, 0],
            "devices": [
                {"index": 0, "name": "GPU", "memory_bytes": 10},
                {"index": 1, "name": "GPU", "memory_bytes": 10},
            ],
            "topology": "GPU0 GPU1",
            "topology_command_status": 0,
        }
        return {
            "schema_version": 1,
            "candidate_evidence": self.candidate,
            "case": "strong-mixed",
            "gate": "strong",
            "sizes": {
                "serial": size,
                "distributed": size,
                "serial_cells": serial_cells,
                "distributed_cells": distributed_cells,
            },
            "measurement": {
                "warmup": self.manifest["reference"]["performance_warmup_steps"],
                "steps": steps,
                "repeats": repeats,
                "profile_steps": 10,
                "threads_per_rank": 1,
            },
            "serial": {
                "raw_seconds": [serial_seconds] * repeats,
                "median_seconds": serial_seconds,
                "steps_per_second": steps / serial_seconds,
                "cells_per_second": serial_cells * steps / serial_seconds,
                "construction_seconds": 0.1,
                "capture_seconds": 0.1,
                "peak_allocated_bytes": 1024,
            },
            "distributed": {
                "raw_seconds": [distributed_seconds] * repeats,
                "median_seconds": distributed_seconds,
                "steps_per_second": steps / distributed_seconds,
                "cells_per_second": distributed_cells * steps / distributed_seconds,
                "rank_raw_seconds": [
                    [distributed_seconds, distributed_seconds] for _ in range(repeats)
                ],
                "storage_addresses_stable": True,
                "construction_seconds": 0.1,
                "capture_seconds": 0.1,
                "peak_allocated_bytes_rank0": 1024,
                "halo_bytes_rank0": 128,
            },
            "decomposition": {},
            "profiles": [{}, {}],
            "environment": environment,
            "acceptance": {
                "ratio": ratio,
                "threshold": 1.6,
                "passed": True,
            },
        }

    def two_gpu_v2_artifact(self):
        from benchmarks import torch_two_gpu
        from tests.test_torch_gpu_closure import TwoGpuEvidenceContractTest

        helper = TwoGpuEvidenceContractTest(
            "test_combiner_requires_complete_both_rank_and_raw_profile_evidence"
        )
        helper.setUp()
        reference = self.manifest["reference"]
        helper.args.warmup = reference["performance_warmup_steps"]
        helper.args.steps = reference["performance_steps_per_repeat"]
        helper.args.repeats = reference["performance_repetitions"]
        helper.args.profile_steps = reference["performance_profile_steps"]
        helper.candidate = self.candidate
        serial, distributed, subprocesses = helper.workers()
        result = torch_two_gpu._combine_worker_results(
            serial,
            distributed,
            subprocesses,
            helper.args,
        )
        self.assertTrue(result["acceptance"]["passed"])

        for rank in range(2):
            events = [
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "ncclKernel",
                    "ts": index * 2.5,
                    "dur": 2.5,
                }
                for index in range(4)
            ]
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "compute-main",
                    "ts": 0.0,
                    "dur": 5.0,
                }
            )
            events.extend(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": f"compute-{index}",
                    "ts": 20.0 + index,
                    "dur": 0.1,
                }
                for index in range(15)
            )
            for name in completion.HALO_ANNOTATIONS:
                events.extend(
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": name,
                        "ts": 40.0 + index,
                        "dur": 1.0 / helper.args.profile_steps,
                    }
                    for index in range(helper.args.profile_steps)
                )
            descriptor = self.write_json(
                f"case/gmes-two-gpu-{rank}.json",
                {"traceEvents": events},
            )
            profile = result["profiles"][rank]
            profile["trace"] = descriptor["path"]
            profile["trace_sha256"] = descriptor["sha256"]
            profile["trace_size_bytes"] = descriptor["size_bytes"]

        result["subprocesses"]["serial"]["command"] = [
            "python",
            "-m",
            "benchmarks.torch_two_gpu",
            "--worker",
            "serial",
        ]
        for role, child in (("serial", serial), ("distributed", distributed)):
            record = result["subprocesses"][role]
            artifact_descriptor = self.write_json(
                f"case/{role}-child.artifact.json", child
            )
            stdout_descriptor = self.write_json(f"case/{role}-child.stdout.json", child)
            stderr_descriptor = self.write_bytes(
                f"case/{role}-child.stderr.txt",
                b"",
                media_type=completion.MEDIA_TYPE_TEXT,
            )
            for name, descriptor in (
                ("artifact", artifact_descriptor),
                ("stdout", stdout_descriptor),
                ("stderr", stderr_descriptor),
            ):
                record[name] = descriptor
                record[f"{name}_sha256"] = descriptor["sha256"]
                record[f"{name}_size_bytes"] = descriptor["size_bytes"]
        artifact = completion.LoadedArtifact(
            {}, self.directory / "two-gpu-v2.json", b"", result
        )
        return artifact, completion.ArtifactReader(self.directory, self.candidate)

    def test_two_gpu_embedded_pass_cannot_hide_raw_scaling_failure(self):
        artifact, reader = self.two_gpu_v2_artifact()
        artifact.document["acceptance"]["ratio"] = 999.0
        with self.assertRaisesRegex(
            completion.EvidenceError, "differs from raw evidence"
        ):
            completion._validate_two_gpu_performance(
                artifact,
                reader,
                self.manifest,
                self.candidate,
            )

    def test_two_gpu_fixed_workload_size_cannot_be_tampered(self):
        artifact, reader = self.two_gpu_v2_artifact()
        artifact.document["sizes"]["serial"] = [2, 2, 2]
        with self.assertRaisesRegex(completion.EvidenceError, "fixed size"):
            completion._validate_two_gpu_performance(
                artifact,
                reader,
                self.manifest,
                self.candidate,
            )

    def test_macos_job_requires_success_on_the_candidate(self):
        job = {
            "name": completion.REQUIRED_JOBS[0],
            "workflow": "CI",
            "run_id": 1,
            "run_attempt": 1,
            "job_id": 2,
            "event": "pull_request",
            "head_sha": self.candidate["candidate_git_commit"],
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://github.com/ruddyscent/gmes/actions/runs/1",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z",
        }
        self.assertEqual(
            completion._validate_job(job, self.candidate, "job"),
            completion.REQUIRED_JOBS[0],
        )
        job["head_sha"] = "b" * 40
        with self.assertRaises(completion.EvidenceError):
            completion._validate_job(job, self.candidate, "job")

    def test_codeql_analysis_binds_the_pr_synthetic_merge(self):
        job = {
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:02:00Z",
        }
        pull_request = {
            "number": 17,
            "head_sha": self.candidate["candidate_git_commit"],
            "merge_sha": "b" * 40,
            "merge_ref": "refs/pull/17/merge",
        }
        analysis = {
            "language": "python",
            "analysis_key": ".github/workflows/codeql.yml:analyze",
            "category": "/language:python",
            "commit_sha": pull_request["merge_sha"],
            "ref": pull_request["merge_ref"],
            "environment": {"language": "python", "build-mode": "none"},
            "created_at": "2026-01-01T00:01:00Z",
            "results_count": 0,
            "rules_count": 1,
            "error": "",
            "warning": "",
            "url": (
                "https://api.github.com/repos/ruddyscent/gmes/"
                "code-scanning/analyses/1"
            ),
            "sarif_id": "sarif-1",
            "tool": {"name": "CodeQL", "version": "2.23.0"},
        }
        self.assertEqual(
            completion._validate_code_scanning_analysis(
                analysis,
                self.candidate,
                pull_request,
                job,
                "analysis",
            ),
            "python",
        )

        analysis["commit_sha"] = self.candidate["candidate_git_commit"]
        with self.assertRaisesRegex(completion.EvidenceError, "cross-run"):
            completion._validate_code_scanning_analysis(
                analysis,
                self.candidate,
                pull_request,
                job,
                "analysis",
            )


if __name__ == "__main__":
    unittest.main()
