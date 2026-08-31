from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmarks import issue123_completion as completion
from benchmarks import two_gpu_failure_evidence as failure_evidence


class Issue123CompletionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        manifest_raw = completion.DEFAULT_MANIFEST.read_bytes()
        self.manifest = json.loads(manifest_raw)
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
        self.common_host = {
            "hostname": "issue-123-linux",
            "platform": "Linux-6.8.0-x86_64-with-glibc2.39",
            "os": {
                "system": "Linux",
                "release": "6.8.0",
                "machine": "x86_64",
            },
            "python": "3.14.0",
            "cxx_version": "c++ 14.2.0",
            "swig_version": "SWIG Version 4.3.1",
            "uv_lock_sha256": "f" * 64,
        }
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

    def patched_validators(self, *, different_gpu_environment=False):
        gpu_environment = {
            "host_contract": copy.deepcopy(self.cuda_host_contract),
            "common_host_identity": copy.deepcopy(self.common_host),
            "platform": self.common_host["platform"],
            "python": self.common_host["python"],
            "torch": self.cuda_runtime["torch"],
            "cuda_runtime": self.cuda_runtime["cuda_runtime"],
            "devices": [
                {"index": 0, "name": "GPU", "memory_bytes": 1},
                {"index": 1, "name": "GPU", "memory_bytes": 1},
            ],
            "topology": "GPU0 GPU1",
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
        return {
            "_validate_cpu_scope": {
                "candidate_evidence": self.candidate,
                "torch_raw_seconds_per_step": cpu_raw,
                "native_raw_seconds_per_step": native_raw,
                "host_contract": copy.deepcopy(self.host_contract),
                "common_host_identity": copy.deepcopy(self.common_host),
                "runtime_identity": copy.deepcopy(self.cpu_runtime),
            },
            "_validate_policy_scope": {
                "candidate_evidence": self.candidate,
                "environment": copy.deepcopy(gpu_environment),
            },
            "_validate_single_gpu_scope": {
                "candidate_evidence": self.candidate,
                "environment": copy.deepcopy(gpu_environment),
                "cuda_raw_seconds_per_step": cuda_raw,
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
                "macos_job": {
                    "run_id": 1,
                    "started_at": "2026-08-31T00:00:00Z",
                    "completed_at": "2026-08-31T00:03:00Z",
                },
            },
        }

    def evaluate_with_patches(self, values):
        index = self.write_top_index()
        with ExitStack() as stack:
            for name, value in values.items():
                stack.enter_context(
                    mock.patch.object(completion, name, return_value=value)
                )
            return completion.evaluate_completion(index)

    def test_only_all_satisfied_scopes_complete_the_issue(self):
        result = self.evaluate_with_patches(self.patched_validators())
        self.assertTrue(result["issue_completion_satisfied"])
        self.assertTrue(all(scope["satisfied"] for scope in result["scopes"].values()))

        values = self.patched_validators()
        index = self.write_top_index()
        with ExitStack() as stack:
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
                    "message": "missing job",
                }
            ],
        )

    def test_cross_gpu_environment_mismatch_fails_closed(self):
        result = self.evaluate_with_patches(
            self.patched_validators(different_gpu_environment=True)
        )
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertFalse(result["scopes"]["single_gpu"]["satisfied"])
        self.assertFalse(result["scopes"]["two_gpu"]["satisfied"])
        self.assertTrue(result["cross_scope_errors"])

    def test_missing_scope_fields_return_false_instead_of_raising(self):
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

        thread = {"OMP_NUM_THREADS": "1"}
        environment = {
            "hostname": "candidate-host",
            "platform": "Linux-baseline",
            "python": "3.14.0",
            "torch": "2.13.0+cpu",
            "cuda_runtime": None,
            "devices": [],
            "cpu_count": 8,
            "cpu_affinity": list(range(8)),
            "cpu_count_physical_affinity": 4,
            "cpu_topology": "0,0\n1,0\n2,0\n3,0",
            "cpu_model": "model\nCPU(s) scaling MHz: 72%",
            "gpu_topology": None,
            "thread_environment": thread,
        }
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
        self.assertEqual(environment["hostname"], "candidate-host")
        environment["torch"] = "2.13.0+cu130"
        environment["cuda_runtime"] = "13.0"
        environment["devices"] = [
            {"index": 0, "name": "synthetic device"},
            {"index": 1, "name": "synthetic device"},
        ]
        environment["gpu_topology"] = "synthetic topology"
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
        environment["cpu_model"] = "model\nCPU(s) scaling MHz: 72%"
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
                    repr(tuple(runtime_preimage)).encode()
                ).hexdigest(),
            },
            "diagnostics": {
                "compile_solver_abi": completion.TORCH_SOLVER_ABI,
                "compiled_region_topology": completion.LOCAL_COMPILED_REGION_TOPOLOGY,
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
            repr(tuple(runtime_preimage)).encode()
        ).hexdigest()
        with self.assertRaisesRegex(completion.EvidenceError, "configuration"):
            self.validate_policy_run(rehashed_preimage_tamper)

        abi_tamper = self.policy_run("dense")
        runtime_preimage = abi_tamper["compile_cache_key_evidence"]["runtime_preimage"]
        runtime_preimage[0] = "torch-fdtd-regions-v8"
        abi_tamper["diagnostics"]["compile_solver_abi"] = "torch-fdtd-regions-v8"
        abi_tamper["runtime"]["compile_cache_key"] = hashlib.sha256(
            repr(tuple(runtime_preimage)).encode()
        ).hexdigest()
        with self.assertRaisesRegex(completion.EvidenceError, "configuration"):
            self.validate_policy_run(abi_tamper)

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
