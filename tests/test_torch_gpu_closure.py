"""CPU-only contract tests for issue #123 GPU closure evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from benchmarks import (
    torch_tuning,
    torch_two_gpu,
    torch_two_gpu_correctness,
    two_gpu_failure_evidence,
)


def _host_contract():
    return {
        "schema_version": 2,
        "common_identity": {
            "hostname": "gpu-host",
            "platform": "linux-x86_64",
            "os": {
                "system": "Linux",
                "release": "6.14.0",
                "machine": "x86_64",
            },
            "python": "3.14.0",
            "cxx_version": "c++ 14",
            "swig_version": "SWIG 4.3",
            "uv_lock_sha256": "f" * 64,
        },
        "runtime_identity": {
            "torch": "2.10.0+cu130",
            "cuda_runtime": "13.0",
        },
    }


def _devices():
    return [
        {
            "index": index,
            "name": f"GPU {index}",
            "memory_bytes": 16 * 1024**3,
            "capability": [9, 0],
            "multiprocessors": 100,
        }
        for index in range(2)
    ]


def _two_gpu_environment():
    return {
        "host_contract": _host_contract(),
        "hostname": "gpu-host",
        "platform": "linux-x86_64",
        "python": "3.14.0",
        "torch": "2.10.0+cu130",
        "cuda_runtime": "13.0",
        "nccl": [2, 28, 0],
        "devices": _devices(),
        "topology": "GPU0 GPU1 NV2",
        "topology_command": ["nvidia-smi", "topo", "-m"],
        "topology_command_status": 0,
    }


class SingleGpuCudaSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = torch_tuning.load_manifest(torch_tuning.MANIFEST)

    def result(self, name):
        reference = self.manifest["reference"]
        repetitions = reference["performance_repetitions"]
        steps = reference["performance_steps_per_repeat"]
        profile_steps = reference["performance_profile_steps"]
        precision = torch_tuning.CUDA_PERFORMANCE_PRECISION_BY_CASE[name]
        return {
            "schema_version": 2,
            "backend": "torch",
            "workload": copy.deepcopy(torch_tuning.find_case(self.manifest, name)),
            "benchmark_contract": {
                "initializer": torch_tuning.FIELD_INITIALIZER,
                "seed": reference["seed"],
                "field_scale": reference["field_scale"],
                "warmup_steps": reference["performance_warmup_steps"],
                "steps_per_repeat": steps,
                "repetitions": repetitions,
                "profile_steps": profile_steps,
                "timer": "time.perf_counter",
                "sample_start": "independently-restored-pre-warmup-state",
            },
            "runtime": {
                "device": "cuda:0",
                "precision": precision,
                "compile_policy": "compile",
                "compile_mode": "default",
                "explicit_cuda_graphs": False,
                "execution_policy": "auto",
                "experimental_dispersive_grouping": False,
                "experimental_dispersive_grouping_scope": "combined",
                "threads": 1,
                "interop_threads": 1,
                "paired_real": False,
                "field_storage_representation": "real-v1",
                "field_storage_channels": 1,
                "field_storage_dtype": f"torch.{precision}",
                "compile_cache_key": "c" * 64,
                "cpu_topology_command_status": 0,
            },
            "measurements": {
                "advance": torch_tuning._timing_summary(
                    [1.0] * repetitions,
                    steps=steps,
                )
            },
            "memory": {
                "cuda_allocated_before_bytes": 1024,
                "cuda_allocated_after_bytes": 1024,
                "cuda_allocated_growth_bytes": 0,
                "cuda_peak_allocated_bytes": 4096,
                "cuda_peak_reserved_bytes": 8192,
                "bounded": True,
            },
            "profiler": {
                "profile_steps": profile_steps,
                "kernel_launches": 10,
                "host_to_device_events": 0,
                "device_to_host_events": 0,
                "field_buffer_sizes_bytes": {
                    f"state.{component}": (40 if precision == "float32" else 80)
                    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
                },
            },
            "diagnostics": {
                "boundaries": {
                    "scheduling": "external",
                    "execution_representation": (
                        torch_tuning.gmes.torch_fdtd.BOUNDARY_SYNC_REPRESENTATION
                    ),
                    "paired_real_scratch_bytes": 0,
                }
            },
            "state_progress": {
                "changed_buffers": ["ex", "ey", "ez", "hx", "hy", "hz"],
                "fields_changed": ["ex", "ey", "ez", "hx", "hy", "hz"],
                "all_fields_changed": True,
            },
            "state_finiteness": {
                "contract_id": torch_tuning.STATE_FINITENESS_CONTRACT_ID,
                "tracked_buffers": ["ex", "ey", "ez", "hx", "hy", "hz"],
                "stages": {
                    stage: {
                        "floating_or_complex_buffer_count": 6,
                        "floating_or_complex_element_count": 60,
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
            "acceptance": {name: True for name in torch_tuning.RUNTIME_ACCEPTANCE_KEYS},
        }

    @staticmethod
    def environment():
        return {
            "host_contract": _host_contract(),
            "platform": "linux-x86_64",
            "python": "3.14.0",
            "torch": "2.10.0+cu130",
            "cuda_runtime": "13.0",
            "devices": _devices(),
            "gpu_topology": "GPU0 GPU1 NV2",
            "gpu_topology_command_status": 0,
            "cpu_model_command_status": 0,
        }

    def correctness_indexes(self):
        candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": "f" * 64,
        }
        indexes = []
        for mode in torch_tuning.CUDA_CORRECTNESS_RUNTIME_MODES:
            index = {
                "candidate_evidence": {
                    **candidate,
                    "solver_abi": "gmes-torch-v1",
                },
                "runtime_mode": copy.deepcopy(mode),
            }
            raw = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
            index["source_artifact"] = {
                "path": f"correctness/{mode['graph_mode']}.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "media_type": "application/json",
                "candidate_evidence": copy.deepcopy(candidate),
            }
            indexes.append(index)
        return indexes

    def gate(self, indexes, correctness_errors=()):
        results = [self.result(name) for name in torch_tuning.CUDA_GATES]
        with mock.patch.object(
            torch_tuning,
            "_profiler_trace_matches",
            return_value=True,
        ):
            return torch_tuning._cuda_suite_gate(
                results,
                indexes,
                list(correctness_errors),
                self.manifest,
                self.environment(),
            )

    def test_cuda_suite_requires_all_cases_raw_traces_and_two_runtime_modes(self):
        results = [self.result(name) for name in torch_tuning.CUDA_GATES]
        with mock.patch.object(
            torch_tuning,
            "_profiler_trace_matches",
            return_value=True,
        ):
            gate = torch_tuning._cuda_suite_gate(
                results,
                self.correctness_indexes(),
                [],
                self.manifest,
                self.environment(),
            )
            self.assertTrue(gate["passed"])
            self.assertTrue(gate["case_closure_complete"])
            self.assertTrue(gate["correctness_evidence_bound"])
            self.assertEqual(gate["contract_id"], "single-gpu-cuda-closure-v2")
            self.assertEqual(
                gate["correctness_indexes"],
                [
                    {
                        "runtime_mode": index["runtime_mode"],
                        "source_artifact": index["source_artifact"],
                    }
                    for index in self.correctness_indexes()
                ],
            )
            self.assertTrue(
                all(
                    set(record) == {"runtime_mode", "source_artifact"}
                    and set(record["source_artifact"])
                    == set(torch_tuning.CUDA_CORRECTNESS_SOURCE_DESCRIPTOR_KEYS)
                    for record in gate["correctness_indexes"]
                )
            )
            self.assertEqual(
                {
                    mode["precision"]
                    for mode in gate["required_correctness_runtime_modes"]
                },
                {"float32"},
            )
            self.assertEqual(
                gate["required_case_precisions"][-1],
                {"case": "single-gpu-3d", "precision": "float64"},
            )

            failed_environment = self.environment()
            failed_environment["gpu_topology_command_status"] = 1
            failed = torch_tuning._cuda_suite_gate(
                results,
                self.correctness_indexes(),
                [],
                self.manifest,
                failed_environment,
            )
            self.assertFalse(failed["environment_complete"])
            self.assertFalse(failed["passed"])

            transfer = copy.deepcopy(results)
            transfer[0]["profiler"]["host_to_device_events"] = 1
            self.assertFalse(
                torch_tuning._cuda_suite_gate(
                    transfer,
                    self.correctness_indexes(),
                    [],
                    self.manifest,
                    self.environment(),
                )["passed"]
            )

            undersized = copy.deepcopy(results)
            for result in undersized:
                for record in result["state_finiteness"]["stages"].values():
                    record["floating_or_complex_buffer_count"] = 1
                    record["floating_or_complex_element_count"] = 1
            self.assertFalse(
                torch_tuning._cuda_suite_gate(
                    undersized,
                    self.correctness_indexes(),
                    [],
                    self.manifest,
                    self.environment(),
                )["passed"]
            )

            mixed_names = copy.deepcopy(results)
            mixed_names[0]["state_progress"]["changed_buffers"] = ["ex", 1]
            self.assertFalse(
                torch_tuning._cuda_suite_gate(
                    mixed_names,
                    self.correctness_indexes(),
                    [],
                    self.manifest,
                    self.environment(),
                )["passed"]
            )

            incomplete = torch_tuning._cuda_suite_gate(
                results[:-1],
                self.correctness_indexes()[:1],
                ["graph CUDA correctness index is absent"],
                self.manifest,
                self.environment(),
            )
            self.assertFalse(incomplete["passed"])
            self.assertFalse(incomplete["case_closure_complete"])
            self.assertFalse(incomplete["correctness_evidence_bound"])

            wrong_precision = copy.deepcopy(results)
            wrong_precision[-1]["runtime"]["precision"] = "float32"
            wrong_precision[-1]["runtime"]["field_storage_dtype"] = "torch.float32"
            self.assertFalse(
                torch_tuning._cuda_suite_gate(
                    wrong_precision,
                    self.correctness_indexes(),
                    [],
                    self.manifest,
                    self.environment(),
                )["passed"]
            )

            nonfinite = copy.deepcopy(results)
            nonfinite[-1]["state_finiteness"]["stages"]["post_timed"][
                "nonfinite_element_count"
            ] = 1
            nonfinite[-1]["state_finiteness"]["stages"]["post_timed"]["finite"] = False
            nonfinite[-1]["state_finiteness"]["passed"] = False
            self.assertFalse(
                torch_tuning._cuda_suite_gate(
                    nonfinite,
                    self.correctness_indexes(),
                    [],
                    self.manifest,
                    self.environment(),
                )["passed"]
            )

            malformed = copy.deepcopy(results)
            for result in malformed:
                result["state_finiteness"]["tracked_buffers"] = ["not-a-real-buffer"]
                for record in result["state_finiteness"]["stages"].values():
                    record["floating_or_complex_buffer_count"] = 1
                    record["floating_or_complex_element_count"] = 1
                    record["nonfinite_element_count"] = False
            self.assertFalse(
                torch_tuning._cuda_suite_gate(
                    malformed,
                    self.correctness_indexes(),
                    [],
                    self.manifest,
                    self.environment(),
                )["passed"]
            )

    def test_cuda_correctness_source_descriptors_are_exact_and_distinct(self):
        valid = self.correctness_indexes()
        gate = self.gate(valid)
        self.assertTrue(gate["correctness_evidence_bound"])
        self.assertEqual(
            [
                record["source_artifact"]["sha256"]
                for record in gate["correctness_indexes"]
            ],
            [
                index["source_artifact"]["sha256"]
                for index in self.correctness_indexes()
            ],
        )
        self.assertEqual(
            len(
                {
                    record["source_artifact"]["sha256"]
                    for record in gate["correctness_indexes"]
                }
            ),
            2,
        )

        mutations = {}
        extra = copy.deepcopy(valid)
        extra[0]["source_artifact"]["unexpected"] = True
        mutations["extra descriptor key"] = extra
        missing = copy.deepcopy(valid)
        del missing[0]["source_artifact"]["size_bytes"]
        mutations["missing descriptor key"] = missing
        extra_candidate = copy.deepcopy(valid)
        extra_candidate[0]["source_artifact"]["candidate_evidence"]["unexpected"] = True
        mutations["extra candidate key"] = extra_candidate
        mismatched_candidate = copy.deepcopy(valid)
        mismatched_candidate[0]["source_artifact"]["candidate_evidence"][
            "candidate_git_commit"
        ] = ("b" * 40)
        mutations["mismatched candidate"] = mismatched_candidate
        for name, indexes in mutations.items():
            with self.subTest(name=name):
                invalid = self.gate(indexes)
                self.assertFalse(invalid["correctness_evidence_bound"])
                self.assertFalse(invalid["passed"])
                self.assertTrue(
                    any(
                        "source artifact descriptor" in error
                        for error in invalid["errors"]
                    )
                )

        duplicate = copy.deepcopy(valid)
        duplicate[1]["source_artifact"] = copy.deepcopy(duplicate[0]["source_artifact"])
        reused = self.gate(duplicate)
        self.assertFalse(reused["correctness_evidence_bound"])
        self.assertFalse(reused["passed"])
        self.assertTrue(
            any("source artifact descriptor" in error for error in reused["errors"])
        )

        same_digest = copy.deepcopy(valid)
        same_digest[1]["source_artifact"]["sha256"] = same_digest[0]["source_artifact"][
            "sha256"
        ]
        reused_digest = self.gate(same_digest)
        self.assertFalse(reused_digest["correctness_evidence_bound"])
        self.assertTrue(
            any(
                "source artifact descriptor" in error
                for error in reused_digest["errors"]
            )
        )

        same_path = copy.deepcopy(valid)
        same_path[1]["source_artifact"]["path"] = same_path[0]["source_artifact"][
            "path"
        ]
        reused_path = self.gate(same_path)
        self.assertFalse(reused_path["correctness_evidence_bound"])
        self.assertTrue(
            any("paths are not distinct" in error for error in reused_path["errors"])
        )

        reordered = self.gate(list(reversed(valid)))
        self.assertFalse(reordered["correctness_evidence_bound"])
        self.assertFalse(reordered["passed"])
        self.assertIn(
            "CUDA correctness runtime mode closure differs",
            reordered["errors"],
        )

        relabeled = copy.deepcopy(valid)
        relabeled[0]["runtime_mode"] = copy.deepcopy(relabeled[1]["runtime_mode"])
        duplicate_mode = self.gate(relabeled)
        self.assertFalse(duplicate_mode["correctness_evidence_bound"])
        self.assertIn(
            "CUDA correctness runtime mode closure differs",
            duplicate_mode["errors"],
        )

        swapped_descriptors = copy.deepcopy(valid)
        (
            swapped_descriptors[0]["source_artifact"],
            swapped_descriptors[1]["source_artifact"],
        ) = (
            swapped_descriptors[1]["source_artifact"],
            swapped_descriptors[0]["source_artifact"],
        )
        swapped = self.gate(swapped_descriptors)
        self.assertFalse(swapped["correctness_evidence_bound"])
        self.assertFalse(swapped["passed"])
        self.assertTrue(
            any("source artifact descriptor" in error for error in swapped["errors"])
        )

    def test_cuda_correctness_loader_revalidates_ordered_eager_and_graph_indexes(
        self,
    ):
        missing, missing_errors = torch_tuning._load_cuda_correctness_indexes(
            (Path("eager.json"), Path("graph.json")),
            None,
            self.manifest,
            {"candidate_git_commit": "a" * 40},
        )
        self.assertEqual(missing, [])
        self.assertEqual(
            missing_errors,
            ["CUDA correctness requires exactly eager and graph receipts"],
        )
        indexes = self.correctness_indexes()
        with (
            mock.patch.object(
                torch_tuning,
                "load_correctness_evidence_index",
                side_effect=indexes,
            ) as loader,
            mock.patch.object(
                torch_tuning,
                "correctness_binding_complete",
                return_value=True,
            ) as binding,
            mock.patch.object(
                torch_tuning,
                "load_runtime_publication_receipt",
                side_effect=({}, {}),
            ) as receipt_loader,
        ):
            loaded, errors = torch_tuning._load_cuda_correctness_indexes(
                (Path("eager.json"), Path("graph.json")),
                (Path("eager-receipt.json"), Path("graph-receipt.json")),
                self.manifest,
                {"candidate_git_commit": "a" * 40},
            )
        self.assertEqual(errors, [])
        self.assertEqual(loaded, indexes)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(receipt_loader.call_count, 2)
        self.assertEqual(binding.call_count, 2)

        duplicate = copy.deepcopy(indexes)
        duplicate[1]["source_artifact"] = copy.deepcopy(duplicate[0]["source_artifact"])
        with (
            mock.patch.object(
                torch_tuning,
                "load_correctness_evidence_index",
                side_effect=duplicate,
            ),
            mock.patch.object(
                torch_tuning,
                "correctness_binding_complete",
                return_value=True,
            ),
            mock.patch.object(
                torch_tuning,
                "load_runtime_publication_receipt",
                side_effect=({}, {}),
            ),
        ):
            _loaded, duplicate_errors = torch_tuning._load_cuda_correctness_indexes(
                (Path("eager.json"), Path("graph.json")),
                (Path("eager-receipt.json"), Path("graph-receipt.json")),
                self.manifest,
                {"candidate_git_commit": "a" * 40},
            )
        self.assertTrue(
            any("source artifact descriptor" in error for error in duplicate_errors)
        )


class TwoGpuEvidenceContractTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            case="strong-mixed",
            warmup=5,
            steps=100,
            repeats=3,
            profile_steps=2,
            threads_per_rank=1,
            trace_directory=Path("/tmp/traces"),
            descriptor_root=Path("/tmp"),
        )
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": "b" * 64,
        }
        self.environment = _two_gpu_environment()

    @staticmethod
    def memory():
        return {
            "allocated_before_bytes": 100,
            "allocated_after_bytes": 100,
            "allocated_growth_bytes": 0,
            "peak_allocated_bytes": 200,
            "peak_reserved_bytes": 300,
            "bounded": True,
        }

    @staticmethod
    def storage():
        return {
            "address_count": 10,
            "initial_address_sha256": "a" * 64,
            "final_address_sha256": "a" * 64,
            "addresses_stable": True,
            "alias_count": 0,
            "tracked_tensor_count": 10,
            "device_resident": True,
            "resident_bytes": 1000,
            "category_bytes": {"state": 1000},
        }

    @staticmethod
    def halo(rank):
        return {
            "buffer_count": 8,
            "bytes": 128,
            "initial_address_sha256": "b" * 64,
            "final_address_sha256": "b" * 64,
            "addresses_stable": True,
            "alias_count": 0,
            "device": f"cuda:{rank}",
            "device_resident": True,
        }

    def profile(self, rank):
        return {
            "trace": f"case/gmes-two-gpu-{rank}.json",
            "trace_size_bytes": 100,
            "trace_sha256": str(rank + 1) * 64,
            "kernel_launches": 20,
            "host_to_device_events": 0,
            "device_to_host_events": 0,
            "nccl_kernel_launches": 4,
            "nccl_device_us": 10.0,
            "nccl_compute_overlap_us": 5.0,
            "nccl_exposed_us": 5.0,
            "overlap_fraction": 0.5,
            "halo_annotations": {
                name: {"count": self.args.profile_steps, "duration_us": 1.0}
                for name in torch_two_gpu.HALO_ANNOTATIONS
            },
        }

    def workers(self):
        spec = torch_two_gpu.CASES[self.args.case]
        measurement = torch_two_gpu._measurement(self.args)
        serial = torch_two_gpu._summary(
            [1.0] * self.args.repeats,
            steps=self.args.steps,
            cells=128 * 96 * 96,
        )
        serial.update(
            {
                "construction_seconds": 1.0,
                "capture_seconds": 1.0,
                "memory": self.memory(),
                "storage": self.storage(),
            }
        )
        rank_samples = [[0.49, 0.5] for _ in range(self.args.repeats)]
        distributed = torch_two_gpu._summary(
            [0.5] * self.args.repeats,
            steps=self.args.steps,
            cells=128 * 96 * 96,
        )
        distributed.update(
            {
                "rank_raw_seconds": rank_samples,
                "construction_seconds": 1.0,
                "capture_seconds": 1.0,
                "peak_allocated_bytes_rank0": 200,
                "halo_bytes_rank0": 128,
                "storage_addresses_stable": True,
            }
        )
        decomposition = {
            "global_shape": list(spec["distributed_size"]),
            "axis": 0,
            "cut": 64,
            "rank_costs": [100.0, 100.0],
            "device_weights": [0.5, 0.5],
            "communication_cells": 96 * 96,
            "source_crossings": 0,
            "axis_name": "x",
        }
        identity_payload = (
            tuple(decomposition["global_shape"]),
            decomposition["axis"],
            decomposition["cut"],
            tuple(decomposition["device_weights"]),
        )
        decomposition["identity"] = hashlib.sha256(
            repr(identity_payload).encode()
        ).hexdigest()
        ranks = [
            {
                "rank": rank,
                "local_rank": rank,
                "device": f"cuda:{rank}",
                "peer_rank": 1 - rank,
                "peer_access": rank == 0,
                "construction_seconds": 1.0,
                "capture_seconds": 1.0,
                "raw_seconds": [0.49 if rank == 0 else 0.5] * self.args.repeats,
                "memory": self.memory(),
                "storage": self.storage(),
                "halo": self.halo(rank),
                "decomposition_identity": decomposition["identity"],
                "local_field_shape": [64, 96, 96],
                "global_offset": [64 * rank, 0, 0],
            }
            for rank in range(2)
        ]
        serial_worker = {
            "schema_version": torch_two_gpu.SCHEMA_VERSION,
            "kind": torch_two_gpu.WORKER_KINDS["serial"],
            "candidate_evidence": self.candidate,
            "environment": self.environment,
            "case": self.args.case,
            "size": list(spec["serial_size"]),
            "measurement": measurement,
            "serial": serial,
        }
        distributed_worker = {
            "schema_version": torch_two_gpu.SCHEMA_VERSION,
            "kind": torch_two_gpu.WORKER_KINDS["distributed"],
            "candidate_evidence": self.candidate,
            "environment": self.environment,
            "case": self.args.case,
            "size": list(spec["distributed_size"]),
            "measurement": measurement,
            "distributed": distributed,
            "decomposition": decomposition,
            "rank_evidence": ranks,
            "profiles": [self.profile(0), self.profile(1)],
        }

        def subprocess_record(role, command, marker):
            descriptors = {
                name: {
                    "path": f"case/{role}-child.{name}",
                    "sha256": marker[name] * 64,
                    "size_bytes": 0 if name == "stderr" else 100,
                    "media_type": (
                        "text/plain; charset=utf-8"
                        if name == "stderr"
                        else "application/json"
                    ),
                    "candidate_evidence": self.candidate,
                }
                for name in ("artifact", "stdout", "stderr")
            }
            return {
                "role": role,
                "command": command,
                "exit_code": 0,
                "stdout_sha256": descriptors["stdout"]["sha256"],
                "stdout_size_bytes": descriptors["stdout"]["size_bytes"],
                "stderr_sha256": descriptors["stderr"]["sha256"],
                "stderr_size_bytes": descriptors["stderr"]["size_bytes"],
                "artifact_sha256": descriptors["artifact"]["sha256"],
                "artifact_size_bytes": descriptors["artifact"]["size_bytes"],
                **descriptors,
            }

        subprocesses = {
            "serial": subprocess_record(
                "serial",
                ["python", "--worker", "serial"],
                {"artifact": "1", "stdout": "2", "stderr": "3"},
            ),
            "distributed": subprocess_record(
                "distributed",
                [
                    "python",
                    "-m",
                    "torch.distributed.run",
                    "--worker",
                    "distributed",
                ],
                {"artifact": "4", "stdout": "5", "stderr": "6"},
            ),
        }
        return serial_worker, distributed_worker, subprocesses

    def test_combiner_requires_complete_both_rank_and_raw_profile_evidence(self):
        serial, distributed, subprocesses = self.workers()
        result = torch_two_gpu._combine_worker_results(
            serial,
            distributed,
            subprocesses,
            self.args,
        )
        self.assertTrue(result["acceptance"]["passed"])
        self.assertTrue(all(result["acceptance"]["checks"].values()))
        self.assertEqual(result["acceptance"]["ratio"], 2.0)
        self.assertEqual(
            result["imbalance"]["rank_seconds_ratio_per_repeat"],
            [0.5 / 0.49] * self.args.repeats,
        )

        tampered = copy.deepcopy(distributed)
        tampered["profiles"][1]["device_to_host_events"] = 1
        failed = torch_two_gpu._combine_worker_results(
            serial,
            tampered,
            subprocesses,
            self.args,
        )
        self.assertFalse(failed["acceptance"]["passed"])
        self.assertFalse(failed["acceptance"]["checks"]["steady_state_transfers_zero"])

        wrong_rank = copy.deepcopy(distributed)
        wrong_rank["rank_evidence"][1]["device"] = "cuda:0"
        failed = torch_two_gpu._combine_worker_results(
            serial,
            wrong_rank,
            subprocesses,
            self.args,
        )
        self.assertFalse(failed["acceptance"]["checks"]["rank_evidence_complete"])

        bad_descriptor = copy.deepcopy(subprocesses)
        bad_descriptor["serial"]["stdout_sha256"] = "f" * 64
        failed = torch_two_gpu._combine_worker_results(
            serial,
            distributed,
            bad_descriptor,
            self.args,
        )
        self.assertFalse(failed["acceptance"]["checks"]["independent_subprocesses"])

    def test_worker_commands_are_separate_serial_and_torchrun_children(self):
        commands = torch_two_gpu._worker_commands(self.args, Path("/tmp/bundle"))
        serial = commands["serial"][0]
        distributed = commands["distributed"][0]
        self.assertIn("benchmarks.torch_two_gpu", serial)
        self.assertNotIn("torch.distributed.run", serial)
        self.assertIn("torch.distributed.run", distributed)
        self.assertNotEqual(serial, distributed)

    def test_trace_summary_reports_raw_transfers_phases_and_overlap(self):
        events = [
            {"ph": "X", "cat": "kernel", "name": "ncclKernel", "ts": 0, "dur": 10},
            {"ph": "X", "cat": "kernel", "name": "compute", "ts": 5, "dur": 10},
            {"ph": "X", "cat": "Memcpy", "name": "Memcpy HtoD", "ts": 1, "dur": 1},
            *[
                {"ph": "X", "cat": "cpu_op", "name": name, "ts": 0, "dur": 1}
                for name in torch_two_gpu.HALO_ANNOTATIONS
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank.json"
            path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
            summary = torch_two_gpu._trace_summary(path)
        self.assertEqual(summary["host_to_device_events"], 1)
        self.assertEqual(summary["device_to_host_events"], 0)
        self.assertEqual(summary["nccl_kernel_launches"], 1)
        self.assertEqual(summary["nccl_compute_overlap_us"], 5)
        self.assertEqual(summary["nccl_exposed_us"], 5)
        self.assertEqual(
            set(summary["halo_annotations"]),
            set(torch_two_gpu.HALO_ANNOTATIONS),
        )


class TwoGpuCorrectnessClosureTest(unittest.TestCase):
    @staticmethod
    def candidate():
        return {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": "b" * 64,
        }

    def raw_evidence(self, path, names, shapes):
        return {
            "artifact": {
                "path": path,
                "sha256": "c" * 64,
                "size_bytes": 100,
                "media_type": torch_two_gpu_correctness.MEDIA_TYPE_NPZ,
                "candidate_evidence": self.candidate(),
            },
            "array_names": names,
            "field_shapes": shapes,
        }

    @staticmethod
    def storage(rank=None):
        value = {
            "address_names": ["fields/Ex", "state/0"],
            "address_count": 2,
            "initial_sha256": "d" * 64,
            "final_sha256": "d" * 64,
            "addresses_stable": True,
        }
        if rank is not None:
            value["rank"] = rank
        return value

    def records(self):
        records = []
        for case in torch_two_gpu_correctness._cases():
            records.append(
                {
                    "name": case["name"],
                    "axis": case["axis"],
                    "cut": 1,
                    "capture_errors": {
                        str(step): 0.0
                        for step in torch_two_gpu_correctness.CAPTURE_STEPS
                    },
                    "checkpoint_replay_fields": list(
                        torch_two_gpu_correctness.COMPONENTS
                    ),
                    "checkpoint_replay_steps": 5,
                    "checkpoint_determinism_error": 0.0,
                    "checkpoint_reference_error": 0.0,
                    "rank_storage": [self.storage(rank) for rank in range(2)],
                    "serial_storage": self.storage(),
                    "raw_evidence": self.raw_evidence(
                        f"correctness-raw/{case['name']}.npz",
                        torch_two_gpu_correctness._case_raw_array_names(),
                        torch_two_gpu_correctness._field_shapes(
                            case["size"], case["resolution"]
                        ),
                    ),
                }
            )
        return records

    def test_enforced_suite_requires_full_matrix_long_run_and_rank_replay(self):
        records = self.records()
        stability = {
            "steps": 1000,
            "maximum_error": 0.0,
            "finite": True,
            "energy_ratio": 1.0,
            "raw_evidence": self.raw_evidence(
                "correctness-raw/long-stability.npz",
                torch_two_gpu_correctness._long_raw_array_names(),
                torch_two_gpu_correctness._field_shapes((8, 6, 4), 2),
            ),
        }
        accepted = torch_two_gpu_correctness._suite_acceptance(
            records,
            stability,
            True,
            _two_gpu_environment(),
        )
        self.assertTrue(accepted["passed"])

        bad_environment = _two_gpu_environment()
        bad_environment["host_contract"]["common_identity"]["os"] = "Linux"
        rejected = torch_two_gpu_correctness._suite_acceptance(
            records,
            stability,
            True,
            bad_environment,
        )
        self.assertFalse(rejected["checks"]["environment_complete"])

        records[0]["rank_storage"].append(self.storage(2))
        rejected = torch_two_gpu_correctness._suite_acceptance(
            records,
            stability,
            True,
            _two_gpu_environment(),
        )
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["checks"]["rank_storage_stable"])

    def test_enforced_suite_rejects_raw_array_self_report_tamper(self):
        records = self.records()
        records[0]["raw_evidence"]["array_names"] = records[0]["raw_evidence"][
            "array_names"
        ][:-1]
        stability = {
            "steps": 1000,
            "maximum_error": 0.0,
            "finite": True,
            "energy_ratio": 1.0,
            "raw_evidence": self.raw_evidence(
                "correctness-raw/long-stability.npz",
                torch_two_gpu_correctness._long_raw_array_names(),
                torch_two_gpu_correctness._field_shapes((8, 6, 4), 2),
            ),
        }
        rejected = torch_two_gpu_correctness._suite_acceptance(
            records, stability, True, _two_gpu_environment()
        )
        self.assertFalse(rejected["checks"]["raw_full_fields_bound"])
        self.assertFalse(rejected["passed"])

    def test_raw_npz_writer_binds_order_shape_dtype_and_candidate(self):
        shapes = torch_two_gpu_correctness._field_shapes((2, 2, 2), 1)
        names = torch_two_gpu_correctness._long_raw_array_names()
        arrays = {
            name: np.zeros(shapes[name.rsplit("/", 1)[-1]], dtype=np.float64)
            for name in names
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "correctness-raw" / "long-stability.npz"
            evidence = torch_two_gpu_correctness._write_raw_evidence(
                path,
                arrays,
                root,
                self.candidate(),
                shapes,
                expected_names=names,
                field_dtype="float64",
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive.files, names)
            self.assertEqual(set(evidence), {"artifact", "array_names", "field_shapes"})
            self.assertEqual(
                set(evidence["artifact"]),
                {
                    "path",
                    "sha256",
                    "size_bytes",
                    "media_type",
                    "candidate_evidence",
                },
            )
            self.assertEqual(
                evidence["artifact"]["path"],
                "correctness-raw/long-stability.npz",
            )

            bad_arrays = dict(arrays)
            bad_arrays[names[0]] = bad_arrays[names[0]].astype(np.float32)
            with self.assertRaisesRegex(ValueError, "raw evidence array is invalid"):
                torch_two_gpu_correctness._write_raw_evidence(
                    root / "correctness-raw" / "bad.npz",
                    bad_arrays,
                    root,
                    self.candidate(),
                    shapes,
                    expected_names=names,
                    field_dtype="float64",
                )

    def test_case_npz_requires_full_fields_and_uint64_addresses(self):
        shapes = torch_two_gpu_correctness._field_shapes((2, 2, 2), 1)
        names = torch_two_gpu_correctness._case_raw_array_names()
        arrays = {}
        for name in names:
            if name.startswith("storage/"):
                arrays[name] = np.asarray([11, 12], dtype=np.uint64)
            else:
                component = name.rsplit("/", 1)[-1]
                arrays[name] = np.zeros(shapes[component], dtype=np.float64)
        self.assertEqual(len(names), 84)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "correctness-raw" / "axis-0-real.npz"
            evidence = torch_two_gpu_correctness._write_raw_evidence(
                path,
                arrays,
                root,
                self.candidate(),
                shapes,
                expected_names=names,
                field_dtype="float64",
            )
            self.assertEqual(evidence["array_names"], names)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive.files, names)
                self.assertEqual(
                    archive["storage/rank/0/initial"].dtype,
                    np.dtype("uint64"),
                )

            bad_arrays = dict(arrays)
            bad_arrays["storage/rank/0/initial"] = np.asarray([11, 12], dtype=np.int64)
            with self.assertRaisesRegex(ValueError, "raw evidence array is invalid"):
                torch_two_gpu_correctness._write_raw_evidence(
                    root / "correctness-raw" / "bad-address.npz",
                    bad_arrays,
                    root,
                    self.candidate(),
                    shapes,
                    expected_names=names,
                    field_dtype="float64",
                )

    def test_storage_record_binds_sorted_names_raw_values_and_digest(self):
        initial = {"state/0": 22, "fields/Ex": 11}
        summary, initial_values, final_values = (
            torch_two_gpu_correctness._storage_record(0, initial, dict(initial))
        )
        self.assertEqual(summary["address_names"], ["fields/Ex", "state/0"])
        self.assertEqual(summary["address_count"], 2)
        self.assertEqual(summary["rank"], 0)
        self.assertEqual(initial_values.dtype, np.dtype("uint64"))
        self.assertEqual(initial_values.tolist(), [11, 22])
        np.testing.assert_array_equal(initial_values, final_values)
        self.assertEqual(summary["initial_sha256"], summary["final_sha256"])
        self.assertEqual(
            summary["initial_sha256"],
            torch_two_gpu_correctness._storage_digest(
                summary["address_names"], initial_values
            ),
        )

        changed = dict(initial)
        changed["state/0"] = 23
        changed_summary, _, _ = torch_two_gpu_correctness._storage_record(
            0, initial, changed
        )
        self.assertFalse(changed_summary["addresses_stable"])
        self.assertNotEqual(
            changed_summary["initial_sha256"], changed_summary["final_sha256"]
        )


class TwoGpuFailureDescriptorTest(unittest.TestCase):
    def test_failure_logs_use_bundle_relative_typed_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "failures" / "probe.stdout"
            path.parent.mkdir()
            path.write_bytes(b"{}\n")
            candidate = {"candidate_git_commit": "a" * 40}
            descriptor = two_gpu_failure_evidence._descriptor(
                path,
                root,
                candidate,
                "text/plain; charset=utf-8",
            )
        self.assertEqual(
            set(descriptor),
            {
                "path",
                "sha256",
                "size_bytes",
                "media_type",
                "candidate_evidence",
            },
        )
        self.assertEqual(descriptor["path"], "failures/probe.stdout")
        self.assertEqual(descriptor["media_type"], "text/plain; charset=utf-8")


if __name__ == "__main__":
    unittest.main()
