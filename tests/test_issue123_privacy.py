from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import shutil
import stat
import struct
import tarfile
import tempfile
import time
import traceback
import unicodedata
import unittest
import zipfile
import zlib
from unittest import mock
from pathlib import Path

import numpy as np

from benchmarks import issue123_completion as completion
from benchmarks import issue123_privacy as privacy
from benchmarks import issue123_publication as publication


def _json_bytes(value):
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _fullwidth_ascii(value):
    return "".join(
        chr(ord(character) + 0xFEE0) if 0x21 <= ord(character) <= 0x7E else character
        for character in value
    )


def _exception_messages(error):
    messages = []
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        messages.append(str(current))
        pending.extend((current.__cause__, current.__context__))
    return messages


def _assert_sanitized_privacy_error(testcase, error, marker):
    testcase.assertIs(type(error), privacy.PrivacyError)
    testcase.assertIsNone(error.__cause__)
    testcase.assertIsNone(error.__context__)
    pending = [error]
    seen = set()
    diagnostics = [
        "".join(traceback.format_exception(type(error), error, error.__traceback__))
    ]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        diagnostics.extend(
            (
                str(current),
                repr(current),
                repr(current.args),
                repr(getattr(current, "__notes__", ())),
            )
        )
        pending.extend(
            item
            for item in (
                current.__cause__,
                current.__context__,
                *getattr(current, "exceptions", ()),
            )
            if item is not None
        )
    testcase.assertNotIn(marker, " ".join(diagnostics))


def _trace_events():
    return [
        {
            "name": "process_name",
            "cat": "",
            "ph": "M",
            "pid": 9017,
            "tid": 41,
            "args": {"name": "private-worker"},
        },
        {
            "name": "[memory]",
            "cat": "",
            "ph": "i",
            "ts": 1_700_000_000_000,
            "pid": 9017,
            "tid": 41,
            "args": {
                "Bytes": 64,
                "Total Allocated": 64,
                "Addr": "allocation-address-9",
            },
        },
        {
            "name": "[memory]",
            "cat": "",
            "ph": "i",
            "ts": 1_700_000_000_001,
            "pid": 9017,
            "tid": 41,
            "args": {
                "Bytes": -64,
                "Total Allocated": 0,
                "Addr": "allocation-address-9",
            },
        },
        {
            "name": "Torch-Compiled Region: 0/0",
            "cat": "user_annotation",
            "ph": "X",
            "ts": 1_700_000_000_002,
            "dur": 20,
            "pid": 9017,
            "tid": 41,
            "args": {},
        },
        {
            "name": "CudaGraphLaunch",
            "cat": "cuda_runtime",
            "ph": "X",
            "ts": 1_700_000_000_003,
            "dur": 1,
            "pid": 9017,
            "tid": 41,
            "args": {"Graph Id": 82, "correlation": "graph-correlation"},
        },
        {
            "name": "generated_compute_kernel",
            "cat": "kernel",
            "ph": "X",
            "ts": 1_700_000_000_005,
            "dur": 10,
            "pid": 44_000,
            "tid": 52,
            "args": {"stream": "stream-17", "correlation": "correlation-1"},
        },
        {
            "name": "ncclKernel_AllReduce",
            "cat": "kernel",
            "ph": "X",
            "ts": 1_700_000_000_010,
            "dur": 10,
            "pid": 44_000,
            "tid": 53,
            "args": {"stream": "stream-22", "correlation": "correlation-2"},
        },
        {
            "name": "Memcpy DtoD",
            "cat": "gpu_memcpy",
            "ph": "X",
            "ts": 1_700_000_000_021,
            "dur": 1,
            "pid": 44_000,
            "tid": 52,
            "args": {"stream": "stream-17", "correlation": "correlation-3"},
        },
        {
            "name": "aten::index_copy_",
            "cat": "cpu_op",
            "ph": "X",
            "ts": 1_700_000_000_022,
            "dur": 2,
            "pid": 9017,
            "tid": 41,
            "args": {},
        },
    ]


def _trace_bytes(events=None):
    return _json_bytes({"traceEvents": _trace_events() if events is None else events})


def _trace_policy():
    return {
        "name": "steady-profile",
        "event_count": 9,
        "semantic_signatures": [
            ["metadata-process-name", "metadata"],
            ["allocation", "instant"],
            ["allocation", "instant"],
            ["compiled-region", "complete"],
            ["cuda-graph", "complete"],
            ["kernel", "complete"],
            ["nccl-kernel", "complete"],
            ["copy-device", "complete"],
            ["indexed-write-index-copy", "complete"],
        ],
        "allocation_events": 2,
        "positive_allocation_events": 1,
        "allocated_bytes": 64,
        "freed_bytes": 64,
        "allocation_net_bytes": 0,
        "live_allocation_baseline_bytes": 0,
        "peak_live_allocated_bytes": 64,
        "final_live_allocated_bytes": 0,
        "live_allocation_growth_bytes": 0,
        "graph_breaks": 0,
        "recompiles": 0,
        "fallbacks": 0,
        "device_copy_events": 1,
        "host_to_device_events": 0,
        "device_to_host_events": 0,
        "kernel_launches": 2,
        "compiled_region_events": 1,
        "cuda_graph_launches": 1,
        "nccl_kernel_launches": 1,
        "require_nccl_overlap": True,
    }


def _witness_trace_policy(
    name,
    semantic_signatures,
    *,
    kernel_launches=0,
    graphs=0,
    compiled_regions=0,
):
    return {
        "name": name,
        "event_count": len(semantic_signatures),
        "semantic_signatures": semantic_signatures,
        "allocation_events": 0,
        "positive_allocation_events": 0,
        "allocated_bytes": 0,
        "freed_bytes": 0,
        "allocation_net_bytes": 0,
        "live_allocation_baseline_bytes": 0,
        "peak_live_allocated_bytes": 0,
        "final_live_allocated_bytes": 0,
        "live_allocation_growth_bytes": 0,
        "graph_breaks": 0,
        "recompiles": 0,
        "fallbacks": 0,
        "device_copy_events": 0,
        "host_to_device_events": 0,
        "device_to_host_events": 0,
        "kernel_launches": kernel_launches,
        "compiled_region_events": compiled_regions,
        "cuda_graph_launches": graphs,
        "nccl_kernel_launches": 0,
        "require_nccl_overlap": False,
    }


def _cpu_eager_trace_bytes():
    return _trace_bytes(
        [
            {
                "name": "process_name",
                "cat": "",
                "ph": "M",
                "pid": 7,
                "tid": 3,
                "args": {"name": "private-cpu-worker"},
            },
            {
                "name": "aten::index_copy_",
                "cat": "cpu_op",
                "ph": "X",
                "ts": 100,
                "dur": 2,
                "pid": 7,
                "tid": 3,
                "args": {},
            },
        ]
    )


def _cuda_eager_trace_bytes():
    return _trace_bytes(
        [
            {
                "name": "process_name",
                "cat": "",
                "ph": "M",
                "pid": 8,
                "tid": 4,
                "args": {"name": "private-cuda-worker"},
            },
            {
                "name": "cudaLaunchKernel",
                "cat": "cuda_runtime",
                "ph": "X",
                "ts": 99,
                "dur": 1,
                "pid": 8,
                "tid": 4,
                "args": {"correlation": 10},
            },
            {
                "name": "generated_compute_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 100,
                "dur": 5,
                "pid": 9,
                "tid": 5,
                "args": {
                    "Device Type": "CUDA",
                    "Device Id": 0,
                    "context": 1,
                    "stream": 3,
                    "correlation": 10,
                },
            },
        ]
    )


def _fixture():
    bindings = {
        "final_sha": "a" * 40,
        "manifest_sha256": "b" * 64,
        "jobs": [
            {
                "name": "Python 3.14 / ubuntu-latest",
                "run_id": 101,
                "run_attempt": 1,
                "job_id": 201,
            },
            {
                "name": "Python 3.14 / macos-latest",
                "run_id": 101,
                "run_attempt": 1,
                "job_id": 202,
            },
            {
                "name": "CodeQL / python",
                "run_id": 102,
                "run_attempt": 2,
                "job_id": 203,
            },
            {
                "name": "CodeQL / c-cpp",
                "run_id": 102,
                "run_attempt": 2,
                "job_id": 204,
            },
        ],
    }
    policy_scopes = []
    private_scopes = []
    timing_samples = [0.003, 0.001, 0.002]
    for index, scope in enumerate(privacy.TECHNICAL_SCOPE_ORDER):
        payload_name = f"evidence/{scope}.bin"
        payload = f"safe-payload-{scope}".encode()
        case = f"case-{scope}"
        comparison = {
            "name": "field",
            "dtype": "float64",
            "shape": [2],
            "comparison_contract": "elementwise",
            "rtol": 1e-6,
            "atol": 1e-9,
            "normalized_limit": None,
        }
        trace_policies = [_trace_policy()]
        private_traces = [
            {
                "name": "steady-profile",
                "trace_bytes": _trace_bytes(),
            }
        ]
        if scope == "cpu":
            trace_policies.append(
                _witness_trace_policy(
                    "cpu-eager-witness",
                    [
                        ["metadata-process-name", "metadata"],
                        ["indexed-write-index-copy", "complete"],
                    ],
                )
            )
            private_traces.append(
                {
                    "name": "cpu-eager-witness",
                    "trace_bytes": _cpu_eager_trace_bytes(),
                }
            )
        elif scope == "single_gpu":
            trace_policies.append(
                _witness_trace_policy(
                    "cuda-eager-witness",
                    [
                        ["metadata-process-name", "metadata"],
                        ["cuda-runtime", "complete"],
                        ["kernel", "complete"],
                    ],
                    kernel_launches=1,
                )
            )
            private_traces.append(
                {
                    "name": "cuda-eager-witness",
                    "trace_bytes": _cuda_eager_trace_bytes(),
                }
            )
        policy_scopes.append(
            {
                "name": scope,
                "identities": ["machine"],
                "timings": [
                    {
                        "name": "steady",
                        "sample_count": len(timing_samples),
                        "samples_sha256": _canonical_sha256(timing_samples),
                    }
                ],
                "traces": trace_policies,
                "payloads": [
                    {
                        "name": payload_name,
                        "media_type": "application/octet-stream",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
                "correctness": [
                    {
                        "name": case,
                        "captures": [
                            {
                                "capture": 0,
                                "arrays": [comparison],
                            }
                        ],
                    }
                ],
            }
        )
        private_scopes.append(
            {
                "name": scope,
                "satisfied": True,
                "identities": {
                    "machine": {
                        "opaque_private_value": f"private-host-{index}",
                    }
                },
                "timings": [
                    {
                        "name": "steady",
                        "unit": "seconds",
                        "samples": list(timing_samples),
                    }
                ],
                "traces": private_traces,
                "payloads": [
                    {
                        "name": payload_name,
                        "media_type": "application/octet-stream",
                        "bytes": payload,
                    }
                ],
                "correctness": [
                    {
                        "name": case,
                        "captures": [
                            {
                                "capture": 0,
                                "arrays": [
                                    {
                                        "name": "field",
                                        "dtype": "float64",
                                        "shape": [2],
                                        "reference_bytes": struct.pack("<2d", 1.0, 2.0),
                                        "candidate_bytes": struct.pack(
                                            "<2d", 1.0 + 1e-10, 2.0
                                        ),
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    policy = {
        "schema_version": privacy.SCHEMA_VERSION,
        "kind": privacy.POLICY_KIND,
        "bindings": bindings,
        "scopes": policy_scopes,
        "execution_witnesses": [
            {
                "claim": "cpu-eager",
                "scope": "cpu",
                "trace_name": "cpu-eager-witness",
                "validation_workflow": "CI",
                "validator_job_name": "Python 3.14 / ubuntu-latest",
            },
            {
                "claim": "cuda-eager",
                "scope": "single_gpu",
                "trace_name": "cuda-eager-witness",
                "validation_workflow": "CI",
                "validator_job_name": "Python 3.14 / ubuntu-latest",
            },
            {
                "claim": "cuda-graph",
                "scope": "two_gpu",
                "trace_name": "steady-profile",
                "validation_workflow": "CI",
                "validator_job_name": "Python 3.14 / ubuntu-latest",
            },
        ],
        "issue115": {
            "timings": [{"scope": "cpu", "name": "steady"}],
            "profilers": [{"scope": "two_gpu", "name": "steady-profile"}],
        },
    }
    private = {
        "schema_version": privacy.SCHEMA_VERSION,
        "kind": privacy.PRIVATE_INPUT_KIND,
        "bindings": copy.deepcopy(bindings),
        "scopes": private_scopes,
    }
    return policy, private


def _production_source_fixture(root, policy, private):
    """Write real-format private sources for the production adapter tests."""

    root = Path(root)
    candidate = {
        "candidate_git_commit": policy["bindings"]["final_sha"],
        "candidate_git_status": "",
        "manifest_sha256": policy["bindings"]["manifest_sha256"],
    }
    runtime_paths = {}
    runtime_records = []
    for ordinal, role in enumerate(privacy.RUNTIME_RECEIPT_ORDER):
        path = root / f"runtime-{ordinal}.json"
        raw = privacy.binding_canonical_json_bytes({"ordinal": ordinal, "role": role})
        path.write_bytes(raw)
        runtime_paths[role] = path
        runtime_records.append(
            {
                "role": role,
                "source_path": path.name,
                "bundle_path": f"runtime/{ordinal}-{role}.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    private_scopes = {scope["name"]: scope for scope in private["scopes"]}
    sources = []
    scope_artifacts = {
        scope: {"production-source": {}} for scope in privacy.TECHNICAL_SCOPE_ORDER
    }
    for ordinal, target in enumerate(privacy._expected_source_targets(policy)):
        scope = private_scopes[target["scope"]]
        semantic_role = target["semantic_role"]
        suffix = "bin"
        if semantic_role == "identity":
            selected = scope["identities"][target["name"]]
            raw = privacy.binding_canonical_json_bytes({"value": selected})
            selector = {"json_pointer": "/value", "expected_type": "object"}
            suffix = "json"
            media_type = "application/json"
        elif semantic_role == "timing":
            selected = next(
                item for item in scope["timings"] if item["name"] == target["name"]
            )["samples"]
            raw = privacy.binding_canonical_json_bytes({"value": selected})
            selector = {"json_pointer": "/value", "expected_type": "array"}
            suffix = "json"
            media_type = "application/json"
        elif semantic_role == "trace":
            raw = next(
                item for item in scope["traces"] if item["name"] == target["name"]
            )["trace_bytes"]
            selector = {"whole_bytes": True}
            suffix = "json"
            media_type = "application/json"
        elif semantic_role == "payload":
            payload = next(
                item for item in scope["payloads"] if item["name"] == target["name"]
            )
            raw = payload["bytes"]
            selector = {"whole_bytes": True}
            media_type = payload["media_type"]
        else:
            case = next(
                item for item in scope["correctness"] if item["name"] == target["case"]
            )
            capture = next(
                item
                for item in case["captures"]
                if item["capture"] == target["capture"]
            )
            array = next(
                item for item in capture["arrays"] if item["name"] == target["name"]
            )
            role = semantic_role.removeprefix("correctness-")
            values = np.frombuffer(
                array[f"{role}_bytes"], dtype=np.dtype(target["dtype"])
            ).reshape(target["shape"])
            path = root / f"source-{ordinal}.npz"
            np.savez(path, **{target["name"]: values})
            raw = path.read_bytes()
            selector = {
                "npz_member": {
                    "name": target["name"],
                    "dtype": target["dtype"],
                    "shape": target["shape"],
                    "byte_order": "little",
                    "order": "C",
                }
            }
            suffix = "npz"
            media_type = "application/x-npz"
        path = root / f"source-{ordinal}.{suffix}"
        if not path.exists():
            path.write_bytes(raw)
        record = {
            "scope": target["scope"],
            "semantic_role": semantic_role,
            "completion_role": (
                f"/artifacts/{target['scope']}/production-source/{ordinal}"
            ),
            "source_path": path.name,
            "bundle_path": f"private-inputs/{ordinal}.{suffix}",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "media_type": media_type,
            "selector": selector,
        }
        sources.append(record)
        scope_artifacts[target["scope"]]["production-source"][str(ordinal)] = {
            "path": record["bundle_path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "media_type": record["media_type"],
            "candidate_evidence": candidate,
        }
    document = {
        "schema_version": privacy.PUBLICATION_SOURCE_SPECIFICATION_VERSION,
        "kind": privacy.PUBLICATION_SOURCE_SPECIFICATION_KIND,
        "candidate_evidence": candidate,
        "policy_sha256": hashlib.sha256(
            privacy.canonical_json_bytes(policy)
        ).hexdigest(),
        "scope_order": list(privacy.TECHNICAL_SCOPE_ORDER),
        "scope_artifacts": scope_artifacts,
        "runtime_receipt_order": list(privacy.RUNTIME_RECEIPT_ORDER),
        "runtime_receipts": runtime_records,
        "sources": sources,
    }
    path = root / "publication-source-spec.json"
    path.write_bytes(privacy.binding_canonical_json_bytes(document))
    return path, runtime_paths, document


def _completion_bundle_fixture(bundle_root, source_root, document):
    """Build the minimal index topology consumed by the production binding adapter."""

    bundle_root = Path(bundle_root)
    source_root = Path(source_root)
    bundle_root.mkdir()
    candidate = document["candidate_evidence"]
    payloads = []

    def copy_payload(source, bundle_path, media_type):
        destination = bundle_root.joinpath(*bundle_path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        raw = destination.read_bytes()
        descriptor = {
            "path": bundle_path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "media_type": media_type,
            "candidate_evidence": candidate,
        }
        payloads.append(descriptor)
        return descriptor

    manifest_source = source_root / "synthetic-manifest.json"
    manifest_source.write_bytes(b"{}\n")
    manifest = copy_payload(
        manifest_source,
        "manifest.json",
        completion.MEDIA_TYPE_JSON,
    )
    for record in document["runtime_receipts"]:
        descriptor = copy_payload(
            source_root / record["source_path"],
            record["bundle_path"],
            completion.MEDIA_TYPE_JSON,
        )
        assert descriptor["sha256"] == record["sha256"]
        assert descriptor["size_bytes"] == record["size_bytes"]
    source_descriptors = {}
    for record in document["sources"]:
        descriptor = copy_payload(
            source_root / record["source_path"],
            record["bundle_path"],
            record["media_type"],
        )
        source_descriptors[record["completion_role"]] = descriptor
    for scope in privacy.TECHNICAL_SCOPE_ORDER:
        for ordinal, descriptor in document["scope_artifacts"][scope][
            "production-source"
        ].items():
            role = f"/artifacts/{scope}/production-source/{ordinal}"
            assert source_descriptors[role] == descriptor
    payloads.sort(key=lambda descriptor: descriptor["path"])
    index = {
        "schema_version": completion.INDEX_SCHEMA_VERSION,
        "kind": completion.INDEX_KIND,
        "issue": 123,
        "bundle": {
            "format": completion.BUNDLE_FORMAT,
            "path_contract": completion.PATH_CONTRACT,
            "artifact_count": len(payloads),
            "artifact_bytes": sum(item["size_bytes"] for item in payloads),
        },
        "candidate_evidence": candidate,
        "manifest": manifest,
        "payloads": payloads,
        "artifacts": {
            **copy.deepcopy(document["scope_artifacts"]),
            "operations": {},
        },
    }
    index_path = bundle_root / "completion-index.json"
    index_path.write_bytes(completion._canonical_json_bytes(index))
    return index_path


def _zip_bytes(entries, *, symlink=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in entries:
            if symlink:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, raw)
            else:
                archive.writestr(name, raw)
    return stream.getvalue()


def _pax_record(key, value):
    payload = f"{key}={value}\n".encode("utf-8")
    length = len(payload) + 2
    while True:
        record = str(length).encode("ascii") + b" " + payload
        if len(record) == length:
            return record
        length = len(record)


def _physical_tar_bytes(records):
    raw = bytearray()
    for name, type_byte, body in records:
        info = tarfile.TarInfo(name)
        info.type = type_byte
        info.size = len(body)
        raw.extend(info.tobuf(format=tarfile.USTAR_FORMAT))
        raw.extend(body)
        raw.extend(b"\0" * (-len(body) % 512))
    raw.extend(b"\0" * 1024)
    return bytes(raw)


def _pax_helper(type_byte, items):
    body = b"".join(_pax_record(key, value) for key, value in items)
    return ("././@PaxHeader", type_byte, body)


def _ordinary_tar_record(name="package/data.bin", body=b"safe"):
    return (name, tarfile.REGTYPE, body)


def _gzip_bytes(raw):
    compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
    return compressor.compress(raw) + compressor.flush()


class Issue123PrivacyProjectionTest(unittest.TestCase):
    def setUp(self):
        self.policy, self.private = _fixture()
        self.salt = bytes(range(32))
        self._production_project_publication = privacy.project_publication

        def deterministic_project_publication(
            private_bundle, policy, *, salt=None, private_openings=None
        ):
            if salt is None:
                return self._production_project_publication(
                    private_bundle,
                    policy,
                    private_openings=private_openings,
                )
            with mock.patch.object(privacy.secrets, "token_bytes", return_value=salt):
                return self._production_project_publication(
                    private_bundle,
                    policy,
                    private_openings=private_openings,
                )

        project_patcher = mock.patch.object(
            privacy, "project_publication", new=deterministic_project_publication
        )
        project_patcher.start()
        self.addCleanup(project_patcher.stop)

    def project(self, *, openings=None):
        return privacy.project_publication(
            self.private,
            self.policy,
            salt=self.salt,
            private_openings=openings,
        )

    def test_synthetic_literal_binding_scaffold_is_deterministic_and_private(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            adapter_policy = copy.deepcopy(self.policy)
            adapter_private = copy.deepcopy(self.private)
            for policy_scope, private_scope in zip(
                adapter_policy["scopes"],
                adapter_private["scopes"],
                strict=True,
            ):
                policy_scope["correctness"] = []
                private_scope["correctness"] = []
            specification_path, runtime_paths, specification_document = (
                _production_source_fixture(root, adapter_policy, adapter_private)
            )
            completion_index = _completion_bundle_fixture(
                root / "completion-bundle",
                root,
                specification_document,
            )
            policy_sha256 = hashlib.sha256(
                privacy.canonical_json_bytes(adapter_policy)
            ).hexdigest()
            literal_bindings = {}
            for target, record in zip(
                privacy._expected_source_targets(adapter_policy),
                specification_document["sources"],
                strict=True,
            ):
                key = privacy._target_key(target)
                literal_bindings[key] = privacy.EvaluatorTargetBinding(
                    target_key=key,
                    primary=privacy.RoleSelector(
                        record["completion_role"],
                        copy.deepcopy(record["selector"]),
                    ),
                )
            with mock.patch.object(
                privacy,
                "CODE_OWNED_LITERAL_TARGET_BINDINGS",
                literal_bindings,
            ):
                specification = privacy.load_publication_source_spec(
                    specification_path,
                    adapter_policy,
                    completion_index=completion_index,
                    policy_sha256=policy_sha256,
                )
            materialized = privacy.materialize_publication_inputs(
                specification,
                runtime_receipt_paths=runtime_paths,
            )
            self.assertEqual(
                [scope["name"] for scope in materialized.private_bundle["scopes"]],
                list(privacy.TECHNICAL_SCOPE_ORDER),
            )
            first_openings = privacy.PrivateOpenings(self.salt)
            second_openings = privacy.PrivateOpenings(self.salt)
            first = privacy.project_publication(
                materialized.private_bundle,
                adapter_policy,
                private_openings=first_openings,
            )
            second = privacy.project_publication(
                materialized.private_bundle,
                adapter_policy,
                private_openings=second_openings,
            )
            self.assertEqual(first, second)
            ledger = [
                {
                    "role": "technical_evidence",
                    "name": "safe.bin",
                    "size_bytes": 1,
                    "sha256": "c" * 64,
                }
            ]
            context = privacy.publication_binding_context(
                materialized,
                first,
                ledger,
            )
            protected = privacy.serialize_private_openings(first_openings, context)
            self.assertNotIn(b"source_path", protected)
            self.assertNotIn(b"private-host", protected)
            self.assertNotIn(struct.pack("<2d", 1.0, 2.0), protected)
            private_directory = root / "authority"
            private_directory.mkdir(mode=0o700)
            protected_path = private_directory / "openings.json"
            protected_path.write_bytes(protected)
            protected_path.chmod(0o600)
            loaded_context = privacy.verify_private_openings(
                protected_path,
                context,
            )
            self.assertEqual(loaded_context, context)
            self.assertEqual(stat.S_IMODE(protected_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(private_directory.stat().st_mode), 0o700)

            legacy = copy.deepcopy(specification_document)
            legacy["schema_version"] = 0
            legacy_path = root / "legacy-source-spec.json"
            legacy_path.write_bytes(privacy.binding_canonical_json_bytes(legacy))
            with self.assertRaisesRegex(privacy.PrivacyError, "identity"):
                privacy.load_publication_source_spec(
                    legacy_path,
                    adapter_policy,
                    completion_index=completion_index,
                )

            first_source = root / specification_document["sources"][0]["source_path"]
            first_source.write_bytes(first_source.read_bytes() + b" ")
            with self.assertRaisesRegex(privacy.PrivacyError, "bytes differ"):
                privacy.materialize_publication_inputs(
                    specification,
                    runtime_receipt_paths=runtime_paths,
                )

            substituted = json.loads(protected)
            substituted["binding"]["value"] = "0" * 64
            with self.assertRaisesRegex(privacy.PrivacyError, "authentication"):
                privacy.load_private_openings(
                    privacy.binding_canonical_json_bytes(substituted)
                )
            with self.assertRaisesRegex(privacy.PrivacyError, "unavailable"):
                privacy.load_private_openings(private_directory / "missing.json")

    def test_production_binding_profile_is_fail_closed_until_owner_profile_exists(
        self,
    ):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            policy = copy.deepcopy(self.policy)
            private = copy.deepcopy(self.private)
            for policy_scope, private_scope in zip(
                policy["scopes"], private["scopes"], strict=True
            ):
                policy_scope["correctness"] = []
                private_scope["correctness"] = []
            specification_path, _runtime_paths, document = _production_source_fixture(
                root,
                policy,
                private,
            )
            completion_index = _completion_bundle_fixture(
                root / "completion-bundle",
                root,
                document,
            )
            self.assertEqual(dict(privacy.CODE_OWNED_LITERAL_TARGET_BINDINGS), {})
            with self.assertRaisesRegex(
                privacy.PrivacyError,
                "no code-owned evaluator binding",
            ):
                privacy.load_publication_source_spec(
                    specification_path,
                    policy,
                    completion_index=completion_index,
                    policy_sha256=hashlib.sha256(
                        privacy.canonical_json_bytes(policy)
                    ).hexdigest(),
                )

    def test_synthetic_binding_scaffold_rejects_same_scope_decoy_rebuild(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            policy = copy.deepcopy(self.policy)
            private = copy.deepcopy(self.private)
            for ordinal, (policy_scope, private_scope) in enumerate(
                zip(policy["scopes"], private["scopes"], strict=True)
            ):
                if ordinal:
                    policy_scope["correctness"] = []
                    private_scope["correctness"] = []
            specification_path, runtime_paths, document = _production_source_fixture(
                root, policy, private
            )
            candidate = document["candidate_evidence"]
            correctness_records = [
                record
                for record in document["sources"]
                if record["scope"] == "cpu"
                and record["semantic_role"].startswith("correctness-")
            ]
            self.assertEqual(
                [record["semantic_role"] for record in correctness_records],
                ["correctness-reference", "correctness-candidate"],
            )
            reference_record, candidate_record = correctness_records

            def record_descriptor(record):
                return {
                    "path": record["bundle_path"],
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                    "media_type": record["media_type"],
                    "candidate_evidence": candidate,
                }

            extra_payloads = []

            def add_payload(bundle_path, raw, media_type):
                source = root / bundle_path.replace("/", "-")
                source.write_bytes(raw)
                descriptor = {
                    "path": bundle_path,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                    "media_type": media_type,
                    "candidate_evidence": candidate,
                }
                extra_payloads.append((source, descriptor))
                return descriptor

            def npz_payload(bundle_path, values):
                source = root / bundle_path.replace("/", "-")
                np.savez(source, field=np.asarray(values, dtype="<f8"))
                raw = source.read_bytes()
                source.unlink()
                return add_payload(bundle_path, raw, "application/x-npz")

            cuda_eager_candidate = npz_payload(
                "evaluator/cuda-eager-candidate.npz",
                [1.0 + 3e-10, 2.0],
            )
            cuda_graph_candidate = npz_payload(
                "evaluator/cuda-graph-candidate.npz",
                [1.0 + 4e-10, 2.0],
            )
            same_scope_decoy = npz_payload(
                "evaluator/cpu-same-scope-decoy.npz",
                [1.0 + 2e-10, 2.0],
            )
            reference_descriptor = record_descriptor(reference_record)
            candidate_descriptor = record_descriptor(candidate_record)
            cpu_index = add_payload(
                "evaluator/cpu-correctness-index.json",
                privacy.binding_canonical_json_bytes(
                    {
                        "artifacts": [
                            {
                                "case": "case-cpu",
                                "reference": reference_descriptor,
                                "candidate": candidate_descriptor,
                            }
                        ]
                    }
                ),
                "application/json",
            )
            cuda_indexes = []
            for mode, mode_candidate in enumerate(
                (cuda_eager_candidate, cuda_graph_candidate)
            ):
                cuda_indexes.append(
                    add_payload(
                        f"evaluator/cuda-{mode}-correctness-index.json",
                        privacy.binding_canonical_json_bytes(
                            {
                                "artifacts": [
                                    {
                                        "case": "case-cpu",
                                        "reference": reference_descriptor,
                                        "candidate": mode_candidate,
                                    }
                                ]
                            }
                        ),
                        "application/json",
                    )
                )
            cuda_gates = add_payload(
                "evaluator/cuda-gates.json",
                privacy.binding_canonical_json_bytes(
                    {
                        "cuda_suite_gate": {
                            "correctness_indexes": [
                                {"source_artifact": descriptor}
                                for descriptor in cuda_indexes
                            ]
                        }
                    }
                ),
                "application/json",
            )
            cpu_sources = document["scope_artifacts"]["cpu"]["production-source"]
            for record in correctness_records:
                del cpu_sources[record["completion_role"].rsplit("/", 1)[1]]
            document["scope_artifacts"]["cpu"].update(
                correctness_index=cpu_index,
                same_scope_decoy=same_scope_decoy,
            )
            document["scope_artifacts"]["single_gpu"]["cuda_gates"] = cuda_gates
            reference_role = (
                "/artifacts/cpu/correctness_index/@document/artifacts/0/reference"
            )
            candidate_role = (
                "/artifacts/cpu/correctness_index/@document/artifacts/0/candidate"
            )
            reference_record["completion_role"] = reference_role
            candidate_record["completion_role"] = candidate_role
            specification_path.write_bytes(
                privacy.binding_canonical_json_bytes(document)
            )
            completion_index = _completion_bundle_fixture(
                root / "completion-b1",
                root,
                document,
            )
            completion_document = completion._strict_json_bytes(
                completion_index.read_bytes(),
                "synthetic completion B1",
            )
            for source, descriptor in extra_payloads:
                destination = completion_index.parent.joinpath(
                    *descriptor["path"].split("/")
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                completion_document["payloads"].append(descriptor)
            completion_document["payloads"].sort(key=lambda item: item["path"])
            completion_document["bundle"]["artifact_count"] = len(
                completion_document["payloads"]
            )
            completion_document["bundle"]["artifact_bytes"] = sum(
                item["size_bytes"] for item in completion_document["payloads"]
            )
            completion_index.write_bytes(
                completion._canonical_json_bytes(completion_document)
            )

            literal_bindings = {}
            for target, record in zip(
                privacy._expected_source_targets(policy),
                document["sources"],
                strict=True,
            ):
                if target["semantic_role"].startswith("correctness-"):
                    continue
                key = privacy._target_key(target)
                literal_bindings[key] = privacy.EvaluatorTargetBinding(
                    target_key=key,
                    primary=privacy.RoleSelector(
                        record["completion_role"],
                        copy.deepcopy(record["selector"]),
                    ),
                )
            policy_sha256 = hashlib.sha256(
                privacy.canonical_json_bytes(policy)
            ).hexdigest()
            with mock.patch.object(
                privacy,
                "CODE_OWNED_LITERAL_TARGET_BINDINGS",
                literal_bindings,
            ):
                specification = privacy.load_publication_source_spec(
                    specification_path,
                    policy,
                    completion_index=completion_index,
                    policy_sha256=policy_sha256,
                )
                materialized = privacy.materialize_publication_inputs(
                    specification,
                    runtime_receipt_paths=runtime_paths,
                )
                base_openings = privacy.PrivateOpenings(bytes(range(32)))
                base_projection = privacy.project_publication(
                    materialized.private_bundle,
                    policy,
                    private_openings=base_openings,
                )

                def public_outputs(projection):
                    built = publication.build_publication_assets(
                        projection,
                        expected_policy=policy,
                        expected_bindings=policy["bindings"],
                    )
                    assets = {
                        role: built[asset_name]
                        for role, asset_name in publication.ASSET_ORDER
                    }
                    ledger = [
                        {
                            "role": role,
                            "name": asset_name,
                            "size_bytes": len(assets[role]),
                            "sha256": hashlib.sha256(assets[role]).hexdigest(),
                        }
                        for role, asset_name in publication.ASSET_ORDER
                    ]
                    return assets, ledger

                base_assets, base_ledger = public_outputs(base_projection)
                base_context = privacy.publication_binding_context(
                    materialized,
                    base_projection,
                    base_ledger,
                )
                authority = root / "authority"
                authority.mkdir(mode=0o700)
                base_openings_path = authority / "base-openings.json"
                privacy.write_private_authority_file(
                    base_openings_path,
                    privacy.serialize_private_openings(
                        base_openings,
                        base_context,
                    ),
                    label="synthetic base openings",
                )
                _reader, _index, catalog, _raw = privacy._completion_source_state(
                    completion_index,
                    base_context,
                )
                shared_roles = (
                    reference_role,
                    "/artifacts/single_gpu/cuda_gates/@document/"
                    "cuda_suite_gate/correctness_indexes/0/source_artifact/"
                    "@document/artifacts/0/reference",
                    "/artifacts/single_gpu/cuda_gates/@document/"
                    "cuda_suite_gate/correctness_indexes/1/source_artifact/"
                    "@document/artifacts/0/reference",
                )
                self.assertTrue(
                    all(
                        catalog[role] == catalog[shared_roles[0]]
                        for role in shared_roles
                    )
                )

                def structurally_valid_b1(
                    _index_path,
                    _manifest_path,
                    _runtime_receipts,
                    *,
                    descriptor_access_log=None,
                ):
                    assert descriptor_access_log is not None
                    descriptor_access_log.extend(
                        {
                            "label": "synthetic evaluator access",
                            "descriptor": descriptor,
                        }
                        for descriptor in catalog.values()
                    )
                    return {
                        "scopes": {
                            scope: {"satisfied": True}
                            for scope in privacy.TECHNICAL_SCOPE_ORDER
                        }
                    }

                with mock.patch.object(
                    completion,
                    "evaluate_completion",
                    side_effect=structurally_valid_b1,
                ):
                    verified = privacy.verify_publication_bundle_binding(
                        index_path=completion_index,
                        protected_openings=base_openings_path,
                        policy=policy,
                        public_assets=base_assets,
                        runtime_receipt_paths=runtime_paths,
                        manifest_path=root / "synthetic-manifest.json",
                    )
                self.assertTrue(verified["first_five_scopes_validated"])

                attack_roles = (
                    "/artifacts/cpu/same_scope_decoy",
                    "/artifacts/single_gpu/cuda_gates/@document/"
                    "cuda_suite_gate/correctness_indexes/0/source_artifact/"
                    "@document/artifacts/0/candidate",
                )
                decoy_candidate_bytes = np.asarray(
                    [1.0 + 2e-10, 2.0], dtype="<f8"
                ).tobytes()
                for attack_ordinal, attack_role in enumerate(attack_roles):
                    attack_descriptor = catalog[attack_role]
                    attack_inventory = copy.deepcopy(materialized.technical_inventory)
                    attack_record = next(
                        item
                        for item in attack_inventory["sources"]
                        if item["scope"] == "cpu"
                        and item["semantic_role"] == "correctness-candidate"
                    )
                    attack_record.update(
                        completion_role=attack_role,
                        bundle_path=attack_descriptor["path"],
                        sha256=attack_descriptor["sha256"],
                        size_bytes=attack_descriptor["size_bytes"],
                        media_type=attack_descriptor["media_type"],
                    )
                    attack_private = copy.deepcopy(materialized.private_bundle)
                    attack_private["scopes"][0]["correctness"][0]["captures"][0][
                        "arrays"
                    ][0]["candidate_bytes"] = decoy_candidate_bytes
                    attack_materialized = privacy.MaterializedPublicationInputs(
                        private_bundle=attack_private,
                        technical_inventory=attack_inventory,
                        technical_input_root=privacy.tagged_canonical_sha256(
                            privacy.TECHNICAL_INPUT_INVENTORY_DOMAIN,
                            attack_inventory,
                        ),
                        source_specification_sha256=hashlib.sha256(
                            privacy.binding_canonical_json_bytes(attack_inventory)
                        ).hexdigest(),
                    )
                    attack_openings = privacy.PrivateOpenings(bytes(range(32)))
                    attack_projection = privacy.project_publication(
                        attack_private,
                        policy,
                        private_openings=attack_openings,
                    )
                    attack_assets, attack_ledger = public_outputs(attack_projection)
                    attack_context = privacy.publication_binding_context(
                        attack_materialized,
                        attack_projection,
                        attack_ledger,
                    )
                    attack_openings_path = (
                        authority / f"attack-{attack_ordinal}-openings.json"
                    )
                    privacy.write_private_authority_file(
                        attack_openings_path,
                        privacy.serialize_private_openings(
                            attack_openings,
                            attack_context,
                        ),
                        label="synthetic attack openings",
                    )
                    with (
                        mock.patch.object(
                            completion,
                            "evaluate_completion",
                            side_effect=structurally_valid_b1,
                        ),
                        self.subTest(attack_role=attack_role),
                        self.assertRaisesRegex(
                            privacy.PrivacyError,
                            "publication source evaluator role assertion differs",
                        ),
                    ):
                        privacy.verify_publication_bundle_binding(
                            index_path=completion_index,
                            protected_openings=attack_openings_path,
                            policy=policy,
                            public_assets=attack_assets,
                            runtime_receipt_paths=runtime_paths,
                            manifest_path=root / "synthetic-manifest.json",
                        )
                    public_raw = b"".join(attack_assets.values())
                    self.assertNotIn(bytes(range(32)).hex().encode(), public_raw)
                    self.assertNotIn(decoy_candidate_bytes, public_raw)

    def test_binding_canonicalization_domains_and_receipt_links_have_fixed_vectors(
        self,
    ):
        value = {"alpha": [1, True, None], "é": {"nested": "ok"}}
        expected_raw = b'{"alpha":[1,true,null],"\xc3\xa9":{"nested":"ok"}}\n'
        self.assertEqual(privacy.binding_canonical_json_bytes(value), expected_raw)
        self.assertEqual(
            privacy.tagged_canonical_sha256(
                privacy.TECHNICAL_INPUT_INVENTORY_DOMAIN,
                value,
            ),
            "eb42d8510ac9c4a51276f193a291a325a3e7d10ac02503c917c2628e14dfc55e",
        )
        self.assertEqual(
            privacy._tagged_hmac(
                bytes(range(32)),
                privacy.PRIVATE_OPENING_BINDING_DOMAIN,
                value,
            ),
            "bd91e96f364e430bb19c92f48b43e9f02dd057edf681e217babcc81496e4167f",
        )
        checked = {
            "body": (
                "## Implementation work\n"
                "- [x] publish the final bundle\n"
                "- [x] complete the post-bundle checklist\n"
            ),
            "updated_at": "2026-01-01T00:00:00Z",
        }
        from benchmarks import issue123_operations as operations

        self.assertEqual(
            operations.checklist_transition_sha256(checked, "checked"),
            "f62d341750742410bbadf27495f924dbc2be6b5d7b5c5cb918e2829aefae54e7",
        )
        receipt_raw = b'{"kind":"synthetic-receipt","schema_version":1}\n'
        self.assertEqual(
            completion._reopen_receipt_file_sha256(receipt_raw),
            "29f55bda72d534eb1dac97d507e258515816779f77a9957d71e7998114146016",
        )
        self.assertFalse(hasattr(completion, "BUNDLE_REOPEN_RECEIPT_DOMAIN"))
        self.assertNotEqual(
            privacy.tagged_canonical_sha256(
                privacy.TECHNICAL_INPUT_INVENTORY_DOMAIN + "-changed",
                value,
            ),
            "eb42d8510ac9c4a51276f193a291a325a3e7d10ac02503c917c2628e14dfc55e",
        )
        for invalid in (
            {"e\u0301": 1},
            {"é": 1, "e\u0301": 2},
            {"outer": {"é": 1, "e\u0301": 2}},
            {1: "non-string"},
            {"safe": "e\u0301"},
        ):
            with (
                self.subTest(invalid=repr(invalid)),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.binding_canonical_json_bytes(invalid)
            self.assertNotIn("e\u0301", str(caught.exception))
            self.assertNotIn("é", str(caught.exception))

    def test_private_authority_writer_is_atomic_private_and_symlink_closed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            private_root = root / "private"
            private_root.mkdir(mode=0o700)
            callbacks = []
            with mock.patch.object(
                privacy.os,
                "fsync",
                wraps=privacy.os.fsync,
            ) as fsync:
                final = privacy.write_private_authority_file(
                    private_root / "authority.json",
                    b"complete-authority\n",
                    label="synthetic private authority",
                    before_commit=lambda: callbacks.append("checked"),
                )
            self.assertEqual(final.read_bytes(), b"complete-authority\n")
            self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o600)
            self.assertEqual(callbacks, ["checked"])
            self.assertGreaterEqual(fsync.call_count, 2)
            self.assertEqual(list(private_root.glob(".*.tmp-*")), [])

            callback_root = root / "callback-failure"
            callback_root.mkdir(mode=0o700)
            sentinel = callback_root / ".result.json.tmp-sentinel"
            sentinel.write_bytes(b"unrelated")

            def reject_commit():
                raise RuntimeError("private-callback-marker")

            with self.assertRaises(privacy.PrivacyError) as caught:
                privacy.write_private_authority_file(
                    callback_root / "result.json",
                    b"partial-must-not-publish",
                    label="synthetic callback authority",
                    before_commit=reject_commit,
                )
            self.assertEqual(
                str(caught.exception), "private authority file could not be committed"
            )
            self.assertNotIn(
                "private-callback-marker",
                " ".join(_exception_messages(caught.exception)),
            )
            self.assertFalse((callback_root / "result.json").exists())
            self.assertEqual(sentinel.read_bytes(), b"unrelated")
            self.assertEqual(
                list(callback_root.glob(".result.json.tmp-*")),
                [sentinel],
            )

            partial_root = root / "partial-write"
            partial_root.mkdir(mode=0o700)
            real_write = privacy.os.write
            writes = 0

            def partial_then_fail(descriptor, view):
                nonlocal writes
                writes += 1
                if writes == 1:
                    return real_write(descriptor, bytes(view[:1]))
                raise OSError("private-write-marker")

            with (
                mock.patch.object(privacy.os, "write", side_effect=partial_then_fail),
                self.assertRaises(privacy.PrivacyError) as partial_error,
            ):
                privacy.write_private_authority_file(
                    partial_root / "result.json",
                    b"complete-value",
                    label="synthetic partial authority",
                )
            self.assertFalse((partial_root / "result.json").exists())
            self.assertEqual(list(partial_root.iterdir()), [])
            self.assertNotIn(
                "private-write-marker",
                " ".join(_exception_messages(partial_error.exception)),
            )

            symlink_target = root / "symlink-target"
            symlink_target.mkdir(mode=0o700)
            ancestor = root / "linked-private"
            ancestor.symlink_to(symlink_target, target_is_directory=True)
            with self.assertRaisesRegex(privacy.PrivacyError, "symbolic link"):
                privacy.write_private_authority_file(
                    ancestor / "result.json",
                    b"must-not-follow",
                    label="synthetic symlink authority",
                )
            self.assertFalse((symlink_target / "result.json").exists())
            with self.assertRaisesRegex(privacy.PrivacyError, "path is invalid"):
                privacy.write_private_authority_file(
                    None,
                    b"invalid-path",
                    label="synthetic invalid authority",
                )
            with self.assertRaisesRegex(privacy.PrivacyError, "forbidden root"):
                privacy.write_private_authority_file(
                    private_root / "forbidden.json",
                    b"forbidden",
                    label="synthetic forbidden authority",
                    forbidden_roots=(private_root,),
                )
            existing = private_root / "existing.json"
            existing.write_bytes(b"sentinel")
            with self.assertRaisesRegex(privacy.PrivacyError, "could not be committed"):
                privacy.write_private_authority_file(
                    existing,
                    b"replacement",
                    label="synthetic no-replace authority",
                )
            self.assertEqual(existing.read_bytes(), b"sentinel")

            unsupported_root = root / "unsupported-descriptor-stat"
            unsupported_root.mkdir(mode=0o700)
            supports_fd_without_stat = set(privacy.os.supports_fd) - {privacy.os.stat}
            with (
                mock.patch.object(
                    privacy.os,
                    "supports_fd",
                    supports_fd_without_stat,
                ),
                self.assertRaisesRegex(
                    privacy.PrivacyError,
                    "atomic publication is unsupported",
                ),
            ):
                privacy.write_private_authority_file(
                    unsupported_root / "result.json",
                    b"unsupported",
                    label="synthetic unsupported authority",
                )
            self.assertEqual(list(unsupported_root.iterdir()), [])

    def test_private_writer_first_fstat_cleanup_is_identity_scoped(self):
        real_fstat = privacy.os.fstat
        real_stat = privacy.os.stat
        for attack in ("persistent-no-replacement", "persistent-with-replacement"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as name:
                parent = Path(name) / "private"
                parent.mkdir(mode=0o700)
                final = parent / "result.json"
                sentinel = parent / ".result.json.tmp-sentinel"
                sentinel.write_bytes(b"unrelated-sentinel")
                sentinel_identity = (sentinel.stat().st_dev, sentinel.stat().st_ino)
                target_descriptor = None
                target_fstat_failures = 0
                replacement_identity = None
                commit_calls = []
                canary = "synthetic-persistent-fstat-failure"

                def fail_persistently_for_temp(descriptor):
                    nonlocal target_descriptor
                    nonlocal target_fstat_failures
                    nonlocal replacement_identity
                    metadata = real_fstat(descriptor)
                    if (
                        target_descriptor is None
                        and stat.S_ISREG(metadata.st_mode)
                        and stat.S_IMODE(metadata.st_mode) == 0o600
                    ):
                        target_descriptor = descriptor
                    if descriptor == target_descriptor:
                        target_fstat_failures += 1
                        if (
                            target_fstat_failures == 1
                            and attack == "persistent-with-replacement"
                        ):
                            temporary = next(
                                path
                                for path in parent.glob(".result.json.tmp-*")
                                if path != sentinel
                            )
                            temporary.unlink()
                            temporary.write_bytes(b"unrelated-replacement")
                            replacement = real_stat(
                                temporary,
                                follow_symlinks=False,
                            )
                            replacement_identity = (
                                replacement.st_dev,
                                replacement.st_ino,
                            )
                        raise OSError(canary)
                    return metadata

                with (
                    mock.patch.object(
                        privacy.os,
                        "fstat",
                        side_effect=fail_persistently_for_temp,
                    ),
                    self.assertRaises(privacy.PrivacyError) as caught,
                ):
                    privacy.write_private_authority_file(
                        final,
                        b"candidate-authority",
                        label="synthetic first-fstat authority",
                        before_commit=lambda: commit_calls.append("committed"),
                    )
                self.assertEqual(
                    str(caught.exception),
                    "private authority file could not be committed",
                )
                diagnostics = " ".join(
                    (
                        str(caught.exception),
                        repr(caught.exception),
                        repr(caught.exception.__cause__),
                        repr(caught.exception.__context__),
                        "".join(traceback.format_exception(caught.exception)),
                    )
                )
                self.assertNotIn(canary, diagnostics)
                self.assertFalse(final.exists())
                self.assertEqual(commit_calls, [])
                self.assertEqual(target_fstat_failures, 2)
                self.assertIsNotNone(target_descriptor)
                with self.assertRaises(OSError):
                    real_fstat(target_descriptor)
                self.assertEqual(
                    (sentinel.stat().st_dev, sentinel.stat().st_ino),
                    sentinel_identity,
                )
                self.assertEqual(sentinel.read_bytes(), b"unrelated-sentinel")
                owned = [
                    path
                    for path in parent.glob(".result.json.tmp-*")
                    if path != sentinel
                ]
                if attack == "persistent-no-replacement":
                    self.assertEqual(owned, [])
                else:
                    self.assertEqual(len(owned), 1)
                    self.assertEqual(
                        (owned[0].stat().st_dev, owned[0].stat().st_ino),
                        replacement_identity,
                    )
                    self.assertEqual(owned[0].read_bytes(), b"unrelated-replacement")

    def test_private_writer_post_link_failures_report_committed_state(self):
        cases = ("parent-fsync", "final-reopen")
        for attack in cases:
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as name:
                parent = Path(name) / "private"
                parent.mkdir(mode=0o700)
                final = parent / "result.json"
                raw = b"complete-authority\n"
                real_fsync = privacy.os.fsync
                real_fstat = privacy.os.fstat
                directory_fsyncs = 0
                regular_fstats = 0

                def fail_parent_fsync(descriptor):
                    nonlocal directory_fsyncs
                    metadata = privacy.os.fstat(descriptor)
                    if stat.S_ISDIR(metadata.st_mode):
                        directory_fsyncs += 1
                        if directory_fsyncs == 1:
                            raise OSError("synthetic-parent-fsync-failure")
                    return real_fsync(descriptor)

                def fail_final_reopen_fstat(descriptor):
                    nonlocal regular_fstats
                    metadata = real_fstat(descriptor)
                    if stat.S_ISREG(metadata.st_mode):
                        regular_fstats += 1
                    if regular_fstats == 3:
                        raise OSError("synthetic-final-reopen-failure")
                    return metadata

                patcher = (
                    mock.patch.object(
                        privacy.os,
                        "fsync",
                        side_effect=fail_parent_fsync,
                    )
                    if attack == "parent-fsync"
                    else mock.patch.object(
                        privacy.os,
                        "fstat",
                        side_effect=fail_final_reopen_fstat,
                    )
                )
                with (
                    patcher,
                    self.assertRaises(privacy.PrivateAuthorityCommitError) as caught,
                ):
                    privacy.write_private_authority_file(
                        final,
                        raw,
                        label="synthetic committed authority",
                    )
                self.assertTrue(caught.exception.committed)
                self.assertEqual(
                    str(caught.exception),
                    "private authority file was committed but final verification failed",
                )
                self.assertEqual(final.read_bytes(), raw)
                self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o600)
                self.assertEqual(list(parent.glob(".result.json.tmp-*")), [])

    def test_projects_exact_public_schema_without_private_material(self):
        openings = privacy.PrivateOpenings()
        projected = self.project(openings=openings)
        self.assertEqual(
            set(projected),
            {
                "schema_version",
                "kind",
                "bindings",
                "technical_scopes",
                "correctness_commitments",
                "execution_witness",
                "raw_timing",
                "event_profiler",
            },
        )
        self.assertEqual(
            [scope["scope"] for scope in projected["technical_scopes"]],
            list(privacy.TECHNICAL_SCOPE_ORDER),
        )
        self.assertNotIn("operations", repr(projected))
        self.assertNotIn("private-host", repr(projected))
        raw = privacy.canonical_json_bytes(projected)
        self.assertNotIn(self.salt.hex().encode(), raw)
        self.assertNotIn(b"private-host", raw)
        self.assertNotIn(struct.pack("<2d", 1.0, 2.0), raw)
        self.assertEqual(repr(openings), "<PrivateOpenings redacted>")
        self.assertEqual(openings.salt_for_private_verification(), self.salt)
        self.assertEqual(
            openings.identity_for_private_verification("cpu", "machine"),
            {"opaque_private_value": "private-host-0"},
        )

    def test_preserves_timing_samples_exactly_and_recomputes_statistics(self):
        projected = self.project()
        timing = projected["technical_scopes"][0]["timings"][0]
        self.assertEqual(timing["samples"], [0.003, 0.001, 0.002])
        self.assertEqual(timing["sample_count"], 3)
        self.assertEqual(timing["median_seconds"], 0.002)
        self.assertEqual(timing["mad_seconds"], 0.001)
        self.assertEqual(timing["relative_mad"], 0.5)
        self.assertEqual(
            projected["raw_timing"]["records"][0]["samples"],
            [0.003, 0.001, 0.002],
        )

    def test_trace_is_event_complete_local_and_dense(self):
        trace = self.project()["technical_scopes"][0]["traces"][0]
        events = trace["events"]
        self.assertEqual(trace["clock"], privacy.LOCAL_CLOCK)
        self.assertEqual([event["ordinal"] for event in events], list(range(9)))
        self.assertEqual(events[0]["start_us"], 0)
        self.assertEqual(events[1]["start_us"], 0.0)
        self.assertEqual(sorted({event["process_ordinal"] for event in events}), [0, 1])
        self.assertEqual(
            sorted(
                {
                    event["stream_ordinal"]
                    for event in events
                    if event["stream_ordinal"] is not None
                }
            ),
            [0, 1],
        )
        self.assertNotIn("9017", repr(trace))
        self.assertNotIn("stream-17", repr(trace))
        self.assertNotIn("allocation-address", repr(trace))
        self.assertEqual(events[1]["live_allocated_bytes"], 64)
        self.assertEqual(events[2]["live_allocated_bytes"], 0)
        self.assertIsNone(events[3]["live_allocated_bytes"])
        summary = trace["summary"]
        self.assertEqual(summary["allocation_net_bytes"], 0)
        self.assertEqual(summary["live_allocation_growth_bytes"], 0)
        self.assertEqual(summary["kernel_launches"], 2)
        self.assertEqual(summary["cuda_graph_launches"], 1)
        self.assertEqual(summary["nccl_compute_overlap_us"], 5.0)
        self.assertEqual(summary["overlap_fraction"], 0.5)

    def test_correctness_publishes_arithmetic_and_commitments_only(self):
        document = self.project()["correctness_commitments"]
        array = document["cases"][0]["captures"][0]["arrays"][0]
        self.assertTrue(array["comparison"]["passed"])
        self.assertFalse(array["comparison"]["reference_all_zero"])
        self.assertTrue(array["comparison"]["zero_reference_exact"])
        self.assertRegex(array["commitments"]["reference"], r"[0-9a-f]{64}\Z")
        self.assertRegex(array["commitments"]["candidate"], r"[0-9a-f]{64}\Z")
        self.assertEqual(document["closure"]["case_count"], 5)
        self.assertEqual(document["closure"]["capture_count"], 5)
        self.assertEqual(document["closure"]["array_count"], 5)
        self.assertEqual(len(document["closure"]["inventory_sha256"]), 64)
        self.assertNotIn("reference_bytes", repr(document))
        self.assertNotIn("candidate_bytes", repr(document))

    def test_execution_witness_binds_claims_traces_and_exact_job_identity(self):
        projected = self.project()
        witness = projected["execution_witness"]
        self.assertEqual(set(witness), {"schema_version", "kind", "bindings", "claims"})
        self.assertEqual(witness["kind"], privacy.EXECUTION_WITNESS_KIND)
        self.assertEqual(privacy.EXECUTION_WITNESS_PATH, "execution/witness.json")
        self.assertEqual(
            [claim["claim"] for claim in witness["claims"]],
            list(privacy.EXECUTION_CLAIM_ORDER),
        )
        scopes = {scope["scope"]: scope for scope in projected["technical_scopes"]}
        for claim in witness["claims"]:
            trace = next(
                trace
                for trace in scopes[claim["scope"]]["traces"]
                if trace["name"] == claim["trace_name"]
            )
            signatures = [
                [event["semantic_token"], event["phase"]] for event in trace["events"]
            ]
            normalized_trace = {
                "clock": trace["clock"],
                "events": trace["events"],
                "summary": trace["summary"],
            }
            self.assertEqual(
                claim["semantic_inventory_sha256"],
                _canonical_sha256(signatures),
            )
            self.assertEqual(
                claim["normalized_trace_sha256"],
                _canonical_sha256(normalized_trace),
            )
            self.assertEqual(claim["event_count"], len(trace["events"]))
            self.assertEqual(claim["validation_workflow"], "CI")
            self.assertEqual(claim["validator_job"], self.policy["bindings"]["jobs"][0])
        self.assertNotIn("private-cpu-worker", repr(witness))
        self.assertNotIn("private-cuda-worker", repr(witness))

    def test_execution_claims_reject_incompatible_semantics_and_job_rebinding(self):
        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        policy["scopes"][0]["traces"][1] = _witness_trace_policy(
            "cpu-eager-witness",
            [
                ["metadata-process-name", "metadata"],
                ["cuda-runtime", "complete"],
                ["kernel", "complete"],
            ],
            kernel_launches=1,
        )
        private["scopes"][0]["traces"][1]["trace_bytes"] = _cuda_eager_trace_bytes()
        with self.assertRaisesRegex(privacy.PrivacyError, "cpu-eager witness"):
            privacy.project_publication(private, policy, salt=self.salt)

        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        policy["scopes"][0]["traces"][1] = _witness_trace_policy(
            "cpu-eager-witness",
            [
                ["metadata-process-name", "metadata"],
                ["indexed-write-index-copy", "complete"],
                ["memset", "complete"],
            ],
        )
        cpu_trace = json.loads(_cpu_eager_trace_bytes())
        cpu_trace["traceEvents"].append(
            {
                "name": "Memset (Device)",
                "cat": "gpu_memset",
                "ph": "X",
                "ts": 103,
                "dur": 1,
                "pid": 9,
                "tid": 5,
                "args": {},
            }
        )
        private["scopes"][0]["traces"][1]["trace_bytes"] = _json_bytes(cpu_trace)
        with self.assertRaisesRegex(privacy.PrivacyError, "cpu-eager witness"):
            privacy.project_publication(private, policy, salt=self.salt)

        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        single_gpu_index = privacy.TECHNICAL_SCOPE_ORDER.index("single_gpu")
        policy["scopes"][single_gpu_index]["traces"][1] = _witness_trace_policy(
            "cuda-eager-witness",
            [
                ["metadata-process-name", "metadata"],
                ["kernel", "complete"],
            ],
            kernel_launches=1,
        )
        eager_trace = json.loads(_cuda_eager_trace_bytes())
        eager_trace["traceEvents"] = [
            event
            for event in eager_trace["traceEvents"]
            if event["cat"] != "cuda_runtime"
        ]
        private["scopes"][single_gpu_index]["traces"][1]["trace_bytes"] = _json_bytes(
            eager_trace
        )
        with self.assertRaisesRegex(privacy.PrivacyError, "cuda-eager witness"):
            privacy.project_publication(private, policy, salt=self.salt)

        policy = copy.deepcopy(self.policy)
        policy["execution_witnesses"][1]["validator_job_name"] = "CodeQL / python"
        with self.assertRaisesRegex(privacy.PrivacyError, "validator job differs"):
            privacy.project_publication(self.private, policy, salt=self.salt)

        policy = copy.deepcopy(self.policy)
        policy["execution_witnesses"][0], policy["execution_witnesses"][1] = (
            policy["execution_witnesses"][1],
            policy["execution_witnesses"][0],
        )
        with self.assertRaisesRegex(privacy.PrivacyError, "claim order"):
            privacy.project_publication(self.private, policy, salt=self.salt)

    def test_eager_witnesses_reject_coherently_declared_compiled_regions(self):
        compiled_event = {
            "name": "Torch-Compiled Region: 0/0",
            "cat": "user_annotation",
            "ph": "X",
            "ts": 106,
            "dur": 2,
            "pid": 7,
            "tid": 3,
            "args": {},
        }

        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        policy["scopes"][0]["traces"][1] = _witness_trace_policy(
            "cpu-eager-witness",
            [
                ["metadata-process-name", "metadata"],
                ["indexed-write-index-copy", "complete"],
                ["compiled-region", "complete"],
            ],
            compiled_regions=1,
        )
        cpu_trace = json.loads(_cpu_eager_trace_bytes())
        cpu_trace["traceEvents"].append(compiled_event)
        private["scopes"][0]["traces"][1]["trace_bytes"] = _json_bytes(cpu_trace)
        normalized = privacy.normalize_trace(_json_bytes(cpu_trace))
        self.assertEqual(normalized["summary"]["compiled_region_events"], 1)
        with self.assertRaisesRegex(
            privacy.PrivacyError, "cpu-eager witness semantics differ"
        ):
            privacy.project_publication(private, policy, salt=self.salt)

        single_gpu_index = privacy.TECHNICAL_SCOPE_ORDER.index("single_gpu")
        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        policy["scopes"][single_gpu_index]["traces"][1] = _witness_trace_policy(
            "cuda-eager-witness",
            [
                ["metadata-process-name", "metadata"],
                ["cuda-runtime", "complete"],
                ["kernel", "complete"],
                ["compiled-region", "complete"],
            ],
            kernel_launches=1,
            compiled_regions=1,
        )
        cuda_trace = json.loads(_cuda_eager_trace_bytes())
        cuda_trace["traceEvents"].append({**compiled_event, "pid": 8, "tid": 4})
        private["scopes"][single_gpu_index]["traces"][1]["trace_bytes"] = _json_bytes(
            cuda_trace
        )
        normalized = privacy.normalize_trace(_json_bytes(cuda_trace))
        self.assertEqual(normalized["summary"]["compiled_region_events"], 1)
        with self.assertRaisesRegex(
            privacy.PrivacyError, "cuda-eager witness semantics differ"
        ):
            privacy.project_publication(private, policy, salt=self.salt)

    def test_fresh_default_salts_change_low_entropy_commitments(self):
        first = privacy.project_publication(self.private, self.policy)
        second = privacy.project_publication(self.private, self.policy)
        self.assertNotEqual(
            first["technical_scopes"][0]["identity_commitments"][0]["commitment"],
            second["technical_scopes"][0]["identity_commitments"][0]["commitment"],
        )
        self.assertNotEqual(
            first["correctness_commitments"]["cases"][0]["captures"][0]["arrays"][0][
                "commitments"
            ],
            second["correctness_commitments"]["cases"][0]["captures"][0]["arrays"][0][
                "commitments"
            ],
        )

    def test_rejects_invalid_or_reused_private_openings(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'salt'"):
            self._production_project_publication(
                self.private, self.policy, salt=self.salt
            )
        with self.assertRaisesRegex(privacy.PrivacyError, "32 private bytes"):
            with mock.patch.object(
                privacy.secrets, "token_bytes", return_value=b"short"
            ):
                self._production_project_publication(self.private, self.policy)
        openings = privacy.PrivateOpenings()
        self.project(openings=openings)
        with self.assertRaisesRegex(privacy.PrivacyError, "already used"):
            self.project(openings=openings)

    def test_private_sixth_scope_and_operations_fields_are_rejected(self):
        private = copy.deepcopy(self.private)
        private["scopes"].append(
            {
                "name": "operations",
                "satisfied": True,
                "identities": {},
                "timings": [],
                "traces": [],
                "payloads": [],
                "correctness": [],
            }
        )
        with self.assertRaisesRegex(privacy.PrivacyError, "scope closure"):
            privacy.project_publication(private, self.policy, salt=self.salt)
        private = copy.deepcopy(self.private)
        private["operations"] = {}
        with self.assertRaisesRegex(privacy.PrivacyError, "fields differ"):
            privacy.project_publication(private, self.policy, salt=self.salt)

    def test_stale_sha_manifest_run_attempt_and_job_bindings_fail(self):
        mutations = (
            ("final_sha", "c" * 40),
            ("manifest_sha256", "d" * 64),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                private = copy.deepcopy(self.private)
                private["bindings"][field] = replacement
                with self.assertRaisesRegex(privacy.PrivacyError, "stale"):
                    privacy.project_publication(private, self.policy, salt=self.salt)
        for field in ("run_id", "run_attempt", "job_id"):
            with self.subTest(field=field):
                private = copy.deepcopy(self.private)
                private["bindings"]["jobs"][0][field] += 1000
                with self.assertRaises(privacy.PrivacyError):
                    privacy.project_publication(private, self.policy, salt=self.salt)

    def test_descriptor_and_manifest_rehashing_cannot_change_trusted_closure(self):
        private = copy.deepcopy(self.private)
        payload = private["scopes"][0]["payloads"][0]
        payload["bytes"] += b"tampered"
        payload["sha256"] = hashlib.sha256(payload["bytes"]).hexdigest()
        private["bindings"]["manifest_sha256"] = "e" * 64
        with self.assertRaises(privacy.PrivacyError):
            privacy.project_publication(private, self.policy, salt=self.salt)

    def test_timing_companion_metadata_and_nonfinite_samples_fail(self):
        private = copy.deepcopy(self.private)
        private["scopes"][0]["timings"][0]["hostname"] = "secret"
        with self.assertRaisesRegex(privacy.PrivacyError, "fields differ"):
            privacy.project_publication(private, self.policy, salt=self.salt)
        for value in (math.nan, math.inf, 0.0, -1.0, True):
            with self.subTest(value=value):
                private = copy.deepcopy(self.private)
                private["scopes"][0]["timings"][0]["samples"][1] = value
                with self.assertRaises(privacy.PrivacyError):
                    privacy.project_publication(private, self.policy, salt=self.salt)

    def test_missing_extra_or_unknown_trace_events_fail_closed(self):
        cases = []
        missing = _trace_events()[:-1]
        cases.append(missing)
        extra = _trace_events()
        extra.append(copy.deepcopy(extra[-1]))
        extra[-1]["ts"] += 3
        cases.append(extra)
        unknown = _trace_events()
        unknown[-1]["name"] = "free form operation"
        unknown[-1]["cat"] = "user_annotation"
        cases.append(unknown)
        for index, events in enumerate(cases):
            with self.subTest(index=index):
                private = copy.deepcopy(self.private)
                private["scopes"][0]["traces"][0]["trace_bytes"] = _trace_bytes(events)
                with self.assertRaises(privacy.PrivacyError):
                    privacy.project_publication(private, self.policy, salt=self.salt)

    def test_trace_rejects_unknown_fields_args_paths_wall_clock_and_duplicates(self):
        traces = []
        events = _trace_events()
        events[-1]["stack"] = ["frame"]
        traces.append(_trace_bytes(events))
        events = _trace_events()
        events[-1]["args"][
            "Call stack"
        ] = "/Users/fixture-person-invalid/project/file.py"
        traces.append(_trace_bytes(events))
        events = _trace_events()
        events[-1]["name"] = "/tmp/private/kernel"
        traces.append(_trace_bytes(events))
        traces.append(
            _json_bytes(
                {
                    "traceEvents": _trace_events(),
                    "baseTimeNanoseconds": 1_700_000_000_000_000_000,
                }
            )
        )
        traces.append(b'{"traceEvents":[],"traceEvents":[]}')
        for index, raw in enumerate(traces):
            with self.subTest(index=index):
                with self.assertRaises(privacy.PrivacyError):
                    privacy.normalize_trace(raw)

    def test_trace_semantic_invariants_cannot_be_redeclared(self):
        changes = []
        events = _trace_events()
        events[2]["args"]["Bytes"] = -32
        events[2]["args"]["Total Allocated"] = 32
        changes.append(events)
        events = _trace_events()
        events[-1]["name"] = "Graph break: secret reason"
        events[-1]["cat"] = "user_annotation"
        changes.append(events)
        events = _trace_events()
        events[-1]["name"] = "torch recompile"
        events[-1]["cat"] = "user_annotation"
        changes.append(events)
        events = _trace_events()
        events[-1]["name"] = "backend fallback"
        events[-1]["cat"] = "user_annotation"
        changes.append(events)
        events = _trace_events()
        events[7]["name"] = "Memcpy HtoD"
        changes.append(events)
        events = _trace_events()
        events[4] = copy.deepcopy(events[-1])
        events[4]["ts"] = 1_700_000_000_003
        changes.append(events)
        events = _trace_events()
        events[6]["ts"] = 1_700_000_000_030
        changes.append(events)
        for index, events in enumerate(changes):
            with self.subTest(index=index):
                private = copy.deepcopy(self.private)
                private["scopes"][0]["traces"][0]["trace_bytes"] = _trace_bytes(events)
                with self.assertRaises(privacy.PrivacyError):
                    privacy.project_publication(private, self.policy, salt=self.salt)

        for field in (
            "allocation_net_bytes",
            "graph_breaks",
            "recompiles",
            "fallbacks",
            "host_to_device_events",
            "device_to_host_events",
        ):
            with self.subTest(redeclared=field):
                policy = copy.deepcopy(self.policy)
                policy["scopes"][0]["traces"][0][field] = 1
                with self.assertRaises(privacy.PrivacyError):
                    privacy.project_publication(self.private, policy, salt=self.salt)

    def test_trace_rejects_unbounded_local_time(self):
        events = _trace_events()
        events[-1]["ts"] = events[1]["ts"] + privacy.MAX_LOCAL_TIMESTAMP_US + 1
        with self.assertRaisesRegex(privacy.PrivacyError, "local-clock bound"):
            privacy.normalize_trace(_trace_bytes(events))

    def test_correctness_closure_tolerance_and_zero_reference_are_recomputed(self):
        private = copy.deepcopy(self.private)
        private["scopes"][0]["correctness"][0]["captures"][0]["arrays"] = []
        with self.assertRaisesRegex(privacy.PrivacyError, "array closure"):
            privacy.project_publication(private, self.policy, salt=self.salt)

        private = copy.deepcopy(self.private)
        array = private["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]
        array["candidate_bytes"] = struct.pack("<2d", 1.1, 2.0)
        with self.assertRaisesRegex(privacy.PrivacyError, "tolerance"):
            privacy.project_publication(private, self.policy, salt=self.salt)

        private = copy.deepcopy(self.private)
        array = private["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]
        array["reference_bytes"] = struct.pack("<2d", 0.0, 0.0)
        array["candidate_bytes"] = struct.pack("<2d", 1e-12, 0.0)
        with self.assertRaisesRegex(privacy.PrivacyError, "tolerance"):
            privacy.project_publication(private, self.policy, salt=self.salt)

    def test_private_capture_type_cannot_alias_trusted_capture(self):
        private = copy.deepcopy(self.private)
        private["scopes"][0]["correctness"][0]["captures"][0]["capture"] = "0"
        with self.assertRaisesRegex(privacy.PrivacyError, "capture order"):
            privacy.project_publication(private, self.policy, salt=self.salt)

    def test_normalized_correctness_arithmetic_is_public_and_recomputed(self):
        policy = copy.deepcopy(self.policy)
        comparison = policy["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]
        comparison.update(
            comparison_contract="normalized-linf-l2",
            rtol=0.0,
            atol=2e-12,
            normalized_limit=1e-6,
        )
        projected = privacy.project_publication(self.private, policy, salt=self.salt)
        arithmetic = projected["correctness_commitments"]["cases"][0]["captures"][0][
            "arrays"
        ][0]["comparison"]
        self.assertIsNone(arithmetic["max_allowed_error"])
        self.assertGreater(arithmetic["normalized_linf"], 0)
        self.assertGreater(arithmetic["normalized_l2"], 0)
        self.assertLessEqual(arithmetic["normalized_linf"], 1e-6)

    def test_correctness_accepts_canonical_hierarchical_array_names(self):
        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        public_name = "step/100/state/Ex/0-Cpml/indices"
        policy["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0][
            "name"
        ] = public_name
        private["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0][
            "name"
        ] = public_name
        projected = privacy.project_publication(private, policy, salt=self.salt)
        self.assertEqual(
            projected["correctness_commitments"]["cases"][0]["captures"][0]["arrays"][
                0
            ]["name"],
            public_name,
        )

    def test_precollects_all_scope_identities_before_scanning_any_payload(self):
        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        private["scopes"][-1]["identities"]["machine"][
            "opaque_private_value"
        ] = "late-private-host"
        payload = b"late-private-host"
        private["scopes"][0]["payloads"][0]["bytes"] = payload
        descriptor = policy["scopes"][0]["payloads"][0]
        descriptor["size_bytes"] = len(payload)
        descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
        with self.assertRaisesRegex(privacy.PrivacyError, "private opening"):
            privacy.project_publication(private, policy, salt=self.salt)

    def test_private_identity_mapping_keys_are_scanned_before_payloads(self):
        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        private["scopes"][0]["identities"]["machine"] = {
            "fixture-private-host.invalid": True
        }
        payload = b"fixture-private-host.invalid"
        private["scopes"][0]["payloads"][0]["bytes"] = payload
        descriptor = policy["scopes"][0]["payloads"][0]
        descriptor["size_bytes"] = len(payload)
        descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
        with self.assertRaisesRegex(privacy.PrivacyError, "private opening"):
            privacy.project_publication(private, policy, salt=self.salt)

    def test_numeric_boolean_and_null_identity_leaves_are_scanned(self):
        cases = (
            (876543210123456789, b"876543210123456789"),
            (True, b"true"),
            (False, b"false"),
            (None, b"null"),
        )
        for identity, payload in cases:
            with self.subTest(identity=identity):
                policy = copy.deepcopy(self.policy)
                private = copy.deepcopy(self.private)
                private["scopes"][0]["identities"]["machine"] = {
                    "opaque_identity_value": identity
                }
                private["scopes"][0]["payloads"][0]["bytes"] = payload
                descriptor = policy["scopes"][0]["payloads"][0]
                descriptor["size_bytes"] = len(payload)
                descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
                with self.assertRaisesRegex(privacy.PrivacyError, "private opening"):
                    privacy.project_publication(private, policy, salt=self.salt)

        private = copy.deepcopy(self.private)
        private["scopes"][0]["identities"]["machine"] = {"opaque_identity_value": 7}
        with self.assertRaisesRegex(privacy.PrivacyError, "unscannably short"):
            privacy.project_publication(private, self.policy, salt=self.salt)

    def test_windows_drive_payload_names_fail_during_projection(self):
        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        windows_name = "C:/a/_work/repo/evidence.bin"
        policy["scopes"][0]["payloads"][0]["name"] = windows_name
        private["scopes"][0]["payloads"][0]["name"] = windows_name
        with self.assertRaisesRegex(privacy.PrivacyError, "relative"):
            privacy.project_publication(private, policy, salt=self.salt)

    def test_identity_values_are_bounded_and_openings_are_defensive(self):
        private = copy.deepcopy(self.private)
        openings = privacy.PrivateOpenings()
        privacy.project_publication(
            private, self.policy, salt=self.salt, private_openings=openings
        )
        private["scopes"][0]["identities"]["machine"][
            "opaque_private_value"
        ] = "mutated-host"
        first = openings.identity_for_private_verification("cpu", "machine")
        self.assertEqual(first["opaque_private_value"], "private-host-0")
        first["opaque_private_value"] = "caller-mutation"
        self.assertEqual(
            openings.identity_for_private_verification("cpu", "machine"),
            {"opaque_private_value": "private-host-0"},
        )

        private = copy.deepcopy(self.private)
        private["scopes"][0]["identities"]["machine"]["opaque_private_value"] = "xy"
        with self.assertRaisesRegex(privacy.PrivacyError, "short string"):
            privacy.project_publication(private, self.policy, salt=self.salt)

        nested = "private-host"
        for _ in range(privacy.MAX_PRIVATE_JSON_DEPTH + 1):
            nested = [nested]
        private = copy.deepcopy(self.private)
        private["scopes"][0]["identities"]["machine"] = nested
        with self.assertRaisesRegex(privacy.PrivacyError, "deeply nested"):
            privacy.project_publication(private, self.policy, salt=self.salt)

    def test_failed_projection_does_not_consume_private_openings(self):
        private = copy.deepcopy(self.private)
        private["scopes"][0]["payloads"][0]["bytes"] += b"tamper"
        openings = privacy.PrivateOpenings()
        with self.assertRaises(privacy.PrivacyError):
            privacy.project_publication(
                private, self.policy, salt=self.salt, private_openings=openings
            )
        self.project(openings=openings)

    def test_timing_count_and_exact_sample_digest_are_trusted(self):
        projected = self.project()
        timing = projected["technical_scopes"][0]["timings"][0]
        self.assertEqual(timing["samples_sha256"], _canonical_sha256(timing["samples"]))

        private = copy.deepcopy(self.private)
        private["scopes"][0]["timings"][0]["samples"].append(0.004)
        with self.assertRaisesRegex(privacy.PrivacyError, "sample closure"):
            privacy.project_publication(private, self.policy, salt=self.salt)

        private = copy.deepcopy(self.private)
        private["scopes"][0]["timings"][0]["samples"][0] = 0.004
        with self.assertRaisesRegex(privacy.PrivacyError, "trusted digest"):
            privacy.project_publication(private, self.policy, salt=self.salt)

        policy = copy.deepcopy(self.policy)
        policy["scopes"][0]["timings"][0]["sample_count"] = (
            privacy.MAX_TIMING_SAMPLES + 1
        )
        with self.assertRaisesRegex(privacy.PrivacyError, "sample count"):
            privacy.project_publication(self.private, policy, salt=self.salt)

    def test_semantic_phase_inventory_and_exact_allocations_are_trusted(self):
        mutations = []
        events = _trace_events()
        events[-1]["name"] = "aten::masked_scatter_"
        mutations.append(events)
        events = _trace_events()
        events[1]["ph"] = "C"
        mutations.append(events)
        events = _trace_events()
        events[1]["args"].update(Bytes=128, **{"Total Allocated": 128})
        events[2]["args"].update(Bytes=-128, **{"Total Allocated": 0})
        mutations.append(events)
        events = _trace_events()
        events[7]["args"]["Bytes"] = -1
        mutations.append(events)
        for index, events in enumerate(mutations):
            with self.subTest(index=index):
                private = copy.deepcopy(self.private)
                private["scopes"][0]["traces"][0]["trace_bytes"] = _trace_bytes(events)
                with self.assertRaises(privacy.PrivacyError):
                    privacy.project_publication(private, self.policy, salt=self.salt)

    def test_trace_ordinals_bind_process_and_full_device_context(self):
        events = []
        contexts = (
            (10, 7, "CUDA", 0, 1),
            (20, 7, "CUDA", 0, 1),
            (10, 7, "CUDA", 0, 2),
        )
        for index, (pid, tid, device_type, device, context) in enumerate(contexts):
            events.append(
                {
                    "name": "CudaGraphLaunch",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 100 + index,
                    "dur": 1,
                    "pid": pid,
                    "tid": tid,
                    "args": {
                        "Device Type": device_type,
                        "Device Id": device,
                        "context": context,
                        "stream": 3,
                        "Graph Id": 5,
                        "correlation": 99,
                    },
                }
            )
        normalized = privacy.normalize_trace(_trace_bytes(events))["events"]
        self.assertEqual([event["thread_ordinal"] for event in normalized], [0, 1, 0])
        self.assertEqual([event["stream_ordinal"] for event in normalized], [0, 1, 2])
        self.assertEqual([event["graph_ordinal"] for event in normalized], [0, 1, 2])
        self.assertEqual(
            [event["correlation_ordinal"] for event in normalized], [0, 0, 0]
        )

    def test_allocation_context_ordinals_close_multi_device_live_totals(self):
        events = []
        changes = (
            (10, 0, 64, 64),
            (20, 1, 32, 32),
            (10, 0, -64, 0),
            (20, 1, -32, 0),
        )
        for index, (pid, device, amount, total) in enumerate(changes):
            events.append(
                {
                    "name": "[memory]",
                    "cat": "",
                    "ph": "i",
                    "ts": 100 + index,
                    "pid": pid,
                    "tid": 7,
                    "args": {
                        "Bytes": amount,
                        "Total Allocated": total,
                        "Device Type": "CUDA",
                        "Device Id": device,
                        "stream": 3,
                        "Addr": "reused-address",
                    },
                }
            )
        trace = privacy.normalize_trace(_trace_bytes(events))
        self.assertEqual(
            [event["allocation_context_ordinal"] for event in trace["events"]],
            [0, 1, 0, 1],
        )
        self.assertEqual(
            [event["allocation_ordinal"] for event in trace["events"]],
            [0, 1, 0, 1],
        )
        self.assertEqual(trace["summary"]["allocated_bytes"], 96)
        self.assertEqual(trace["summary"]["freed_bytes"], 96)
        self.assertEqual(trace["summary"]["peak_live_allocated_bytes"], 96)
        self.assertEqual(trace["summary"]["final_live_allocated_bytes"], 0)

    def test_allocation_context_ordinals_distinguish_cuda_contexts(self):
        events = []
        changes = (
            (1, 64, 64),
            (2, 32, 32),
            (1, -64, 0),
            (2, -32, 0),
        )
        for index, (context, amount, total) in enumerate(changes):
            events.append(
                {
                    "name": "[memory]",
                    "cat": "",
                    "ph": "i",
                    "ts": 100 + index,
                    "pid": 10,
                    "tid": 7,
                    "args": {
                        "Bytes": amount,
                        "Total Allocated": total,
                        "Device Type": "CUDA",
                        "Device Id": 0,
                        "context": context,
                        "stream": 3,
                        "Addr": "reused-address",
                    },
                }
            )
        trace = privacy.normalize_trace(_trace_bytes(events))
        self.assertEqual(
            [event["allocation_context_ordinal"] for event in trace["events"]],
            [0, 1, 0, 1],
        )
        self.assertEqual(
            [event["allocation_ordinal"] for event in trace["events"]],
            [0, 1, 0, 1],
        )
        self.assertEqual(trace["summary"]["peak_live_allocated_bytes"], 96)
        self.assertEqual(trace["summary"]["final_live_allocated_bytes"], 0)

    def test_correlation_flows_require_ids_and_complete_global_topology(self):
        def flow(phase, correlation, timestamp):
            event = {
                "name": "cpu-to-gpu",
                "cat": "ac2g",
                "ph": phase,
                "ts": timestamp,
                "pid": 10,
                "tid": 7,
                "args": {},
            }
            if correlation is not None:
                event["id"] = correlation
            return event

        interleaved = [
            flow("s", "first", 100),
            flow("s", "second", 101),
            flow("t", "first", 102),
            flow("f", "second", 103),
            flow("f", "first", 104),
        ]
        normalized = privacy.normalize_trace(_trace_bytes(interleaved))["events"]
        self.assertEqual(
            [event["correlation_ordinal"] for event in normalized],
            [0, 1, 0, 1, 0],
        )

        equal_time = [flow("s", "same-time", 100), flow("f", "same-time", 100)]
        privacy.normalize_trace(_trace_bytes(equal_time))

        with self.assertRaisesRegex(privacy.PrivacyError, "timestamps decrease"):
            privacy.normalize_trace(
                _trace_bytes(
                    [flow("s", "reverse-time", 200), flow("f", "reverse-time", 100)]
                )
            )

        with self.assertRaisesRegex(privacy.PrivacyError, "flow has no id"):
            privacy.normalize_trace(
                _trace_bytes([flow("s", None, 100), flow("f", "first", 101)])
            )

        invalid_topologies = (
            ("end before start", ("f", "s")),
            ("step before start", ("t", "s", "f")),
            ("duplicate start", ("s", "s", "f")),
            ("duplicate end", ("s", "f", "f")),
            ("unclosed", ("s", "t")),
        )
        for case, phases in invalid_topologies:
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(privacy.PrivacyError, "topology is incomplete"),
            ):
                privacy.normalize_trace(
                    _trace_bytes(
                        [
                            flow(phase, "shared", 100 + index)
                            for index, phase in enumerate(phases)
                        ]
                    )
                )

    def test_large_integer_timestamps_keep_exact_local_deltas(self):
        origin = 1 << 60
        events = [
            {
                "name": "aten::index_copy_",
                "cat": "cpu_op",
                "ph": "X",
                "ts": origin + index,
                "dur": 0,
                "pid": 10,
                "tid": 7,
                "args": {},
            }
            for index in range(2)
        ]
        normalized = privacy.normalize_trace(_trace_bytes(events))["events"]
        self.assertEqual([event["start_us"] for event in normalized], [0.0, 1.0])

    def test_floating_timestamp_literals_are_exact_and_bounded_before_subtraction(self):
        raw = (
            b'{"traceEvents":['
            b'{"name":"aten::index_copy_","cat":"cpu_op","ph":"X",'
            b'"ts":1152921504606846976.0,"dur":0,"pid":1,"tid":1,"args":{}},'
            b'{"name":"aten::index_copy_","cat":"cpu_op","ph":"X",'
            b'"ts":1152921504606846977.0,"dur":0,"pid":1,"tid":1,"args":{}}]}'
        )
        normalized = privacy.normalize_trace(raw)["events"]
        self.assertEqual([event["start_us"] for event in normalized], [0.0, 1.0])

        oversized_exponent = raw.replace(b"1152921504606846976.0", b"1e999999", 1)
        with self.assertRaisesRegex(privacy.PrivacyError, "number literal"):
            privacy.normalize_trace(oversized_exponent)

    def test_private_keys_are_redacted_from_errors_and_exception_chains(self):
        marker = "authorization_private_probe_value"
        calls = (
            lambda: privacy._exact_keys(
                {marker: True}, {"expected"}, "private fixture object"
            ),
            lambda: privacy.normalize_trace(
                ("{" + json.dumps(marker) + ":1," + json.dumps(marker) + ":2}").encode(
                    "utf-8"
                ),
                label="private trace fixture",
            ),
        )
        for call in calls:
            with (
                self.subTest(call=call),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                call()
            self.assertTrue(_exception_messages(caught.exception))
            self.assertTrue(
                all(
                    marker not in message
                    for message in _exception_messages(caught.exception)
                )
            )

    def test_metadata_clocks_and_sort_index_are_rejected(self):
        events = _trace_events()
        events[0]["ts"] = events[1]["ts"]
        with self.assertRaisesRegex(privacy.PrivacyError, "carries a clock"):
            privacy.normalize_trace(_trace_bytes(events))

        events = _trace_events()
        events[0]["name"] = "process_sort_index"
        events[0]["args"] = {"sort_index": "first"}
        with self.assertRaisesRegex(privacy.PrivacyError, "sort index"):
            privacy.normalize_trace(_trace_bytes(events))

    def test_correctness_arithmetic_closes_elementwise_extrema_and_empty_arrays(self):
        comparison = self.project()["correctness_commitments"]["cases"][0]["captures"][
            0
        ]["arrays"][0]["comparison"]
        self.assertEqual(comparison["reference_abs_max"], 2.0)
        self.assertAlmostEqual(comparison["reference_l2"], math.sqrt(5.0))
        self.assertAlmostEqual(comparison["error_l2"], comparison["max_abs_error"])
        self.assertLessEqual(comparison["max_tolerance_excess"], 0)

        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        descriptor = policy["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]
        descriptor.update(rtol=1e-3, atol=0.0)
        array = private["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]
        array["reference_bytes"] = struct.pack("<2d", 1000.0, 0.0)
        array["candidate_bytes"] = struct.pack("<2d", 1000.0, 0.001)
        with self.assertRaisesRegex(privacy.PrivacyError, "tolerance"):
            privacy.project_publication(private, policy, salt=self.salt)

        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        descriptor = policy["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]
        descriptor["shape"] = [0]
        array = private["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]
        array["shape"] = [0]
        array["reference_bytes"] = b""
        array["candidate_bytes"] = b""
        with self.assertRaisesRegex(privacy.PrivacyError, "contain an element"):
            privacy.project_publication(private, policy, salt=self.salt)

        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        descriptor = policy["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]
        descriptor["shape"] = [1]
        array = private["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]
        array["shape"] = [True]
        array["reference_bytes"] = struct.pack("<d", 1.0)
        array["candidate_bytes"] = struct.pack("<d", 1.0)
        with self.assertRaisesRegex(privacy.PrivacyError, "shape"):
            privacy.project_publication(private, policy, salt=self.salt)

    def test_operations_v2_job_names_order_and_workflow_pairing_are_exact(self):
        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        policy["bindings"]["jobs"][0], policy["bindings"]["jobs"][1] = (
            policy["bindings"]["jobs"][1],
            policy["bindings"]["jobs"][0],
        )
        private["bindings"] = copy.deepcopy(policy["bindings"])
        with self.assertRaisesRegex(privacy.PrivacyError, "job order"):
            privacy.project_publication(private, policy, salt=self.salt)

        policy = copy.deepcopy(self.policy)
        private = copy.deepcopy(self.private)
        policy["bindings"]["jobs"][1]["run_attempt"] += 1
        private["bindings"] = copy.deepcopy(policy["bindings"])
        with self.assertRaisesRegex(privacy.PrivacyError, "attempts differ"):
            privacy.project_publication(private, policy, salt=self.salt)


class Issue123PrivacyScannerTest(unittest.TestCase):
    def _private_sdist(self, *, pax_items=(), uname="builder", gname="builders"):
        info = tarfile.TarInfo("gmes-0.10.0/package/data.py")
        info.uid = 42
        info.gid = 7
        info.uname = uname
        info.gname = gname
        body = b"value = 1\n"
        info.size = len(body)
        ordinary = (
            info.tobuf(format=tarfile.USTAR_FORMAT) + body + b"\0" * (-len(body) % 512)
        )
        records = []
        if pax_items:
            records.append(_pax_helper(tarfile.XHDTYPE, pax_items))
        raw = _physical_tar_bytes(records)[:-1024] + ordinary + b"\0" * 1024
        return _gzip_bytes(raw)

    def _validate_private_sdist(self, raw, *, openings=(), limits=None):
        with tempfile.NamedTemporaryFile() as stream:
            stream.write(raw)
            stream.flush()
            fd = os.open(stream.name, os.O_RDONLY)
            try:
                with privacy._retain_private_sdist_fd(fd) as view:
                    return privacy._validate_private_sdist_raw_first(
                        view,
                        openings,
                        limits=(
                            privacy._default_private_sdist_validation_limits()
                            if limits is None
                            else limits
                        ),
                    )
            finally:
                os.close(fd)

    def test_private_sdist_descriptor_raw_first_owner_contract(self):
        raw = self._private_sdist(
            pax_items=(
                ("uid", "43"),
                ("gid", "8"),
                ("uname", "pax-builder"),
                ("gname", "pax-builders"),
                ("comment", "safe"),
            )
        )
        with (
            mock.patch("pwd.getpwuid", side_effect=AssertionError("lookup")),
            mock.patch("grp.getgrgid", side_effect=AssertionError("lookup")),
            mock.patch.object(
                privacy,
                "_normalization_non_inert_codepoints",
                side_effect=AssertionError("Unicode table rebuilt"),
            ) as rebuilt,
        ):
            started = time.monotonic()
            result = self._validate_private_sdist(raw)
        self.assertLess(time.monotonic() - started, 5.0)
        rebuilt.assert_not_called()
        self.assertEqual(result.archive_size, len(raw))
        self.assertEqual(result.archive_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(result.physical_ordinary_count, 1)
        self.assertEqual(result.logical_member_count, 1)
        self.assertEqual(result.total_member_bytes, len(b"value = 1\n"))
        self.assertEqual(len(result.members), 1)
        member = result.members[0]
        self.assertEqual(member.name, "gmes-0.10.0/package/data.py")
        self.assertEqual(member.type_code, "file")
        self.assertEqual(member.size, len(b"value = 1\n"))
        self.assertGreaterEqual(member.body_offset, 512)
        self.assertEqual(member.sha256, hashlib.sha256(b"value = 1\n").hexdigest())
        self.assertFalse(any("owner" in field for field in result.__dataclass_fields__))
        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_payload("packages/public.tar.gz", raw)

        marker = "private-owner-opening"
        with self.assertRaises(privacy._PrivateSdistValidationError) as caught:
            self._validate_private_sdist(
                self._private_sdist(uname=marker), openings=(marker,)
            )
        self.assertEqual(
            str(caught.exception), privacy._PrivateSdistFailure.ARCHIVE_REJECTED.value
        )
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(marker, " ".join(_exception_messages(caught.exception)))

        with self.assertRaises(privacy._PrivateSdistValidationError) as fabricated:
            privacy._validate_private_sdist_raw_first(
                privacy._PrivateSdistReadView(0, object(), object()),
                (),
                limits=privacy._default_private_sdist_validation_limits(),
            )
        self.assertEqual(
            str(fabricated.exception), privacy._PrivateSdistFailure.SOURCE_INVALID.value
        )

        marker = "forged-private-sdist-fd"

        class ForgedFd:
            def __index__(self):
                raise ValueError(marker)

        class IntSubclass(int):
            pass

        identity_values = {
            "device": 0,
            "inode": 1,
            "mode": stat.S_IFREG | 0o600,
            "nlink": 1,
            "size": 1,
            "mtime_ns": 1,
            "ctime_ns": 1,
        }

        def identity_with(**overrides):
            return privacy._PrivateSdistIdentity(**{**identity_values, **overrides})

        valid_identity = identity_with()
        missing_identity_field = identity_with()
        object.__delattr__(missing_identity_field, "ctime_ns")
        forged_sources = [
            (
                "forged-index",
                privacy._PrivateSdistReadView(
                    ForgedFd(), valid_identity, privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "bool-fd",
                privacy._PrivateSdistReadView(
                    True, valid_identity, privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "int-subclass",
                privacy._PrivateSdistReadView(
                    IntSubclass(0), valid_identity, privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "bool-identity-field",
                privacy._PrivateSdistReadView(
                    0,
                    identity_with(size=True),
                    privacy._PRIVATE_SDIST_VIEW_SEAL,
                ),
            ),
            (
                "missing-identity-field",
                privacy._PrivateSdistReadView(
                    0, missing_identity_field, privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "negative-fd",
                privacy._PrivateSdistReadView(
                    -1, valid_identity, privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "negative-device",
                privacy._PrivateSdistReadView(
                    0, identity_with(device=-1), privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "negative-inode",
                privacy._PrivateSdistReadView(
                    0, identity_with(inode=-1), privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "directory-mode",
                privacy._PrivateSdistReadView(
                    0,
                    identity_with(mode=stat.S_IFDIR | 0o755),
                    privacy._PRIVATE_SDIST_VIEW_SEAL,
                ),
            ),
            (
                "zero-nlink",
                privacy._PrivateSdistReadView(
                    0, identity_with(nlink=0), privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "negative-size",
                privacy._PrivateSdistReadView(
                    0, identity_with(size=-1), privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "negative-mtime",
                privacy._PrivateSdistReadView(
                    0, identity_with(mtime_ns=-1), privacy._PRIVATE_SDIST_VIEW_SEAL
                ),
            ),
            (
                "timestamp-overflow",
                privacy._PrivateSdistReadView(
                    0,
                    identity_with(ctime_ns=privacy._PRIVATE_SDIST_MAX_TIMESTAMP_NS + 1),
                    privacy._PRIVATE_SDIST_VIEW_SEAL,
                ),
            ),
        ]
        for field_name, value in identity_values.items():
            forged_sources.extend(
                (
                    (
                        f"bool-{field_name}",
                        privacy._PrivateSdistReadView(
                            0,
                            identity_with(**{field_name: True}),
                            privacy._PRIVATE_SDIST_VIEW_SEAL,
                        ),
                    ),
                    (
                        f"int-subclass-{field_name}",
                        privacy._PrivateSdistReadView(
                            0,
                            identity_with(**{field_name: IntSubclass(value)}),
                            privacy._PRIVATE_SDIST_VIEW_SEAL,
                        ),
                    ),
                )
            )
        with mock.patch.object(
            privacy.os, "fstat", side_effect=AssertionError("fstat reached")
        ) as fstat:
            for source_label, source in forged_sources:
                with self.subTest(source_type=source_label):
                    with self.assertRaises(
                        privacy._PrivateSdistValidationError
                    ) as caught:
                        privacy._validate_private_sdist_raw_first(
                            source,
                            (),
                            limits=privacy._default_private_sdist_validation_limits(),
                        )
                    error = caught.exception
                    self.assertIs(type(error), privacy._PrivateSdistValidationError)
                    self.assertEqual(
                        error.args,
                        (privacy._PrivateSdistFailure.SOURCE_INVALID.value,),
                    )
                    self.assertIsNone(error.__cause__)
                    self.assertIsNone(error.__context__)
                    diagnostics = " ".join(
                        (
                            str(error),
                            repr(error),
                            repr(error.args),
                            repr(getattr(error, "__notes__", ())),
                            "".join(
                                traceback.format_exception(
                                    type(error), error, error.__traceback__
                                )
                            ),
                        )
                    )
                    self.assertNotIn(marker, diagnostics)
        fstat.assert_not_called()

        defaults = privacy._default_private_sdist_validation_limits()
        limited_values = {
            field_name: getattr(defaults, field_name)
            for field_name in defaults.__dataclass_fields__
        }
        limited_values["matcher_states"] = 1
        with (
            mock.patch.object(
                privacy,
                "_scan_tar",
                side_effect=AssertionError("tar scan entered"),
            ) as tar_scan,
            self.assertRaises(privacy._PrivateSdistValidationError) as caught,
        ):
            self._validate_private_sdist(
                raw,
                openings=("bounded-private-opening",),
                limits=privacy._PrivateSdistValidationLimits(**limited_values),
            )
        tar_scan.assert_not_called()
        self.assertEqual(
            str(caught.exception), privacy._PrivateSdistFailure.ARCHIVE_REJECTED.value
        )

        limited_values["matcher_states"] = defaults.matcher_states
        limited_values["normalization_work_bytes"] = 1
        with (
            mock.patch.object(
                privacy.tarfile,
                "open",
                side_effect=AssertionError("logical tar entered"),
            ) as logical_open,
            self.assertRaises(privacy._PrivateSdistValidationError) as caught,
        ):
            self._validate_private_sdist(
                raw,
                limits=privacy._PrivateSdistValidationLimits(**limited_values),
            )
        logical_open.assert_not_called()
        self.assertEqual(
            str(caught.exception), privacy._PrivateSdistFailure.ARCHIVE_REJECTED.value
        )

    def test_full_normalization_tail_boundary_lookahead(self):
        tail = "\N{COMBINING ACUTE ACCENT}" * 8
        encodings = (
            ("utf-8", tail + "AA"),
            ("utf-16-le", tail + "AA"),
            ("utf-16-be", tail + "AA"),
        )
        with (
            mock.patch.object(privacy, "MAX_PRIVACY_NORMALIZATION_TAIL_CHARACTERS", 8),
            mock.patch.object(privacy, "MAX_PRIVACY_NORMALIZATION_TAIL_BYTES", 32),
            mock.patch.object(privacy, "PRIVACY_TEXT_INPUT_CHUNK_BYTES", 2),
        ):
            for encoding, value in encodings:
                with self.subTest(encoding=encoding):
                    context = privacy._privacy_scan_context(())
                    privacy._scan_decoded_privacy_view(
                        value.encode(encoding),
                        encoding,
                        0,
                        "tail boundary",
                        context,
                        check_patterns=True,
                        check_openings=True,
                    )
            with self.assertRaisesRegex(privacy.PrivacyError, "normalization exceeds"):
                privacy._scan_decoded_privacy_view(
                    (tail + "A\N{COMBINING ACUTE ACCENT}").encode(),
                    "utf-8",
                    0,
                    "tail boundary",
                    privacy._privacy_scan_context(()),
                    check_patterns=True,
                    check_openings=True,
                )

    def test_utf8_surrogate_failures_are_chain_free(self):
        marker = "private-surrogate-marker"
        for value in ("\ud800", "x\udc00", "\udfff"):
            with (
                self.subTest(value=repr(value)),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy._utf8_bytes(value, "JSON string value")
            _assert_sanitized_privacy_error(self, caught.exception, marker)

        escaped = json.dumps({marker: "\ud800"})
        for name, raw, media_type in (
            ("packages/value.bin", escaped.encode(), "application/octet-stream"),
            (
                "packages/value.zip",
                _zip_bytes((("package/config.json", escaped.encode()),)),
                "application/zip",
            ),
            (
                "packages/value.tar",
                _physical_tar_bytes(
                    [_ordinary_tar_record("package/config.json", escaped.encode())]
                ),
                "application/x-tar",
            ),
            (
                "packages/value.tar.gz",
                _gzip_bytes(
                    _physical_tar_bytes(
                        [_ordinary_tar_record("package/config.json", escaped.encode())]
                    )
                ),
                "application/gzip",
            ),
        ):
            with (
                self.subTest(name=name),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.scan_payload(name, raw, media_type=media_type)
            _assert_sanitized_privacy_error(self, caught.exception, marker)

    def test_decoded_json_keys_use_shared_privacy_context(self):
        def escaped(value):
            return "".join(f"\\u{ord(character):04x}" for character in value)

        opening = "caller-canary-q7m9tag"
        documents = (
            "{" + json.dumps(escaped(opening)) + ':"safe"}',
            '{"nested": {' + json.dumps(escaped("HOST=synthetic-node")) + ':"safe"}}',
        )
        for document in documents:
            raw = document.encode()
            carriers = (
                ("packages/keys.bin", raw, "application/octet-stream"),
                (
                    "packages/keys.zip",
                    _zip_bytes((("package/config.json", raw),)),
                    "application/zip",
                ),
                (
                    "packages/keys.tar",
                    _physical_tar_bytes(
                        [_ordinary_tar_record("package/config.json", raw)]
                    ),
                    "application/x-tar",
                ),
                (
                    "packages/keys.tar.gz",
                    _gzip_bytes(
                        _physical_tar_bytes(
                            [_ordinary_tar_record("package/config.json", raw)]
                        )
                    ),
                    "application/gzip",
                ),
            )
            for name, payload, media_type in carriers:
                with (
                    self.subTest(document=document, name=name),
                    self.assertRaises(privacy.PrivacyError),
                ):
                    privacy.scan_payload(
                        name,
                        payload,
                        media_type=media_type,
                        forbidden_values=(opening,),
                    )

        safe = b'{"nested":{"public_key":"safe"}}'
        prepared = privacy._prepare_forbidden_value_plan
        with mock.patch.object(
            privacy, "_prepare_forbidden_value_plan", wraps=prepared
        ) as plan:
            privacy.scan_payload(
                "packages/safe.bin", safe, forbidden_values=("absent-opening",)
            )
        self.assertEqual(plan.call_count, 1)

    def test_pax_resident_ownership_is_preflighted(self):
        helper_body = _pax_record("comment", "q" * (64 * 1024 - 32))
        raw = _physical_tar_bytes(
            [
                _pax_helper(tarfile.XHDTYPE, (("comment", "q" * (64 * 1024 - 32)),)),
                _ordinary_tar_record(),
            ]
        )
        helper_size = len(helper_body)
        compressed = _gzip_bytes(raw)
        with mock.patch.object(privacy, "MAX_TAR_SCAN_TRANSIENT_BYTES", 1024):
            reserve = privacy._pax_resident_reserve(helper_size, 1, 1, 1)
        gzip_resident = len(compressed) + 2 * len(raw) + 2 * 512 + 512 + reserve
        raw_resident = len(raw) + reserve
        pax_limits = {
            "MAX_TAR_BYTES": len(raw),
            "MAX_ARCHIVE_TOTAL_BYTES": len(raw),
            "MAX_TAR_SCAN_TRANSIENT_BYTES": 1024,
            "MAX_TAR_PAX_HELPER_BYTES": helper_size,
            "MAX_TAR_HELPER_TOTAL_BYTES": helper_size,
            "MAX_TAR_PAX_RECORDS": 1,
            "MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS": 1,
            "MAX_ARCHIVE_MEMBERS": 1,
            "TAR_GZIP_INPUT_CHUNK_BYTES": 512,
            "TAR_GZIP_OUTPUT_CHUNK_BYTES": 512,
            "MIN_TAR_GZIP_RATIO_BYTES": len(raw),
            "MAX_PUBLIC_JSON_BYTES": helper_size,
        }
        reader_inputs = []
        resource_states = []
        real_reader = privacy._TarBufferReader

        class RecordingReader(real_reader):
            def __init__(self, source, *, pax_resources=None):
                reader_inputs.append(source)
                resource_states.append(pax_resources)
                super().__init__(source, pax_resources=pax_resources)

        with (
            mock.patch.multiple(
                privacy,
                **{**pax_limits, "MAX_TAR_RESIDENT_BYTES": gzip_resident},
            ),
            mock.patch.object(privacy, "_TarBufferReader", RecordingReader),
        ):
            privacy.scan_payload("packages/pax.tar.gz", compressed)
            privacy.scan_payload("packages/pax.tar", raw)
        self.assertTrue(reader_inputs)
        self.assertTrue(all(type(source) is bytes for source in reader_inputs))
        state = resource_states[0]
        self.assertIsNotNone(state)
        self.assertEqual(state.physical_helper_bytes, helper_size)
        self.assertEqual(state.logical_reparse_bytes, helper_size)
        self.assertEqual(state.logical_reparse_peak_read_bytes, helper_size)
        self.assertEqual(state.logical_reparse_raw_bytes, 3 * helper_size)
        self.assertEqual(state.logical_reparse_decoded_bytes, 4 * helper_size)
        self.assertEqual(state.logical_reparse_record_object_bytes, 512)
        self.assertEqual(state.association_references, 1)
        self.assertEqual(state.association_members, 1)
        self.assertEqual(
            state.physical_association_bytes,
            privacy._PAX_PHYSICAL_ASSOCIATION_RESERVE_BYTES,
        )
        self.assertEqual(
            state.logical_cached_association_bytes,
            privacy._PAX_LOGICAL_ASSOCIATION_RESERVE_BYTES,
        )
        self.assertEqual(state.current_same_value_owners, 8)
        self.assertLessEqual(
            state.resident_bytes(), reserve - pax_limits["MAX_TAR_SCAN_TRANSIENT_BYTES"]
        )

        reader = privacy._TarBufferReader(b"immutable")
        self.assertIs(type(reader._raw), bytes)
        self.assertNotIsInstance(reader._raw, memoryview)
        with self.assertRaises(TypeError):
            privacy._TarBufferReader(bytearray(b"mutable"))

        with (
            mock.patch.multiple(
                privacy,
                **{**pax_limits, "MAX_TAR_RESIDENT_BYTES": raw_resident - 1},
            ),
            mock.patch.object(
                privacy,
                "_scan_tar_physical_records",
                side_effect=AssertionError("physical scan entered"),
            ) as physical_scan,
            self.assertRaisesRegex(privacy.PrivacyError, "resident-memory bound"),
        ):
            privacy.scan_payload("packages/pax.tar", raw)
        physical_scan.assert_not_called()

        with (
            mock.patch.multiple(
                privacy,
                **{
                    **pax_limits,
                    "MAX_TAR_RESIDENT_BYTES": gzip_resident - 1,
                },
            ),
            mock.patch.object(privacy, "_allocate_tar_buffer") as allocate,
            self.assertRaisesRegex(privacy.PrivacyError, "oversized"),
        ):
            privacy.scan_payload("packages/pax.tar.gz", compressed)
        allocate.assert_not_called()

        association_items = (
            ("uid", "0"),
            ("gid", "0"),
            ("uname", ""),
            ("gname", ""),
            ("comment", "safe"),
            ("mtime", "0"),
            ("hdrcharset", "ISO-IR 10646 2000 UTF-8"),
        )
        association_members = 10
        association_references = len(association_items) * association_members
        association_helper = _pax_helper(tarfile.XGLTYPE, association_items)
        association_raw = _physical_tar_bytes(
            [association_helper]
            + [
                _ordinary_tar_record(f"package/member-{index}.bin", b"")
                for index in range(association_members)
            ]
        )
        association_helper_size = len(association_helper[2])
        with mock.patch.object(privacy, "MAX_TAR_SCAN_TRANSIENT_BYTES", 1024):
            association_reserve = privacy._pax_resident_reserve(
                association_helper_size,
                len(association_items),
                association_references,
                association_members,
            )
        association_limits = {
            "MAX_TAR_BYTES": len(association_raw),
            "MAX_ARCHIVE_TOTAL_BYTES": len(association_raw),
            "MAX_ARCHIVE_MEMBERS": association_members,
            "MAX_TAR_SCAN_TRANSIENT_BYTES": 1024,
            "MAX_TAR_PAX_HELPER_BYTES": association_helper_size,
            "MAX_TAR_HELPER_TOTAL_BYTES": association_helper_size,
            "MAX_TAR_PAX_RECORDS": len(association_items),
            "MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS": association_references,
            "MAX_PUBLIC_JSON_BYTES": association_helper_size,
        }
        association_states = []

        class AssociationReader(real_reader):
            def __init__(self, source, *, pax_resources=None):
                association_states.append(pax_resources)
                super().__init__(source, pax_resources=pax_resources)

        with (
            mock.patch.multiple(
                privacy,
                **{
                    **association_limits,
                    "MAX_TAR_RESIDENT_BYTES": len(association_raw)
                    + association_reserve,
                },
            ),
            mock.patch.object(privacy, "_TarBufferReader", AssociationReader),
        ):
            privacy.scan_payload("packages/global-pax.tar", association_raw)
        association_state = association_states[0]
        self.assertEqual(
            association_state.association_references, association_references
        )
        self.assertEqual(association_state.association_members, association_members)
        self.assertEqual(
            association_state.physical_association_bytes,
            privacy._PAX_PHYSICAL_ASSOCIATION_RESERVE_BYTES * association_references,
        )
        self.assertEqual(
            association_state.physical_associated_member_bytes,
            privacy._PAX_PHYSICAL_ASSOCIATED_MEMBER_RESERVE_BYTES * association_members,
        )
        self.assertEqual(
            association_state.logical_cached_association_bytes,
            privacy._PAX_LOGICAL_ASSOCIATION_RESERVE_BYTES * association_references,
        )
        self.assertEqual(
            association_state.logical_cached_member_bytes,
            privacy._PAX_LOGICAL_ASSOCIATED_MEMBER_RESERVE_BYTES * association_members,
        )
        with (
            mock.patch.multiple(
                privacy,
                **{
                    **association_limits,
                    "MAX_TAR_RESIDENT_BYTES": len(association_raw)
                    + association_reserve
                    - 1,
                },
            ),
            mock.patch.object(
                privacy,
                "_scan_tar_physical_records",
                side_effect=AssertionError("physical scan entered"),
            ) as physical_scan,
            self.assertRaisesRegex(privacy.PrivacyError, "resident-memory bound"),
        ):
            privacy.scan_payload("packages/global-pax.tar", association_raw)
        physical_scan.assert_not_called()

    def test_forbidden_openings_are_normalized_across_payload_encodings(self):
        opening = "Straße-Private-Opening"
        normalized = _fullwidth_ascii("STRASSE-PRIVATE-OPENING")
        encoded_variants = (
            normalized.encode("utf-8"),
            normalized.encode("utf-16-le"),
            normalized.encode("utf-16-be"),
            b"\xff\xfe" + normalized.encode("utf-16-le"),
            b"\xfe\xff" + normalized.encode("utf-16-be"),
            b"x" + normalized.encode("utf-16-le"),
            b"x" + normalized.encode("utf-16-be"),
        )
        for index, raw in enumerate(encoded_variants):
            with (
                self.subTest(container="direct", index=index),
                self.assertRaisesRegex(privacy.PrivacyError, "private opening"),
            ):
                privacy.scan_public_bytes(raw, forbidden_values=(opening,))
            with (
                self.subTest(container="opaque", index=index),
                self.assertRaisesRegex(privacy.PrivacyError, "private opening"),
            ):
                privacy.scan_payload(
                    f"packages/normalized-{index}.bin",
                    raw,
                    media_type="application/octet-stream",
                    forbidden_values=(opening,),
                )

        for suffix, raw in (
            (
                "tar",
                _physical_tar_bytes([_ordinary_tar_record(body=encoded_variants[0])]),
            ),
            (
                "tar.gz",
                _gzip_bytes(
                    _physical_tar_bytes(
                        [_ordinary_tar_record(body=encoded_variants[2])]
                    )
                ),
            ),
            (
                "pax.tar",
                _physical_tar_bytes(
                    [
                        _pax_helper(tarfile.XHDTYPE, (("comment", normalized),)),
                        _ordinary_tar_record(),
                    ]
                ),
            ),
        ):
            with (
                self.subTest(container=suffix),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaisesRegex(privacy.PrivacyError, "private opening"),
            ):
                privacy.scan_payload(
                    f"packages/normalized.{suffix}",
                    raw,
                    forbidden_values=(opening,),
                )
            opened.assert_not_called()

    def test_unicode_stream_scans_are_stateful_and_malformed_units_separate(self):
        opening = "Probe\N{GRINNING FACE}Opening"
        rendered = "PROBE\N{GRINNING FACE}OPENING"
        variants = (
            rendered.encode("utf-8"),
            rendered.encode("utf-16-le"),
            rendered.encode("utf-16-be"),
            b"\xff\xfe" + rendered.encode("utf-16-le"),
            b"\xfe\xff" + rendered.encode("utf-16-be"),
            b"x" + rendered.encode("utf-16-le"),
            b"x" + rendered.encode("utf-16-be"),
        )
        with mock.patch.object(privacy, "PRIVACY_TEXT_INPUT_CHUNK_BYTES", 5):
            for index, raw in enumerate(variants):
                with (
                    self.subTest(index=index),
                    self.assertRaisesRegex(privacy.PrivacyError, "private opening"),
                ):
                    privacy.scan_public_bytes(raw, forbidden_values=(opening,))

            malformed = (
                "Probe".encode("utf-16-le")
                + b"\x00\xd8"
                + "Opening".encode("utf-16-le")
            )
            privacy.scan_public_bytes(
                malformed,
                forbidden_values=("ProbeOpening",),
            )
            privacy.scan_public_bytes(
                b"Probe\xffOpening",
                forbidden_values=("Probe\N{REPLACEMENT CHARACTER}Opening",),
            )
            with self.assertRaises(privacy.PrivacyError):
                privacy.scan_public_bytes(
                    b"\xff" + _fullwidth_ascii("HOST=synthetic-node").encode("utf-8")
                )

        tail_limit = 10
        with (
            mock.patch.object(
                privacy, "MAX_PRIVACY_NORMALIZATION_TAIL_CHARACTERS", tail_limit
            ),
            mock.patch.object(privacy, "MAX_PRIVACY_NORMALIZATION_TAIL_BYTES", 64),
        ):
            privacy.scan_public_bytes(("x" + "\N{COMBINING ACUTE ACCENT}" * 9).encode())
            with self.assertRaisesRegex(privacy.PrivacyError, "normalization exceeds"):
                privacy.scan_public_bytes(
                    ("x" + "\N{COMBINING ACUTE ACCENT}" * 10).encode()
                )

    def test_non_ascii_normalization_suffixes_match_the_whole_string_oracle(self):
        def encoded_variants(value):
            utf8 = value.encode("utf-8")
            utf16_le = value.encode("utf-16-le")
            utf16_be = value.encode("utf-16-be")
            return (
                ("utf8", utf8),
                ("utf8-aligned", b"x" + utf8),
                ("utf8-bom", b"\xef\xbb\xbf" + utf8),
                ("utf8-bom-aligned", b"x\xef\xbb\xbf" + utf8),
                ("utf16-le", utf16_le),
                ("utf16-le-aligned", b"x" + utf16_le),
                ("utf16-le-bom", b"\xff\xfe" + utf16_le),
                ("utf16-le-bom-aligned", b"x\xff\xfe" + utf16_le),
                ("utf16-be", utf16_be),
                ("utf16-be-aligned", b"x" + utf16_be),
                ("utf16-be-bom", b"\xfe\xff" + utf16_be),
                ("utf16-be-bom-aligned", b"x\xfe\xff" + utf16_be),
            )

        hangul_opening = "\uac01" * 3
        hangul_decomposed = ("\u1100\u1161\u11a8") * 3
        cases = [
            (
                "hangul",
                hangul_opening,
                hangul_decomposed,
            )
        ]
        cases.extend(
            (
                f"tibetan-{ord(character):04x}-{count}",
                character * count,
                character * count,
            )
            for character in ("\u0f73", "\u0f75", "\u0f81")
            for count in (5, 6)
        )
        for case, opening, rendered in cases:
            self.assertEqual(
                privacy._privacy_nfkc_casefold(rendered),
                privacy._privacy_nfkc_casefold(opening),
            )
            context = privacy._privacy_scan_context((opening,))
            for chunk_bytes in range(2, 10):
                with mock.patch.object(
                    privacy, "PRIVACY_TEXT_INPUT_CHUNK_BYTES", chunk_bytes
                ):
                    for encoding, raw in encoded_variants(rendered):
                        with (
                            self.subTest(
                                case=case,
                                chunk_bytes=chunk_bytes,
                                encoding=encoding,
                            ),
                            self.assertRaisesRegex(
                                privacy.PrivacyError, "private opening"
                            ),
                        ):
                            privacy.scan_public_bytes(
                                raw,
                                forbidden_values=context,
                            )

        for case, opening, rendered in (
            ("hangul", hangul_opening, hangul_decomposed),
            ("tibetan", "\u0f73" * 6, "\u0f73" * 6),
        ):
            stream = io.BytesIO()
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("package/data.bin", rendered.encode())
            carriers = (
                (
                    "opaque",
                    f"packages/{case}.bin",
                    rendered.encode(),
                    "application/octet-stream",
                ),
                (
                    "zip",
                    f"packages/{case}.zip",
                    stream.getvalue(),
                    "application/zip",
                ),
            )
            with mock.patch.object(privacy, "PRIVACY_TEXT_INPUT_CHUNK_BYTES", 2):
                for carrier, name, raw, media_type in carriers:
                    with (
                        self.subTest(case=case, carrier=carrier),
                        self.assertRaisesRegex(privacy.PrivacyError, "private opening"),
                    ):
                        privacy.scan_payload(
                            name,
                            raw,
                            media_type=media_type,
                            forbidden_values=(opening,),
                        )

                tar_carriers = (
                    (
                        "tar",
                        _physical_tar_bytes(
                            [_ordinary_tar_record(body=rendered.encode())]
                        ),
                    ),
                    (
                        "tar.gz",
                        _gzip_bytes(
                            _physical_tar_bytes(
                                [_ordinary_tar_record(body=rendered.encode())]
                            )
                        ),
                    ),
                    (
                        "pax",
                        _physical_tar_bytes(
                            [
                                _pax_helper(
                                    tarfile.XHDTYPE,
                                    (("comment", rendered),),
                                ),
                                _ordinary_tar_record(),
                            ]
                        ),
                    ),
                )
                for carrier, raw in tar_carriers:
                    with (
                        self.subTest(case=case, carrier=carrier),
                        mock.patch.object(
                            privacy.tarfile,
                            "open",
                            side_effect=AssertionError("logical parser entered"),
                        ) as opened,
                        self.assertRaisesRegex(privacy.PrivacyError, "private opening"),
                    ):
                        privacy.scan_payload(
                            (
                                f"packages/{case}-pax.tar"
                                if carrier == "pax"
                                else f"packages/{case}.{carrier}"
                            ),
                            raw,
                            media_type=(
                                "application/gzip"
                                if carrier == "tar.gz"
                                else "application/x-tar"
                            ),
                            forbidden_values=(opening,),
                        )
                    opened.assert_not_called()

        unstable_starts = 0
        unstable_ends = 0
        composition_participants = set()
        inert_violations = []
        for codepoint in range(0x110000):
            if 0xD800 <= codepoint <= 0xDFFF:
                continue
            character = chr(codepoint)
            source_decomposition = unicodedata.normalize("NFKD", character)
            direct_decomposition = unicodedata.decomposition(character)
            if direct_decomposition and not direct_decomposition.startswith("<"):
                parts = direct_decomposition.split()
                if len(parts) == 2:
                    composition_participants.update(int(part, 16) for part in parts)
            transformed = unicodedata.normalize(
                "NFKD", privacy._privacy_nfkc_casefold(character)
            )
            starts_nonstarter = bool(
                transformed and unicodedata.combining(transformed[0])
            )
            ends_nonstarter = bool(
                transformed and unicodedata.combining(transformed[-1])
            )
            if starts_nonstarter:
                unstable_starts += 1
                self.assertFalse(
                    privacy._starts_normalization_starter_group(character + "A", 0)
                )
            if ends_nonstarter:
                unstable_ends += 1
                self.assertFalse(
                    privacy._starts_normalization_starter_group(character + "A", 0)
                )
            if codepoint not in privacy._NORMALIZATION_NON_INERT_CODEPOINTS and (
                unicodedata.combining(character) != 0
                or unicodedata.category(character).startswith("M")
                or source_decomposition != character
                or character.casefold() != character
            ):
                inert_violations.append(codepoint)
        composition_participants.update(range(0x1100, 0x1113))
        composition_participants.update(range(0x1161, 0x1176))
        composition_participants.update(range(0x11A8, 0x11C3))
        composition_participants.update(range(0xAC00, 0xD7A4, 28))
        self.assertEqual(inert_violations, [])
        self.assertEqual(
            composition_participants - privacy._NORMALIZATION_NON_INERT_CODEPOINTS,
            set(),
        )
        self.assertGreater(unstable_starts, 0)
        self.assertGreater(unstable_ends, 0)
        self.assertTrue(
            all(
                privacy._starts_normalization_starter_group(chr(codepoint) + "A", 0)
                for codepoint in range(128)
            )
        )
        self.assertFalse(
            privacy._starts_normalization_starter_group("A\N{COMBINING RING ABOVE}", 0)
        )
        for left, right in (
            (
                "A\N{COMBINING ACUTE ACCENT}",
                "AA\N{COMBINING RING ABOVE}",
            ),
            (
                "A\N{COMBINING ACUTE ACCENT}",
                "\u7373\N{COMBINING RING ABOVE}",
            ),
        ):
            self.assertEqual(
                privacy._privacy_nfkc_casefold(left + right),
                privacy._privacy_nfkc_casefold(left)
                + privacy._privacy_nfkc_casefold(right),
            )

        with (
            mock.patch.object(privacy, "MAX_PRIVACY_NORMALIZATION_TAIL_CHARACTERS", 8),
            mock.patch.object(privacy, "MAX_PRIVACY_NORMALIZATION_TAIL_BYTES", 32),
            mock.patch.object(privacy, "PRIVACY_TEXT_INPUT_CHUNK_BYTES", 2),
        ):
            privacy.scan_public_bytes("\u0301".encode() * 8)
            with self.assertRaisesRegex(privacy.PrivacyError, "normalization exceeds"):
                privacy.scan_public_bytes("\u0301".encode() * 9)

    def test_normalized_pattern_streams_close_whitespace_and_email_bounds(self):
        long_space = b" " * (privacy.MAX_PRIVACY_BUILTIN_PATTERN_BYTES + 257)
        unsafe = (
            b"token" + long_space + b"=x",
            b"HOST" + long_space + b"=synthetic-node",
            b'"hostname"' + long_space + b':"synthetic-node"',
            (
                _fullwidth_ascii("HOST")
                + "\N{EM SPACE}" * (privacy.MAX_PRIVACY_BUILTIN_PATTERN_BYTES + 257)
                + _fullwidth_ascii("=synthetic-node")
            ).encode(),
            (
                _fullwidth_ascii("TOKEN")
                + "\N{EM SPACE}" * (privacy.MAX_PRIVACY_BUILTIN_PATTERN_BYTES + 257)
                + _fullwidth_ascii("=x")
            ).encode(),
            (
                '"'
                + _fullwidth_ascii("hostname")
                + '"'
                + "\N{EM SPACE}" * (privacy.MAX_PRIVACY_BUILTIN_PATTERN_BYTES + 257)
                + _fullwidth_ascii(':"synthetic-node"')
            ).encode(),
        )
        with mock.patch.object(privacy, "PRIVACY_TEXT_INPUT_CHUNK_BYTES", 7):
            for raw in unsafe:
                with (
                    self.subTest(raw=raw[:12]),
                    self.assertRaises(privacy.PrivacyError),
                ):
                    privacy.scan_public_bytes(raw)
            with self.assertRaisesRegex(privacy.PrivacyError, "private opening"):
                privacy.scan_public_bytes(
                    b"prefix-private-opening-suffix",
                    forbidden_values=("private-opening",),
                )

        privacy.scan_public_bytes(b"a" * 65 + b"@example.com")
        with self.assertRaisesRegex(privacy.PrivacyError, "email address"):
            privacy.scan_public_bytes(b"a" * 64 + b"@example.com")

    def test_forbidden_plan_is_frozen_once_and_has_an_aggregate_work_bound(self):
        class OneShotValues:
            def __init__(self):
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations != 1:
                    raise AssertionError("forbidden values iterated twice")
                return iter(("absent-private-opening",))

        values = OneShotValues()
        raw = _physical_tar_bytes(
            [
                _ordinary_tar_record("package/one.bin"),
                _ordinary_tar_record("package/two.bin"),
                _ordinary_tar_record("package/three.bin"),
            ]
        )
        prepare = privacy._prepare_forbidden_value_plan
        with mock.patch.object(
            privacy, "_prepare_forbidden_value_plan", wraps=prepare
        ) as prepared:
            privacy.scan_payload(
                "packages/one-shot.tar",
                raw,
                forbidden_values=values,
            )
        self.assertEqual(values.iterations, 1)
        self.assertEqual(prepared.call_count, 1)

        for invalid in ("private-opening", b"private-opening"):
            with (
                self.subTest(container=type(invalid).__name__),
                self.assertRaisesRegex(privacy.PrivacyError, "scan exceeds"),
            ):
                privacy.scan_public_bytes(b"safe", forbidden_values=invalid)

        class InfiniteValues:
            def __init__(self):
                self.next_calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.next_calls += 1
                return "bounded-opening"

        infinite = InfiniteValues()
        with (
            mock.patch.object(privacy, "MAX_IDENTITY_SCAN_VALUES", 2),
            self.assertRaisesRegex(privacy.PrivacyError, "scan exceeds"),
        ):
            privacy.scan_public_bytes(b"safe", forbidden_values=infinite)
        self.assertEqual(infinite.next_calls, 3)

        iterator_marker = "private-iterator-exception-marker"

        class RaisingValues:
            def __iter__(self):
                raise RuntimeError(iterator_marker)

        with self.assertRaises(privacy.PrivacyError) as caught:
            privacy.scan_public_bytes(b"safe", forbidden_values=RaisingValues())
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(
            iterator_marker,
            " ".join(
                (
                    str(caught.exception),
                    repr(caught.exception),
                    "".join(
                        traceback.format_exception(
                            type(caught.exception),
                            caught.exception,
                            caught.exception.__traceback__,
                        )
                    ),
                )
            ),
        )

        with (
            mock.patch.object(privacy, "MAX_PRIVACY_SCAN_CANONICAL_BYTES", 8),
            self.assertRaisesRegex(privacy.PrivacyError, "scan exceeds"),
        ):
            privacy.scan_public_bytes(
                b"public-data-that-does-not-match",
                forbidden_values=("absent-private-opening",),
            )

        canonicalize = privacy._privacy_nfkc_casefold
        with (
            mock.patch.object(privacy, "MAX_FORBIDDEN_PLAN_BYTES", 1),
            mock.patch.object(
                privacy,
                "_privacy_nfkc_casefold",
                wraps=canonicalize,
            ) as transformed,
            self.assertRaisesRegex(privacy.PrivacyError, "scan exceeds"),
        ):
            privacy._prepare_forbidden_value_plan(("abcd", "efgh", "ijkl"))
        self.assertEqual(transformed.call_count, 1)

    def test_canonical_matcher_duplicate_prefix_and_failure_links(self):
        duplicate = privacy._build_canonical_matcher((b"abc", b"abc"))
        self.assertEqual(len(duplicate.transitions), 4)

        patterns = (b"he", b"she", b"his", b"hers", b"he")
        matcher = privacy._build_canonical_matcher(patterns)
        candidates = (
            b"",
            b"her",
            b"ushers",
            b"ahishers",
            b"nomatch",
            b"sh",
            b"she",
        )
        for candidate in candidates:
            expected = any(pattern in candidate for pattern in patterns)
            for split in range(len(candidate) + 1):
                state, first = matcher.scan(candidate[:split])
                _state, second = matcher.scan(candidate[split:], state)
                with self.subTest(candidate=candidate, split=split):
                    self.assertEqual(first or second, expected)

        prefix = privacy._build_canonical_matcher((b"ab", b"abcd"))
        _state, matched = prefix.scan(b"ab")
        self.assertTrue(matched)
        with self.assertRaises(TypeError):
            prefix.transitions[0][ord("z")] = 1

    def test_gzip_preflight_uses_exact_bounded_storage_and_no_copy_reader(self):
        tar_raw = _physical_tar_bytes([_ordinary_tar_record()])
        compressed = _gzip_bytes(tar_raw)
        allocation = bytearray(len(tar_raw))
        with mock.patch.object(
            privacy, "_allocate_tar_buffer", return_value=allocation
        ) as allocate:
            decoded = privacy._decompress_gzip_bounded("fixture.tar.gz", compressed)
        allocate.assert_called_once_with(len(tar_raw))
        self.assertIs(decoded, allocation)
        self.assertEqual(decoded, tar_raw)

        with self.assertRaises(TypeError):
            privacy._TarBufferReader(decoded)
        reader = privacy._TarBufferReader(bytes(decoded))
        self.assertEqual(reader.read(7), tar_raw[:7])
        self.assertEqual(reader.seek(-4, io.SEEK_END), len(tar_raw) - 4)
        self.assertEqual(reader.read(), tar_raw[-4:])
        past_eof = len(tar_raw) + 17
        self.assertEqual(reader.seek(past_eof), past_eof)
        self.assertEqual(reader.read(2), b"")
        self.assertEqual(reader.tell(), past_eof)
        self.assertEqual(reader.seek(-past_eof - 1, io.SEEK_CUR), 0)
        self.assertEqual(reader.seek(-len(tar_raw) - 1, io.SEEK_END), 0)
        with self.assertRaises(ValueError):
            reader.seek(-1)

        advertised_too_large = compressed[:-4] + (len(tar_raw) + 1).to_bytes(
            4, "little"
        )
        fixed = (
            len(compressed)
            + 2 * privacy.TAR_GZIP_INPUT_CHUNK_BYTES
            + privacy.TAR_GZIP_OUTPUT_CHUNK_BYTES
            + privacy._pax_resident_reserve()
        )
        with (
            mock.patch.object(
                privacy, "MAX_TAR_RESIDENT_BYTES", fixed + 2 * len(tar_raw)
            ),
            mock.patch.object(privacy, "_allocate_tar_buffer") as allocate,
            self.assertRaisesRegex(privacy.PrivacyError, "oversized"),
        ):
            privacy._decompress_gzip_bounded(
                "resident-preflight.tar.gz", advertised_too_large
            )
        allocate.assert_not_called()

        inconsistent_isize = compressed[:-4] + (len(tar_raw) - 1).to_bytes(4, "little")
        with self.assertRaises(privacy.PrivacyError) as caught:
            privacy._decompress_gzip_bounded(
                "actual-size-check.tar.gz", inconsistent_isize
            )
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_tar_member_scan_avoids_exfileobject_request_sized_allocations(self):
        body = b"s" * (64 * 1024)
        tar_raw = _physical_tar_bytes([_ordinary_tar_record("package/data.bin", body)])
        for suffix, raw in (
            ("tar", tar_raw),
            ("tar.gz", _gzip_bytes(tar_raw)),
        ):
            with (
                self.subTest(suffix=suffix),
                mock.patch.object(
                    privacy.tarfile.TarFile,
                    "extractfile",
                    side_effect=AssertionError("ExFileObject storage requested"),
                ) as extractfile,
            ):
                privacy.scan_payload(f"packages/member.{suffix}", raw)
            extractfile.assert_not_called()

    def test_tar_library_exceptions_are_chain_free_and_non_leaking(self):
        raw_tar = _physical_tar_bytes([_ordinary_tar_record()])
        marker = "private-library-exception-marker"
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as archive:
            logical_info = next(iter(archive))

        def assert_sanitized(caught):
            error = caught.exception
            self.assertIsNone(error.__cause__)
            self.assertIsNone(error.__context__)
            rendered = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            diagnostics = " ".join((str(error), repr(error), rendered))
            self.assertNotIn(marker, diagnostics)

        for suffix, payload in (("tar", raw_tar), ("tar.gz", _gzip_bytes(raw_tar))):
            with (
                self.subTest(stage="header", suffix=suffix),
                mock.patch.object(
                    privacy.tarfile.TarInfo,
                    "frombuf",
                    side_effect=RecursionError(marker),
                ),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.scan_payload(f"packages/header.{suffix}", payload)
            assert_sanitized(caught)

            for stage in ("iteration", "close"):
                fake_archive = mock.MagicMock()
                fake_archive.pax_headers = {}
                fake_archive.__iter__.return_value = iter((logical_info,))
                if stage == "iteration":
                    fake_archive.__iter__.side_effect = OSError(marker)
                else:
                    fake_archive.close.side_effect = OSError(marker)
                with (
                    self.subTest(stage=stage, suffix=suffix),
                    mock.patch.object(
                        privacy.tarfile, "open", return_value=fake_archive
                    ),
                    self.assertRaises(privacy.PrivacyError) as caught,
                ):
                    privacy.scan_payload(f"packages/{stage}.{suffix}", payload)
                assert_sanitized(caught)

            with (
                self.subTest(stage="logical", suffix=suffix),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=OSError(marker),
                ),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.scan_payload(f"packages/logical.{suffix}", payload)
            assert_sanitized(caught)

        with self.assertRaises(privacy.PrivacyError) as caught:
            privacy.scan_public_bytes(
                marker.upper().encode(),
                forbidden_values=(marker,),
            )
        assert_sanitized(caught)

        class FailingDecompressor:
            def decompress(self, _data, _max_length):
                raise zlib.error(marker)

        with (
            mock.patch.object(
                privacy.zlib, "decompressobj", return_value=FailingDecompressor()
            ),
            self.assertRaises(privacy.PrivacyError) as caught,
        ):
            privacy.scan_payload("packages/zlib.tar.gz", _gzip_bytes(raw_tar))
        assert_sanitized(caught)

        with (
            mock.patch.object(
                privacy.zlib, "decompressobj", side_effect=ValueError(marker)
            ),
            self.assertRaises(privacy.PrivacyError) as caught,
        ):
            privacy.scan_payload("packages/zlib-init.tar.gz", _gzip_bytes(raw_tar))
        assert_sanitized(caught)

        invalid_helpers = (
            _physical_tar_bytes(
                [("././@PaxHeader", tarfile.XHDTYPE, b"19 comment=\xffxxxxxx\n")]
            ),
            _physical_tar_bytes(
                [("././@LongLink", tarfile.GNUTYPE_LONGNAME, b"\xff" + b"\0")]
            ),
        )
        for raw in invalid_helpers:
            with self.assertRaises(privacy.PrivacyError) as caught:
                privacy.scan_payload("packages/invalid-helper.tar", raw)
            self.assertIsNone(caught.exception.__cause__)
            self.assertIsNone(caught.exception.__context__)

    def test_pax_key_normalization_and_raw_budget_edges_fail_closed(self):
        for key, expected in (
            (_fullwidth_ascii("SIZE"), "unsupported tar size override"),
            ("LiNkPaTh", "unsupported link path"),
            ("Schily.ACL.ACE", "unsupported tar metadata override"),
        ):
            raw = _physical_tar_bytes(
                [
                    _pax_helper(tarfile.XHDTYPE, ((key, "4"),)),
                    _ordinary_tar_record(),
                ]
            )
            with (
                self.subTest(key=key),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaisesRegex(privacy.PrivacyError, expected),
            ):
                privacy.scan_payload("packages/normalized-key.tar", raw)
            opened.assert_not_called()

        within = _physical_tar_bytes(
            [
                _pax_helper(
                    tarfile.XGLTYPE,
                    (("comment", "public"), ("mtime", "1")),
                ),
                _ordinary_tar_record(),
            ]
        )
        with mock.patch.object(privacy, "MAX_TAR_PAX_RECORDS", 2):
            privacy.scan_payload("packages/raw-budget-edge.tar", within)

        beyond = _physical_tar_bytes(
            [
                _pax_helper(
                    tarfile.XGLTYPE,
                    (("comment", "public"), ("mtime", "1")),
                ),
                _pax_helper(tarfile.XHDTYPE, (("uid", "0"),)),
                _ordinary_tar_record(),
            ]
        )
        with (
            mock.patch.object(privacy, "MAX_TAR_PAX_RECORDS", 2),
            mock.patch.object(
                privacy.tarfile,
                "open",
                side_effect=AssertionError("logical parser entered"),
            ) as opened,
            self.assertRaisesRegex(privacy.PrivacyError, "PAX record closure"),
        ):
            privacy.scan_payload("packages/raw-budget-over.tar", beyond)
        opened.assert_not_called()

    def test_pax_allowlist_aliases_and_identical_overlay_budget_matrix(self):
        values = {
            "path": "package/pax-name.bin",
            "mtime": "1.5",
            "comment": "public",
            "hdrcharset": "ISO-IR 10646 2000 UTF-8",
            "uid": "0",
            "gid": "0",
            "uname": "",
            "gname": "",
        }
        vendor_values = {
            "SUN.holesdata": "0",
            "SCHILY.acl.ace": "public",
            "RHT.security.selinux": "public",
            "LIBARCHIVE.symlinktype": "file",
            "SCHILY.devmajor": "0",
            "SCHILY.devminor": "0",
            "SCHILY.ino": "1",
        }
        rejected = []
        for key, value in values.items():
            for alias in (key.swapcase(), _fullwidth_ascii(key.upper())):
                for helper_type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
                    rejected.append((f"allowed-alias-{key}", helper_type, alias, value))
        for key, value in vendor_values.items():
            aliases = (
                key,
                key.swapcase(),
                _fullwidth_ascii(key.upper()),
            )
            for alias in aliases:
                for helper_type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
                    rejected.append((f"vendor-{key}", helper_type, alias, value))
        rejected.extend(
            (
                ("unknown-local", tarfile.XHDTYPE, "unknown", "public"),
                ("unknown-global", tarfile.XGLTYPE, "unknown", "public"),
                (
                    "global-path",
                    tarfile.XGLTYPE,
                    "path",
                    "package/global.bin",
                ),
            )
        )
        for case, helper_type, key, value in rejected:
            raw = _physical_tar_bytes(
                [
                    _pax_helper(helper_type, ((key, value),)),
                    _ordinary_tar_record(),
                ]
            )
            with (
                self.subTest(case=case, key=key, helper_type=helper_type),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaises(privacy.PrivacyError),
            ):
                privacy.scan_payload("packages/rejected-key.tar", raw)
            opened.assert_not_called()

        positive = []
        for key, value in values.items():
            positive.append(
                (
                    f"local-{key}",
                    [_pax_helper(tarfile.XHDTYPE, ((key, value),))],
                )
            )
            if key != "path":
                positive.append(
                    (
                        f"global-{key}",
                        [_pax_helper(tarfile.XGLTYPE, ((key, value),))],
                    )
                )
                positive.append(
                    (
                        f"identical-{key}",
                        [
                            _pax_helper(tarfile.XGLTYPE, ((key, value),)),
                            _pax_helper(tarfile.XHDTYPE, ((key, value),)),
                        ],
                    )
                )
        with (
            mock.patch.object(
                privacy.tarfile.TarFile,
                "getmembers",
                side_effect=AssertionError("unbounded logical list"),
            ) as getmembers,
            mock.patch.object(
                privacy.tarfile.TarFile,
                "extract",
                side_effect=AssertionError("filesystem extraction"),
            ) as extract,
            mock.patch.object(
                privacy.tarfile.TarFile,
                "extractall",
                side_effect=AssertionError("filesystem extraction"),
            ) as extractall,
            mock.patch.object(
                privacy.tarfile.TarFile,
                "extractfile",
                side_effect=AssertionError("logical member copy"),
            ) as extractfile,
        ):
            for case, helpers in positive:
                raw = _physical_tar_bytes([*helpers, _ordinary_tar_record()])
                with self.subTest(case=case):
                    privacy.scan_payload("packages/allowed-key.tar", raw)
        getmembers.assert_not_called()
        extract.assert_not_called()
        extractall.assert_not_called()
        extractfile.assert_not_called()

        global_helper = _pax_helper(tarfile.XGLTYPE, (("comment", "public"),))
        local_helper = _pax_helper(tarfile.XHDTYPE, (("comment", "public"),))
        identical = _physical_tar_bytes(
            [global_helper, local_helper, _ordinary_tar_record()]
        )
        thresholds = (
            ("MAX_TAR_PAX_HELPER_BYTES", len(global_helper[2])),
            (
                "MAX_TAR_HELPER_TOTAL_BYTES",
                len(global_helper[2]) + len(local_helper[2]),
            ),
            ("MAX_TAR_PAX_RECORDS", 2),
            ("MAX_TAR_CONSECUTIVE_HELPERS", 2),
            ("MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS", 1),
        )
        for attribute, exact in thresholds:
            with (
                self.subTest(attribute=attribute, boundary="exact"),
                mock.patch.object(privacy, attribute, exact),
            ):
                privacy.scan_payload("packages/exact-overlay.tar", identical)
            with (
                self.subTest(attribute=attribute, boundary="one-below"),
                mock.patch.object(privacy, attribute, exact - 1),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaises(privacy.PrivacyError),
            ):
                privacy.scan_payload("packages/below-overlay.tar", identical)
            opened.assert_not_called()

        marker = "pax-shadow-canary-q7m9tag"
        for global_value, local_value in (
            (marker, "public"),
            ("public", marker),
        ):
            raw = _physical_tar_bytes(
                [
                    _pax_helper(
                        tarfile.XGLTYPE,
                        (("comment", global_value),),
                    ),
                    _pax_helper(
                        tarfile.XHDTYPE,
                        (("comment", local_value),),
                    ),
                    _ordinary_tar_record(),
                ]
            )
            with (
                self.subTest(
                    global_value=global_value,
                    local_value=local_value,
                ),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaisesRegex(privacy.PrivacyError, "private opening"),
            ):
                privacy.scan_payload(
                    "packages/shadow-overlay.tar",
                    raw,
                    forbidden_values=(marker,),
                )
            opened.assert_not_called()

    def test_rejects_text_binary_and_casefolded_personal_metadata(self):
        bad = (
            b"/home/alice/project",
            b"/Users/alice/project",
            b"/tmp/build-root/file",
            b"/workspace/project/file",
            b"/runner/_work/repo/repo",
            b"file:///private/source.py",
            b"C:\\Users\\alice\\repo",
            rb"C:\a\_work\repo\evidence.bin",
            b"C:/a/_work/repo/evidence.bin",
            b"github_pat_abcdefghijklmnopqrstuvwxyz123456",
            b"alice@example.com",
            b"123e4567-e89b-12d3-a456-426614174000",
            b"123e4567-e89b-72d3-a456-426614174000",
            b"GPU-123e4567-e89b-82d3-f456-426614174000",
            "ALICE".encode("utf-16-le"),
            "/Users/alice/private".encode("utf-16-le"),
            "TOKEN=abcdefghijklmnopqrstuvwxyz".encode("utf-16-be"),
        )
        for raw in bad:
            with self.subTest(raw=raw[:30]):
                with self.assertRaises(privacy.PrivacyError):
                    privacy.scan_public_bytes(
                        raw,
                        forbidden_values=("Alice",),
                    )

    def test_recursively_scans_wheels_and_rejects_aliases_links_and_traversal(self):
        safe = _zip_bytes(
            [
                ("package/__init__.py", b"value = 1\n"),
                ("package/data.bin", b"safe bytes"),
            ]
        )
        privacy.scan_payload("packages/safe.whl", safe)

        bad_archives = (
            _zip_bytes([("../escape", b"safe")]),
            _zip_bytes([("A.txt", b"one"), ("a.txt", b"two")]),
            _zip_bytes([("package/data", b"/Users/alice/private")]),
            _zip_bytes([("package/link", b"target")], symlink=True),
            _zip_bytes([("package/ALICE.txt", b"safe")]),
        )
        for index, raw in enumerate(bad_archives):
            with self.subTest(index=index):
                with self.assertRaises(privacy.PrivacyError):
                    privacy.scan_payload(
                        "packages/unsafe.whl",
                        raw,
                        forbidden_values=("alice",),
                    )

    def test_rejects_nonempty_compressed_archive_directory_bodies(self):
        for body in (b"safe hidden bytes", b"token=x"):
            zip_stream = io.BytesIO()
            with zipfile.ZipFile(zip_stream, "w") as archive:
                info = zipfile.ZipInfo("hidden/")
                info.create_system = 3
                info.external_attr = (stat.S_IFDIR | 0o755) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, body)
            with (
                self.subTest(archive="zip", body=body),
                self.assertRaises(privacy.PrivacyError),
            ):
                privacy.scan_payload("packages/hidden.zip", zip_stream.getvalue())

            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w:gz") as archive:
                info = tarfile.TarInfo("hidden/")
                info.type = tarfile.DIRTYPE
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
            with (
                self.subTest(archive="tar", body=body),
                self.assertRaises(privacy.PrivacyError),
            ):
                privacy.scan_payload("packages/hidden.tar.gz", tar_stream.getvalue())

        pax_body = b"safe"
        pax_stream = io.BytesIO()
        with tarfile.open(
            fileobj=pax_stream, mode="w:gz", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("hidden/")
            info.type = tarfile.DIRTYPE
            info.size = len(pax_body)
            info.pax_headers = {"size": "0"}
            archive.addfile(info, io.BytesIO(pax_body))
        with self.assertRaisesRegex(
            privacy.PrivacyError, "unsupported tar size override"
        ):
            privacy.scan_payload("packages/pax-hidden.tar.gz", pax_stream.getvalue())

        zip_stream = io.BytesIO()
        with zipfile.ZipFile(
            zip_stream, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            info = zipfile.ZipInfo("hidden/")
            info.create_system = 3
            info.external_attr = (stat.S_IFDIR | 0o755) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, b"")
        mutated = bytearray(zip_stream.getvalue())
        central = mutated.index(b"PK\x01\x02")
        eocd = mutated.rindex(b"PK\x05\x06")
        name_size = int.from_bytes(mutated[26:28], "little")
        extra_size = int.from_bytes(mutated[28:30], "little")
        data_start = 30 + name_size + extra_size
        compressed_size = int.from_bytes(mutated[18:22], "little")
        hidden = b"HIDDEN-BYTES"
        mutated[data_start + compressed_size : data_start + compressed_size] = hidden
        central += len(hidden)
        eocd += len(hidden)
        expanded_size = compressed_size + len(hidden)
        mutated[18:22] = expanded_size.to_bytes(4, "little")
        mutated[central + 20 : central + 24] = expanded_size.to_bytes(4, "little")
        central_offset = int.from_bytes(mutated[eocd + 16 : eocd + 20], "little")
        mutated[eocd + 16 : eocd + 20] = (central_offset + len(hidden)).to_bytes(
            4, "little"
        )
        with self.assertRaisesRegex(
            privacy.PrivacyError,
            "compressed stream has trailing or inconsistent content",
        ):
            privacy.scan_payload("packages/zip-hidden.zip", bytes(mutated))

    def test_tar_member_padding_is_scanned_and_required_to_be_zero(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        original = stream.getvalue()
        with tarfile.open(fileobj=io.BytesIO(original), mode="r:") as archive:
            padding_start = archive.getmembers()[0].offset_data + 1

        for hidden in (b"token=padding-secret", b"\x01"):
            mutated = bytearray(original)
            mutated[padding_start : padding_start + len(hidden)] = hidden
            compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
            compressed = compressor.compress(bytes(mutated)) + compressor.flush()
            with (
                self.subTest(hidden=hidden),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.scan_payload("packages/padded.tar.gz", compressed)
            self.assertNotIn("padding-secret", str(caught.exception))

        pax_stream = io.BytesIO()
        with tarfile.open(
            fileobj=pax_stream, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.pax_headers = {
                "comment": "public",
                "mtime": "1.5",
                "hdrcharset": "ISO-IR 10646 2000 UTF-8",
            }
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        pax_raw = bytearray(pax_stream.getvalue())
        physical = tarfile.TarInfo.frombuf(
            pax_raw[:512], encoding="utf-8", errors="surrogateescape"
        )
        pax_padding = 512 + physical.size
        hidden = b"token=pax-padding-secret"
        pax_raw[pax_padding : pax_padding + len(hidden)] = hidden
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        compressed = compressor.compress(bytes(pax_raw)) + compressor.flush()
        with self.assertRaises(privacy.PrivacyError) as caught:
            privacy.scan_payload("packages/pax-padding.tar.gz", compressed)
        self.assertNotIn("pax-padding-secret", str(caught.exception))

        override_body = b"safe"
        override_stream = io.BytesIO()
        with tarfile.open(
            fileobj=override_stream, mode="w:gz", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("package/hidden.bin")
            info.size = len(override_body)
            info.pax_headers = {"size": "0"}
            archive.addfile(info, io.BytesIO(override_body))
        with self.assertRaisesRegex(
            privacy.PrivacyError, "unsupported tar size override"
        ):
            privacy.scan_payload(
                "packages/pax-size-override.tar.gz", override_stream.getvalue()
            )

    def test_prefixed_and_nested_gzip_and_tar_payloads_are_detected_boundedly(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))
        tar_raw = stream.getvalue()
        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        gzip_raw = compressor.compress(tar_raw) + compressor.flush()

        v7_raw = bytearray(tar_raw)
        v7_raw[257:265] = b"\0" * 8
        v7_raw[148:156] = b" " * 8
        checksum = sum(v7_raw[:512])
        v7_raw[148:156] = f"{checksum:06o}\0 ".encode("ascii")

        for payload in (
            b"launcher-stub" + gzip_raw,
            b"launcher-stub" + tar_raw,
            bytes(v7_raw),
            b"launcher-stub" + bytes(v7_raw),
        ):
            with self.subTest(kind="opaque"), self.assertRaises(privacy.PrivacyError):
                privacy.scan_payload(
                    "packages/launcher.bin",
                    payload,
                    media_type="application/octet-stream",
                )

        nested = _zip_bytes([("package/data.bin", b"launcher-stub" + gzip_raw)])
        with self.assertRaisesRegex(privacy.PrivacyError, "nested archive"):
            privacy.scan_payload("packages/nested.zip", nested)

        invalid_embedded_header = b"x" * 300 + b"ustar" + b"x" * 300
        privacy.scan_payload(
            "packages/not-a-tar.bin",
            invalid_embedded_header,
            media_type="application/octet-stream",
        )

    def test_rejects_assignments_short_credentials_and_private_json_keys(self):
        bad = (
            b"HOSTNAME=runner-42",
            b"USER=alice",
            b"CUDA_VISIBLE_DEVICES=0",
            b"PATH=/usr/bin",
            b"Path=C:/Windows/System32",
            b"cuda_visible_devices=0",
            b"token=x",
            b"tokens=!",
            b"credentials=$X",
            b"Bearer !",
            b"API_KEY=x",
            b"AWS_SECRET_ACCESS_KEY=x",
            b"HOST=runner-42",
            b"COMPUTERNAME=runner-42",
            b"RUNNER_NAME=runner-42",
            b"GITHUB_ACTOR=alice",
            b'{"hostname":"runner-42"}',
            b'{"api_key":"x"}',
            b'{"api-key":"x"}',
            b'{"apikey":"x"}',
            b'{"aws_secret_access_key":"x"}',
            b'{"client_secret":"x"}',
            b'{"PATH":"/usr/bin"}',
            b'{"cuda_visible_devices":"0"}',
        )
        for raw in bad:
            with (
                self.subTest(raw=raw),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.scan_public_bytes(raw)
            self.assertNotIn(raw.decode("ascii"), str(caught.exception))

        stream = io.BytesIO()
        with tarfile.open(
            fileobj=stream, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.pax_headers = {"uname": "private-pax-owner"}
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))
        with self.assertRaises(privacy.PrivacyError) as caught:
            privacy.scan_payload("packages/owner-pax.tar", stream.getvalue())
        self.assertNotIn("private-pax-owner", str(caught.exception))

    def test_nfkc_archive_aliases_cannot_introduce_paths_or_separators(self):
        zip_names = (
            "．．/data.bin",
            "package／data.bin",
            "package＼data.bin",
        )
        for name in zip_names:
            with (
                self.subTest(archive="zip", name=name),
                self.assertRaises(privacy.PrivacyError),
            ):
                privacy.scan_payload(
                    "packages/alias.zip", _zip_bytes([(name, b"safe")])
                )

        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            info = tarfile.TarInfo("．．/data.bin")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))
        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_payload("packages/alias.tar", stream.getvalue())

    def test_rejects_zip_local_central_flag_disagreement_and_strong_encryption(self):
        raw = bytearray(_zip_bytes([("package/data", b"safe")]))
        central = raw.index(b"PK\x01\x02")
        raw[central + 8] |= 0x40
        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_payload("packages/flags.whl", bytes(raw))

        raw = bytearray(_zip_bytes([("package/data", b"safe")]))
        central = raw.index(b"PK\x01\x02")
        raw[6] |= 0x40
        raw[central + 8] |= 0x40
        with self.assertRaisesRegex(privacy.PrivacyError, "encrypted"):
            privacy.scan_payload("packages/encrypted.whl", bytes(raw))

    def test_recursively_scans_sdist_and_rejects_links(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            info = tarfile.TarInfo("package/module.py")
            raw = b"value = 1\n"
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        privacy.scan_payload("packages/safe.tar.gz", stream.getvalue())

        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            info = tarfile.TarInfo("package/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/Users/alice/private"
            archive.addfile(info)
        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_payload("packages/link.tar.gz", stream.getvalue())

    def test_media_suffix_signatures_withheld_roles_and_array_magic_are_closed(self):
        archive = _zip_bytes([("package/data.bin", b"safe")])
        bad = (
            ("packages/disguised.bin", archive, "application/octet-stream"),
            ("packages/prefixed.bin", b"stub" + archive, "application/octet-stream"),
            ("packages/data.bin", b"prefix\x93NUMPYsuffix", "application/octet-stream"),
            ("packages/private.bin", b"safe", "application/octet-stream"),
            ("packages/operations.bin", b"safe", "application/octet-stream"),
            (
                "packages/correctness-arrays.bin",
                b"safe",
                "application/octet-stream",
            ),
            ("packages/data.npz", archive, "application/zip"),
            ("packages/data.bin", b"safe", "application/zip"),
        )
        for name, raw, media_type in bad:
            with self.subTest(name=name, media_type=media_type):
                with self.assertRaises(privacy.PrivacyError):
                    privacy.scan_payload(name, raw, media_type=media_type)

    def test_zip_structure_comments_extras_prefix_trailing_and_nested_are_closed(self):
        safe = _zip_bytes([("package/data.bin", b"safe")])
        for raw in (b"prefix" + safe, safe + b"trailing"):
            with self.subTest(kind="boundary"):
                with self.assertRaises(privacy.PrivacyError):
                    privacy.scan_payload("packages/unsafe.zip", raw)

        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.comment = b"/Users/alice/private"
            archive.writestr("package/data.bin", b"safe")
        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_payload("packages/comment.zip", stream.getvalue())

        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("package/data.bin")
            leak = b"/Users/alice/private"
            info.extra = struct.pack("<HH", 0xCAFE, len(leak)) + leak
            archive.writestr(info, b"safe")
        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_payload("packages/extra.zip", stream.getvalue())

        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("package/data.bin")
            original_name = info.filename.encode("utf-8")
            alternate_name = b"../escape.bin"
            unicode_path = (
                b"\x01"
                + zlib.crc32(original_name).to_bytes(4, "little")
                + alternate_name
            )
            info.extra = struct.pack("<HH", 0x7075, len(unicode_path)) + unicode_path
            archive.writestr(info, b"safe")
        with self.assertRaisesRegex(privacy.PrivacyError, "unsupported ZIP metadata"):
            privacy.scan_payload("packages/unicode-path.zip", stream.getvalue())

        inner_stream = io.BytesIO()
        with zipfile.ZipFile(
            inner_stream, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("hidden/data.bin", b"/Users/alice/private")
        outer = _zip_bytes([("package/hidden.zip", inner_stream.getvalue())])
        with self.assertRaisesRegex(privacy.PrivacyError, "nested archive"):
            privacy.scan_payload("packages/nested.zip", outer)

        aliased = _zip_bytes([("package/name", b"one"), ("package/name.", b"two")])
        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_payload("packages/alias.zip", aliased)

    def test_zip_parser_exceptions_are_chain_free_and_non_leaking(self):
        marker = "zip-parser-canary-q7m9tag"
        member_name = f"{marker}.bin"
        safe = _zip_bytes([(member_name, b"safe")])

        damaged = bytearray(safe)
        with zipfile.ZipFile(io.BytesIO(safe)) as archive:
            info = archive.infolist()[0]
        name_size, extra_size = struct.unpack_from(
            "<HH", damaged, info.header_offset + 26
        )
        damaged[info.header_offset + 30 + name_size + extra_size] ^= 1
        with self.assertRaises(privacy.PrivacyError) as caught:
            privacy.scan_payload("packages/crc-failure.zip", bytes(damaged))
        _assert_sanitized_privacy_error(self, caught.exception, marker)

        fake_archive = mock.MagicMock()
        fake_archive.infolist.side_effect = RecursionError(marker)

        class CloseFailureArchive:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __getattr__(self, attribute):
                return getattr(self._wrapped, attribute)

            def close(self):
                self._wrapped.close()
                raise RuntimeError(marker)

        close_failure = CloseFailureArchive(zipfile.ZipFile(io.BytesIO(safe)))
        stages = (
            (
                "constructor",
                mock.patch.object(
                    privacy.zipfile,
                    "ZipFile",
                    side_effect=zipfile.BadZipFile(marker),
                ),
            ),
            (
                "infolist",
                mock.patch.object(
                    privacy.zipfile,
                    "ZipFile",
                    return_value=fake_archive,
                ),
            ),
            (
                "member-read",
                mock.patch.object(
                    privacy.zipfile.ZipFile,
                    "read",
                    side_effect=zipfile.BadZipFile(marker),
                ),
            ),
            (
                "testzip",
                mock.patch.object(
                    privacy.zipfile.ZipFile,
                    "testzip",
                    side_effect=RuntimeError(marker),
                ),
            ),
            (
                "close",
                mock.patch.object(
                    privacy.zipfile,
                    "ZipFile",
                    return_value=close_failure,
                ),
            ),
        )
        for stage, patch in stages:
            with (
                self.subTest(stage=stage),
                patch,
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.scan_payload("packages/library-failure.zip", safe)
            _assert_sanitized_privacy_error(self, caught.exception, marker)

        primary_close_failure = CloseFailureArchive(zipfile.ZipFile(io.BytesIO(safe)))
        with (
            mock.patch.object(
                privacy,
                "_validate_zip_structure",
                side_effect=privacy.PrivacyError("safe primary rejection"),
            ),
            mock.patch.object(
                privacy.zipfile,
                "ZipFile",
                return_value=primary_close_failure,
            ),
            self.assertRaisesRegex(
                privacy.PrivacyError, "safe primary rejection"
            ) as caught,
        ):
            privacy.scan_payload("packages/primary.zip", safe)
        _assert_sanitized_privacy_error(self, caught.exception, marker)

        with (
            mock.patch.object(
                privacy.struct,
                "unpack_from",
                side_effect=struct.error(marker),
            ),
            self.assertRaises(privacy.PrivacyError) as caught,
        ):
            privacy._zip_unpack("<H", b"\0\0", 0, "ZIP metadata is malformed")
        _assert_sanitized_privacy_error(self, caught.exception, marker)

        with self.assertRaises(privacy.PrivacyError) as caught:
            privacy._zip_decode(
                b"\xff" + marker.encode(),
                "utf-8",
                "ZIP name encoding is invalid",
            )
        _assert_sanitized_privacy_error(self, caught.exception, marker)

        deflate = io.BytesIO()
        with zipfile.ZipFile(deflate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("package/data.bin", b"safe")
        with (
            mock.patch.object(
                privacy.zlib,
                "decompressobj",
                side_effect=zlib.error(marker),
            ),
            self.assertRaises(privacy.PrivacyError) as caught,
        ):
            privacy.scan_payload("packages/deflate-failure.zip", deflate.getvalue())
        _assert_sanitized_privacy_error(self, caught.exception, marker)

    def test_strict_and_contextual_json_exceptions_are_fully_sanitized(self):
        marker = "json-parser-canary-q7m9tag"
        malformed = b'{"public":"' + b"\xff" + marker.encode() + b'"}'
        with self.assertRaises(privacy.PrivacyError) as caught:
            privacy._strict_json(malformed, "private JSON")
        _assert_sanitized_privacy_error(self, caught.exception, marker)

        with (
            mock.patch.object(
                privacy.json,
                "loads",
                side_effect=RecursionError(marker),
            ),
            self.assertRaises(privacy.PrivacyError) as caught,
        ):
            privacy._strict_json(b"{}", "private JSON")
        _assert_sanitized_privacy_error(self, caught.exception, marker)

        valid = b'{"public":"safe"}'
        zip_raw = _zip_bytes([("package/config.json", malformed)])
        valid_zip_raw = _zip_bytes([("package/config.json", valid)])
        tar_raw = _physical_tar_bytes(
            [_ordinary_tar_record("package/config.json", malformed)]
        )
        valid_tar_raw = _physical_tar_bytes(
            [_ordinary_tar_record("package/config.json", valid)]
        )
        containers = (
            (
                "zip",
                "packages/config.zip",
                zip_raw,
                valid_zip_raw,
                "application/zip",
            ),
            (
                "wheel",
                "packages/config.whl",
                zip_raw,
                valid_zip_raw,
                "application/vnd.python.wheel",
            ),
            (
                "tar",
                "packages/config.tar",
                tar_raw,
                valid_tar_raw,
                "application/x-tar",
            ),
            (
                "gzip",
                "packages/config.tar.gz",
                _gzip_bytes(tar_raw),
                _gzip_bytes(valid_tar_raw),
                "application/gzip",
            ),
        )
        for stage, name, raw, parser_raw, media_type in containers:
            with (
                self.subTest(stage=stage, failure="decode"),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.scan_payload(name, raw, media_type=media_type)
            _assert_sanitized_privacy_error(self, caught.exception, marker)

            with (
                self.subTest(stage=stage, failure="parser"),
                mock.patch.object(
                    privacy.json,
                    "loads",
                    side_effect=ValueError(marker),
                ),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.scan_payload(name, parser_raw, media_type=media_type)
            _assert_sanitized_privacy_error(self, caught.exception, marker)

    def test_wheel_source_schema_words_are_safe_but_json_identity_values_are_not(self):
        safe = _zip_bytes(
            [("package/module.py", b'CONFIG = {"public_label": "placeholder"}\n')]
        )
        privacy.scan_payload("packages/source.whl", safe)

        unsafe = _zip_bytes(
            [("package/config.json", _json_bytes({"hostname": "private-host"}))]
        )
        with self.assertRaisesRegex(privacy.PrivacyError, "private metadata"):
            privacy.scan_payload("packages/metadata.whl", unsafe)

    def test_benign_pax_and_gnu_names_reconcile_without_getmembers(self):
        local_stream = io.BytesIO()
        with tarfile.open(
            fileobj=local_stream, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.pax_headers = {"comment": "public"}
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))

        global_stream = io.BytesIO()
        with tarfile.open(
            fileobj=global_stream,
            mode="w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": "public"},
        ) as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))

        identical_stream = io.BytesIO()
        with tarfile.open(
            fileobj=identical_stream,
            mode="w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": "public"},
        ) as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.pax_headers = {"comment": "public"}
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))

        pax_name = "package/" + "p" * 120 + ".bin"
        pax_path_stream = io.BytesIO()
        with tarfile.open(
            fileobj=pax_path_stream, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo(pax_name)
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))

        gnu_name = "package/" + "g" * 120 + ".bin"
        gnu_stream = io.BytesIO()
        with tarfile.open(
            fileobj=gnu_stream, mode="w", format=tarfile.GNU_FORMAT
        ) as archive:
            info = tarfile.TarInfo(gnu_name)
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))

        shadowed_path = _physical_tar_bytes(
            [
                _pax_helper(tarfile.XGLTYPE, (("path", "package/global-name.bin"),)),
                _pax_helper(tarfile.XHDTYPE, (("path", "package/local-name.bin"),)),
                _ordinary_tar_record("package/physical-name.bin"),
            ]
        )

        archives = (
            ("packages/local-pax.tar", local_stream.getvalue()),
            ("packages/global-pax.tar", global_stream.getvalue()),
            ("packages/identical-pax.tar", identical_stream.getvalue()),
            ("packages/path-pax.tar", pax_path_stream.getvalue()),
            ("packages/path-pax.tar.gz", _gzip_bytes(pax_path_stream.getvalue())),
            ("packages/long-gnu.tar", gnu_stream.getvalue()),
        )
        identical_allowed = tuple(
            (
                f"packages/identical-{key}.tar",
                _physical_tar_bytes(
                    [
                        _pax_helper(tarfile.XGLTYPE, ((key, value),)),
                        _pax_helper(tarfile.XHDTYPE, ((key, value),)),
                        _ordinary_tar_record(),
                    ]
                ),
            )
            for key, value in (
                ("mtime", "1.5"),
                ("uid", "0"),
                ("gid", "0"),
                ("uname", ""),
                ("gname", ""),
                ("hdrcharset", "ISO-IR 10646 2000 UTF-8"),
            )
        )
        with (
            mock.patch.object(
                privacy.tarfile.TarFile,
                "getmembers",
                side_effect=AssertionError("unbounded logical list"),
            ) as getmembers,
            mock.patch.object(
                privacy.tarfile.TarFile,
                "extract",
                side_effect=AssertionError("filesystem extraction"),
            ) as extract,
            mock.patch.object(
                privacy.tarfile.TarFile,
                "extractall",
                side_effect=AssertionError("filesystem extraction"),
            ) as extractall,
        ):
            for name, raw in archives + identical_allowed:
                with self.subTest(name=name):
                    privacy.scan_payload(name, raw)
        getmembers.assert_not_called()
        extract.assert_not_called()
        extractall.assert_not_called()

        with (
            mock.patch.object(
                privacy.tarfile,
                "open",
                side_effect=AssertionError("logical parser entered"),
            ) as opened,
            self.assertRaisesRegex(privacy.PrivacyError, "unsupported global PAX path"),
        ):
            privacy.scan_payload("packages/global-path-pax.tar", shadowed_path)
        opened.assert_not_called()

        with mock.patch.object(privacy, "PRIVACY_TEXT_INPUT_CHUNK_BYTES", 519):
            privacy.scan_payload("packages/path-pax.tar", pax_path_stream.getvalue())

        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_public_bytes(b"path=package/public.bin")

        with (
            mock.patch.object(
                privacy.tarfile,
                "open",
                side_effect=AssertionError("logical parser entered"),
            ) as opened,
            self.assertRaisesRegex(privacy.PrivacyError, "private opening"),
        ):
            privacy.scan_payload(
                "packages/path-pax.tar",
                pax_path_stream.getvalue(),
                forbidden_values=("path=",),
            )
        opened.assert_not_called()

    def test_parser_active_pax_controls_are_rejected_before_tarfile_open(self):
        controls = (
            ("size", "4", "unsupported tar size override"),
            ("linkpath", "package/other.bin", "unsupported link path"),
            ("GNU.sparse.map", "0,4", "unsupported tar size override"),
            ("GNU.sparse.size", "4", "unsupported tar size override"),
            ("GNU.sparse.major", "1", "unsupported tar size override"),
            ("GNU.sparse.minor", "0", "unsupported tar size override"),
            ("GNU.sparse.realsize", "4", "unsupported tar size override"),
            ("SCHILY.realsize", "4", "unsupported tar size override"),
            ("SCHILY.filetype", "sparse", "unsupported tar size override"),
            (
                "SCHILY.xattr.security.selinux",
                "private-shadow-marker",
                "unsupported tar metadata override",
            ),
            (
                "LIBARCHIVE.xattr.security.selinux",
                "public",
                "unsupported tar metadata override",
            ),
            ("security.selinux", "public", "unsupported tar metadata override"),
            ("HOST", "public-host", "unsupported tar metadata override"),
            ("hdrcharset", "BINARY", "unsupported tar character encoding"),
            ("mtime", "1e1000000", "invalid tar timestamp"),
            ("mtime", "1.1234567890", "invalid tar timestamp"),
            ("mtime", "9223372036854775808", "invalid tar timestamp"),
        )
        for key, value, expected in controls:
            raw = _physical_tar_bytes(
                [
                    _pax_helper(tarfile.XGLTYPE, ((key, value),)),
                    _pax_helper(tarfile.XGLTYPE, ((key, ""),)),
                    _ordinary_tar_record(),
                ]
            )
            with (
                self.subTest(key=key),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaisesRegex(privacy.PrivacyError, expected) as caught,
            ):
                privacy.scan_payload("packages/structural.tar", raw)
            opened.assert_not_called()
            self.assertNotIn(
                "private-shadow-marker",
                " ".join(_exception_messages(caught.exception)),
            )

    def test_contiguous_tar_scan_closes_physical_record_boundaries(self):
        header_body = bytearray(
            _physical_tar_bytes(
                [_ordinary_tar_record("package/header-body.bin", b"alice/public")]
            )
        )
        header_body[506:512] = b"/home/"
        header_body[148:156] = b" " * 8
        checksum = sum(header_body[:512])
        header_body[148:156] = f"{checksum:06o}\0 ".encode("ascii")
        self.assertTrue(privacy._tar_checksum_matches(bytes(header_body[:512])))

        body_header = _physical_tar_bytes(
            [
                _ordinary_tar_record(
                    "package/body-header.bin",
                    b"a" * (512 - len(b"token=")) + b"token=",
                ),
                _ordinary_tar_record("x", b"safe"),
            ]
        )
        header_header = bytearray(
            _physical_tar_bytes(
                [
                    _ordinary_tar_record("package/empty.bin", b""),
                    _ordinary_tar_record("=synthetic-node", b"safe"),
                ]
            )
        )
        header_header[508:512] = b"HOST"
        header_header[148:156] = b" " * 8
        checksum = sum(header_header[:512])
        header_header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
        self.assertTrue(privacy._tar_checksum_matches(bytes(header_header[:512])))
        boundary_cases = (
            ("header-body.tar", bytes(header_body)),
            ("header-body.tar.gz", _gzip_bytes(bytes(header_body))),
            ("body-header.tar", body_header),
            ("body-header.tar.gz", _gzip_bytes(body_header)),
            ("header-header.tar", bytes(header_header)),
            ("header-header.tar.gz", _gzip_bytes(bytes(header_header))),
        )
        for name, raw in boundary_cases:
            with (
                self.subTest(name=name),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy.scan_payload(f"packages/{name}", raw)
            opened.assert_not_called()
            self.assertNotIn("alice", str(caught.exception))
            self.assertNotIn("token=x", str(caught.exception))

        with (
            mock.patch.object(privacy, "PRIVACY_TEXT_INPUT_CHUNK_BYTES", 512),
            mock.patch.object(
                privacy.tarfile,
                "open",
                side_effect=AssertionError("logical parser entered"),
            ) as opened,
            self.assertRaises(privacy.PrivacyError),
        ):
            privacy.scan_payload("packages/rolling-boundary.tar", bytes(header_body))
        opened.assert_not_called()

        padding_boundary = _physical_tar_bytes(
            [
                _ordinary_tar_record("package/padded.bin", b"Q"),
                _ordinary_tar_record("tail", b"safe"),
            ]
        )
        padding_opening = "Q" + "\0" * 511 + "tail"
        for name, raw in (
            ("padding-boundary.tar", padding_boundary),
            ("padding-boundary.tar.gz", _gzip_bytes(padding_boundary)),
        ):
            with (
                self.subTest(name=name),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaisesRegex(privacy.PrivacyError, "private opening"),
            ):
                privacy.scan_payload(
                    f"packages/{name}",
                    raw,
                    forbidden_values=(padding_opening,),
                )
            opened.assert_not_called()

        ordinary_path_assignment = _physical_tar_bytes(
            [_ordinary_tar_record(body=b"path=package/public.bin")]
        )
        with (
            mock.patch.object(
                privacy.tarfile,
                "open",
                side_effect=AssertionError("logical parser entered"),
            ) as opened,
            self.assertRaisesRegex(
                privacy.PrivacyError, "environment or identity assignment"
            ),
        ):
            privacy.scan_payload("packages/ordinary-path.tar", ordinary_path_assignment)
        opened.assert_not_called()

        pax_values = (
            (
                "path",
                "package/HOST=synthetic-node.bin",
                "environment or identity assignment",
            ),
            (
                "comment",
                "path=package/public.bin",
                "environment or identity assignment",
            ),
        )
        for key, value, expected in pax_values:
            raw = _physical_tar_bytes(
                [
                    _pax_helper(tarfile.XHDTYPE, ((key, value),)),
                    _ordinary_tar_record(),
                ]
            )
            with (
                self.subTest(key=key),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaisesRegex(privacy.PrivacyError, expected),
            ):
                privacy.scan_payload("packages/pax-value.tar", raw)
            opened.assert_not_called()

    def test_effective_pax_association_budget_precedes_member_copy_and_open(self):
        raw = _physical_tar_bytes(
            [
                _pax_helper(
                    tarfile.XGLTYPE,
                    (("comment", "public"), ("mtime", "1")),
                ),
                _ordinary_tar_record("package/one.bin"),
                _ordinary_tar_record("package/two.bin"),
            ]
        )
        merge = privacy._merge_pax_items
        with (
            mock.patch.object(privacy, "MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS", 3),
            mock.patch.object(privacy, "_merge_pax_items", wraps=merge) as merged,
            mock.patch.object(
                privacy.tarfile,
                "open",
                side_effect=AssertionError("logical parser entered"),
            ) as opened,
            self.assertRaisesRegex(
                privacy.PrivacyError, "effective PAX associations exceed their bound"
            ),
        ):
            privacy.scan_payload("packages/pax-product.tar", raw)
        opened.assert_not_called()
        self.assertEqual(merged.call_count, 2)

        for local_items in (
            (("comment", "updated"),),
            (("comment", "public"),),
        ):
            bounded_overlay = _physical_tar_bytes(
                [
                    _pax_helper(
                        tarfile.XGLTYPE,
                        (("comment", "public"), ("mtime", "1")),
                    ),
                    _pax_helper(tarfile.XHDTYPE, local_items),
                    _ordinary_tar_record(),
                ]
            )
            with (
                self.subTest(local_items=local_items),
                mock.patch.object(privacy, "MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS", 2),
            ):
                privacy.scan_payload("packages/pax-overlay.tar", bounded_overlay)

        new_local_key = _physical_tar_bytes(
            [
                _pax_helper(
                    tarfile.XGLTYPE,
                    (("comment", "public"), ("mtime", "1")),
                ),
                _pax_helper(
                    tarfile.XHDTYPE,
                    (("hdrcharset", "ISO-IR 10646 2000 UTF-8"),),
                ),
                _ordinary_tar_record(),
            ]
        )
        with (
            mock.patch.object(privacy, "MAX_TAR_EFFECTIVE_PAX_ASSOCIATIONS", 2),
            mock.patch.object(privacy, "_merge_pax_items", wraps=merge) as merged,
            mock.patch.object(
                privacy.tarfile,
                "open",
                side_effect=AssertionError("logical parser entered"),
            ) as opened,
            self.assertRaisesRegex(
                privacy.PrivacyError, "effective PAX associations exceed their bound"
            ),
        ):
            privacy.scan_payload("packages/pax-new-key.tar", new_local_key)
        opened.assert_not_called()
        self.assertEqual(merged.call_count, 1)

    def test_extraction_active_vendor_pax_metadata_precedes_tarfile_open(self):
        vendor_fields = (
            ("SUN.holesdata", "0"),
            ("SCHILY.acl.access", "public"),
            ("SCHILY.acl.default", "public"),
            ("SCHILY.acl.ace", "public"),
            ("RHT.security.selinux", "private-vendor-marker"),
            ("LIBARCHIVE.symlinktype", "file"),
            ("SCHILY.devmajor", "0"),
            ("SCHILY.devminor", "0"),
            ("SCHILY.filetype", "regular"),
            ("SCHILY.ino", "1"),
            ("SCHILY.dev", "1"),
            ("SCHILY.nlink", "1"),
            ("SCHILY.nlinks", "1"),
            ("SUN.devmajor", "0"),
            ("SUN.devminor", "0"),
            ("LIBARCHIVE.creationtime", "1"),
            ("GNU.dumpdir", "safe"),
            ("RHT.unknown", "safe"),
            ("realtime.any", "safe"),
            ("atime", "1"),
            ("ctime", "1"),
            ("charset", "UTF-8"),
            ("note", "safe"),
        )
        for key, value in vendor_fields:
            raw = _physical_tar_bytes(
                [
                    _pax_helper(tarfile.XHDTYPE, ((key, value),)),
                    _ordinary_tar_record(),
                ]
            )
            with (
                self.subTest(key=key),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaisesRegex(
                    privacy.PrivacyError, "unsupported tar metadata override"
                ) as caught,
            ):
                privacy.scan_payload("packages/vendor.tar", raw)
            opened.assert_not_called()
            self.assertNotIn(
                "private-vendor-marker",
                " ".join(_exception_messages(caught.exception)),
            )

    def test_gzip_expansion_ratio_is_bounded_before_physical_tar_parsing(self):
        tar_raw = _physical_tar_bytes([_ordinary_tar_record(body=b"\0" * 64 * 1024)])
        compressed = _gzip_bytes(tar_raw)
        self.assertGreater(
            len(tar_raw), len(compressed) * privacy.MAX_COMPRESSION_RATIO
        )
        privacy.scan_payload("packages/floor-compatible.tar.gz", compressed)

        with (
            mock.patch.object(privacy, "MIN_TAR_GZIP_RATIO_BYTES", 0),
            mock.patch.object(privacy, "MAX_COMPRESSION_RATIO", 2),
            mock.patch.object(
                privacy,
                "_scan_tar_physical_records",
                side_effect=AssertionError("physical parser entered"),
            ) as physical_scan,
            self.assertRaisesRegex(privacy.PrivacyError, "oversized"),
        ):
            privacy.scan_payload("packages/ratio-bomb.tar.gz", compressed)
        physical_scan.assert_not_called()

        inflated_denominator = compressed + b"x" * (len(compressed) * 10)
        with (
            mock.patch.object(privacy, "MIN_TAR_GZIP_RATIO_BYTES", 0),
            mock.patch.object(privacy, "MAX_COMPRESSION_RATIO", 2),
            mock.patch.object(
                privacy,
                "_scan_tar_physical_records",
                side_effect=AssertionError("physical parser entered"),
            ) as physical_scan,
            self.assertRaisesRegex(privacy.PrivacyError, "oversized"),
        ):
            privacy.scan_payload(
                "packages/trailing-ratio-bomb.tar.gz", inflated_denominator
            )
        physical_scan.assert_not_called()

        safe_tar = _physical_tar_bytes([_ordinary_tar_record()])
        safe_gzip = _gzip_bytes(safe_tar)
        real_decompressobj = zlib.decompressobj
        input_sizes = []
        output_limits = []
        flush_calls = []

        class RecordingDecompressor:
            def __init__(self, *args, **kwargs):
                self._wrapped = real_decompressobj(*args, **kwargs)

            def decompress(self, data, max_length=0):
                input_sizes.append(len(data))
                output_limits.append(max_length)
                return self._wrapped.decompress(data, max_length)

            def flush(self, *args, **kwargs):
                flush_calls.append(True)
                return self._wrapped.flush(*args, **kwargs)

            def __getattr__(self, attribute):
                return getattr(self._wrapped, attribute)

        with (
            mock.patch.object(privacy, "TAR_GZIP_INPUT_CHUNK_BYTES", 7),
            mock.patch.object(privacy, "TAR_GZIP_OUTPUT_CHUNK_BYTES", 29),
            mock.patch.object(
                privacy.zlib,
                "decompressobj",
                side_effect=lambda *args, **kwargs: RecordingDecompressor(
                    *args, **kwargs
                ),
            ),
        ):
            privacy.scan_payload("packages/chunked.tar.gz", safe_gzip)
        self.assertGreater(len(input_sizes), 1)
        self.assertTrue(all(0 < size <= 7 for size in input_sizes))
        self.assertTrue(all(0 < limit <= 29 for limit in output_limits))
        self.assertEqual(flush_calls, [])

        for malformed in (
            safe_gzip + safe_gzip,
            safe_gzip + b"trailer-marker",
            safe_gzip[:-1],
        ):
            with (
                self.subTest(malformed_length=len(malformed)),
                mock.patch.object(
                    privacy,
                    "_scan_tar_physical_records",
                    side_effect=AssertionError("physical parser entered"),
                ) as physical_scan,
                self.assertRaisesRegex(
                    privacy.PrivacyError,
                    "trailing, concatenated, or oversized",
                ),
            ):
                privacy.scan_payload("packages/malformed.tar.gz", malformed)
            physical_scan.assert_not_called()

        class FailingDecompressor:
            def decompress(self, _data, _max_length):
                raise zlib.error("private-gzip-marker")

        with (
            mock.patch.object(
                privacy.zlib, "decompressobj", return_value=FailingDecompressor()
            ),
            self.assertRaisesRegex(
                privacy.PrivacyError, "is not a valid gzip payload"
            ) as caught,
        ):
            privacy.scan_payload("packages/invalid.tar.gz", safe_gzip)
        self.assertNotIn(
            "private-gzip-marker",
            " ".join(_exception_messages(caught.exception)),
        )

    def test_tar_helper_budgets_and_physical_member_bound_precede_open(self):
        pax_one = _pax_helper(tarfile.XGLTYPE, (("comment", "one"),))
        pax_two = _pax_helper(tarfile.XGLTYPE, (("mtime", "2"),))
        pax_size = len(pax_one[2])
        gnu_body = b"package/long-public-name.bin\0"
        cases = (
            (
                "PAX byte bound",
                _physical_tar_bytes([pax_one, _ordinary_tar_record()]),
                {"MAX_TAR_PAX_HELPER_BYTES": pax_size - 1},
            ),
            (
                "GNU long-name byte bound",
                _physical_tar_bytes(
                    [
                        ("././@LongLink", tarfile.GNUTYPE_LONGNAME, gnu_body),
                        _ordinary_tar_record(),
                    ]
                ),
                {"MAX_TAR_GNU_LONGNAME_BYTES": len(gnu_body) - 1},
            ),
            (
                "helpers exceed their byte bound",
                _physical_tar_bytes([pax_one, pax_two, _ordinary_tar_record()]),
                {"MAX_TAR_HELPER_TOTAL_BYTES": len(pax_one[2]) + len(pax_two[2]) - 1},
            ),
            (
                "PAX record closure",
                _physical_tar_bytes([pax_one, pax_two, _ordinary_tar_record()]),
                {"MAX_TAR_PAX_RECORDS": 1},
            ),
            (
                "consecutive tar helpers",
                _physical_tar_bytes([pax_one, pax_two, _ordinary_tar_record()]),
                {"MAX_TAR_CONSECUTIVE_HELPERS": 1},
            ),
            (
                "too many members",
                _physical_tar_bytes(
                    [
                        _ordinary_tar_record("package/one.bin"),
                        _ordinary_tar_record("package/two.bin"),
                    ]
                ),
                {"MAX_ARCHIVE_MEMBERS": 1},
            ),
        )
        for expected, raw, limits in cases:
            patches = [
                mock.patch.object(privacy, attribute, value)
                for attribute, value in limits.items()
            ]
            for patch in patches:
                patch.start()
            try:
                with (
                    self.subTest(expected=expected),
                    mock.patch.object(
                        privacy.tarfile,
                        "open",
                        side_effect=AssertionError("logical parser entered"),
                    ) as opened,
                    self.assertRaisesRegex(privacy.PrivacyError, expected),
                ):
                    privacy.scan_payload("packages/bounded.tar", raw)
                opened.assert_not_called()
            finally:
                for patch in reversed(patches):
                    patch.stop()

    def test_malformed_and_conflicting_tar_helpers_fail_before_open(self):
        local = _pax_helper(tarfile.XHDTYPE, (("comment", "public"),))
        global_same = _pax_helper(tarfile.XGLTYPE, (("comment", "public"),))
        path = _pax_helper(tarfile.XHDTYPE, (("path", "package/pax-name.bin"),))
        longname = (
            "././@LongLink",
            tarfile.GNUTYPE_LONGNAME,
            b"package/gnu-name.bin\0",
        )
        invalid = (
            _physical_tar_bytes([("././@PaxHeader", tarfile.XHDTYPE, b"")]),
            _physical_tar_bytes([local]),
            _physical_tar_bytes([longname]),
            _physical_tar_bytes([global_same]),
            _physical_tar_bytes([local, local, _ordinary_tar_record()]),
            _physical_tar_bytes([path, longname, _ordinary_tar_record()]),
            _physical_tar_bytes(
                [
                    ("././@LongLink", tarfile.GNUTYPE_LONGLINK, b"target\0"),
                    _ordinary_tar_record(),
                ]
            ),
            _physical_tar_bytes(
                [
                    ("package/sparse.bin", tarfile.GNUTYPE_SPARSE, b"safe"),
                ]
            ),
        )
        for index, raw in enumerate(invalid):
            with (
                self.subTest(index=index),
                mock.patch.object(
                    privacy.tarfile,
                    "open",
                    side_effect=AssertionError("logical parser entered"),
                ) as opened,
                self.assertRaises(privacy.PrivacyError),
            ):
                privacy.scan_payload("packages/helpers.tar", raw)
            opened.assert_not_called()

    def test_tar_logical_members_reconcile_with_the_immutable_ledger(self):
        raw = _physical_tar_bytes(
            [
                _ordinary_tar_record("package/one.bin", b"one"),
                _ordinary_tar_record("package/two.bin", b"two"),
            ]
        )
        ledger = privacy._scan_tar_physical_records("fixture.tar", raw, ())
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            infos = list(archive)

        privacy._reconcile_tar_member("fixture.tar", infos[0], ledger.members[0])
        mismatches = []
        changed = copy.deepcopy(infos[0])
        changed.name = "package/private-marker.bin"
        mismatches.append(changed)
        changed = copy.deepcopy(infos[0])
        changed.offset_data += 512
        mismatches.append(changed)
        changed = copy.deepcopy(infos[0])
        changed.size += 1
        mismatches.append(changed)
        changed = copy.deepcopy(infos[0])
        changed.type = tarfile.DIRTYPE
        mismatches.append(changed)
        changed = copy.deepcopy(infos[0])
        changed.pax_headers = {"comment": "private-marker"}
        mismatches.append(changed)
        changed = copy.deepcopy(infos[0])
        changed.sparse = [(0, 3)]
        mismatches.append(changed)
        mismatches.append(infos[1])

        for changed in mismatches:
            with (
                self.subTest(field=changed),
                self.assertRaises(privacy.PrivacyError) as caught,
            ):
                privacy._reconcile_tar_member("fixture.tar", changed, ledger.members[0])
            self.assertNotIn("private-marker", str(caught.exception))

        for logical_members in ([], infos + [copy.deepcopy(infos[-1])]):
            fake_archive = mock.MagicMock()
            fake_archive.__iter__.return_value = iter(logical_members)
            fake_archive.pax_headers = {}
            fake_archive.extractfile.side_effect = lambda info: io.BytesIO(
                raw[info.offset_data : info.offset_data + info.size]
            )
            with (
                mock.patch.object(privacy.tarfile, "open", return_value=fake_archive),
                self.assertRaisesRegex(privacy.PrivacyError, "membership differs"),
            ):
                privacy._scan_tar("fixture.tar", raw, ())

    def test_tar_recursion_error_is_converted_without_private_context(self):
        raw = _physical_tar_bytes([_ordinary_tar_record()])
        marker = "private-recursion-marker"
        with (
            mock.patch.object(
                privacy.tarfile, "open", side_effect=RecursionError(marker)
            ),
            self.assertRaisesRegex(
                privacy.PrivacyError, "is not a valid tar payload"
            ) as caught,
        ):
            privacy.scan_payload("packages/recursive.tar", raw)
        self.assertNotIn(marker, " ".join(_exception_messages(caught.exception)))

    def test_physical_tar_headers_and_helper_bodies_precede_logical_overrides(self):
        owner_stream = io.BytesIO()
        with tarfile.open(
            fileobj=owner_stream, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.uid = 12345
            info.gid = 23456
            info.uname = "synthetic-owner"
            info.gname = "synthetic-group"
            info.pax_headers = {"uid": "0", "gid": "0", "uname": "", "gname": ""}
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))
        with self.assertRaisesRegex(privacy.PrivacyError, "physical tar owner"):
            privacy.scan_payload("packages/owner-override.tar", owner_stream.getvalue())

        pax_owner_stream = io.BytesIO()
        with tarfile.open(
            fileobj=pax_owner_stream, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.pax_headers = {"uid": "12345"}
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))
        with self.assertRaisesRegex(privacy.PrivacyError, "owner identifiers"):
            privacy.scan_payload(
                "packages/pax-owner-id.tar", pax_owner_stream.getvalue()
            )

        path_stream = io.BytesIO()
        with tarfile.open(
            fileobj=path_stream, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("../synthetic-escape.bin")
            info.pax_headers = {"path": "package/data.bin"}
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))
        with self.assertRaisesRegex(privacy.PrivacyError, "physical tar path"):
            privacy._scan_tar("packages/path-override.tar", path_stream.getvalue(), ())

        nested_zip = _zip_bytes([])
        pax_stream = io.BytesIO()
        with tarfile.open(
            fileobj=pax_stream, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("package/data.bin")
            info.pax_headers = {"comment": nested_zip.decode("ascii")}
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))
        with self.assertRaisesRegex(privacy.PrivacyError, "nested archive"):
            privacy.scan_payload("packages/pax-nested.tar", pax_stream.getvalue())

        gnu_helper = tarfile.TarInfo("././@LongLink")
        gnu_helper.type = tarfile.GNUTYPE_LONGNAME
        with self.assertRaisesRegex(privacy.PrivacyError, "nested archive"):
            privacy._scan_tar_helper_body(
                gnu_helper, nested_zip, "GNU helper fixture", ()
            )

        gnu_stream = io.BytesIO()
        with tarfile.open(
            fileobj=gnu_stream, mode="w", format=tarfile.GNU_FORMAT
        ) as archive:
            info = tarfile.TarInfo("package/" + "a" * 120 + ".bin")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"safe"))
        privacy.scan_payload("packages/gnu-long-name.tar", gnu_stream.getvalue())

    def test_embedded_gzip_candidate_work_is_capped_fail_closed(self):
        storm = b"\x1f\x8b\x08\xe0" * (privacy.MAX_EMBEDDED_ARCHIVE_CANDIDATES + 1)
        with self.assertRaisesRegex(privacy.PrivacyError, "archive"):
            privacy.scan_payload(
                "packages/signature-storm.bin",
                storm,
                media_type="application/octet-stream",
            )

    def test_windows_nonportable_payload_aliases_are_rejected(self):
        names = (
            "C:relative/evidence.bin",
            "evidence/CON.bin",
            "evidence/file.bin:stream.bin",
            "evidence/name./data.bin",
            "evidence/COM\N{SUPERSCRIPT ONE}.bin",
            "．．/evidence.bin",
        )
        for name in names:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(privacy.PrivacyError, "portable"),
            ):
                privacy.scan_payload(
                    name, b"safe", media_type="application/octet-stream"
                )

    def test_general_text_scans_apply_nfkc_and_casefold(self):
        documents = (
            {_fullwidth_ascii("hostname"): "samplehost"},
            {"public_label": _fullwidth_ascii("C:/Users/sampleuser/private-location")},
        )
        for document in documents:
            raw = json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            with (
                self.subTest(document=document),
                self.assertRaises(privacy.PrivacyError),
            ):
                privacy.scan_public_bytes(raw)

        utf16_alias = _fullwidth_ascii("PATH=/home/sampleuser/private").encode(
            "utf-16-le"
        )
        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_public_bytes(utf16_alias)

    def test_tar_owner_pax_trailer_and_gzip_expansion_are_closed(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            info = tarfile.TarInfo("package/module.py")
            info.uname = "fixture-owner-invalid"
            info.gname = "fixture-group-invalid"
            raw = b"value = 1\n"
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        with self.assertRaisesRegex(privacy.PrivacyError, "owner names") as caught:
            privacy.scan_payload(
                "packages/owner.tar",
                stream.getvalue(),
            )
        self.assertNotIn("fixture-owner-invalid", str(caught.exception))
        self.assertNotIn("fixture-group-invalid", str(caught.exception))

        stream = io.BytesIO()
        with tarfile.open(
            fileobj=stream, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            info = tarfile.TarInfo("package/module.py")
            info.pax_headers = {"source": "/Users/fixture-person-invalid/private"}
            raw = b"value = 1\n"
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        with self.assertRaises(privacy.PrivacyError):
            privacy.scan_payload("packages/pax.tar", stream.getvalue())

        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            info = tarfile.TarInfo("package/module.py")
            raw = b"value = 1\n"
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        hidden = bytearray(stream.getvalue())
        hidden[-1] = 1
        with self.assertRaisesRegex(privacy.PrivacyError, "trailer"):
            privacy.scan_payload("packages/trailing.tar", bytes(hidden))

        compressor = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        bomb = compressor.compress(b"x" * 2048) + compressor.flush()
        original_bound = privacy.MAX_ARCHIVE_TOTAL_BYTES
        try:
            privacy.MAX_ARCHIVE_TOTAL_BYTES = 1024
            with self.assertRaisesRegex(privacy.PrivacyError, "oversized"):
                privacy.scan_payload("packages/bomb.tar.gz", bomb)
        finally:
            privacy.MAX_ARCHIVE_TOTAL_BYTES = original_bound


if __name__ == "__main__":
    unittest.main()
