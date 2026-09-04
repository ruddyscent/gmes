from __future__ import annotations

import copy
import hashlib
import io
import json
import stat
import statistics
import tempfile
import traceback
import unittest
import unicodedata
import zipfile
from pathlib import Path
from unittest import mock

from benchmarks import issue123_operations as operations
from benchmarks import issue123_privacy as privacy
from benchmarks import issue123_publication as publication
from tests import test_issue123_privacy as privacy_fixture


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _inventory_digest(value):
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


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


class Issue123PublicationTest(unittest.TestCase):
    def setUp(self):
        self.bindings = {
            "final_sha": "a" * 40,
            "manifest_sha256": "b" * 64,
            "jobs": [
                {
                    "name": "Python 3.14 / ubuntu-latest",
                    "run_id": 101,
                    "run_attempt": 2,
                    "job_id": 1001,
                },
                {
                    "name": "Python 3.14 / macos-latest",
                    "run_id": 101,
                    "run_attempt": 2,
                    "job_id": 1002,
                },
                {
                    "name": "CodeQL / python",
                    "run_id": 102,
                    "run_attempt": 3,
                    "job_id": 1003,
                },
                {
                    "name": "CodeQL / c-cpp",
                    "run_id": 102,
                    "run_attempt": 3,
                    "job_id": 1004,
                },
            ],
        }
        self.policy, self.projection = self._fixture()
        self.assets = publication.build_publication_assets(
            self.projection,
            expected_policy=self.policy,
            expected_bindings=self.bindings,
        )
        self.ledger = {
            role: {
                "name": name,
                "size_bytes": len(self.assets[name]),
                "sha256": hashlib.sha256(self.assets[name]).hexdigest(),
            }
            for role, name in publication.ASSET_ORDER
        }
        self.release_identity = self._release_identity(self._release_capture())

    def _release_capture(self, assets=None, ledger=None):
        assets = self.assets if assets is None else assets
        ledger = self.ledger if ledger is None else ledger
        final_sha = self.bindings["final_sha"]
        release_id = 7001
        tag = f"{publication.RELEASE_TAG_PREFIX}{final_sha}"
        api_root = f"https://api.github.com/repos/{publication.REPOSITORY}"
        web_root = f"https://github.com/{publication.REPOSITORY}"
        records = []
        for index, (role, name) in enumerate(publication.ASSET_ORDER, start=1):
            asset_id = 8000 + index
            records.append(
                {
                    "role": role,
                    "asset_id": asset_id,
                    "release_id": release_id,
                    "name": name,
                    "api_url": f"{api_root}/releases/assets/{asset_id}",
                    "browser_download_url": (
                        f"{web_root}/releases/download/{tag}/{name}"
                    ),
                    "state": "uploaded",
                    "size_bytes": len(assets[name]),
                    "sha256": ledger[role]["sha256"],
                }
            )
        return {
            "schema_version": publication.SCHEMA_VERSION,
            "kind": publication.RELEASE_CAPTURE_KIND,
            "repository": publication.REPOSITORY,
            "release_id": release_id,
            "tag_name": tag,
            "target_commitish": final_sha,
            "api_url": f"{api_root}/releases/{release_id}",
            "html_url": f"{web_root}/releases/tag/{tag}",
            "immutable": True,
            "draft": False,
            "prerelease": False,
            "tag_ref": {
                "ref": f"refs/tags/{tag}",
                "api_url": f"{api_root}/git/refs/tags/{tag}",
                "object_type": "commit",
                "object_sha": final_sha,
                "object_url": f"{api_root}/git/commits/{final_sha}",
            },
            "assets": records,
        }

    @staticmethod
    def _release_identity(release):
        release_fields = (
            "repository",
            "release_id",
            "tag_name",
            "target_commitish",
            "api_url",
            "html_url",
        )
        asset_fields = (
            "role",
            "asset_id",
            "release_id",
            "name",
            "api_url",
            "browser_download_url",
        )
        return {
            **{field: release[field] for field in release_fields},
            "tag_ref": copy.deepcopy(release["tag_ref"]),
            "assets": [
                {field: record[field] for field in asset_fields}
                for record in release["assets"]
            ],
        }

    @staticmethod
    def _event(
        ordinal,
        kind,
        start,
        duration=0.0,
        *,
        phase="complete",
        stream=None,
        correlation=None,
        allocation=None,
        allocation_context=None,
        graph=None,
        amount=None,
        live=None,
    ):
        return {
            "ordinal": ordinal,
            "kind": kind,
            "semantic_token": kind,
            "phase": phase,
            "start_us": float(start),
            "duration_us": float(duration),
            "process_ordinal": 0,
            "thread_ordinal": 0,
            "stream_ordinal": stream,
            "correlation_ordinal": correlation,
            "allocation_ordinal": allocation,
            "allocation_context_ordinal": (
                0
                if kind == "allocation" and allocation_context is None
                else allocation_context
            ),
            "graph_ordinal": graph,
            "bytes": amount,
            "live_allocated_bytes": live,
        }

    def _trace(self, name, *, mode, nccl=False):
        events = [
            self._event(
                0,
                "allocation",
                0,
                phase="instant",
                allocation=0,
                amount=64,
                live=64,
            ),
            self._event(
                1,
                "allocation",
                1,
                phase="instant",
                allocation=0,
                amount=-64,
                live=0,
            ),
        ]
        if mode == "cpu":
            events.append(self._event(2, "cpu-operation", 2, 2))
        elif mode == "cuda-eager":
            events.extend(
                [
                    self._event(2, "cuda-runtime", 2, 1),
                    self._event(3, "kernel", 10, 10, stream=0, correlation=0),
                ]
            )
        else:
            events.extend(
                [
                    self._event(2, "cuda-graph", 2, 1, stream=0, graph=0),
                    self._event(3, "compiled-region", 3, 5),
                    self._event(4, "kernel", 10, 10, stream=0, correlation=0),
                ]
            )
        if mode == "cuda-graph" and nccl:
            events.append(
                self._event(
                    len(events),
                    "nccl-kernel",
                    12,
                    4,
                    stream=1,
                    correlation=1,
                )
            )
        summary = {
            "event_count": len(events),
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
            "device_copy_events": 0,
            "host_to_device_events": 0,
            "device_to_host_events": 0,
            "kernel_launches": (1 if mode != "cpu" else 0) + (1 if nccl else 0),
            "compiled_region_events": 1 if mode == "cuda-graph" else 0,
            "cuda_graph_launches": 1 if mode == "cuda-graph" else 0,
            "nccl_kernel_launches": 1 if nccl else 0,
            "nccl_device_us": 4.0 if nccl else 0.0,
            "compute_device_us": 10.0 if mode != "cpu" else 0.0,
            "nccl_compute_overlap_us": 4.0 if nccl else 0.0,
            "nccl_exposed_us": 0.0,
            "overlap_fraction": 1.0 if nccl else 0.0,
        }
        return {
            "name": name,
            "clock": publication.LOCAL_CLOCK,
            "events": events,
            "summary": summary,
        }

    @staticmethod
    def _timing(name):
        samples = [0.009, 0.01, 0.011]
        middle = statistics.median(samples)
        mad = statistics.median(abs(sample - middle) for sample in samples)
        return {
            "name": name,
            "unit": "seconds",
            "samples": samples,
            "sample_count": len(samples),
            "samples_sha256": _inventory_digest(samples),
            "median_seconds": middle,
            "mad_seconds": mad,
            "relative_mad": mad / middle,
        }

    def _fixture(self):
        policy_scopes = []
        public_scopes = []
        cases = []
        inventory = []
        for index, scope in enumerate(publication.TECHNICAL_SCOPE_ORDER):
            timing_name = f"{scope}-timing".replace("_", "-")
            trace_name = f"{scope}-trace".replace("_", "-")
            case_name = f"{scope}-case".replace("_", "-")
            mode = {
                "cpu": "cpu",
                "single_gpu": "cuda-eager",
                "two_gpu": "cuda-graph",
            }.get(scope, "cuda-eager")
            nccl = scope == "two_gpu"
            trace = self._trace(trace_name, mode=mode, nccl=nccl)
            timing = self._timing(timing_name)
            expectation = {
                "name": trace_name,
                "event_count": trace["summary"]["event_count"],
                "semantic_signatures": [
                    [event["semantic_token"], event["phase"]]
                    for event in trace["events"]
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
                "device_copy_events": 0,
                "host_to_device_events": 0,
                "device_to_host_events": 0,
                "kernel_launches": trace["summary"]["kernel_launches"],
                "compiled_region_events": trace["summary"]["compiled_region_events"],
                "cuda_graph_launches": trace["summary"]["cuda_graph_launches"],
                "nccl_kernel_launches": 1 if nccl else 0,
                "require_nccl_overlap": nccl,
            }
            array_policy = {
                "name": "field",
                "dtype": "float64",
                "shape": [2],
                "comparison_contract": "elementwise",
                "rtol": 1e-6,
                "atol": 1e-8,
                "normalized_limit": None,
            }
            payload_policy = (
                [
                    {
                        "name": "gmes.whl",
                        "media_type": "application/vnd.python.wheel",
                        "size_bytes": 10,
                        "sha256": _digest("wheel"),
                    }
                ]
                if scope == "macos"
                else []
            )
            policy_scope = {
                "name": scope,
                "identities": ["host"],
                "timings": [
                    {
                        "name": timing_name,
                        "sample_count": timing["sample_count"],
                        "samples_sha256": _inventory_digest(timing["samples"]),
                    }
                ],
                "traces": [expectation],
                "payloads": payload_policy,
                "correctness": [
                    {
                        "name": case_name,
                        "captures": [{"capture": 0, "arrays": [array_policy]}],
                    }
                ],
            }
            policy_scopes.append(policy_scope)
            public_array = {
                "name": "field",
                "dtype": "float64",
                "shape": [2],
                "element_count": 2,
                "comparison": {
                    "contract": "elementwise",
                    "rtol": 1e-6,
                    "atol": 1e-8,
                    "normalized_limit": None,
                    "max_abs_error": 0.0,
                    "max_allowed_error": 1e-8 + 1e-6 * 1.0,
                    "max_tolerance_excess": -1e-8,
                    "reference_abs_max": 1.0,
                    "reference_l2": 1.0,
                    "error_l2": 0.0,
                    "normalized_linf": None,
                    "normalized_l2": None,
                    "reference_all_zero": False,
                    "zero_reference_exact": True,
                    "passed": True,
                },
                "commitments": {
                    "algorithm": publication.COMMITMENT_ALGORITHM,
                    "reference": _digest(f"reference-{scope}"),
                    "candidate": _digest(f"candidate-{scope}"),
                },
            }
            case = {
                "scope": scope,
                "name": case_name,
                "captures": [{"capture": 0, "arrays": [public_array]}],
            }
            cases.append(case)
            inventory.append([scope, case_name, "0", "field"])
            public_scopes.append(
                {
                    "schema_version": publication.SCHEMA_VERSION,
                    "kind": publication.SCOPE_KIND,
                    "scope": scope,
                    "bindings": copy.deepcopy(self.bindings),
                    "identity_commitments": [
                        {
                            "name": "host",
                            "algorithm": publication.COMMITMENT_ALGORITHM,
                            "commitment": _digest(f"host-{scope}-{index}"),
                        }
                    ],
                    "timings": [timing],
                    "traces": [trace],
                    "payloads": [
                        {**item, "scan_contract": publication.SCAN_CONTRACT}
                        for item in payload_policy
                    ],
                    "closure": {
                        "identity_names": ["host"],
                        "timing_names": [timing_name],
                        "trace_names": [trace_name],
                        "payload_names": [item["name"] for item in payload_policy],
                        "correctness_case_names": [case_name],
                        "correctness_capture_count": 1,
                        "correctness_array_count": 1,
                    },
                }
            )
        policy = {
            "schema_version": publication.SCHEMA_VERSION,
            "kind": publication.POLICY_KIND,
            "bindings": copy.deepcopy(self.bindings),
            "scopes": policy_scopes,
            "execution_witnesses": [
                {
                    "claim": "cpu-eager",
                    "scope": "cpu",
                    "trace_name": "cpu-trace",
                    "validation_workflow": "CI",
                    "validator_job_name": "Python 3.14 / ubuntu-latest",
                },
                {
                    "claim": "cuda-eager",
                    "scope": "single_gpu",
                    "trace_name": "single-gpu-trace",
                    "validation_workflow": "CI",
                    "validator_job_name": "Python 3.14 / ubuntu-latest",
                },
                {
                    "claim": "cuda-graph",
                    "scope": "two_gpu",
                    "trace_name": "two-gpu-trace",
                    "validation_workflow": "CI",
                    "validator_job_name": "Python 3.14 / ubuntu-latest",
                },
            ],
            "issue115": {
                "timings": [{"scope": "cpu", "name": "cpu-timing"}],
                "profilers": [{"scope": "two_gpu", "name": "two-gpu-trace"}],
            },
        }
        correctness = {
            "schema_version": publication.SCHEMA_VERSION,
            "kind": publication.COMMITMENTS_KIND,
            "bindings": copy.deepcopy(self.bindings),
            "algorithm": publication.COMMITMENT_ALGORITHM,
            "cases": cases,
            "closure": {
                "scope_order": list(publication.TECHNICAL_SCOPE_ORDER),
                "case_count": 5,
                "capture_count": 5,
                "array_count": 5,
                "inventory_sha256": _inventory_digest(inventory),
            },
        }
        timing = copy.deepcopy(public_scopes[0]["timings"][0])
        raw_timing = {
            "schema_version": publication.SCHEMA_VERSION,
            "kind": publication.RAW_TIMING_KIND,
            "contract_id": publication.RAW_TIMING_CONTRACT_ID,
            "bindings": copy.deepcopy(self.bindings),
            "records": [{"scope": "cpu", **timing}],
            "closure": {
                "record_count": 1,
                "inventory_sha256": _inventory_digest([["cpu", "cpu-timing"]]),
            },
        }
        two_gpu = public_scopes[3]
        profiler = copy.deepcopy(two_gpu["traces"][0])
        event_profiler = {
            "schema_version": publication.SCHEMA_VERSION,
            "kind": publication.EVENT_PROFILER_KIND,
            "contract_id": publication.EVENT_PROFILER_CONTRACT_ID,
            "bindings": copy.deepcopy(self.bindings),
            "records": [{"scope": "two_gpu", **profiler}],
            "closure": {
                "record_count": 1,
                "inventory_sha256": _inventory_digest([["two_gpu", "two-gpu-trace"]]),
            },
        }
        projection = {
            "schema_version": publication.SCHEMA_VERSION,
            "kind": publication.PROJECTION_KIND,
            "bindings": copy.deepcopy(self.bindings),
            "technical_scopes": public_scopes,
            "correctness_commitments": correctness,
            "execution_witness": publication._derive_execution_witness(
                policy["execution_witnesses"], public_scopes, self.bindings
            ),
            "raw_timing": raw_timing,
            "event_profiler": event_profiler,
        }
        return policy, projection

    @staticmethod
    def _entries(archive):
        with zipfile.ZipFile(io.BytesIO(archive)) as handle:
            return {name: handle.read(name) for name in handle.namelist()}

    def _rehash_archive(self, mutate):
        entries = self._entries(self.assets[publication.TECHNICAL_EVIDENCE_ASSET])
        mutate(entries)
        manifest = json.loads(entries[publication.MANIFEST_PATH])
        for descriptor in manifest["payloads"]:
            raw = entries[descriptor["path"]]
            descriptor["size_bytes"] = len(raw)
            descriptor["sha256"] = hashlib.sha256(raw).hexdigest()
        entries[publication.MANIFEST_PATH] = publication.canonical_json_bytes(manifest)
        return publication._encode_archive(entries)

    def test_assets_are_deterministic_reopenable_and_non_circular(self):
        self.assertEqual(
            publication.ASSET_ORDER,
            tuple(operations.TECHNICAL_RELEASE_ASSETS.items()),
        )
        self.assertEqual(
            publication.REQUIRED_JOB_NAMES,
            (*operations.REQUIRED_STATUS_CONTEXTS, *operations.REQUIRED_CODEQL_JOBS),
        )
        again = publication.build_publication_assets(
            self.projection,
            expected_policy=self.policy,
            expected_bindings=self.bindings,
        )
        self.assertEqual(self.assets, again)
        self.assertEqual(
            list(self.assets),
            [name for _role, name in publication.ASSET_ORDER],
        )
        archive = self.assets[publication.TECHNICAL_EVIDENCE_ASSET]
        with zipfile.ZipFile(io.BytesIO(archive)) as handle:
            self.assertEqual(tuple(handle.namelist()), publication.ARCHIVE_ENTRY_ORDER)
            self.assertEqual(handle.comment, b"")
            for info in handle.infolist():
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
                self.assertEqual(info.date_time, publication.FIXED_ZIP_TIMESTAMP)
                self.assertEqual(info.extra, b"")
                self.assertEqual(info.comment, b"")
                self.assertEqual(info.external_attr >> 16, stat.S_IFREG | 0o644)
            manifest = json.loads(handle.read(publication.MANIFEST_PATH))
        self.assertNotIn(
            publication.MANIFEST_PATH,
            [item["path"] for item in manifest["payloads"]],
        )
        self.assertIn(
            publication.EXECUTION_WITNESS_PATH,
            [item["path"] for item in manifest["payloads"]],
        )
        summary = json.loads(self.assets[publication.TECHNICAL_SUMMARY_ASSET])
        self.assertEqual(
            [item["role"] for item in summary["assets"]],
            ["technical_evidence", "raw_timing", "event_profiler"],
        )
        downloaded = {name: bytes(bytearray(raw)) for name, raw in self.assets.items()}
        ledger = {
            role: {
                "name": name,
                "size_bytes": len(downloaded[name]),
                "sha256": hashlib.sha256(downloaded[name]).hexdigest(),
            }
            for role, name in publication.ASSET_ORDER
        }
        result = publication.validate_publication_assets(
            downloaded,
            expected_policy=self.policy,
            expected_bindings=self.bindings,
            expected_assets=ledger,
        )
        self.assertEqual(
            result["asset_order"], [role for role, _ in publication.ASSET_ORDER]
        )
        self.assertEqual(result["kind"], publication.VALIDATION_KIND)
        bad_ledger = copy.deepcopy(ledger)
        bad_ledger["technical_evidence"]["sha256"] = "0" * 64
        with self.assertRaises(publication.PublicationError):
            publication.validate_publication_assets(
                downloaded,
                expected_policy=self.policy,
                expected_bindings=self.bindings,
                expected_assets=bad_ledger,
            )

    def test_offline_release_capture_and_receipt_are_independently_reopenable(self):
        release = self._release_capture()
        receipt = publication.finalize_publication(
            self.assets,
            release,
            expected_policy=self.policy,
            expected_release_identity=self.release_identity,
            expected_bindings=self.bindings,
            expected_assets=self.ledger,
        )
        self.assertEqual(
            receipt,
            publication.finalize_publication(
                self.assets,
                publication.canonical_json_bytes(release),
                expected_policy=self.policy,
                expected_release_identity=self.release_identity,
                expected_bindings=self.bindings,
                expected_assets=self.ledger,
            ),
        )
        downloaded = {name: bytes(bytearray(raw)) for name, raw in self.assets.items()}
        reopened = publication.validate_publication_receipt(
            bytes(bytearray(receipt)),
            downloaded,
            expected_policy=copy.deepcopy(self.policy),
            expected_release_identity=copy.deepcopy(self.release_identity),
            expected_bindings=copy.deepcopy(self.bindings),
            expected_assets=copy.deepcopy(self.ledger),
        )
        self.assertEqual(reopened["kind"], publication.PUBLICATION_RECEIPT_KIND)
        self.assertEqual(
            reopened["release_capture"]["tag_ref"]["object_sha"],
            self.bindings["final_sha"],
        )
        self.assertEqual(
            reopened["execution_witness"]["claims"][0]["validation_workflow"],
            "CI",
        )
        self.assertNotIn("receipt_sha256", reopened["hashes"])
        self.assertEqual(
            reopened["hashes"]["release_capture_sha256"],
            hashlib.sha256(publication.canonical_json_bytes(release)).hexdigest(),
        )
        witness_member = reopened["execution_witness_member"]
        self.assertEqual(witness_member["path"], publication.EXECUTION_WITNESS_PATH)
        self.assertEqual(
            witness_member["sha256"],
            hashlib.sha256(
                publication.canonical_json_bytes(reopened["execution_witness"])
            ).hexdigest(),
        )

        for label, mutate in (
            (
                "release-api-query",
                lambda value: value.__setitem__("api_url", value["api_url"] + "?x=1"),
            ),
            (
                "tag-ref-alias",
                lambda value: value["tag_ref"].__setitem__(
                    "api_url", value["tag_ref"]["api_url"].replace("/refs/", "/ref/")
                ),
            ),
            ("mutable", lambda value: value.__setitem__("immutable", False)),
            (
                "asset-id-alias",
                lambda value: value["assets"][1].__setitem__(
                    "asset_id", value["assets"][0]["asset_id"]
                ),
            ),
            (
                "download-query",
                lambda value: value["assets"][0].__setitem__(
                    "browser_download_url",
                    value["assets"][0]["browser_download_url"] + "?download=1",
                ),
            ),
        ):
            forged = copy.deepcopy(release)
            mutate(forged)
            with (
                self.subTest(label=label),
                self.assertRaises(publication.PublicationError),
            ):
                publication.finalize_publication(
                    self.assets,
                    forged,
                    expected_policy=self.policy,
                    expected_release_identity=self.release_identity,
                    expected_bindings=self.bindings,
                    expected_assets=self.ledger,
                )

        bool_release_ids = copy.deepcopy(release)
        bool_release_ids["release_id"] = 1
        bool_release_ids["api_url"] = (
            f"https://api.github.com/repos/{publication.REPOSITORY}/releases/1"
        )
        for asset in bool_release_ids["assets"]:
            asset["release_id"] = True
        with self.assertRaises(publication.PublicationError):
            publication.finalize_publication(
                self.assets,
                bool_release_ids,
                expected_policy=self.policy,
                expected_release_identity=self.release_identity,
                expected_bindings=self.bindings,
                expected_assets=self.ledger,
            )

        forged_receipt = json.loads(receipt)
        forged_receipt["execution_witness"]["claims"][0]["normalized_trace_sha256"] = (
            "0" * 64
        )
        forged_witness_raw = publication.canonical_json_bytes(
            forged_receipt["execution_witness"]
        )
        forged_receipt["execution_witness_member"]["size_bytes"] = len(
            forged_witness_raw
        )
        forged_receipt["execution_witness_member"]["sha256"] = hashlib.sha256(
            forged_witness_raw
        ).hexdigest()
        forged_receipt["hashes"]["execution_witness_member_sha256"] = forged_receipt[
            "execution_witness_member"
        ]["sha256"]
        with self.assertRaises(publication.PublicationError):
            publication.validate_publication_receipt(
                publication.canonical_json_bytes(forged_receipt),
                downloaded,
                expected_policy=self.policy,
                expected_release_identity=self.release_identity,
                expected_bindings=self.bindings,
                expected_assets=self.ledger,
            )

        bool_receipt = json.loads(receipt)
        bool_receipt["execution_witness"]["claims"][0]["validator_job"][
            "run_attempt"
        ] = True
        bool_witness_raw = publication.canonical_json_bytes(
            bool_receipt["execution_witness"]
        )
        bool_receipt["execution_witness_member"]["size_bytes"] = len(bool_witness_raw)
        bool_receipt["execution_witness_member"]["sha256"] = hashlib.sha256(
            bool_witness_raw
        ).hexdigest()
        bool_receipt["hashes"]["execution_witness_member_sha256"] = bool_receipt[
            "execution_witness_member"
        ]["sha256"]
        with self.assertRaises(publication.PublicationError):
            publication.validate_publication_receipt(
                publication.canonical_json_bytes(bool_receipt),
                downloaded,
                expected_policy=self.policy,
                expected_release_identity=self.release_identity,
                expected_bindings=self.bindings,
                expected_assets=self.ledger,
            )

    def test_refreshed_release_descriptors_do_not_replace_external_ledger(self):
        def relabel_witness(entries):
            witness = json.loads(entries[publication.EXECUTION_WITNESS_PATH])
            witness["claims"][0]["trace_name"] = "forged-trace"
            entries[publication.EXECUTION_WITNESS_PATH] = (
                publication.canonical_json_bytes(witness)
            )

        forged_archive = self._rehash_archive(relabel_witness)
        forged_assets = dict(self.assets)
        forged_assets[publication.TECHNICAL_EVIDENCE_ASSET] = forged_archive
        summary = json.loads(forged_assets[publication.TECHNICAL_SUMMARY_ASSET])
        archive_descriptor = summary["assets"][0]
        archive_descriptor["size_bytes"] = len(forged_archive)
        archive_descriptor["sha256"] = hashlib.sha256(forged_archive).hexdigest()
        forged_assets[publication.TECHNICAL_SUMMARY_ASSET] = (
            publication.canonical_json_bytes(summary)
        )
        refreshed_ledger = {
            role: {
                "name": asset_name,
                "size_bytes": len(forged_assets[asset_name]),
                "sha256": hashlib.sha256(forged_assets[asset_name]).hexdigest(),
            }
            for role, asset_name in publication.ASSET_ORDER
        }
        refreshed_capture = self._release_capture(forged_assets, refreshed_ledger)
        with self.assertRaises(publication.PublicationError):
            publication.finalize_publication(
                forged_assets,
                refreshed_capture,
                expected_policy=self.policy,
                expected_release_identity=self.release_identity,
                expected_bindings=self.bindings,
                expected_assets=self.ledger,
            )
        with self.assertRaises(publication.PublicationError):
            publication.finalize_publication(
                forged_assets,
                refreshed_capture,
                expected_policy=self.policy,
                expected_release_identity=self.release_identity,
                expected_bindings=self.bindings,
                expected_assets=refreshed_ledger,
            )

    def test_release_and_asset_ids_require_an_independent_identity_anchor(self):
        substituted = copy.deepcopy(self._release_capture())
        substituted["release_id"] = 9001
        substituted["api_url"] = (
            f"https://api.github.com/repos/{publication.REPOSITORY}/releases/9001"
        )
        for record in substituted["assets"]:
            record["asset_id"] += 10_000
            record["release_id"] = 9001
            record["api_url"] = (
                f"https://api.github.com/repos/{publication.REPOSITORY}/"
                f"releases/assets/{record['asset_id']}"
            )

        with self.assertRaises(publication.PublicationError):
            publication.finalize_publication(
                self.assets,
                substituted,
                expected_policy=self.policy,
                expected_release_identity=self.release_identity,
                expected_bindings=self.bindings,
                expected_assets=self.ledger,
            )

        substituted_identity = self._release_identity(substituted)
        receipt = publication.finalize_publication(
            self.assets,
            substituted,
            expected_policy=self.policy,
            expected_release_identity=substituted_identity,
            expected_bindings=self.bindings,
            expected_assets=self.ledger,
        )
        with self.assertRaises(publication.PublicationError):
            publication.validate_publication_receipt(
                receipt,
                self.assets,
                expected_policy=self.policy,
                expected_release_identity=self.release_identity,
                expected_bindings=self.bindings,
                expected_assets=self.ledger,
            )

        extra_field = copy.deepcopy(self.release_identity)
        extra_field["unexpected"] = 1
        with self.assertRaises(publication.PublicationError):
            publication.finalize_publication(
                self.assets,
                self._release_capture(),
                expected_policy=self.policy,
                expected_release_identity=extra_field,
                expected_bindings=self.bindings,
                expected_assets=self.ledger,
            )

    def test_execution_witness_is_derived_from_event_complete_traces(self):
        policy = copy.deepcopy(self.policy)
        scopes = copy.deepcopy(self.projection["technical_scopes"])
        cpu_trace = scopes[0]["traces"][0]
        cpu_trace["events"].append(
            self._event(len(cpu_trace["events"]), "memset", 4, 1)
        )
        cpu_trace["summary"]["event_count"] += 1
        cpu_expectation = policy["scopes"][0]["traces"][0]
        cpu_expectation["semantic_signatures"].append(["memset", "complete"])
        cpu_expectation["event_count"] += 1
        with self.assertRaisesRegex(publication.PublicationError, "cpu-eager"):
            publication._derive_execution_witness(
                policy["execution_witnesses"], scopes, self.bindings
            )

    def test_compiled_regions_cannot_be_relabelled_as_eager_downloads(self):
        for scope_index, claim_index, error in (
            (0, 0, "cpu-eager"),
            (2, 1, "cuda-eager"),
        ):
            with self.subTest(claim=error):
                policy = copy.deepcopy(self.policy)
                scope_name = publication.TECHNICAL_SCOPE_ORDER[scope_index]

                def add_compiled_region(entries):
                    path = publication.SCOPE_PATHS[scope_name]
                    scope = json.loads(entries[path])
                    trace = scope["traces"][0]
                    trace["events"].append(
                        self._event(len(trace["events"]), "compiled-region", 21, 1)
                    )
                    trace["summary"]["event_count"] += 1
                    trace["summary"]["compiled_region_events"] += 1
                    entries[path] = publication.canonical_json_bytes(scope)

                    expectation = policy["scopes"][scope_index]["traces"][0]
                    expectation["semantic_signatures"].append(
                        ["compiled-region", "complete"]
                    )
                    expectation["event_count"] += 1
                    expectation["compiled_region_events"] += 1

                    witness = json.loads(entries[publication.EXECUTION_WITNESS_PATH])
                    claim = witness["claims"][claim_index]
                    signatures = [
                        [event["semantic_token"], event["phase"]]
                        for event in trace["events"]
                    ]
                    normalized_trace = {
                        "clock": trace["clock"],
                        "events": trace["events"],
                        "summary": trace["summary"],
                    }
                    claim["event_count"] = trace["summary"]["event_count"]
                    claim["semantic_inventory_sha256"] = _inventory_digest(signatures)
                    claim["normalized_trace_sha256"] = _inventory_digest(
                        normalized_trace
                    )
                    entries[publication.EXECUTION_WITNESS_PATH] = (
                        publication.canonical_json_bytes(witness)
                    )

                archive = self._rehash_archive(add_compiled_region)
                downloaded = dict(self.assets)
                downloaded[publication.TECHNICAL_EVIDENCE_ASSET] = archive
                summary = json.loads(downloaded[publication.TECHNICAL_SUMMARY_ASSET])
                summary["assets"][0]["size_bytes"] = len(archive)
                summary["assets"][0]["sha256"] = hashlib.sha256(archive).hexdigest()
                downloaded[publication.TECHNICAL_SUMMARY_ASSET] = (
                    publication.canonical_json_bytes(summary)
                )
                downloaded = {
                    name: bytes(bytearray(raw)) for name, raw in downloaded.items()
                }
                refreshed_ledger = {
                    role: {
                        "name": name,
                        "size_bytes": len(downloaded[name]),
                        "sha256": hashlib.sha256(downloaded[name]).hexdigest(),
                    }
                    for role, name in publication.ASSET_ORDER
                }
                with self.assertRaisesRegex(publication.PublicationError, error):
                    publication.validate_publication_assets(
                        downloaded,
                        expected_policy=policy,
                        expected_bindings=self.bindings,
                        expected_assets=refreshed_ledger,
                    )

    def test_boolean_integer_aliases_fail_closed(self):
        policy = copy.deepcopy(self.policy)
        projection = copy.deepcopy(self.projection)
        policy["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]["shape"] = [1]
        public_array = projection["correctness_commitments"]["cases"][0]["captures"][0][
            "arrays"
        ][0]
        public_array["shape"] = [True]
        public_array["element_count"] = 1
        with self.assertRaises(publication.PublicationError):
            publication.build_publication_assets(
                projection,
                expected_policy=policy,
                expected_bindings=self.bindings,
            )

        policy = copy.deepcopy(self.policy)
        projection = copy.deepcopy(self.projection)
        policy["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]["rtol"] = 0
        comparison = projection["correctness_commitments"]["cases"][0]["captures"][0][
            "arrays"
        ][0]["comparison"]
        comparison["rtol"] = False
        comparison["max_allowed_error"] = comparison["atol"]
        with self.assertRaises(publication.PublicationError):
            publication.build_publication_assets(
                projection,
                expected_policy=policy,
                expected_bindings=self.bindings,
            )

        def bool_capture(entries):
            correctness = json.loads(entries[publication.COMMITMENTS_PATH])
            correctness["cases"][0]["captures"][0]["capture"] = False
            entries[publication.COMMITMENTS_PATH] = publication.canonical_json_bytes(
                correctness
            )

        with self.assertRaises(publication.PublicationError):
            publication.validate_public_archive(
                self._rehash_archive(bool_capture),
                expected_policy=self.policy,
                expected_bindings=self.bindings,
            )

        def bool_witness_attempt(entries):
            witness = json.loads(entries[publication.EXECUTION_WITNESS_PATH])
            witness["claims"][0]["validator_job"]["run_attempt"] = True
            entries[publication.EXECUTION_WITNESS_PATH] = (
                publication.canonical_json_bytes(witness)
            )

        with self.assertRaises(publication.PublicationError):
            publication.validate_public_archive(
                self._rehash_archive(bool_witness_attempt),
                expected_policy=self.policy,
                expected_bindings=self.bindings,
            )

        policy = copy.deepcopy(self.policy)
        scopes = copy.deepcopy(self.projection["technical_scopes"])
        eager_trace = scopes[2]["traces"][0]
        del eager_trace["events"][2]
        eager_trace["events"][2]["ordinal"] = 2
        eager_trace["summary"]["event_count"] -= 1
        eager_expectation = policy["scopes"][2]["traces"][0]
        del eager_expectation["semantic_signatures"][2]
        eager_expectation["event_count"] -= 1
        with self.assertRaisesRegex(publication.PublicationError, "cuda-eager"):
            publication._derive_execution_witness(
                policy["execution_witnesses"], scopes, self.bindings
            )

    def test_real_projector_to_downloaded_byte_validator(self):
        policy, private = privacy_fixture._fixture()
        salt = bytes(range(32))
        with mock.patch.object(privacy.secrets, "token_bytes", return_value=salt):
            projection = privacy.project_publication(private, policy)
        assets = publication.build_publication_assets(
            projection,
            expected_policy=policy,
            expected_bindings=policy["bindings"],
        )
        downloaded = {name: bytes(bytearray(raw)) for name, raw in assets.items()}
        ledger = {
            role: {
                "name": name,
                "size_bytes": len(downloaded[name]),
                "sha256": hashlib.sha256(downloaded[name]).hexdigest(),
            }
            for role, name in publication.ASSET_ORDER
        }
        validated = publication.validate_publication_assets(
            downloaded,
            expected_policy=policy,
            expected_bindings=policy["bindings"],
            expected_assets=ledger,
        )
        self.assertEqual(validated["kind"], publication.VALIDATION_KIND)
        self.assertEqual(
            [scope["scope"] for scope in validated["technical_scopes"]],
            list(publication.TECHNICAL_SCOPE_ORDER),
        )
        for raw in downloaded.values():
            self.assertNotIn(salt.hex().encode(), raw)

    def test_synthetic_literal_profile_cli_is_deterministic_and_path_free(self):
        policy, private = privacy_fixture._fixture()
        for policy_scope, private_scope in zip(
            policy["scopes"], private["scopes"], strict=True
        ):
            policy_scope["correctness"] = []
            private_scope["correctness"] = []
        salt = bytes(range(32))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            specification, runtime_paths, document = (
                privacy_fixture._production_source_fixture(root, policy, private)
            )
            completion_index = privacy_fixture._completion_bundle_fixture(
                root / "completion-bundle",
                root,
                document,
            )
            literal_bindings = {}
            for target, record in zip(
                privacy._expected_source_targets(policy),
                document["sources"],
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
            literal_patcher = mock.patch.object(
                privacy,
                "CODE_OWNED_LITERAL_TARGET_BINDINGS",
                literal_bindings,
            )
            literal_patcher.start()
            self.addCleanup(literal_patcher.stop)
            policy_path = root / "policy.json"
            policy_raw = publication.canonical_json_bytes(policy)
            policy_path.write_bytes(policy_raw)
            policy_sha256 = hashlib.sha256(policy_raw).hexdigest()

            results = []
            for ordinal in range(2):
                private_directory = root / f"authority-{ordinal}"
                private_directory.mkdir(mode=0o700)
                result = publication.prepare_publication(
                    source_specification=specification,
                    completion_index=completion_index,
                    policy_path=policy_path,
                    policy_sha256=policy_sha256,
                    runtime_receipt_paths=runtime_paths,
                    asset_output_directory=root / f"assets-{ordinal}",
                    private_openings_output=private_directory / "openings.json",
                    salt=salt,
                )
                results.append(result)
                self.assertEqual(
                    list(result["asset_paths"]),
                    [role for role, _asset in publication.ASSET_ORDER],
                )
                self.assertEqual(
                    stat.S_IMODE(result["private_openings"].stat().st_mode),
                    0o600,
                )
                self.assertEqual(stat.S_IMODE(private_directory.stat().st_mode), 0o700)
            self.assertEqual(results[0]["asset_ledger"], results[1]["asset_ledger"])
            for role, _asset in publication.ASSET_ORDER:
                self.assertEqual(
                    results[0]["asset_paths"][role].read_bytes(),
                    results[1]["asset_paths"][role].read_bytes(),
                )
                public_raw = results[0]["asset_paths"][role].read_bytes()
                self.assertNotIn(salt.hex().encode(), public_raw)
                self.assertNotIn(str(root).encode(), public_raw)

            cli_private_directory = root / "cli-authority"
            cli_private_directory.mkdir(mode=0o700)
            cli_assets = root / "cli-assets"
            cli_openings = cli_private_directory / "openings.json"
            prepare_arguments = [
                "prepare",
                "--source-spec",
                str(specification),
                "--completion-index",
                str(completion_index),
                "--policy",
                str(policy_path),
                "--policy-sha256",
                policy_sha256,
            ]
            for role in privacy.RUNTIME_RECEIPT_ORDER:
                prepare_arguments.extend(
                    ["--runtime-receipt", f"{role}={runtime_paths[role]}"]
                )
            prepare_arguments.extend(
                [
                    "--asset-output-directory",
                    str(cli_assets),
                    "--private-openings-output",
                    str(cli_openings),
                ]
            )
            with mock.patch("builtins.print") as printed:
                self.assertEqual(publication.main(prepare_arguments), 0)
            printed.assert_called_once_with("issue123-publication-prepare-ok")
            self.assertNotIn(str(root), printed.call_args.args[0])

            cli_asset_bytes = {
                asset: (cli_assets / asset).read_bytes()
                for _role, asset in publication.ASSET_ORDER
            }
            cli_ledger = {
                role: {
                    "name": asset,
                    "size_bytes": len(cli_asset_bytes[asset]),
                    "sha256": hashlib.sha256(cli_asset_bytes[asset]).hexdigest(),
                }
                for role, asset in publication.ASSET_ORDER
            }
            release_capture = self._release_capture(cli_asset_bytes, cli_ledger)
            release_identity = self._release_identity(release_capture)
            release_path = root / "release.json"
            identity_path = root / "release-identity.json"
            release_path.write_bytes(publication.canonical_json_bytes(release_capture))
            identity_path.write_bytes(
                publication.canonical_json_bytes(release_identity)
            )
            receipt_path = root / "publication-receipt.json"
            with mock.patch("builtins.print") as printed:
                self.assertEqual(
                    publication.main(
                        [
                            "finalize",
                            "--asset-directory",
                            str(cli_assets),
                            "--release-capture",
                            str(release_path),
                            "--release-identity",
                            str(identity_path),
                            "--policy",
                            str(policy_path),
                            "--policy-sha256",
                            policy_sha256,
                            "--receipt-output",
                            str(receipt_path),
                        ]
                    ),
                    0,
                )
            printed.assert_called_once_with("issue123-publication-finalize-ok")
            self.assertNotIn(str(root), printed.call_args.args[0])
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o600)

    def test_publication_outputs_reject_bundle_public_and_alias_overlap(self):
        policy, private = privacy_fixture._fixture()
        for policy_scope, private_scope in zip(
            policy["scopes"], private["scopes"], strict=True
        ):
            policy_scope["correctness"] = []
            private_scope["correctness"] = []
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            specification, runtime_paths, document = (
                privacy_fixture._production_source_fixture(root, policy, private)
            )
            completion_index = privacy_fixture._completion_bundle_fixture(
                root / "completion-bundle",
                root,
                document,
            )
            literal_bindings = {}
            for target, record in zip(
                privacy._expected_source_targets(policy),
                document["sources"],
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
            policy_path = root / "policy.json"
            policy_raw = publication.canonical_json_bytes(policy)
            policy_path.write_bytes(policy_raw)
            policy_sha256 = hashlib.sha256(policy_raw).hexdigest()
            private_root = root / "private"
            private_root.mkdir(mode=0o700)
            alias = root / "completion-alias"
            alias.symlink_to(completion_index.parent, target_is_directory=True)
            bundle_before = {
                path.relative_to(completion_index.parent).as_posix(): (
                    path.stat().st_dev,
                    path.stat().st_ino,
                    path.read_bytes(),
                )
                for path in completion_index.parent.rglob("*")
                if path.is_file()
            }
            cases = (
                (
                    "asset-under-bundle",
                    completion_index.parent / "nested-assets",
                    private_root / "asset-under-bundle.json",
                ),
                (
                    "sidecar-under-bundle",
                    root / "assets-sidecar-bundle",
                    completion_index.parent / "sidecar.json",
                ),
                (
                    "sidecar-equals-public-root",
                    root / "assets-equal-sidecar",
                    root / "assets-equal-sidecar",
                ),
                (
                    "sidecar-nested-under-public-root",
                    root / "assets-nested-sidecar",
                    root / "assets-nested-sidecar" / "sidecar.json",
                ),
                (
                    "asset-through-bundle-alias",
                    alias / "nested-assets",
                    private_root / "asset-through-alias.json",
                ),
            )
            with mock.patch.object(
                privacy,
                "CODE_OWNED_LITERAL_TARGET_BINDINGS",
                literal_bindings,
            ):
                for label, asset_output, sidecar_output in cases:
                    with (
                        self.subTest(label=label),
                        self.assertRaises(publication.PublicationError),
                    ):
                        publication.prepare_publication(
                            source_specification=specification,
                            completion_index=completion_index,
                            policy_path=policy_path,
                            policy_sha256=policy_sha256,
                            runtime_receipt_paths=runtime_paths,
                            asset_output_directory=asset_output,
                            private_openings_output=sidecar_output,
                            salt=bytes(range(32)),
                        )
                    self.assertFalse(asset_output.exists())
                    self.assertFalse(sidecar_output.exists())
            bundle_after = {
                path.relative_to(completion_index.parent).as_posix(): (
                    path.stat().st_dev,
                    path.stat().st_ino,
                    path.read_bytes(),
                )
                for path in completion_index.parent.rglob("*")
                if path.is_file()
            }
            self.assertEqual(bundle_after, bundle_before)

            asset_directory = root / "downloaded-assets"
            asset_directory.mkdir()
            for _role, asset_name in publication.ASSET_ORDER:
                (asset_directory / asset_name).write_bytes(self.assets[asset_name])
            release_capture = self._release_capture()
            release_identity = self._release_identity(release_capture)
            release_path = root / "release.json"
            identity_path = root / "release-identity.json"
            release_path.write_bytes(publication.canonical_json_bytes(release_capture))
            identity_path.write_bytes(
                publication.canonical_json_bytes(release_identity)
            )
            final_before = {
                path.name: path.read_bytes() for path in asset_directory.iterdir()
            }
            asset_alias = root / "downloaded-assets-alias"
            asset_alias.symlink_to(asset_directory, target_is_directory=True)
            for receipt_output in (
                asset_directory / "receipt.json",
                asset_alias / "receipt.json",
            ):
                with self.assertRaises(publication.PublicationError):
                    publication.finalize_publication_files(
                        asset_directory=asset_directory,
                        release_capture_path=release_path,
                        release_identity_path=identity_path,
                        policy_path=policy_path,
                        policy_sha256=policy_sha256,
                        receipt_output=receipt_output,
                    )
                self.assertFalse(receipt_output.exists())
            self.assertEqual(
                {path.name: path.read_bytes() for path in asset_directory.iterdir()},
                final_before,
            )

    def test_publication_library_path_failures_are_typed_fixed_and_context_free(self):
        marker = "SYNTHETIC-PATH-CANARY-É"

        def assert_private(error, expected):
            self.assertIs(type(error), publication.PublicationError)
            self.assertEqual(error.args, (expected,))
            self.assertIsNone(error.__cause__)
            self.assertIsNone(error.__context__)
            rendered = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            rendered += " ".join(_exception_messages(error)) + repr(error)
            folded = unicodedata.normalize("NFKC", rendered).casefold()
            self.assertNotIn(unicodedata.normalize("NFKC", marker).casefold(), folded)

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            policy_path = root / "policy.json"
            policy_raw = publication.canonical_json_bytes(self.policy)
            policy_path.write_bytes(policy_raw)
            policy_sha256 = hashlib.sha256(policy_raw).hexdigest()
            private_root = root / "private"
            private_root.mkdir(mode=0o700)
            source_target = root / "source-target.json"
            source_target.write_bytes(b"{}\n")
            source_alias = root / f"{marker}-source-alias"
            source_alias.symlink_to(source_target)
            source_cases = (
                root / f"{marker}-missing-source",
                None,
                source_alias,
            )
            for source in source_cases:
                with self.assertRaises(publication.PublicationError) as caught:
                    publication.prepare_publication(
                        source_specification=source,
                        completion_index=root / "unused-index.json",
                        policy_path=policy_path,
                        policy_sha256=policy_sha256,
                        runtime_receipt_paths={},
                        asset_output_directory=root / "unused-assets",
                        private_openings_output=private_root / "unused-openings.json",
                    )
                assert_private(
                    caught.exception, "publication source preparation failed"
                )

            real_private_reader = privacy._private_file_bytes

            def deny_source(path, label, *, maximum=privacy.MAX_PAYLOAD_BYTES):
                if label == "publication source specification":
                    raise PermissionError(marker)
                return real_private_reader(path, label, maximum=maximum)

            with (
                mock.patch.object(
                    privacy,
                    "_private_file_bytes",
                    side_effect=deny_source,
                ),
                self.assertRaises(publication.PublicationError) as caught,
            ):
                publication.prepare_publication(
                    source_specification=source_target,
                    completion_index=root / "unused-index.json",
                    policy_path=policy_path,
                    policy_sha256=policy_sha256,
                    runtime_receipt_paths={},
                    asset_output_directory=root / "unused-assets",
                    private_openings_output=private_root / "unused-openings.json",
                )
            assert_private(caught.exception, "publication source preparation failed")

            asset_target = root / "asset-target"
            asset_target.mkdir()
            asset_alias = root / f"{marker}-asset-alias"
            asset_alias.symlink_to(asset_target, target_is_directory=True)
            for asset_directory in (
                root / f"{marker}-missing-assets",
                None,
                asset_alias,
            ):
                with self.assertRaises(publication.PublicationError) as caught:
                    publication.finalize_publication_files(
                        asset_directory=asset_directory,
                        release_capture_path=root / "unused-release.json",
                        release_identity_path=root / "unused-identity.json",
                        policy_path=policy_path,
                        policy_sha256=policy_sha256,
                        receipt_output=private_root / "unused-receipt.json",
                    )
                assert_private(
                    caught.exception, "publication asset directory is unavailable"
                )

            real_lexical_path = privacy._lexical_path_without_symlinks

            def deny_assets(path, label, *, require_leaf):
                if label == "publication asset directory":
                    raise PermissionError(marker)
                return real_lexical_path(path, label, require_leaf=require_leaf)

            with (
                mock.patch.object(
                    privacy,
                    "_lexical_path_without_symlinks",
                    side_effect=deny_assets,
                ),
                self.assertRaises(publication.PublicationError) as caught,
            ):
                publication.finalize_publication_files(
                    asset_directory=asset_target,
                    release_capture_path=root / "unused-release.json",
                    release_identity_path=root / "unused-identity.json",
                    policy_path=policy_path,
                    policy_sha256=policy_sha256,
                    receipt_output=private_root / "unused-receipt.json",
                )
            assert_private(
                caught.exception, "publication asset directory is unavailable"
            )

    def test_publication_cli_failure_tokens_never_render_private_text(self):
        marker = (
            "/tmp/synthetic-private.invalid/identity "
            + "salt="
            + "ab" * 32
            + " hmac="
            + "cd" * 32
            + " raw-body=fixture-private-value"
        )
        for command, token in (
            ("prepare", "issue123-publication-prepare-failed\n"),
            ("finalize", "issue123-publication-finalize-failed\n"),
        ):
            for boundary in (publication.main, publication._cli):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    self.subTest(command=command, boundary=boundary.__name__),
                    mock.patch.object(
                        publication,
                        "_main",
                        side_effect=publication.PublicationError(marker),
                    ),
                    mock.patch("sys.stdout", new=stdout),
                    mock.patch("sys.stderr", new=stderr),
                ):
                    status = boundary([command])
                self.assertEqual(status, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(stderr.getvalue(), token)
                rendered = stdout.getvalue() + stderr.getvalue()
                self.assertNotIn("Traceback", rendered)
                self.assertNotIn(marker, rendered)
        with (
            mock.patch.object(
                publication,
                "_main",
                side_effect=RuntimeError(marker),
            ),
            self.assertRaisesRegex(RuntimeError, "synthetic-private"),
        ):
            publication.main(["prepare"])

    def test_stale_binding_components_fail_independently(self):
        mutations = [
            ("final_sha", "c" * 40),
            ("manifest_sha256", "d" * 64),
        ]
        for field, replacement in mutations:
            expected = copy.deepcopy(self.bindings)
            expected[field] = replacement
            with (
                self.subTest(field=field),
                self.assertRaises(publication.PublicationError),
            ):
                publication.validate_publication_assets(
                    self.assets,
                    expected_policy=self.policy,
                    expected_bindings=expected,
                    expected_assets=self.ledger,
                )
        for field in ("run_id", "run_attempt", "job_id"):
            expected = copy.deepcopy(self.bindings)
            expected["jobs"][0][field] += 1
            with (
                self.subTest(field=field),
                self.assertRaises(publication.PublicationError),
            ):
                publication.validate_publication_assets(
                    self.assets,
                    expected_policy=self.policy,
                    expected_bindings=expected,
                    expected_assets=self.ledger,
                )

    def test_rehashed_event_and_array_removal_cannot_bypass_policy(self):
        def remove_event(entries):
            path = publication.SCOPE_PATHS["cpu"]
            scope = json.loads(entries[path])
            scope["traces"][0]["events"].pop(2)
            for ordinal, event in enumerate(scope["traces"][0]["events"]):
                event["ordinal"] = ordinal
            summary = scope["traces"][0]["summary"]
            summary["event_count"] -= 1
            summary["cuda_graph_launches"] -= 1
            entries[path] = publication.canonical_json_bytes(scope)

        archive = self._rehash_archive(remove_event)
        with self.assertRaises(publication.PublicationError):
            publication.validate_public_archive(
                archive,
                expected_policy=self.policy,
                expected_bindings=self.bindings,
            )

        def remove_case(entries):
            correctness = json.loads(entries[publication.COMMITMENTS_PATH])
            correctness["cases"].pop(0)
            correctness["closure"]["case_count"] -= 1
            correctness["closure"]["capture_count"] -= 1
            correctness["closure"]["array_count"] -= 1
            correctness["closure"]["inventory_sha256"] = _inventory_digest(
                [
                    [item["scope"], item["name"], "0", "field"]
                    for item in correctness["cases"]
                ]
            )
            entries[publication.COMMITMENTS_PATH] = publication.canonical_json_bytes(
                correctness
            )
            scope_path = publication.SCOPE_PATHS["cpu"]
            scope = json.loads(entries[scope_path])
            scope["closure"]["correctness_case_names"].clear()
            scope["closure"]["correctness_capture_count"] = 0
            scope["closure"]["correctness_array_count"] = 0
            entries[scope_path] = publication.canonical_json_bytes(scope)

        archive = self._rehash_archive(remove_case)
        with self.assertRaises(publication.PublicationError):
            publication.validate_public_archive(
                archive,
                expected_policy=self.policy,
                expected_bindings=self.bindings,
            )

        def remove_array(entries):
            correctness = json.loads(entries[publication.COMMITMENTS_PATH])
            correctness["cases"][0]["captures"][0]["arrays"].clear()
            correctness["closure"]["array_count"] -= 1
            inventory = [
                [scope, case, "0", "field"]
                for scope, case in (
                    (item["scope"], item["name"]) for item in correctness["cases"][1:]
                )
            ]
            correctness["closure"]["inventory_sha256"] = _inventory_digest(inventory)
            entries[publication.COMMITMENTS_PATH] = publication.canonical_json_bytes(
                correctness
            )
            scope_path = publication.SCOPE_PATHS["cpu"]
            scope = json.loads(entries[scope_path])
            scope["closure"]["correctness_array_count"] -= 1
            entries[scope_path] = publication.canonical_json_bytes(scope)

        archive = self._rehash_archive(remove_array)
        with self.assertRaises(publication.PublicationError):
            publication.validate_public_archive(
                archive,
                expected_policy=self.policy,
                expected_bindings=self.bindings,
            )

    def test_rehashed_privacy_unknown_and_withheld_array_leaks_fail(self):
        def leak(entries):
            path = publication.SCOPE_PATHS["cpu"]
            scope = json.loads(entries[path])
            scope["metadata"] = {
                "hostname": "fixture-host.invalid",
                "path": "/home/fixture-person-invalid/work/repository",
                "token": "github_pat_" + "syntheticinvalid" * 2,
                "array": "correctness.npz",
            }
            scope["traces"][0]["events"][0]["unknown_arg"] = "raw"
            entries[path] = publication.canonical_json_bytes(scope)

        archive = self._rehash_archive(leak)
        with self.assertRaises(publication.PublicationError):
            publication.validate_public_archive(
                archive,
                expected_policy=self.policy,
                expected_bindings=self.bindings,
            )

    def test_timing_digest_and_identifier_context_are_externally_anchored(self):
        def replace_samples(entries):
            path = publication.SCOPE_PATHS["cpu"]
            scope = json.loads(entries[path])
            timing = scope["timings"][0]
            timing["samples"] = [0.018, 0.02, 0.022]
            timing["samples_sha256"] = _inventory_digest(timing["samples"])
            timing["median_seconds"] = 0.02
            timing["mad_seconds"] = 0.002
            timing["relative_mad"] = 0.1
            entries[path] = publication.canonical_json_bytes(scope)

        archive = self._rehash_archive(replace_samples)
        with self.assertRaises(publication.PublicationError):
            publication.validate_public_archive(
                archive,
                expected_policy=self.policy,
                expected_bindings=self.bindings,
            )

    def test_public_trace_validator_reconstructs_complete_flow_topology(self):
        valid_events = [
            self._event(0, "correlation-flow", 0, phase="flow-start", correlation=0),
            self._event(1, "correlation-flow", 1, phase="flow-start", correlation=1),
            self._event(2, "correlation-flow", 2, phase="flow-step", correlation=0),
            self._event(3, "correlation-flow", 3, phase="flow-end", correlation=1),
            self._event(4, "correlation-flow", 4, phase="flow-end", correlation=0),
        ]
        raw_flow_events = [
            {
                "name": "cpu-to-gpu",
                "cat": "ac2g",
                "ph": phase,
                "ts": index,
                "pid": 10,
                "tid": 7,
                "args": {},
                "id": correlation,
            }
            for index, (phase, correlation) in enumerate(
                (("s", 0), ("s", 1), ("t", 0), ("f", 1), ("f", 0))
            )
        ]
        summary = privacy.normalize_trace(
            privacy_fixture._trace_bytes(raw_flow_events)
        )["summary"]
        signatures = [
            [event["semantic_token"], event["phase"]] for event in valid_events
        ]
        self.assertEqual(
            publication._recompute_trace_summary(
                valid_events, summary, "valid flow trace", signatures
            ),
            summary,
        )

        equal_time = [
            self._event(0, "correlation-flow", 0, phase="flow-start", correlation=0),
            self._event(1, "correlation-flow", 0, phase="flow-end", correlation=0),
        ]
        equal_summary = copy.deepcopy(summary)
        equal_summary["event_count"] = len(equal_time)
        equal_signatures = [
            [event["semantic_token"], event["phase"]] for event in equal_time
        ]
        self.assertEqual(
            publication._recompute_trace_summary(
                equal_time, equal_summary, "equal-time flow", equal_signatures
            ),
            equal_summary,
        )

        reverse_time = copy.deepcopy(equal_time)
        reverse_time[0]["start_us"] = 1.0
        with self.assertRaisesRegex(
            publication.PublicationError, "timestamps decrease"
        ):
            publication._recompute_trace_summary(
                reverse_time,
                equal_summary,
                "reverse-time flow",
                equal_signatures,
            )

        missing_ordinal = [
            self._event(0, "correlation-flow", 0, phase="flow-start"),
            self._event(1, "correlation-flow", 1, phase="flow-end", correlation=0),
        ]
        missing_summary = copy.deepcopy(summary)
        missing_summary["event_count"] = len(missing_ordinal)
        with self.assertRaisesRegex(
            publication.PublicationError, "flow has no ordinal"
        ):
            publication._recompute_trace_summary(
                missing_ordinal,
                missing_summary,
                "missing flow id",
                [
                    [event["semantic_token"], event["phase"]]
                    for event in missing_ordinal
                ],
            )

        invalid_topologies = (
            ("end before start", ("flow-end", "flow-start")),
            ("step before start", ("flow-step", "flow-start", "flow-end")),
            ("duplicate start", ("flow-start", "flow-start", "flow-end")),
            ("duplicate end", ("flow-start", "flow-end", "flow-end")),
            ("unclosed", ("flow-start", "flow-step")),
        )
        for case, phases in invalid_topologies:
            events = [
                self._event(
                    index,
                    "correlation-flow",
                    index,
                    phase=phase,
                    correlation=0,
                )
                for index, phase in enumerate(phases)
            ]
            observed_summary = copy.deepcopy(summary)
            observed_summary["event_count"] = len(events)
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(
                    publication.PublicationError, "topology is incomplete"
                ),
            ):
                publication._recompute_trace_summary(
                    events,
                    observed_summary,
                    f"{case} flow",
                    [[event["semantic_token"], event["phase"]] for event in events],
                )

    def test_public_trace_origin_uses_only_clocked_events(self):
        normalized = privacy.normalize_trace(privacy_fixture._cpu_eager_trace_bytes())
        trace = {
            "name": "metadata-origin",
            "clock": normalized["clock"],
            "events": copy.deepcopy(normalized["events"]),
            "summary": normalized["summary"],
        }
        for event in trace["events"]:
            if event["phase"] != "metadata":
                event["start_us"] += 100
        expectation = privacy_fixture._witness_trace_policy(
            "metadata-origin",
            [[event["semantic_token"], event["phase"]] for event in trace["events"]],
        )
        with self.assertRaisesRegex(publication.PublicationError, "local origin"):
            publication._validate_trace_record(
                trace,
                expectation,
                "metadata-assisted origin",
            )

    def test_semantic_and_tolerance_rehashes_cannot_bypass_validation(self):
        def swap_event_kinds(entries):
            path = publication.SCOPE_PATHS["two_gpu"]
            scope = json.loads(entries[path])
            events = scope["traces"][0]["events"]
            events[2]["kind"], events[3]["kind"] = (
                events[3]["kind"],
                events[2]["kind"],
            )
            entries[path] = publication.canonical_json_bytes(scope)

        def add_event(entries):
            path = publication.SCOPE_PATHS["cpu"]
            scope = json.loads(entries[path])
            trace = scope["traces"][0]
            event = self._event(
                len(trace["events"]),
                "profiler-step",
                21,
                1,
            )
            trace["events"].append(event)
            trace["summary"]["event_count"] += 1
            entries[path] = publication.canonical_json_bytes(scope)

        def forge_tolerance(entries):
            correctness = json.loads(entries[publication.COMMITMENTS_PATH])
            comparison = correctness["cases"][0]["captures"][0]["arrays"][0][
                "comparison"
            ]
            comparison["max_abs_error"] = 1e90
            comparison["error_l2"] = 1e90
            comparison["max_tolerance_excess"] = -1.0
            entries[publication.COMMITMENTS_PATH] = publication.canonical_json_bytes(
                correctness
            )

        def forge_zero_reference_excess(entries):
            correctness = json.loads(entries[publication.COMMITMENTS_PATH])
            comparison = correctness["cases"][0]["captures"][0]["arrays"][0][
                "comparison"
            ]
            comparison.update(
                {
                    "max_abs_error": 0.0,
                    "max_allowed_error": comparison["atol"],
                    "max_tolerance_excess": 0.0,
                    "reference_abs_max": 0.0,
                    "reference_l2": 0.0,
                    "error_l2": 0.0,
                    "reference_all_zero": True,
                }
            )
            entries[publication.COMMITMENTS_PATH] = publication.canonical_json_bytes(
                correctness
            )

        for mutate in (
            swap_event_kinds,
            add_event,
            forge_tolerance,
            forge_zero_reference_excess,
        ):
            archive = self._rehash_archive(mutate)
            with (
                self.subTest(mutate=mutate.__name__),
                self.assertRaises(publication.PublicationError),
            ):
                publication.validate_public_archive(
                    archive,
                    expected_policy=self.policy,
                    expected_bindings=self.bindings,
                )

        def collapse_thread_context(entries):
            path = publication.SCOPE_PATHS["cpu"]
            scope = json.loads(entries[path])
            scope["traces"][0]["events"][2]["process_ordinal"] = 1
            entries[path] = publication.canonical_json_bytes(scope)

        archive = self._rehash_archive(collapse_thread_context)
        with self.assertRaises(publication.PublicationError):
            publication.validate_public_archive(
                archive,
                expected_policy=self.policy,
                expected_bindings=self.bindings,
            )

    def test_scanner_and_untrusted_parser_fail_closed(self):
        attacks = (
            {"note": "/var/folders/aa/bb"},
            {"note": "C:\\runner\\_work\\repo"},
            {"note": "Bearer abcdefghijklmnop"},
            {"credentials": "abcdefghijklmnop"},
            {"tokens": "abcdefghijklmnop"},
        )
        for attack in attacks:
            with (
                self.subTest(attack=attack),
                self.assertRaises(publication.PublicationError),
            ):
                publication.scan_public_bytes(publication.canonical_json_bytes(attack))
        malformed = (
            b"[" * 2000 + b"0" + b"]" * 2000 + b"\n",
            b'{"value":' + b"9" * 5000 + b"}\n",
        )
        for raw in malformed:
            with self.assertRaises(publication.PublicationError):
                publication.scan_public_bytes(raw)

        policy = copy.deepcopy(self.policy)
        policy["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0]["shape"] = [0]
        with self.assertRaises(publication.PublicationError):
            publication.validate_public_archive(
                self.assets[publication.TECHNICAL_EVIDENCE_ASSET],
                expected_policy=policy,
                expected_bindings=self.bindings,
            )

        for payload_name, size_bytes in (
            ("nested/private/data.whl", 10),
            ("nested/\x01data.whl", 10),
            ("．．/data.whl", 10),
            ("package／data.whl", 10),
            ("package＼data.whl", 10),
            ("C:/build/data.whl", 10),
            ("gmes.whl", publication.MAX_PAYLOAD_BYTES + 1),
        ):
            policy = copy.deepcopy(self.policy)
            policy["scopes"][4]["payloads"][0]["name"] = payload_name
            policy["scopes"][4]["payloads"][0]["size_bytes"] = size_bytes
            with self.assertRaises(publication.PublicationError):
                publication.validate_public_archive(
                    self.assets[publication.TECHNICAL_EVIDENCE_ASSET],
                    expected_policy=policy,
                    expected_bindings=self.bindings,
                )

        malformed_fields = (
            ("media_type", []),
            ("dtype", []),
            ("comparison_contract", []),
        )
        for field, malformed_value in malformed_fields:
            policy = copy.deepcopy(self.policy)
            if field == "media_type":
                policy["scopes"][4]["payloads"][0][field] = malformed_value
            else:
                policy["scopes"][0]["correctness"][0]["captures"][0]["arrays"][0][
                    field
                ] = malformed_value
            with self.assertRaises(publication.PublicationError):
                publication.validate_public_archive(
                    self.assets[publication.TECHNICAL_EVIDENCE_ASSET],
                    expected_policy=policy,
                    expected_bindings=self.bindings,
                )

    def test_independent_payload_path_validator_rejects_windows_aliases(self):
        names = (
            "C:relative/evidence.bin",
            "evidence/NUL.bin",
            "evidence/file.bin:stream.bin",
            "evidence/name./data.bin",
            "evidence/LPT\N{SUPERSCRIPT TWO}.bin",
            "．．/evidence.bin",
            "evidence/ｐｒｉｖａｔｅ/data.bin",
        )
        for name in names:
            with (
                self.subTest(name=name),
                self.assertRaises(publication.PublicationError),
            ):
                publication._portable_payload_name(
                    name, "downloaded payload", "application/octet-stream"
                )
            policy = copy.deepcopy(self.policy)
            projection = copy.deepcopy(self.projection)
            policy["scopes"][4]["payloads"][0]["name"] = name
            projection["technical_scopes"][4]["payloads"][0]["name"] = name
            with self.assertRaises(publication.PublicationError):
                publication.build_publication_assets(
                    projection,
                    expected_policy=policy,
                    expected_bindings=self.bindings,
                )

        def build_with_payload_name(name):
            policy = copy.deepcopy(self.policy)
            projection = copy.deepcopy(self.projection)
            policy["scopes"][4]["payloads"][0]["name"] = name
            projection["technical_scopes"][4]["payloads"][0]["name"] = name
            projection["technical_scopes"][4]["closure"]["payload_names"][0] = name
            return publication.build_publication_assets(
                projection,
                expected_policy=policy,
                expected_bindings=self.bindings,
            )

        assignments = (
            "CUDA_VISIBLE_DEVICES=0",
            _fullwidth_ascii("CUDA_VISIBLE_DEVICES=0"),
        )
        for assignment in assignments:
            with (
                self.subTest(assignment=assignment),
                self.assertRaisesRegex(
                    publication.PublicationError,
                    "environment or identity assignment",
                ),
            ):
                build_with_payload_name(f"{assignment}/evidence.whl")

        portable_name = "portable/PATH/evidence.whl"
        self.assertEqual(
            publication._portable_payload_name(
                portable_name, "downloaded payload", "application/vnd.python.wheel"
            ),
            portable_name,
        )
        portable_assets = build_with_payload_name(portable_name)
        self.assertEqual(
            list(portable_assets), [name for _role, name in publication.ASSET_ORDER]
        )

    def test_independent_text_scans_apply_nfkc_and_redact_keys(self):
        marker = "authorization_private_probe_value"
        with self.assertRaises(publication.PublicationError) as exact_keys:
            publication._exact_keys(
                {marker: True}, {"expected"}, "downloaded fixture object"
            )
        self.assertTrue(
            all(
                marker not in message
                for message in _exception_messages(exact_keys.exception)
            )
        )
        attacks = (
            {_fullwidth_ascii("hostname"): "samplehost"},
            {"public_label": _fullwidth_ascii("C:/Users/sampleuser/private-location")},
            {"public": {marker: "value"}},
        )
        for attack in attacks:
            raw = publication.canonical_json_bytes(attack)
            with (
                self.subTest(attack=attack),
                self.assertRaises(publication.PublicationError) as caught,
            ):
                publication.scan_public_bytes(raw, "downloaded fixture")
            self.assertTrue(_exception_messages(caught.exception))
            self.assertTrue(
                all(
                    marker not in message
                    for message in _exception_messages(caught.exception)
                )
            )

        environment_names = (
            "PATH",
            "HOME",
            "CUDA_VISIBLE_DEVICES",
            "RUNNER_NAME",
            "RUNNER_OS",
            "RUNNER_ARCH",
            "RUNNER_TEMP",
            "RUNNER_TOOL_CACHE",
            "RUNNER_WORKSPACE",
            "GITHUB_ACTOR",
            "GITHUB_TRIGGERING_ACTOR",
            "GITHUB_WORKSPACE",
            "GITHUB_REPOSITORY",
            "GITHUB_REPOSITORY_OWNER",
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "GITHUB_JOB",
            "GITHUB_WORKFLOW",
            "GITHUB_SHA",
            "GITHUB_REF",
            "GITHUB_HEAD_REF",
            "GITHUB_BASE_REF",
            "SSH_AUTH_SOCK",
            "SHELL",
            "TMPDIR",
            "TEMP",
            "TMP",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
        )
        for environment_name in environment_names:
            variants = (
                (
                    environment_name,
                    f"{environment_name}=synthetic-value",
                ),
                (
                    _fullwidth_ascii(environment_name),
                    _fullwidth_ascii(f"{environment_name}=synthetic-value"),
                ),
            )
            for key, assignment in variants:
                for attack in (
                    {key: "synthetic-value"},
                    {"public_label": assignment},
                ):
                    with (
                        self.subTest(
                            environment_name=environment_name,
                            attack=attack,
                        ),
                        self.assertRaises(publication.PublicationError),
                    ):
                        publication.scan_public_bytes(
                            publication.canonical_json_bytes(attack),
                            "downloaded fixture",
                        )

        publication.scan_public_bytes(
            publication.canonical_json_bytes(
                {"portable_path": ("portable/PATH/GITHUB_WORKSPACE/evidence.whl")}
            ),
            "downloaded fixture",
        )

        duplicate = (
            "{" + json.dumps(marker) + ":1," + json.dumps(marker) + ":2}\n"
        ).encode("utf-8")
        with self.assertRaises(publication.PublicationError) as caught:
            publication.scan_public_bytes(duplicate, "downloaded fixture")
        self.assertTrue(
            all(
                marker not in message
                for message in _exception_messages(caught.exception)
            )
        )

    def test_zip_metadata_and_name_attacks_fail(self):
        valid_entries = self._entries(self.assets[publication.TECHNICAL_EVIDENCE_ASSET])

        def archive_with(
            names, *, compression=zipfile.ZIP_STORED, mode=None, comment=b"", extra=b""
        ):
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w") as handle:
                handle.comment = comment
                for name, raw in names:
                    info = zipfile.ZipInfo(
                        name, date_time=publication.FIXED_ZIP_TIMESTAMP
                    )
                    info.compress_type = compression
                    info.create_system = 3
                    info.external_attr = (
                        mode if mode is not None else stat.S_IFREG | 0o644
                    ) << 16
                    info.extra = extra
                    handle.writestr(info, raw)
            return output.getvalue()

        ordered = [
            (name, valid_entries[name]) for name in publication.ARCHIVE_ENTRY_ORDER
        ]
        attacks = [
            archive_with(ordered, compression=zipfile.ZIP_DEFLATED),
            archive_with(ordered, mode=stat.S_IFLNK | 0o777),
            archive_with(ordered, comment=b"comment"),
            archive_with(ordered, extra=b"\x0a\x00\x00\x00"),
            archive_with(
                [
                    *ordered,
                    (
                        publication.MANIFEST_PATH,
                        valid_entries[publication.MANIFEST_PATH],
                    ),
                ]
            ),
            archive_with(
                [
                    (
                        ("../manifest.json", raw)
                        if name == publication.MANIFEST_PATH
                        else (name, raw)
                    )
                    for name, raw in ordered
                ]
            ),
            archive_with(
                [
                    (
                        ("SCOPES/01-CPU.JSON", raw)
                        if name == publication.SCOPE_PATHS["cpu"]
                        else (name, raw)
                    )
                    for name, raw in ordered
                ]
            ),
        ]
        local_name_mismatch = bytearray(
            self.assets[publication.TECHNICAL_EVIDENCE_ASSET]
        )
        local_name_mismatch[30] ^= 1
        attacks.append(bytes(local_name_mismatch))

        malformed_utf8 = bytearray(self.assets[publication.TECHNICAL_EVIDENCE_ASSET])
        central = malformed_utf8.index(b"PK\x01\x02")
        malformed_utf8[6:8] = (0x800).to_bytes(2, "little")
        malformed_utf8[central + 8 : central + 10] = (0x800).to_bytes(2, "little")
        malformed_utf8[30] = 0xFF
        malformed_utf8[central + 46] = 0xFF
        attacks.append(bytes(malformed_utf8))
        for index, archive in enumerate(attacks):
            with (
                self.subTest(index=index),
                self.assertRaises(publication.PublicationError),
            ):
                publication.validate_public_archive(
                    archive,
                    expected_policy=self.policy,
                    expected_bindings=self.bindings,
                )


if __name__ == "__main__":
    unittest.main()
