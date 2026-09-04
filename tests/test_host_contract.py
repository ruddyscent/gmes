from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmarks import host_contract


class HostContractTest(unittest.TestCase):
    def setUp(self):
        self.common = {
            "hostname": "issue-123-host",
            "platform": "Linux-6.8.0-x86_64-with-glibc2.39",
            "os": {
                "system": "Linux",
                "release": "6.8.0",
                "machine": "x86_64",
            },
            "python": "3.14.0",
            "cxx_version": "c++ 14.2.0",
            "swig_version": "SWIG Version 4.3.1",
            "uv_lock_sha256": "a" * 64,
        }

    def contract(self, torch_version, cuda_runtime):
        return {
            "schema_version": 2,
            "common_identity": copy.deepcopy(self.common),
            "runtime_identity": {
                "torch": torch_version,
                "cuda_runtime": cuda_runtime,
            },
        }

    def test_complete_accepts_separate_cpu_and_cuda_runtime_identities(self):
        cpu = self.contract("2.9.0+cpu", None)
        cuda = self.contract("2.9.0+cu130", "13.0")
        self.assertTrue(host_contract.host_contract_complete(cpu))
        self.assertTrue(host_contract.host_contract_complete(cuda))
        self.assertEqual(cpu["common_identity"], cuda["common_identity"])
        self.assertNotEqual(cpu["runtime_identity"], cuda["runtime_identity"])

    def test_complete_rejects_non_exact_or_incomplete_documents(self):
        mutations = []
        wrong_schema = self.contract("2.9.0+cpu", None)
        wrong_schema["schema_version"] = True
        mutations.append(wrong_schema)
        extra = self.contract("2.9.0+cpu", None)
        extra["extra"] = None
        mutations.append(extra)
        missing = self.contract("2.9.0+cpu", None)
        del missing["common_identity"]["os"]
        mutations.append(missing)
        bad_digest = self.contract("2.9.0+cpu", None)
        bad_digest["common_identity"]["uv_lock_sha256"] = "A" * 64
        mutations.append(bad_digest)
        empty_runtime = self.contract("", None)
        mutations.append(empty_runtime)
        malformed_cuda = self.contract("2.9.0+cu130", True)
        mutations.append(malformed_cuda)

        for value in mutations:
            with self.subTest(value=value):
                self.assertFalse(host_contract.host_contract_complete(value))

    def test_command_text_fails_closed_on_execution_exit_and_empty_output(self):
        with mock.patch.object(
            host_contract.subprocess,
            "run",
            side_effect=OSError("unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be executed"):
                host_contract._command_text("tool", "--version")

        completed = SimpleNamespace(returncode=7, stdout="ignored")
        with mock.patch.object(host_contract.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "exited with 7"):
                host_contract._command_text("tool", "--version")

        completed = SimpleNamespace(returncode=0, stdout="  \n")
        with mock.patch.object(host_contract.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "empty output"):
                host_contract._command_text("tool", "--version")
            self.assertEqual(
                host_contract._command_text("git", "status", allow_empty=True),
                "",
            )

    def test_candidate_evidence_requires_full_commit_and_clean_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_bytes(b"{}\n")
            with mock.patch.object(
                host_contract,
                "_command_text",
                side_effect=["b" * 40, ""],
            ):
                value = host_contract.candidate_evidence(manifest)
        self.assertEqual(value["candidate_git_commit"], "b" * 40)
        self.assertEqual(value["candidate_git_status"], "")
        self.assertEqual(
            value["manifest_sha256"],
            hashlib.sha256(b"{}\n").hexdigest(),
        )

        with mock.patch.object(
            host_contract,
            "_command_text",
            side_effect=["b" * 40, " M tracked.py"],
        ):
            with self.assertRaisesRegex(RuntimeError, "not clean"):
                host_contract.candidate_evidence(host_contract.DEFAULT_MANIFEST)

        with mock.patch.object(
            host_contract,
            "_command_text",
            side_effect=["short", ""],
        ):
            with self.assertRaisesRegex(RuntimeError, "full lowercase"):
                host_contract.candidate_evidence(host_contract.DEFAULT_MANIFEST)

    def test_capture_emits_schema_v2_and_propagates_command_failure(self):
        torch_module = SimpleNamespace(
            __version__="2.9.0+cu130",
            version=SimpleNamespace(cuda="13.0"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "uv.lock"
            lock.write_bytes(b"frozen-lock\n")
            with (
                mock.patch.object(host_contract, "ROOT", root),
                mock.patch.object(
                    host_contract,
                    "_command_text",
                    side_effect=["c++ 14.2.0", "SWIG Version 4.3.1"],
                ),
                mock.patch.object(host_contract.platform, "node", return_value="host"),
                mock.patch.object(
                    host_contract.platform,
                    "platform",
                    return_value="Linux-platform",
                ),
                mock.patch.object(
                    host_contract.platform,
                    "system",
                    return_value="Linux",
                ),
                mock.patch.object(
                    host_contract.platform,
                    "release",
                    return_value="6.8.0",
                ),
                mock.patch.object(
                    host_contract.platform,
                    "machine",
                    return_value="x86_64",
                ),
                mock.patch.object(
                    host_contract.platform,
                    "python_version",
                    return_value="3.14.0",
                ),
            ):
                result = host_contract.capture_host_contract(torch_module)
        self.assertTrue(host_contract.host_contract_complete(result))
        self.assertEqual(
            result["common_identity"]["uv_lock_sha256"],
            hashlib.sha256(b"frozen-lock\n").hexdigest(),
        )
        self.assertEqual(
            result["runtime_identity"],
            {"torch": "2.9.0+cu130", "cuda_runtime": "13.0"},
        )

        with mock.patch.object(
            host_contract,
            "_command_text",
            side_effect=RuntimeError("command failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "command failed"):
                host_contract.capture_host_contract(torch_module)


if __name__ == "__main__":
    unittest.main()
