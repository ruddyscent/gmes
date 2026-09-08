from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from benchmarks import host_contract


class TestHostContract:
    @pytest.fixture(autouse=True)
    def _common_identity(self):
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
        assert host_contract.host_contract_complete(cpu)
        assert host_contract.host_contract_complete(cuda)
        assert cpu["common_identity"] == cuda["common_identity"]
        assert cpu["runtime_identity"] != cuda["runtime_identity"]

    @pytest.mark.parametrize(
        "mutation",
        (
            "wrong-schema",
            "extra-key",
            "missing-os",
            "bad-digest",
            "empty-runtime",
            "malformed-cuda",
        ),
        ids=str,
    )
    def test_complete_rejects_non_exact_or_incomplete_documents(self, mutation):
        value = self.contract("2.9.0+cpu", None)
        if mutation == "wrong-schema":
            value["schema_version"] = True
        elif mutation == "extra-key":
            value["extra"] = None
        elif mutation == "missing-os":
            del value["common_identity"]["os"]
        elif mutation == "bad-digest":
            value["common_identity"]["uv_lock_sha256"] = "A" * 64
        elif mutation == "empty-runtime":
            value["runtime_identity"]["torch"] = ""
        else:
            value["runtime_identity"] = {
                "torch": "2.9.0+cu130",
                "cuda_runtime": True,
            }

        assert not host_contract.host_contract_complete(value)

    def test_command_text_fails_closed_on_execution_exit_and_empty_output(self):
        with mock.patch.object(
            host_contract.subprocess,
            "run",
            side_effect=OSError("unavailable"),
        ):
            with pytest.raises(RuntimeError, match="could not be executed"):
                host_contract._command_text("tool", "--version")

        completed = SimpleNamespace(returncode=7, stdout="ignored")
        with mock.patch.object(host_contract.subprocess, "run", return_value=completed):
            with pytest.raises(RuntimeError, match="exited with 7"):
                host_contract._command_text("tool", "--version")

        completed = SimpleNamespace(returncode=0, stdout="  \n")
        with mock.patch.object(host_contract.subprocess, "run", return_value=completed):
            with pytest.raises(RuntimeError, match="empty output"):
                host_contract._command_text("tool", "--version")
            assert host_contract._command_text("git", "status", allow_empty=True) == ""

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
        assert value["candidate_git_commit"] == "b" * 40
        assert value["candidate_git_status"] == ""
        assert value["manifest_sha256"] == hashlib.sha256(b"{}\n").hexdigest()

        with mock.patch.object(
            host_contract,
            "_command_text",
            side_effect=["b" * 40, " M tracked.py"],
        ):
            with pytest.raises(RuntimeError, match="not clean"):
                host_contract.candidate_evidence(host_contract.DEFAULT_MANIFEST)

        with mock.patch.object(
            host_contract,
            "_command_text",
            side_effect=["short", ""],
        ):
            with pytest.raises(RuntimeError, match="full lowercase"):
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
        assert host_contract.host_contract_complete(result)
        assert (
            result["common_identity"]["uv_lock_sha256"]
            == hashlib.sha256(b"frozen-lock\n").hexdigest()
        )
        assert result["runtime_identity"] == {
            "torch": "2.9.0+cu130",
            "cuda_runtime": "13.0",
        }

        with mock.patch.object(
            host_contract,
            "_command_text",
            side_effect=RuntimeError("command failed"),
        ):
            with pytest.raises(RuntimeError, match="command failed"):
                host_contract.capture_host_contract(torch_module)
