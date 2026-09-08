"""Unit-only checks for the local installed-package cutover helper."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

_HELPER_PATH = Path(__file__).parents[1] / "benchmarks" / "package_cutover.py"
_SPEC = importlib.util.spec_from_file_location("package_cutover", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cutover = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cutover
_SPEC.loader.exec_module(cutover)


class _Distribution:
    def __init__(self, root, digest):
        self.root = root
        self.digest = digest
        self.version = "0.10.0"
        self.metadata = {"Name": "gmes"}

    def read_text(self, name):
        assert name == "direct_url.json"
        return json.dumps(
            {
                "url": "https://token@example.invalid/gmes.whl?secret=no",
                "archive_info": {"hash": f"sha256={self.digest}"},
            }
        )

    def locate_file(self, name):
        assert name == "gmes"
        return self.root


class TestPackageCutoverUnit:
    """These tests use fake metadata only; they are not installed-artifact proof."""

    @pytest.fixture(autouse=True)
    def _temporary_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            self.root = Path(directory)
            self.installed = self.root / "site-packages" / "gmes"
            self.installed.mkdir(parents=True)
            (self.installed / "__init__.py").write_text("\n")
            self.checkout = self.root / "checkout"
            self.checkout.mkdir()
            self.archive = self.root / "gmes-0.10.0-py3-none-any.whl"
            self.archive.write_bytes(b"unit-only archive")
            yield

    def test_provenance_requires_exact_pep610_archive_hash_and_scrubs_url(self):
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        result = cutover.installed_provenance(
            self.archive, distribution=_Distribution(self.installed, digest)
        )
        assert result["archive"]["sha256"] == digest
        assert result["direct_url"]["url"] == "https://example.invalid/gmes.whl"
        with pytest.raises(cutover.CutoverError, match="does not match"):
            cutover.installed_provenance(
                self.archive, distribution=_Distribution(self.installed, "0" * 64)
            )

    def test_origin_validation_rejects_checkout_and_native_gmes_modules(self):
        module = types.ModuleType("gmes")
        module.__file__ = str(self.installed / "__init__.py")
        assert cutover.verify_module_origins(
            self.installed, (self.checkout,), modules={"gmes": module}
        ) == {"gmes": str((self.installed / "__init__.py").resolve())}
        module.__file__ = str(self.checkout / "gmes.py")
        (self.checkout / "gmes.py").write_text("\n")
        with pytest.raises(cutover.CutoverError, match="outside"):
            cutover.verify_module_origins(
                self.installed, (self.checkout,), modules={"gmes": module}
            )
        native = types.ModuleType("gmes._removed")
        native_path = self.installed / "_removed.so"
        native_path.write_bytes(b"")
        native.__file__ = str(native_path)
        base = types.ModuleType("gmes")
        base.__file__ = str(self.installed / "__init__.py")
        with pytest.raises(cutover.CutoverError, match="native"):
            cutover.verify_module_origins(
                self.installed,
                (self.checkout,),
                modules={"gmes": base, "gmes._removed": native},
            )

    def test_cpu_contract_and_two_gpu_device_requirements_are_fail_closed(self):
        assert cutover._require_device("cpu", 0) is None
        with pytest.raises(cutover.CutoverError, match="requires --required"):
            cutover._require_device("cpu", 1)
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 2
            ),
            distributed=types.SimpleNamespace(is_nccl_available=lambda: True),
        )
        with mock.patch.object(
            cutover.importlib, "import_module", return_value=fake_torch
        ):
            assert cutover._require_device("cuda:0", 2) is fake_torch
            with pytest.raises(cutover.CutoverError, match="exactly two"):
                cutover._require_device("cuda:0", 3)

    def test_two_gpu_result_requires_real_rank_ownership_and_replay_evidence(self):
        provenance = {"archive": {"sha256": "a" * 64}}
        ranks = [
            {
                "rank": rank,
                "local_rank": rank,
                "isolated": True,
                "device": f"cuda:{rank}",
                "current_device": rank,
                "source_batches": 1 if rank == 0 else 0,
                "probe_samples": 2 if rank == 0 else 0,
                "checkpoint_replay": True,
                "distributed_seconds": 0.2,
                "provenance": provenance,
                "module_origins_after": {"gmes": f"/installed/{rank}/gmes/__init__.py"},
            }
            for rank in (0, 1)
        ]
        result = {
            "ranks": ranks,
            "initial_field_sha256": "first",
            "checkpoint_field_sha256": "second",
            "restored_field_sha256": "second",
            "global_checkpoint_replay": True,
            "single_gpu": {
                "seconds": 0.1,
                "probe_samples": 2,
                "checkpoint_replay": True,
            },
            "single_vs_two_maximum_error": 0.0,
            "two_gpu_seconds": 0.2,
            "informational_speedup": 0.5,
        }
        validated = cutover._validate_two_gpu_result(result, provenance)
        assert validated["device_count"] == 2
        assert validated["single_vs_two_maximum_error"] == 0.0
        ranks[1]["device"] = "cuda:0"
        with pytest.raises(cutover.CutoverError, match="ownership"):
            cutover._validate_two_gpu_result(result, provenance)

    def test_field_comparison_rejects_nonfinite_later_component(self):
        fields = {
            name: np.zeros((2, 2, 2), dtype=np.float64)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
        altered = {name: value.copy() for name, value in fields.items()}
        altered["Hz"][0, 0, 0] = np.nan
        with pytest.raises(cutover.CutoverError, match="Hz.*non-finite"):
            cutover._maximum_field_error(fields, altered)
        altered["Hz"] = np.zeros((1, 2, 2), dtype=np.float64)
        with pytest.raises(cutover.CutoverError, match="Hz.*mismatched shapes"):
            cutover._maximum_field_error(fields, altered)

    def test_evidence_record_uses_exclusive_creation(self):
        evidence = self.root / "evidence"
        first = {"schema": cutover.EVIDENCE_SCHEMA, "first": True}
        path = cutover._write_record(evidence, first)
        with pytest.raises(cutover.CutoverError, match="already exists"):
            cutover._write_record(evidence, {"schema": cutover.EVIDENCE_SCHEMA})
        assert json.loads(path.read_text()) == first

    def test_main_preserves_failed_worker_output_in_evidence(self):
        evidence = self.root / "worker-evidence"
        request = [
            "--candidate-label",
            "candidate-under-test",
            "--archive",
            str(self.archive),
            "--forbidden-root",
            str(self.checkout),
            "--device",
            "cuda:0",
            "--required-device-count",
            "2",
            "--evidence-dir",
            str(evidence),
        ]
        worker = {
            "argv": ["torchrun"],
            "exit_code": 1,
            "stdout": "out",
            "stderr": "err",
        }
        with mock.patch.object(
            cutover,
            "run_cutover",
            side_effect=cutover.WorkerCutoverError("worker failed", worker),
        ):
            assert cutover.main(request) == 1
        record = json.loads((evidence / cutover.EVIDENCE_FILENAME).read_text())
        assert record["result"]["worker"] == worker

    def test_main_records_real_outcome_without_claiming_publication(self):
        evidence = self.root / "evidence"
        request = [
            "--candidate-label",
            "candidate-under-test",
            "--archive",
            str(self.archive),
            "--forbidden-root",
            str(self.checkout),
            "--device",
            "cpu",
            "--required-device-count",
            "0",
            "--evidence-dir",
            str(evidence),
        ]
        with mock.patch.object(
            cutover,
            "run_cutover",
            return_value={"schema": cutover.EVIDENCE_SCHEMA, "passed": True},
        ):
            assert cutover.main(request) == 0
        record = json.loads((evidence / cutover.EVIDENCE_FILENAME).read_text())
        assert record["status"] == "passed"
        assert record["exit_code"] == 0
        assert "--candidate-label" in record["command"]["argv"]
        assert "not publication evidence" in record["scope"]
