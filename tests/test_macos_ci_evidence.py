from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

from benchmarks import issue123_completion as completion
from benchmarks import macos_ci_evidence as evidence


class MacOSCiEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "evidence"
        self.records = self.root / "records"
        self.work = self.root / "work"
        self.packages = self.directory / "packages"
        self.logs = self.directory / "logs"
        self.packages.mkdir(parents=True)
        self.logs.mkdir()
        self.records.mkdir()
        self.work.mkdir()
        manifest_raw = evidence.DEFAULT_MANIFEST.read_bytes()
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
        self.platform = {
            "system": "Darwin",
            "machine": "arm64",
            "python": "3.14.7",
        }
        self.host = {
            "schema_version": 2,
            "common_identity": {
                "hostname": "macos-runner",
                "platform": "macOS-26-arm64",
                "os": {"system": "Darwin", "release": "25.4.0", "machine": "arm64"},
                "python": "3.14.7",
                "cxx_version": "Apple clang version 18",
                "swig_version": "SWIG Version 4.4.1",
                "uv_lock_sha256": "b" * 64,
            },
            "runtime_identity": {"torch": "2.10.0+cpu", "cuda_runtime": None},
        }
        self._write_sdist()
        self._write_wheel()
        for role in evidence.RUNTIME_ROLES:
            self._write_command_record(role)

    def _write_sdist(self):
        path = self.packages / "gmes-0.10.0.tar.gz"
        raw = b"Metadata-Version: 2.4\nName: gmes\n"
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo("gmes-0.10.0/PKG-INFO")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))

    def _write_wheel(self):
        path = self.packages / "gmes-0.10.0-cp314-cp314-macosx_11_0_arm64.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("gmes/__init__.py", b"__version__ = '0.10.0'\n")

    def _package(self, role):
        pattern = "*.whl" if role.startswith("wheel-") else "*.tar.gz"
        return next(self.packages.glob(pattern))

    @staticmethod
    def _field_record(name, initial=0.1, final=0.2):
        return evidence._array_record(name, [initial], [final])

    def _native(self):
        fields = [self._field_record(name) for name in evidence.FIELD_NAMES]
        addresses = {name: index + 1 for index, name in enumerate(evidence.FIELD_NAMES)}
        return {
            "openmp_enabled": False,
            "steps": {"initial": 0, "final": 2},
            "fields": fields,
            "initial_field_sha256": evidence._records_sha256(fields, "initial"),
            "final_field_sha256": evidence._records_sha256(fields, "final"),
            "storage_addresses": {"initial": addresses, "final": addresses},
            "storage_stable": True,
            "finite": True,
            "progressed": True,
            "passed": True,
        }

    def _torch_mode(self, mode):
        fields = [self._field_record(name) for name in evidence.FIELD_NAMES]
        states = [evidence._array_record("step_count", [0], [2])]
        addresses = {"state.ex": 100, "state.step_count": 200}
        zero = {"unique_graphs": 0, "calls_captured": 0, "graph_breaks": 0}
        hot = (
            zero
            if mode == "eager"
            else {"unique_graphs": 1, "calls_captured": 12, "graph_breaks": 0}
        )
        policy = "eager" if mode == "eager" else "compile"
        return {
            "mode": mode,
            "runtime": {
                "device": "cpu",
                "precision": "float64",
                "compile_policy": policy,
                "compile_mode": "default",
                "cpu_threads": 1,
                "cpu_interop_threads": 1,
            },
            "steps": {"initial": 0, "warmup": 1, "final": 2},
            "fields": fields,
            "state_buffers": states,
            "initial_field_sha256": evidence._records_sha256(fields, "initial"),
            "final_field_sha256": evidence._records_sha256(fields, "final"),
            "initial_state_sha256": evidence._records_sha256(states, "initial"),
            "final_state_sha256": evidence._records_sha256(states, "final"),
            "storage_addresses": {
                "initial": addresses,
                "warmup": addresses,
                "final": addresses,
            },
            "storage_stable": True,
            "compiler": {
                "before": copy.deepcopy(zero),
                "warmup": copy.deepcopy(hot),
                "final": copy.deepcopy(hot),
            },
            "finite": True,
            "progressed": True,
            "passed": True,
        }

    def _comparison(self, left, right):
        return {
            "left": left,
            "right": right,
            "tolerance": {"rtol": 1e-13, "atol": 1e-14},
            "fields": [
                {"name": name, "max_abs_error": 0.0, "passed": True}
                for name in evidence.FIELD_NAMES
            ],
            "passed": True,
        }

    def _result(self, role):
        package_sha256 = hashlib.sha256(self._package(role).read_bytes()).hexdigest()
        if role.endswith("-import"):
            return {
                "kind": evidence.IMPORT_RESULT_KIND,
                "role": role,
                "package_sha256": package_sha256,
                "platform": self.platform,
                "host_contract": self.host,
                "distribution": {
                    "name": "gmes",
                    "version": "0.10.0",
                    "module_path": "/venv/gmes/__init__.py",
                    "native_module_paths": [
                        "/venv/gmes/_constant.so",
                        "/venv/gmes/_pw_material.so",
                    ],
                    "outside_source": True,
                },
                "passed": True,
            }
        eager = self._torch_mode("eager")
        compiled = self._torch_mode("compile")
        return {
            "kind": evidence.SUITE_RESULT_KIND,
            "role": role,
            "mode": evidence.SUITE_MODES[role],
            "package_sha256": package_sha256,
            "platform": self.platform,
            "host_contract": self.host,
            "native": self._native(),
            "torch_cpu": {
                "modes": [eager, compiled],
                "comparisons": [
                    self._comparison("eager", "compile"),
                    self._comparison("eager", "native"),
                    self._comparison("compile", "native"),
                ],
            },
            "passed": True,
        }

    def _write_command_record(self, role):
        result = self._result(role)
        stdout = (
            json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode()
        stderr = b""
        (self.logs / f"{role}.stdout.json").write_bytes(stdout)
        (self.logs / f"{role}.stderr.txt").write_bytes(stderr)
        mode = evidence.SUITE_MODES.get(role)
        cache_directory = self.records / "torchinductor" / role
        cache_directory.mkdir(parents=True, exist_ok=True)
        environment = {
            "GMES_ENABLE_OPENMP": "0" if mode == "serial" else "auto",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(cache_directory),
        }
        package = self._package(role).resolve(strict=True)
        record = {
            "schema_version": 2,
            "kind": evidence.COMMAND_RECORD_KIND,
            "role": role,
            "command": {
                "argv": evidence._probe_argv(
                    sys.executable,
                    evidence.ROOT,
                    evidence.ROOT,
                    package,
                    role,
                    mode,
                ),
                "cwd": str(self.work.resolve(strict=True)),
                "environment": environment,
            },
            "exit_code": 0,
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stdout_size_bytes": len(stdout),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stderr_size_bytes": len(stderr),
            "result": result,
        }
        (self.records / f"{role}.json").write_text(json.dumps(record))

    def _capture(self):
        with (
            mock.patch.object(
                evidence,
                "_candidate_evidence",
                return_value=self.candidate,
            ),
            mock.patch.object(evidence, "_platform_record", return_value=self.platform),
        ):
            return evidence.capture_runtime_index(
                self.directory,
                self.records,
                evidence.ROOT,
                evidence.DEFAULT_MANIFEST,
                self.candidate["candidate_git_commit"],
            )

    def _write_actions_archive(self, runtime_index):
        runtime = json.loads(runtime_index.read_text())
        archive_path = self.root / "actions-archive.zip"
        descriptors = [item["artifact"] for item in runtime["packages"]]
        for check in runtime["runtime_checks"]:
            descriptors.extend((check["stdout"], check["stderr"]))
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(runtime_index, "runtime-index.json")
            for descriptor in descriptors:
                archive.write(self.directory / descriptor["path"], descriptor["path"])
        return archive_path

    def _github_responses(self, archive_path):
        candidate_sha = self.candidate["candidate_git_commit"]
        archive_raw = archive_path.read_bytes()
        artifact = {
            "id": 401,
            "name": f"issue-123-macos-{candidate_sha}",
            "size_in_bytes": len(archive_raw),
            "archive_download_url": (
                "https://api.github.com/repos/ruddyscent/gmes/actions/artifacts/401/zip"
            ),
            "expired": False,
            "created_at": "2026-08-31T00:05:00Z",
            "updated_at": "2026-08-31T00:06:00Z",
            "digest": f"sha256:{hashlib.sha256(archive_raw).hexdigest()}",
            "workflow_run": {
                "head_branch": "perf/123-candidate",
                "head_repository_id": 1,
                "head_sha": candidate_sha,
                "id": 101,
                "repository_id": 1,
            },
        }
        return {
            "repos/ruddyscent/gmes/actions/runs/101": {
                "id": 101,
                "name": "CI",
                "event": "pull_request",
                "head_sha": candidate_sha,
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
            },
            "repos/ruddyscent/gmes/actions/runs/101/jobs": {
                "jobs": [
                    {
                        "id": 2,
                        "run_id": 101,
                        "run_attempt": 1,
                        "name": evidence.MACOS_REQUIRED_JOB,
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2026-08-31T00:00:00Z",
                        "completed_at": "2026-08-31T00:10:00Z",
                    }
                ]
            },
            "repos/ruddyscent/gmes/actions/runs/101/artifacts": {
                "artifacts": [artifact]
            },
        }

    def test_capture_binds_packages_and_structured_command_outputs(self):
        runtime_index = self._capture()
        document = json.loads(runtime_index.read_text())

        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(
            [item["role"] for item in document["runtime_checks"]],
            list(evidence.RUNTIME_ROLES),
        )
        for check in document["runtime_checks"]:
            self.assertEqual(check["stdout"]["media_type"], evidence.MEDIA_TYPE_JSON)
            self.assertEqual(check["stderr"]["media_type"], evidence.MEDIA_TYPE_TEXT)
            self.assertEqual(check["result"]["host_contract"]["schema_version"], 2)
        evidence._load_runtime_index(runtime_index, evidence.DEFAULT_MANIFEST)

    def test_capture_rejects_embedded_result_that_differs_from_raw_stdout(self):
        role = "wheel-import"
        record_path = self.records / f"{role}.json"
        record = json.loads(record_path.read_text())
        record["result"]["passed"] = False
        record_path.write_text(json.dumps(record))

        with self.assertRaisesRegex(evidence.EvidenceError, "result differs"):
            self._capture()

    def test_capture_rejects_boolean_exit_code(self):
        role = "wheel-import"
        record_path = self.records / f"{role}.json"
        record = json.loads(record_path.read_text())
        record["exit_code"] = False
        record_path.write_text(json.dumps(record))

        with self.assertRaisesRegex(evidence.EvidenceError, "did not succeed"):
            self._capture()

    def test_suite_validation_recomputes_compiler_hot_path(self):
        role = "wheel-default-suite"
        result = self._result(role)
        result["torch_cpu"]["modes"][1]["compiler"]["final"]["unique_graphs"] = 2

        with self.assertRaisesRegex(evidence.EvidenceError, "summaries differ"):
            evidence._validate_probe_result(
                result,
                role,
                result["package_sha256"],
                self.platform,
            )

    def test_allocated_addresses_omit_zero_sized_tensors(self):
        addresses = evidence._allocated_addresses(
            {"state.field": 101, "plan.empty_targets": 0},
            "captured Torch addresses",
        )

        self.assertEqual(addresses, {"state.field": 101})
        evidence._validate_addresses(addresses, "captured Torch addresses")

    def test_allocated_addresses_reject_invalid_or_unallocated_maps(self):
        for addresses in (
            {"state.field": -1},
            {"state.field": False},
            {"plan.empty_targets": 0},
        ):
            with (
                self.subTest(addresses=addresses),
                self.assertRaisesRegex(evidence.EvidenceError, "addresses"),
            ):
                evidence._allocated_addresses(addresses, "captured Torch addresses")

        for addresses in ({"state.field": 0}, {"state.field": -1}):
            with (
                self.subTest(serialized=addresses),
                self.assertRaisesRegex(evidence.EvidenceError, "addresses"),
            ):
                evidence._validate_addresses(addresses, "captured Torch addresses")

    def test_inductor_import_filter_ignores_only_the_exact_warning(self):
        class Simulation:
            def __init__(self, message, module):
                self.message = message
                self.module = module
                self.calls = []

            def advance(self, steps):
                self.calls.append(steps)
                warnings.warn_explicit(
                    self.message,
                    DeprecationWarning,
                    filename="torch/jit/_script.py",
                    lineno=359,
                    module=self.module,
                )

        exact = Simulation(
            evidence.TORCH_JIT_SCRIPT_METHOD_PY314_WARNING, "torch.jit._script"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            evidence._run_under_inductor_import_warning_filter(exact.advance, 1)
            self.assertEqual(exact.calls, [1])
            with self.assertRaises(DeprecationWarning):
                warnings.warn_explicit(
                    evidence.TORCH_JIT_SCRIPT_METHOD_PY314_WARNING,
                    DeprecationWarning,
                    filename="torch/jit/_script.py",
                    lineno=359,
                    module="torch.jit._script",
                )

        for message, module in (
            ("an unrelated deprecation", "torch.jit._script"),
            (evidence.TORCH_JIT_SCRIPT_METHOD_PY314_WARNING, "torch.jit._other"),
        ):
            with (
                self.subTest(message=message, module=module),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("error")
                with self.assertRaises(DeprecationWarning):
                    simulation = Simulation(message, module)
                    evidence._run_under_inductor_import_warning_filter(
                        simulation.advance, 1
                    )

    def test_installed_archive_requires_exact_pep610_url_and_sha(self):
        package = self._package("wheel-import").resolve()
        digest = hashlib.sha256(package.read_bytes()).hexdigest()
        distribution = mock.Mock()
        distribution.read_text.return_value = json.dumps(
            {"archive_info": {"hashes": {"sha256": digest}}, "url": package.as_uri()}
        )
        with mock.patch.object(
            evidence.metadata, "distribution", return_value=distribution
        ):
            self.assertEqual(evidence._installed_archive_sha256(package), digest)

        distribution.read_text.return_value = json.dumps(
            {"archive_info": {"hashes": {"sha256": "0" * 64}}, "url": package.as_uri()}
        )
        with (
            mock.patch.object(
                evidence.metadata, "distribution", return_value=distribution
            ),
            self.assertRaisesRegex(evidence.EvidenceError, "digest differs"),
        ):
            evidence._installed_archive_sha256(package)

    def test_installed_archive_accepts_uv_empty_pep610_archive_info(self):
        package = self._package("wheel-import").resolve()
        digest = hashlib.sha256(package.read_bytes()).hexdigest()
        distribution = mock.Mock()
        distribution.read_text.return_value = json.dumps(
            {"archive_info": {}, "url": package.as_uri()}
        )
        with mock.patch.object(
            evidence.metadata, "distribution", return_value=distribution
        ):
            self.assertEqual(evidence._installed_archive_sha256(package), digest)

    def test_installed_archive_rejects_malformed_or_symlinked_pep610_url(self):
        package = self._package("wheel-import").resolve()
        digest = hashlib.sha256(package.read_bytes()).hexdigest()
        distribution = mock.Mock()
        distribution.read_text.return_value = json.dumps(
            {"archive_info": {"hash": f"sha256={digest}"}, "url": package.as_uri()}
        )
        with mock.patch.object(
            evidence.metadata, "distribution", return_value=distribution
        ):
            self.assertEqual(evidence._installed_archive_sha256(package), digest)

        distribution.read_text.return_value = json.dumps(
            {"archive_info": {"hash": "sha512=not-a-sha256"}, "url": package.as_uri()}
        )
        with (
            mock.patch.object(
                evidence.metadata, "distribution", return_value=distribution
            ),
            self.assertRaisesRegex(evidence.EvidenceError, "digest differs"),
        ):
            evidence._installed_archive_sha256(package)

        distribution.read_text.return_value = json.dumps(
            {"archive_info": {"hashes": None}, "url": package.as_uri()}
        )
        with (
            mock.patch.object(
                evidence.metadata, "distribution", return_value=distribution
            ),
            self.assertRaisesRegex(evidence.EvidenceError, "digest differs"),
        ):
            evidence._installed_archive_sha256(package)

        alias = self.root / "package-alias.whl"
        alias.symlink_to(package)
        distribution.read_text.return_value = json.dumps(
            {"archive_info": {}, "url": alias.as_uri()}
        )
        with (
            mock.patch.object(
                evidence.metadata, "distribution", return_value=distribution
            ),
            self.assertRaisesRegex(evidence.EvidenceError, "local evidence archive"),
        ):
            evidence._installed_archive_sha256(package)

        distribution.read_text.return_value = json.dumps(
            {"archive_info": {}, "url": f"{package.as_uri()}#unexpected"}
        )
        with (
            mock.patch.object(
                evidence.metadata, "distribution", return_value=distribution
            ),
            self.assertRaisesRegex(evidence.EvidenceError, "local evidence archive"),
        ):
            evidence._installed_archive_sha256(package)

    def test_actions_archive_requires_exact_fifteen_files(self):
        runtime_index = self._capture()
        runtime = json.loads(runtime_index.read_text())
        archive_path = self._write_actions_archive(runtime_index)
        evidence._validate_actions_archive(
            archive_path,
            runtime_index,
            runtime,
            self.candidate,
        )

        with zipfile.ZipFile(archive_path, "a") as archive:
            archive.writestr("unexpected.txt", b"extra")
        with self.assertRaisesRegex(evidence.EvidenceError, "member closure"):
            evidence._validate_actions_archive(
                archive_path,
                runtime_index,
                runtime,
                self.candidate,
            )

    def test_assemble_contains_only_macos_runtime_and_artifact_binding(self):
        runtime_index = self._capture()
        source_archive = self._write_actions_archive(runtime_index)
        archive_path = self.directory / "actions-archive.zip"
        archive_path.write_bytes(source_archive.read_bytes())
        responses = self._github_responses(archive_path)

        def github_api(endpoint, *_fields):
            return copy.deepcopy(responses[endpoint])

        with mock.patch.object(evidence, "_github_api", side_effect=github_api):
            index_path, scope_path = evidence.assemble_macos_index(
                runtime_index=runtime_index,
                manifest=evidence.DEFAULT_MANIFEST,
                repository="ruddyscent/gmes",
                ci_run_id=101,
                actions_archive=archive_path,
                output=self.directory / "index.json",
            )
        document = json.loads(index_path.read_text())
        scope = json.loads(scope_path.read_text())
        self.assertEqual(
            set(document),
            {
                "schema_version",
                "kind",
                "candidate_evidence",
                "actions_artifact",
                "packages",
                "runtime_checks",
                "passed",
            },
        )
        self.assertNotIn("jobs", document)
        self.assertNotIn("pull_request", document)
        self.assertNotIn("code_scanning_analyses", document)
        self.assertEqual(set(scope), {"index", "actions_archive"})
        self.assertEqual(scope["index"]["media_type"], evidence.MEDIA_TYPE_JSON)
        self.assertEqual(
            scope["actions_archive"]["media_type"], evidence.MEDIA_TYPE_ZIP
        )
        result = completion._validate_macos_scope(
            scope,
            completion.ArtifactReader(self.directory, self.candidate),
            self.candidate,
        )
        self.assertEqual(result["runtime_checks"], list(evidence.RUNTIME_ROLES))

        stdout_path = self.logs / "wheel-default-suite.stdout.json"
        stdout = json.loads(stdout_path.read_text())
        stdout["torch_cpu"]["modes"][1]["compiler"]["final"]["unique_graphs"] = 2
        stdout_path.write_text(json.dumps(stdout))
        with self.assertRaisesRegex(
            completion.EvidenceError, "(?:byte size|digest) differs"
        ):
            completion._validate_macos_scope(
                scope,
                completion.ArtifactReader(self.directory, self.candidate),
                self.candidate,
            )

    def test_assemble_rejects_non_pull_request_artifact_run(self):
        runtime_index = self._capture()
        source_archive = self._write_actions_archive(runtime_index)
        archive_path = self.directory / "actions-archive.zip"
        archive_path.write_bytes(source_archive.read_bytes())
        responses = self._github_responses(archive_path)
        responses["repos/ruddyscent/gmes/actions/runs/101"]["event"] = "push"

        def github_api(endpoint, *_fields):
            return copy.deepcopy(responses[endpoint])

        with (
            mock.patch.object(evidence, "_github_api", side_effect=github_api),
            self.assertRaisesRegex(evidence.EvidenceError, "successful candidate run"),
        ):
            evidence.assemble_macos_index(
                runtime_index=runtime_index,
                manifest=evidence.DEFAULT_MANIFEST,
                repository="ruddyscent/gmes",
                ci_run_id=101,
                actions_archive=archive_path,
                output=self.directory / "index.json",
            )

    def test_record_preserves_real_argv_exit_stdout_and_stderr(self):
        role = "wheel-import"
        expected = self._result(role)
        stdout = json.dumps(expected, separators=(",", ":"), sort_keys=True).encode()
        stderr = b"native diagnostic\n"
        completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr)
        package = self._package(role)

        with mock.patch.object(
            evidence.subprocess, "run", return_value=completed
        ) as run:
            exit_code = evidence.record_runtime_command(
                role=role,
                mode=None,
                repository=evidence.ROOT,
                forbidden_root=evidence.ROOT,
                expected_package=package,
                evidence_directory=self.directory,
                records_directory=self.records,
                working_directory=self.work,
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual((self.logs / f"{role}.stdout.json").read_bytes(), stdout)
        self.assertEqual((self.logs / f"{role}.stderr.txt").read_bytes(), stderr)
        record = json.loads((self.records / f"{role}.json").read_text())
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["command"]["argv"], run.call_args.args[0])
        self.assertEqual(
            record["command"]["argv"][:4], [sys.executable, "-I", "-W", "error"]
        )
        self.assertEqual(
            set(record["command"]["environment"]), evidence.COMMAND_ENVIRONMENT_KEYS
        )

    def test_runtime_index_rejects_noncanonical_cache_directory(self):
        role = "wheel-import"
        record_path = self.records / f"{role}.json"
        record = json.loads(record_path.read_text())
        canonical_cache = self.records / "torchinductor" / role
        canonical_cache.mkdir(parents=True, exist_ok=True)
        alias = self.root / "cache-alias"
        alias.symlink_to(canonical_cache, target_is_directory=True)
        record["command"]["environment"]["TORCHINDUCTOR_CACHE_DIR"] = str(alias)
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(evidence.EvidenceError, "environment differs"):
            self._capture()
        record["command"]["environment"]["TORCHINDUCTOR_CACHE_DIR"] = "relative"
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(evidence.EvidenceError, "environment differs"):
            self._capture()

    def test_cache_path_accepts_only_darwin_system_root_aliases(self):
        expected = Path("/private/var/folders/example/torchinductor/wheel-import")
        with mock.patch.object(evidence.platform, "system", return_value="Darwin"):
            self.assertTrue(
                evidence._path_is_exact(
                    "/var/folders/example/torchinductor/wheel-import", expected
                )
            )
            self.assertTrue(evidence._path_is_exact(str(expected), expected))
            self.assertFalse(
                evidence._path_is_exact(
                    "/opt/folders/example/torchinductor/wheel-import", expected
                )
            )
        with mock.patch.object(evidence.platform, "system", return_value="Linux"):
            self.assertFalse(
                evidence._path_is_exact(
                    "/var/folders/example/torchinductor/wheel-import", expected
                )
            )


if __name__ == "__main__":
    unittest.main()
