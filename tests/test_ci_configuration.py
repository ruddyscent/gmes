"""Regression tests for the pure-package GitHub Actions policy."""

import unittest
from pathlib import Path


class CiConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow_directory = (
            Path(__file__).resolve().parents[1] / ".github" / "workflows"
        )
        cls.ci_workflow = (workflow_directory / "ci.yml").read_text()
        cls.codeql_workflow = (workflow_directory / "codeql.yml").read_text()
        cls.prerelease_workflow = (workflow_directory / "prerelease.yml").read_text()
        cls.release_workflow = (workflow_directory / "release.yml").read_text()

    def test_required_ci_is_limited_to_master_and_cancels_stale_runs(self):
        self.assertEqual(self.ci_workflow.count("      - master"), 2)
        self.assertIn(
            "group: ci-${{ github.workflow }}-${{ github.ref }}", self.ci_workflow
        )
        self.assertIn("cancel-in-progress: true", self.ci_workflow)
        self.assertIn("name: Python 3.14 / ${{ matrix.os }}", self.ci_workflow)
        self.assertIn("          - ubuntu-latest", self.ci_workflow)
        self.assertIn("          - macos-latest", self.ci_workflow)
        self.assertIn("      fail-fast: false", self.ci_workflow)
        self.assertIn("permissions:\n  contents: read", self.ci_workflow)
        self.assertNotIn("prerelease:", self.ci_workflow)

    def test_required_ci_uses_locked_cpu_torch_and_pure_build(self):
        primary_steps = self.ci_workflow.split(
            "      - name: Check out candidate head", 1
        )[0]
        self.assertIn("uv sync --locked --extra torch-cpu --extra hdf5", primary_steps)
        self.assertIn("uv run --no-sync python -m unittest discover -v", primary_steps)
        self.assertIn("uv build", primary_steps)
        self.assertNotIn("brew install", primary_steps)
        self.assertNotIn("GMES_ENABLE_OPENMP", primary_steps)
        self.assertNotIn("swig", primary_steps)
        self.assertNotIn("mpi", primary_steps.lower())

    def test_required_ci_runs_static_and_supported_stub_checks(self):
        self.assertIn("python -m mypy", self.ci_workflow)
        self.assertIn(
            "python -m mypy.stubtest --mypy-config-file pyproject.toml gmes.constant",
            self.ci_workflow,
        )
        self.assertNotIn("gmes.pw_material", self.ci_workflow)

    def test_required_ci_lints_only_tracked_python_sources(self):
        self.assertIn(
            "python -m pylint $(git ls-files 'gmes/*.py') setup.py",
            self.ci_workflow,
        )

    def test_macos_required_job_uses_the_same_locked_cpu_environment(self):
        self.assertIn(
            "          - macos-latest",
            self.ci_workflow,
        )
        self.assertIn(
            "uv sync --locked --extra torch-cpu --extra hdf5", self.ci_workflow
        )

    def test_prerelease_is_scheduled_advisory_using_nightly_wheels(self):
        self.assertIn("name: Python prerelease (advisory)", self.prerelease_workflow)
        self.assertIn("schedule:", self.prerelease_workflow)
        self.assertIn("workflow_dispatch:", self.prerelease_workflow)
        self.assertIn("--only-binary=:all:", self.prerelease_workflow)
        self.assertIn(
            "scientific-python-nightly-wheels/simple", self.prerelease_workflow
        )
        self.assertIn("python -m pip install --no-deps -e .", self.prerelease_workflow)
        self.assertIn("python -m unittest discover -v", self.prerelease_workflow)

    def test_workflow_checkouts_disable_persisted_credentials(self):
        checkout = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
        for workflow in (
            self.ci_workflow,
            self.codeql_workflow,
            self.prerelease_workflow,
            self.release_workflow,
        ):
            with self.subTest(workflow=workflow):
                self.assertEqual(
                    workflow.count(checkout),
                    workflow.count("persist-credentials: false"),
                )

    def test_codeql_analyzes_python_without_a_native_build(self):
        self.assertIn("name: CodeQL / python", self.codeql_workflow)
        self.assertNotIn("matrix.", self.codeql_workflow)
        self.assertIn("security-events: write", self.codeql_workflow)
        self.assertIn("languages: python", self.codeql_workflow)
        self.assertIn("build-mode: none", self.codeql_workflow)
        self.assertNotIn("languages: c-cpp", self.codeql_workflow)
        self.assertNotIn("swig", self.codeql_workflow)

    def test_release_builds_and_reuses_exact_universal_archives(self):
        self.assertIn("uv build --clear --out-dir release-dist", self.release_workflow)
        self.assertIn("gmes-*-py3-none-any.whl", self.release_workflow)
        self.assertIn(
            "--archive release-dist/*.whl --archive release-dist/*.tar.gz",
            self.release_workflow,
        )
        self.assertIn(
            "--collect-dir downloaded-artifacts --dist-dir dist", self.release_workflow
        )
        self.assertIn("name: release-distributions-source", self.release_workflow)
        self.assertIn("name: release-distributions", self.release_workflow)
        self.assertEqual(self.release_workflow.count("if-no-files-found: error"), 3)
        publish_job = self.release_workflow.split("  publish-pypi:", 1)[1].split(
            "  verify-pypi:", 1
        )[0]
        self.assertIn("id-token: write", publish_job)
        self.assertIn("name: release-distributions", publish_job)
        self.assertNotIn("uv build", publish_job)
        self.assertNotIn("skip-existing", self.release_workflow)
        self.assertIn("needs: [metadata, assemble, verify-pypi]", self.release_workflow)
        self.assertIn("gh release create", self.release_workflow)
        self.assertIn("--verify-tag", self.release_workflow)

    def test_release_cpu_evidence_installs_each_archive_on_linux_and_macos(self):
        self.assertIn(
            "name: Verify installed release artifacts / ${{ matrix.os }}",
            self.release_workflow,
        )
        self.assertIn("os: [ubuntu-24.04, macos-15]", self.release_workflow)
        self.assertIn("needs: build-distributions", self.release_workflow)
        self.assertIn(
            "needs: [metadata, build-distributions, verify-cpu-artifacts]",
            self.release_workflow,
        )
        self.assertIn("uv venv --clear --python 3.14", self.release_workflow)
        self.assertIn("benchmarks/package_cutover.py", self.release_workflow)
        self.assertIn("--device cpu", self.release_workflow)
        self.assertIn("--required-device-count 0", self.release_workflow)
        self.assertIn(
            "name: issue-124-release-cpu-${{ matrix.os }}", self.release_workflow
        )

    def test_macos_candidate_evidence_uses_installed_issue124_packages(self):
        checkout = (
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 " "# v7.0.1"
        )
        upload = (
            "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a "
            "# v7.0.1"
        )
        self.assertEqual(
            self.ci_workflow.count("name: Python 3.14 / ${{ matrix.os }}"), 1
        )
        self.assertEqual(self.ci_workflow.count(checkout), 2)
        self.assertEqual(self.ci_workflow.count("persist-credentials: false"), 2)
        self.assertNotIn("actions/checkout@v7", self.ci_workflow)
        self.assertIn(
            "ref: ${{ github.event.pull_request.head.sha || github.sha }}",
            self.ci_workflow,
        )
        self.assertIn("path: issue124-package-evidence-candidate", self.ci_workflow)
        self.assertEqual(
            self.ci_workflow.count(
                'test -z "$(git -C "$CANDIDATE_DIR" status --porcelain=v1 --untracked-files=all)"'
            ),
            2,
        )
        self.assertIn("persist-credentials: false", self.ci_workflow)
        evidence_condition = (
            "if: runner.os == 'macOS' && github.event_name == 'pull_request'"
        )
        self.assertEqual(self.ci_workflow.count(evidence_condition), 3)
        self.assertIn('test "$(uname -m)" = arm64', self.ci_workflow)
        self.assertIn("--clear --no-create-gitignore", self.ci_workflow)
        self.assertIn("*-py3-none-any.whl", self.ci_workflow)
        self.assertIn(
            "UV_CACHE_DIR: ${{ runner.temp }}/issue-124-package-cache", self.ci_workflow
        )
        self.assertIn("uv venv --clear --python 3.14", self.ci_workflow)
        self.assertIn("benchmarks/package_cutover.py", self.ci_workflow)
        self.assertIn('--forbidden-root "$GITHUB_WORKSPACE"', self.ci_workflow)
        self.assertIn('--forbidden-root "$CANDIDATE_DIR"', self.ci_workflow)
        self.assertIn("--device cpu", self.ci_workflow)
        self.assertIn("--required-device-count 0", self.ci_workflow)
        self.assertNotIn("macos_ci_evidence.py", self.ci_workflow)
        self.assertNotIn("native_oracle_workloads.json", self.ci_workflow)
        self.assertNotIn("--required-device-count 2", self.ci_workflow)
        self.assertIn(upload, self.ci_workflow)
        self.assertIn(
            "name: issue-124-package-${{ github.event.pull_request.head.sha || github.sha }}",
            self.ci_workflow,
        )
        self.assertIn(
            "path: ${{ runner.temp }}/issue-124-package-evidence", self.ci_workflow
        )
        self.assertIn("if-no-files-found: error", self.ci_workflow)
        self.assertIn("retention-days: 90", self.ci_workflow)
        self.assertIn("overwrite: true", self.ci_workflow)

    def test_cpu_artifact_installers_preserve_pip_hash_and_sdist_backend_provenance(
        self,
    ):
        for workflow, build_constraints in (
            (self.ci_workflow, "$CANDIDATE_DIR/build-constraints.txt"),
            (self.release_workflow, "$GITHUB_WORKSPACE/build-constraints.txt"),
        ):
            with self.subTest(build_constraints=build_constraints):
                self.assertIn("uv sync --locked --no-install-project", workflow)
                self.assertIn("--extra torch-cpu --extra hdf5", workflow)
                self.assertIn("-m ensurepip", workflow)
                self.assertIn("-m pip --version", workflow)
                self.assertNotIn("pip 26.2.1", workflow)
                self.assertIn(f'--constraint "{build_constraints}"', workflow)
                self.assertIn("setuptools==84.0.0 wheel==0.48.0", workflow)
                self.assertIn(
                    "installer=(--no-deps --no-index --force-reinstall)", workflow
                )
                self.assertIn("installer+=(--no-build-isolation)", workflow)
                self.assertIn('"gmes @ file://${archive}#sha256=${digest}"', workflow)
        candidate_install = self.ci_workflow.split(
            'helper="$CANDIDATE_DIR/benchmarks/package_cutover.py"', 1
        )[1]
        release_install = self.release_workflow.split("  verify-pypi:", 1)[0]
        self.assertNotIn("uv pip install --python", candidate_install)
        self.assertNotIn("uv pip install --python", release_install)
        self.assertIn(
            'cd "$CANDIDATE_DIR"\n'
            '              UV_PROJECT_ENVIRONMENT="$environment" uv sync',
            candidate_install,
        )
        self.assertIn(
            'cd "$RUNNER_TEMP"\n              "$environment/bin/python" -I "$helper"',
            candidate_install,
        )
        self.assertIn(
            'cd "${GITHUB_WORKSPACE}"\n'
            '              UV_PROJECT_ENVIRONMENT="${environment}" uv sync',
            release_install,
        )
        self.assertIn(
            'cd "$GITHUB_WORKSPACE"\n'
            '              UV_PROJECT_ENVIRONMENT="$environment" uv sync',
            release_install,
        )
        self.assertIn(
            'cd "$RUNNER_TEMP"\n'
            '              "$environment/bin/python" -I "$GITHUB_WORKSPACE/benchmarks/package_cutover.py"',
            release_install,
        )


if __name__ == "__main__":
    unittest.main()
