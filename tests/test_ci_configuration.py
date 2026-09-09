"""Regression tests for the pure-package GitHub Actions policy."""

from pathlib import Path

import pytest


@pytest.fixture(scope="class", autouse=True)
def workflow_files(request):
    workflow_directory = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    request.cls.ci_workflow = (workflow_directory / "ci.yml").read_text()
    request.cls.codeql_workflow = (workflow_directory / "codeql.yml").read_text()
    request.cls.prerelease_workflow = (
        workflow_directory / "prerelease.yml"
    ).read_text()
    request.cls.release_workflow = (workflow_directory / "release.yml").read_text()


class TestCiConfiguration:

    def test_required_ci_is_limited_to_master_and_cancels_stale_runs(self):
        assert (self.ci_workflow.count("      - master")) == (2)
        assert ("group: ci-${{ github.workflow }}-${{ github.ref }}") in (
            self.ci_workflow
        )
        assert ("cancel-in-progress: true") in (self.ci_workflow)
        assert ("name: Python 3.14 / ${{ matrix.os }}") in (self.ci_workflow)
        assert ("          - ubuntu-latest") in (self.ci_workflow)
        assert ("          - macos-latest") in (self.ci_workflow)
        assert ("      fail-fast: false") in (self.ci_workflow)
        assert ("permissions:\n  contents: read") in (self.ci_workflow)
        assert ("prerelease:") not in (self.ci_workflow)

    def test_required_ci_uses_locked_cpu_torch_and_pure_build(self):
        primary_steps = self.ci_workflow.split(
            "      - name: Check out candidate head", 1
        )[0]
        assert ("uv sync --locked --extra torch-cpu --extra hdf5") in (primary_steps)
        assert ("uv run --no-sync python -m pytest -v") in (primary_steps)
        assert ("uv build") in (primary_steps)
        assert ("brew install") not in (primary_steps)
        assert ("GMES_ENABLE_OPENMP") not in (primary_steps)
        assert ("swig") not in (primary_steps)
        assert ("mpi") not in (primary_steps.lower())

    def test_required_ci_runs_static_and_supported_stub_checks(self):
        assert ("python -m mypy") in (self.ci_workflow)
        assert (
            "python -m mypy.stubtest --mypy-config-file pyproject.toml gmes.constant"
        ) in (self.ci_workflow)
        assert ("gmes.pw_material") not in (self.ci_workflow)

    def test_required_ci_lints_only_tracked_python_sources(self):
        assert ("python -m pylint $(git ls-files 'gmes/*.py') setup.py") in (
            self.ci_workflow
        )

    def test_macos_required_job_uses_the_same_locked_cpu_environment(self):
        assert ("          - macos-latest") in (self.ci_workflow)
        assert ("uv sync --locked --extra torch-cpu --extra hdf5") in (self.ci_workflow)

    def test_prerelease_is_scheduled_advisory_using_nightly_wheels(self):
        assert ("name: Python prerelease (advisory)") in (self.prerelease_workflow)
        assert ("schedule:") in (self.prerelease_workflow)
        assert ("workflow_dispatch:") in (self.prerelease_workflow)
        assert ("--only-binary=:all:") in (self.prerelease_workflow)
        assert ("scientific-python-nightly-wheels/simple") in (self.prerelease_workflow)
        assert ("--no-deps") not in (self.prerelease_workflow)
        assert ("-e .") in (self.prerelease_workflow)
        assert ("--extra-index-url https://download.pytorch.org/whl/cpu") in (
            self.prerelease_workflow
        )
        assert ("python -m pip check") in (self.prerelease_workflow)
        assert ("        run: python -m pytest -v") in (self.prerelease_workflow)
        assert ("uv run --no-sync python -m pytest -v") not in (
            self.prerelease_workflow
        )
        assert ('python-version: "3.15-dev"') in (self.prerelease_workflow)
        assert ("uv run --no-sync python -m pytest -v") in (self.ci_workflow)
        assert ("uv run --no-sync python -m pytest -v") in (self.release_workflow)

    def test_prerelease_installs_uv_for_packaging_tests(self):
        assert ("uses: astral-sh/setup-uv@") in (self.prerelease_workflow)
        assert ('version: "0.12.5"') in (self.prerelease_workflow)

    @pytest.mark.parametrize(
        "workflow_name",
        ("ci_workflow", "codeql_workflow", "prerelease_workflow", "release_workflow"),
    )
    def test_workflow_checkouts_disable_persisted_credentials(self, workflow_name):
        checkout = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
        workflow = getattr(self, workflow_name)
        assert workflow.count(checkout) == workflow.count("persist-credentials: false")

    def test_codeql_analyzes_python_without_a_native_build(self):
        assert ("name: CodeQL / python") in (self.codeql_workflow)
        assert ("matrix.") not in (self.codeql_workflow)
        assert ("security-events: write") in (self.codeql_workflow)
        assert ("languages: python") in (self.codeql_workflow)
        assert ("build-mode: none") in (self.codeql_workflow)
        assert ("languages: c-cpp") not in (self.codeql_workflow)
        assert ("swig") not in (self.codeql_workflow)

    def test_release_builds_and_reuses_exact_universal_archives(self):
        assert ("uv build --clear --out-dir release-dist") in (self.release_workflow)
        assert ("gmes-*-py3-none-any.whl") in (self.release_workflow)
        assert ("--archive release-dist/*.whl --archive release-dist/*.tar.gz") in (
            self.release_workflow
        )
        assert ("--collect-dir downloaded-artifacts --dist-dir dist") in (
            self.release_workflow
        )
        assert ("name: release-distributions-source") in (self.release_workflow)
        assert ("name: release-distributions") in (self.release_workflow)
        assert (self.release_workflow.count("if-no-files-found: error")) == (3)
        publish_job = self.release_workflow.split("  publish-pypi:", 1)[1].split(
            "  verify-pypi:", 1
        )[0]
        assert ("id-token: write") in (publish_job)
        assert ("name: release-distributions") in (publish_job)
        assert ("uv build") not in (publish_job)
        assert ("skip-existing") not in (self.release_workflow)
        assert ("needs: [metadata, assemble, verify-pypi]") in (self.release_workflow)
        assert ("gh release create") in (self.release_workflow)
        assert ("--verify-tag") in (self.release_workflow)

    def test_release_cpu_evidence_installs_each_archive_on_linux_and_macos(self):
        assert ("name: Verify installed release artifacts / ${{ matrix.os }}") in (
            self.release_workflow
        )
        assert ("os: [ubuntu-24.04, macos-15]") in (self.release_workflow)
        assert ("needs: build-distributions") in (self.release_workflow)
        assert ("needs: [metadata, build-distributions, verify-cpu-artifacts]") in (
            self.release_workflow
        )
        assert ("uv venv --clear --python 3.14") in (self.release_workflow)
        assert ("benchmarks/package_cutover.py") in (self.release_workflow)
        assert ("--device cpu") in (self.release_workflow)
        assert ("--required-device-count 0") in (self.release_workflow)
        assert ("name: issue-124-release-cpu-${{ matrix.os }}") in (
            self.release_workflow
        )

    def test_macos_candidate_evidence_uses_installed_issue124_packages(self):
        checkout = (
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 " "# v7.0.1"
        )
        upload = (
            "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a "
            "# v7.0.1"
        )
        assert (self.ci_workflow.count("name: Python 3.14 / ${{ matrix.os }}")) == (1)
        assert (self.ci_workflow.count(checkout)) == (2)
        assert (self.ci_workflow.count("persist-credentials: false")) == (2)
        assert ("actions/checkout@v7") not in (self.ci_workflow)
        assert ("ref: ${{ github.event.pull_request.head.sha || github.sha }}") in (
            self.ci_workflow
        )
        assert ("path: issue124-package-evidence-candidate") in (self.ci_workflow)
        assert (
            self.ci_workflow.count(
                'test -z "$(git -C "$CANDIDATE_DIR" status --porcelain=v1 --untracked-files=all)"'
            )
        ) == (2)
        assert ("persist-credentials: false") in (self.ci_workflow)
        evidence_condition = (
            "if: runner.os == 'macOS' && github.event_name == 'pull_request'"
        )
        assert (self.ci_workflow.count(evidence_condition)) == (3)
        assert ('test "$(uname -m)" = arm64') in (self.ci_workflow)
        assert ("--clear --no-create-gitignore") in (self.ci_workflow)
        assert ("*-py3-none-any.whl") in (self.ci_workflow)
        assert ("UV_CACHE_DIR: ${{ runner.temp }}/issue-124-package-cache") in (
            self.ci_workflow
        )
        assert ("uv venv --clear --python 3.14") in (self.ci_workflow)
        assert ("benchmarks/package_cutover.py") in (self.ci_workflow)
        assert ('--forbidden-root "$GITHUB_WORKSPACE"') in (self.ci_workflow)
        assert ('--forbidden-root "$CANDIDATE_DIR"') in (self.ci_workflow)
        assert ("--device cpu") in (self.ci_workflow)
        assert ("--required-device-count 0") in (self.ci_workflow)
        assert ("macos_ci_evidence.py") not in (self.ci_workflow)
        assert ("native_oracle_workloads.json") not in (self.ci_workflow)
        assert ("--required-device-count 2") not in (self.ci_workflow)
        assert (upload) in (self.ci_workflow)
        assert (
            "name: issue-124-package-${{ github.event.pull_request.head.sha || github.sha }}"
        ) in (self.ci_workflow)
        assert ("path: ${{ runner.temp }}/issue-124-package-evidence") in (
            self.ci_workflow
        )
        assert ("if-no-files-found: error") in (self.ci_workflow)
        assert ("retention-days: 90") in (self.ci_workflow)
        assert ("overwrite: true") in (self.ci_workflow)

    @pytest.mark.parametrize(
        "workflow_build_constraints_case", range(2), ids=("ci", "release")
    )
    def test_cpu_artifact_installers_preserve_pip_hash_and_sdist_backend_provenance(
        self,
        workflow_build_constraints_case,
    ):
        workflow_build_constraints_case_values = tuple(
            (
                (self.ci_workflow, "$CANDIDATE_DIR/build-constraints.txt"),
                (self.release_workflow, "$GITHUB_WORKSPACE/build-constraints.txt"),
            )
        )
        assert len(workflow_build_constraints_case_values) == 2
        workflow, build_constraints = workflow_build_constraints_case_values[
            workflow_build_constraints_case
        ]
        assert ("uv sync --locked --no-install-project") in (workflow)
        assert ("--extra torch-cpu --extra hdf5") in (workflow)
        assert ("-m ensurepip") in (workflow)
        assert ("-m pip --version") in (workflow)
        assert ("pip 26.2.1") not in (workflow)
        assert (f'--constraint "{build_constraints}"') in (workflow)
        assert ("setuptools==84.0.0 wheel==0.48.0") in (workflow)
        assert ("installer=(--no-deps --no-index --force-reinstall)") in (workflow)
        assert ("installer+=(--no-build-isolation)") in (workflow)
        assert ('"gmes @ file://${archive}#sha256=${digest}"') in (workflow)
        candidate_install = self.ci_workflow.split(
            'helper="$CANDIDATE_DIR/benchmarks/package_cutover.py"', 1
        )[1]
        release_install = self.release_workflow.split("  verify-pypi:", 1)[0]
        assert ("uv pip install --python") not in (candidate_install)
        assert ("uv pip install --python") not in (release_install)
        assert (
            'cd "$CANDIDATE_DIR"\n'
            '              UV_PROJECT_ENVIRONMENT="$environment" uv sync'
        ) in (candidate_install)
        assert (
            'cd "$RUNNER_TEMP"\n              "$environment/bin/python" -I "$helper"'
        ) in (candidate_install)
        assert (
            'cd "${GITHUB_WORKSPACE}"\n'
            '              UV_PROJECT_ENVIRONMENT="${environment}" uv sync'
        ) in (release_install)
        assert (
            'cd "$GITHUB_WORKSPACE"\n'
            '              UV_PROJECT_ENVIRONMENT="$environment" uv sync'
        ) in (release_install)
        assert (
            'cd "$RUNNER_TEMP"\n'
            '              "$environment/bin/python" -I "$GITHUB_WORKSPACE/benchmarks/package_cutover.py"'
        ) in (release_install)
