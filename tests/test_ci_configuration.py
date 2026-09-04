"""Regression tests for GitHub Actions trigger and prerelease policy."""

import unittest
from pathlib import Path


class CiConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow_directory = (
            Path(__file__).resolve().parents[1] / ".github" / "workflows"
        )
        cls.ci_workflow = (workflow_directory / "ci.yml").read_text()
        cls.prerelease_workflow = (workflow_directory / "prerelease.yml").read_text()

    def test_required_ci_is_limited_to_master_and_cancels_stale_runs(self):
        self.assertEqual(self.ci_workflow.count("      - master"), 2)
        self.assertIn(
            "group: ci-${{ github.workflow }}-${{ github.ref }}", self.ci_workflow
        )
        self.assertIn("cancel-in-progress: true", self.ci_workflow)
        self.assertIn("name: Python 3.14 / ${{ matrix.os }}", self.ci_workflow)
        self.assertNotIn("prerelease:", self.ci_workflow)

    def test_required_ci_exercises_all_openmp_paths(self):
        self.assertIn("brew install swig libomp", self.ci_workflow)
        self.assertIn("assert pw_material.openmp_enabled()", self.ci_workflow)
        self.assertIn("assert not pw_material.openmp_enabled()", self.ci_workflow)
        self.assertIn("GMES_OPENMP_THRESHOLD=0", self.ci_workflow)
        self.assertIn("GMES_ENABLE_OPENMP=0", self.ci_workflow)
        self.assertIn("GMES_ENABLE_OPENMP=1 uv build", self.ci_workflow)
        self.assertIn("GMES_ENABLE_OPENMP=auto uv build", self.ci_workflow)

    def test_required_ci_runs_static_and_native_stub_checks(self):
        self.assertIn("python -m mypy", self.ci_workflow)
        self.assertIn(
            "python -m mypy.stubtest --mypy-config-file pyproject.toml gmes.constant gmes.pw_material",
            self.ci_workflow,
        )

    def test_required_ci_lints_only_tracked_python_sources(self):
        self.assertIn(
            "python -m pylint $(git ls-files 'gmes/*.py') setup.py",
            self.ci_workflow,
        )

    def test_macos_default_suite_runs_without_plot_extra(self):
        self.assertIn(
            """- name: Install macOS test dependencies without plot extra
        if: runner.os == 'macOS'
        run: uv sync --locked --extra torch-cpu --extra hdf5""",
            self.ci_workflow,
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

    def test_macos_required_job_uploads_candidate_bound_runtime_evidence(self):
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
        self.assertNotIn("actions/checkout@v7", self.ci_workflow)
        self.assertIn(
            "ref: ${{ github.event.pull_request.head.sha || github.sha }}",
            self.ci_workflow,
        )
        self.assertIn("persist-credentials: false", self.ci_workflow)
        evidence_condition = (
            "if: runner.os == 'macOS' && github.event_name == 'pull_request'"
        )
        self.assertEqual(self.ci_workflow.count(evidence_condition), 3)
        self.assertIn('test "$(uname -m)" = arm64', self.ci_workflow)
        self.assertIn("--clear --no-create-gitignore", self.ci_workflow)
        for role in (
            "wheel-import",
            "wheel-default-suite",
            "wheel-serial-suite",
            "sdist-import",
            "sdist-default-suite",
            "sdist-serial-suite",
        ):
            self.assertIn(f"record_probe {role}", self.ci_workflow)
        self.assertIn('probe_args=(record --role "$1")', self.ci_workflow)
        self.assertIn('probe_args+=(--mode "$2")', self.ci_workflow)
        self.assertNotIn("mode_args=(", self.ci_workflow)
        self.assertIn("local probe_status stderr_path", self.ci_workflow)
        self.assertIn('tail -c 65536 "$stderr_path" >&2', self.ci_workflow)
        self.assertIn('return "$probe_status"', self.ci_workflow)
        self.assertIn('"$python" -I "$helper" capture', self.ci_workflow)
        self.assertIn(upload, self.ci_workflow)
        self.assertIn(
            "name: issue-123-macos-${{ github.event.pull_request.head.sha || github.sha }}",
            self.ci_workflow,
        )
        self.assertIn(
            "path: ${{ runner.temp }}/issue-123-macos-evidence", self.ci_workflow
        )
        self.assertIn("if-no-files-found: error", self.ci_workflow)
        self.assertIn("retention-days: 90", self.ci_workflow)
        self.assertIn("overwrite: true", self.ci_workflow)


if __name__ == "__main__":
    unittest.main()
