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
            "python -m mypy.stubtest gmes.constant gmes.pw_material",
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


if __name__ == "__main__":
    unittest.main()
