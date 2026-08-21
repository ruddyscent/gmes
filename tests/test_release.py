"""Regression tests for the release identity and supported wheel matrix."""

import re
import tempfile
import tomllib
import unittest
from pathlib import Path

from utils.release import (
    EXPECTED_WHEEL_PLATFORMS,
    _wheel_platform,
    collect_distributions,
    verify_distribution_set,
    verify_release_identity,
)


class ReleaseConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.workflow = (
            cls.project_root / ".github" / "workflows" / "release.yml"
        ).read_text()
        cls.configuration = tomllib.loads(
            (cls.project_root / "pyproject.toml").read_text()
        )

    def test_release_identity_uses_version_tag_and_dated_changelog(self):
        version, title = verify_release_identity(self.project_root, "v0.10.0")

        self.assertEqual(version, "0.10.0")
        self.assertEqual(title, "GMES 0.10.0")

    def test_release_identity_rejects_a_mismatched_tag(self):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            verify_release_identity(self.project_root, "v0.10.1")

    def test_wheel_matrix_is_python_314_linux_and_macos_arm64(self):
        filenames = {
            "gmes-0.10.0-cp314-cp314-manylinux_2_34_x86_64.whl",
            "gmes-0.10.0-cp314-cp314-macosx_11_0_arm64.whl",
        }

        platforms = {_wheel_platform(name, "0.10.0") for name in filenames}

        self.assertEqual(platforms, EXPECTED_WHEEL_PLATFORMS)

    def test_cibuildwheel_matches_the_documented_platforms(self):
        cibuildwheel = self.configuration["tool"]["cibuildwheel"]

        self.assertEqual(
            cibuildwheel["build"],
            "cp314-manylinux_x86_64 cp314-macosx_arm64",
        )
        self.assertEqual(cibuildwheel["linux"]["archs"], ["x86_64"])
        self.assertEqual(cibuildwheel["macos"]["archs"], ["arm64"])
        self.assertEqual(
            cibuildwheel["macos"]["environment"]["MACOSX_DEPLOYMENT_TARGET"],
            "11.0",
        )

    def test_release_actions_are_immutable_and_publish_with_oidc(self):
        action_references = re.findall(
            r"^\s*- uses: [^@\s]+@([^\s]+)", self.workflow, re.MULTILINE
        )

        self.assertTrue(action_references)
        self.assertTrue(
            all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_references)
        )
        self.assertEqual(self.workflow.count("id-token: write"), 1)
        self.assertIn("environment:\n      name: pypi", self.workflow)
        self.assertNotIn("skip-existing", self.workflow)
        self.assertIn("gh release create", self.workflow)
        self.assertIn("pull_request:\n    branches:\n      - master", self.workflow)
        self.assertIn("--from cibuildwheel==4.2.0", self.workflow)
        self.assertIn("--with uv==0.12.5", self.workflow)

    def test_publish_job_has_no_checkout_or_build_step(self):
        publish_job = self.workflow.split("  publish-pypi:", 1)[1].split(
            "  verify-pypi:", 1
        )[0]

        self.assertNotIn("actions/checkout", publish_job)
        self.assertNotIn("run:", publish_job)

    def test_distribution_set_rejects_local_or_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "developer-build.whl").touch()

            with self.assertRaisesRegex(RuntimeError, "exactly one sdist"):
                verify_distribution_set(directory, "0.10.0")

    def test_collection_rejects_duplicate_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            (artifacts / "linux").mkdir(parents=True)
            (artifacts / "macos").mkdir()
            filename = "gmes-0.10.0-cp314-cp314-manylinux_2_34_x86_64.whl"
            (artifacts / "linux" / filename).touch()
            (artifacts / "macos" / filename).touch()

            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                collect_distributions(artifacts, Path(directory) / "dist", "0.10.0")


if __name__ == "__main__":
    unittest.main()
