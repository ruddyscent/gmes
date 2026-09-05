"""Regression tests for the release identity and supported wheel matrix."""

import base64
import hashlib
import io
import re
import stat
import tarfile
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

from utils.release import (
    REQUIRED_SDIST_PATHS,
    REQUIRED_WHEEL_MODULES,
    RETIRED_GMES_PATH_PREFIXES,
    UNIVERSAL_WHEEL_TAG,
    _wheel_platform,
    collect_distributions,
    verify_distribution_set,
    verify_release_identity,
)


def _write_sdist(path, version, paths=REQUIRED_SDIST_PATHS):
    root = f"gmes-{version}"
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(paths | {"examples/example.py", "tests/test_example.py"}):
            content = (
                b"" if name != "PKG-INFO" else b"Name: gmes\nVersion: 0.10.0\n# GMES\n"
            )
            member = tarfile.TarInfo(f"{root}/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        metadata = b"Name: gmes\nVersion: 0.10.0\n# GMES\n"
        member = tarfile.TarInfo(f"{root}/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))


def _write_wheel(
    path,
    version,
    *,
    purelib=True,
    tag=UNIVERSAL_WHEEL_TAG,
    members=REQUIRED_WHEEL_MODULES,
    include_record=True,
    record_digest_override=None,
    record_size_offset=0,
):
    dist_info = f"gmes-{version}.dist-info"
    entries = {name: b"" for name in members}
    entries[f"{dist_info}/METADATA"] = (
        f"Name: gmes\nVersion: {version}\n# GMES\n".encode()
    )
    entries[f"{dist_info}/WHEEL"] = (
        f"Wheel-Version: 1.0\nRoot-Is-Purelib: {'true' if purelib else 'false'}\nTag: {tag}\n".encode()
    )
    entries[f"{dist_info}/licenses/LICENSE"] = b"GPL-3.0-or-later"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in sorted(entries.items()):
            archive.writestr(name, content)
        if include_record:
            rows = []
            for name, content in sorted(entries.items()):
                digest = (
                    base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                    .rstrip(b"=")
                    .decode()
                )
                if not rows and record_digest_override is not None:
                    digest = record_digest_override
                size = len(content) + (record_size_offset if not rows else 0)
                rows.append(f"{name},sha256={digest},{size}")
            rows.append(f"{dist_info}/RECORD,,")
            archive.writestr(f"{dist_info}/RECORD", "\n".join(rows) + "\n")


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

    def test_wheel_is_one_universal_python_distribution(self):
        self.assertEqual(
            _wheel_platform("gmes-0.10.0-py3-none-any.whl", "0.10.0"),
            UNIVERSAL_WHEEL_TAG,
        )

    def test_pure_archives_require_canonical_and_torch_modules(self):
        expected = {
            "gmes/constant.py",
            "gmes/constant.pyi",
            "gmes/geometry.py",
            "gmes/py.typed",
            "gmes/torch_dispersive.py",
            "gmes/torch_dm2.py",
        }
        self.assertTrue(expected <= REQUIRED_SDIST_PATHS)
        self.assertTrue(expected <= REQUIRED_WHEEL_MODULES)
        self.assertTrue(
            {"gmes/torch_fdtd.py", "gmes/torch_plan.py", "gmes/torch_source.py"}
            <= REQUIRED_WHEEL_MODULES
        )
        self.assertIn("src/", RETIRED_GMES_PATH_PREFIXES)
        self.assertIn("gmes/_pw_material.", RETIRED_GMES_PATH_PREFIXES)
        self.assertNotIn("cibuildwheel", self.configuration["tool"])

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

    def test_publish_job_has_no_checkout_or_build_step(self):
        publish_job = self.workflow.split("  publish-pypi:", 1)[1].split(
            "  verify-pypi:", 1
        )[0]

        self.assertNotIn("actions/checkout", publish_job)
        self.assertNotIn("run:", publish_job)

    def test_full_quality_suite_skips_pull_requests(self):
        quality_job = self.workflow.split("  quality:", 1)[1].split(
            "  build-sdist:", 1
        )[0]

        self.assertIn("if: github.event_name != 'pull_request'", quality_job)

    def test_github_release_job_has_explicit_repository_context(self):
        github_release_job = self.workflow.split("  github-release:", 1)[1]

        self.assertNotIn("actions/checkout", github_release_job)
        self.assertIn("GH_REPO: ${{ github.repository }}", github_release_job)
        self.assertIn("gh release create", github_release_job)
        self.assertIn("--verify-tag", github_release_job)

    def test_release_workflow_avoids_known_runner_warnings(self):
        verify_pypi_job = self.workflow.split("  verify-pypi:", 1)[1].split(
            "  github-release:", 1
        )[0]

        self.assertEqual(self.workflow.count("cache-suffix: ${{ github.job }}"), 3)
        self.assertIn("ignore-empty-workdir: true", verify_pypi_job)
        self.assertNotIn("brew ", self.workflow)
        self.assertNotIn("swig", self.workflow)
        self.assertNotIn("HOMEBREW_NO_REQUIRE_TAP_TRUST", self.workflow)

    def test_distribution_set_rejects_local_or_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "developer-build.whl").touch()

            with self.assertRaisesRegex(RuntimeError, "exactly one sdist"):
                verify_distribution_set(directory, "0.10.0")

    def test_distribution_set_accepts_one_pure_wheel_and_sdist(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _write_sdist(directory / "gmes-0.10.0.tar.gz", "0.10.0")
            _write_wheel(directory / "gmes-0.10.0-py3-none-any.whl", "0.10.0")
            verify_distribution_set(directory, "0.10.0")

    def test_wheel_rejects_non_pure_retired_or_native_members(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            archive = directory / "gmes-0.10.0-py3-none-any.whl"
            _write_sdist(directory / "gmes-0.10.0.tar.gz", "0.10.0")
            _write_wheel(archive, "0.10.0", purelib=False)
            with self.assertRaisesRegex(RuntimeError, "Root-Is-Purelib"):
                verify_distribution_set(directory, "0.10.0")
            _write_wheel(
                archive,
                "0.10.0",
                members=REQUIRED_WHEEL_MODULES | {"gmes/_pw_material.so"},
            )
            with self.assertRaisesRegex(RuntimeError, "retired native"):
                verify_distribution_set(directory, "0.10.0")
            _write_wheel(
                archive,
                "0.10.0",
                members=REQUIRED_WHEEL_MODULES | {"gmes/unreviewed_native.so"},
            )
            with self.assertRaisesRegex(RuntimeError, "native members"):
                verify_distribution_set(directory, "0.10.0")

    def test_wheel_rejects_missing_internal_tag_record_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            archive = directory / "gmes-0.10.0-py3-none-any.whl"
            _write_sdist(directory / "gmes-0.10.0.tar.gz", "0.10.0")
            _write_wheel(archive, "0.10.0", tag="cp314-cp314-any")
            with self.assertRaisesRegex(RuntimeError, "universal"):
                verify_distribution_set(directory, "0.10.0")
            _write_wheel(archive, "0.10.0", include_record=False)
            with self.assertRaisesRegex(RuntimeError, "dist-info metadata"):
                verify_distribution_set(directory, "0.10.0")
            _write_wheel(archive, "0.10.0", record_size_offset=1)
            with self.assertRaisesRegex(RuntimeError, "RECORD digest or size"):
                verify_distribution_set(directory, "0.10.0")
            _write_wheel(archive, "0.10.0", record_digest_override="forged")
            with self.assertRaisesRegex(RuntimeError, "RECORD digest or size"):
                verify_distribution_set(directory, "0.10.0")
            _write_wheel(archive, "0.10.0")
            with zipfile.ZipFile(archive, "a") as wheel:
                link = zipfile.ZipInfo("gmes/linked")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                wheel.writestr(link, "../outside")
            with self.assertRaisesRegex(RuntimeError, "non-regular"):
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
