"""Validate release identity and built distribution contents."""

import argparse
import re
import shutil
import tarfile
import tomllib
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

PROJECT_NAME = "gmes"
EXPECTED_WHEEL_PLATFORMS = {
    "manylinux_2_34_x86_64",
    "macosx_11_0_arm64",
}
REQUIRED_SDIST_PATHS = {
    "CHANGELOG.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "VERSION",
    "build-constraints.txt",
    "gmes/__init__.py",
    "pyproject.toml",
    "setup.py",
    "src/constant.cc",
    "src/constant.hh",
    "src/constant.i",
    "src/cpp23_support.hh",
    "src/material.pyx",
    "src/numpy.i",
    "src/pw_material.i",
    "src/pygeom.pyx",
}
REQUIRED_WHEEL_MODULES = {
    "gmes/__init__.py",
    "gmes/constant.py",
    "gmes/pw_material.py",
}
NATIVE_MODULE_PREFIXES = (
    "gmes/_constant.",
    "gmes/_pw_material.",
    "gmes/material.",
    "gmes/pygeom.",
)
FORBIDDEN_DIRECTORY_NAMES = {".git", "__pycache__", "build", "dist"}
FORBIDDEN_ARCHIVE_SUFFIXES = {".h5", ".hdf5", ".pyc", ".pyo"}


def read_version(project_root):
    """Return and validate the repository version."""
    version = (Path(project_root) / "VERSION").read_text().strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError(f"VERSION must contain a release version, got {version!r}")
    return version


def verify_release_identity(project_root, tag):
    """Ensure tag, package metadata, changelog, and release title agree."""
    project_root = Path(project_root)
    version = read_version(project_root)
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise RuntimeError(f"release tag {tag!r} does not match {expected_tag!r}")

    configuration = tomllib.loads((project_root / "pyproject.toml").read_text())
    project = configuration["project"]
    if project["name"] != PROJECT_NAME:
        raise RuntimeError(f"project name must remain {PROJECT_NAME!r}")
    if project.get("dynamic") != ["version"]:
        raise RuntimeError("project version must be read dynamically from VERSION")
    version_file = configuration["tool"]["setuptools"]["dynamic"]["version"]["file"]
    if version_file != "VERSION":
        raise RuntimeError("setuptools package metadata must read VERSION")

    changelog = (project_root / "CHANGELOG.md").read_text()
    heading = re.search(
        rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$",
        changelog,
        re.MULTILINE,
    )
    if heading is None:
        raise RuntimeError(
            f"CHANGELOG.md must contain a dated [{version}] release heading"
        )
    try:
        date.fromisoformat(heading.group(1))
    except ValueError as error:
        raise RuntimeError("CHANGELOG.md release date is not valid ISO 8601") from error
    if re.search(
        rf"^## \[{re.escape(version)}\] - Unreleased$", changelog, re.MULTILINE
    ):
        raise RuntimeError(f"CHANGELOG.md still marks {version} as Unreleased")

    return version, f"GMES {version}"


def _normalized_sdist_paths(archive, version):
    expected_root = f"{PROJECT_NAME}-{version}"
    paths = set()
    with tarfile.open(archive, "r:gz") as distribution:
        for member in distribution.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe sdist path: {member.name}")
            if not path.parts or path.parts[0] != expected_root:
                raise RuntimeError(
                    f"sdist member {member.name!r} is outside {expected_root!r}"
                )
            relative = PurePosixPath(*path.parts[1:])
            if relative.parts:
                paths.add(relative.as_posix())
    return paths


def _reject_generated_products(paths, archive_kind):
    for name in paths:
        path = PurePosixPath(name)
        if FORBIDDEN_DIRECTORY_NAMES.intersection(path.parts):
            raise RuntimeError(f"{archive_kind} contains generated directory {name!r}")
        if path.suffix.lower() in FORBIDDEN_ARCHIVE_SUFFIXES:
            raise RuntimeError(f"{archive_kind} contains generated file {name!r}")


def verify_sdist(archive, version):
    """Check that an sdist contains source inputs but no local build output."""
    archive = Path(archive)
    expected_name = f"{PROJECT_NAME}-{version}.tar.gz"
    if archive.name != expected_name:
        raise RuntimeError(f"unexpected sdist name {archive.name!r}")

    paths = _normalized_sdist_paths(archive, version)
    missing = REQUIRED_SDIST_PATHS - paths
    if missing:
        raise RuntimeError(f"sdist is missing required paths: {sorted(missing)}")
    if not any(name.startswith("tests/test_") for name in paths):
        raise RuntimeError("sdist does not contain the unit tests")
    if not any(name.startswith("examples/") and name.endswith(".py") for name in paths):
        raise RuntimeError("sdist does not contain the examples")
    compiled_suffixes = {".dll", ".dylib", ".o", ".obj", ".so"}
    compiled = sorted(
        name for name in paths if PurePosixPath(name).suffix in compiled_suffixes
    )
    if compiled:
        raise RuntimeError(f"sdist contains compiled build products: {compiled}")
    _reject_generated_products(paths, "sdist")

    metadata_path = f"{PROJECT_NAME}-{version}/PKG-INFO"
    with tarfile.open(archive, "r:gz") as distribution:
        metadata_file = distribution.extractfile(metadata_path)
        if metadata_file is None:
            raise RuntimeError("sdist does not contain PKG-INFO")
        metadata = metadata_file.read().decode("utf-8")
    _verify_core_metadata(metadata, version, "sdist")


def _wheel_platform(filename, version):
    pattern = re.compile(
        rf"^{PROJECT_NAME}-{re.escape(version)}-cp314-cp314-(.+)\.whl$"
    )
    match = pattern.fullmatch(filename)
    if match is None:
        raise RuntimeError(f"unexpected wheel name {filename!r}")
    return match.group(1)


def _verify_core_metadata(metadata, version, archive_kind):
    if f"Name: {PROJECT_NAME}\n" not in metadata:
        raise RuntimeError(f"{archive_kind} metadata has the wrong project name")
    if f"Version: {version}\n" not in metadata:
        raise RuntimeError(f"{archive_kind} metadata has the wrong version")
    if "# GMES" not in metadata:
        raise RuntimeError(f"{archive_kind} metadata does not contain the README")


def verify_wheel(archive, version):
    """Check wheel tags, native modules, proxies, metadata, and cleanliness."""
    archive = Path(archive)
    platform = _wheel_platform(archive.name, version)
    if platform not in EXPECTED_WHEEL_PLATFORMS:
        raise RuntimeError(f"unsupported wheel platform {platform!r}")

    with zipfile.ZipFile(archive) as distribution:
        paths = set(distribution.namelist())
        for name in paths:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe wheel path: {name}")

        missing = REQUIRED_WHEEL_MODULES - paths
        if missing:
            raise RuntimeError(f"wheel is missing proxy modules: {sorted(missing)}")
        for prefix in NATIVE_MODULE_PREFIXES:
            if not any(
                name.startswith(prefix) and name.endswith(".so") for name in paths
            ):
                raise RuntimeError(f"wheel is missing native module {prefix!r}")

        metadata_names = [
            name for name in paths if name.endswith(".dist-info/METADATA")
        ]
        license_names = [
            name for name in paths if ".dist-info/licenses/LICENSE" in name
        ]
        if len(metadata_names) != 1 or len(license_names) != 1:
            raise RuntimeError(
                "wheel must contain one metadata file and the GPL license"
            )
        metadata = distribution.read(metadata_names[0]).decode("utf-8")
        _verify_core_metadata(metadata, version, "wheel")

    _reject_generated_products(paths, "wheel")
    return platform


def verify_distribution_set(dist_directory, version):
    """Require exactly one sdist and the two declared supported wheels."""
    dist_directory = Path(dist_directory)
    archives = sorted(path for path in dist_directory.iterdir() if path.is_file())
    sdist = [path for path in archives if path.name.endswith(".tar.gz")]
    wheels = [path for path in archives if path.suffix == ".whl"]
    unexpected = [path.name for path in archives if path not in sdist + wheels]
    if len(sdist) != 1 or len(wheels) != 2 or unexpected:
        raise RuntimeError(
            "release must contain exactly one sdist and two wheels; "
            f"found {[path.name for path in archives]}"
        )

    verify_sdist(sdist[0], version)
    platforms = {verify_wheel(wheel, version) for wheel in wheels}
    if platforms != EXPECTED_WHEEL_PLATFORMS:
        raise RuntimeError(
            f"wheel platforms {sorted(platforms)} do not match "
            f"{sorted(EXPECTED_WHEEL_PLATFORMS)}"
        )


def collect_distributions(artifact_directory, dist_directory, version):
    """Collect separate job artifacts without silently overwriting duplicates."""
    artifact_directory = Path(artifact_directory)
    archives = sorted(path for path in artifact_directory.rglob("*") if path.is_file())
    names = [path.name for path in archives]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate distribution filenames: {duplicates}")

    dist_directory = Path(dist_directory)
    dist_directory.mkdir(parents=True, exist_ok=False)
    for archive in archives:
        shutil.copy2(archive, dist_directory / archive.name)
    verify_distribution_set(dist_directory, version)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--archive", action="append", type=Path, default=[])
    parser.add_argument("--collect-dir", type=Path)
    parser.add_argument("--dist-dir", type=Path)
    arguments = parser.parse_args()

    version, release_title = verify_release_identity(
        arguments.project_root, arguments.tag
    )
    if arguments.collect_dir is not None:
        if arguments.dist_dir is None:
            parser.error("--collect-dir requires --dist-dir")
        collect_distributions(arguments.collect_dir, arguments.dist_dir, version)
    elif arguments.dist_dir is not None:
        verify_distribution_set(arguments.dist_dir, version)
    for archive in arguments.archive:
        if archive.suffix == ".whl":
            verify_wheel(archive, version)
        elif archive.name.endswith(".tar.gz"):
            verify_sdist(archive, version)
        else:
            raise RuntimeError(f"unsupported distribution archive {archive}")
    print(f"release identity verified: {arguments.tag} / {release_title}")


if __name__ == "__main__":
    main()
