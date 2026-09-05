"""Validate release identity and built distribution contents."""

import argparse
import base64
import csv
import hashlib
import re
import shutil
import stat
import tarfile
import tomllib
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

PROJECT_NAME = "gmes"
UNIVERSAL_WHEEL_TAG = "py3-none-any"
REQUIRED_SDIST_PATHS = {
    "CHANGELOG.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "VERSION",
    "build-constraints.txt",
    "gmes/__init__.py",
    "gmes/constant.py",
    "gmes/constant.pyi",
    "gmes/file_io.py",
    "gmes/geometry.py",
    "gmes/material.py",
    "gmes/pygeom.py",
    "gmes/py.typed",
    "gmes/source.py",
    "gmes/torch_distributed.py",
    "gmes/torch_dispersive.py",
    "gmes/torch_dm2.py",
    "gmes/torch_fdtd.py",
    "gmes/torch_output.py",
    "gmes/torch_plan.py",
    "gmes/torch_source.py",
    "pyproject.toml",
    "setup.py",
}
REQUIRED_WHEEL_MODULES = {
    "gmes/__init__.py",
    "gmes/constant.py",
    "gmes/constant.pyi",
    "gmes/file_io.py",
    "gmes/geometry.py",
    "gmes/material.py",
    "gmes/pygeom.py",
    "gmes/py.typed",
    "gmes/source.py",
    "gmes/torch_distributed.py",
    "gmes/torch_dispersive.py",
    "gmes/torch_dm2.py",
    "gmes/torch_fdtd.py",
    "gmes/torch_output.py",
    "gmes/torch_plan.py",
    "gmes/torch_source.py",
}
RETIRED_GMES_PATH_PREFIXES = (
    "gmes/_constant.",
    "gmes/_pw_material.",
    "gmes/fdtd.py",
    "gmes/pw_material.",
    "gmes/pw_source.py",
    "gmes/show.py",
    "src/",
)
FORBIDDEN_DIRECTORY_NAMES = {".git", "__pycache__", "build", "dist"}
FORBIDDEN_ARCHIVE_SUFFIXES = {".h5", ".hdf5", ".pyc", ".pyo"}
NATIVE_ARCHIVE_SUFFIXES = {".a", ".dll", ".dylib", ".lib", ".o", ".obj", ".pyd", ".so"}


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
            if member.issym() or member.islnk():
                raise RuntimeError(f"sdist contains link member {member.name!r}")
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe sdist path: {member.name}")
            if not path.parts or path.parts[0] != expected_root:
                raise RuntimeError(
                    f"sdist member {member.name!r} is outside {expected_root!r}"
                )
            relative = PurePosixPath(*path.parts[1:])
            if relative.parts:
                name = relative.as_posix()
                if name in paths:
                    raise RuntimeError(f"sdist contains duplicate member {name!r}")
                paths.add(name)
    return paths


def _reject_generated_products(paths, archive_kind):
    for name in paths:
        path = PurePosixPath(name)
        if FORBIDDEN_DIRECTORY_NAMES.intersection(path.parts):
            raise RuntimeError(f"{archive_kind} contains generated directory {name!r}")
        if path.suffix.lower() in FORBIDDEN_ARCHIVE_SUFFIXES:
            raise RuntimeError(f"{archive_kind} contains generated file {name!r}")


def _is_native_archive_member(name):
    path = PurePosixPath(name)
    return path.suffix.lower() in NATIVE_ARCHIVE_SUFFIXES or ".so." in path.name.lower()


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
    retired_members = sorted(
        name for name in paths if name.startswith(RETIRED_GMES_PATH_PREFIXES)
    )
    if retired_members:
        raise RuntimeError(
            f"sdist contains retired native or proxy members: {retired_members}"
        )
    if not any(name.startswith("tests/test_") for name in paths):
        raise RuntimeError("sdist does not contain the unit tests")
    if not any(name.startswith("examples/") and name.endswith(".py") for name in paths):
        raise RuntimeError("sdist does not contain the examples")
    compiled = sorted(name for name in paths if _is_native_archive_member(name))
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
        rf"^{PROJECT_NAME}-{re.escape(version)}-({re.escape(UNIVERSAL_WHEEL_TAG)})\.whl$"
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


def _verify_wheel_record(distribution, paths, record_name):
    """Require one complete SHA-256 RECORD matching every wheel member."""
    try:
        rows = list(
            csv.reader(distribution.read(record_name).decode("utf-8").splitlines())
        )
    except UnicodeDecodeError as error:
        raise RuntimeError("wheel RECORD must be UTF-8") from error
    records = {}
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in records:
            raise RuntimeError(
                "wheel RECORD must contain unique path, hash, and size rows"
            )
        records[row[0]] = row[1:]
    if set(records) != paths:
        raise RuntimeError("wheel RECORD members do not match archive members")
    for name, (digest, size) in records.items():
        if name == record_name:
            if digest or size:
                raise RuntimeError(
                    "wheel RECORD self-entry must not have a digest or size"
                )
            continue
        if not digest.startswith("sha256=") or not size.isdecimal():
            raise RuntimeError(
                "wheel RECORD requires SHA-256 digests and decimal sizes"
            )
        payload = distribution.read(name)
        expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(
            b"="
        )
        if digest.partition("=")[2].encode("ascii") != expected or int(size) != len(
            payload
        ):
            raise RuntimeError(f"wheel RECORD digest or size differs for {name!r}")


def verify_wheel(archive, version):
    """Check wheel tags, modules, metadata, and cleanliness."""
    archive = Path(archive)
    wheel_tag = _wheel_platform(archive.name, version)

    dist_info = f"{PROJECT_NAME}-{version}.dist-info"
    metadata_name = f"{dist_info}/METADATA"
    wheel_name = f"{dist_info}/WHEEL"
    record_name = f"{dist_info}/RECORD"
    license_name = f"{dist_info}/licenses/LICENSE"
    with zipfile.ZipFile(archive) as distribution:
        members = distribution.infolist()
        paths = {member.filename for member in members}
        if len(paths) != len(members):
            raise RuntimeError("wheel contains duplicate members")
        for member in members:
            name = member.filename
            path = PurePosixPath(name)
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if member.is_dir() or kind not in (0, stat.S_IFREG):
                raise RuntimeError(f"wheel contains non-regular member {name!r}")
            if not name or path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe wheel path: {name}")

        missing = REQUIRED_WHEEL_MODULES - paths
        if missing:
            raise RuntimeError(f"wheel is missing Python modules: {sorted(missing)}")
        retired_members = sorted(
            name for name in paths if name.startswith(RETIRED_GMES_PATH_PREFIXES)
        )
        if retired_members:
            raise RuntimeError(
                f"wheel contains retired native or proxy members: {retired_members}"
            )
        native_members = sorted(
            name for name in paths if _is_native_archive_member(name)
        )
        if native_members:
            raise RuntimeError(f"wheel contains native members: {native_members}")
        if not {metadata_name, wheel_name, record_name, license_name} <= paths:
            raise RuntimeError("wheel is missing required dist-info metadata")
        unexpected_dist_info = sorted(
            name
            for name in paths
            if ".dist-info/" in name and not name.startswith(f"{dist_info}/")
        )
        if unexpected_dist_info:
            raise RuntimeError(
                f"wheel contains unexpected dist-info metadata: {unexpected_dist_info}"
            )
        metadata = distribution.read(metadata_name).decode("utf-8")
        _verify_core_metadata(metadata, version, "wheel")
        wheel_metadata = distribution.read(wheel_name).decode("utf-8")
        fields = {}
        for line in wheel_metadata.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                fields.setdefault(key, []).append(value)
        if fields.get("Wheel-Version") != ["1.0"]:
            raise RuntimeError("wheel must declare Wheel-Version: 1.0")
        if fields.get("Root-Is-Purelib") != ["true"]:
            raise RuntimeError("wheel must declare Root-Is-Purelib: true")
        if fields.get("Tag") != [UNIVERSAL_WHEEL_TAG]:
            raise RuntimeError("wheel must declare the universal py3-none-any tag")
        _verify_wheel_record(distribution, paths, record_name)

    _reject_generated_products(paths, "wheel")
    return wheel_tag


def verify_distribution_set(dist_directory, version):
    """Require exactly one sdist and one universal pure-Python wheel."""
    dist_directory = Path(dist_directory)
    archives = sorted(path for path in dist_directory.iterdir() if path.is_file())
    sdist = [path for path in archives if path.name.endswith(".tar.gz")]
    wheels = [path for path in archives if path.suffix == ".whl"]
    unexpected = [path.name for path in archives if path not in sdist + wheels]
    if len(sdist) != 1 or len(wheels) != 1 or unexpected:
        raise RuntimeError(
            "release must contain exactly one sdist and one universal wheel; "
            f"found {[path.name for path in archives]}"
        )

    verify_sdist(sdist[0], version)
    wheel_tags = {verify_wheel(wheel, version) for wheel in wheels}
    if wheel_tags != {UNIVERSAL_WHEEL_TAG}:
        raise RuntimeError(
            f"wheel tags {sorted(wheel_tags)} do not match {UNIVERSAL_WHEEL_TAG!r}"
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
