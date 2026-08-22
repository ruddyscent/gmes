"""Validation helpers for reproducible macOS wheel compatibility."""

import subprocess
from pathlib import Path

MINIMUM_MACOS_VERSION = "11.0"


def _version_tuple(version):
    """Return a version tuple suitable for comparing macOS targets."""
    return tuple(int(part) for part in version.split("."))


def verify_wheel_platform_tag(platform_tag, target):
    """Ensure every wheel platform tag uses the requested macOS target."""
    target_tag = "_".join(str(part) for part in _version_tuple(target)[:2])
    expected_prefix = f"macosx_{target_tag}_"
    tags = platform_tag.split(".")
    if not tags or any(not tag.startswith(expected_prefix) for tag in tags):
        raise RuntimeError(f"macOS wheel tag {platform_tag!r} does not target {target}")


def extract_macos_targets(otool_output):
    """Extract minimum macOS versions from Mach-O load commands."""
    targets = []
    command = None
    for line in otool_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("cmd "):
            command = stripped.removeprefix("cmd ")
        elif command == "LC_BUILD_VERSION" and stripped.startswith("minos "):
            targets.append(stripped.removeprefix("minos ").split()[0])
        elif command == "LC_VERSION_MIN_MACOSX" and stripped.startswith("version "):
            targets.append(stripped.removeprefix("version ").split()[0])
    return targets


def inspect_macos_targets(path):
    """Return the minimum macOS versions recorded in a Mach-O binary."""
    try:
        result = subprocess.run(
            ["otool", "-l", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"could not inspect macOS target metadata in {path}"
        ) from error
    targets = extract_macos_targets(result.stdout)
    if not targets:
        raise RuntimeError(f"could not find a minimum macOS version in {path}")
    return targets


def verify_macos_dependency_target(path, target):
    """Reject a Mach-O dependency that requires a newer macOS target."""
    expected = _version_tuple(target)
    targets = inspect_macos_targets(path)
    incompatible = [
        version for version in targets if _version_tuple(version) > expected
    ]
    if incompatible:
        versions = ", ".join(incompatible)
        raise RuntimeError(
            f"OpenMP runtime {path} targets macOS {versions}, "
            f"newer than requested {target}"
        )


def select_macos_openmp_prefix(prefixes, target):
    """Select the first complete libomp installation compatible with target."""
    incompatibilities = []
    for candidate in map(Path, prefixes):
        header = candidate / "include" / "omp.h"
        runtime = candidate / "lib" / "libomp.dylib"
        if not header.is_file() or not runtime.is_file():
            continue
        try:
            verify_macos_dependency_target(runtime, target)
        except RuntimeError as error:
            incompatibilities.append(str(error))
            continue
        return candidate, incompatibilities
    return None, incompatibilities


def verify_extension_targets(extension_paths, target):
    """Ensure built Mach-O extensions use the requested minimum macOS version."""
    expected = _version_tuple(target)
    native_paths = [
        Path(path) for path in extension_paths if Path(path).suffix == ".so"
    ]
    if not native_paths:
        raise RuntimeError("macOS wheel build produced no native extensions")

    for path in native_paths:
        targets = inspect_macos_targets(path)
        mismatches = [
            version for version in targets if _version_tuple(version) != expected
        ]
        if mismatches:
            raise RuntimeError(
                f"native extension {path} targets {mismatches}, expected {target}"
            )
