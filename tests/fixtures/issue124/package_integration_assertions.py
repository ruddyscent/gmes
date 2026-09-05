"""Complete sdist-loadability/wheel-exclusion assertions for issue #124."""

from __future__ import annotations

import importlib.util
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


FIXTURE_ROOT = "tests/fixtures/issue124"
PROFILE_FILES = (
    "canonical-dm2-0766dbf93288.json",
    "long-dummy-345c284da712.json",
    "long-gaussian-2b721c7d152f.json",
    "long-tfsf-e6289d035614.json",
    "small-dcp-d7781a0517f3.json",
    "small-drude1-e00d2eae1726.json",
    "small-gaussian-095e5d4cdf76.json",
    "small-index-matrix-5f1106504c56.json",
    "small-initial-eb42a7849f3b.json",
    "small-main-fields-5dec520153b5.json",
    "small-material-matrix-5355ca306f08.json",
    "small-mixed-26ebbf7ce60d.json",
    "small-stability-c3d3ea9a05fc.json",
    "small-tfsf-808b0f4710a6.json",
)
REQUIRED = (
    "benchmarks/historical_probes.py",
    f"{FIXTURE_ROOT}/historical_probe_loader.py",
    f"{FIXTURE_ROOT}/pre-cutover-native-numeric-probes-v5-4.json.gz",
    f"{FIXTURE_ROOT}/pre-cutover-native-numeric-probes-v5-4.sha256",
    f"{FIXTURE_ROOT}/BUNDLE-MANIFEST.json",
    f"{FIXTURE_ROOT}/BUNDLE-MANIFEST.sha256",
    f"{FIXTURE_ROOT}/generator.py",
    f"{FIXTURE_ROOT}/package_integration_assertions.py",
    f"{FIXTURE_ROOT}/self_test.py",
    f"{FIXTURE_ROOT}/schema.json",
    f"{FIXTURE_ROOT}/README.md",
    f"{FIXTURE_ROOT}/REPRODUCTION.md",
    f"{FIXTURE_ROOT}/inputs/native_oracle_workloads.json",
    *(f"{FIXTURE_ROOT}/profiles/{name}" for name in PROFILE_FILES),
)


def assert_required_closure(side: dict) -> None:
    """Keep the synthetic sdist equal to the loader's complete file closure."""
    fixture = side["fixture"]["file"]
    if not fixture.endswith(".json.gz"):
        raise ValueError("fixture path cannot derive its checksum path")
    closure = {
        "benchmarks/historical_probes.py",  # public API placement
        f"{FIXTURE_ROOT}/self_test.py",  # caller-owned external anchor carrier
        f"{FIXTURE_ROOT}/BUNDLE-MANIFEST.json",
        f"{FIXTURE_ROOT}/BUNDLE-MANIFEST.sha256",
        f"{FIXTURE_ROOT}/{fixture}",
        f"{FIXTURE_ROOT}/{fixture.removesuffix('.json.gz')}.sha256",
        f"{FIXTURE_ROOT}/{side['generator']['file']}",
        f"{FIXTURE_ROOT}/{side['generator']['loader_file']}",
        f"{FIXTURE_ROOT}/{side['inputs']['source_manifest']['file']}",
        *(f"{FIXTURE_ROOT}/{descriptor['file']}" for descriptor in side["support"].values()),
        *(f"{FIXTURE_ROOT}/{descriptor['file']}" for descriptor in side["profiles"].values()),
    }
    if len(REQUIRED) != len(set(REQUIRED)) or set(REQUIRED) != closure:
        missing, extra = sorted(closure - set(REQUIRED)), sorted(set(REQUIRED) - closure)
        raise ValueError(f"sdist required set is not the bundle closure: missing={missing!r}, extra={extra!r}")


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("packaged_historical_probes", path)
    if spec is None or spec.loader is None:
        raise ValueError("packaged loader cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def assert_sdist(path: str | Path, *, trust: dict) -> None:
    """Require every dependency exactly once, extract safely, and load it."""
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if not names:
            raise ValueError("sdist is empty")
        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        if len(roots) != 1:
            raise ValueError("sdist must have exactly one top-level directory")
        root = next(iter(roots))
        relative = [name.removeprefix(f"{root}/") for name in names if name != root]
        for required in REQUIRED:
            if relative.count(required) != 1:
                raise ValueError(f"sdist fixture placement failure: {required}")
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts) or member.issym() or member.islnk():
                raise ValueError("sdist contains an unsafe path/link")
        with tempfile.TemporaryDirectory() as temporary:
            archive.extractall(temporary, filter="data")
            extracted = Path(temporary) / root
            loader = _load_module(extracted / "benchmarks/historical_probes.py")
            side, fixture = loader.load_bundle(extracted / FIXTURE_ROOT, **trust)
            assert_required_closure(side)
            if (len(side["profiles"]), len(fixture["captures"]), len(side["runtime_bindings"])) != (14, 35, 56):
                raise ValueError("packaged bundle loaded with incomplete counts")


def assert_wheel(path: str | Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    leaked = [name for name in names if "fixtures/issue124" in name or name.endswith(("historical_probes.py", "historical_probe_loader.py"))]
    if leaked:
        raise ValueError(f"wheel contains test-only fixture/support: {leaked!r}")
