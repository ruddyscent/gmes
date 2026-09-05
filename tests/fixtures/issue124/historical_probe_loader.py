"""Pure, bounded reference loader/comparator for the V5.4 test-only fixture."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np


MAX_COMPRESSED = 5 * 1024 * 1024
MAX_UNCOMPRESSED = 5 * 1024 * 1024
MAX_SIDE_MANIFEST = 1024 * 1024
MAX_NPZ_BYTES = 128 * 1024 * 1024
MAX_NPZ_MEMBERS = 10_000
MAX_NPY_HEADER_BYTES = 64 * 1024
MAX_NPY_PAYLOAD_BYTES = 128 * 1024 * 1024
MAX_TOTAL_ARRAY_BYTES = 256 * 1024 * 1024
FIXTURE_SCHEMA_VERSION = 8
BUNDLE_SCHEMA_VERSION = 3
FIXTURE_NAME = "pre-cutover-native-numeric-probes-v5-4.json.gz"
FIXTURE_CHECKSUM_NAME = "pre-cutover-native-numeric-probes-v5-4.sha256"
FIXTURE_KIND = "pre-cutover-native-numeric-probes"
PROJECTION_ALGORITHM = "validated-probe-projection-v5-storage-dtype-contract"
PROJECTION_PREFIX = b"gmes-issue124-probe-projection-v5\0"
SAMPLING_ALGORITHM = "reference-selector-candidate-fixed-positions-v3"
EXPECTED_COUNTS = {
    "profiles": 14,
    "captures": 35,
    "arrays": 4377,
    "runtime_bindings": 56,
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,119}$")
SAFE_KEY = re.compile(r"^(?:map/[EH][xyz]/(?:material_ids|underlying_ids)|step/[0-9]+/[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+){0,5})$")
SAFE_DTYPE = re.compile(r"^(?:bool|int(?:8|16|32|64)|uint(?:8|16|32|64)|float(?:16|32|64)|complex(?:64|128))$")
COORDINATE_INDEX_GRAMMARS = (
    re.compile(r"^step/[0-9]+/source/[EH][xyz]/[A-Za-z0-9_.+-]+/indices$"),
    re.compile(r"^step/[0-9]+/source_aux_material/[0-9]+/[EH][xyz]/[A-Za-z0-9_.+-]+/indices$"),
    re.compile(r"^step/[0-9]+/state/[EH][xyz]/[A-Za-z0-9_.+-]+/indices$"),
)
MAP_ID_PATH = re.compile(r"^map/[EH][xyz]/(?:material_ids|underlying_ids)$")
SAFE_LABEL = r"[A-Za-z0-9_.+-]+"
PUBLIC_KEY_GRAMMARS = (
    ("map", MAP_ID_PATH),
    ("field", re.compile(r"^step/[0-9]+/field/[EH][xyz]$")),
    ("physical", re.compile(r"^step/[0-9]+/physical/(?:spectrum/[EH][xyz]|summary)$")),
    ("time", re.compile(r"^step/[0-9]+/time$")),
    ("source_indices", re.compile(rf"^step/[0-9]+/source/[EH][xyz]/{SAFE_LABEL}/indices$")),
    ("source_aux", re.compile(rf"^step/[0-9]+/source_aux/[0-9]+-{SAFE_LABEL}/(?:field/[EH][xyz]|time)$")),
    ("source_aux_material", re.compile(rf"^step/[0-9]+/source_aux_material/[0-9]+/[EH][xyz]/{SAFE_LABEL}/(?:indices|values)$")),
    ("state", re.compile(rf"^step/[0-9]+/state/[EH][xyz]/{SAFE_LABEL}/(?:indices|values)$")),
)
OPAQUE_DIRECT_SOURCE_VALUE = re.compile(rf"^step/[0-9]+/source/[EH][xyz]/{SAFE_LABEL}/values$")
EXPECTED_FAMILY_COUNTS = {
    "map": 420,
    "field": 498,
    "physical": 581,
    "time": 83,
    "source_indices": 145,
    "source_aux": 154,
    "source_aux_material": 264,
    "state": 2232,
}
RUNTIME_KEYS = {"device", "precision", "graph_mode", "compile_mode"}
STRATEGY_TOLERANCE_MODEL = {
    "Const": "dielectric",
    "Cpml": "pml",
    "DcpAde": "dcp-ade",
    "DcpPlrc": "dcp-plrc",
    "DcpRc": "dcp-rc",
    "Dielectric": "dielectric",
    "Dm2": "dm2",
    "Drude": "drude",
    "Lorentz": "lorentz",
    "Upml": "pml",
}


@dataclass(frozen=True)
class LoadedBundle:
    side: dict
    fixture: dict
    profiles: dict

    def __iter__(self):
        yield self.side
        yield self.fixture


@dataclass(frozen=True)
class ResolvedCase:
    profile_id: str
    case: str
    manifest: dict
    runtime: dict
    consumer_line: int
    capture: dict


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def projection_sha256(profile: str, case: str, archive_schema_version: int, arrays: list) -> str:
    """Hash exactly the domain-separated stable probe projection."""
    projection = {
        "profile": profile,
        "case": case,
        "archive_schema_version": archive_schema_version,
        "arrays": arrays,
    }
    encoded = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(PROJECTION_PREFIX + encoded).hexdigest()


def projection_class(key: str) -> str | None:
    """Classify one raw key as public, paired opaque, or forbidden."""
    matches = [name for name, grammar in PUBLIC_KEY_GRAMMARS if grammar.fullmatch(key)]
    if len(matches) > 1:
        raise ValueError("semantic projection grammar is ambiguous")
    if matches:
        return matches[0]
    if OPAQUE_DIRECT_SOURCE_VALUE.fullmatch(key):
        return "opaque_source_value"
    return None


def coordinate_index_family(key: str) -> int | None:
    matches = [index for index, grammar in enumerate(COORDINATE_INDEX_GRAMMARS) if grammar.fullmatch(key)]
    if len(matches) > 1:
        raise ValueError("coordinate-index grammar is ambiguous")
    return matches[0] if matches else None


def canonical_semantic_projection_array(key: str, value: np.ndarray) -> np.ndarray:
    """Accept candidate arrays without casting reviewed coordinate semantics."""
    data = np.asarray(value)
    if coordinate_index_family(key) is None:
        return data
    if (
        str(data.dtype) != "int64"
        or data.ndim != 2
        or data.shape[1:] != (3,)
    ):
        raise ValueError(f"coordinate-index candidate storage contract differs: {key}")
    return data


def validate_storage_reference_dtype(key: str, reference_dtype: str) -> str:
    """Validate the total V5.4 semantic storage-dtype grammar."""
    if MAP_ID_PATH.fullmatch(key):
        family, allowed = "map", {"int32"}
    elif coordinate_index_family(key) is not None:
        family, allowed = "coordinate_index", {"int64"}
    elif re.fullmatch(r"step/[0-9]+/field/[EH][xyz]", key):
        family, allowed = "primary_field", {"float64", "complex128"}
    elif re.fullmatch(r"step/[0-9]+/physical/spectrum/[EH][xyz]", key):
        family, allowed = "physical_spectrum", {"float64"}
    elif re.fullmatch(r"step/[0-9]+/physical/summary", key):
        family, allowed = "physical_summary", {"float64"}
    elif re.fullmatch(r"step/[0-9]+/time", key):
        family, allowed = "primary_time", {"float64"}
    elif re.fullmatch(rf"step/[0-9]+/source_aux/[0-9]+-{SAFE_LABEL}/(?:field/[EH][xyz]|time)", key):
        family, allowed = "source_aux", {"float64"}
    elif re.fullmatch(rf"step/[0-9]+/source_aux_material/[0-9]+/[EH][xyz]/{SAFE_LABEL}/indices", key):
        family, allowed = "source_aux_material_indices", {"int64"}
    elif re.fullmatch(rf"step/[0-9]+/source_aux_material/[0-9]+/[EH][xyz]/{SAFE_LABEL}/values", key):
        family, allowed = "source_aux_material_values", {"complex128"}
    elif re.fullmatch(rf"step/[0-9]+/state/[EH][xyz]/{SAFE_LABEL}/indices", key):
        family, allowed = "state_indices", {"int64"}
    elif re.fullmatch(rf"step/[0-9]+/state/[EH][xyz]/{SAFE_LABEL}/values", key):
        family, allowed = "state_values", {"complex128"}
    elif re.fullmatch(rf"step/[0-9]+/source/[EH][xyz]/{SAFE_LABEL}/indices", key):
        family, allowed = "source_indices", {"int64"}
    else:
        raise ValueError(f"unreviewed V5.4 storage-dtype key: {key}")
    if reference_dtype not in allowed:
        raise ValueError(f"V5.4 reference storage dtype differs: {key}")
    return family


def _selector_positions(data: np.ndarray) -> list[int]:
    """Use the reviewed append-once reference selector with NumPy first-ties."""
    flat, count = data.reshape(-1), int(data.size)
    magnitude = np.abs(flat)
    first_nonzero = int(np.flatnonzero(flat)[0]) if np.any(flat != 0) else None
    maxabs = int(np.argmax(magnitude)) if count else None
    positions = []
    for index in (0, count - 1, count // 2, first_nonzero, maxabs):
        if index is not None and 0 <= index < count and index not in positions:
            positions.append(index)
    for part in range(8):
        index = (count - 1) * part // 7 if count else None
        if index is not None and index not in positions:
            positions.append(index)
    return positions


def _reference_selector_positions(reference: list) -> list[int]:
    """Validate and return the immutable descriptor-selected positions."""
    _key, _dtype, shape, complex_flag, stats, samples = reference
    count = stats[0]
    positions = [sample[0] for sample in samples]
    base = []
    for index in (0, count - 1, count // 2):
        if count and index not in base:
            base.append(index)
    if positions[:len(base)] != base:
        raise ValueError("reference selector base ordering differs")
    values = {sample[0]: sample[2] for sample in samples}
    grid = [(count - 1) * part // 7 for part in range(8)] if count else []

    def nonzero(index: int) -> bool:
        value = values[index]
        return value[0] != 0 or (complex_flag and value[1] != 0)

    def maximum(index: int) -> bool:
        value = values[index]
        magnitude = math.hypot(value[0], value[1]) if complex_flag else abs(value[0])
        return math.isclose(magnitude, stats[4], rel_tol=2e-15, abs_tol=0.0)

    def matches(stage: int, selected: list[int]) -> bool:
        if stage == 2:
            expected = list(selected)
            for index in grid:
                if index not in expected:
                    expected.append(index)
            return expected == positions
        if matches(stage + 1, selected):
            return True
        if len(selected) >= len(positions):
            return False
        index = positions[len(selected)]
        predicate = nonzero if stage == 0 else maximum
        return index not in selected and predicate(index) and matches(stage + 1, selected + [index])

    if not matches(0, list(base)):
        raise ValueError("reference selector append-once ordering differs")
    return positions


def reference_value_selected_positions(reference: list) -> list[int]:
    """Expose the retained non-base/non-grid selector inventory for review."""
    count = reference[4][0]
    base = {index for index in (0, count - 1, count // 2) if count}
    grid = {(count - 1) * part // 7 for part in range(8)} if count else set()
    return [index for index in _reference_selector_positions(reference) if index not in base | grid]


def _confined_file(root: Path, relative: str) -> Path:
    """Resolve one normalized regular file without symlink/hard-link escape."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("descriptor path is not normalized relative POSIX form")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("descriptor path is not normalized relative POSIX form")
    candidate = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ValueError(f"bundle file is missing: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"bundle path traverses a symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"bundle path escapes bundle: {relative}") from error
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"bundle path is not a regular file: {relative}")
    if metadata.st_nlink != 1:
        raise ValueError(f"bundle file is hard-linked: {relative}")
    return resolved


def _checksum(path: Path, digest: str, basename: str) -> None:
    if path.read_bytes() != f"{digest}  {basename}\n".encode():
        raise ValueError(f"checksum file differs: {path.name}")


def _descriptor_file(root: Path, descriptor: dict, label: str) -> tuple[Path, bytes]:
    if not isinstance(descriptor, dict) or set(descriptor) != {"file", "bytes", "sha256"}:
        raise ValueError(f"invalid {label} descriptor topology")
    if type(descriptor["bytes"]) is not int or descriptor["bytes"] < 0:
        raise ValueError(f"invalid {label} descriptor bytes")
    if not isinstance(descriptor["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]):
        raise ValueError(f"invalid {label} descriptor SHA-256")
    path = _confined_file(root, descriptor["file"])
    data = path.read_bytes()
    if len(data) != descriptor["bytes"] or hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"{label} byte/hash mismatch")
    return path, data


def _bounded_gzip(path: Path) -> bytes:
    if path.stat().st_size > MAX_COMPRESSED:
        raise ValueError("fixture compressed-size ceiling exceeded")
    data = bytearray()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(65536):
            data.extend(block)
            if len(data) > MAX_UNCOMPRESSED:
                raise ValueError("fixture uncompressed-size ceiling exceeded")
    return bytes(data)


def load_bundle(
    bundle: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_manifest_bytes: int,
    expected_fixture_sha256: str,
    expected_fixture_bytes: int,
) -> LoadedBundle:
    """Load only after verifying caller-owned immutable size/hash anchors."""
    if not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        for value in (expected_manifest_sha256, expected_fixture_sha256)
    ):
        raise ValueError("invalid caller SHA-256 trust anchor")
    if (
        type(expected_manifest_bytes) is not int
        or not 0 < expected_manifest_bytes <= MAX_SIDE_MANIFEST
        or type(expected_fixture_bytes) is not int
        or not 0 < expected_fixture_bytes <= MAX_COMPRESSED
    ):
        raise ValueError("invalid caller byte-size trust anchor")
    root_input = Path(bundle)
    if root_input.is_symlink():
        raise ValueError("bundle root must not be a symlink")
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("bundle root is not a directory")
    side_path = _confined_file(root, "BUNDLE-MANIFEST.json")
    if side_path.stat().st_size != expected_manifest_bytes:
        raise ValueError("side-manifest byte trust anchor differs")
    side_bytes = side_path.read_bytes()
    if hashlib.sha256(side_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("side-manifest SHA-256 trust anchor differs")
    _checksum(
        _confined_file(root, "BUNDLE-MANIFEST.sha256"),
        expected_manifest_sha256,
        "BUNDLE-MANIFEST.json",
    )
    side = json.loads(side_bytes, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    if set(side) != {"bundle_schema_version", "fixture_schema_version", "kind", "fixture", "generator", "inputs", "profiles", "provenance", "runtime_bindings", "sampling", "support"}:
        raise ValueError("unexpected side-manifest topology")
    if side["bundle_schema_version"] != BUNDLE_SCHEMA_VERSION or side["fixture_schema_version"] != FIXTURE_SCHEMA_VERSION or side["kind"] != FIXTURE_KIND:
        raise ValueError("unexpected bundle schema version")
    if not isinstance(side["profiles"], dict) or list(side["profiles"]) != sorted(side["profiles"]) or len(side["profiles"]) != EXPECTED_COUNTS["profiles"]:
        raise ValueError("profile descriptors must be sorted and complete")
    profile_documents = {}
    for profile_id, descriptor in side["profiles"].items():
        if not SAFE_ID.fullmatch(profile_id):
            raise ValueError("unsafe profile identifier")
        _profile_path, profile_data = _descriptor_file(root, descriptor, f"profile {profile_id}")
        profile_documents[profile_id] = json.loads(profile_data, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
        if not isinstance(profile_documents[profile_id], dict):
            raise ValueError("profile is not a JSON object")
    if not isinstance(side["inputs"], dict) or set(side["inputs"]) != {"source_manifest"}:
        raise ValueError("input descriptor topology differs")
    source_descriptor = side["inputs"]["source_manifest"]
    _source_path, source_data = _descriptor_file(root, source_descriptor, "source manifest")
    json.loads(source_data, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    if not isinstance(side["generator"], dict) or set(side["generator"]) != {"file", "bytes", "sha256", "loader_file", "loader_bytes", "loader_sha256"}:
        raise ValueError("generator descriptor topology differs")
    for file_key, hash_key in (("file", "sha256"), ("loader_file", "loader_sha256")):
        artifact = _confined_file(root, side["generator"][file_key])
        bytes_key = "bytes" if file_key == "file" else "loader_bytes"
        if type(side["generator"][bytes_key]) is not int or artifact.stat().st_size != side["generator"][bytes_key] or _sha(artifact) != side["generator"][hash_key]:
            raise ValueError("generator/loader hash mismatch")
    support = side["support"]
    if not isinstance(support, dict) or set(support) != {"package_assertions", "schema", "readme", "reproduction"}:
        raise ValueError("support descriptor topology differs")
    for label, descriptor in support.items():
        _descriptor_file(root, descriptor, label)
    fixture_info = side["fixture"]
    if not isinstance(fixture_info, dict) or set(fixture_info) != {"file", "compressed_bytes", "compressed_sha256", "uncompressed_bytes", "uncompressed_sha256"}:
        raise ValueError("fixture descriptor topology differs")
    if (
        fixture_info["compressed_bytes"] != expected_fixture_bytes
        or fixture_info["compressed_sha256"] != expected_fixture_sha256
    ):
        raise ValueError("fixture descriptor differs from caller trust anchor")
    fixture_path = _confined_file(root, fixture_info["file"])
    if fixture_path.name != FIXTURE_NAME:
        raise ValueError("unexpected fixture basename")
    if fixture_path.stat().st_size != fixture_info["compressed_bytes"] or _sha(fixture_path) != fixture_info["compressed_sha256"]:
        raise ValueError("fixture compressed digest/size mismatch")
    _checksum(
        _confined_file(root, FIXTURE_CHECKSUM_NAME),
        expected_fixture_sha256,
        fixture_info["file"],
    )
    data = _bounded_gzip(fixture_path)
    if len(data) != fixture_info["uncompressed_bytes"] or hashlib.sha256(data).hexdigest() != fixture_info["uncompressed_sha256"]:
        raise ValueError("fixture uncompressed digest/size mismatch")
    if not data.endswith(b"\n") or data != json.dumps(json.loads(data, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant), sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n":
        raise ValueError("fixture is not canonical JSON with one final LF")
    fixture = json.loads(data, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    if set(fixture) != {"captures", "fixture_schema_version", "kind", "profiles", "provenance", "runtime_bindings", "sampling", "tolerances_numeric"}:
        raise ValueError("unexpected fixture topology")
    if fixture["fixture_schema_version"] != FIXTURE_SCHEMA_VERSION or fixture["kind"] != FIXTURE_KIND:
        raise ValueError("fixture schema/kind mismatch")
    if fixture["fixture_schema_version"] != side["fixture_schema_version"] or fixture["kind"] != side["kind"] or fixture["provenance"] != side["provenance"] or fixture["runtime_bindings"] != side["runtime_bindings"] or fixture["profiles"] != side["profiles"]:
        raise ValueError("fixture/side-manifest binding mismatch")
    _validate_fixture(fixture, side, profile_documents)
    return LoadedBundle(side, fixture, profile_documents)


def _validate_fixture(fixture: dict, side: dict, profile_documents: dict) -> None:
    if not isinstance(side["sampling"], dict) or set(side["sampling"]) != {"algorithm", "limits", "max_samples_per_array"}:
        raise ValueError("sampling topology differs")
    if side["sampling"] != fixture["sampling"] or side["sampling"]["algorithm"] != SAMPLING_ALGORITHM or side["sampling"]["max_samples_per_array"] != 13:
        raise ValueError("sampling contract differs")
    provenance = side["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"repository", "capture_source_commit", "native_oracle_sha256", "historical_observer", "source_manifest_sha256", "generator_sha256"}:
        raise ValueError("provenance topology differs")
    if provenance["source_manifest_sha256"] != side["inputs"]["source_manifest"]["sha256"] or provenance["generator_sha256"] != side["generator"]["sha256"]:
        raise ValueError("provenance file binding differs")
    observer = provenance["historical_observer"]
    if not isinstance(observer, dict) or set(observer) != {"tag", "tag_object", "peeled_commit", "role"} or observer["role"] != "historical-manifest-reference-not-capture-source":
        raise ValueError("historical observer provenance differs")
    for name in ("capture_source_commit", "native_oracle_sha256", "source_manifest_sha256", "generator_sha256"):
        if not isinstance(provenance[name], str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", provenance[name]):
            raise ValueError("invalid provenance digest")
    if set(profile_documents) != set(side["profiles"]):
        raise ValueError("profile document coverage differs")
    profile_cases, profile_case_steps = {}, {}
    for profile_id, profile in profile_documents.items():
        if set(profile) != {"schema_version", "reference", "correctness", "physical_checks", "benchmarks", "tolerances", "performance_gates"}:
            raise ValueError("profile topology differs")
        if type(profile["schema_version"]) is not int or not isinstance(profile["reference"], dict):
            raise ValueError("profile schema/reference differs")
        steps = profile["reference"].get("capture_steps")
        if not isinstance(steps, list) or not steps or any(type(step) is not int or step <= 0 for step in steps) or steps != sorted(set(steps)):
            raise ValueError("profile capture steps differ")
        cases, case_steps = [], {}
        for section in ("correctness", "physical_checks"):
            if not isinstance(profile[section], list):
                raise ValueError("profile case list differs")
            for spec in profile[section]:
                if not isinstance(spec, dict) or not SAFE_ID.fullmatch(spec.get("name", "")):
                    raise ValueError("profile case identity differs")
                cases.append(spec["name"])
                selected_steps = spec.get("capture_steps", steps)
                if (
                    not isinstance(selected_steps, list)
                    or not selected_steps
                    or any(type(step) is not int or step <= 0 for step in selected_steps)
                    or selected_steps != sorted(set(selected_steps))
                ):
                    raise ValueError("profile case capture schedule differs")
                case_steps[spec["name"]] = selected_steps
        if len(cases) != len(set(cases)):
            raise ValueError("duplicate profile case")
        profile_cases[profile_id] = set(cases)
        profile_case_steps[profile_id] = case_steps
    bindings = side["runtime_bindings"]
    if not isinstance(bindings, list) or len(bindings) != EXPECTED_COUNTS["runtime_bindings"]:
        raise ValueError("runtime binding count differs")
    binding_ids = set()
    covered = set()
    cuda = set()
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"test_line", "profile", "profile_sha256", "case", "capture_id", "capture_steps", "runtime", "probe_projection_sha256"}:
            raise ValueError("runtime binding topology differs")
        profile_id, case, runtime = binding["profile"], binding["case"], binding["runtime"]
        if type(binding["test_line"]) is not int or binding["test_line"] <= 0 or profile_id not in profile_cases or case not in profile_cases[profile_id]:
            raise ValueError("runtime binding profile/case differs")
        binding_steps = binding["capture_steps"]
        if (
            binding["profile_sha256"] != side["profiles"][profile_id]["sha256"]
            or binding["capture_id"] != f"{profile_id}:{case}"
            or not isinstance(binding_steps, list)
            or not binding_steps
            or any(type(step) is not int or step <= 0 for step in binding_steps)
            or binding_steps != sorted(set(binding_steps))
            or not set(binding_steps).issubset(case_steps := profile_case_steps[profile_id][case])
            or not isinstance(binding["probe_projection_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", binding["probe_projection_sha256"])
        ):
            raise ValueError("runtime binding stable projection differs")
        _validate_runtime(runtime)
        identity = (binding["test_line"], profile_id, case, tuple(runtime.items()))
        if identity in binding_ids:
            raise ValueError("duplicate runtime binding")
        binding_ids.add(identity)
        covered.add((profile_id, case))
        if runtime["device"] == "cuda:0":
            cuda.add((runtime["precision"], runtime["graph_mode"], runtime["compile_mode"]))
    if cuda != {("float32", "eager", "default"), ("float32", "graph", "reduce-overhead")}:
        raise ValueError("CUDA contracts differ")
    float64_lines = {1426, 1753, 1930, 1975, 1990, 2012}
    actual_float64 = {binding["test_line"] for binding in bindings if binding["test_line"] in float64_lines and binding["runtime"]["precision"] == "float64"}
    if actual_float64 != float64_lines:
        raise ValueError("six reviewed default-precision consumers differ")
    captures = fixture["captures"]
    if not isinstance(captures, dict) or list(captures) != sorted(captures) or len(captures) != EXPECTED_COUNTS["captures"]:
        raise ValueError("capture IDs must be unique/sorted")
    array_count = numeric_count = dynamic_record_count = dynamic_position_count = 0
    family_counts = {name: 0 for name in EXPECTED_FAMILY_COUNTS}
    for capture_id, capture in captures.items():
        if not isinstance(capture, dict) or set(capture) != {"profile", "case", "archive_schema_version", "probe_projection", "arrays"}:
            raise ValueError("capture topology differs")
        if not SAFE_ID.fullmatch(capture_id) or not SAFE_ID.fullmatch(capture["profile"]) or not SAFE_ID.fullmatch(capture["case"]) or capture_id != f"{capture['profile']}:{capture['case']}":
            raise ValueError("unsafe capture identifier")
        if (capture["profile"], capture["case"]) not in covered or capture["archive_schema_version"] != 2:
            raise ValueError("capture profile/schema coverage differs")
        projection = capture["probe_projection"]
        if not isinstance(projection, dict) or set(projection) != {"algorithm", "sha256"} or projection["algorithm"] != PROJECTION_ALGORITHM or not isinstance(projection["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", projection["sha256"]):
            raise ValueError("probe projection descriptor differs")
        if not isinstance(capture["arrays"], list) or not capture["arrays"]:
            raise ValueError("capture arrays differ")
        prior = None
        for probe in capture["arrays"]:
            if not isinstance(probe, list) or len(probe) != 6:
                raise ValueError("probe record topology differs")
            key, dtype, shape, complex_flag, stats, samples = probe
            classification = projection_class(key) if isinstance(key, str) else None
            if not isinstance(key, str) or classification is None or classification == "opaque_source_value" or not isinstance(dtype, str) or not SAFE_DTYPE.fullmatch(dtype) or (prior is not None and key <= prior):
                raise ValueError("unsafe/non-increasing/duplicate array key")
            prior = key
            family_counts[classification] += 1
            coordinate_index = coordinate_index_family(key)
            if coordinate_index is not None and (
                dtype != "int64"
                or complex_flag is not False
                or len(shape) != 2
                or shape[-1:] != [3]
            ):
                raise ValueError("coordinate-index semantic descriptor differs")
            if key.endswith("/indices") and coordinate_index is None:
                raise ValueError("unreviewed coordinate-index descriptor differs")
            if type(complex_flag) is not bool or complex_flag != dtype.startswith("complex"):
                raise ValueError("complex flag/dtype mismatch")
            validate_storage_reference_dtype(key, dtype)
            if not isinstance(shape, list) or len(shape) > 8 or any(type(item) is not int or item < 0 or item > 1_000_000 for item in shape):
                raise ValueError("invalid shape")
            if not isinstance(stats, list) or len(stats) != 9:
                raise ValueError("statistics tuple length differs")
            count, finite_count, nonzero_count, *numeric = stats
            if any(type(item) is not int or item < 0 for item in (count, finite_count, nonzero_count)) or finite_count != count or nonzero_count > count:
                raise ValueError("invalid/non-finite probe statistics")
            if any(type(item) not in {int, float} or not math.isfinite(float(item)) for item in numeric):
                raise ValueError("non-finite aggregate")
            expected_count = math.prod(shape) if shape else 1
            if count != expected_count or count > 100_000_000 or stats[3] < 0 or stats[4] < 0 or stats[5] > stats[6] or stats[7] > stats[8]:
                raise ValueError("shape/count mismatch")
            if not isinstance(samples, list) or len(samples) > side["sampling"]["max_samples_per_array"]:
                raise ValueError("sample list length differs")
            seen = set()
            for sample in samples:
                if not isinstance(sample, list) or len(sample) != 3:
                    raise ValueError("sample tuple length differs")
                flat, coordinate, value = sample
                if type(flat) is not int or flat < 0 or flat >= count or flat in seen:
                    raise ValueError("invalid sample index")
                seen.add(flat)
                if not isinstance(coordinate, list) or any(type(item) is not int for item in coordinate) or (list(np.unravel_index(flat, tuple(shape))) if shape else []) != coordinate:
                    raise ValueError("sample coordinate mismatch")
                if not isinstance(value, list) or len(value) != 2 or type(value[0]) not in {int, float} or (complex_flag and type(value[1]) not in {int, float}) or (not complex_flag and value[1] is not None):
                    raise ValueError("invalid sample value")
                if coordinate_index is not None and type(value[0]) is not int:
                    raise ValueError("coordinate-index sample is not exact int64")
                if any(component is not None and not math.isfinite(float(component)) for component in value):
                    raise ValueError("non-finite sample")
            base = []
            for index in (0, count - 1, count // 2):
                if count and index not in base:
                    base.append(index)
            if [sample[0] for sample in samples[:len(base)]] != base:
                raise ValueError("missing deterministic base sentinels")
            if nonzero_count and not any(sample[2][0] != 0 or (complex_flag and sample[2][1] != 0) for sample in samples):
                raise ValueError("missing first-nonzero sentinel")
            if count and not any(math.isclose(math.hypot(sample[2][0], sample[2][1]) if complex_flag else abs(sample[2][0]), stats[4], rel_tol=2e-15, abs_tol=0.0) for sample in samples):
                raise ValueError("missing max-absolute sentinel")
            _reference_selector_positions(probe)
            if not dtype.startswith(("bool", "int", "uint")):
                numeric_count += 1
                dynamic = reference_value_selected_positions(probe)
                dynamic_record_count += bool(dynamic)
                dynamic_position_count += len(dynamic)
        if projection["sha256"] != projection_sha256(capture["profile"], capture["case"], capture["archive_schema_version"], capture["arrays"]):
            raise ValueError("probe projection digest differs")
        array_count += len(capture["arrays"])
    if set(captures) != {f"{profile}:{case}" for profile, case in covered}:
        raise ValueError("capture/runtime binding coverage differs")
    for binding in bindings:
        capture = captures.get(binding["capture_id"])
        capture_steps = {
            int(probe[0].split("/", 2)[1])
            for probe in capture["arrays"]
            if probe[0].startswith("step/")
        }
        if (
            capture is None
            or capture["profile"] != binding["profile"]
            or capture["case"] != binding["case"]
            or capture["probe_projection"]["sha256"] != binding["probe_projection_sha256"]
            or not set(binding["capture_steps"]).issubset(capture_steps)
        ):
            raise ValueError("runtime binding capture/projection differs")
    if (
        family_counts != EXPECTED_FAMILY_COUNTS
        or array_count != EXPECTED_COUNTS["arrays"]
        or (numeric_count, dynamic_record_count, dynamic_position_count) != (2564, 729, 771)
    ):
        raise ValueError("array probe count differs")


def _validate_runtime(runtime: dict) -> None:
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_KEYS:
        raise ValueError("runtime topology differs")
    if runtime["device"] not in {"cpu", "cuda:0"} or runtime["precision"] not in {"float32", "float64"} or runtime["graph_mode"] not in {"eager", "graph"} or runtime["compile_mode"] not in {"default", "reduce-overhead"}:
        raise ValueError("runtime value differs")
    if runtime["graph_mode"] == "eager" and runtime["compile_mode"] != "default":
        raise ValueError("eager runtime compile mode differs")
    if runtime["compile_mode"] == "reduce-overhead" and (runtime["device"], runtime["graph_mode"]) != ("cuda:0", "graph"):
        raise ValueError("reduce-overhead runtime differs")


def probe_array(key: str, value: np.ndarray, *, sample_positions: list[int] | None = None) -> list:
    """Recompute a probe, optionally at immutable validated reference positions."""
    classification = projection_class(key)
    if classification is None or classification == "opaque_source_value":
        raise ValueError(f"unprojectable semantic array key: {key!r}")
    data = canonical_semantic_projection_array(key, value)
    coordinate_index = coordinate_index_family(key) is not None
    dtype = str(data.dtype)
    if not SAFE_DTYPE.fullmatch(dtype):
        raise ValueError(f"unsafe dtype: {dtype!r}")
    flat, count = data.reshape(-1), int(data.size)
    complex_value = bool(np.iscomplexobj(flat))
    if not bool(np.all(np.isfinite(flat))):
        raise ValueError(f"non-finite candidate array: {key}")

    def finite_number(item) -> float:
        result = float(item)
        if not math.isfinite(result):
            raise ValueError(f"non-finite candidate array: {key}")
        return result

    magnitude = np.abs(flat)
    if sample_positions is None:
        positions = _selector_positions(data)
    else:
        if (
            not isinstance(sample_positions, list)
            or any(type(index) is not int or index < 0 or index >= count for index in sample_positions)
            or len(sample_positions) != len(set(sample_positions))
        ):
            raise ValueError(f"invalid immutable reference selector: {key}")
        positions = list(sample_positions)
    samples = []
    for index in positions:
        coordinate = [int(item) for item in np.unravel_index(index, data.shape)] if data.shape else []
        item = flat[index]
        samples.append([index, coordinate, [int(item.real) if coordinate_index else finite_number(item.real), finite_number(item.imag) if complex_value else None]])
    real = flat.real
    imag = flat.imag if complex_value else None
    stats = [
        count,
        count,
        int(np.count_nonzero(flat)),
        finite_number(np.linalg.norm(magnitude.astype(np.float64))),
        finite_number(np.max(magnitude)) if count else 0.0,
        finite_number(np.min(real)) if count else 0.0,
        finite_number(np.max(real)) if count else 0.0,
        finite_number(np.min(imag)) if complex_value and count else 0.0,
        finite_number(np.max(imag)) if complex_value and count else 0.0,
    ]
    return [key, dtype, [int(item) for item in data.shape], complex_value, stats, samples]


def resolve_case(
    bundle: LoadedBundle,
    *,
    profile_id: str,
    case: str,
    manifest: dict,
    runtime: dict,
    consumer_line: int,
) -> ResolvedCase:
    """Resolve exactly one reviewed binding with no profile/case fallback."""
    if profile_id not in bundle.profiles or not SAFE_ID.fullmatch(case):
        raise ValueError("unknown or unsafe profile/case")
    if manifest != bundle.profiles[profile_id]:
        raise ValueError("current manifest differs from reviewed profile")
    _validate_runtime(runtime)
    if type(consumer_line) is not int or consumer_line <= 0:
        raise ValueError("consumer line differs")
    matches = [
        binding
        for binding in bundle.side["runtime_bindings"]
        if binding["profile"] == profile_id
        and binding["case"] == case
        and binding["runtime"] == runtime
        and binding["test_line"] == consumer_line
    ]
    if len(matches) != 1:
        raise ValueError("profile/case/runtime must resolve exactly one binding")
    capture = bundle.fixture["captures"].get(f"{profile_id}:{case}")
    if capture is None:
        raise ValueError("resolved capture is missing")
    binding = matches[0]
    if binding["capture_id"] != f"{profile_id}:{case}" or binding["probe_projection_sha256"] != capture["probe_projection"]["sha256"] or capture["probe_projection"]["sha256"] != projection_sha256(capture["profile"], capture["case"], capture["archive_schema_version"], capture["arrays"]):
        raise ValueError("resolved capture projection differs")
    return ResolvedCase(profile_id, case, manifest, dict(runtime), consumer_line, capture)


def _capture_strategies(capture: dict) -> set[str]:
    strategies = set()
    known = {*STRATEGY_TOLERANCE_MODEL, "Dummy"}
    for record in capture["arrays"]:
        for segment in record[0].split("/"):
            if "-" in segment and segment.split("-", 1)[1] in known:
                strategies.add(segment.split("-", 1)[1])
    if not strategies:
        raise ValueError("capture has no material strategy evidence")
    return strategies


def _checked_tolerance(value, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"rtol", "atol"}:
        raise ValueError(f"current manifest {label} tolerance is invalid")
    result = {name: float(value[name]) for name in ("rtol", "atol")}
    if any(not math.isfinite(item) or item < 0 for item in result.values()):
        raise ValueError(f"current manifest {label} tolerance is invalid")
    return result


def _manifest_tolerance(resolved: ResolvedCase, key: str, dtype: str) -> dict:
    """Select only from the caller's exactly reviewed current manifest."""
    if dtype.startswith(("bool", "int", "uint")):
        return {"rtol": 0.0, "atol": 0.0, "scope": "exact/integer"}
    tree = resolved.manifest.get("tolerances", {}).get("torch", {})
    if "/source_aux/" in key:
        source = tree.get("source_auxiliary", {}).get(resolved.case, {}).get(dtype)
        if source is not None:
            return {
                **_checked_tolerance(source, "source/auxiliary"),
                "scope": f"source_auxiliary/{resolved.case}/{dtype}",
            }
    selected = _capture_strategies(resolved.capture)
    for segment in key.split("/"):
        if "-" in segment and segment.split("-", 1)[1] in selected:
            selected = {segment.split("-", 1)[1]}
            break
    models = {
        STRATEGY_TOLERANCE_MODEL[strategy]
        for strategy in selected
        if strategy != "Dummy"
    }
    scope = None
    parts = key.split("/")
    if not models:
        dummy_dynamic = (
            selected == {"Dummy"}
            and resolved.case == "dummy"
            and len(parts) > 3
            and parts[0] == "step"
            and parts[1].isdigit()
            and parts[2] in {"field", "physical"}
        )
        if not dummy_dynamic:
            return {"rtol": 0.0, "atol": 0.0, "scope": "exact/dummy"}
        models = {"dielectric"}
        scope = f"dummy-source-numerics/dielectric/{dtype}"
    tolerances = []
    for model in sorted(models):
        model_dtype = "float64" if model == "dm2" and dtype == "complex128" else dtype
        try:
            tolerances.append(_checked_tolerance(tree[model][model_dtype], model))
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"current manifest has no pinned {model_dtype} tolerance for {model}"
            ) from error
    result = {
        name: max(tolerance[name] for tolerance in tolerances)
        for name in ("rtol", "atol")
    }
    result["scope"] = scope or f"strategies/{','.join(sorted(models))}/{dtype}"
    return result


def comparison_storage_dtype(resolved: ResolvedCase, key: str, reference_dtype: str) -> str:
    """Return the V5.4 candidate storage dtype; never cast or infer loosely."""
    family = validate_storage_reference_dtype(key, reference_dtype)
    precision = resolved.runtime["precision"]
    if precision not in {"float32", "float64"}:
        raise ValueError("runtime precision differs")
    if family in {"map", "coordinate_index", "source_indices", "source_aux_material_indices", "state_indices"}:
        return reference_dtype
    if family == "primary_field":
        return ("complex64" if precision == "float32" else "complex128") if reference_dtype == "complex128" else precision
    if family == "physical_spectrum":
        return precision
    if family in {"physical_summary", "primary_time", "source_aux"}:
        return "float64"
    if family in {"source_aux_material_values", "state_values"}:
        return "complex128"
    raise ValueError("V5.4 storage-dtype family differs")


def _comparison_tolerance_dtype(resolved: ResolvedCase, key: str, dtype: str) -> str:
    if dtype.startswith(("bool", "int", "uint")):
        return dtype
    if "/source_aux/" in key or "/source_aux_material/" in key:
        return "complex128" if dtype.startswith("complex") else "float64"
    return resolved.runtime["precision"]


def _preflight_npz_bytes(candidate: bytes) -> None:
    if type(candidate) is not bytes or len(candidate) > MAX_NPZ_BYTES:
        raise ValueError("candidate NPZ byte bound differs")
    try:
        with zipfile.ZipFile(io.BytesIO(candidate)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_NPZ_MEMBERS or len(infos) != len({item.filename for item in infos}):
                raise ValueError("candidate NPZ member bound/uniqueness differs")
            total = 0
            for info in infos:
                pure = PurePosixPath(info.filename)
                if (
                    pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or not info.filename.endswith(".npy")
                    or info.file_size > MAX_NPY_PAYLOAD_BYTES + MAX_NPY_HEADER_BYTES
                ):
                    raise ValueError("candidate NPZ member path/size differs")
                with archive.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(
                            member, max_header_size=MAX_NPY_HEADER_BYTES
                        )
                    elif version in {(2, 0), (3, 0)}:
                        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(
                            member, max_header_size=MAX_NPY_HEADER_BYTES
                        )
                    else:
                        raise ValueError("candidate NPY version differs")
                    if dtype.hasobject:
                        raise ValueError("candidate NPZ object dtype is forbidden")
                    payload = math.prod(shape) * dtype.itemsize
                    if payload > MAX_NPY_PAYLOAD_BYTES:
                        raise ValueError("candidate NPY payload bound differs")
                    total += payload
                    if total > MAX_TOTAL_ARRAY_BYTES:
                        raise ValueError("candidate total allocation bound differs")
    except (EOFError, OSError, zipfile.BadZipFile) as error:
        raise ValueError("candidate is not a valid bounded NPZ") from error


def _read_candidate_snapshot(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_NPZ_BYTES:
        raise ValueError("candidate NPZ byte bound differs")
    snapshot = path.read_bytes()
    _preflight_npz_bytes(snapshot)
    return snapshot


def _preflight_npz(path: Path) -> None:
    """Retain the path preflight API while validating one immutable snapshot."""
    _read_candidate_snapshot(path)


def nonzero_count_is_exact(dtype: str) -> bool:
    """Only reviewed integral/topology descriptors make nonzero count exact."""
    return isinstance(dtype, str) and dtype.startswith(("bool", "int", "uint"))


def _tolerance_record(resolved, key, dtype, tolerance, label, error, limit, diagnostic=False):
    return {
        "key": key,
        "profile": resolved.profile_id,
        "case": resolved.case,
        "runtime": dict(resolved.runtime),
        "dtype": dtype,
        "scope": tolerance["scope"],
        "rtol": tolerance["rtol"],
        "atol": tolerance["atol"],
        "label": label,
        "absolute_error": float(error),
        "limit": None if limit is None else float(limit),
        "diagnostic": diagnostic,
    }


def _candidate_projection(archive, expected: dict) -> dict:
    """Filter only paired direct-source payloads; reject all other unknowns."""
    observed, source_indices, opaque_values = {}, set(), set()
    for key in sorted(name for name in archive.files if name != "metadata.json" and not name.startswith("torch/")):
        classification = projection_class(key)
        if classification is None:
            raise ValueError(f"unclassified candidate semantic key: {key}")
        if classification == "opaque_source_value":
            opaque_values.add(key.removesuffix("/values"))
            continue
        reference = expected.get(key)
        positions = None
        if reference is not None and not reference[1].startswith(("bool", "int", "uint")):
            positions = _reference_selector_positions(reference)
        observed[key] = probe_array(key, archive[key], sample_positions=positions)
        if classification == "source_indices":
            source_indices.add(key.removesuffix("/indices"))
    if source_indices != opaque_values:
        raise ValueError("candidate direct-source opaque/value topology differs")
    return observed


def compare_candidate_bytes(candidate: bytes, resolved: ResolvedCase) -> dict:
    """Compare one immutable bounded NPZ snapshot against a resolved capture."""
    if type(candidate) is not bytes or len(candidate) > MAX_NPZ_BYTES:
        return {"passed": False, "failures": [{"key": "candidate/archive", "error": "candidate immutable byte bound differs"}], "tolerance_results": []}
    expected = {record[0]: record for record in resolved.capture["arrays"]}
    try:
        _preflight_npz_bytes(candidate)
        with np.load(io.BytesIO(candidate), allow_pickle=False) as archive:
            observed = _candidate_projection(archive, expected)
    except (MemoryError, OSError, ValueError, zipfile.BadZipFile) as error:
        return {"passed": False, "failures": [{"key": "candidate/archive", "error": str(error)}], "tolerance_results": []}
    if set(observed) != set(expected):
        return {"passed": False, "failures": [{"key": "candidate/topology", "error": "missing or extra non-torch arrays"}], "tolerance_results": []}
    failures, records = [], []
    for key in sorted(expected):
        actual, reference = observed[key], expected[key]
        expected_dtype = comparison_storage_dtype(resolved, key, reference[1])
        if actual[1] != expected_dtype or actual[2:4] != reference[2:4] or actual[4][:2] != reference[4][:2]:
            failures.append({"key": key, "error": "dtype/shape/complex/count contract differs"})
            continue
        tolerance_dtype = _comparison_tolerance_dtype(resolved, key, expected_dtype)
        tolerance = _manifest_tolerance(resolved, key, tolerance_dtype)
        rtol, atol = tolerance["rtol"], tolerance["atol"]
        integer = nonzero_count_is_exact(expected_dtype)
        nonzero_error = abs(actual[4][2] - reference[4][2])
        structural = integer
        records.append(_tolerance_record(resolved, key, tolerance_dtype, tolerance, "nonzero_count", nonzero_error, 0.0 if structural else None, not structural))
        if structural and nonzero_error:
            failures.append({"key": key, "error": "nonzero_count differs"})
        for label, got, want in zip(("l2", "max_abs", "real_min", "real_max", "imag_min", "imag_max"), actual[4][3:], reference[4][3:], strict=True):
            delta = abs(got - want)
            stat_limit = 0.0 if integer else (atol * math.sqrt(reference[4][0]) + rtol * abs(want) if label == "l2" else atol + rtol * abs(want))
            records.append(_tolerance_record(resolved, key, tolerance_dtype, tolerance, label, delta, stat_limit))
            if delta > stat_limit:
                failures.append({"key": key, "error": f"{label} differs"})
        if len(actual[5]) != len(reference[5]):
            failures.append({"key": key, "error": "sample-list length differs"})
            continue
        for ordinal, (got, want) in enumerate(zip(actual[5], reference[5], strict=True)):
            if got[:2] != want[:2]:
                failures.append({"key": key, "error": "sample index/coordinate differs"})
                continue
            for part, g, w in zip(("real", "imag"), got[2], want[2], strict=True):
                if w is None:
                    if g is not None:
                        failures.append({"key": key, "error": f"sample {part} null sentinel differs"})
                    continue
                sample_limit = 0.0 if integer else atol + rtol * abs(w)
                delta = abs(g - w)
                records.append(_tolerance_record(resolved, key, tolerance_dtype, tolerance, f"sample[{ordinal}]/{part}", delta, sample_limit))
                if delta > sample_limit:
                    failures.append({"key": key, "error": f"sample {part} differs"})
    return {"passed": not failures, "failures": failures, "tolerance_results": records}


def compare_candidate(candidate: str | Path, resolved: ResolvedCase) -> dict:
    """Read one bounded path once, then compare its immutable byte snapshot."""
    path = Path(candidate)
    try:
        snapshot = _read_candidate_snapshot(path)
    except (MemoryError, OSError, ValueError, zipfile.BadZipFile) as error:
        return {"passed": False, "failures": [{"key": "candidate/archive", "error": str(error)}], "tolerance_results": []}
    return compare_candidate_bytes(snapshot, resolved)
