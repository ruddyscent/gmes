#!/usr/bin/env python3
"""Self-contained V5.4 pre-cutover native numeric-probe bundle generator."""

from __future__ import annotations

import argparse, ast, copy, gzip, hashlib, json, math, os, re, shutil, subprocess, sys
from pathlib import Path

import numpy as np


COMMIT = "66a0a1aa8d6f163134967e8b8a7e9dc46530717b"
REPOSITORY = "ruddyscent/gmes"
OBSERVER_TAG = "native-oracle-observer-v6"
OBSERVER_OBJECT = "440a6c262b0344a051b8f90c7b07734f3af750a4"
OBSERVER_COMMIT = "2d5810cebf610fa6384235d9771f4ac699c23fc5"
CONSUMER_SOURCE_SHA256 = "9d14d911f4aa609fa2377cdc5e551c5b89b5b2827e81dad83f064a57ac1baed4"
FIXTURE_SCHEMA_VERSION = 8
BUNDLE_SCHEMA_VERSION = 3
FIXTURE_NAME = "pre-cutover-native-numeric-probes-v5-4.json.gz"
FIXTURE_CHECKSUM_NAME = "pre-cutover-native-numeric-probes-v5-4.sha256"
V5_3_FIXTURE_NAME = "pre-cutover-native-numeric-probes-v5-3.json.gz"
V5_3_FIXTURE_CHECKSUM_NAME = "pre-cutover-native-numeric-probes-v5-3.sha256"
LEGACY_FIXTURE_NAME = "pre-cutover-native-numeric-probes-v5-2.json.gz"
LEGACY_FIXTURE_CHECKSUM_NAME = "pre-cutover-native-numeric-probes-v5-2.sha256"
PROJECTION_ALGORITHM = "validated-probe-projection-v5-storage-dtype-contract"
PROJECTION_PREFIX = b"gmes-issue124-probe-projection-v5\0"
V5_3_PROJECTION_ALGORITHM = "validated-probe-projection-v4-reference-sampled"
V5_3_PROJECTION_PREFIX = b"gmes-issue124-probe-projection-v4\0"
LEGACY_PROJECTION_ALGORITHM = "validated-probe-projection-v3-source-values-omitted"
LEGACY_PROJECTION_PREFIX = b"gmes-issue124-probe-projection-v3\0"
SAMPLING_ALGORITHM = "reference-selector-candidate-fixed-positions-v3"
TOLERANCES_NUMERIC_SHA256 = "208b8816eafd97b8a3a295bdff0c68db8b2a6cccc47ffc3301fcc895cbeb4c3f"
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
FORBIDDEN = re.compile(r"(?:@|(?:github_pat_|gh[opsu]_)[A-Za-z0-9_]{8,}|(?i:password|secret|token|hostname|host_identity|environment|command|/home/|\\\\Users\\\\))")
LONG = [1, 2, 5, 20, 100]
INITIAL = ("dummy", "upml", "drude-4", "lorentz-4", "dcp-ade", "dcp-plrc-bloch", "dcp-rc", "dm2-1", "cpml-bloch", "mixed-2d", "overlapping-sources", "tfsf-transparent", "gaussian-auxiliary")


def sha_path(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _projection_sha256(prefix: bytes, profile: str, case: str, archive_schema_version: int, arrays: list) -> str:
    projection = {
        "profile": profile,
        "case": case,
        "archive_schema_version": archive_schema_version,
        "arrays": arrays,
    }
    encoded = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(prefix + encoded).hexdigest()


def projection_sha256(profile: str, case: str, archive_schema_version: int, arrays: list) -> str:
    """Hash the V5.4 strict-storage-dtype semantic probe projection."""
    return _projection_sha256(PROJECTION_PREFIX, profile, case, archive_schema_version, arrays)


def projection_class(key: str) -> str | None:
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
    """Widen only reviewed coordinate-index semantics to contiguous int64."""
    data = np.asarray(value)
    if coordinate_index_family(key) is None:
        return data
    if (
        data.dtype.kind != "i"
        or data.dtype.itemsize not in {1, 2, 4, 8}
        or data.dtype.byteorder not in {"=", "|"}
        or data.ndim != 2
        or data.shape[1:] != (3,)
    ):
        raise ValueError(f"coordinate-index projection contract differs: {key}")
    canonical = np.ascontiguousarray(data, dtype=np.int64)
    if canonical.shape != data.shape or not np.array_equal(canonical, data):
        raise ValueError(f"coordinate-index projection conversion differs: {key}")
    return canonical


def fnum(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric data is not allowed in V5")
    return value


def array_probe(key: str, raw: np.ndarray) -> list:
    """Topology, aggregates, and mandatory deterministic/extrema sentinels."""
    classification = projection_class(key)
    if classification is None or classification == "opaque_source_value":
        raise ValueError(f"unprojectable semantic array key: {key!r}")
    data = canonical_semantic_projection_array(key, raw)
    coordinate_index = coordinate_index_family(key) is not None
    dtype = str(data.dtype)
    if not SAFE_DTYPE.fullmatch(dtype):
        raise ValueError(f"unsafe dtype: {dtype!r}")
    flat, count = data.reshape(-1), int(data.size)
    complex_value = bool(np.iscomplexobj(flat))
    finite = np.isfinite(flat)
    if not bool(np.all(finite)):
        # Fail rather than omitting a non-finite sentinel from a privacy-safe fixture.
        raise ValueError(f"non-finite source array: {key}")
    magnitude = np.abs(flat)
    first_nonzero = int(np.flatnonzero(flat)[0]) if np.any(flat != 0) else None
    maxabs = int(np.argmax(magnitude)) if count else None
    positions = []
    for value in (0, count - 1, count // 2, first_nonzero, maxabs):
        if value is not None and 0 <= value < count and value not in positions:
            positions.append(value)
    for part in range(8):
        value = (count - 1) * part // 7 if count else None
        if value is not None and value not in positions:
            positions.append(value)
    samples = []
    for index in positions:
        coordinate = [int(item) for item in np.unravel_index(index, data.shape)] if data.shape else []
        value = flat[index]
        samples.append([index, coordinate, [int(value.real) if coordinate_index else fnum(value.real), fnum(value.imag) if complex_value else None]])
    real = flat.real
    imag = flat.imag if complex_value else None
    stats = [count, count, int(np.count_nonzero(flat)), fnum(np.linalg.norm(magnitude.astype(np.float64))), fnum(np.max(magnitude)) if count else 0.0, fnum(np.min(real)) if count else 0.0, fnum(np.max(real)) if count else 0.0, fnum(np.min(imag)) if complex_value and count else 0.0, fnum(np.max(imag)) if complex_value and count else 0.0]
    return [key, dtype, [int(item) for item in data.shape], complex_value, stats, samples]


def numeric_tree(value):
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)) and not isinstance(value, bool): return fnum(value)
    if isinstance(value, list): return [numeric_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: numeric_tree(item) for key, item in sorted(value.items()) if re.fullmatch(r"[A-Za-z0-9_.+-]+", key) and not isinstance(item, str)}
    return None


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=checkout, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()


def small(base, native_oracle, cases):
    result = copy.deepcopy(base); result["reference"]["capture_steps"] = [1]
    def alter(case):
        spec = copy.deepcopy(native_oracle.find_case(result, case)); spec["capture_steps"] = [1]
        if spec["recipe"] != "mixed": spec.update(size=[2, 2, 2], resolution=2)
        return spec
    result["correctness"] = [alter(case) for case in cases]; result["physical_checks"] = []
    return result


def long(base, native_oracle, case):
    result = small(base, native_oracle, (case,)); result["reference"]["capture_steps"] = LONG; result["correctness"][0]["capture_steps"] = LONG
    return result


def case_capture_steps(profile, case):
    """Return the exact reviewed schedule for one selected profile case."""
    matches = [
        spec
        for section in ("correctness", "physical_checks")
        for spec in profile[section]
        if spec.get("name") == case
    ]
    if len(matches) != 1:
        raise ValueError("profile case schedule is ambiguous")
    steps = matches[0].get("capture_steps", profile["reference"].get("capture_steps"))
    if (
        not isinstance(steps, list)
        or not steps
        or any(type(step) is not int or step <= 0 for step in steps)
        or steps != sorted(set(steps))
    ):
        raise ValueError("profile case capture schedule is invalid")
    return list(steps)


def _seed_coordinate_sample(value) -> int:
    if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("approved V5 coordinate-index sample is not finite")
    result = int(value)
    if value != result or not -(2**63) <= result < 2**63:
        raise ValueError("approved V5 coordinate-index sample is not exact int64")
    return result


def canonicalize_seed_probe(probe: list) -> tuple[list, bool, bool]:
    """Migrate an approved V5 probe descriptor without rebuilding raw arrays."""
    if not isinstance(probe, list) or len(probe) != 6 or not isinstance(probe[0], str):
        raise ValueError("approved V5 probe topology differs")
    key, dtype, shape, complex_flag, _stats, samples = probe
    family = coordinate_index_family(key)
    if family is not None:
        if (
            dtype != "int64"
            or not isinstance(shape, list)
            or len(shape) != 2
            or shape[-1:] != [3]
            or complex_flag is not False
            or not isinstance(samples, list)
            or any(
                not isinstance(sample, list)
                or len(sample) != 3
                or not isinstance(sample[2], list)
                or len(sample[2]) != 2
                or sample[2][1] is not None
                for sample in samples
            )
        ):
            raise ValueError(f"approved V5 coordinate-index descriptor differs: {key}")
        canonical = copy.deepcopy(probe)
        canonical[1] = "int64"
        for sample in canonical[5]:
            sample[2][0] = _seed_coordinate_sample(sample[2][0])
        return canonical, True, False
    if key.endswith("/indices"):
        raise ValueError(f"unreviewed coordinate-index path: {key}")
    if MAP_ID_PATH.fullmatch(key):
        if dtype != "int32":
            raise ValueError(f"map-ID descriptor is not strict int32: {key}")
        return probe, False, True
    return probe, False, False


def validate_storage_reference_dtype(key: str, dtype: str) -> str:
    """Accept only the reviewed V5.4 descriptor storage-dtype table."""
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
    if dtype not in allowed:
        raise ValueError(f"V5.4 reference storage dtype differs: {key}")
    return family


def reference_value_selected_positions(probe: list) -> list[int]:
    """Return the retained non-base/non-grid reference selector positions."""
    count = probe[4][0]
    base = []
    for index in (0, count - 1, count // 2):
        if count and index not in base:
            base.append(index)
    grid = {(count - 1) * part // 7 for part in range(8)} if count else set()
    fixed = set(base) | grid
    return [sample[0] for sample in probe[5] if sample[0] not in fixed]


def _seed_projection_digest(capture: dict) -> str:
    projection = capture.get("probe_projection")
    if not isinstance(projection, dict) or set(projection) != {"algorithm", "sha256"}:
        raise ValueError("approved V5 projection descriptor differs")
    prefixes = {
        PROJECTION_ALGORITHM: PROJECTION_PREFIX,
        V5_3_PROJECTION_ALGORITHM: V5_3_PROJECTION_PREFIX,
        LEGACY_PROJECTION_ALGORITHM: LEGACY_PROJECTION_PREFIX,
    }
    try:
        prefix = prefixes[projection["algorithm"]]
    except KeyError as error:
        raise ValueError("approved V5 projection algorithm differs") from error
    return _projection_sha256(
        prefix,
        capture["profile"],
        capture["case"],
        capture["archive_schema_version"],
        capture["arrays"],
    )


def consumers():
    rows = [[234, "small-initial", case, "cpu", "float64", "eager", "default"] for case in INITIAL]
    rows += [[253,"long-tfsf","tfsf-transparent","cpu","float32","eager","default"],[338,"small-tfsf","tfsf-transparent","cpu","float32","eager","default"],[458,"long-gaussian","gaussian-auxiliary","cpu","float32","eager","default"],[513,"small-gaussian","gaussian-auxiliary","cpu","float32","eager","default"],[562,"long-dummy","dummy","cpu","float64","graph","default"]]
    rows += [[line,"small-dcp","dcp-plrc-bloch","cpu","float64","eager","default"] for line in (615,659,671,697,779,1052,1085,1179,1412,1485,1509)]
    rows += [[712,"small-mixed","mixed-2d","cpu","float64","eager","default"],[843,"small-drude1","drude-1","cpu","float64","eager","default"]]
    rows += [[line,"small-tfsf","tfsf-transparent","cpu","float64","eager","default"] for line in (922,1008,1227)]
    rows += [[1292,"small-main-fields",case,"cpu","float64","eager","default"] for case in ("stability-energy-dielectric","dcp-plrc-bloch")]
    rows += [[1371,"small-material-matrix",case,"cpu","float64","eager","default"] for case in ("upml","drude-4","lorentz-4","dcp-ade","dcp-plrc-bloch","dcp-rc","dm2-1","tfsf-transparent")]
    # H1: omitted precision at 1426 and five stability sites is float64.
    rows += [[1426,"small-gaussian","gaussian-auxiliary","cpu","float64","eager","default"]]
    rows += [[line,"small-stability","stability-energy-dielectric","cpu",("float32" if line in (1523,1812) else "float64"),"eager","default"] for line in (1523,1753,1812,1930,1975,1990,2012)]
    rows += [[1847,"small-index-matrix",case,"cpu","float64","eager","default"] for case in ("dcp-plrc-bloch","stability-energy-dielectric")]
    # H1/L1: one native reference producer, two concrete candidate contracts.
    rows += [[2190,"canonical-dm2","ziolkowski-dm2","cuda:0","float32","eager","default"],[2190,"canonical-dm2","ziolkowski-dm2","cuda:0","float32","graph","reduce-overhead"]]
    return rows


def validate_ast(checkout: Path, rows) -> None:
    source = (checkout / "tests/test_torch_correctness.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != CONSUMER_SOURCE_SHA256:
        raise ValueError("reviewed consumer source digest differs")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_capture_pair"]
    if len(calls) != 33: raise ValueError("expected exactly 33 shared helper call sites")
    by_line = {node.lineno: node for node in calls}
    for line, _profile, _case, _device, precision, _graph, _mode in rows:
        if line == 2190: continue
        node = by_line.get(line)
        if node is None: raise ValueError(f"consumer line drift: {line}")
        actual = next((ast.literal_eval(item.value) for item in node.keywords if item.arg == "precision"), "float64")
        if actual != precision: raise ValueError(f"precision mismatch at {line}: {actual!r} != {precision!r}")
    if len(rows) != 56: raise ValueError("expected 56 candidate runtime executions")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkout", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    checkout, output = args.checkout.resolve(), args.output.resolve()
    if git(checkout,"rev-parse","HEAD") != COMMIT: raise ValueError("checkout HEAD mismatch")
    changed = git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "gmes",
        "benchmarks",
    ).splitlines()
    if changed: raise ValueError("capture source tree is dirty")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in ("generator.py","historical_probe_loader.py","package_integration_assertions.py","self_test.py","README.md","REPRODUCTION.md","schema.json"):
        source, target = Path(__file__).resolve().parent / name, output / name
        if source.resolve() != target.resolve(): shutil.copy2(source, target)
    sys.path.insert(0, str(checkout)); import benchmarks.native_oracle as native_oracle
    manifest_bytes = subprocess.run(["git","show",f"{COMMIT}:benchmarks/native_oracle_workloads.json"],cwd=checkout,check=True,stdout=subprocess.PIPE).stdout
    base = json.loads(manifest_bytes)
    if base["reference"]["observer_tag"] != OBSERVER_TAG or base["reference"]["observer_commit"] != OBSERVER_COMMIT: raise ValueError("historical observer identity mismatch")
    if git(checkout,"rev-parse",f"refs/tags/{OBSERVER_TAG}") != OBSERVER_OBJECT or git(checkout,"rev-parse",f"{OBSERVER_TAG}^{{commit}}") != OBSERVER_COMMIT: raise ValueError("observer tag identity mismatch")
    profiles_raw = {"small-initial":small(base,native_oracle,INITIAL),"small-dcp":small(base,native_oracle,("dcp-plrc-bloch",)),"small-tfsf":small(base,native_oracle,("tfsf-transparent",)),"small-gaussian":small(base,native_oracle,("gaussian-auxiliary",)),"small-mixed":small(base,native_oracle,("mixed-2d",)),"small-drude1":small(base,native_oracle,("drude-1",)),"small-stability":small(base,native_oracle,("stability-energy-dielectric",)),"small-main-fields":small(base,native_oracle,("stability-energy-dielectric","dcp-plrc-bloch")),"small-material-matrix":small(base,native_oracle,("upml","drude-4","lorentz-4","dcp-ade","dcp-plrc-bloch","dcp-rc","dm2-1","tfsf-transparent")),"small-index-matrix":small(base,native_oracle,("dcp-plrc-bloch","stability-energy-dielectric")),"long-tfsf":long(base,native_oracle,"tfsf-transparent"),"long-gaussian":long(base,native_oracle,"gaussian-auxiliary"),"long-dummy":long(base,native_oracle,"dummy"),"canonical-dm2":base}
    profiles_dir, inputs_dir = output / "profiles", output / "inputs"; profiles_dir.mkdir(exist_ok=True); inputs_dir.mkdir(exist_ok=True)
    (inputs_dir / "native_oracle_workloads.json").write_bytes(manifest_bytes)
    aliases, profiles = {}, {}
    for alias, profile in profiles_raw.items():
        data = manifest_bytes if alias == "canonical-dm2" else canonical(profile); identifier = f"{alias}-{hashlib.sha256(data).hexdigest()[:12]}"; path = profiles_dir / f"{identifier}.json"; path.write_bytes(data); aliases[alias] = identifier; profiles[identifier] = {"file":f"profiles/{path.name}","bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    rows = consumers(); validate_ast(checkout, rows)
    bindings = [
        {
            "test_line": line,
            "profile": aliases[alias],
            "profile_sha256": profiles[aliases[alias]]["sha256"],
            "case": case,
            "capture_id": f"{aliases[alias]}:{case}",
            "capture_steps": case_capture_steps(profiles_raw[alias], case),
            "runtime": {
                "device": device,
                "precision": precision,
                "graph_mode": graph,
                "compile_mode": mode,
            },
        }
        for line, alias, case, device, precision, graph, mode in rows
    ]
    # V5.4 changes only the external comparison dtype contract. Reuse the
    # approved semantic projection rather than recapturing native NPZ input.
    seed_path = next(
        (output / name for name in (FIXTURE_NAME, V5_3_FIXTURE_NAME, LEGACY_FIXTURE_NAME) if (output / name).is_file()),
        None,
    )
    if seed_path is None:
        raise ValueError("approved V5 projection seed is missing")
    seed = json.loads(gzip.decompress(seed_path.read_bytes()))
    captures = seed.get("captures")
    expected_capture_ids = {binding["capture_id"] for binding in bindings}
    if not isinstance(captures, dict) or set(captures) != expected_capture_ids:
        raise ValueError("approved V5 capture coverage differs")
    seed_schema = seed.get("fixture_schema_version")
    if seed_schema == 6:
        seed_algorithm = LEGACY_PROJECTION_ALGORITHM
        seed_array_count = 4377
        expected_opaque_count = 0
    elif seed_schema == 7:
        seed_algorithm = V5_3_PROJECTION_ALGORITHM
        seed_array_count = 4377
        expected_opaque_count = 0
    elif seed_schema == 8:
        seed_algorithm = PROJECTION_ALGORITHM
        seed_array_count = 4377
        expected_opaque_count = 0
    else:
        raise ValueError("approved V5 seed schema differs")
    if sum(len(capture.get("arrays", ())) for capture in captures.values()) != seed_array_count:
        raise ValueError("approved V5 seed probe count differs")
    coordinate_count = map_id_count = opaque_count = 0
    numeric_count = dynamic_record_count = dynamic_position_count = 0
    family_counts = {name: 0 for name in EXPECTED_FAMILY_COUNTS}
    for capture_id, capture in captures.items():
        profile_id, case = capture_id.split(":", 1)
        if (
            not isinstance(capture, dict)
            or set(capture) != {"profile", "case", "archive_schema_version", "probe_projection", "arrays"}
            or capture["profile"] != profile_id
            or capture["case"] != case
            or capture["probe_projection"].get("algorithm") != seed_algorithm
            or capture["probe_projection"].get("sha256") != _seed_projection_digest(capture)
        ):
            raise ValueError("approved V5 semantic projection differs")
        arrays, retained_source_indices, opaque_source_values = [], set(), set()
        for probe in capture["arrays"]:
            if not isinstance(probe, list) or len(probe) != 6 or not isinstance(probe[0], str):
                raise ValueError("approved V5.1 probe topology differs")
            key = probe[0]
            classification = projection_class(key)
            if classification is None:
                raise ValueError(f"unclassified V5.1 semantic key: {key}")
            if classification == "opaque_source_value":
                raise ValueError("approved V5.2/V5.3 seed serializes opaque direct-source values")
            canonical_probe, coordinate, map_id = canonicalize_seed_probe(probe)
            validate_storage_reference_dtype(canonical_probe[0], canonical_probe[1])
            arrays.append(canonical_probe)
            coordinate_count += coordinate
            map_id_count += map_id
            family_counts[classification] += 1
            if not canonical_probe[1].startswith(("bool", "int", "uint")):
                numeric_count += 1
                dynamic_positions = reference_value_selected_positions(canonical_probe)
                dynamic_record_count += bool(dynamic_positions)
                dynamic_position_count += len(dynamic_positions)
            if classification == "source_indices":
                retained_source_indices.add(key.removesuffix("/indices"))
        if opaque_source_values:
            raise ValueError("opaque direct-source seed topology differs")
        capture["arrays"] = arrays
        capture["probe_projection"] = {
            "algorithm": PROJECTION_ALGORITHM,
            "sha256": projection_sha256(profile_id, case, capture["archive_schema_version"], arrays),
        }
    if (
        family_counts != EXPECTED_FAMILY_COUNTS
        or opaque_count != expected_opaque_count
        or (coordinate_count, map_id_count) != (1393, 420)
        or (numeric_count, dynamic_record_count, dynamic_position_count) != (2564, 729, 771)
    ):
        raise ValueError("approved V5 index-path inventory differs")
    for binding in bindings:
        binding["probe_projection_sha256"] = captures[binding["capture_id"]]["probe_projection"]["sha256"]
    provenance={"repository":REPOSITORY,"capture_source_commit":COMMIT,"native_oracle_sha256":sha_path(checkout/"benchmarks/native_oracle.py"),"historical_observer":{"tag":OBSERVER_TAG,"tag_object":OBSERVER_OBJECT,"peeled_commit":OBSERVER_COMMIT,"role":"historical-manifest-reference-not-capture-source"},"source_manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),"generator_sha256":sha_path(Path(__file__).resolve())}
    tolerances_numeric = numeric_tree(base["tolerances"])
    if hashlib.sha256(canonical(tolerances_numeric)).hexdigest() != TOLERANCES_NUMERIC_SHA256:
        raise ValueError("pinned floating tolerance tree differs")
    fixture={"fixture_schema_version":FIXTURE_SCHEMA_VERSION,"kind":"pre-cutover-native-numeric-probes","provenance":provenance,"profiles":profiles,"runtime_bindings":bindings,"sampling":{"algorithm":SAMPLING_ALGORITHM,"max_samples_per_array":13,"limits":"sampled-and-aggregate-historical-evidence; not full-array physics equality"},"tolerances_numeric":tolerances_numeric,"captures":dict(sorted(captures.items()))}
    raw=canonical(fixture); gz=gzip.compress(raw,compresslevel=9,mtime=0)
    if len(gz)>5*1024*1024 or len(raw)>5*1024*1024: raise ValueError("fixture size ceiling exceeded")
    fixture_path=output/FIXTURE_NAME; fixture_path.write_bytes(gz)
    for name in (V5_3_FIXTURE_NAME, V5_3_FIXTURE_CHECKSUM_NAME, LEGACY_FIXTURE_NAME, LEGACY_FIXTURE_CHECKSUM_NAME):
        legacy = output / name
        if legacy != fixture_path:
            legacy.unlink(missing_ok=True)
    # self_test.py carries the caller-owned, final bundle anchors.  Describing
    # that mutable anchor carrier in BUNDLE-MANIFEST would make a hash cycle;
    # the remaining support is immutable and covered below.
    support = {}
    for key, name in (("package_assertions","package_integration_assertions.py"),("schema","schema.json"),("readme","README.md"),("reproduction","REPRODUCTION.md")):
        path = output / name
        support[key] = {"file":name,"bytes":path.stat().st_size,"sha256":sha_path(path)}
    side={"bundle_schema_version":BUNDLE_SCHEMA_VERSION,"fixture_schema_version":FIXTURE_SCHEMA_VERSION,"kind":fixture["kind"],"provenance":provenance,"inputs":{"source_manifest":{"file":"inputs/native_oracle_workloads.json","bytes":len(manifest_bytes),"sha256":hashlib.sha256(manifest_bytes).hexdigest()}},"profiles":profiles,"runtime_bindings":bindings,"sampling":fixture["sampling"],"generator":{"file":"generator.py","bytes":(output/"generator.py").stat().st_size,"sha256":sha_path(output/"generator.py"),"loader_file":"historical_probe_loader.py","loader_bytes":(output/"historical_probe_loader.py").stat().st_size,"loader_sha256":sha_path(output/"historical_probe_loader.py")},"support":support,"fixture":{"file":fixture_path.name,"compressed_bytes":len(gz),"compressed_sha256":sha_path(fixture_path),"uncompressed_bytes":len(raw),"uncompressed_sha256":hashlib.sha256(raw).hexdigest()}}
    side_bytes=canonical(side); (output/"BUNDLE-MANIFEST.json").write_bytes(side_bytes); (output/"BUNDLE-MANIFEST.sha256").write_text(f"{hashlib.sha256(side_bytes).hexdigest()}  BUNDLE-MANIFEST.json\n"); (output/FIXTURE_CHECKSUM_NAME).write_text(f"{side['fixture']['compressed_sha256']}  {fixture_path.name}\n")
    trust = {"expected_manifest_sha256": hashlib.sha256(side_bytes).hexdigest(), "expected_manifest_bytes": len(side_bytes), "expected_fixture_sha256": side["fixture"]["compressed_sha256"], "expected_fixture_bytes": len(gz)}
    anchor_path = output / "self_test.py"
    anchored, replacements = re.subn(r"(?m)^TRUST = .*$", "TRUST = " + repr(trust), anchor_path.read_text(), count=1)
    if replacements != 1:
        raise ValueError("self-test trust anchor marker is missing")
    anchor_path.write_text(anchored)
    data_files=[fixture_path,output/"BUNDLE-MANIFEST.json",inputs_dir/"native_oracle_workloads.json",*profiles_dir.glob("*.json")]
    if FORBIDDEN.search(raw.decode()) or any(FORBIDDEN.search(path.read_bytes().decode()) for path in data_files if path.suffix != ".gz"): raise ValueError("privacy scan failed")
    (output/"REPORT.md").write_text(f"# V5.4 fixture candidate\n\n## Verdict: generated; independent self-test and review pending\n\n* source: `{COMMIT}`; native module SHA `{provenance['native_oracle_sha256']}`\n* profiles: {len(profiles)}; captures: {len(captures)}; public arrays: {sum(len(x['arrays']) for x in captures.values())}\n* opaque direct-source values omitted: 145; retained family counts: {family_counts}\n* float/complex descriptors: {numeric_count}; records/positions with non-grid reference selectors: {dynamic_record_count}/{dynamic_position_count}\n* coordinate-index descriptors: {coordinate_count} canonical int64; map IDs: {map_id_count} strict int32\n* compressed/uncompressed bytes: {len(gz)}/{len(raw)}\n* compressed SHA-256: `{side['fixture']['compressed_sha256']}`\n* semantic projection: `{PROJECTION_ALGORITHM}` domain-separated SHA-256; raw serialization identity omitted\n* storage dtype: strict V5.4 grammar; tolerance dtype remains separately pinned\n* raw NPZ: no native recapture; approved V5.3-r2 projection seed only\n* privacy: data-artifact scan passed; raw metadata excluded\n\nRun the locked-interpreter command in REPRODUCTION.md.\n")


if __name__ == "__main__": main()
