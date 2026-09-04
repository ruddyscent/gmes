#!/usr/bin/env python3
"""Produce and bind independently-derived Torch correctness archives."""

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import zipfile
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath

import numpy as np

import gmes
import gmes.torch_fdtd
import gmes.torch_source
from benchmarks import native_oracle
from benchmarks.native_oracle import COMPONENT_NAMES
from gmes.torch_source import TorchPointSourceBatch, TorchTransparentBatch

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "native_oracle_workloads.json"
INDEX_KIND = "torch-correctness-evidence-index"
INDEX_CONTRACT = "complete-field-state-and-runtime-receipt-v2"
PRODUCER = "gmes-torch-correctness-v2"
TORCH_ARRAY_CONTRACT = "torch-plan-state-source-arrays-v2"
RUNTIME_RECEIPT_KIND = "issue123-runtime-publication-receipt"
TRUSTED_MANIFEST_SHA256 = (
    "0766dbf932882dfec7a40abfbcd78eb67978ed8cd65e38625193a16502cc29a9"
)

MAX_CORRECTNESS_JSON_BYTES = 16 * 1024**2
MAX_CORRECTNESS_NPZ_BYTES = 4 * 1024**3
MAX_CORRECTNESS_NPZ_MEMBERS = 16_384
MAX_CORRECTNESS_NPY_HEADER_BYTES = 64 * 1024
MAX_CORRECTNESS_NPY_PAYLOAD_BYTES = 2 * 1024**3
MAX_CORRECTNESS_TOTAL_ARRAY_BYTES = 8 * 1024**3

_ZIP_EOCD = struct.Struct("<4s4H2LH")
_ZIP_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
_ZIP_LOCAL_HEADER = struct.Struct("<4s5H3L2H")

_POINT_SOURCE_LIVE_ARRAYS = (
    "overwrite_targets",
    "overwrite_models",
    "overwrite_parameters",
    "overwrite_amplitudes",
    "_overwrite_values",
    "additive_targets",
    "additive_models",
    "additive_parameters",
    "additive_amplitudes",
    "_additive_values",
)
_PLANNER_COMPONENT_ARRAYS = (
    "material_ids",
    "underlying_ids",
    "ownership",
    "dense_inverse",
    "constant_targets",
    "constant_values",
)
_PLANNER_BUCKET_ARRAYS = (
    "targets",
    "target_region_indices",
    "region_keys",
    "region_coefficient_indices",
    "coefficient_table",
    "cell_coefficients",
    "stencil_indices",
    "tile_origins",
    "tile_region_indices",
)
_PLANNER_CPML_RESIDUAL_ARRAYS = (
    "positions",
    "targets",
    "stencil_indices",
    "parameters",
)

_STRATEGY_MODELS = {
    "Const": "const",
    "Cpml": "cpml",
    "DcpAde": "dcp-ade",
    "DcpPlrc": "dcp-plrc",
    "DcpRc": "dcp-rc",
    "Dielectric": "dielectric",
    "Dm2": "dm2",
    "Drude": "drude",
    "Dummy": "dummy",
    "Lorentz": "lorentz",
    "Upml": "upml",
}


def _hash_open_file(handle):
    digest = hashlib.sha256()
    handle.seek(0)
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256(path):
    with Path(path).open("rb") as handle:
        return _hash_open_file(handle)


def _read_exact(handle, size, label):
    value = handle.read(size)
    if len(value) != size:
        raise ValueError(f"{label} is truncated")
    return value


def _canonical_npz_member_name(raw, flags):
    if flags & ~0x800:
        raise ValueError("NPZ member uses unsupported ZIP flags")
    try:
        value = raw.decode("utf-8" if flags & 0x800 else "ascii")
    except UnicodeDecodeError as error:
        raise ValueError("NPZ member name is not canonical text") from error
    if not value.endswith(".npy"):
        raise ValueError("NPZ contains a non-NPY member")
    key = value[:-4]
    if (
        not key
        or "\\" in key
        or "\x00" in key
        or PurePosixPath(key).is_absolute()
        or PurePosixPath(key).as_posix() != key
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise ValueError("NPZ member name is not canonical")
    return value


def _zip64_local_sizes(extra, uncompressed_size, compressed_size):
    position = 0
    records = []
    while position < len(extra):
        if len(extra) - position < 4:
            raise ValueError("NPZ local ZIP extra field is truncated")
        identifier, size = struct.unpack_from("<HH", extra, position)
        position += 4
        payload = extra[position : position + size]
        if len(payload) != size:
            raise ValueError("NPZ local ZIP extra payload is truncated")
        position += size
        records.append((identifier, payload))
    if len(records) != 1 or records[0][0] != 0x0001 or len(records[0][1]) != 16:
        raise ValueError("NPZ local ZIP extra field is not canonical ZIP64")
    actual_uncompressed, actual_compressed = struct.unpack("<QQ", records[0][1])
    if (actual_uncompressed, actual_compressed) != (
        uncompressed_size,
        compressed_size,
    ):
        raise ValueError("NPZ local ZIP64 sizes differ from the central directory")


def _preflight_npz_file(handle, label):
    descriptor = os.fstat(handle.fileno())
    if not stat.S_ISREG(descriptor.st_mode):
        raise ValueError(f"{label} must be a regular file")
    file_size = descriptor.st_size
    if file_size <= 0 or file_size > MAX_CORRECTNESS_NPZ_BYTES:
        raise ValueError(f"{label} exceeds the bounded NPZ size contract")
    digest = _hash_open_file(handle)

    tail_size = min(file_size, 22 + 65_535)
    handle.seek(file_size - tail_size)
    tail = _read_exact(handle, tail_size, f"{label} ZIP trailer")
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0 or len(tail) - marker < _ZIP_EOCD.size:
        raise ValueError(f"{label} has no complete ZIP end record")
    eocd_offset = file_size - tail_size + marker
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = _ZIP_EOCD.unpack_from(tail, marker)
    if signature != b"PK\x05\x06":
        raise ValueError(f"{label} ZIP end signature differs")
    if (
        disk_number != 0
        or central_disk != 0
        or disk_entries != total_entries
        or total_entries in {0, 0xFFFF}
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
        or total_entries > MAX_CORRECTNESS_NPZ_MEMBERS
    ):
        raise ValueError(f"{label} ZIP topology is unsupported or out of bounds")
    if comment_size != 0 or eocd_offset + _ZIP_EOCD.size != file_size:
        raise ValueError(f"{label} has a ZIP comment or trailing bytes")
    if central_offset + central_size != eocd_offset:
        raise ValueError(f"{label} has a gap around the ZIP central directory")

    handle.seek(central_offset)
    central_records = []
    names = set()
    total_uncompressed = 0
    for _ordinal in range(total_entries):
        raw = _read_exact(handle, _ZIP_CENTRAL_HEADER.size, f"{label} ZIP member")
        (
            signature,
            _version_made,
            _version_needed,
            flags,
            compression,
            _modified_time,
            _modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
            member_comment_size,
            disk_start,
            _internal_attributes,
            _external_attributes,
            local_offset,
        ) = _ZIP_CENTRAL_HEADER.unpack(raw)
        if signature != b"PK\x01\x02":
            raise ValueError(f"{label} central directory is malformed")
        raw_name = _read_exact(handle, name_size, f"{label} ZIP member name")
        extra = _read_exact(handle, extra_size, f"{label} ZIP central extra")
        member_comment = _read_exact(
            handle, member_comment_size, f"{label} ZIP member comment"
        )
        name = _canonical_npz_member_name(raw_name, flags)
        if (
            name in names
            or extra
            or member_comment
            or disk_start != 0
            or compression not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or uncompressed_size > MAX_CORRECTNESS_NPY_PAYLOAD_BYTES
        ):
            raise ValueError(f"{label} ZIP member topology is not canonical")
        names.add(name)
        total_uncompressed += uncompressed_size
        if total_uncompressed > MAX_CORRECTNESS_TOTAL_ARRAY_BYTES:
            raise ValueError(f"{label} exceeds the total array byte bound")
        central_records.append(
            {
                "name": name,
                "raw_name": raw_name,
                "flags": flags,
                "compression": compression,
                "crc": crc,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_offset": local_offset,
            }
        )
    if handle.tell() != central_offset + central_size:
        raise ValueError(f"{label} central directory byte coverage differs")

    ordered = sorted(central_records, key=lambda value: value["local_offset"])
    expected_offset = 0
    for record in ordered:
        if record["local_offset"] != expected_offset:
            raise ValueError(f"{label} has an unindexed or gapped local ZIP record")
        handle.seek(record["local_offset"])
        raw = _read_exact(handle, _ZIP_LOCAL_HEADER.size, f"{label} local ZIP header")
        (
            signature,
            _version_needed,
            flags,
            compression,
            _modified_time,
            _modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
        ) = _ZIP_LOCAL_HEADER.unpack(raw)
        raw_name = _read_exact(handle, name_size, f"{label} local ZIP member name")
        extra = _read_exact(handle, extra_size, f"{label} local ZIP extra")
        if (
            signature != b"PK\x03\x04"
            or flags != record["flags"]
            or compression != record["compression"]
            or raw_name != record["raw_name"]
            or flags & 0x08
            or crc != record["crc"]
        ):
            raise ValueError(f"{label} local and central ZIP records differ")
        if compressed_size == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF:
            if (compressed_size, uncompressed_size) != (0xFFFFFFFF, 0xFFFFFFFF):
                raise ValueError(f"{label} has partial ZIP64 local sizes")
            _zip64_local_sizes(
                extra,
                record["uncompressed_size"],
                record["compressed_size"],
            )
        elif (
            extra
            or compressed_size != record["compressed_size"]
            or uncompressed_size != record["uncompressed_size"]
        ):
            raise ValueError(f"{label} local ZIP sizes or extras differ")
        expected_offset = handle.tell() + record["compressed_size"]
        if expected_offset > central_offset:
            raise ValueError(f"{label} local ZIP payload overlaps metadata")
    if expected_offset != central_offset:
        raise ValueError(f"{label} has bytes outside the indexed local records")

    handle.seek(0)
    with zipfile.ZipFile(handle, mode="r") as archive:
        infos = archive.infolist()
        if len(infos) != total_entries or [info.filename for info in infos] != [
            record["name"] for record in central_records
        ]:
            raise ValueError(f"{label} ZIP parser topology differs")
        total_payload = 0
        for info, record in zip(infos, central_records, strict=True):
            if (
                info.header_offset != record["local_offset"]
                or info.CRC != record["crc"]
                or info.compress_size != record["compressed_size"]
                or info.file_size != record["uncompressed_size"]
            ):
                raise ValueError(f"{label} ZIP parser descriptor differs")
            with archive.open(info, mode="r") as member:
                version = np.lib.format.read_magic(member)
                if version not in {(1, 0), (2, 0)}:
                    raise ValueError(f"{label} NPY version is unsupported")
                header_reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _fortran_order, dtype = header_reader(
                    member, max_header_size=MAX_CORRECTNESS_NPY_HEADER_BYTES
                )
                dtype = np.dtype(dtype)
                if dtype.hasobject:
                    raise ValueError(f"{label} contains an object NPY array")
                payload_size = math.prod(shape) * dtype.itemsize
                if (
                    payload_size > MAX_CORRECTNESS_NPY_PAYLOAD_BYTES
                    or payload_size != info.file_size - member.tell()
                ):
                    raise ValueError(
                        f"{label} NPY payload size differs or is too large"
                    )
                total_payload += payload_size
                if total_payload > MAX_CORRECTNESS_TOTAL_ARRAY_BYTES:
                    raise ValueError(f"{label} exceeds the total NPY payload bound")
                while member.read(1024 * 1024):
                    pass
    return {
        "sha256": digest,
        "size_bytes": file_size,
        "member_count": total_entries,
        "array_payload_bytes": total_payload,
    }


@contextmanager
def _open_bounded_npz(path):
    path = Path(path).resolve(strict=True)
    try:
        with path.open("rb") as handle:
            preflight = _preflight_npz_file(handle, str(path))
            handle.seek(0)
            archive = np.load(handle, allow_pickle=False)
            try:
                yield archive
            except MemoryError as error:
                raise ValueError(
                    "NPZ allocation failed within the bounded contract"
                ) from error
            finally:
                archive.close()
            descriptor = os.fstat(handle.fileno())
            if descriptor.st_size != preflight["size_bytes"]:
                raise ValueError("NPZ changed size while it was being validated")
            if _hash_open_file(handle) != preflight["sha256"]:
                raise ValueError("NPZ bytes changed while they were being validated")
    except (MemoryError, RecursionError) as error:
        raise ValueError(
            "NPZ parsing exceeded the bounded resource contract"
        ) from error


def _load_bounded_json(
    path,
    label,
    *,
    maximum=None,
    require_canonical=False,
    return_raw=False,
):
    path = Path(path).resolve(strict=True)
    maximum = MAX_CORRECTNESS_JSON_BYTES if maximum is None else maximum
    try:
        with path.open("rb") as handle:
            descriptor = os.fstat(handle.fileno())
            if not stat.S_ISREG(descriptor.st_mode):
                raise ValueError(f"{label} must be a regular file")
            if descriptor.st_size <= 0 or descriptor.st_size > maximum:
                raise ValueError(f"{label} exceeds the JSON byte bound")
            raw = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
        if (
            len(raw) != descriptor.st_size
            or len(raw) > maximum
            or (
                descriptor.st_dev,
                descriptor.st_ino,
                descriptor.st_size,
                descriptor.st_mtime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
        ):
            raise ValueError(f"{label} exceeds or changed within the JSON byte bound")
        value = json.loads(
            raw,
            object_pairs_hook=native_oracle._object_without_duplicate_keys,
            parse_constant=native_oracle._reject_json_constant,
        )
        if require_canonical:
            rendered = (
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
            ).encode()
            if raw != rendered:
                raise ValueError(f"{label} is not canonical JSON")
        return (value, raw) if return_raw else value
    except (MemoryError, RecursionError) as error:
        raise ValueError(f"{label} exceeds safe JSON parsing limits") from error


def _load_trusted_manifest(path=DEFAULT_MANIFEST):
    manifest, raw = _load_bounded_json(
        path,
        "trusted repository manifest",
        return_raw=True,
    )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != TRUSTED_MANIFEST_SHA256:
        raise ValueError(
            "trusted repository manifest digest differs from the frozen contract"
        )
    if not isinstance(manifest, dict):
        raise ValueError("trusted repository manifest must be an object")
    return manifest, digest


def _canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _exact_keys(value, expected, label, optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    expected = set(expected)
    optional = set(optional)
    if not expected <= actual <= expected | optional:
        raise ValueError(
            f"{label} keys are invalid: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected - optional)}"
        )


def _required_cases(manifest):
    return tuple(
        (group, case)
        for group in ("correctness", "physical_checks")
        for case in manifest.get(group, ())
    )


def _candidate_geometry(spec):
    if spec["recipe"] == "mixed":
        return native_oracle._mixed_geometry(spec, gmes)
    if spec["recipe"] == "coverage":
        return native_oracle._coverage_geometry(spec, gmes)
    if spec["recipe"] == "heterogeneous":
        return native_oracle._heterogeneous_geometry(spec, gmes)
    material = native_oracle.material_from_name(spec["material"], gmes)
    if spec["material"] in {"upml", "cpml"}:
        return [
            gmes.DefaultMedium(native_oracle.material_from_name("dielectric", gmes)),
            gmes.Shell(
                material=material,
                thickness=spec.get("pml_thickness", 1),
            ),
        ]
    if spec["material"] == "dielectric":
        return [gmes.DefaultMedium(material)]

    # Torch uses a dense Dielectric base.  Preserve the homogeneous native
    # coefficients at the base and overlay the stateful strategy on every
    # active cell.  The canonical step/0 maps remain the archive input.
    base = gmes.Dielectric(eps_inf=material.eps_inf, mu_inf=material.mu_inf)
    minimum = 1.0 / spec["resolution"]
    extent = tuple(max(abs(float(value)), minimum) for value in spec["size"])
    return [
        gmes.DefaultMedium(base),
        gmes.Block(material=material, center=(0, 0, 0), size=extent),
    ]


def _candidate_sources(spec):
    source_name = spec.get("source", "point")
    if source_name == "none":
        return []
    source_time = gmes.Continuous(freq=0.35)
    amplitude = float(spec.get("source_amp", 1e-3))
    if source_name in {"point", "overlap-point"}:
        component = getattr(gmes, spec.get("source_component", "Ex"))
        sources = [
            gmes.PointSource(
                src_time=source_time,
                center=(0, 0, 0),
                component=component,
                amp=amplitude,
            )
        ]
        if source_name == "overlap-point":
            sources.append(
                gmes.PointSource(
                    src_time=gmes.Continuous(freq=0.55, phase=0.2),
                    center=(0, 0, 0),
                    component=component,
                    amp=0.25 * amplitude,
                )
            )
        return sources
    size = np.asarray(spec["size"], dtype=float)
    common = {
        "src_time": source_time,
        "center": (0, 0, 0),
        "size": tuple(max(value * 0.55, 1.0) for value in size),
        "direction": (1, 0, 0),
        "polarization": (0, 1, 0),
        "amp": amplitude,
    }
    if source_name == "tfsf":
        return [gmes.TotalFieldScatteredField(**common)]
    if source_name == "gaussian":
        return [gmes.GaussianBeam(directivity=gmes.PlusX, **common)]
    raise ValueError(f"unknown source recipe: {source_name}")


def _runtime_contract(device, precision, graph_mode, compile_mode):
    if not isinstance(device, str) or (
        device != "cpu"
        and not (
            device.startswith("cuda:")
            and device[5:].isdigit()
            and str(int(device[5:])) == device[5:]
        )
    ):
        raise ValueError("device must be 'cpu' or an explicit canonical 'cuda:N'")
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be float32 or float64")
    if graph_mode not in {"eager", "graph"}:
        raise ValueError("graph_mode must be eager or graph")
    if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError("compile_mode is invalid")
    if graph_mode == "eager" and compile_mode != "default":
        raise ValueError("a non-default compile mode requires graph mode")
    if device == "cpu" and compile_mode != "default":
        raise ValueError("CPU correctness graph mode requires default compile mode")
    return {
        "device": device,
        "precision": precision,
        "graph_mode": graph_mode,
        "compile_policy": "compile" if graph_mode == "graph" else "eager",
        "compile_mode": compile_mode,
    }


def _build_torch_simulation(
    spec,
    *,
    dt,
    threads=1,
    device="cpu",
    precision="float64",
    graph_mode="eager",
    compile_mode="default",
):
    mode = _runtime_contract(device, precision, graph_mode, compile_mode)
    bloch = (0.07, 0.11, 0.13) if spec.get("complex") else None
    return gmes.TorchSimulation(
        space=gmes.Cartesian(tuple(spec["size"]), spec["resolution"]),
        geometry=_candidate_geometry(spec),
        sources=_candidate_sources(spec),
        runtime=gmes.TorchRuntimeConfig(
            device=mode["device"],
            precision=mode["precision"],
            compile_policy=mode["compile_policy"],
            compile_mode=mode["compile_mode"],
            cpu_threads=threads,
            cpu_interop_threads=1,
        ),
        dt=dt,
        bloch=bloch,
    )


def _host(value):
    return value.detach().cpu().numpy().copy()


def _complex_channels(value, paired_real):
    value = _host(value)
    if paired_real:
        return value[..., 0] + 1j * value[..., 1]
    return value[..., 0].astype(np.complex128)


def _dispersive_rows(simulation, descriptor):
    prefix = descriptor.prefix
    paired = simulation.state.paired_real
    poles = descriptor.pole_count
    points = descriptor.point_count

    def state(suffix):
        return getattr(simulation.state, f"{prefix}_{suffix}")

    if descriptor.model in {"drude", "lorentz"}:
        return np.concatenate(
            (
                _complex_channels(state("previous"), paired).T,
                _complex_channels(state("current"), paired).T,
            ),
            axis=1,
        )
    if descriptor.model == "dcp-ade":
        return np.concatenate(
            (
                _complex_channels(state("field_old"), paired)[:, None],
                _complex_channels(state("pole_old"), paired).T,
                _complex_channels(state("pole_now"), paired).T,
                _complex_channels(state("point_old"), paired).T,
                _complex_channels(state("point_now"), paired).T,
            ),
            axis=1,
        )

    pole = _host(state("pole_state"))
    point = _host(state("point_state"))
    channels = 2 if paired else 1
    pole_blocks = [pole[..., channel].T for channel in range(channels)]
    point_blocks = [
        (point[..., channel, 0] + 1j * point[..., channel, 1]).T
        for channel in range(channels)
    ]
    if not paired:
        pole_blocks.append(np.zeros((descriptor.target_count, poles)))
        point_blocks.append(np.zeros((descriptor.target_count, points)))
    return np.concatenate((*pole_blocks, *point_blocks), axis=1)


def _state_parts(simulation, component, models):
    parts = []
    plan = simulation.plan.components[component]
    pml = simulation.state.pml_state_snapshot()
    for index, bucket in enumerate(plan.buckets):
        model = bucket.signature.model
        if model not in models or model not in {"upml", "cpml"}:
            continue
        key = f"pml_{component.lower()}_{index}_state"
        targets = np.asarray(bucket.targets, dtype=np.int64)

        parts.append(
            {
                "model": model,
                "targets": targets,
                "rows": np.asarray(pml[key], dtype=np.complex128),
                "representation": key,
            }
        )

    for descriptor in simulation.plan.dispersive_buckets:
        if descriptor.component != component or descriptor.model not in models:
            continue
        targets = _host(
            getattr(simulation.plan, f"{descriptor.prefix}_targets")
        ).astype(np.int64)

        parts.append(
            {
                "model": descriptor.model,
                "targets": targets,
                "rows": _dispersive_rows(simulation, descriptor),
                "representation": descriptor.prefix,
            }
        )

    dm2_snapshots = {
        (snapshot["component"], tuple(snapshot["targets"])): snapshot
        for snapshot in simulation.dm2_state_snapshot()
    }
    for bucket in simulation.state.dm2_buckets:
        metadata = bucket.metadata
        if component != metadata.component or "dm2" not in models:
            continue
        targets = _host(getattr(simulation.plan, f"{metadata.prefix}_targets")).astype(
            np.int64
        )
        snapshot = dm2_snapshots[(component, tuple(targets))]

        parts.append(
            {
                "model": "dm2",
                "targets": targets,
                "rows": np.asarray(snapshot["u"], dtype=np.complex128).reshape(
                    len(targets), -1
                ),
                "representation": metadata.prefix,
            }
        )
    return parts


def _physical(fields, step, arrays):
    prefix = f"step/{step}/physical"
    summary = [0.0, 0.0, 0.0, 0.0, 1.0]
    for name in COMPONENT_NAMES:
        field = np.asarray(fields[name])
        magnitude = np.abs(field)
        summary[0] += float(np.sum(magnitude * magnitude))
        summary[1] = max(summary[1], float(np.max(magnitude)))
        summary[2] += float(np.sum(magnitude[0] * magnitude[0]))
        summary[3] += float(np.sum(magnitude[-1] * magnitude[-1]))
        summary[4] = float(bool(summary[4]) and np.isfinite(field).all())
        axes = tuple(range(1, field.ndim))
        line = np.mean(field, axis=axes) if axes else field
        arrays[f"{prefix}/spectrum/{name}"] = np.abs(np.fft.fft(line))
    arrays[f"{prefix}/summary"] = np.asarray(summary)
    return {
        "energy": summary[0],
        "maximum_abs_field": summary[1],
        "boundary_low_energy": summary[2],
        "boundary_high_energy": summary[3],
        "finite": bool(summary[4]),
    }


def _strategy_family(strategy):
    return "DcpPlrc" if strategy in {"DcpPlrc", "DcpRc"} else strategy


def _material_plan_groups(simulation, component):
    plan = simulation.plan.components[component]
    material_ids = np.asarray(plan.material_ids).reshape(-1)
    groups = {}
    target_families = {}
    for bucket_index, bucket in enumerate(plan.buckets):
        for target in np.asarray(bucket.targets, dtype=np.int64):
            geometry_index = int(material_ids[int(target)])
            strategy = type(simulation.geometry[geometry_index].material).__name__
            family = _strategy_family(strategy)
            group = groups.setdefault(
                family,
                {"targets": [], "strategies": set(), "buckets": set()},
            )
            group["targets"].append(int(target))
            group["strategies"].add(strategy)
            group["buckets"].add(bucket_index)
            if int(target) in target_families:
                raise ValueError(f"Torch material target is duplicated for {component}")
            target_families[int(target)] = family
    inactive = np.flatnonzero(np.asarray(plan.ownership).reshape(-1) < 0)
    if len(inactive):
        group = groups.setdefault(
            "Dummy",
            {"targets": [], "strategies": set(), "buckets": set()},
        )
        group["strategies"].add("Dummy")
        for value in inactive:
            target = int(value)
            if target in target_families:
                raise ValueError(f"Torch material target is duplicated for {component}")
            group["targets"].append(target)
            target_families[target] = "Dummy"
    if len(target_families) != int(np.prod(plan.shape)):
        raise ValueError(f"Torch material topology is incomplete for {component}")
    normalized = []
    for family, group in sorted(groups.items()):
        targets = np.asarray(sorted(group["targets"]), dtype=np.int64)
        if len(targets) != len(np.unique(targets)):
            raise ValueError(f"Torch material group {family} contains duplicates")
        normalized.append(
            {
                "family": family,
                "strategies": sorted(group["strategies"]),
                "targets": targets,
                "buckets": sorted(group["buckets"]),
            }
        )
    return normalized, target_families


def _group_state_rows(simulation, component, group, target_families):
    parts = []
    for part in _state_parts(simulation, component, set(_STRATEGY_MODELS.values())):
        families = {target_families[int(value)] for value in part["targets"]}
        if len(families) != 1:
            raise ValueError("one Torch persistent-state buffer spans material groups")
        if families == {group["family"]}:
            parts.append(part)
    if not parts:
        return np.empty((len(group["targets"]), 0), dtype=np.complex128), []
    by_target = {}
    width = None
    for part in parts:
        rows = np.asarray(part["rows"], dtype=np.complex128).reshape(
            len(part["targets"]), -1
        )
        if width is None:
            width = rows.shape[1]
        elif rows.shape[1] != width:
            raise ValueError("Torch material state widths differ inside one group")
        for target, row in zip(part["targets"], rows, strict=True):
            target = int(target)
            if target in by_target:
                raise ValueError("Torch material state target is duplicated")
            by_target[target] = row
    expected = {int(value) for value in group["targets"]}
    if set(by_target) != expected:
        raise ValueError("Torch persistent state does not cover its material group")
    return np.asarray([by_target[int(value)] for value in group["targets"]]), parts


def _independent_material_records(
    simulation, step, arrays, *, prefix="state", components=COMPONENT_NAMES
):
    records = []
    suffix = "Cmplx" if simulation.state.paired_real else "Real"
    for component in components:
        groups, target_families = _material_plan_groups(simulation, component)
        shape = tuple(simulation.plan.shapes[component])
        for ordinal, group in enumerate(groups):
            strategy = "+".join(group["strategies"])
            record_prefix = f"step/{step}/{prefix}/{component}/{ordinal}-{strategy}"
            targets = group["targets"]
            indices = np.column_stack(np.unravel_index(targets, shape)).astype(
                np.int64, copy=False
            )
            rows, parts = _group_state_rows(
                simulation, component, group, target_families
            )
            values = np.asarray(rows, dtype=np.complex128).reshape(-1)
            arrays[f"{record_prefix}/indices"] = indices.copy()
            arrays[f"{record_prefix}/values"] = values.copy()
            cells = len(targets)
            runs = native_oracle._linear_run_count(indices, shape)
            plan_bytes = sum(
                int(
                    simulation.plan.components[component].buckets[index].estimated_bytes
                )
                for index in group["buckets"]
            )
            state_bytes = int(values.nbytes)
            native_type = f"{group['family']}{component}{suffix}"
            records.append(
                {
                    "component": component,
                    "strategy": strategy,
                    "strategies": group["strategies"],
                    "native_type": native_type,
                    "cells": cells,
                    "coverage": cells / int(np.prod(shape)),
                    "fragmentation_runs": runs,
                    "fragmentation_ratio": runs / cells if cells else 0.0,
                    "state_values": int(values.size),
                    "state_nonzero_values": int(np.count_nonzero(values)),
                    "state_width": values.size / cells if cells else 0.0,
                    "state_key": f"{record_prefix}/values",
                    "state_bytes": state_bytes,
                    "plan_bytes": plan_bytes,
                    "index_bytes": int(indices.nbytes),
                    "parameter_bytes": 0,
                    "live_updater_bytes": plan_bytes + int(indices.nbytes),
                    "plan_runs": runs,
                    "bucket_signature": [
                        component,
                        strategy,
                        native_type,
                        cells,
                        int(values.size),
                    ],
                    "backend_metadata": {
                        "producer": PRODUCER,
                        "representations": [part["representation"] for part in parts],
                        "plan_bucket_ordinals": group["buckets"],
                        "state_origin": "live-torch-state-buffers",
                    },
                }
            )
    return records


def _source_batch_payload(simulation, batch):
    component = batch.component
    shape = tuple(simulation.plan.shapes[component])
    if isinstance(batch, TorchPointSourceBatch):
        rows = []
        for kind, prefix in enumerate(("overwrite", "additive")):
            targets = _host(getattr(batch, f"{prefix}_targets")).astype(np.int64)
            models = _host(getattr(batch, f"{prefix}_models")).astype(np.float64)
            parameters = _host(getattr(batch, f"{prefix}_parameters")).astype(
                np.float64
            )
            amplitudes = _host(getattr(batch, f"{prefix}_amplitudes")).astype(
                np.float64
            )
            for target, model, parameter, amplitude in zip(
                targets, models, parameters, amplitudes, strict=True
            ):
                rows.append(
                    (
                        int(target),
                        np.concatenate(
                            (
                                np.asarray([kind, model], dtype=np.float64),
                                parameter,
                                np.asarray([amplitude], dtype=np.float64),
                            )
                        ),
                    )
                )
        rows.sort(key=lambda item: (item[0], item[1][0]))
        native_type = f"PointSource{component}"
        representation = "torch-point-source-batch-v1"
    elif isinstance(batch, TorchTransparentBatch):
        targets = _host(batch.targets).astype(np.int64)
        samples = _host(batch.samples).astype(np.float64)
        weights = _host(batch.weights).astype(np.float64)
        width = -1.0 if batch.gaussian_width is None else batch.gaussian_width
        rows = [
            (
                int(target),
                np.concatenate(
                    (np.asarray([width]), sample.reshape(-1), weight.reshape(-1))
                ),
            )
            for target, sample, weight in zip(targets, samples, weights, strict=True)
        ]
        rows.sort(key=lambda item: item[0])
        native_type = f"Transparent{component}"
        representation = "torch-transparent-source-batch-v1"
    else:
        raise TypeError(f"unsupported Torch source batch {type(batch).__name__}")
    targets = np.asarray([target for target, _row in rows], dtype=np.int64)
    indices = (
        np.column_stack(np.unravel_index(targets, shape)).astype(np.int64, copy=False)
        if len(targets)
        else np.empty((0, 3), dtype=np.int64)
    )
    # Preserve the exact IEEE-754 payload while satisfying the base archive's
    # finite-number contract. Point-source plans legitimately use infinities
    # as open-interval sentinels; the raw named buffers are archived separately.
    values = (
        np.concatenate([row for _target, row in rows])
        .astype("<f8", copy=False)
        .view("<u8")
        .copy()
        if rows
        else np.empty(0, dtype="<u8")
    )
    return native_type, representation, indices, values


def _auxiliary_source_name(simulation, auxiliary):
    batches = [
        batch
        for batch in simulation.sources.batches
        if isinstance(batch, TorchTransparentBatch) and batch.auxiliary is auxiliary
    ]
    if not batches:
        raise ValueError("Torch auxiliary simulation has no transparent source batch")
    return (
        "GaussianBeam"
        if any(batch.gaussian_width is not None for batch in batches)
        else "TotalFieldScatteredField"
    )


def _independent_source_records(simulation, step, arrays):
    records = []
    ordinals = {component: 0 for component in COMPONENT_NAMES}
    ordered = sorted(
        enumerate(simulation.sources.batches),
        key=lambda item: (item[1].component, type(item[1]).__name__, item[0]),
    )
    for source_ordinal, batch in ordered:
        native_type, representation, indices, values = _source_batch_payload(
            simulation, batch
        )
        component = batch.component
        ordinal = ordinals[component]
        ordinals[component] += 1
        prefix = f"step/{step}/source/{component}/{ordinal}-{native_type}"
        arrays[f"{prefix}/indices"] = indices.copy()
        arrays[f"{prefix}/values"] = values.copy()
        records.append(
            {
                "component": component,
                "native_type": native_type,
                "cells": len(indices),
                "state_values": int(values.size),
                "backend_metadata": {
                    "producer": PRODUCER,
                    "representation": representation,
                    "source_origin": "live-torch-source-batch",
                    "batch_ordinal": source_ordinal,
                },
            }
        )

    auxiliaries = []
    for ordinal, auxiliary in enumerate(simulation.sources.auxiliaries):
        source_name = _auxiliary_source_name(simulation, auxiliary)
        prefix = f"step/{step}/source_aux/{ordinal}-{source_name}"
        fields = auxiliary.host_snapshot()
        active_components = tuple(
            component
            for component in COMPONENT_NAMES
            if np.count_nonzero(fields[component])
        )
        if not active_components:
            raise ValueError("Torch auxiliary simulation has no active field component")
        field_shapes = {}
        for component in COMPONENT_NAMES:
            values = np.asarray(fields[component]).copy()
            canonical = (
                values
                if component in active_components
                else np.zeros((1, 1, 1), dtype=values.dtype)
            )
            arrays[f"{prefix}/field/{component}"] = canonical
            field_shapes[component] = list(canonical.shape)
        arrays[f"{prefix}/time"] = np.asarray(
            [
                int(auxiliary.state.step_count.detach().cpu()),
                int(auxiliary.state.step_count.detach().cpu()) * auxiliary.plan.dt,
                auxiliary.plan.dt,
            ]
        )
        auxiliaries.append(
            {
                "source": source_name,
                "fields": field_shapes,
                "materials": _independent_material_records(
                    auxiliary,
                    step,
                    arrays,
                    components=active_components,
                    prefix=f"source_aux_material/{ordinal}",
                ),
                "backend_metadata": {
                    "producer": PRODUCER,
                    "canonical_components": list(active_components),
                    "precision": auxiliary.runtime.precision,
                    "raw_state_origin": "torch-named-buffers",
                    "representation": "live-torch-auxiliary-simulation-v1",
                },
            }
        )
    return {"updaters": records, "auxiliary": auxiliaries}


def _safe_buffer_name(name):
    parts = name.split(".")
    if not parts or any(not part or "/" in part for part in parts):
        raise ValueError(f"Torch buffer name is not canonical: {name!r}")
    return "/".join(parts)


def _store_module_buffers(module, prefix, arrays):
    for name, value in module.named_buffers():
        key = f"{prefix}/{_safe_buffer_name(name)}"
        if key in arrays:
            raise ValueError(f"duplicate Torch buffer archive key: {key}")
        arrays[key] = _host(value)


def _store_planner_arrays(simulation, arrays, prefix="torch/planner"):
    for component, plan in simulation.plan.components.items():
        component_prefix = f"{prefix}/{component}"
        for name in _PLANNER_COMPONENT_ARRAYS:
            arrays[f"{component_prefix}/{name}"] = np.asarray(
                getattr(plan, name)
            ).copy()
        for ordinal, bucket in enumerate(plan.buckets):
            bucket_prefix = (
                f"{component_prefix}/bucket/{ordinal}-{bucket.signature.model}"
            )
            for name in _PLANNER_BUCKET_ARRAYS:
                arrays[f"{bucket_prefix}/{name}"] = np.asarray(
                    getattr(bucket, name)
                ).copy()
            for axis, residual in enumerate(bucket.cpml_residual_axes):
                residual_prefix = f"{bucket_prefix}/cpml/{axis}"
                for name in _PLANNER_CPML_RESIDUAL_ARRAYS:
                    arrays[f"{residual_prefix}/{name}"] = np.asarray(
                        getattr(residual, name)
                    ).copy()


def _store_live_torch_state(simulation, step, arrays):
    _store_module_buffers(simulation.state, f"torch/step/{step}/state", arrays)
    _store_module_buffers(simulation.sources, f"torch/step/{step}/sources", arrays)
    for ordinal, auxiliary in enumerate(simulation.sources.auxiliaries):
        prefix = f"torch/step/{step}/auxiliary/{ordinal}"
        _store_module_buffers(auxiliary.state, f"{prefix}/state", arrays)
        _store_module_buffers(auxiliary.sources, f"{prefix}/sources", arrays)


def _independent_snapshot(simulation, step, arrays):
    fields = simulation.host_snapshot()
    for name in COMPONENT_NAMES:
        arrays[f"step/{step}/field/{name}"] = np.asarray(fields[name]).copy()
    arrays[f"step/{step}/time"] = np.asarray(
        [
            int(simulation.state.step_count.detach().cpu()),
            int(simulation.state.step_count.detach().cpu()) * simulation.plan.dt,
            simulation.plan.dt,
        ]
    )
    snapshot = {
        "materials": _independent_material_records(simulation, step, arrays),
        "sources": _independent_source_records(simulation, step, arrays),
        "physical": _physical(fields, step, arrays),
    }
    _store_live_torch_state(simulation, step, arrays)
    return snapshot


def _logical_geometry_id_projection(actual_geometry, logical_geometry):
    if native_oracle._same_json_value(actual_geometry, logical_geometry):
        return {index: index for index in range(len(actual_geometry))}

    projection = {}
    for actual_index, actual in enumerate(actual_geometry):
        exact = [
            index
            for index, logical in enumerate(logical_geometry)
            if native_oracle._same_json_value(actual, logical)
        ]
        if len(exact) == 1:
            projection[actual_index] = exact[0]
            continue
        material = [
            index
            for index, logical in enumerate(logical_geometry)
            if native_oracle._same_json_value(actual["material"], logical["material"])
        ]
        if len(material) == 1:
            projection[actual_index] = material[0]
    return projection


def _project_logical_map_ids(
    material_ids, underlying_ids, actual_geometry, logical_geometry
):
    material_ids = np.asarray(material_ids).reshape(-1)
    underlying_ids = np.asarray(underlying_ids).reshape(-1)
    if (
        material_ids.shape != underlying_ids.shape
        or material_ids.dtype.kind not in "iu"
        or underlying_ids.dtype.kind not in "iu"
        or np.any(material_ids < 0)
        or np.any(material_ids >= len(actual_geometry))
        or np.any(underlying_ids < -1)
        or np.any(underlying_ids >= len(actual_geometry))
    ):
        raise ValueError("Torch planner map shapes differ")
    if len(logical_geometry) == 1:
        return (
            np.zeros(material_ids.shape, dtype=material_ids.dtype),
            np.full(underlying_ids.shape, -1, dtype=underlying_ids.dtype),
        )
    projection = _logical_geometry_id_projection(actual_geometry, logical_geometry)
    canonical_material = np.empty(material_ids.shape, dtype=material_ids.dtype)
    canonical_underlying = np.full(underlying_ids.shape, -1, dtype=underlying_ids.dtype)
    for index, value in enumerate(material_ids):
        material_id = int(value)
        if material_id not in projection:
            raise ValueError("Torch planner material has no logical geometry mapping")
        canonical_material[index] = projection[material_id]
    for index, value in enumerate(underlying_ids):
        underlying_id = int(value)
        if underlying_id < 0:
            continue
        if underlying_id in projection:
            canonical_underlying[index] = projection[underlying_id]
    return canonical_material, canonical_underlying


def _active_component_targets(component, shape):
    bounds = {
        "Ex": ((0, shape[0]), (0, shape[1] - 1), (0, shape[2] - 1)),
        "Ey": ((0, shape[0] - 1), (0, shape[1]), (0, shape[2] - 1)),
        "Ez": ((0, shape[0] - 1), (0, shape[1] - 1), (0, shape[2])),
        "Hx": ((0, shape[0]), (1, shape[1]), (1, shape[2])),
        "Hy": ((1, shape[0]), (0, shape[1]), (1, shape[2])),
        "Hz": ((1, shape[0]), (1, shape[1]), (0, shape[2])),
    }[component]
    active = np.zeros(shape, dtype=np.bool_)
    active[tuple(slice(start, stop) for start, stop in bounds)] = True
    return np.flatnonzero(active.reshape(-1)).astype(np.int64, copy=False)


def _transparent_auxiliary_solver(value):
    nested = getattr(value, "aux_fdtd", None)
    return nested if not hasattr(value, "time_step") and nested is not None else value


def _transparent_batch_arrays(
    component, parameters, *, main_shape, auxiliary_shape, gaussian_width, dt, dr
):
    active = np.zeros(int(np.prod(main_shape)), dtype=np.bool_)
    active[_active_component_targets(component, main_shape)] = True
    targets = []
    terms = []
    for target, parameter in parameters:
        linear_target = int(np.ravel_multi_index(target, main_shape))
        if not active[linear_target]:
            continue
        target_terms = []
        for face in parameter.face_list:
            coefficient = gmes.torch_source._transparent_coefficient(
                component, face, parameter, dt, dr
            )
            for sample, weight in (
                (parameter.samp_idx0[face], parameter.r0[face]),
                (parameter.samp_idx1[face], parameter.r1[face]),
            ):
                target_terms.append(
                    (
                        int(np.ravel_multi_index(sample, auxiliary_shape)),
                        coefficient * weight,
                    )
                )
        consolidated = {}
        for sample, weight in target_terms:
            consolidated[sample] = consolidated.get(sample, 0.0) + weight
        targets.append(linear_target)
        terms.append(tuple(consolidated.items()))
    width = max((len(row) for row in terms), default=0)
    samples = np.zeros((len(terms), width), dtype=np.int64)
    weights = np.zeros((len(terms), width), dtype=np.float64)
    for row, values in enumerate(terms):
        for column, (sample, weight) in enumerate(values):
            samples[row, column] = sample
            weights[row, column] = weight
    return {
        "component": component,
        "native_type": f"Transparent{component}",
        "targets": np.asarray(targets, dtype=np.int64),
        "samples": samples,
        "weights": weights,
        "gaussian_width": gaussian_width,
    }


def _expected_transparent_live_buffers(workload, dt, backend, reference):
    simulation = _build_torch_simulation(
        workload,
        dt=dt,
        threads=1,
        device=backend["device"],
        precision=backend["precision"],
        graph_mode=backend["graph_mode"],
        compile_mode=backend["compile_mode"],
    )
    initial_fields = native_oracle.initial_field_values(
        simulation.plan.shapes,
        reference["seed"],
        reference["field_scale"],
        complex_fields=simulation.state.paired_real,
    )
    simulation.load_host_fields(initial_fields)
    if backend["graph_mode"] == "graph" and simulation.device.type == "cuda":
        simulation.capture_cuda_graphs()
    simulation.advance(reference["precondition_steps"])

    def snapshot():
        records = []
        for batch in simulation.sources.batches:
            if not isinstance(batch, TorchTransparentBatch):
                raise ValueError(
                    "transparent workload contains a non-transparent Torch batch"
                )
            records.append(
                {
                    name: _host(value)
                    for name, value in batch.named_buffers(recurse=False)
                }
            )
        return tuple(records)

    buffers = {"0": snapshot()}
    completed = 0
    for target in reference["capture_steps"]:
        simulation.advance(target - completed)
        completed = target
        buffers[str(target)] = snapshot()
    return buffers


def _derive_transparent_source_contract(workload, dt, backend, reference):
    if workload.get("source", "point") not in {"tfsf", "gaussian"}:
        return None
    simulation = native_oracle.build_simulation(workload, gmes)
    simulation.init()
    if not np.array_equal(
        np.asarray(simulation.time_step.dt, dtype=np.float64),
        np.asarray(dt, dtype=np.float64),
    ):
        raise ValueError("transparent source time step differs from the workload")

    auxiliary_by_id = {}
    auxiliaries = []
    for native_ordinal, source in enumerate(simulation.src_list):
        native_auxiliary = getattr(source, "aux_fdtd", None)
        if native_auxiliary is None:
            continue
        solver = _transparent_auxiliary_solver(native_auxiliary)
        gaussian_width = (
            float(source.src_time.width)
            if isinstance(source, gmes.GaussianBeam)
            else None
        )
        auxiliary = {
            "candidate_ordinal": len(auxiliaries),
            "native_ordinal": native_ordinal,
            "solver": solver,
            "gaussian_width": gaussian_width,
            "initial_step_count": int(solver.time_step.n),
        }
        auxiliaries.append(auxiliary)
        auxiliary_by_id[id(native_auxiliary)] = auxiliary
        auxiliary_by_id[id(solver)] = auxiliary

    native_records = []
    batches = []
    for component_type, updaters in sorted(
        simulation.pw_source.items(), key=lambda item: item[0].__name__
    ):
        component = component_type.__name__
        main_shape = tuple(simulation.field[component_type].shape)
        for updater in sorted(
            updaters.values(), key=lambda value: type(value).__name__
        ):
            if not type(updater).__name__.startswith("Transparent"):
                continue
            ordered = sorted(updater._param.items())
            native_records.append(
                {
                    "component": component,
                    "native_type": type(updater).__name__,
                    "parameters": ordered,
                }
            )
            grouped = {}
            for target, parameter in updater._param.items():
                auxiliary = auxiliary_by_id.get(id(parameter.aux_fdtd))
                if auxiliary is None:
                    raise ValueError(
                        "transparent source parameter has no workload auxiliary"
                    )
                grouped.setdefault(auxiliary["candidate_ordinal"], []).append(
                    (target, parameter)
                )
            for auxiliary_ordinal, parameters in grouped.items():
                auxiliary = auxiliaries[auxiliary_ordinal]
                auxiliary_component = "Hy" if component.startswith("E") else "Ex"
                auxiliary_shape = tuple(
                    auxiliary["solver"].field[getattr(gmes, auxiliary_component)].shape
                )
                batch = _transparent_batch_arrays(
                    component,
                    parameters,
                    main_shape=main_shape,
                    auxiliary_shape=auxiliary_shape,
                    gaussian_width=auxiliary["gaussian_width"],
                    dt=dt,
                    dr=tuple(float(value) for value in simulation.space.dr),
                )
                batch.update(
                    auxiliary_ordinal=auxiliary_ordinal,
                    auxiliary_component=auxiliary_component,
                )
                batches.append(batch)
    native_records.sort(key=lambda item: (item["component"], item["native_type"]))
    batches.sort(
        key=lambda item: (
            item["component"],
            item["native_type"],
            item["auxiliary_ordinal"],
        )
    )
    for batch_ordinal, batch in enumerate(batches):
        batch["batch_ordinal"] = batch_ordinal

    device_type = backend["device"].split(":", 1)[0]
    for auxiliary in auxiliaries:
        solver = auxiliary["solver"]
        space, component_plans = _build_expected_component_plans(
            solver.space,
            solver.geom_list,
            dt=dt,
            precision="float64",
            device_type=device_type,
            compile_policy=backend["compile_policy"],
        )
        auxiliary["space"] = space
        auxiliary["component_plans"] = component_plans
    return {
        "simulation": simulation,
        "native_records": tuple(native_records),
        "batches": tuple(batches),
        "auxiliaries": tuple(auxiliaries),
        "live_buffers": _expected_transparent_live_buffers(
            workload, dt, backend, reference
        ),
    }


def _validate_planner_map_self_consistency(archive, planner_root, component, shape):
    shape = tuple(shape)
    component_root = f"{planner_root}/{component}"
    material_ids = np.asarray(archive[f"{component_root}/material_ids"])
    underlying_ids = np.asarray(archive[f"{component_root}/underlying_ids"])
    ownership = np.asarray(archive[f"{component_root}/ownership"])
    if (
        material_ids.dtype != np.dtype("int32")
        or material_ids.shape != shape
        or underlying_ids.dtype != np.dtype("int32")
        or underlying_ids.shape != shape
        or ownership.dtype != np.dtype("int16")
        or ownership.shape != shape
    ):
        raise ValueError(f"Torch planner map planes are invalid for {component}")

    bucket_prefix = f"{component_root}/bucket/"
    bucket_roots = {}
    for key in archive.files:
        if (
            not key.startswith(bucket_prefix)
            or not key.endswith("/targets")
            or "/cpml/" in key
        ):
            continue
        identity = key.removeprefix(bucket_prefix).removesuffix("/targets")
        ordinal_text, separator, model = identity.partition("-")
        if (
            not separator
            or not ordinal_text.isdigit()
            or ordinal_text != str(int(ordinal_text))
            or not model
            or int(ordinal_text) in bucket_roots
        ):
            raise ValueError(
                f"Torch planner bucket identity is invalid for {component}"
            )
        bucket_roots[int(ordinal_text)] = f"{bucket_prefix}{identity}"
    if sorted(bucket_roots) != list(range(len(bucket_roots))):
        raise ValueError(f"Torch planner bucket ordinals differ for {component}")

    active = _active_component_targets(component, shape)
    flat_ownership = ownership.reshape(-1)
    if not np.array_equal(np.flatnonzero(flat_ownership >= 0), active):
        raise ValueError(f"Torch planner ownership coverage differs for {component}")
    bucket_targets = []
    flat_material = material_ids.reshape(-1)
    flat_underlying = underlying_ids.reshape(-1)
    for ordinal, bucket_root in sorted(bucket_roots.items()):
        targets = np.asarray(archive[f"{bucket_root}/targets"])
        target_region_indices = np.asarray(
            archive[f"{bucket_root}/target_region_indices"]
        )
        region_keys = np.asarray(archive[f"{bucket_root}/region_keys"])
        if (
            targets.dtype != np.dtype("int64")
            or targets.ndim != 1
            or np.any(targets < 0)
            or np.any(targets >= material_ids.size)
            or target_region_indices.dtype != np.dtype("int32")
            or target_region_indices.shape != targets.shape
            or region_keys.dtype != np.dtype("int32")
            or region_keys.ndim != 2
            or region_keys.shape[1:] != (2,)
            or np.any(target_region_indices < 0)
            or np.any(target_region_indices >= len(region_keys))
        ):
            raise ValueError(f"Torch planner bucket maps are invalid for {component}")
        if not np.array_equal(np.flatnonzero(flat_ownership == ordinal), targets):
            raise ValueError(f"Torch planner bucket ownership differs for {component}")
        actual_keys = np.column_stack(
            (flat_material[targets], flat_underlying[targets])
        )
        if not np.array_equal(actual_keys, region_keys[target_region_indices]):
            raise ValueError(
                f"Torch planner region indirection differs for {component}"
            )
        bucket_targets.append(targets)
    covered = (
        np.sort(np.concatenate(bucket_targets))
        if bucket_targets
        else np.empty(0, dtype=np.int64)
    )
    if not np.array_equal(covered, active) or len(covered) != len(np.unique(covered)):
        raise ValueError(f"Torch planner bucket coverage differs for {component}")


def _component_maps(simulation, fields, arrays, actual_geometry, logical_geometry):
    metadata = {}
    for component in COMPONENT_NAMES:
        plan = simulation.plan.components[component]
        material_ids, underlying_ids = _project_logical_map_ids(
            plan.material_ids,
            plan.underlying_ids,
            actual_geometry,
            logical_geometry,
        )
        arrays[f"map/{component}/material_ids"] = material_ids
        arrays[f"map/{component}/underlying_ids"] = underlying_ids
        underlying = underlying_ids[underlying_ids >= 0]
        metadata[component] = {
            "shape": list(plan.shape),
            "dtype": str(np.asarray(fields[component]).dtype),
            "active_cells": int(material_ids.size),
            "material_regions": int(np.unique(material_ids).size),
            "underlying_regions": int(np.unique(underlying).size),
        }
    return metadata


def _logical_geometry(spec):
    if spec["recipe"] in {"mixed", "coverage", "heterogeneous"}:
        return _candidate_geometry(spec)
    material = native_oracle.material_from_name(spec["material"], gmes)
    if spec["material"] in {"upml", "cpml"}:
        return [
            gmes.DefaultMedium(native_oracle.material_from_name("dielectric", gmes)),
            gmes.Shell(
                material=material,
                thickness=spec.get("pml_thickness", 1),
            ),
        ]
    return [gmes.DefaultMedium(material)]


def _geometry_metadata(geometry):
    return [
        {
            "geometry": type(item).__name__,
            "material": native_oracle._json_value(item.material),
        }
        for item in geometry
    ]


def _initialized_geometry_metadata(spec, dt):
    space = gmes.Cartesian(tuple(spec["size"]), spec["resolution"])
    space.dt = float(dt)
    geometry = _candidate_geometry(spec)
    for geometric_object in geometry:
        geometric_object.init(space)
    return _geometry_metadata(geometry)


def _logical_geometry_metadata(spec, dt):
    space = gmes.Cartesian(tuple(spec["size"]), spec["resolution"])
    space.dt = float(dt)
    geometry = _logical_geometry(spec)
    for geometric_object in geometry:
        geometric_object.init(space)
    return _geometry_metadata(geometry)


def _build_expected_component_plans(
    space, geometry, *, dt, precision, device_type, compile_policy
):
    space.dt = float(dt)
    geometry = tuple(geometry)
    for geometric_object in geometry:
        geometric_object.init(space)
    runtime = gmes.TorchRuntimeConfig(
        device="cpu" if device_type == "cpu" else "cuda:0",
        precision=precision,
        compile_policy=compile_policy,
        compile_mode=(
            "default"
            if device_type == "cpu"
            else ("reduce-overhead" if compile_policy == "compile" else "default")
        ),
        cpu_threads=1,
        cpu_interop_threads=1,
    )
    runtime.validate_static()
    planner = gmes.torch_fdtd.TorchExecutionPlanner(
        geom_tree=gmes.torch_fdtd.GeomBoxTree(geometry),
        space=space,
        shapes=gmes.torch_fdtd._field_shapes(space),
        precision=precision,
        device_type=device_type,
        policy=runtime.execution_policy,
        execution_tile_size=runtime.planner_tile_size,
        cpml_sparse_residual=(compile_policy == "compile" and device_type == "cpu"),
        avoid_dense_auto=(
            runtime.execution_policy == "auto"
            and compile_policy == "compile"
            and device_type == "cpu"
        ),
        avoid_tiled_auto=(
            runtime.execution_policy == "auto"
            and compile_policy == "compile"
            and device_type == "cpu"
        ),
    )
    component_plans = planner.build()
    return space, {plan.name: plan for plan in component_plans}


def _expected_primary_component_plans(metadata):
    backend = metadata["backend_metadata"]
    workload = metadata["workload"]
    space = gmes.Cartesian(tuple(workload["size"]), workload["resolution"])
    return _build_expected_component_plans(
        space,
        _candidate_geometry(workload),
        dt=np.asarray(metadata["_step_zero_time"])[2],
        precision=backend["precision"],
        device_type=backend["device"].split(":", 1)[0],
        compile_policy=backend["compile_policy"],
    )


def _planner_array_values(component_plans, prefix):
    values = {}
    for component in COMPONENT_NAMES:
        plan = component_plans[component]
        component_prefix = f"{prefix}/{component}"
        for name in _PLANNER_COMPONENT_ARRAYS:
            values[f"{component_prefix}/{name}"] = np.asarray(getattr(plan, name))
        for ordinal, bucket in enumerate(plan.buckets):
            bucket_prefix = (
                f"{component_prefix}/bucket/{ordinal}-{bucket.signature.model}"
            )
            for name in _PLANNER_BUCKET_ARRAYS:
                values[f"{bucket_prefix}/{name}"] = np.asarray(getattr(bucket, name))
            for axis, residual in enumerate(bucket.cpml_residual_axes):
                residual_prefix = f"{bucket_prefix}/cpml/{axis}"
                for name in _PLANNER_CPML_RESIDUAL_ARRAYS:
                    values[f"{residual_prefix}/{name}"] = np.asarray(
                        getattr(residual, name)
                    )
    return values


def _finalized_plan_buffer_values(component_plans, *, space, dt, precision, bloch):
    module = gmes.torch_fdtd.TorchSimulationPlan(
        tuple(component_plans[name] for name in COMPONENT_NAMES),
        dr=space.dr,
        dt=dt,
        bloch=bloch,
        device="cpu",
        dtype={
            "float32": gmes.torch_fdtd.torch.float32,
            "float64": gmes.torch_fdtd.torch.float64,
        }[precision],
    )
    return {
        _safe_buffer_name(name): _host(value) for name, value in module.named_buffers()
    }


def _require_exact_array(actual, expected, label):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if (
        actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or not np.array_equal(actual, expected)
    ):
        raise ValueError(f"{label} differs from the immutable workload plan")


def _validate_expected_planner(
    archive,
    component_plans,
    *,
    space,
    dt,
    precision,
    bloch,
    planner_root,
    plan_roots,
):
    expected_planner = _planner_array_values(component_plans, planner_root)
    actual_planner_keys = {
        key for key in archive.files if key.startswith(f"{planner_root}/")
    }
    if actual_planner_keys != set(expected_planner):
        raise ValueError(f"{planner_root} array topology differs from the workload")
    for key, expected in expected_planner.items():
        _require_exact_array(archive[key], expected, key)

    expected_buffers = _finalized_plan_buffer_values(
        component_plans,
        space=space,
        dt=dt,
        precision=precision,
        bloch=bloch,
    )
    for root in plan_roots:
        expected_keys = {f"{root}/{name}" for name in expected_buffers}
        actual_keys = {key for key in archive.files if key.startswith(f"{root}/")}
        if actual_keys != expected_keys:
            raise ValueError(f"{root} buffer topology differs from the workload")
        for name, expected in expected_buffers.items():
            key = f"{root}/{name}"
            _require_exact_array(archive[key], expected, key)


def _validate_expected_primary_planner(archive, metadata, expected_steps):
    step_zero_time = np.asarray(archive["step/0/time"])
    derived_metadata = dict(metadata)
    derived_metadata["_step_zero_time"] = step_zero_time
    space, component_plans = _expected_primary_component_plans(derived_metadata)
    backend = metadata["backend_metadata"]
    _validate_expected_planner(
        archive,
        component_plans,
        space=space,
        dt=float(step_zero_time[2]),
        precision=backend["precision"],
        bloch=((0.07, 0.11, 0.13) if metadata["workload"].get("complex") else None),
        planner_root="torch/planner",
        plan_roots=(
            "torch/plan",
            *(
                f"torch/step/{step}/state/plan"
                for step in sorted(expected_steps, key=int)
            ),
        ),
    )


def _array_descriptor(value):
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "size_bytes": int(contiguous.nbytes),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _array_descriptors(arrays, predicate):
    return {
        key: _array_descriptor(arrays[key]) for key in sorted(arrays) if predicate(key)
    }


def _candidate_provenance():
    source = Path(gmes.torch_fdtd.__file__).resolve()
    controller = Path(__file__).resolve()
    records = {
        "source": native_oracle._checkout_provenance(
            native_oracle._git_checkout(source), source
        ),
        "controller": native_oracle._checkout_provenance(
            native_oracle._git_checkout(controller), controller
        ),
    }
    if any(record["clean"] is not True for record in records.values()):
        raise RuntimeError("Torch correctness evidence requires clean Git checkouts")
    if records["source"]["commit"] != records["controller"]["commit"]:
        raise RuntimeError("Torch source and correctness controller commits differ")
    return records, source


def capture_torch_candidate(
    reference_path,
    manifest,
    output,
    *,
    threads=1,
    device="cpu",
    precision="float64",
    graph_mode="eager",
    compile_mode="default",
):
    """Independently execute Torch and emit a strict schema-2 candidate."""
    mode = _runtime_contract(device, precision, graph_mode, compile_mode)
    reference_path = Path(reference_path).resolve(strict=True)
    output = Path(output).resolve()
    if output == reference_path:
        raise ValueError("candidate output must differ from the native reference")
    with _open_bounded_npz(reference_path) as reference:
        reference_metadata = native_oracle._validate_archive(
            reference, manifest, "reference"
        )
        spec = reference_metadata["workload"]
        dt = float(np.asarray(reference["step/0/time"])[2])
        input_arrays = {
            key: _array_descriptor(reference[key])
            for key in sorted(reference.files)
            if key.startswith(("map/", "step/0/"))
        }
    simulation = _build_torch_simulation(
        spec,
        dt=dt,
        threads=threads,
        device=device,
        precision=precision,
        graph_mode=graph_mode,
        compile_mode=compile_mode,
    )
    for name in COMPONENT_NAMES:
        shape = tuple(reference_metadata["maps"][name]["shape"])
        if tuple(simulation.plan.shapes[name]) != shape:
            raise ValueError(f"Torch field shape differs for {name}")
    initial_fields = native_oracle.initial_field_values(
        simulation.plan.shapes,
        manifest["reference"]["seed"],
        manifest["reference"]["field_scale"],
        complex_fields=simulation.state.paired_real,
    )
    simulation.load_host_fields(initial_fields)
    if graph_mode == "graph" and simulation.device.type == "cuda":
        simulation.capture_cuda_graphs()
    simulation.advance(manifest["reference"]["precondition_steps"])

    arrays = {}
    _store_planner_arrays(simulation, arrays)
    _store_module_buffers(simulation.plan, "torch/plan", arrays)
    for ordinal, auxiliary in enumerate(simulation.sources.auxiliaries):
        _store_planner_arrays(
            auxiliary,
            arrays,
            prefix=f"torch/auxiliary/{ordinal}/planner",
        )
    fields = simulation.host_snapshot()
    actual_geometry = _geometry_metadata(simulation.geometry)
    logical_geometry = _logical_geometry_metadata(spec, dt)
    map_metadata = _component_maps(
        simulation,
        fields,
        arrays,
        actual_geometry,
        logical_geometry,
    )
    steps = {"0": _independent_snapshot(simulation, 0, arrays)}
    completed = 0
    capture_steps = spec.get("capture_steps", manifest["reference"]["capture_steps"])
    for target in capture_steps:
        simulation.advance(target - completed)
        completed = target
        steps[str(target)] = _independent_snapshot(simulation, target, arrays)

    final_records = steps[str(completed)]["materials"]
    all_records = [record for value in steps.values() for record in value["materials"]]
    totals = {
        "active_cells": sum(record["cells"] for record in final_records),
        "state_bytes": sum(record["state_bytes"] for record in final_records),
        "plan_bytes": sum(record["plan_bytes"] for record in final_records),
        "index_bytes": sum(record["index_bytes"] for record in final_records),
        "parameter_bytes": sum(record["parameter_bytes"] for record in final_records),
    }
    totals["live_updater_bytes"] = (
        totals["plan_bytes"] + totals["index_bytes"] + totals["parameter_bytes"]
    )
    torch_arrays = _array_descriptors(arrays, lambda key: key.startswith("torch/"))
    source_arrays = _array_descriptors(
        arrays,
        lambda key: len(key.split("/")) > 2 and key.split("/")[2] == "source",
    )
    base_bytes = sum(
        value.nbytes for key, value in arrays.items() if not key.startswith("torch/")
    )
    provenance, source = _candidate_provenance()
    metadata = {
        "schema_version": native_oracle.ARCHIVE_SCHEMA_VERSION,
        "backend": "torch",
        "backend_metadata": {
            "producer": PRODUCER,
            "solver_abi": gmes.torch_fdtd.TORCH_SOLVER_ABI,
            "cuda_graph_execution_representation": (
                gmes.torch_fdtd.CUDA_GRAPH_EXECUTION_REPRESENTATION
            ),
            **mode,
            "resolved_device": str(simulation.device),
            "paired_real": bool(simulation.state.paired_real),
            "auxiliary_precisions": [
                auxiliary.runtime.precision
                for auxiliary in simulation.sources.auxiliaries
            ],
            "manifest_contract_sha256": _canonical_sha256(manifest),
            "input_archive": {
                "sha256": _sha256(reference_path),
                "size_bytes": reference_path.stat().st_size,
                "media_type": "application/x-npz",
                "prefix": "step/0",
            },
            "input_step_zero_contract": {
                "array_contract": "native-step-zero-and-maps-v1",
                "array_bytes": sum(
                    descriptor["size_bytes"] for descriptor in input_arrays.values()
                ),
                "arrays": input_arrays,
                "reconstruction": {
                    "mode": "independent-workload-replay-v1",
                    "field_initializer": manifest["reference"]["field_initializer"],
                    "seed": manifest["reference"]["seed"],
                    "field_scale": manifest["reference"]["field_scale"],
                    "precondition_steps": manifest["reference"]["precondition_steps"],
                    "field_origin": "manifest-initializer",
                    "source_origin": "live-torch-plan",
                    "material_origin": "live-torch-plan",
                },
            },
            "logical_map_source": "live-torch-plan",
            "actual_geometry_and_coefficients": actual_geometry,
            "plan_identity": simulation.plan_identity,
            "compile_cache_key": simulation.compile_cache_key,
            "cuda_graph_regions": list(sorted(simulation._cuda_graphs)),
            "array_contract": TORCH_ARRAY_CONTRACT,
            "torch_array_bytes": sum(
                descriptor["size_bytes"] for descriptor in torch_arrays.values()
            ),
            "torch_arrays": torch_arrays,
            "source_arrays": source_arrays,
        },
        "workload": spec,
        "reference": native_oracle._correctness_reference_contract(
            manifest["reference"]
        ),
        "capture_steps": capture_steps,
        "input_state": {
            "archive_prefix": "step/0",
            "precondition_steps": manifest["reference"]["precondition_steps"],
            "relative_capture_steps": True,
        },
        "maps": map_metadata,
        "steps": steps,
        "geometry_and_coefficients": logical_geometry,
        **totals,
        "archive_array_bytes": int(base_bytes),
        "nonzero_seed": all(
            np.count_nonzero(arrays[f"step/0/field/{name}"])
            == arrays[f"step/0/field/{name}"].size
            for name in COMPONENT_NAMES
        ),
        "nonzero_persistent_state": all(
            record["state_values"] == 0 or record["state_nonzero_values"] > 0
            for record in all_records
        ),
        "provenance": provenance,
        "reference_source": str(source),
    }
    arrays["metadata.json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    validation = compare_torch_archives(reference_path, output, manifest)
    if validation != {"passed": True, "failures": []}:
        raise ValueError(
            "independent Torch reconstruction differs from native step/0 contract: "
            f"{validation['failures'][:1]}"
        )
    return metadata


class _BaseArchiveView:
    def __init__(self, archive):
        self._archive = archive
        self.files = [key for key in archive.files if not key.startswith("torch/")]

    def __getitem__(self, key):
        if key not in self.files:
            raise KeyError(key)
        return self._archive[key]


def _descriptor_map_complete(archive, descriptors, expected_keys, label):
    _exact_keys(descriptors, expected_keys, label)
    for key in sorted(expected_keys):
        descriptor = descriptors[key]
        _exact_keys(
            descriptor,
            {"shape", "dtype", "size_bytes", "sha256"},
            f"{label}.{key}",
        )
        if not native_oracle._same_json_value(
            descriptor, _array_descriptor(archive[key])
        ):
            raise ValueError(f"{label} descriptor differs for {key}")


def _input_step_zero_contract_complete(contract, manifest):
    try:
        _exact_keys(
            contract,
            {"array_contract", "array_bytes", "arrays", "reconstruction"},
            "Torch native step/0 input contract",
        )
        arrays = contract["arrays"]
        if not isinstance(arrays, dict) or not arrays:
            return False
        required = {"step/0/time"}
        required.update(f"step/0/field/{name}" for name in COMPONENT_NAMES)
        required.update(
            f"map/{name}/{suffix}"
            for name in COMPONENT_NAMES
            for suffix in ("material_ids", "underlying_ids")
        )
        if not required <= set(arrays):
            return False
        size_bytes = 0
        for key, descriptor in arrays.items():
            if not isinstance(key, str) or not key.startswith(("map/", "step/0/")):
                return False
            _exact_keys(
                descriptor,
                {"shape", "dtype", "size_bytes", "sha256"},
                f"Torch native step/0 descriptor {key}",
            )
            shape = descriptor["shape"]
            if not isinstance(shape, list) or any(
                type(dimension) is not int or dimension < 0 for dimension in shape
            ):
                return False
            dtype_name = descriptor["dtype"]
            if not isinstance(dtype_name, str) or not dtype_name:
                return False
            dtype = np.dtype(dtype_name)
            if dtype.hasobject:
                return False
            expected_size = int(dtype.itemsize)
            for dimension in shape:
                expected_size *= dimension
            if (
                type(descriptor["size_bytes"]) is not int
                or descriptor["size_bytes"] != expected_size
                or not _hex_string(descriptor["sha256"], 64)
            ):
                return False
            size_bytes += expected_size
        reconstruction = contract["reconstruction"]
        expected_reconstruction = {
            "mode": "independent-workload-replay-v1",
            "field_initializer": manifest["reference"]["field_initializer"],
            "seed": manifest["reference"]["seed"],
            "field_scale": manifest["reference"]["field_scale"],
            "precondition_steps": manifest["reference"]["precondition_steps"],
            "field_origin": "manifest-initializer",
            "source_origin": "live-torch-plan",
            "material_origin": "live-torch-plan",
        }
        return (
            contract["array_contract"] == "native-step-zero-and-maps-v1"
            and type(contract["array_bytes"]) is int
            and contract["array_bytes"] == size_bytes
            and native_oracle._same_json_value(reconstruction, expected_reconstruction)
        )
    except AttributeError, KeyError, TypeError, ValueError, OverflowError:
        return False


def _validate_live_clock(archive, prefix, precision, canonical_time):
    step_count = np.asarray(archive[f"{prefix}/step_count"])
    source_time = np.asarray(archive[f"{prefix}/source_time"])
    time_step = np.asarray(archive[f"{prefix}/time_step"])
    clock_dtype = np.dtype(precision)
    if (
        step_count.shape != ()
        or step_count.dtype != np.dtype("int64")
        or int(step_count) < 0
        or source_time.shape != ()
        or time_step.shape != ()
        or source_time.dtype != clock_dtype
        or time_step.dtype != clock_dtype
        or not bool(np.isfinite(source_time))
        or not bool(np.isfinite(time_step))
        or float(time_step) <= 0.0
    ):
        raise ValueError(f"Torch live clock is invalid: {prefix}")
    expected = np.multiply(step_count.astype(clock_dtype), time_step, dtype=clock_dtype)
    canonical_time = np.asarray(canonical_time)
    if canonical_time.shape != (3,):
        raise ValueError(f"Torch canonical clock is invalid: {prefix}")
    canonical_dt = np.asarray(canonical_time[2], dtype=clock_dtype)
    if (
        float(canonical_time[0]) != int(step_count)
        or not np.array_equal(time_step, canonical_dt)
        or not np.array_equal(source_time, expected)
    ):
        raise ValueError(f"Torch live clock differs from canonical n * dt: {prefix}")


def _live_field_value(value, canonical, paired_real, label):
    value = np.asarray(value)
    canonical = np.asarray(canonical)
    if paired_real:
        if value.shape != canonical.shape + (2,) or value.dtype.kind != "f":
            raise ValueError(f"Torch paired-real live field is invalid: {label}")
        value = value[..., 0] + 1j * value[..., 1]
    elif value.shape != canonical.shape or value.dtype != canonical.dtype:
        raise ValueError(f"Torch live field shape or dtype differs: {label}")
    if value.dtype != canonical.dtype or not np.array_equal(value, canonical):
        raise ValueError(f"Torch live field differs from canonical field: {label}")


def _point_source_model_values(models, parameters, time, paired_real, dtype):
    values = []
    for model, row in zip(models, parameters, strict=True):
        model = int(model)
        parameters_row = [float(value) for value in row]
        if model == 0:
            frequency, phase, start, end, width, _unused = parameters_row
            ts = time - start
            te = end - time
            rise = np.sin(0.5 * np.pi * ts / width) ** 2 if ts < width else 1.0
            fall = np.sin(0.5 * np.pi * te / width) ** 2 if te < width else 1.0
            envelope = rise * fall if ts >= 0 and te >= 0 else 0.0
            angle = 2.0 * np.pi * frequency * time + phase
            value = envelope * complex(np.cos(angle), np.sin(angle))
        elif model == 1:
            frequency, phase, width, peak_time, cutoff, _unused = parameters_row
            offset = time - peak_time
            envelope = (
                np.exp(-0.5 * (offset / width) ** 2) if abs(offset) <= cutoff else 0.0
            )
            angle = 2.0 * np.pi * frequency * time + phase
            value = (
                envelope
                * complex(-np.sin(angle), np.cos(angle))
                / (2.0 * np.pi * frequency)
            )
        elif model == 2:
            width, peak_time, *_unused = parameters_row
            offset = (time - peak_time) / width
            value = complex(-2.0 * offset * np.exp(-(offset**2)), 0.0)
        else:
            raise ValueError("Torch PointSource time model is invalid")
        values.append((value.real, value.imag) if paired_real else (value.real,))
    channels = 2 if paired_real else 1
    return np.asarray(values, dtype=dtype).reshape(-1, channels)


def _point_source_values_match(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return False
    if not expected.size or np.all(expected == 0):
        return bool(np.array_equal(actual, expected))
    # Torch and NumPy evaluate the trigonometric model through different libm
    # kernels. Keep this far below the numerical archive tolerances while
    # allowing the accumulated auxiliary clock's expected ULP drift.
    tolerance = (
        512 * np.finfo(expected.dtype).eps * max(1.0, float(np.max(np.abs(expected))))
    )
    return bool(np.all(np.abs(actual - expected) <= tolerance))


def _point_source_live_payload(
    archive,
    live_root,
    *,
    precision,
    paired_real,
    canonical_time,
    component,
    field_shape,
    expected_kind,
    expected_model,
    expected_parameters,
    expected_amplitude,
):
    precision = np.dtype(precision)
    channels = 2 if paired_real else 1
    live = {
        name: np.asarray(archive[f"{live_root}/{name}"])
        for name in _POINT_SOURCE_LIVE_ARRAYS
    }
    rows = []
    for kind, prefix in enumerate(("overwrite", "additive")):
        targets = live[f"{prefix}_targets"]
        models = live[f"{prefix}_models"]
        parameters = live[f"{prefix}_parameters"]
        amplitudes = live[f"{prefix}_amplitudes"]
        evaluated = live[f"_{prefix}_values"]
        count = len(targets) if targets.ndim == 1 else -1
        if (
            targets.dtype != np.dtype("int64")
            or targets.shape != (count,)
            or models.dtype != np.dtype("int8")
            or models.shape != (count,)
            or parameters.dtype != precision
            or parameters.shape != (count, 6)
            or amplitudes.dtype != precision
            or amplitudes.shape != (count,)
            or evaluated.dtype != precision
            or evaluated.shape != (count, channels)
            or not np.isfinite(amplitudes).all()
        ):
            raise ValueError(
                f"Torch PointSource live buffer contract differs: {live_root}"
            )
        canonical_time = np.asarray(canonical_time)
        if canonical_time.shape != (3,):
            raise ValueError(f"Torch PointSource canonical time differs: {live_root}")
        time = float(canonical_time[1])
        if component.startswith("E"):
            time -= 0.5 * float(canonical_time[2])
        expected_values = _point_source_model_values(
            models, parameters, time, paired_real, precision
        )
        expected_values *= amplitudes[:, None]
        if not _point_source_values_match(evaluated, expected_values):
            raise ValueError(f"Torch PointSource evaluated values differ: {live_root}")
        for target, model, parameter, amplitude in zip(
            targets, models, parameters, amplitudes, strict=True
        ):
            rows.append(
                (
                    int(target),
                    kind,
                    int(model),
                    np.asarray(parameter),
                    np.asarray(amplitude),
                )
            )
    rows.sort(key=lambda item: (item[0], item[1]))
    expected_parameters = np.asarray(expected_parameters, dtype=precision)
    expected_amplitude = np.asarray(expected_amplitude, dtype=precision)
    if (
        len(rows) != 1
        or rows[0][1] != expected_kind
        or rows[0][2] != expected_model
        or not np.array_equal(rows[0][3], expected_parameters)
        or not np.array_equal(rows[0][4], expected_amplitude)
    ):
        raise ValueError("Torch PointSource semantics differ from expected source")
    field_shape = tuple(field_shape)
    try:
        expected_indices = np.asarray(
            [np.unravel_index(target, field_shape) for target, *_rest in rows],
            dtype=np.int64,
        ).reshape(-1, 3)
    except ValueError as error:
        raise ValueError("Torch PointSource target is outside its field") from error
    packed_rows = [
        np.concatenate(
            (
                np.asarray([kind, model], dtype=np.float64),
                parameter.astype(np.float64),
                amplitude.reshape(1).astype(np.float64),
            )
        )
        for _target, kind, model, parameter, amplitude in rows
    ]
    expected_packed = (
        np.concatenate(packed_rows).astype("<f8", copy=False).view("<u8")
        if packed_rows
        else np.empty(0, dtype="<u8")
    )
    return expected_indices, expected_packed, len(rows)


def _validate_point_source_batch(archive, metadata, step, record, record_ordinal):
    backend = metadata["backend_metadata"]
    precision = np.dtype(backend["precision"])
    paired_real = backend["paired_real"]
    batch_ordinal = record["backend_metadata"]["batch_ordinal"]
    live_root = f"torch/step/{step}/sources/batches/{batch_ordinal}"
    source_name = metadata["workload"].get("source", "point")
    if source_name not in {"point", "overlap-point"}:
        raise ValueError("Torch PointSource workload semantics are absent")
    source_component = metadata["workload"].get("source_component", "Ex")
    expected_kind = 1 if source_component.startswith(("J", "M")) else 0
    expected_component = {
        "J": "E",
        "M": "H",
    }.get(
        source_component[0], source_component[0]
    ) + source_component[1:]
    frequency, phase, scale = (
        (0.55, 0.2, 0.25) if source_name == "overlap-point" else (0.35, 0.0, 1.0)
    )
    if record["component"] != expected_component:
        raise ValueError("Torch PointSource component differs from workload")
    expected_indices, expected_packed, cells = _point_source_live_payload(
        archive,
        live_root,
        precision=precision,
        paired_real=paired_real,
        canonical_time=archive[f"step/{step}/time"],
        component=record["component"],
        field_shape=np.asarray(
            archive[f"step/{step}/field/{record['component']}"]
        ).shape,
        expected_kind=expected_kind,
        expected_model=0,
        expected_parameters=(
            frequency,
            phase,
            0.0,
            np.inf,
            5.0 / frequency,
            0.0,
        ),
        expected_amplitude=scale * float(metadata["workload"].get("source_amp", 1e-3)),
    )
    if record["cells"] != cells or record["state_values"] != 9 * cells:
        raise ValueError("Torch PointSource packed state closure differs")
    canonical_root = (
        f"step/{step}/source/{record['component']}/"
        f"{record_ordinal}-{record['native_type']}"
    )
    indices = np.asarray(archive[f"{canonical_root}/indices"])
    values = np.asarray(archive[f"{canonical_root}/values"])
    if (
        indices.dtype != np.dtype("int64")
        or not np.array_equal(indices, expected_indices)
        or values.dtype != np.dtype("<u8")
        or not np.array_equal(values, expected_packed)
    ):
        raise ValueError("Torch PointSource packed/live semantics differ")


def _validate_auxiliary_point_source_batch(
    archive,
    torch_keys,
    metadata,
    step,
    auxiliary_ordinal,
    record,
):
    source_root = f"torch/step/{step}/auxiliary/{auxiliary_ordinal}/sources/batches/"
    batch_suffixes = {}
    for key in torch_keys:
        if not key.startswith(source_root):
            continue
        remainder = key.removeprefix(source_root)
        parts = remainder.split("/")
        if len(parts) != 2 or not parts[0].isdigit() or parts[0] != str(int(parts[0])):
            raise ValueError("Torch auxiliary source batch key is not canonical")
        batch_suffixes.setdefault(int(parts[0]), set()).add(parts[1])
    if set(batch_suffixes) != {0}:
        raise ValueError("Torch auxiliary PointSource batch closure differs")
    if batch_suffixes[0] != set(_POINT_SOURCE_LIVE_ARRAYS):
        raise ValueError("Torch auxiliary PointSource live buffer closure differs")
    if record["source"] not in {"TotalFieldScatteredField", "GaussianBeam"}:
        raise ValueError("Torch auxiliary PointSource source is unsupported")

    planner_root = f"torch/auxiliary/{auxiliary_ordinal}/planner"
    field_shape = np.asarray(archive[f"{planner_root}/Ex/material_ids"]).shape
    canonical_time = archive[
        f"step/{step}/source_aux/{auxiliary_ordinal}-{record['source']}/time"
    ]
    indices, packed, cells = _point_source_live_payload(
        archive,
        f"{source_root}0",
        precision=record["backend_metadata"]["precision"],
        paired_real=metadata["backend_metadata"]["paired_real"],
        canonical_time=canonical_time,
        component="Ex",
        field_shape=field_shape,
        expected_kind=0,
        expected_model=0,
        expected_parameters=(0.35, 0.0, 0.0, np.inf, 5.0 / 0.35, 0.0),
        expected_amplitude=1.0,
    )
    if cells != 1:
        raise ValueError("Torch auxiliary PointSource cardinality differs")
    ownership = np.asarray(archive[f"{planner_root}/Ex/ownership"])
    target = int(np.ravel_multi_index(indices[0], field_shape))
    if (
        ownership.dtype != np.dtype("int16")
        or ownership.shape != field_shape
        or int(ownership.reshape(-1)[target]) < 0
    ):
        raise ValueError("Torch auxiliary PointSource target is not owned")
    return indices, packed


def _validate_point_source_capture_invariance(archive, metadata):
    baselines = {}
    for step in ("0", *(str(value) for value in metadata["capture_steps"])):
        ordinals = {component: 0 for component in COMPONENT_NAMES}
        for record in metadata["steps"][step]["sources"]["updaters"]:
            component = record["component"]
            ordinal = ordinals[component]
            ordinals[component] += 1
            if not record["native_type"].startswith("PointSource"):
                continue
            identity = (component, ordinal, record["native_type"])
            root = f"step/{step}/source/{component}/{ordinal}-{record['native_type']}"
            current = tuple(
                np.asarray(archive[f"{root}/{suffix}"])
                for suffix in ("indices", "values")
            )
            baseline = baselines.setdefault(
                identity,
                tuple(value.copy() for value in current),
            )
            if any(
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not np.array_equal(actual, expected)
                for actual, expected in zip(current, baseline, strict=True)
            ):
                raise ValueError(
                    "Torch PointSource canonical plan differs across captures"
                )


def _live_complex_channels(value, paired_real, precision):
    value = np.asarray(value)
    if value.dtype != np.dtype(precision):
        raise ValueError("Torch material state precision differs")
    if paired_real:
        if value.shape[-1:] != (2,) or value.dtype.kind != "f":
            raise ValueError("Torch paired-real material state is invalid")
        return (value[..., 0] + 1j * value[..., 1]).astype(np.complex128, copy=False)
    if value.shape[-1:] != (1,) or value.dtype.kind != "f":
        raise ValueError("Torch real material state is invalid")
    return value[..., 0].astype(np.complex128)


def _material_representation(archive, planner_root, component, representation):
    parts = representation.split("_")
    if (
        len(parts) == 4
        and parts[0] == "pml"
        and parts[1] == component.lower()
        and parts[2].isdigit()
        and parts[3] == "state"
    ):
        bucket_ordinal = int(parts[2])
    elif (
        len(parts) == 3
        and parts[0] == "bucket"
        and parts[1] == component.lower()
        and parts[2].isdigit()
    ):
        bucket_ordinal = int(parts[2])
    else:
        raise ValueError("Torch material representation is not canonical")
    prefix = f"{planner_root}/{component}/bucket/{bucket_ordinal}-"
    candidates = [
        key.removeprefix(prefix).removesuffix("/targets")
        for key in archive.files
        if key.startswith(prefix) and key.endswith("/targets") and "/cpml/" not in key
    ]
    if len(candidates) != 1:
        raise ValueError("Torch material representation has no unique plan bucket")
    model = candidates[0]
    bucket_root = f"{prefix}{model}"
    targets = np.asarray(archive[f"{bucket_root}/targets"])
    if targets.dtype != np.dtype("int64") or targets.ndim != 1:
        raise ValueError("Torch material plan targets are invalid")
    return bucket_ordinal, model, bucket_root, targets


def _dm2_bucket_ordinal(archive, planner_root, component, bucket_ordinal):
    identities = []
    for component_name in COMPONENT_NAMES:
        prefix = f"{planner_root}/{component_name}/bucket/"
        for key in archive.files:
            if not key.startswith(prefix) or not key.endswith("-dm2/targets"):
                continue
            token = key.removeprefix(prefix).split("-", 1)[0]
            if token.isdigit() and token == str(int(token)):
                identities.append((component_name, int(token)))
    identity = (component, bucket_ordinal)
    if identities.count(identity) != 1:
        raise ValueError("Torch DM2 plan identity is invalid")
    ordered = sorted(
        identities,
        key=lambda item: (COMPONENT_NAMES.index(item[0]), item[1]),
    )
    return ordered.index(identity)


def _planner_material_representations(archive, planner_root, components):
    persistent_models = {
        "upml",
        "cpml",
        "drude",
        "lorentz",
        "dcp-ade",
        "dcp-plrc",
        "dcp-rc",
        "dm2",
    }
    representations = {}
    for component in components:
        prefix = f"{planner_root}/{component}/bucket/"
        for key in archive.files:
            if not key.startswith(prefix) or not key.endswith("/targets"):
                continue
            identity = key.removeprefix(prefix).removesuffix("/targets")
            if "/" in identity or "-" not in identity:
                continue
            ordinal, model = identity.split("-", 1)
            if (
                not ordinal.isdigit()
                or ordinal != str(int(ordinal))
                or model not in persistent_models
            ):
                continue
            ordinal = int(ordinal)
            representation = (
                f"pml_{component.lower()}_{ordinal}_state"
                if model in {"upml", "cpml"}
                else f"bucket_{component.lower()}_{ordinal}"
            )
            identity = (component, representation)
            if identity in representations:
                raise ValueError("Torch material plan representation is duplicated")
            representations[identity] = (ordinal, model)
    return representations


def _pml_live_rows(
    archive,
    state_root,
    bucket_root,
    representation,
    targets,
    paired_real,
    precision,
):
    direct_key = f"{state_root}/{representation}"
    if direct_key in archive.files:
        value = np.asarray(archive[direct_key])
        if value.dtype != np.dtype(precision) or value.shape[0] != len(targets):
            raise ValueError("Torch PML state shape or precision differs")
        if paired_real:
            if value.shape[-1:] != (2,):
                raise ValueError("Torch paired-real PML state is invalid")
            return (value[..., 0] + 1j * value[..., 1]).astype(np.complex128)
        if value.dtype.kind != "f":
            raise ValueError("Torch PML state dtype is invalid")
        return value.astype(np.complex128)

    component = representation.split("_")[1]
    bucket_ordinal = representation.split("_")[2]
    plane = (2,) if paired_real else ()
    logical = None
    axis_count = 0
    for axis in range(2):
        position_key = f"{bucket_root}/cpml/{axis}/positions"
        if position_key not in archive.files:
            continue
        positions = np.asarray(archive[position_key])
        state_key = f"{state_root}/_pml_{component}_{bucket_ordinal}_axis{axis}_state"
        state = np.asarray(archive[state_key])
        if (
            positions.dtype != np.dtype("int64")
            or positions.ndim != 1
            or np.any(positions < 0)
            or np.any(positions >= len(targets))
            or len(np.unique(positions)) != len(positions)
            or state.shape != (len(positions),) + plane
            or state.dtype != np.dtype(precision)
        ):
            raise ValueError("Torch sparse CPML live state is invalid")
        if logical is None:
            logical = np.zeros((len(targets), 2) + plane, dtype=state.dtype)
        elif logical.dtype != state.dtype:
            raise ValueError("Torch sparse CPML live state dtypes differ")
        logical[positions, axis] = state
        axis_count += 1
    if axis_count != 2 or logical is None:
        raise ValueError("Torch sparse CPML live state closure differs")
    if paired_real:
        return (logical[..., 0] + 1j * logical[..., 1]).astype(np.complex128)
    return logical.astype(np.complex128)


def _material_live_rows(
    archive,
    planner_root,
    state_root,
    component,
    representation,
    paired_real,
    precision,
):
    bucket_ordinal, model, bucket_root, targets = _material_representation(
        archive, planner_root, component, representation
    )

    def state(suffix):
        return np.asarray(archive[f"{state_root}/{representation}_{suffix}"])

    if model in {"upml", "cpml"}:
        rows = _pml_live_rows(
            archive,
            state_root,
            bucket_root,
            representation,
            targets,
            paired_real,
            precision,
        )
    elif model in {"drude", "lorentz"}:
        rows = np.concatenate(
            (
                _live_complex_channels(state("previous"), paired_real, precision).T,
                _live_complex_channels(state("current"), paired_real, precision).T,
            ),
            axis=1,
        )
    elif model == "dcp-ade":
        rows = np.concatenate(
            (
                _live_complex_channels(state("field_old"), paired_real, precision)[
                    :, None
                ],
                _live_complex_channels(state("pole_old"), paired_real, precision).T,
                _live_complex_channels(state("pole_now"), paired_real, precision).T,
                _live_complex_channels(state("point_old"), paired_real, precision).T,
                _live_complex_channels(state("point_now"), paired_real, precision).T,
            ),
            axis=1,
        )
    elif model in {"dcp-plrc", "dcp-rc"}:
        pole = np.asarray(state("pole_state"))
        point = np.asarray(state("point_state"))
        if pole.dtype != np.dtype(precision) or point.dtype != np.dtype(precision):
            raise ValueError("Torch material state precision differs")
        channels = 2 if paired_real else 1
        pole_blocks = [pole[..., channel].T for channel in range(channels)]
        point_blocks = [
            (point[..., channel, 0] + 1j * point[..., channel, 1]).T
            for channel in range(channels)
        ]
        if not paired_real:
            pole_blocks.append(np.zeros((len(targets), pole.shape[0])))
            point_blocks.append(np.zeros((len(targets), point.shape[0])))
        rows = np.concatenate((*pole_blocks, *point_blocks), axis=1).astype(
            np.complex128
        )
    elif model == "dm2":
        dm2_ordinal = _dm2_bucket_ordinal(
            archive, planner_root, component, bucket_ordinal
        )
        raw = np.asarray(archive[f"{state_root}/dm2_buckets/{dm2_ordinal}/u"])
        if (
            raw.dtype != np.dtype(precision)
            or raw.ndim != 3
            or raw.shape[0] != 3
            or raw.shape[1] != len(targets)
        ):
            raise ValueError("Torch DM2 live state is invalid")
        rows = raw.transpose(1, 2, 0).reshape(len(targets), -1).astype(np.complex128)
    else:
        raise ValueError("Torch material representation model is unsupported")
    rows = np.asarray(rows, dtype=np.complex128).reshape(len(targets), -1)
    return targets, rows, bucket_ordinal, model


def _validate_live_material_records(
    archive,
    metadata,
    step,
    records,
    canonical_prefix,
    planner_root,
    state_root,
    field_shapes=None,
    precision=None,
):
    components = {record["component"] for record in records}
    expected_representations = _planner_material_representations(
        archive, planner_root, components
    )
    observed_representations = set()
    ordinals = {component: 0 for component in COMPONENT_NAMES}
    for record in records:
        component = record["component"]
        ordinal = ordinals[component]
        ordinals[component] += 1
        backend = record.get("backend_metadata")
        _exact_keys(
            backend,
            {
                "producer",
                "representations",
                "plan_bucket_ordinals",
                "state_origin",
            },
            "Torch material backend_metadata",
        )
        representations = backend["representations"]
        bucket_ordinals = backend["plan_bucket_ordinals"]
        if (
            backend["producer"] != PRODUCER
            or backend["state_origin"] != "live-torch-state-buffers"
            or not isinstance(representations, list)
            or len(representations) != len(set(representations))
            or any(not isinstance(value, str) or not value for value in representations)
            or not isinstance(bucket_ordinals, list)
            or len(bucket_ordinals) != len(set(bucket_ordinals))
            or any(type(value) is not int or value < 0 for value in bucket_ordinals)
        ):
            raise ValueError("Torch material backend identity is invalid")
        root = (
            f"step/{step}/{canonical_prefix}/{component}/"
            f"{ordinal}-{record['strategy']}"
        )
        indices = np.asarray(archive[f"{root}/indices"])
        values = np.asarray(archive[f"{root}/values"])
        shape = tuple(
            metadata["maps"][component]["shape"]
            if field_shapes is None
            else field_shapes[component]
        )
        targets = (
            np.ravel_multi_index(indices.T, shape) if len(indices) else np.empty(0)
        )
        by_target = {}
        for representation in representations:
            part_targets, rows, bucket_ordinal, model = _material_live_rows(
                archive,
                planner_root,
                state_root,
                component,
                representation,
                metadata["backend_metadata"]["paired_real"],
                (
                    metadata["backend_metadata"]["precision"]
                    if precision is None
                    else precision
                ),
            )
            identity = (component, representation)
            expected_identity = expected_representations.get(identity)
            expected_models = {
                _STRATEGY_MODELS[strategy] for strategy in record["strategies"]
            }
            if (
                identity in observed_representations
                or expected_identity != (bucket_ordinal, model)
                or bucket_ordinal not in bucket_ordinals
                or model not in expected_models
            ):
                raise ValueError("Torch material representation binding differs")
            observed_representations.add(identity)
            for target, row in zip(part_targets, rows, strict=True):
                target = int(target)
                if target in by_target:
                    raise ValueError("Torch live material state target is duplicated")
                by_target[target] = row
        if set(by_target) != {int(value) for value in targets}:
            if values.size or by_target:
                raise ValueError("Torch live material state target closure differs")
            expected = np.empty(0, dtype=np.complex128)
        else:
            expected = np.asarray(
                [by_target[int(target)] for target in targets], dtype=np.complex128
            ).reshape(-1)
        if values.dtype != np.dtype("complex128") or not np.array_equal(
            values, expected
        ):
            raise ValueError("Torch live material state differs from canonical state")
    if observed_representations != set(expected_representations):
        raise ValueError("Torch live material representation closure differs")


def _validate_zero_live_material_representations(
    archive,
    planner_root,
    state_root,
    components,
    paired_real,
    precision,
):
    representations = _planner_material_representations(
        archive, planner_root, components
    )
    for component, representation in sorted(representations):
        _targets, rows, _bucket_ordinal, _model = _material_live_rows(
            archive,
            planner_root,
            state_root,
            component,
            representation,
            paired_real,
            precision,
        )
        if np.count_nonzero(rows):
            raise ValueError(
                "Torch inactive auxiliary material state is not zero: "
                f"{component}/{representation}"
            )


def _packed_transparent_source(batch, main_shape):
    order = np.argsort(batch["targets"], kind="stable")
    targets = batch["targets"][order]
    samples = batch["samples"][order]
    weights = batch["weights"][order]
    indices = (
        np.column_stack(np.unravel_index(targets, main_shape)).astype(
            np.int64, copy=False
        )
        if len(targets)
        else np.empty((0, 3), dtype=np.int64)
    )
    width = -1.0 if batch["gaussian_width"] is None else batch["gaussian_width"]
    rows = [
        np.concatenate(
            (
                np.asarray([width], dtype=np.float64),
                sample.astype(np.float64, copy=False),
                weight.astype(np.float64, copy=False),
            )
        )
        for sample, weight in zip(samples, weights, strict=True)
    ]
    values = (
        np.concatenate(rows).astype("<f8", copy=False).view("<u8").copy()
        if rows
        else np.empty(0, dtype="<u8")
    )
    return indices, values


def _require_finite_derived_array(actual, expected, label):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if (
        actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or not bool(np.isfinite(actual).all())
        or not bool(np.isfinite(expected).all())
    ):
        raise ValueError(f"{label} shape, precision, or finiteness differs")
    if not actual.size:
        return
    tolerance = (
        np.finfo(actual.dtype).eps
        * max(16, actual.shape[-1] * 8 if actual.ndim else 16)
        * max(1.0, float(np.max(np.abs(expected))))
    )
    if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} differs from derived live source semantics")


def _validate_transparent_batch_semantics(
    archive, metadata, step, record, record_ordinal, contract
):
    if contract is None:
        raise ValueError("transparent source workload contract is absent")
    batch_ordinal = record["backend_metadata"]["batch_ordinal"]
    try:
        batch = contract["batches"][batch_ordinal]
    except (IndexError, TypeError) as error:
        raise ValueError("transparent source batch ordinal differs") from error
    if (
        batch["batch_ordinal"] != batch_ordinal
        or batch["component"] != record["component"]
        or batch["native_type"] != record["native_type"]
    ):
        raise ValueError("transparent source batch identity differs")

    batch_root = f"torch/step/{step}/sources/batches/{batch_ordinal}"
    raw = {
        name: np.asarray(archive[f"{batch_root}/{name}"])
        for name in (
            "targets",
            "samples",
            "weights",
            "_sample_values",
            "_values",
            "_outer_values",
        )
    }
    expected_static_dtypes = {
        "targets": np.dtype("int64"),
        "samples": np.dtype("int64"),
        "weights": np.dtype("float64"),
    }
    for name, expected_dtype in expected_static_dtypes.items():
        if raw[name].dtype != expected_dtype:
            raise ValueError(
                f"Torch transparent raw precision differs for {batch_root}/{name}"
            )
    for name in ("targets", "samples", "weights"):
        _require_exact_array(
            raw[name],
            batch[name],
            f"{batch_root}/{name}",
        )
    if not bool(np.isfinite(raw["weights"]).all()):
        raise ValueError(f"{batch_root}/weights contains non-finite values")

    main_shape = tuple(metadata["maps"][record["component"]]["shape"])
    indices, packed_values = _packed_transparent_source(batch, main_shape)
    canonical_root = (
        f"step/{step}/source/{record['component']}/"
        f"{record_ordinal}-{record['native_type']}"
    )
    _require_exact_array(
        archive[f"{canonical_root}/indices"],
        indices,
        f"{canonical_root}/indices",
    )
    _require_exact_array(
        archive[f"{canonical_root}/values"],
        packed_values,
        f"{canonical_root}/values",
    )
    if record["cells"] != len(indices) or record["state_values"] != len(packed_values):
        raise ValueError("transparent source record counts differ")

    backend = metadata["backend_metadata"]
    plane = 2 if backend["paired_real"] else 1
    expected_sample_shape = batch["samples"].shape + (plane,)
    expected_value_shape = (len(batch["targets"]), plane)
    if (
        raw["_sample_values"].shape != expected_sample_shape
        or raw["_sample_values"].dtype != np.dtype("float64")
        or raw["_values"].shape != expected_value_shape
        or raw["_values"].dtype != np.dtype("float64")
        or raw["_outer_values"].shape != expected_value_shape
        or raw["_outer_values"].dtype != np.dtype(backend["precision"])
    ):
        raise ValueError("transparent source live scratch shape or precision differs")

    auxiliary = contract["auxiliaries"][batch["auxiliary_ordinal"]]
    auxiliary_record = metadata["steps"][step]["sources"]["auxiliary"][
        batch["auxiliary_ordinal"]
    ]
    auxiliary_root = f"torch/step/{step}/auxiliary/{batch['auxiliary_ordinal']}/state"
    auxiliary_size = int(
        np.prod(auxiliary["component_plans"][batch["auxiliary_component"]].shape)
    )
    if np.any(batch["samples"] < 0) or np.any(batch["samples"] >= auxiliary_size):
        raise ValueError("transparent source sample index is outside the auxiliary")

    try:
        expected_live = contract["live_buffers"][step][batch_ordinal]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("transparent source live replay identity differs") from error
    expected_names = {
        "targets",
        "samples",
        "weights",
        "_sample_values",
        "_values",
        "_outer_values",
    }
    if batch["gaussian_width"] is not None:
        expected_names.update({"_envelope_step", "_envelope_step_offset", "_envelope"})
    if set(expected_live) != expected_names:
        raise ValueError("transparent source live replay topology differs")

    if batch["gaussian_width"] is not None:
        for name, dtype, shape in (
            ("_envelope_step", np.dtype("int64"), ()),
            ("_envelope_step_offset", np.dtype("int64"), ()),
            ("_envelope", np.dtype("float64"), ()),
        ):
            value = np.asarray(archive[f"{batch_root}/{name}"])
            if value.dtype != dtype or value.shape != shape:
                raise ValueError(f"{batch_root}/{name} shape or precision differs")
        step_count = int(np.asarray(archive[f"{auxiliary_root}/step_count"]))
        time_step = float(np.asarray(archive[f"{auxiliary_root}/time_step"]))
        expected_offset = int(expected_live["_envelope_step_offset"])
        expected_step = (
            step_count - expected_offset - int(record["component"].startswith("E"))
        )
        raw_step = int(np.asarray(archive[f"{batch_root}/_envelope_step"]))
        if (
            raw_step != expected_step
            or int(archive[f"{batch_root}/_envelope_step_offset"]) != expected_offset
        ):
            raise ValueError("transparent Gaussian envelope step differs")
        envelope_time = float(raw_step) * time_step
        width = batch["gaussian_width"]
        if width > 0:
            envelope_time = min(envelope_time, width)
            expected_envelope = np.sin(0.5 * np.pi * envelope_time / width) ** 2
        else:
            expected_envelope = 1.0
        _require_finite_derived_array(
            archive[f"{batch_root}/_envelope"],
            np.asarray(expected_envelope, dtype=np.float64),
            f"{batch_root}/_envelope",
        )
        for name in ("_envelope_step", "_envelope_step_offset"):
            _require_exact_array(
                archive[f"{batch_root}/{name}"],
                expected_live[name],
                f"{batch_root}/{name}",
            )
        _require_finite_derived_array(
            archive[f"{batch_root}/_envelope"],
            expected_live["_envelope"],
            f"{batch_root}/_envelope",
        )
        _require_exact_array(
            archive[f"{batch_root}/_envelope"],
            expected_live["_envelope"],
            f"{batch_root}/_envelope",
        )
    elif auxiliary_record["source"] != "TotalFieldScatteredField":
        raise ValueError("transparent TFSF auxiliary identity differs")

    for name in ("_sample_values", "_values", "_outer_values"):
        if not bool(np.isfinite(raw[name]).all()) or not bool(
            np.isfinite(expected_live[name]).all()
        ):
            raise ValueError(f"{batch_root}/{name} contains non-finite values")
        _require_exact_array(raw[name], expected_live[name], f"{batch_root}/{name}")


def _validate_expected_auxiliary_planners(archive, metadata, expected_steps, contract):
    if contract is None:
        return
    backend = metadata["backend_metadata"]
    expected_auxiliaries = contract["auxiliaries"]
    for step in expected_steps:
        if len(metadata["steps"][step]["sources"]["auxiliary"]) != len(
            expected_auxiliaries
        ):
            raise ValueError("transparent source auxiliary closure differs")
    for ordinal, auxiliary in enumerate(expected_auxiliaries):
        _validate_expected_planner(
            archive,
            auxiliary["component_plans"],
            space=auxiliary["space"],
            dt=float(np.asarray(archive["step/0/time"])[2]),
            precision="float64",
            bloch=(0.0, 0.0, 0.0) if backend["paired_real"] else None,
            planner_root=f"torch/auxiliary/{ordinal}/planner",
            plan_roots=tuple(
                f"torch/step/{step}/auxiliary/{ordinal}/state/plan"
                for step in sorted(expected_steps, key=int)
            ),
        )


def _validate_reference_transparent_sources(archive, metadata, contract):
    if contract is None:
        return
    expected_records = contract["native_records"]
    for step in ("0", *(str(value) for value in metadata["capture_steps"])):
        actual_records = metadata["steps"][step]["sources"]["updaters"]
        ordinals = {component: 0 for component in COMPONENT_NAMES}
        transparent_records = []
        for record in actual_records:
            ordinal = ordinals[record["component"]]
            ordinals[record["component"]] += 1
            if record["native_type"].startswith("Transparent"):
                transparent_records.append((ordinal, record))
        if len(transparent_records) != len(expected_records):
            raise ValueError("native transparent source record closure differs")
        time = float(np.asarray(archive[f"step/{step}/time"])[1])
        for (ordinal, record), expected in zip(
            transparent_records, expected_records, strict=True
        ):
            if (
                record["component"] != expected["component"]
                or record["native_type"] != expected["native_type"]
            ):
                raise ValueError("native transparent source identity differs")
            ordered = expected["parameters"]
            indices = np.asarray(
                [target for target, _parameter in ordered], dtype=np.intc
            )
            if not len(indices):
                indices = np.empty((0, 3), dtype=np.intc)
            values = np.asarray(
                [
                    value
                    for _target, parameter in ordered
                    for value in native_oracle._source_param_values(parameter, time)
                ],
                dtype=np.complex128,
            )
            root = (
                f"step/{step}/source/{record['component']}/"
                f"{ordinal}-{record['native_type']}"
            )
            _require_exact_array(archive[f"{root}/indices"], indices, f"{root}/indices")
            _require_exact_array(archive[f"{root}/values"], values, f"{root}/values")
            if record["cells"] != len(indices) or record["state_values"] != len(values):
                raise ValueError("native transparent source record counts differ")


def _validate_torch_candidate_archive(archive, manifest, *, transparent_contract=None):
    metadata = native_oracle._validate_archive(
        _BaseArchiveView(archive), manifest, "candidate"
    )
    if metadata["backend"] != "torch":
        raise ValueError("candidate correctness archive is not Torch")
    backend = metadata.get("backend_metadata")
    _exact_keys(
        backend,
        {
            "producer",
            "solver_abi",
            "cuda_graph_execution_representation",
            "device",
            "precision",
            "graph_mode",
            "compile_policy",
            "compile_mode",
            "resolved_device",
            "paired_real",
            "auxiliary_precisions",
            "manifest_contract_sha256",
            "input_archive",
            "input_step_zero_contract",
            "logical_map_source",
            "actual_geometry_and_coefficients",
            "plan_identity",
            "compile_cache_key",
            "cuda_graph_regions",
            "array_contract",
            "torch_array_bytes",
            "torch_arrays",
            "source_arrays",
        },
        "Torch correctness backend_metadata",
    )
    mode = _runtime_contract(
        backend["device"],
        backend["precision"],
        backend["graph_mode"],
        backend["compile_mode"],
    )
    if any(backend[key] != value for key, value in mode.items()):
        raise ValueError("Torch correctness runtime mode is inconsistent")
    if (
        backend["producer"] != PRODUCER
        or backend["solver_abi"] != gmes.torch_fdtd.TORCH_SOLVER_ABI
        or backend["cuda_graph_execution_representation"]
        != gmes.torch_fdtd.CUDA_GRAPH_EXECUTION_REPRESENTATION
        or backend["resolved_device"] != backend["device"]
        or type(backend["paired_real"]) is not bool
        or backend["paired_real"] is not bool(metadata["workload"].get("complex"))
        or backend["manifest_contract_sha256"] != _canonical_sha256(manifest)
        or backend["logical_map_source"] != "live-torch-plan"
        or backend["array_contract"] != TORCH_ARRAY_CONTRACT
        or not isinstance(backend["plan_identity"], str)
        or not backend["plan_identity"]
        or not isinstance(backend["compile_cache_key"], str)
        or not backend["compile_cache_key"]
    ):
        raise ValueError("Torch correctness backend identity is invalid")
    auxiliary_precisions = backend["auxiliary_precisions"]
    if not isinstance(auxiliary_precisions, list) or any(
        precision != "float64" for precision in auxiliary_precisions
    ):
        raise ValueError("Torch auxiliary precision metadata is invalid")
    _exact_keys(
        backend["input_archive"],
        {"sha256", "size_bytes", "media_type", "prefix"},
        "Torch correctness input_archive",
    )
    input_archive = backend["input_archive"]
    if (
        not _hex_string(input_archive["sha256"], 64)
        or type(input_archive["size_bytes"]) is not int
        or input_archive["size_bytes"] < 1
        or input_archive["media_type"] != "application/x-npz"
        or input_archive["prefix"] != "step/0"
    ):
        raise ValueError("Torch correctness input archive descriptor is invalid")
    if not _input_step_zero_contract_complete(
        backend["input_step_zero_contract"], manifest
    ):
        raise ValueError("Torch native step/0 input contract is invalid")
    expected_geometry = _initialized_geometry_metadata(
        metadata["workload"], np.asarray(archive["step/0/time"])[2]
    )
    if not native_oracle._same_json_value(
        backend["actual_geometry_and_coefficients"], expected_geometry
    ):
        raise ValueError("Torch correctness actual geometry differs from workload")
    graph_regions = backend["cuda_graph_regions"]
    if not isinstance(graph_regions, list) or not all(
        isinstance(value, str) and value for value in graph_regions
    ):
        raise ValueError("Torch correctness CUDA graph regions are invalid")
    if backend["device"] == "cpu" or backend["graph_mode"] == "eager":
        if graph_regions != []:
            raise ValueError("eager/CPU correctness cannot claim CUDA graphs")
    elif not graph_regions:
        raise ValueError("CUDA graph correctness has no captured graph regions")

    torch_keys = {key for key in archive.files if key.startswith("torch/")}
    _descriptor_map_complete(
        archive, backend["torch_arrays"], torch_keys, "Torch arrays"
    )
    if type(backend["torch_array_bytes"]) is not int or backend[
        "torch_array_bytes"
    ] != sum(value["size_bytes"] for value in backend["torch_arrays"].values()):
        raise ValueError("Torch array byte accounting is inaccurate")
    source_keys = {
        key
        for key in archive.files
        if len(key.split("/")) > 2 and key.split("/")[2] == "source"
    }
    _descriptor_map_complete(
        archive, backend["source_arrays"], source_keys, "Torch source arrays"
    )
    expected_steps = {"0"} | {str(value) for value in metadata["capture_steps"]}
    if transparent_contract is None:
        transparent_contract = _derive_transparent_source_contract(
            metadata["workload"],
            float(np.asarray(archive["step/0/time"])[2]),
            backend,
            metadata["reference"],
        )
    _validate_expected_primary_planner(archive, metadata, expected_steps)
    _validate_expected_auxiliary_planners(
        archive, metadata, expected_steps, transparent_contract
    )
    expected_field_dtype = np.dtype(
        "complex64"
        if backend["paired_real"] and backend["precision"] == "float32"
        else "complex128" if backend["paired_real"] else backend["precision"]
    )
    auxiliary_point_plans = {}
    for step in expected_steps:
        for component in COMPONENT_NAMES:
            live_key = f"torch/step/{step}/state/{component.lower()}"
            canonical_key = f"step/{step}/field/{component}"
            if live_key not in torch_keys:
                raise ValueError(f"Torch live field is absent: {live_key}")
            _live_field_value(
                archive[live_key],
                archive[canonical_key],
                backend["paired_real"],
                live_key,
            )
        _validate_live_clock(
            archive,
            f"torch/step/{step}/state",
            backend["precision"],
            archive[f"step/{step}/time"],
        )
        _validate_live_material_records(
            archive,
            metadata,
            step,
            metadata["steps"][step]["materials"],
            "state",
            "torch/planner",
            f"torch/step/{step}/state",
        )
        auxiliaries = metadata["steps"][step]["sources"]["auxiliary"]
        if len(auxiliaries) != len(auxiliary_precisions):
            raise ValueError("Torch auxiliary precision count is inconsistent")
        for ordinal, record in enumerate(auxiliaries):
            auxiliary_precision = record["backend_metadata"]["precision"]
            if auxiliary_precision != auxiliary_precisions[ordinal]:
                raise ValueError("Torch auxiliary precision is inconsistent")
            expected_auxiliary_dtype = np.dtype(auxiliary_precision)
            state_prefix = f"torch/step/{step}/auxiliary/{ordinal}/state"
            required_state_fields = {
                f"{state_prefix}/{component.lower()}" for component in COMPONENT_NAMES
            }
            if not required_state_fields.issubset(torch_keys):
                raise ValueError("Torch auxiliary raw state fields are incomplete")
            for component in COMPONENT_NAMES:
                planner_root = f"torch/auxiliary/{ordinal}/planner"
                plan_shape = np.asarray(
                    archive[f"{planner_root}/{component}/material_ids"]
                ).shape
                live_field_key = f"{state_prefix}/{component.lower()}"
                expected_live_shape = plan_shape + (
                    (2,) if backend["paired_real"] else ()
                )
                if (
                    archive[live_field_key].dtype != expected_auxiliary_dtype
                    or archive[live_field_key].shape != expected_live_shape
                ):
                    raise ValueError(
                        f"Torch auxiliary raw field contract differs for "
                        f"{live_field_key}"
                    )
                for suffix in ("material_ids", "underlying_ids"):
                    planner_key = (
                        f"torch/auxiliary/{ordinal}/planner/{component}/{suffix}"
                    )
                    live_key = f"{state_prefix}/plan/{suffix}_{component.lower()}"
                    planner = np.asarray(archive[planner_key])
                    live = np.asarray(archive[live_key])
                    if live.dtype != planner.dtype or not np.array_equal(live, planner):
                        raise ValueError(
                            "Torch auxiliary planner map differs from its live "
                            f"plan copy for {component}"
                        )
                _validate_planner_map_self_consistency(
                    archive,
                    planner_root,
                    component,
                    plan_shape,
                )
            for raw_prefix in (
                f"{state_prefix}/",
                f"torch/step/{step}/auxiliary/{ordinal}/sources/",
            ):
                for key in torch_keys:
                    if not key.startswith(raw_prefix):
                        continue
                    dtype = archive[key].dtype
                    if np.issubdtype(dtype, np.inexact) and dtype != (
                        expected_auxiliary_dtype
                    ):
                        raise ValueError(
                            f"Torch auxiliary raw precision differs for {key}"
                        )
            _validate_live_clock(
                archive,
                state_prefix,
                auxiliary_precision,
                archive[f"step/{step}/source_aux/{ordinal}-{record['source']}/time"],
            )
            _validate_live_material_records(
                archive,
                metadata,
                step,
                record["materials"],
                f"source_aux_material/{ordinal}",
                f"torch/auxiliary/{ordinal}/planner",
                state_prefix,
                record["fields"],
                auxiliary_precision,
            )
            active_components = set(record["backend_metadata"]["canonical_components"])
            _validate_zero_live_material_representations(
                archive,
                f"torch/auxiliary/{ordinal}/planner",
                state_prefix,
                set(COMPONENT_NAMES) - active_components,
                backend["paired_real"],
                auxiliary_precision,
            )
            auxiliary_point_plan = _validate_auxiliary_point_source_batch(
                archive,
                torch_keys,
                metadata,
                step,
                ordinal,
                record,
            )
            baseline = auxiliary_point_plans.setdefault(
                ordinal,
                tuple(value.copy() for value in auxiliary_point_plan),
            )
            if any(
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not np.array_equal(actual, expected)
                for actual, expected in zip(auxiliary_point_plan, baseline, strict=True)
            ):
                raise ValueError(
                    "Torch auxiliary PointSource plan differs across captures"
                )
            auxiliary_field_dtype = np.dtype(
                "complex128" if backend["paired_real"] else auxiliary_precision
            )
            for component in COMPONENT_NAMES:
                key = (
                    f"step/{step}/source_aux/{ordinal}-{record['source']}/"
                    f"field/{component}"
                )
                if archive[key].dtype != auxiliary_field_dtype:
                    raise ValueError(
                        f"Torch auxiliary field precision differs for {key}"
                    )
                live_key = f"{state_prefix}/{component.lower()}"
                canonical = np.asarray(archive[key])
                if component in active_components:
                    _live_field_value(
                        archive[live_key],
                        canonical,
                        backend["paired_real"],
                        live_key,
                    )
                elif (
                    canonical.shape != (1, 1, 1)
                    or np.count_nonzero(canonical)
                    or np.count_nonzero(archive[live_key])
                ):
                    raise ValueError(
                        f"Torch inactive auxiliary field is not zero: {live_key}"
                    )
        source_batch_root = f"torch/step/{step}/sources/batches/"
        updater_records = metadata["steps"][step]["sources"]["updaters"]
        batch_suffixes = {}
        for key in torch_keys:
            if not key.startswith(source_batch_root):
                continue
            remainder = key.removeprefix(source_batch_root)
            parts = remainder.split("/")
            if (
                len(parts) != 2
                or not parts[0].isdigit()
                or parts[0] != str(int(parts[0]))
            ):
                raise ValueError("Torch source batch array key is not canonical")
            batch_suffixes.setdefault(int(parts[0]), set()).add(parts[1])
        record_ordinals = [
            record["backend_metadata"]["batch_ordinal"] for record in updater_records
        ]
        if (
            len(record_ordinals) != len(set(record_ordinals))
            or set(record_ordinals) != set(batch_suffixes)
            or set(record_ordinals) != set(range(len(record_ordinals)))
        ):
            raise ValueError("Torch live source batch closure differs")
        component_ordinals = {component: 0 for component in COMPONENT_NAMES}
        for record in updater_records:
            component = record["component"]
            record_ordinal = component_ordinals[component]
            component_ordinals[component] += 1
            batch_ordinal = record["backend_metadata"]["batch_ordinal"]
            suffixes = batch_suffixes[batch_ordinal]
            if record["native_type"].startswith("PointSource"):
                if suffixes != set(_POINT_SOURCE_LIVE_ARRAYS):
                    raise ValueError("Torch PointSource live buffer closure differs")
                _validate_point_source_batch(
                    archive,
                    metadata,
                    step,
                    record,
                    record_ordinal,
                )
            elif record["native_type"].startswith("Transparent"):
                expected_suffixes = {
                    "targets",
                    "samples",
                    "weights",
                    "_sample_values",
                    "_values",
                    "_outer_values",
                }
                if metadata["workload"].get("source") == "gaussian":
                    expected_suffixes.update(
                        {"_envelope_step", "_envelope_step_offset", "_envelope"}
                    )
                if suffixes != expected_suffixes:
                    raise ValueError("Torch transparent live buffer closure differs")
                _validate_transparent_batch_semantics(
                    archive,
                    metadata,
                    step,
                    record,
                    record_ordinal,
                    transparent_contract,
                )
            else:
                raise ValueError("Torch source batch type is unsupported")
        transparent_batch_prefixes = set()
        for key in torch_keys:
            if not key.startswith(source_batch_root):
                continue
            remainder = key.removeprefix(source_batch_root)
            parts = remainder.split("/")
            if (
                len(parts) == 2
                and parts[0].isdigit()
                and parts[0] == str(int(parts[0]))
                and parts[1] == "weights"
            ):
                transparent_batch_prefixes.add(f"{source_batch_root}{parts[0]}")
        transparent_updaters = sum(
            record["native_type"].startswith("Transparent")
            for record in metadata["steps"][step]["sources"]["updaters"]
        )
        if len(transparent_batch_prefixes) != transparent_updaters:
            raise ValueError("Torch transparent raw batch count is inconsistent")
        for batch_prefix in transparent_batch_prefixes:
            expected_dtypes = {
                "targets": np.dtype("int64"),
                "samples": np.dtype("int64"),
                "weights": np.dtype("float64"),
                "_sample_values": np.dtype("float64"),
                "_values": np.dtype("float64"),
                "_outer_values": np.dtype(backend["precision"]),
            }
            for name, expected_dtype in expected_dtypes.items():
                key = f"{batch_prefix}/{name}"
                if key not in torch_keys:
                    raise ValueError(
                        f"Torch transparent raw batch is incomplete: {batch_prefix}"
                    )
                if archive[key].dtype != expected_dtype:
                    raise ValueError(
                        f"Torch transparent raw precision differs for {key}"
                    )
            envelope_dtypes = {
                "_envelope_step": np.dtype("int64"),
                "_envelope_step_offset": np.dtype("int64"),
                "_envelope": np.dtype("float64"),
            }
            envelope_keys = {f"{batch_prefix}/{name}" for name in envelope_dtypes}
            present_envelope_keys = envelope_keys & torch_keys
            expected_envelope_keys = (
                envelope_keys
                if metadata["workload"].get("source") == "gaussian"
                else set()
            )
            if present_envelope_keys != expected_envelope_keys:
                raise ValueError(
                    f"Torch Gaussian envelope raw state is incomplete: {batch_prefix}"
                )
            for name, expected_dtype in envelope_dtypes.items():
                key = f"{batch_prefix}/{name}"
                if key in torch_keys and archive[key].dtype != expected_dtype:
                    raise ValueError(
                        f"Torch Gaussian envelope raw precision differs for {key}"
                    )
        for component in COMPONENT_NAMES:
            if archive[f"step/{step}/field/{component}"].dtype != expected_field_dtype:
                raise ValueError(
                    f"Torch field precision differs for step/{step}/field/{component}"
                )

    for component in COMPONENT_NAMES:
        material_ids, underlying_ids = _project_logical_map_ids(
            archive[f"torch/planner/{component}/material_ids"],
            archive[f"torch/planner/{component}/underlying_ids"],
            backend["actual_geometry_and_coefficients"],
            metadata["geometry_and_coefficients"],
        )
        for suffix, projected in (
            ("material_ids", material_ids),
            ("underlying_ids", underlying_ids),
        ):
            planner = np.asarray(archive[f"torch/planner/{component}/{suffix}"])
            mirror_keys = {
                f"torch/plan/{suffix}_{component.lower()}",
                *(
                    f"torch/step/{step}/state/plan/" f"{suffix}_{component.lower()}"
                    for step in expected_steps
                ),
            }
            for mirror_key in mirror_keys:
                mirror = np.asarray(archive[mirror_key])
                if mirror.dtype != planner.dtype or not np.array_equal(mirror, planner):
                    raise ValueError(
                        "Torch planner map differs from its live plan copies for "
                        f"{component}"
                    )
            canonical = np.asarray(archive[f"map/{component}/{suffix}"]).reshape(-1)
            if canonical.dtype != projected.dtype or not np.array_equal(
                canonical, projected
            ):
                raise ValueError(
                    f"Torch logical map differs from planner semantics for {component}"
                )
    if not _candidate_source_metadata_complete(metadata, torch_keys):
        raise ValueError("Torch source topology metadata is invalid")
    _validate_point_source_capture_invariance(archive, metadata)
    return metadata


def _candidate_source_metadata_complete(metadata, torch_keys):
    try:
        for step in ("0", *(str(value) for value in metadata["capture_steps"])):
            sources = metadata["steps"][step]["sources"]
            for record in sources["updaters"]:
                backend = record["backend_metadata"]
                _exact_keys(
                    backend,
                    {
                        "producer",
                        "representation",
                        "source_origin",
                        "batch_ordinal",
                    },
                    "Torch source backend_metadata",
                )
                native_type = record["native_type"]
                representation = (
                    "torch-point-source-batch-v1"
                    if native_type.startswith("PointSource")
                    else (
                        "torch-transparent-source-batch-v1"
                        if native_type.startswith("Transparent")
                        else None
                    )
                )
                if (
                    backend["producer"] != PRODUCER
                    or backend["source_origin"] != "live-torch-source-batch"
                    or backend["representation"] != representation
                    or type(backend["batch_ordinal"]) is not int
                    or backend["batch_ordinal"] < 0
                ):
                    return False
            for ordinal, record in enumerate(sources["auxiliary"]):
                backend = record["backend_metadata"]
                _exact_keys(
                    backend,
                    {
                        "producer",
                        "representation",
                        "canonical_components",
                        "precision",
                        "raw_state_origin",
                    },
                    "Torch auxiliary backend_metadata",
                )
                components = backend["canonical_components"]
                if (
                    backend["producer"] != PRODUCER
                    or backend["representation"] != "live-torch-auxiliary-simulation-v1"
                    or backend["precision"] != "float64"
                    or backend["raw_state_origin"] != "torch-named-buffers"
                    or not isinstance(components, list)
                    or components
                    != [name for name in COMPONENT_NAMES if name in components]
                    or not components
                ):
                    return False
                for component in COMPONENT_NAMES:
                    canonical = component in components
                    if (record["fields"][component] != [1, 1, 1]) is not canonical:
                        return False
                    key = (
                        f"torch/step/{step}/auxiliary/{ordinal}/state/"
                        f"{component.lower()}"
                    )
                    if key not in torch_keys:
                        return False
        return True
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def _runtime_mode(metadata):
    backend = metadata["backend_metadata"]
    return {
        key: backend[key]
        for key in (
            "device",
            "precision",
            "graph_mode",
            "compile_policy",
            "compile_mode",
        )
    }


_MATERIAL_TOPOLOGY_KEYS = (
    "component",
    "strategies",
    "native_type",
    "cells",
    "state_values",
    "state_width",
    "state_key",
    "bucket_signature",
)

_STRATEGY_TOLERANCE_MODEL = {
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


def _material_topology(records):
    return [{key: record[key] for key in _MATERIAL_TOPOLOGY_KEYS} for record in records]


def _material_topology_matches(reference_metadata, candidate_metadata):
    for step in ("0", *(str(value) for value in reference_metadata["capture_steps"])):
        expected = reference_metadata["steps"][step]
        actual = candidate_metadata["steps"][step]
        if not native_oracle._same_json_value(
            _material_topology(expected["materials"]),
            _material_topology(actual["materials"]),
        ):
            return False
        expected_auxiliary = expected["sources"]["auxiliary"]
        actual_auxiliary = actual["sources"]["auxiliary"]
        if len(expected_auxiliary) != len(actual_auxiliary):
            return False
        for left, right in zip(expected_auxiliary, actual_auxiliary, strict=True):
            if not native_oracle._same_json_value(
                _material_topology(left["materials"]),
                _material_topology(right["materials"]),
            ):
                return False
    return True


def _reference_strategies(metadata):
    strategies = set()
    for step in ("0", *(str(value) for value in metadata["capture_steps"])):
        snapshot = metadata["steps"][step]
        records = list(snapshot["materials"])
        for auxiliary in snapshot["sources"]["auxiliary"]:
            records.extend(auxiliary["materials"])
        for record in records:
            strategies.update(record["strategies"])
    return strategies


def _manifest_tolerance(manifest, key, dtype, strategies, workload_name):
    if "/source_aux/" in key:
        source = (
            manifest["tolerances"]["torch"]
            .get("source_auxiliary", {})
            .get(workload_name, {})
            .get(dtype)
        )
        if source is not None:
            if not isinstance(source, dict) or set(source) != {"rtol", "atol"}:
                raise ValueError("source/auxiliary tolerance is invalid")
            values = {name: float(source[name]) for name in ("rtol", "atol")}
            if any(not np.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError("source/auxiliary tolerance is invalid")
            return {
                **values,
                "scope": f"source_auxiliary/{workload_name}/{dtype}",
            }

    selected = set(strategies)
    for segment in key.split("/"):
        if "-" not in segment:
            continue
        candidate = segment.split("-", 1)[1]
        if candidate in selected:
            selected = {candidate}
            break
    models = {
        _STRATEGY_TOLERANCE_MODEL[strategy]
        for strategy in selected
        if strategy != "Dummy"
    }
    scope = None
    if not models:
        parts = key.split("/")
        dummy_source_numerics = (
            selected == {"Dummy"}
            and workload_name == "dummy"
            and len(parts) > 3
            and parts[0] == "step"
            and parts[1].isdigit()
            and parts[2] in {"field", "physical"}
        )
        if dummy_source_numerics:
            # CPython and compiled Torch can use different libm oscillator
            # kernels, so the driven cell may differ at the ULP scale. Reuse
            # the pinned non-dispersive tolerance without relaxing topology.
            models = {"dielectric"}
            scope = f"dummy-source-numerics/dielectric/{dtype}"
        else:
            return {"rtol": 0.0, "atol": 0.0, "scope": "exact/dummy"}
    tolerances = []
    for model in sorted(models):
        model_dtype = "float64" if model == "dm2" and dtype == "complex128" else dtype
        try:
            tolerance = manifest["tolerances"]["torch"][model][model_dtype]
            if set(tolerance) != {"rtol", "atol"}:
                raise KeyError(model_dtype)
            values = {name: float(tolerance[name]) for name in ("rtol", "atol")}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"manifest has no pinned {model_dtype} tolerance for {model}"
            ) from error
        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f"manifest tolerance is invalid for {model}")
        tolerances.append(values)
    result = {
        name: max(tolerance[name] for tolerance in tolerances)
        for name in ("rtol", "atol")
    }
    result["scope"] = scope or f"strategies/{','.join(sorted(models))}/{dtype}"
    return result


def _source_topology_matches(reference_metadata, candidate_metadata):
    for step in ("0", *(str(value) for value in reference_metadata["capture_steps"])):
        expected = reference_metadata["steps"][step]["sources"]
        actual = candidate_metadata["steps"][step]["sources"]
        expected_updaters = expected["updaters"]
        actual_updaters = actual["updaters"]
        if len(expected_updaters) != len(actual_updaters):
            return False
        for left, right in zip(expected_updaters, actual_updaters, strict=True):
            if {key: left[key] for key in ("component", "native_type", "cells")} != {
                key: right[key] for key in ("component", "native_type", "cells")
            }:
                return False
            backend = right.get("backend_metadata")
            if (
                not isinstance(backend, dict)
                or backend.get("producer") != PRODUCER
                or backend.get("source_origin") != "live-torch-source-batch"
            ):
                return False
        expected_aux = expected["auxiliary"]
        actual_aux = actual["auxiliary"]
        if len(expected_aux) != len(actual_aux):
            return False
        for left, right in zip(expected_aux, actual_aux, strict=True):
            if left["source"] != right["source"] or left["fields"] != right["fields"]:
                return False
            backend = right.get("backend_metadata")
            if (
                not isinstance(backend, dict)
                or backend.get("producer") != PRODUCER
                or backend.get("representation") != "live-torch-auxiliary-simulation-v1"
            ):
                return False
    return True


def _comparison_precision(candidate_metadata, key):
    parts = key.split("/")
    if (
        len(parts) > 3
        and parts[0] == "step"
        and parts[2] in {"source_aux", "source_aux_material"}
    ):
        ordinal_value = parts[3].split("-", 1)[0]
        if ordinal_value.isdigit():
            auxiliaries = candidate_metadata["steps"][parts[1]]["sources"]["auxiliary"]
            ordinal = int(ordinal_value)
            if ordinal < len(auxiliaries):
                return auxiliaries[ordinal]["backend_metadata"]["precision"]
    return candidate_metadata["backend_metadata"]["precision"]


def _comparison_dtype(candidate_metadata, key, expected, actual):
    if _comparison_precision(candidate_metadata, key) == "float32":
        return "float32"
    if np.iscomplexobj(expected) or np.iscomplexobj(actual):
        return "complex128"
    return "float64"


def _is_source_array(key):
    parts = key.split("/")
    return len(parts) > 2 and parts[2] == "source"


def compare_torch_archives(
    reference_path, candidate_path, manifest, *, include_tolerances=False
):
    """Fail closed while strictly comparing two untrusted evidence archives."""
    try:
        return _compare_torch_archives_loaded(
            reference_path,
            candidate_path,
            manifest,
            include_tolerances=include_tolerances,
        )
    except (
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        UnicodeError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        result = {
            "passed": False,
            "failures": [{"key": "archive/container", "error": str(error)}],
        }
        if include_tolerances:
            result["tolerance_results"] = []
        return result


def _compare_torch_archives_loaded(
    reference_path, candidate_path, manifest, *, include_tolerances=False
):
    """Strictly compare native schema-2 state with a Torch schema-2 archive."""
    failures = []
    tolerance_results = []
    with ExitStack() as stack:
        reference = stack.enter_context(_open_bounded_npz(reference_path))
        candidate = stack.enter_context(_open_bounded_npz(candidate_path))
        try:
            reference_metadata = native_oracle._validate_archive(
                reference, manifest, "reference"
            )
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            failures.append({"key": "reference/archive-contract", "error": str(error)})
            reference_metadata = None
        try:
            candidate_metadata = _validate_torch_candidate_archive(candidate, manifest)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            failures.append({"key": "candidate/archive-contract", "error": str(error)})
            candidate_metadata = None
        if reference_metadata is None or candidate_metadata is None:
            return {"passed": False, "failures": failures}
        try:
            transparent_contract = _derive_transparent_source_contract(
                candidate_metadata["workload"],
                float(np.asarray(candidate["step/0/time"])[2]),
                candidate_metadata["backend_metadata"],
                candidate_metadata["reference"],
            )
            _validate_reference_transparent_sources(
                reference, reference_metadata, transparent_contract
            )
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            failures.append(
                {
                    "key": "reference/source-contract",
                    "error": str(error),
                }
            )
        input_archive = candidate_metadata["backend_metadata"]["input_archive"]
        if (
            input_archive["sha256"] != _sha256(reference_path)
            or input_archive["size_bytes"] != Path(reference_path).stat().st_size
        ):
            failures.append(
                {"key": "candidate/input_archive", "error": "reference bytes differ"}
            )
        reference_inputs = {
            key: _array_descriptor(reference[key])
            for key in sorted(reference.files)
            if key.startswith(("map/", "step/0/"))
        }
        input_contract = candidate_metadata["backend_metadata"][
            "input_step_zero_contract"
        ]
        if not native_oracle._same_json_value(
            input_contract["arrays"], reference_inputs
        ) or input_contract["array_bytes"] != sum(
            value["size_bytes"] for value in reference_inputs.values()
        ):
            failures.append(
                {
                    "key": "candidate/input_step_zero_contract",
                    "error": "native step/0 or map array descriptors differ",
                }
            )
        if not native_oracle._same_json_value(
            candidate_metadata["geometry_and_coefficients"],
            reference_metadata["geometry_and_coefficients"],
        ):
            failures.append(
                {
                    "key": "geometry_and_coefficients",
                    "error": "logical geometry and coefficients differ",
                }
            )
        if not _source_topology_matches(reference_metadata, candidate_metadata):
            failures.append(
                {"key": "source/topology", "error": "Torch source topology differs"}
            )
        if not _material_topology_matches(reference_metadata, candidate_metadata):
            failures.append(
                {
                    "key": "material/topology",
                    "error": "Torch material topology differs",
                }
            )
        for component in COMPONENT_NAMES:
            for suffix in ("material_ids", "underlying_ids"):
                key = f"map/{component}/{suffix}"
                expected = np.asarray(reference[key])
                actual = np.asarray(candidate[key])
                if expected.dtype != actual.dtype or not np.array_equal(
                    expected, actual
                ):
                    failures.append(
                        {
                            "key": key,
                            "error": "Torch logical map differs from native reference",
                        }
                    )
            try:
                _validate_planner_map_self_consistency(
                    candidate,
                    "torch/planner",
                    component,
                    candidate_metadata["maps"][component]["shape"],
                )
            except (IndexError, KeyError, TypeError, ValueError) as error:
                failures.append(
                    {
                        "key": f"torch/planner/{component}/map-consistency",
                        "error": str(error),
                    }
                )
        strategies = _reference_strategies(reference_metadata)
        reference_source_keys = {
            key for key in reference.files if _is_source_array(key)
        }
        candidate_source_keys = {
            key for key in candidate.files if _is_source_array(key)
        }
        if reference_source_keys != candidate_source_keys:
            failures.append(
                {
                    "key": "source/array-topology",
                    "missing": sorted(reference_source_keys - candidate_source_keys),
                    "unexpected": sorted(candidate_source_keys - reference_source_keys),
                }
            )
        for key in sorted(reference_source_keys & candidate_source_keys):
            # Native source values are dynamic complex observables, whereas
            # Torch PointSource values are static packed plan words. The Torch
            # values are checked exactly against live semantics and across
            # captures by the candidate validator above.
            if not key.endswith("/indices"):
                continue
            if not np.array_equal(reference[key], candidate[key]):
                failures.append({"key": key, "error": "source indices differ"})

        reference_keys = {
            key
            for key in reference.files
            if key != "metadata.json"
            and not key.startswith("map/")
            and not _is_source_array(key)
        }
        candidate_keys = {
            key
            for key in candidate.files
            if key != "metadata.json"
            and not key.startswith(("map/", "torch/"))
            and not _is_source_array(key)
        }
        if reference_keys != candidate_keys:
            failures.append(
                {
                    "key": "archive/topology",
                    "missing": sorted(reference_keys - candidate_keys),
                    "unexpected": sorted(candidate_keys - reference_keys),
                }
            )
        for key in sorted(reference_keys & candidate_keys):
            expected = reference[key]
            actual = candidate[key]
            same_shape = expected.shape == actual.shape
            if np.issubdtype(expected.dtype, np.integer):
                equal = same_shape and np.array_equal(expected, actual)
                tolerance = {
                    "rtol": 0.0,
                    "atol": 0.0,
                    "scope": "exact/integer",
                }
                dtype_label = str(expected.dtype)
            else:
                dtype = _comparison_dtype(candidate_metadata, key, expected, actual)
                dtype_label = dtype
                try:
                    tolerance = _manifest_tolerance(
                        manifest,
                        key,
                        dtype,
                        strategies,
                        reference_metadata["workload"]["name"],
                    )
                except ValueError as error:
                    failures.append({"key": key, "error": str(error)})
                    continue
                equal = same_shape and np.allclose(
                    expected,
                    actual,
                    rtol=tolerance["rtol"],
                    atol=tolerance["atol"],
                    equal_nan=False,
                )
            if expected.shape == actual.shape:
                if np.issubdtype(expected.dtype, np.integer):
                    difference = np.abs(
                        expected.astype(np.float64) - actual.astype(np.float64)
                    )
                else:
                    difference = np.abs(expected - actual)
            else:
                difference = np.asarray([np.inf])
            tolerance_results.append(
                {
                    "key": key,
                    "dtype": dtype_label,
                    "scope": tolerance["scope"],
                    "rtol": tolerance["rtol"],
                    "atol": tolerance["atol"],
                    "max_abs_error": (
                        float(np.max(difference)) if difference.size else 0.0
                    ),
                }
            )
            if not equal:
                failures.append(
                    {
                        "key": key,
                        "expected_shape": list(expected.shape),
                        "actual_shape": list(actual.shape),
                        "max_abs_error": float(np.max(difference)),
                        **tolerance,
                    }
                )
    result = {"passed": not failures, "failures": failures}
    if include_tolerances:
        result["tolerance_results"] = tolerance_results
    return result


def _archive_record(path, manifest, role):
    path = Path(path).resolve(strict=True)
    with _open_bounded_npz(path) as archive:
        metadata = (
            _validate_torch_candidate_archive(archive, manifest)
            if role == "candidate"
            else native_oracle._validate_archive(archive, manifest, role)
        )
    return path, metadata


def _descriptor_candidate_evidence(candidate_evidence):
    value = {
        key: candidate_evidence.get(key)
        for key in (
            "candidate_git_commit",
            "candidate_git_status",
            "manifest_sha256",
        )
    }
    if (
        not _hex_string(value["candidate_git_commit"], 40)
        or value["candidate_git_status"] != ""
        or not _hex_string(value["manifest_sha256"], 64)
    ):
        raise ValueError("candidate evidence has no portable three-key binding")
    return value


def _canonical_descriptor_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or (len(value) >= 2 and value[0].isalpha() and value[1] == ":")
    ):
        raise ValueError("artifact descriptor path is not canonical POSIX")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact descriptor path contains a dot or empty segment")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("artifact descriptor path must be canonical and relative")
    return path


def _descriptor_root(path):
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("correctness descriptor root must be a directory")
    return root


def _relative_descriptor_path(path, root):
    try:
        relative = Path(path).resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise ValueError("correctness artifact is outside descriptor root") from error
    value = PurePosixPath(*relative.parts).as_posix()
    _canonical_descriptor_path(value)
    return value


def _artifact_descriptor(path, root, candidate_evidence, media_type):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("correctness artifact must be a regular file")
    return {
        "path": _relative_descriptor_path(path, root),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "media_type": media_type,
        "candidate_evidence": _descriptor_candidate_evidence(candidate_evidence),
    }


def _runtime_receipt_descriptor(path, root):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("runtime publication receipt must be a regular file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_CORRECTNESS_JSON_BYTES:
        raise ValueError("runtime publication receipt exceeds the JSON byte bound")
    return {
        "path": _relative_descriptor_path(path, root),
        "sha256": _sha256(path),
        "size_bytes": size,
        "media_type": "application/json",
    }


def _runtime_receipt_file_identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("runtime publication receipt must be a regular file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_CORRECTNESS_JSON_BYTES:
        raise ValueError("runtime publication receipt exceeds the JSON byte bound")
    return {
        "sha256": _sha256(path),
        "size_bytes": size,
        "media_type": "application/json",
    }


def _runtime_receipt_candidates(artifacts):
    return [
        {
            "case": artifact["case"],
            "sha256": artifact["candidate"]["sha256"],
            "size_bytes": artifact["candidate"]["size_bytes"],
        }
        for artifact in artifacts
    ]


def runtime_publication_receipt_complete(
    receipt,
    manifest,
    expected_evidence,
    runtime_mode,
    candidate_archives,
):
    """Validate one publication-owned receipt against exact local archives."""
    try:
        _exact_keys(
            receipt,
            {
                "schema_version",
                "kind",
                "final_sha",
                "manifest_sha256",
                "workflow",
                "profiler_witness",
                "runtime_mode",
                "candidate_archives",
            },
            "runtime publication receipt",
        )
        workflow = receipt["workflow"]
        witness = receipt["profiler_witness"]
        _exact_keys(
            workflow,
            {"repository", "run_id", "run_attempt", "job_id", "job_name"},
            "runtime publication workflow",
        )
        _exact_keys(
            witness,
            {"name", "sha256", "size_bytes", "media_type"},
            "runtime profiler witness",
        )
        archives = receipt["candidate_archives"]
        if not isinstance(archives, list):
            return False
        for archive in archives:
            _exact_keys(
                archive,
                {"case", "sha256", "size_bytes"},
                "runtime receipt candidate archive",
            )
            if (
                not isinstance(archive["case"], str)
                or not archive["case"]
                or not _hex_string(archive["sha256"], 64)
                or type(archive["size_bytes"]) is not int
                or archive["size_bytes"] <= 0
                or archive["size_bytes"] > MAX_CORRECTNESS_NPZ_BYTES
            ):
                return False
        job_name = workflow["job_name"]
        witness_name = witness["name"]
        media_type = witness["media_type"]
        return (
            type(receipt["schema_version"]) is int
            and receipt["schema_version"] == 1
            and receipt["kind"] == RUNTIME_RECEIPT_KIND
            and receipt["final_sha"] == expected_evidence["candidate_git_commit"]
            and _hex_string(receipt["final_sha"], 40)
            and receipt["manifest_sha256"] == expected_evidence["manifest_sha256"]
            and _hex_string(receipt["manifest_sha256"], 64)
            and receipt["manifest_sha256"] == TRUSTED_MANIFEST_SHA256
            and _load_trusted_manifest(DEFAULT_MANIFEST)[1] == TRUSTED_MANIFEST_SHA256
            and workflow["repository"] == "ruddyscent/gmes"
            and all(
                type(workflow[name]) is int and workflow[name] > 0
                for name in ("run_id", "run_attempt", "job_id")
            )
            and isinstance(job_name, str)
            and 0 < len(job_name) <= 256
            and job_name == job_name.strip()
            and not any(ord(character) < 32 for character in job_name)
            and isinstance(witness_name, str)
            and 0 < len(witness_name) <= 256
            and witness_name == PurePosixPath(witness_name).name
            and witness_name not in {".", ".."}
            and "\\" not in witness_name
            and "\x00" not in witness_name
            and _hex_string(witness["sha256"], 64)
            and type(witness["size_bytes"]) is int
            and 0 < witness["size_bytes"] <= MAX_CORRECTNESS_NPZ_BYTES
            and isinstance(media_type, str)
            and media_type == media_type.lower()
            and media_type.count("/") == 1
            and not any(character.isspace() for character in media_type)
            and _runtime_mode_complete(receipt["runtime_mode"])
            and native_oracle._same_json_value(receipt["runtime_mode"], runtime_mode)
            and native_oracle._same_json_value(archives, candidate_archives)
        )
    except AttributeError, KeyError, OSError, TypeError, ValueError, OverflowError:
        return False


def load_runtime_publication_receipt(
    path,
    manifest,
    expected_evidence,
    runtime_mode,
    candidate_archives,
):
    """Load canonical trusted receipt bytes and bind them to one matrix."""
    try:
        receipt = _load_bounded_json(
            path,
            "runtime publication receipt",
            require_canonical=True,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("runtime publication receipt is not valid JSON") from error
    if not runtime_publication_receipt_complete(
        receipt,
        manifest,
        expected_evidence,
        runtime_mode,
        candidate_archives,
    ):
        raise ValueError("runtime publication receipt differs from local evidence")
    return receipt


def _resolve_artifact_descriptor(root, descriptor):
    portable = _canonical_descriptor_path(descriptor["path"])
    path = root.joinpath(*portable.parts).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("correctness artifact escapes descriptor root") from error
    if not path.is_file():
        raise ValueError("correctness artifact must be a regular file")
    return path


def build_correctness_evidence_index(
    reference_paths,
    candidate_paths,
    manifest,
    candidate_evidence,
    *,
    descriptor_root,
    runtime_receipt,
):
    """Recompare the exact correctness matrix and return a deterministic index."""
    if not isinstance(candidate_evidence, dict):
        raise ValueError("candidate evidence must be an object")
    descriptor_root = _descriptor_root(descriptor_root)
    _descriptor_candidate_evidence(candidate_evidence)
    commit = candidate_evidence.get("candidate_git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or candidate_evidence.get("candidate_git_status") != ""
    ):
        raise ValueError("candidate evidence must identify one clean full commit")

    references = {}
    for path in reference_paths:
        resolved, metadata = _archive_record(path, manifest, "reference")
        name = metadata["workload"]["name"]
        if name in references:
            raise ValueError(f"duplicate reference archive for {name}")
        references[name] = (resolved, metadata)
    candidates = {}
    for path in candidate_paths:
        resolved, metadata = _archive_record(path, manifest, "candidate")
        name = metadata["workload"]["name"]
        if name in candidates:
            raise ValueError(f"duplicate candidate archive for {name}")
        if metadata["backend"] != "torch":
            raise ValueError(f"candidate archive for {name} is not Torch")
        backend = metadata.get("backend_metadata", {})
        if backend.get("producer") != PRODUCER or backend.get(
            "solver_abi"
        ) != candidate_evidence.get("solver_abi"):
            raise ValueError(f"candidate backend metadata differs for {name}")
        provenance = metadata["provenance"]
        if any(
            provenance[key]["commit"] != commit
            or provenance[key]["git_status"] != ""
            or provenance[key]["clean"] is not True
            for key in ("source", "controller")
        ):
            raise ValueError(f"candidate provenance differs for {name}")
        candidates[name] = (resolved, metadata)

    required = _required_cases(manifest)
    required_names = [case["name"] for _group, case in required]
    if set(references) != set(required_names) or set(candidates) != set(required_names):
        raise ValueError(
            "correctness evidence archives do not exactly cover correctness and "
            "physical_checks"
        )
    runtime_modes = [_runtime_mode(metadata) for _path, metadata in candidates.values()]
    runtime_mode = runtime_modes[0]
    if any(
        not native_oracle._same_json_value(value, runtime_mode)
        for value in runtime_modes[1:]
    ):
        raise ValueError("candidate correctness archives mix runtime modes")
    artifacts = []
    for group, case in required:
        name = case["name"]
        reference_path, reference_metadata = references[name]
        candidate_path, candidate_metadata = candidates[name]
        comparison = compare_torch_archives(
            reference_path,
            candidate_path,
            manifest,
            include_tolerances=True,
        )
        tolerance_results = comparison.pop("tolerance_results", None)
        if comparison.get("passed") is not True or comparison.get("failures") != []:
            raise ValueError(f"Torch correctness comparison failed for {name}")
        if not _tolerance_results_complete(tolerance_results):
            raise ValueError(f"Torch tolerance results are invalid for {name}")
        artifacts.append(
            {
                "case": name,
                "group": group,
                "reference": _artifact_descriptor(
                    reference_path,
                    descriptor_root,
                    candidate_evidence,
                    "application/x-npz",
                ),
                "reference_observer_commit": reference_metadata["provenance"]["source"][
                    "commit"
                ],
                "candidate": _artifact_descriptor(
                    candidate_path,
                    descriptor_root,
                    candidate_evidence,
                    "application/x-npz",
                ),
                "candidate_provenance": {
                    "commit": candidate_metadata["provenance"]["source"]["commit"],
                    "source_sha256": candidate_metadata["provenance"]["source"][
                        "source_sha256"
                    ],
                    "controller_sha256": candidate_metadata["provenance"]["controller"][
                        "source_sha256"
                    ],
                },
                "comparison": comparison,
                "tolerance_results": tolerance_results,
            }
        )
    receipt_path = Path(runtime_receipt).resolve(strict=True)
    receipt_candidates = _runtime_receipt_candidates(artifacts)
    load_runtime_publication_receipt(
        receipt_path,
        manifest,
        candidate_evidence,
        runtime_mode,
        receipt_candidates,
    )
    return {
        "schema_version": 2,
        "kind": INDEX_KIND,
        "contract_id": INDEX_CONTRACT,
        "manifest_contract_sha256": _canonical_sha256(manifest),
        "candidate_evidence": candidate_evidence,
        "runtime_mode": runtime_mode,
        "runtime_receipt": _runtime_receipt_descriptor(receipt_path, descriptor_root),
        "required_cases": required_names,
        "artifacts": artifacts,
        "suite_acceptance": {
            "correctness_case_count": len(manifest.get("correctness", ())),
            "physical_check_case_count": len(manifest.get("physical_checks", ())),
            "evaluated_case_count": len(artifacts),
            "complete_fields": True,
            "persistent_state": True,
            "source_and_auxiliary_state": True,
            "physical_observables": True,
            "passed": True,
        },
    }


def _hex_string(value, width):
    return (
        isinstance(value, str)
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value)
    )


def _portable_descriptor_complete(descriptor, expected_evidence, media_type):
    try:
        _exact_keys(
            descriptor,
            {"path", "sha256", "size_bytes", "media_type", "candidate_evidence"},
            "correctness artifact descriptor",
        )
        _canonical_descriptor_path(descriptor["path"])
        return (
            _hex_string(descriptor["sha256"], 64)
            and type(descriptor["size_bytes"]) is int
            and descriptor["size_bytes"] > 0
            and descriptor["media_type"] == media_type
            and native_oracle._same_json_value(
                descriptor["candidate_evidence"],
                _descriptor_candidate_evidence(expected_evidence),
            )
        )
    except AttributeError, KeyError, TypeError, ValueError, OverflowError:
        return False


def _tolerance_results_complete(results):
    try:
        if not isinstance(results, list) or not results:
            return False
        keys = []
        for result in results:
            _exact_keys(
                result,
                {"key", "dtype", "scope", "rtol", "atol", "max_abs_error"},
                "correctness tolerance result",
            )
            keys.append(result["key"])
            if not all(
                isinstance(result[name], str) and result[name]
                for name in ("key", "dtype", "scope")
            ):
                return False
            for name in ("rtol", "atol", "max_abs_error"):
                value = result[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value < 0
                ):
                    return False
        return keys == sorted(set(keys))
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def _artifact_binding_complete(artifact, group, name, manifest, expected_evidence):
    try:
        _exact_keys(
            artifact,
            {
                "case",
                "group",
                "reference",
                "reference_observer_commit",
                "candidate",
                "candidate_provenance",
                "comparison",
                "tolerance_results",
            },
            f"correctness artifact {name}",
        )
        reference = artifact["reference"]
        candidate = artifact["candidate"]
        provenance = artifact["candidate_provenance"]
        _exact_keys(
            provenance,
            {"commit", "source_sha256", "controller_sha256"},
            f"correctness candidate provenance {name}",
        )
        return (
            artifact["case"] == name
            and artifact["group"] == group
            and native_oracle._same_json_value(
                artifact["comparison"], {"passed": True, "failures": []}
            )
            and _portable_descriptor_complete(
                reference, expected_evidence, "application/x-npz"
            )
            and artifact["reference_observer_commit"]
            == manifest["reference"]["observer_commit"]
            and _portable_descriptor_complete(
                candidate, expected_evidence, "application/x-npz"
            )
            and reference["path"] != candidate["path"]
            and _tolerance_results_complete(artifact["tolerance_results"])
            and provenance["commit"] == expected_evidence["candidate_git_commit"]
            and _hex_string(provenance["commit"], 40)
            and _hex_string(provenance["source_sha256"], 64)
            and _hex_string(provenance["controller_sha256"], 64)
        )
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def _source_artifact_complete(source, expected_evidence):
    return _portable_descriptor_complete(source, expected_evidence, "application/json")


def _runtime_mode_complete(mode):
    try:
        _exact_keys(
            mode,
            {"device", "precision", "graph_mode", "compile_policy", "compile_mode"},
            "correctness runtime_mode",
        )
        expected = _runtime_contract(
            mode["device"],
            mode["precision"],
            mode["graph_mode"],
            mode["compile_mode"],
        )
        return native_oracle._same_json_value(mode, expected)
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def _runtime_receipt_descriptor_complete(descriptor):
    try:
        _exact_keys(
            descriptor,
            {"path", "sha256", "size_bytes", "media_type"},
            "runtime receipt descriptor",
        )
        _canonical_descriptor_path(descriptor["path"])
        return (
            _hex_string(descriptor["sha256"], 64)
            and type(descriptor["size_bytes"]) is int
            and 0 < descriptor["size_bytes"] <= MAX_CORRECTNESS_JSON_BYTES
            and descriptor["media_type"] == "application/json"
        )
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def correctness_binding_complete(
    index,
    manifest,
    expected_evidence,
    *,
    runtime_receipt,
    require_source_artifact=False,
):
    """Return whether an already-revalidated evidence index binds exactly."""
    try:
        _exact_keys(
            index,
            {
                "schema_version",
                "kind",
                "contract_id",
                "manifest_contract_sha256",
                "candidate_evidence",
                "runtime_mode",
                "runtime_receipt",
                "required_cases",
                "artifacts",
                "suite_acceptance",
            },
            "correctness index",
            optional={"source_artifact"},
        )
        required_pairs = _required_cases(manifest)
        required = [case["name"] for _group, case in required_pairs]
        suite = index["suite_acceptance"]
        _exact_keys(
            suite,
            {
                "correctness_case_count",
                "physical_check_case_count",
                "evaluated_case_count",
                "complete_fields",
                "persistent_state",
                "source_and_auxiliary_state",
                "physical_observables",
                "passed",
            },
            "correctness index suite_acceptance",
        )
        artifacts = index["artifacts"]
        candidate_binding_complete = (
            isinstance(expected_evidence, dict)
            and _hex_string(expected_evidence.get("candidate_git_commit"), 40)
            and expected_evidence.get("candidate_git_status") == ""
            and _hex_string(expected_evidence.get("manifest_sha256"), 64)
            and isinstance(expected_evidence.get("solver_abi"), str)
            and bool(expected_evidence["solver_abi"])
        )
        artifacts_complete = (
            isinstance(artifacts, list)
            and len(artifacts) == len(required_pairs)
            and all(
                _artifact_binding_complete(
                    artifact,
                    group,
                    case["name"],
                    manifest,
                    expected_evidence,
                )
                for artifact, (group, case) in zip(
                    artifacts, required_pairs, strict=True
                )
            )
        )
        artifact_paths_complete = artifacts_complete and len(
            {
                artifact[role]["path"]
                for artifact in artifacts
                for role in ("reference", "candidate")
            }
        ) == 2 * len(artifacts)
        source = index.get("source_artifact")
        source_complete = source is None or _source_artifact_complete(
            source, expected_evidence
        )
        receipt_bytes = (
            json.dumps(
                runtime_receipt,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        receipt_descriptor = index["runtime_receipt"]
        receipt_complete = (
            runtime_publication_receipt_complete(
                runtime_receipt,
                manifest,
                expected_evidence,
                index["runtime_mode"],
                _runtime_receipt_candidates(artifacts),
            )
            and receipt_descriptor["size_bytes"] == len(receipt_bytes)
            and receipt_descriptor["sha256"]
            == hashlib.sha256(receipt_bytes).hexdigest()
        )
        return (
            type(index["schema_version"]) is int
            and index["schema_version"] == 2
            and index["kind"] == INDEX_KIND
            and index["contract_id"] == INDEX_CONTRACT
            and index["manifest_contract_sha256"] == _canonical_sha256(manifest)
            and native_oracle._same_json_value(
                index["candidate_evidence"], expected_evidence
            )
            and candidate_binding_complete
            and _runtime_mode_complete(index["runtime_mode"])
            and _runtime_receipt_descriptor_complete(index["runtime_receipt"])
            and receipt_complete
            and index["required_cases"] == required
            and artifacts_complete
            and artifact_paths_complete
            and source_complete
            and (not require_source_artifact or source is not None)
            and native_oracle._same_json_value(
                suite,
                {
                    "correctness_case_count": len(manifest.get("correctness", ())),
                    "physical_check_case_count": len(
                        manifest.get("physical_checks", ())
                    ),
                    "evaluated_case_count": len(required),
                    "complete_fields": True,
                    "persistent_state": True,
                    "source_and_auxiliary_state": True,
                    "physical_observables": True,
                    "passed": True,
                },
            )
        )
    except AttributeError, KeyError, TypeError, ValueError:
        return False


def load_correctness_evidence_index(
    path,
    manifest,
    expected_evidence,
    *,
    descriptor_root,
    runtime_receipt,
):
    path = Path(path).resolve(strict=True)
    descriptor_root = _descriptor_root(descriptor_root)
    _relative_descriptor_path(path, descriptor_root)
    try:
        document = _load_bounded_json(path, "correctness evidence index")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("correctness evidence index is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("correctness evidence index must be an object")
    references = [
        _resolve_artifact_descriptor(descriptor_root, item["reference"])
        for item in document.get("artifacts", ())
    ]
    candidates = [
        _resolve_artifact_descriptor(descriptor_root, item["candidate"])
        for item in document.get("artifacts", ())
    ]
    receipt_path = Path(runtime_receipt).resolve(strict=True)
    indexed_receipt_path = _resolve_artifact_descriptor(
        descriptor_root, document["runtime_receipt"]
    )
    try:
        receipt_path.relative_to(descriptor_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "trusted runtime publication receipt must be outside descriptor root"
        )
    if os.path.samefile(receipt_path, indexed_receipt_path):
        raise ValueError(
            "trusted runtime publication receipt must be a distinct external file"
        )
    indexed_receipt = _runtime_receipt_descriptor(indexed_receipt_path, descriptor_root)
    supplied_receipt = _runtime_receipt_file_identity(receipt_path)
    if not native_oracle._same_json_value(
        indexed_receipt, document["runtime_receipt"]
    ) or not native_oracle._same_json_value(
        {key: indexed_receipt[key] for key in ("sha256", "size_bytes", "media_type")},
        supplied_receipt,
    ):
        raise ValueError("trusted runtime publication receipt bytes differ")
    receipt_document = load_runtime_publication_receipt(
        receipt_path,
        manifest,
        expected_evidence,
        document.get("runtime_mode"),
        _runtime_receipt_candidates(document.get("artifacts", ())),
    )
    if not correctness_binding_complete(
        document,
        manifest,
        expected_evidence,
        runtime_receipt=receipt_document,
    ):
        raise ValueError("correctness evidence index differs from recomputed evidence")
    rebuilt = build_correctness_evidence_index(
        references,
        candidates,
        manifest,
        expected_evidence,
        descriptor_root=descriptor_root,
        runtime_receipt=indexed_receipt_path,
    )
    if not native_oracle._same_json_value(document, rebuilt):
        raise ValueError("correctness evidence index differs from recomputed evidence")
    result = deepcopy(rebuilt)
    result["source_artifact"] = _artifact_descriptor(
        path, descriptor_root, expected_evidence, "application/json"
    )
    return result


def _load_candidate_evidence(path):
    try:
        value = _load_bounded_json(path, "candidate evidence")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("candidate evidence is not valid JSON") from error
    if isinstance(value, dict) and "evidence" in value:
        value = value["evidence"]
    if not isinstance(value, dict):
        raise ValueError("candidate evidence JSON must contain an object")
    if not native_oracle._finite_json_value(value):
        raise ValueError("candidate evidence contains a non-finite value")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--reference", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--threads", type=int, default=1)
    capture.add_argument("--device", default="cpu")
    capture.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    capture.add_argument("--graph-mode", choices=("eager", "graph"), default="eager")
    capture.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    index = commands.add_parser("index")
    index.add_argument("--references", type=Path, nargs="+", required=True)
    index.add_argument("--candidates", type=Path, nargs="+", required=True)
    index.add_argument("--candidate-evidence", type=Path, required=True)
    index.add_argument("--descriptor-root", type=Path, required=True)
    index.add_argument("--runtime-receipt", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-index")
    validate.add_argument("--index", type=Path, required=True)
    validate.add_argument("--candidate-evidence", type=Path, required=True)
    validate.add_argument("--descriptor-root", type=Path, required=True)
    validate.add_argument("--runtime-receipt", type=Path, required=True)
    args = parser.parse_args()
    manifest, manifest_sha256 = _load_trusted_manifest(args.manifest)
    if args.command == "capture":
        if args.threads < 1:
            parser.error("threads must be positive")
        value = capture_torch_candidate(
            args.reference,
            manifest,
            args.output,
            threads=args.threads,
            device=args.device,
            precision=args.precision,
            graph_mode=args.graph_mode,
            compile_mode=args.compile_mode,
        )
    elif args.command == "index":
        evidence = _load_candidate_evidence(args.candidate_evidence)
        if evidence.get("manifest_sha256") != manifest_sha256:
            raise ValueError("candidate evidence manifest bytes differ")
        value = build_correctness_evidence_index(
            args.references,
            args.candidates,
            manifest,
            evidence,
            descriptor_root=args.descriptor_root,
            runtime_receipt=args.runtime_receipt,
        )
        rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        evidence = _load_candidate_evidence(args.candidate_evidence)
        if evidence.get("manifest_sha256") != manifest_sha256:
            raise ValueError("candidate evidence manifest bytes differ")
        value = load_correctness_evidence_index(
            args.index,
            manifest,
            evidence,
            descriptor_root=args.descriptor_root,
            runtime_receipt=args.runtime_receipt,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
