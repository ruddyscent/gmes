#!/usr/bin/env python3
"""Record one local installed-artifact Torch cutover smoke for Issue #124.

This helper deliberately has no dependency on the checkout that contains it.
Run it by absolute path with an installed environment's ``python -I`` from a
directory outside every forbidden checkout root.  Its JSON record is local CI
evidence only; it neither publishes nor authorizes a candidate.
"""

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

import numpy as np

EVIDENCE_FILENAME = "package-cutover.json"
EVIDENCE_SCHEMA = "gmes.issue124.installed-package-cutover.v1"
PENDING_EXIT_CODE = 3


class CutoverError(RuntimeError):
    """Raised when installed-artifact cutover acceptance cannot pass."""


class WorkerCutoverError(CutoverError):
    """Carry a failed torchrun result into the local evidence record."""

    def __init__(self, message: str, worker: Mapping[str, object]) -> None:
        super().__init__(message)
        self.worker = dict(worker)


class CutoverPending(CutoverError):
    """Raised for a deliberately unimplemented installed two-GPU provider."""


@dataclass(frozen=True)
class CutoverRequest:
    """Required inputs for one installed-package cutover execution."""

    candidate_label: str
    archive: Path
    forbidden_roots: tuple[Path, ...]
    device: str
    required_device_count: int
    evidence_dir: Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular, non-symlink archive."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CutoverError(f"archive must be a regular file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_url_label(value: object) -> str | None:
    """Return a credential-free URL label suitable for a local evidence record."""

    if not isinstance(value, str) or not value:
        return None
    parsed = urlsplit(value)
    if not parsed.scheme:
        return None
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, host + port, parsed.path, "", ""))


def installed_provenance(
    archive: Path, *, distribution: importlib.metadata.Distribution | None = None
) -> dict[str, object]:
    """Verify PEP 610 archive digest provenance for installed ``gmes``."""

    supplied_archive = Path(archive)
    archive_digest = sha256_file(supplied_archive)
    archive = supplied_archive.resolve(strict=True)
    distribution = (
        importlib.metadata.distribution("gmes")
        if distribution is None
        else distribution
    )
    direct_url_raw = distribution.read_text("direct_url.json")
    if direct_url_raw is None:
        raise CutoverError("installed gmes distribution has no PEP 610 direct_url.json")
    try:
        direct_url = json.loads(direct_url_raw)
    except json.JSONDecodeError as error:
        raise CutoverError("installed gmes direct_url.json is invalid JSON") from error
    if not isinstance(direct_url, dict):
        raise CutoverError("installed gmes direct_url.json must be an object")
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, dict):
        raise CutoverError("installed gmes provenance has no archive_info")
    expected_hash = archive_info.get("hash")
    if not isinstance(expected_hash, str):
        raise CutoverError("installed gmes provenance has no archive SHA-256 hash")
    if expected_hash != f"sha256={archive_digest}":
        raise CutoverError(
            "installed gmes PEP 610 archive hash does not match the supplied archive"
        )
    distribution_root = Path(distribution.locate_file("gmes")).resolve(strict=True)
    if not distribution_root.is_dir():
        raise CutoverError("installed gmes distribution root is not a directory")
    name = distribution.metadata.get("Name")
    if not isinstance(name, str) or name.casefold() != "gmes":
        raise CutoverError("installed distribution metadata is not gmes")
    return {
        "name": name,
        "version": distribution.version,
        "distribution_root": str(distribution_root),
        "archive": {
            "path": str(archive),
            "sha256": archive_digest,
            "size_bytes": archive.stat().st_size,
        },
        "direct_url": {
            "archive_hash": expected_hash,
            "url": _safe_url_label(direct_url.get("url")),
        },
    }


def _is_under(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by a resolved root."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def verify_module_origins(
    distribution_root: Path,
    forbidden_roots: tuple[Path, ...],
    *,
    modules: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Reject checkout or native origins for every currently loaded gmes module."""

    root = Path(distribution_root).resolve(strict=True)
    forbidden = tuple(Path(path).resolve(strict=True) for path in forbidden_roots)
    if not forbidden:
        raise CutoverError("at least one forbidden checkout root is required")
    if any(_is_under(root, item) for item in forbidden):
        raise CutoverError(
            "installed gmes distribution resolves inside a forbidden root"
        )
    modules = sys.modules if modules is None else modules
    extensions = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    origins: dict[str, str] = {}
    for name, module in sorted(modules.items()):
        if name != "gmes" and not name.startswith("gmes."):
            continue
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            raise CutoverError(f"loaded gmes module {name!r} has no file origin")
        path = Path(origin).resolve(strict=True)
        if not _is_under(path, root):
            raise CutoverError(
                f"loaded gmes module {name!r} is outside the installed distribution"
            )
        if any(_is_under(path, item) for item in forbidden):
            raise CutoverError(
                f"loaded gmes module {name!r} resolves in a forbidden root"
            )
        if name.startswith("gmes._") or path.name.endswith(extensions):
            raise CutoverError(f"loaded native GMES module is forbidden: {name!r}")
        origins[name] = str(path)
    if "gmes" not in origins:
        raise CutoverError("gmes was not imported before origin validation")
    return origins


def _require_device(device: str, required_device_count: int):
    """Validate a requested CPU or single-CUDA-device smoke target."""

    if isinstance(required_device_count, bool) or required_device_count < 0:
        raise CutoverError("required device count must be a non-negative integer")
    if device == "cpu":
        if required_device_count != 0:
            raise CutoverError("CPU cutover requires --required-device-count 0")
        return None
    if not device.startswith("cuda"):
        raise CutoverError("device must be cpu or cuda[:INDEX]")
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise CutoverError("CUDA was requested but PyTorch reports it unavailable")
    if required_device_count > 2:
        raise CutoverError("multi-GPU cutover supports exactly two CUDA devices")
    count = int(torch.cuda.device_count())
    if count < required_device_count:
        raise CutoverError(
            f"CUDA cutover requires {required_device_count} visible devices, found {count}"
        )
    suffix = device.removeprefix("cuda")
    if suffix in ("", ":0"):
        index = 0
    elif suffix.startswith(":") and suffix[1:].isdigit():
        index = int(suffix[1:])
    else:
        raise CutoverError("CUDA device must use cuda or cuda:INDEX")
    if index >= count:
        raise CutoverError(f"requested CUDA device cuda:{index} is unavailable")
    if required_device_count == 2:
        distributed = getattr(torch, "distributed", None)
        if distributed is None or not distributed.is_nccl_available():
            raise CutoverError("multi-GPU cutover requires an available NCCL backend")
        return torch
    if required_device_count != 1:
        raise CutoverError("CUDA cutover requires --required-device-count 1")
    return torch


def _mixed_geometry(gmes: Any) -> tuple[object, ...]:
    """Return a small exact built-in material/PML geometry for CUDA smokes."""

    return (
        gmes.DefaultMedium(gmes.Dielectric(eps_inf=1.7, mu_inf=1.05)),
        gmes.Shell(gmes.Cpml(), thickness=0.5),
        gmes.Block(
            gmes.Drude(
                eps_inf=1.2,
                sigma=0.01,
                dps=(gmes.DrudePole(omega=0.7, gamma=0.03),),
            ),
            center=(-1, 0, 0),
            size=(1.5, 2, 2),
        ),
    )


def _maximum_field_error(
    left: Mapping[str, object], right: Mapping[str, object]
) -> float:
    """Validate every field and return the largest finite component-wise error."""

    names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    if set(left) != set(names) or set(right) != set(names):
        raise CutoverError(
            "single-vs-two GPU fields must contain exactly six components"
        )
    errors = []
    for name in names:
        first = np.asarray(left[name])
        second = np.asarray(right[name])
        if first.shape != second.shape:
            raise CutoverError(f"single-vs-two GPU field {name} has mismatched shapes")
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise CutoverError(f"single-vs-two GPU field {name} is non-finite")
        difference = np.abs(first - second)
        if not np.isfinite(difference).all():
            raise CutoverError(
                f"single-vs-two GPU field {name} difference is non-finite"
            )
        errors.append(float(np.max(difference)))
    return max(errors)


def _two_gpu_worker(request: CutoverRequest) -> None:
    """Run one installed-origin rank of the two-GPU source/replay smoke."""

    torch = _require_device(request.device, request.required_device_count)
    if request.required_device_count != 2:
        raise CutoverError("two-GPU worker requires exactly two CUDA devices")
    gmes = importlib.import_module("gmes")
    provenance = installed_provenance(request.archive)
    distribution_root = Path(str(provenance["distribution_root"]))
    origins_before = verify_module_origins(distribution_root, request.forbidden_roots)
    launch = gmes.distributed_launch_from_environment()
    if (
        launch.world_size != 2
        or launch.local_world_size != 2
        or launch.rank not in (0, 1)
        or launch.local_rank not in (0, 1)
        or launch.rank != launch.local_rank
    ):
        raise CutoverError("two-GPU worker requires one local rank per CUDA device")
    simulation = None
    serial = None
    try:
        space = gmes.Cartesian((4, 4, 4), 1)
        geometry = _mixed_geometry(gmes)
        source = gmes.PointSource(
            gmes.DifferentiatedGaussian(1.0, 0.0), (-1, 0, 0), gmes.Ex
        )
        probes = (gmes.TorchProbeSpec("Ex", (2, 2, 2), capacity=4),)
        if launch.rank == 0:
            serial = gmes.TorchSimulation(
                space=space,
                geometry=geometry,
                sources=(source,),
                probes=probes,
                runtime=gmes.TorchRuntimeConfig(
                    device="cuda:0", precision="float64", cpu_threads=1
                ),
                courant_ratio=0.5,
            )
        simulation = gmes.TorchDistributedSimulation(
            space=space,
            geometry=geometry,
            sources=(source,),
            probes=probes,
            runtime=gmes.TorchRuntimeConfig(
                device=f"cuda:{launch.local_rank}",
                precision="float64",
                cpu_threads=1,
                launch=launch,
            ),
            courant_ratio=0.5,
            split_axis=0,
            cut=2,
            require_peer_access=False,
        )
        initial = simulation.global_field_snapshot()
        initial_digest = _field_digest(initial) if launch.rank == 0 else None
        if launch.rank == 0:
            torch.cuda.synchronize(0)
            serial_started = time.perf_counter()
            serial.advance(2)
            torch.cuda.synchronize(0)
            serial_seconds = time.perf_counter() - serial_started
            serial_checkpoint = serial.checkpoint()
            serial_checkpoint_digest = _field_digest(serial.host_snapshot())
            serial.advance(1)
            serial.load_checkpoint(serial_checkpoint)
            if _field_digest(serial.host_snapshot()) != serial_checkpoint_digest:
                raise CutoverError(
                    "single-GPU checkpoint replay did not restore fields"
                )
            serial_snapshot = serial.host_snapshot()
            serial_probe_samples = sum(
                len(sample.times) for sample in serial.flush_probes()
            )
        else:
            serial_seconds = None
            serial_snapshot = None
            serial_probe_samples = None
        torch.distributed.barrier()
        torch.cuda.synchronize(simulation.device)
        distributed_started = time.perf_counter()
        simulation.advance(2)
        torch.cuda.synchronize(simulation.device)
        distributed_seconds = time.perf_counter() - distributed_started
        checkpoint = simulation.checkpoint()
        checkpoint_local_digest = _field_digest(simulation.host_snapshot())
        checkpoint_global = simulation.global_field_snapshot()
        checkpoint_global_digest = (
            _field_digest(checkpoint_global) if launch.rank == 0 else None
        )
        simulation.advance(1)
        simulation.load_checkpoint(checkpoint)
        restored_local_digest = _field_digest(simulation.host_snapshot())
        restored_global = simulation.global_field_snapshot()
        restored_global_digest = (
            _field_digest(restored_global) if launch.rank == 0 else None
        )
        distributed_probes = simulation.flush_probes()
        local_samples = sum(
            len(sample.times) for sample in distributed_probes["samples"]
        )
        field_error = (
            _maximum_field_error(serial_snapshot, checkpoint_global)
            if launch.rank == 0
            else None
        )
        rank_record = {
            "rank": launch.rank,
            "local_rank": launch.local_rank,
            "isolated": sys.flags.isolated == 1,
            "device": str(simulation.device),
            "current_device": int(torch.cuda.current_device()),
            "source_batches": len(simulation.sources.batches),
            "probe_samples": local_samples,
            "checkpoint_replay": checkpoint_local_digest == restored_local_digest,
            "distributed_seconds": distributed_seconds,
            "provenance": provenance,
            "module_origins_before": origins_before,
            "module_origins_after": verify_module_origins(
                distribution_root, request.forbidden_roots
            ),
        }
        gathered: list[object] = [None, None]
        torch.distributed.all_gather_object(gathered, rank_record)
        if launch.rank == 0:
            timing_seconds = [record["distributed_seconds"] for record in gathered]
            result = {
                "initial_field_sha256": initial_digest,
                "checkpoint_field_sha256": checkpoint_global_digest,
                "restored_field_sha256": restored_global_digest,
                "global_checkpoint_replay": (
                    checkpoint_global_digest == restored_global_digest
                ),
                "single_gpu": {
                    "seconds": serial_seconds,
                    "probe_samples": serial_probe_samples,
                    "checkpoint_replay": True,
                },
                "single_vs_two_maximum_error": field_error,
                "two_gpu_seconds": max(timing_seconds),
                "informational_speedup": (
                    serial_seconds / max(timing_seconds)
                    if max(timing_seconds) > 0
                    else None
                ),
                "ranks": gathered,
            }
            print(json.dumps(result, sort_keys=True))
    finally:
        if simulation is not None:
            gmes.TorchDistributedSimulation.close()


def _two_gpu_result(stdout: str) -> dict[str, object]:
    """Return the rank-zero JSON record from a torchrun worker's stdout."""

    for line in reversed(stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    raise CutoverError("two-GPU worker did not emit a JSON result")


def _validate_two_gpu_result(
    result: Mapping[str, object], provenance: Mapping[str, object]
) -> dict[str, object]:
    """Validate real rank/device ownership and replay evidence from two workers."""

    ranks = result.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise CutoverError("two-GPU worker did not report exactly two ranks")
    reports = []
    for record in ranks:
        if not isinstance(record, dict):
            raise CutoverError("two-GPU worker emitted a malformed rank report")
        reports.append(record)
    reports.sort(key=lambda record: record.get("rank", -1))
    for rank, record in enumerate(reports):
        if (
            record.get("rank") != rank
            or record.get("local_rank") != rank
            or record.get("isolated") is not True
            or record.get("device") != f"cuda:{rank}"
            or record.get("current_device") != rank
            or record.get("checkpoint_replay") is not True
            or record.get("provenance") != provenance
        ):
            raise CutoverError("two-GPU rank/device ownership evidence is invalid")
        origins = record.get("module_origins_after")
        if not isinstance(origins, dict) or not origins:
            raise CutoverError("two-GPU worker did not verify installed module origins")
        seconds = record.get("distributed_seconds")
        if not isinstance(seconds, float) or not math.isfinite(seconds) or seconds <= 0:
            raise CutoverError("two-GPU timing evidence is invalid")
    if sum(record.get("source_batches", -1) for record in reports) != 1:
        raise CutoverError("two-GPU source was not owned by exactly one rank")
    if sum(record.get("probe_samples", -1) for record in reports) != 2:
        raise CutoverError("two-GPU probe did not retain both smoke samples")
    single_gpu = result.get("single_gpu")
    if (
        not isinstance(single_gpu, dict)
        or single_gpu.get("checkpoint_replay") is not True
        or single_gpu.get("probe_samples") != 2
        or not isinstance(single_gpu.get("seconds"), float)
        or single_gpu["seconds"] <= 0
    ):
        raise CutoverError("single-GPU source/checkpoint evidence is invalid")
    maximum_error = result.get("single_vs_two_maximum_error")
    if not isinstance(maximum_error, float) or not math.isfinite(maximum_error):
        raise CutoverError("single-vs-two GPU field comparison is invalid")
    if maximum_error > 2e-10:
        raise CutoverError(
            "single-vs-two GPU mixed-material field comparison exceeded 2e-10"
        )
    two_gpu_seconds = result.get("two_gpu_seconds")
    informational_speedup = result.get("informational_speedup")
    if (
        not isinstance(two_gpu_seconds, float)
        or not math.isfinite(two_gpu_seconds)
        or two_gpu_seconds <= 0
        or not isinstance(informational_speedup, float)
        or not math.isfinite(informational_speedup)
        or informational_speedup <= 0
    ):
        raise CutoverError("single-vs-two GPU timing evidence is invalid")
    if (
        not isinstance(result.get("initial_field_sha256"), str)
        or result["initial_field_sha256"] == result.get("checkpoint_field_sha256")
        or result.get("global_checkpoint_replay") is not True
        or result.get("checkpoint_field_sha256") != result.get("restored_field_sha256")
    ):
        raise CutoverError("two-GPU source/checkpoint evidence is invalid")
    return {
        "device_count": 2,
        "ranks": reports,
        "initial_field_sha256": result["initial_field_sha256"],
        "checkpoint_field_sha256": result["checkpoint_field_sha256"],
        "checkpoint_replay": True,
        "single_gpu": single_gpu,
        "single_vs_two_maximum_error": maximum_error,
        "two_gpu_seconds": two_gpu_seconds,
        "informational_speedup": informational_speedup,
    }


def run_two_gpu_smoke(
    request: CutoverRequest, provenance: Mapping[str, object]
) -> dict[str, object]:
    """Launch two installed-origin CUDA ranks and validate their real evidence."""

    worker_command = [
        sys.executable,
        "-I",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "--no-python",
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--two-gpu-worker",
        "--candidate-label",
        request.candidate_label,
        "--archive",
        str(request.archive),
        "--device",
        request.device,
        "--required-device-count",
        "2",
        "--evidence-dir",
        str(request.evidence_dir),
    ]
    for root in request.forbidden_roots:
        worker_command.extend(("--forbidden-root", str(root)))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        worker_command,
        check=False,
        capture_output=True,
        cwd=Path.cwd(),
        env=environment,
        text=True,
    )
    worker = {
        "argv": worker_command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode:
        raise WorkerCutoverError(
            "two-GPU worker failed with exit "
            f"{completed.returncode}: {completed.stderr.strip()}",
            worker,
        )
    result = _validate_two_gpu_result(_two_gpu_result(completed.stdout), provenance)
    result["worker"] = worker
    return result


def _field_digest(fields: Mapping[str, object]) -> str:
    """Return a stable digest of host snapshot buffers without NumPy helpers."""

    hasher = hashlib.sha256()
    for name in sorted(fields):
        value = fields[name]
        hasher.update(name.encode("utf-8"))
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def run_torch_smoke(device: str) -> dict[str, object]:
    """Run a real source, step, probe, and checkpoint-replay Torch smoke."""

    gmes = importlib.import_module("gmes")
    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian((2, 2, 2), 1),
        geometry=(gmes.DefaultMedium(gmes.Dielectric()),),
        sources=(
            gmes.PointSource(gmes.DifferentiatedGaussian(1.0, 0.0), (0, 0, 0), gmes.Ex),
        ),
        probes=(gmes.TorchProbeSpec("Ex", (1, 1, 1), capacity=4),),
        runtime=gmes.TorchRuntimeConfig(
            device=device, precision="float64", cpu_threads=1
        ),
        courant_ratio=0.5,
    )
    initial = simulation.host_snapshot()
    initial_digest = _field_digest(initial)
    simulation.advance(2)
    checkpoint = simulation.checkpoint()
    checkpoint_digest = _field_digest(simulation.host_snapshot())
    simulation.advance(1)
    simulation.load_checkpoint(checkpoint)
    restored = simulation.host_snapshot()
    if _field_digest(restored) != checkpoint_digest:
        raise CutoverError("checkpoint replay did not restore the Torch field state")
    probe = simulation.flush_probes()
    if len(probe) != 1 or len(probe[0].times) != 2:
        raise CutoverError("Torch probe did not retain both smoke samples")
    if not all(math.isfinite(float(value)) for value in probe[0].values.reshape(-1)):
        raise CutoverError("Torch probe contains a non-finite smoke value")
    if initial_digest == checkpoint_digest:
        raise CutoverError("Torch source smoke did not change any field")
    return {
        "device": str(simulation.device),
        "steps": int(simulation.state.step_count.detach().cpu()),
        "probe_samples": len(probe[0].times),
        "initial_field_sha256": initial_digest,
        "checkpoint_field_sha256": checkpoint_digest,
        "checkpoint_replay": True,
    }


def run_cutover(request: CutoverRequest) -> dict[str, object]:
    """Perform the non-I/O portion of one installed package cutover check."""

    if (
        not isinstance(request.candidate_label, str)
        or not request.candidate_label.strip()
    ):
        raise CutoverError("candidate label must be a non-empty explicit label")
    archive = request.archive
    if archive.suffix != ".whl" and not archive.name.endswith(".tar.gz"):
        raise CutoverError("archive must be a wheel or sdist (.whl or .tar.gz)")
    provenance = installed_provenance(archive)
    _require_device(request.device, request.required_device_count)
    importlib.import_module("gmes")
    distribution_root = Path(str(provenance["distribution_root"]))
    origins_before = verify_module_origins(distribution_root, request.forbidden_roots)
    smoke = (
        run_two_gpu_smoke(request, provenance)
        if request.required_device_count == 2
        else run_torch_smoke(request.device)
    )
    origins_after = verify_module_origins(distribution_root, request.forbidden_roots)
    return {
        "schema": EVIDENCE_SCHEMA,
        "scope": "local installed-artifact smoke; not publication evidence",
        "candidate_label": request.candidate_label,
        "provenance": provenance,
        "module_origins_before": origins_before,
        "module_origins_after": origins_after,
        "torch_smoke": smoke,
        "passed": True,
    }


def _write_record(evidence_dir: Path, record: Mapping[str, object]) -> Path:
    """Write one non-overwritable local evidence record."""

    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / EVIDENCE_FILENAME
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise CutoverError(f"evidence output already exists: {path}") from error
    return path


def _request_from_args(args: argparse.Namespace) -> CutoverRequest:
    """Validate CLI arguments and construct the stable request object."""

    if not args.forbidden_root:
        raise CutoverError("at least one --forbidden-root is required")
    roots = tuple(Path(value).resolve(strict=True) for value in args.forbidden_root)
    if any(not root.is_dir() for root in roots):
        raise CutoverError("every forbidden root must be a directory")
    return CutoverRequest(
        candidate_label=args.candidate_label,
        archive=Path(args.archive),
        forbidden_roots=roots,
        device=args.device,
        required_device_count=args.required_device_count,
        evidence_dir=Path(args.evidence_dir),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the stable installed-package cutover command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--forbidden-root", action="append", default=[])
    parser.add_argument("--device", required=True)
    parser.add_argument("--required-device-count", required=True, type=int)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--two-gpu-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and write the real command outcome as local evidence."""

    parser = build_parser()
    args = parser.parse_args(argv)
    request = _request_from_args(args)
    if args.two_gpu_worker:
        _two_gpu_worker(request)
        return 0
    actual_argv = list(sys.argv if argv is None else [sys.argv[0], *argv])
    stdout = ""
    stderr = ""
    exit_code = 0
    status = "passed"
    try:
        result = run_cutover(request)
        stdout = json.dumps(result, sort_keys=True) + "\n"
    except CutoverPending as error:
        status = "pending"
        exit_code = PENDING_EXIT_CODE
        stderr = f"{error}\n"
        result = {"schema": EVIDENCE_SCHEMA, "passed": False, "pending": True}
    except CutoverError as error:
        status = "failed"
        exit_code = 1
        stderr = f"{error}\n"
        result = {"schema": EVIDENCE_SCHEMA, "passed": False, "pending": False}
        worker = getattr(error, "worker", None)
        if worker is not None:
            result["worker"] = worker
    record = {
        "schema": EVIDENCE_SCHEMA,
        "scope": "local installed-artifact smoke; not publication evidence",
        "status": status,
        "command": {"argv": actual_argv},
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "result": result,
    }
    _write_record(request.evidence_dir, record)
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
