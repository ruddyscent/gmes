#!/usr/bin/env python3
"""Validate immutable historical native archives and benchmark records.

Candidate checkouts run the retained Torch providers.  Native capture and
timing execution are deliberately restricted to the manifest-pinned historical
observer checkout.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import sysconfig
from pathlib import Path
from statistics import median, pstdev
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "native_oracle_workloads.json"
COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
FIELD_INITIALIZER = "native-affine-ramp-v1"
ARCHIVE_SCHEMA_VERSION = 2
MIXED_SOURCE_FREE_MAX_ENERGY_RATIO = 100.0
PERFORMANCE_ONLY_REFERENCE_FIELDS = frozenset(
    {"performance_observer_tag", "performance_observer_commit"}
)
TORCH_CPU_BASELINE_RELEASE_ASSET_URLS = {
    "one": (
        "https://github.com/ruddyscent/gmes/releases/download/"
        "issue-123-torch-cpu-baseline-v3/torch-cpu-baseline-one.json"
    ),
    "physical": (
        "https://github.com/ruddyscent/gmes/releases/download/"
        "issue-123-torch-cpu-baseline-v3/torch-cpu-baseline-physical.json"
    ),
}


def _correctness_reference_contract(reference):
    """Project the manifest reference to fields serialized in correctness archives."""
    return {
        key: value
        for key, value in reference.items()
        if key not in PERFORMANCE_ONLY_REFERENCE_FIELDS
    }


def load_manifest(path=DEFAULT_MANIFEST):
    """Load and validate the portable workload and frozen gate description."""
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 2:
        raise ValueError("unsupported native oracle workload schema")
    reference = data["reference"]
    observer_commit = reference.get("observer_commit", "")
    if len(observer_commit) != 40 or any(
        character not in "0123456789abcdef" for character in observer_commit
    ):
        raise ValueError("observer_commit must be a full lowercase Git commit")
    performance_observer_commit = reference.get("performance_observer_commit", "")
    if len(performance_observer_commit) != 40 or any(
        character not in "0123456789abcdef" for character in performance_observer_commit
    ):
        raise ValueError(
            "performance_observer_commit must be a full lowercase Git commit"
        )
    for field in ("observer_tag", "performance_observer_tag"):
        if not isinstance(reference.get(field), str) or not reference[field]:
            raise ValueError(f"{field} must be a non-empty string")
    performance_summary_sha256 = reference.get("performance_summary_sha256", "")
    if len(performance_summary_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in performance_summary_sha256
    ):
        raise ValueError(
            "performance_summary_sha256 must be a lowercase SHA-256 digest"
        )
    if reference.get("field_initializer") != FIELD_INITIALIZER:
        raise ValueError("unsupported native oracle field initializer")
    for name in (
        "performance_warmup_steps",
        "performance_steps_per_repeat",
        "performance_repetitions",
        "performance_profile_steps",
    ):
        if not isinstance(reference.get(name), int) or reference[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    acceptance = data.get("performance_gates", {}).get("cpu_acceptance", {})
    if acceptance.get("contract_id") != "cpu-acceptance-v2":
        raise ValueError("unsupported cpu_acceptance contract")
    timing_reference = acceptance.get("timing_reference", {})
    if timing_reference.get("backend") != "torch":
        raise ValueError("cpu_acceptance timing reference must use Torch")
    root_commit = timing_reference.get("root_commit", "")
    if len(root_commit) != 40 or any(
        character not in "0123456789abcdef" for character in root_commit
    ):
        raise ValueError(
            "cpu_acceptance timing reference root_commit must be a full "
            "lowercase Git commit"
        )
    slice_artifacts = timing_reference.get("slice_artifacts")
    if not isinstance(slice_artifacts, list) or len(slice_artifacts) != 2:
        raise ValueError("cpu_acceptance requires two pinned Torch slice artifacts")
    expected_modes = (("one", 1), ("physical", 4))
    for artifact, (expected_mode, expected_threads) in zip(
        slice_artifacts, expected_modes, strict=True
    ):
        if not isinstance(artifact, dict) or set(artifact) != {
            "thread_mode",
            "threads",
            "publication_url",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("cpu_acceptance Torch slice artifact schema is invalid")
        publication_url = artifact["publication_url"]
        parsed_url = (
            urlsplit(publication_url) if isinstance(publication_url, str) else None
        )
        if (
            artifact["thread_mode"] != expected_mode
            or type(artifact["threads"]) is not int
            or artifact["threads"] != expected_threads
            or parsed_url is None
            or parsed_url.scheme != "https"
            or parsed_url.netloc != "github.com"
            or parsed_url.query
            or parsed_url.fragment
            or publication_url != TORCH_CPU_BASELINE_RELEASE_ASSET_URLS[expected_mode]
            or type(artifact["size_bytes"]) is not int
            or artifact["size_bytes"] < 1
        ):
            raise ValueError("cpu_acceptance Torch slice artifact modes are invalid")
        digest = artifact["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "cpu_acceptance Torch slice artifact must have a lowercase "
                "SHA-256 digest"
            )
    legacy_evidence = timing_reference.get("legacy_evidence", {})
    if legacy_evidence.get("evidence_contract_id") != "torch-cpu-acceptance-v7":
        raise ValueError("unsupported legacy CPU evidence contract")
    if legacy_evidence.get("cpu_contract_id") != "cpu-acceptance-v1":
        raise ValueError("unsupported legacy CPU acceptance contract")
    for name in ("manifest_sha256", "runner_sha256", "solver_sha256"):
        digest = legacy_evidence.get(name, "")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"cpu_acceptance legacy {name} must be a lowercase SHA-256 digest"
            )
    if legacy_evidence.get("solver_abi") != "torch-fdtd-regions-v8":
        raise ValueError("unsupported legacy CPU solver ABI")
    expected_timing_reference = {
        "backend": "torch",
        "root_commit": "821c075b9328e02c3f3e5d16488a44b64ff08c04",
        "timing_runtime_identity": {
            "schema_version": 1,
            "torch": "2.13.0+cpu",
            "cuda_runtime": None,
        },
        "slice_artifacts": [
            {
                "thread_mode": "one",
                "threads": 1,
                "publication_url": TORCH_CPU_BASELINE_RELEASE_ASSET_URLS["one"],
                "size_bytes": 18281,
                "sha256": (
                    "c8eba3c17ccae5ba744a8fbc90b89d72a77dcf0624339cda1deb4d7f594395ed"
                ),
            },
            {
                "thread_mode": "physical",
                "threads": 4,
                "publication_url": TORCH_CPU_BASELINE_RELEASE_ASSET_URLS["physical"],
                "size_bytes": 18292,
                "sha256": (
                    "b1a3c82a069c2475560468a7b8d0a237db89e857bdce96f8fc812449b5c35602"
                ),
            },
        ],
        "legacy_evidence": {
            "evidence_contract_id": "torch-cpu-acceptance-v7",
            "cpu_contract_id": "cpu-acceptance-v1",
            "manifest_sha256": (
                "6d7fe084c558cf69771f0c3928bc9be96fc6bb5b55ba777d674151fbbe6cbe19"
            ),
            "runner_sha256": (
                "fee6d418bb50729ddb26ff14e931a4f51bb8d2a92cb0ad537c2757846247a770"
            ),
            "solver_sha256": (
                "9cd8decc801a6f9d93551c6e6f427afeff1c65e3092e54b03e5abe0a3e9192d5"
            ),
            "solver_abi": "torch-fdtd-regions-v8",
        },
    }
    if timing_reference != expected_timing_reference:
        raise ValueError("cpu_acceptance timing reference is not the frozen baseline")
    known_cases = {
        case["name"]
        for group in ("correctness", "benchmarks")
        for case in data.get(group, ())
    }
    cases = acceptance.get("cases")
    expected_cases = [
        "cpu-crossover-2d",
        "cpu-crossover-3d",
        "cpu-large-2d",
        "cpu-large-3d",
        "bloch-2d",
        "bloch-3d",
    ]
    if cases != expected_cases or any(case not in known_cases for case in cases):
        raise ValueError("cpu_acceptance cases must match the frozen workload matrix")
    if acceptance.get("thread_modes") != ["one", "physical"]:
        raise ValueError("cpu_acceptance thread_modes must be one and physical")
    if acceptance.get("precision") != "float64":
        raise ValueError("cpu_acceptance precision must be float64")
    ratio = acceptance.get("max_individual_ratio")
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio != 1.05:
        raise ValueError("cpu_acceptance max_individual_ratio must be exactly 1.05")
    if acceptance.get("native_comparison") != "informational":
        raise ValueError("cpu_acceptance native comparison must be informational")
    allocation = acceptance.get("allocation_contract", {})
    if allocation.get("method") != "reviewed-fixed-temporary-provenance-v1":
        raise ValueError("unsupported cpu_acceptance allocation contract")
    fixed_temporaries = allocation.get("fixed_temporaries", {})
    if fixed_temporaries.get("allowed") is not True:
        raise ValueError("cpu_acceptance must allow reviewed fixed temporaries")
    if fixed_temporaries.get("reviewed_provenance_required") is not True:
        raise ValueError(
            "cpu_acceptance fixed temporaries must require reviewed provenance"
        )
    for name in (
        "max_net_live_growth_bytes",
        "max_final_live_growth_bytes",
        "max_full_field_or_domain_clones",
    ):
        if type(allocation.get(name)) is not int or allocation[name] != 0:
            raise ValueError(f"cpu_acceptance {name} must be zero")
    if allocation.get("rss_growth") != "bounded":
        raise ValueError("cpu_acceptance RSS growth must be bounded")
    if allocation.get("public_upstream_issue_required") is not True:
        raise ValueError(
            "cpu_acceptance fixed temporaries must require a public upstream issue"
        )
    statistics = acceptance.get("statistics", {})
    if statistics.get("method") != "independent-stratified-bootstrap-log-geomean-v1":
        raise ValueError("unsupported cpu_acceptance statistics method")
    if type(statistics.get("resamples")) is not int or statistics["resamples"] < 1:
        raise ValueError("cpu_acceptance resamples must be a positive integer")
    if type(statistics.get("seed")) is not int:
        raise ValueError("cpu_acceptance seed must be an integer")
    confidence = statistics.get("one_sided_confidence")
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("cpu_acceptance confidence must be between zero and one")
    regression_ratio = statistics.get("regression_ratio")
    if type(regression_ratio) not in (int, float) or regression_ratio <= 0:
        raise ValueError("cpu_acceptance regression_ratio must be positive")
    relative_mad = statistics.get("max_relative_mad")
    if type(relative_mad) not in (int, float) or not 0 <= relative_mad < 1:
        raise ValueError("cpu_acceptance max_relative_mad must be in [0, 1)")
    expected_statistics = {
        "method": "independent-stratified-bootstrap-log-geomean-v1",
        "resamples": 20000,
        "seed": 123,
        "one_sided_confidence": 0.95,
        "regression_ratio": 1.0,
        "max_relative_mad": 0.05,
    }
    if statistics != expected_statistics:
        raise ValueError("cpu_acceptance statistics do not match the frozen contract")
    if reference["capture_steps"] != sorted(set(reference["capture_steps"])):
        raise ValueError("capture_steps must be unique and increasing")
    cases = (
        data["correctness"]
        + data.get("benchmarks", [])
        + data.get("physical_checks", [])
    )
    names = [case["name"] for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("workload names must be unique")
    known = set(names)
    for gate_name, gate in data["performance_gates"].items():
        if isinstance(gate, dict) and "cases" in gate:
            unknown = set(gate["cases"]) - known
            if unknown:
                raise ValueError(
                    f"{gate_name} references unknown workloads: {sorted(unknown)}"
                )
    return data


def _json_value(value):
    """Convert coefficient and configuration metadata to stable JSON values."""
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN coefficient metadata is not allowed")
        if math.isinf(value):
            return {
                "nonfinite": ("positive-infinity" if value > 0 else "negative-infinity")
            }
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, complex):
        return {"real": _json_value(value.real), "imag": _json_value(value.imag)}
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if hasattr(value, "__dict__"):
        return {
            "type": type(value).__name__,
            "values": {
                name: _json_value(item)
                for name, item in sorted(vars(value).items())
                if not name.startswith("_")
                and name not in {"f", "geom_tree", "space", "aux_fdtd"}
            },
        }
    return repr(value)


def _command_output(command):
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError, subprocess.CalledProcessError:
        return None


def _git_output(checkout, *arguments):
    """Return required Git metadata or fail instead of emitting weak evidence."""
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot determine Git provenance for {Path(checkout).resolve()}"
        ) from error


def _git_checkout(path):
    source = Path(path).resolve(strict=True)
    checkout = Path(_git_output(source.parent, "rev-parse", "--show-toplevel"))
    return checkout.resolve(strict=True)


def _checkout_provenance(checkout, source):
    checkout = Path(checkout).resolve(strict=True)
    source = Path(source).resolve(strict=True)
    if not source.is_relative_to(checkout):
        raise RuntimeError(f"provenance source is outside its checkout: {source}")
    actual_checkout = Path(
        _git_output(checkout, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    if actual_checkout != checkout:
        raise RuntimeError(
            f"requested checkout {checkout} resolves to Git checkout {actual_checkout}"
        )
    commit = _git_output(checkout, "rev-parse", "HEAD")
    status = _git_output(checkout, "status", "--short", "--untracked-files=all")
    return {
        "checkout": str(checkout),
        "commit": commit,
        "git_status": status,
        "clean": status == "",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def material_from_name(name, gmes):
    """Build one material strategy from its backend-neutral manifest name."""
    drude_poles = tuple(
        gmes.DrudePole(omega=0.6 + 0.1 * index, gamma=0.03 + 0.01 * index)
        for index in range(4 if name.endswith("-4") else 1)
    )
    lorentz_poles = tuple(
        gmes.LorentzPole(
            amp=0.05 + 0.01 * index,
            omega=0.8 + 0.1 * index,
            gamma=0.03 + 0.01 * index,
        )
        for index in range(4 if name.endswith("-4") else 1)
    )
    critical_points = (
        gmes.CriticalPoint(amp=0.04, phi=0.2, omega=0.9, gamma=0.03),
        gmes.CriticalPoint(amp=0.02, phi=-0.1, omega=1.1, gamma=0.04),
    )
    factories = {
        "dummy": lambda: gmes.Dummy(eps_inf=1.1, mu_inf=1.05),
        "const": lambda: gmes.Const(value=0.25, eps_inf=1.1, mu_inf=1.05),
        "dielectric": lambda: gmes.Dielectric(eps_inf=1.7, mu_inf=1.05),
        "upml": gmes.Upml,
        "cpml": gmes.Cpml,
        "drude-1": lambda: gmes.Drude(eps_inf=1.2, dps=drude_poles),
        "drude-4": lambda: gmes.Drude(eps_inf=1.2, dps=drude_poles),
        "lorentz-1": lambda: gmes.Lorentz(eps_inf=1.2, lps=lorentz_poles),
        "lorentz-4": lambda: gmes.Lorentz(eps_inf=1.2, lps=lorentz_poles),
        "dcp-ade": lambda: gmes.DcpAde(
            eps_inf=1.2, dps=drude_poles[:1], cps=critical_points
        ),
        "dcp-plrc": lambda: gmes.DcpPlrc(
            eps_inf=1.2, dps=drude_poles[:1], cps=critical_points
        ),
        "dcp-rc": lambda: gmes.DcpRc(
            eps_inf=1.2, dps=drude_poles[:1], cps=critical_points
        ),
        "dm2-1": lambda: gmes.Dm2(
            eps_inf=1.2,
            omega=(0.8,),
            n_atom=(0.01,),
            gamma=0.02,
            rtol=1e-4,
        ),
        "dm2-4": lambda: gmes.Dm2(
            eps_inf=1.2,
            omega=(0.7, 0.8, 0.9, 1.0),
            n_atom=(0.01, 0.01, 0.01, 0.01),
            gamma=0.02,
            rtol=1e-4,
        ),
    }
    try:
        return factories[name]()
    except KeyError as error:
        raise ValueError(f"unknown material strategy: {name}") from error


def _mixed_geometry(spec, gmes):
    size = tuple(float(value) for value in spec["size"])
    width = float(size[0]) / 9
    active_size = min(value for value in size if value > 0)
    names = (
        "upml",
        "drude-1",
        "lorentz-1",
        "dcp-ade",
        "dcp-plrc",
        "dcp-rc",
        "dm2-1",
    )
    background = material_from_name("dielectric", gmes)
    if size[2] > 0:
        # Match the instantaneous dispersive coefficients so the fixed,
        # source-free seed does not excite a nonphysical interface mode.
        background = gmes.Dielectric(eps_inf=1.2, mu_inf=1.0)
    geometry = [gmes.DefaultMedium(background)]
    geometry.append(
        gmes.Shell(
            material=material_from_name("cpml", gmes),
            thickness=min(max(0.25, width / 4), active_size / 8),
        )
    )
    z_size = max(float(size[2]) * 0.65, 1.0)
    for index, name in enumerate(names):
        center_x = -float(size[0]) / 2 + (index + 1.5) * width
        region_size = (width * 0.72, float(size[1]) * 0.65, z_size)
        if name == "dm2-1" and float(size[2]) > 0:
            geometry.append(
                gmes.Sphere(
                    material=material_from_name(name, gmes),
                    center=(
                        float(size[0]) / 2 - 0.5 / spec["resolution"],
                        0,
                        0,
                    ),
                    radius=0.2 / spec["resolution"],
                )
            )
        elif name == "upml":
            geometry.append(
                gmes.Shell(
                    material=material_from_name(name, gmes),
                    center=(center_x, 0, 0),
                    size=region_size,
                    thickness=min(max(0.25, width * 0.2), active_size / 4),
                )
            )
        else:
            geometry.append(
                gmes.Block(
                    material=material_from_name(name, gmes),
                    center=(center_x, 0, 0),
                    size=region_size,
                )
            )
    return geometry


def _coverage_geometry(spec, gmes):
    """Build measured contiguous or fragmented material coverage workloads."""
    size = np.maximum(np.asarray(spec["size"], dtype=float), 1.0)
    fraction = float(spec["coverage_percent"]) / 100
    families = spec.get(
        "families",
        [
            "drude-1",
            "lorentz-1",
            "dcp-ade",
            "dcp-plrc",
            "dcp-rc",
            "dm2-1",
        ],
    )
    if float(spec["size"][2]) > 0:
        families = [name for name in families if not name.startswith("dm2-")]
    geometry = [gmes.DefaultMedium(material_from_name("dielectric", gmes))]
    if spec.get("include_pml", True):
        geometry.append(
            gmes.Shell(
                material=material_from_name("cpml", gmes),
                thickness=max(0.25, min(size) * 0.04),
            )
        )
    total_width = max(size[0] * fraction, len(families) / spec["resolution"])
    family_width = total_width / len(families)
    origin = -0.5 * total_width
    fragments = 1 if spec.get("layout") == "contiguous" else 4
    for family_index, family in enumerate(families):
        for fragment in range(fragments):
            width = family_width / fragments
            center_x = (
                origin + (family_index + (fragment + 0.5) / fragments) * family_width
            )
            if fragments == 1:
                center_y = 0.0
                height = size[1]
            else:
                height = size[1] / fragments
                center_y = -0.5 * size[1] + (fragment + 0.5) * height
            geometry.append(
                gmes.Block(
                    material=material_from_name(family, gmes),
                    center=(center_x, center_y, 0),
                    size=(width * 0.92, height * 0.82, size[2]),
                )
            )
    return geometry


def _heterogeneous_geometry(spec, gmes):
    geometry = [gmes.DefaultMedium(material_from_name("dielectric", gmes))]
    count = int(spec.get("region_count", 16))
    side = int(np.ceil(np.sqrt(count)))
    spacing_x = float(spec["size"][0]) / (side + 1)
    spacing_y = max(float(spec["size"][1]), 1.0) / (side + 1)
    for index in range(count):
        x_index, y_index = divmod(index, side)
        geometry.append(
            gmes.Cylinder(
                material=gmes.Dielectric(eps_inf=2.0 + 0.1 * (index % 5)),
                center=(
                    -0.5 * float(spec["size"][0]) + (x_index + 1) * spacing_x,
                    -0.5 * float(spec["size"][1]) + (y_index + 1) * spacing_y,
                    0,
                ),
                axis=(0, 0, 1),
                radius=0.22 * min(spacing_x, spacing_y),
            )
        )
    if spec.get("overlap"):
        geometry.extend(
            gmes.Block(
                material=gmes.Dielectric(eps_inf=3.0 + index),
                center=(0.08 * index, -0.08 * index, 0),
                size=(
                    float(spec["size"][0]) * (0.75 - index * 0.08),
                    max(float(spec["size"][1]), 1.0) * (0.75 - index * 0.08),
                    max(float(spec["size"][2]), 1.0),
                ),
            )
            for index in range(5)
        )
    return geometry


def _build_sources(spec, gmes):
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
    interface_size = tuple(max(value * 0.55, 1.0) for value in size)
    common = {
        "src_time": source_time,
        "center": (0, 0, 0),
        "size": interface_size,
        "direction": (1, 0, 0),
        "polarization": (0, 1, 0),
        "amp": amplitude,
    }
    if source_name == "tfsf":
        return [gmes.TotalFieldScatteredField(**common)]
    if source_name == "gaussian":
        return [gmes.GaussianBeam(directivity=gmes.PlusX, **common)]
    raise ValueError(f"unknown source recipe: {source_name}")


def build_simulation(spec, gmes):
    """Reject native construction in the candidate checkout.

    Historical captures are executable only by ``run_isolated_oracle.py`` from
    the manifest-pinned observer checkout.  This module remains a reader and
    archive validator for those immutable records.
    """
    raise RuntimeError(
        "native simulation construction is retired in this checkout; use the "
        "manifest-pinned historical observer"
    )


def initial_field_values(shapes, seed, scale=1e-3, *, complex_fields=False):
    """Build backend-neutral fixed-seed fields in canonical component order."""
    if set(shapes) != set(COMPONENT_NAMES):
        raise ValueError("field shapes must contain all canonical Yee components")
    rng = np.random.default_rng(seed)
    result = {}
    for name in COMPONENT_NAMES:
        shape = tuple(int(length) for length in shapes[name])
        values = scale * (1 + 0.1 * rng.random())
        for axis, length in enumerate(shape):
            ramp_shape = [1] * len(shape)
            ramp_shape[axis] = length
            values = values + (
                scale
                * 1e-6
                * (axis + 1)
                * np.linspace(0, 1, length).reshape(ramp_shape)
            )
        if complex_fields:
            values = values + 1j * scale * (1 + 0.1 * rng.random())
        result[name] = np.broadcast_to(values, shape).copy()
    return result


def _linear_run_count(indices, shape):
    if not len(indices):
        return 0
    linear = np.sort(np.ravel_multi_index(indices.T, shape))
    return int(1 + np.count_nonzero(np.diff(linear) != 1))


def _source_param_values(parameter, time):
    values = []
    for name in ("amp", "eps_inf", "mu_inf"):
        value = getattr(parameter, name, None)
        if isinstance(value, dict):
            values.extend(complex(item) for item in value.values())
        elif value is not None:
            values.append(complex(value))
    source_time = getattr(parameter, "src_time", None)
    if source_time is not None:
        values.append(complex(source_time.oscillator(time)))
    for name in ("r0", "r1"):
        values.extend(complex(value) for value in getattr(parameter, name, {}).values())
    for name in ("samp_idx0", "samp_idx1"):
        for index in getattr(parameter, name, {}).values():
            values.extend(complex(value) for value in index)
    return values


def _source_records(*_args, **_kwargs):
    """Reject candidate-checkout use of the retired native source adapter."""
    raise RuntimeError(
        "native source-record capture is retired in this checkout; use the "
        "manifest-pinned historical observer"
    )


def capture_case(spec, manifest, output):
    """Reject generation of a native archive in the candidate checkout."""
    raise RuntimeError(
        "native oracle capture is retired in this checkout; invoke the pinned "
        "historical observer through run_isolated_oracle.py"
    )


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def read_metadata(archive):
    if archive.files.count("metadata.json") != 1:
        raise ValueError("archive must contain exactly one metadata.json")
    encoded = archive["metadata.json"]
    if encoded.shape != () or encoded.dtype.kind not in {"U", "S"}:
        raise ValueError("metadata.json must be a scalar string array")
    metadata = json.loads(
        str(encoded),
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must decode to an object")
    return metadata


def _contract(condition, message):
    if not condition:
        raise ValueError(message)


def _same_json_value(actual, expected):
    """Compare decoded JSON values without bool/int or signed-zero aliases."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, float):
        return actual == expected and math.copysign(1.0, actual) == math.copysign(
            1.0, expected
        )
    return actual == expected


def _finite_json_value(value):
    """Return whether a decoded metadata value is finite, canonical JSON."""
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite_json_value(item)
            for key, item in value.items()
        )
    return False


def _exact_keys(value, expected, label, optional=()):
    _contract(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    expected = set(expected)
    optional = set(optional)
    _contract(
        expected <= actual <= expected | optional,
        f"{label} keys are invalid: missing={sorted(expected - actual)}, "
        f"unexpected={sorted(actual - expected - optional)}",
    )


def _full_commit(value):
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_archive_provenance(metadata, manifest, role):
    provenance = metadata["provenance"]
    _exact_keys(provenance, {"source", "controller"}, "provenance")
    for name in ("source", "controller"):
        record = provenance[name]
        _exact_keys(
            record,
            {
                "checkout",
                "commit",
                "git_status",
                "clean",
                "source",
                "source_sha256",
            },
            f"provenance.{name}",
        )
        _contract(_full_commit(record["commit"]), f"{name} commit is not full")
        _contract(
            isinstance(record["git_status"], str),
            f"{name} git_status must be a string",
        )
        _contract(
            record["clean"] is (record["git_status"] == ""),
            f"{name} clean flag disagrees with git_status",
        )
        _contract(record["clean"] is True, f"{name} checkout is not clean")
        checkout = Path(record["checkout"])
        source = Path(record["source"])
        _contract(checkout.is_absolute(), f"{name} checkout must be absolute")
        _contract(source.is_absolute(), f"{name} source must be absolute")
        _contract(
            source.is_relative_to(checkout),
            f"{name} source is outside its checkout",
        )
        _contract(
            _sha256_digest(record["source_sha256"]),
            f"{name} source_sha256 is invalid",
        )
    if role == "reference":
        _contract(
            provenance["source"]["commit"] == manifest["reference"]["observer_commit"],
            "reference checkout does not match reference.observer_commit",
        )
    _contract(
        metadata["reference_source"] == provenance["source"]["source"],
        "reference_source disagrees with source provenance",
    )


def _close_number(actual, expected):
    return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15)


def _validate_material_records(
    archive, records, step, prefix, field_shapes, required_keys
):
    _contract(isinstance(records, list), f"step/{step}/{prefix} must be a list")
    ordinals = {component: 0 for component in COMPONENT_NAMES}
    signatures = []
    required_record_keys = {
        "component",
        "strategy",
        "strategies",
        "native_type",
        "cells",
        "coverage",
        "fragmentation_runs",
        "fragmentation_ratio",
        "state_values",
        "state_nonzero_values",
        "state_width",
        "state_key",
        "state_bytes",
        "plan_bytes",
        "index_bytes",
        "parameter_bytes",
        "live_updater_bytes",
        "plan_runs",
        "bucket_signature",
    }
    for record in records:
        _exact_keys(
            record,
            required_record_keys,
            f"step/{step}/{prefix} material record",
            optional={"backend_metadata"},
        )
        component = record["component"]
        _contract(component in COMPONENT_NAMES, "material component is unknown")
        strategies = record["strategies"]
        _contract(
            isinstance(strategies, list)
            and strategies
            and all(isinstance(value, str) and value for value in strategies),
            "material strategies are invalid",
        )
        _contract(
            record["strategy"] == "+".join(strategies),
            "material strategy is not canonical",
        )
        cells = record["cells"]
        state_values = record["state_values"]
        _contract(type(cells) is int and cells >= 0, "material cells is invalid")
        _contract(
            type(state_values) is int and state_values >= 0,
            "material state_values is invalid",
        )
        ordinal = ordinals[component]
        ordinals[component] += 1
        record_prefix = (
            f"step/{step}/{prefix}/{component}/{ordinal}-{record['strategy']}"
        )
        index_key = f"{record_prefix}/indices"
        state_key = f"{record_prefix}/values"
        _contract(record["state_key"] == state_key, "material state_key is invalid")
        _contract(index_key in archive.files, f"missing required array: {index_key}")
        _contract(state_key in archive.files, f"missing required array: {state_key}")
        required_keys.update((index_key, state_key))
        indices = archive[index_key]
        state = archive[state_key]
        _contract(
            np.issubdtype(indices.dtype, np.integer) and indices.shape == (cells, 3),
            "material indices shape or dtype is invalid",
        )
        _contract(
            state.shape == (state_values,)
            and np.issubdtype(state.dtype, np.number)
            and bool(np.isfinite(state).all()),
            "material state shape, dtype, or finiteness is invalid",
        )
        _contract(
            record["state_nonzero_values"] == int(np.count_nonzero(state)),
            "material state_nonzero_values is inaccurate",
        )
        shape = field_shapes[component]
        expected_coverage = cells / int(np.prod(shape))
        _contract(
            _close_number(record["coverage"], expected_coverage),
            "material coverage is inaccurate",
        )
        runs = _linear_run_count(indices, shape)
        _contract(
            record["fragmentation_runs"] == runs
            and _close_number(
                record["fragmentation_ratio"], runs / cells if cells else 0.0
            )
            and _close_number(
                record["state_width"], state_values / cells if cells else 0.0
            ),
            "material fragmentation or state width is inaccurate",
        )
        for name in (
            "state_bytes",
            "plan_bytes",
            "index_bytes",
            "parameter_bytes",
            "live_updater_bytes",
            "plan_runs",
        ):
            _contract(
                type(record[name]) is int and record[name] >= 0,
                f"material {name} is invalid",
            )
        _contract(
            record["live_updater_bytes"]
            == record["plan_bytes"] + record["index_bytes"] + record["parameter_bytes"],
            "material live_updater_bytes is inaccurate",
        )
        signature = [
            component,
            record["strategy"],
            record["native_type"],
            cells,
            state_values,
        ]
        _contract(
            record["bucket_signature"] == signature,
            "material bucket_signature is inaccurate",
        )
        signatures.append(tuple(signature))
    return signatures


def _validate_source_arrays(archive, sources, step, required_keys):
    _exact_keys(sources, {"updaters", "auxiliary"}, f"step/{step}/sources")
    ordinals = {component: 0 for component in COMPONENT_NAMES}
    for record in sources["updaters"]:
        _exact_keys(
            record,
            {"component", "native_type", "cells", "state_values"},
            "source updater record",
            optional={"backend_metadata"},
        )
        component = record["component"]
        _contract(component in COMPONENT_NAMES, "source component is unknown")
        _contract(
            isinstance(record["native_type"], str) and record["native_type"],
            "source native_type is invalid",
        )
        _contract(
            type(record["cells"]) is int and record["cells"] >= 0,
            "source cells is invalid",
        )
        _contract(
            type(record["state_values"]) is int and record["state_values"] >= 0,
            "source state_values is invalid",
        )
        ordinal = ordinals[component]
        ordinals[component] += 1
        prefix = f"step/{step}/source/{component}/{ordinal}-{record['native_type']}"
        for suffix in ("indices", "values"):
            key = f"{prefix}/{suffix}"
            _contract(key in archive.files, f"missing required array: {key}")
            required_keys.add(key)
        indices = archive[f"{prefix}/indices"]
        values = archive[f"{prefix}/values"]
        _contract(
            indices.shape == (record["cells"], 3)
            and np.issubdtype(indices.dtype, np.integer),
            "source updater indices are invalid",
        )
        _contract(
            values.shape == (record["state_values"],)
            and np.issubdtype(values.dtype, np.number)
            and bool(np.isfinite(values).all()),
            "source updater state is invalid",
        )
    for ordinal, record in enumerate(sources["auxiliary"]):
        _exact_keys(
            record,
            {"source", "fields", "materials"},
            "source auxiliary record",
            optional={"backend_metadata"},
        )
        _exact_keys(record["fields"], COMPONENT_NAMES, "auxiliary fields")
        _contract(
            isinstance(record["source"], str) and record["source"],
            "auxiliary source is invalid",
        )
        prefix = f"step/{step}/source_aux/{ordinal}-{record['source']}"
        time_key = f"{prefix}/time"
        _contract(time_key in archive.files, f"missing required array: {time_key}")
        required_keys.add(time_key)
        auxiliary_time = archive[time_key]
        _contract(
            auxiliary_time.shape == (3,)
            and np.issubdtype(auxiliary_time.dtype, np.number)
            and bool(np.isfinite(auxiliary_time).all())
            and float(auxiliary_time[2]) > 0,
            "auxiliary time is invalid",
        )
        shapes = {}
        for component, shape in record["fields"].items():
            _contract(
                isinstance(shape, list)
                and shape
                and all(type(length) is int and length > 0 for length in shape),
                "auxiliary field shape metadata is invalid",
            )
            key = f"{prefix}/field/{component}"
            _contract(key in archive.files, f"missing required array: {key}")
            required_keys.add(key)
            field = archive[key]
            _contract(
                list(field.shape) == shape
                and np.issubdtype(field.dtype, np.number)
                and bool(np.isfinite(field).all()),
                "auxiliary field is invalid",
            )
            shapes[component] = tuple(shape)
        _validate_material_records(
            archive,
            record["materials"],
            step,
            f"source_aux_material/{ordinal}",
            shapes,
            required_keys,
        )


def _validate_physical_arrays(archive, physical, step, fields, required_keys):
    names = (
        "energy",
        "maximum_abs_field",
        "boundary_low_energy",
        "boundary_high_energy",
        "finite",
    )
    _exact_keys(physical, names, f"step/{step}/physical")
    prefix = f"step/{step}/physical"
    summary_key = f"{prefix}/summary"
    _contract(summary_key in archive.files, f"missing required array: {summary_key}")
    required_keys.add(summary_key)
    summary = archive[summary_key]
    _contract(summary.shape == (5,), "physical summary shape is invalid")
    calculated = [0.0, 0.0, 0.0, 0.0, 1.0]
    low_precision = False
    for component in COMPONENT_NAMES:
        field = fields[component]
        low_precision = low_precision or field.dtype.itemsize <= 4
        magnitude = np.abs(field)
        calculated[0] += float(np.sum(magnitude * magnitude))
        calculated[1] = max(calculated[1], float(np.max(magnitude)))
        calculated[2] += float(np.sum(magnitude[0] * magnitude[0]))
        calculated[3] += float(np.sum(magnitude[-1] * magnitude[-1]))
        calculated[4] = float(bool(calculated[4]) and np.isfinite(field).all())
        axes = tuple(range(1, field.ndim))
        line = np.mean(field, axis=axes) if axes else field
        expected = np.abs(np.fft.fft(line))
        key = f"{prefix}/spectrum/{component}"
        _contract(key in archive.files, f"missing required array: {key}")
        required_keys.add(key)
        tolerance = 2e-5 if low_precision else 2e-12
        _contract(
            archive[key].shape == expected.shape
            and np.allclose(archive[key], expected, rtol=tolerance, atol=0),
            f"physical spectrum is inconsistent for {component}",
        )
    tolerance = 2e-5 if low_precision else 2e-12
    recorded = [physical[name] for name in names[:-1]] + [float(physical["finite"])]
    _contract(
        physical["finite"] is True
        and np.allclose(summary, calculated, rtol=tolerance, atol=0)
        and np.allclose(summary, recorded, rtol=tolerance, atol=0),
        "physical summary is inconsistent with complete fields",
    )
    return calculated[0]


def _validate_mixed_source_free_stability(workload, field_energies, capture_steps):
    if workload.get("recipe") != "mixed" or workload.get("source") != "none":
        return
    initial_energy = field_energies["0"]
    _contract(
        isinstance(initial_energy, (int, float))
        and not isinstance(initial_energy, bool)
        and math.isfinite(initial_energy)
        and initial_energy > 0,
        "mixed source-free initial energy is invalid",
    )
    for step in capture_steps:
        energy = field_energies[str(step)]
        _contract(
            isinstance(energy, (int, float))
            and not isinstance(energy, bool)
            and math.isfinite(energy)
            and 0 <= energy < initial_energy * MIXED_SOURCE_FREE_MAX_ENERGY_RATIO,
            f"mixed source-free energy is unstable at step {step}",
        )


def _validate_archive(archive, manifest, role):
    _contract(
        len(archive.files) == len(set(archive.files)),
        "archive array names must be unique",
    )
    metadata = read_metadata(archive)
    _exact_keys(
        metadata,
        {
            "schema_version",
            "backend",
            "workload",
            "reference",
            "capture_steps",
            "input_state",
            "maps",
            "steps",
            "geometry_and_coefficients",
            "active_cells",
            "state_bytes",
            "plan_bytes",
            "index_bytes",
            "parameter_bytes",
            "live_updater_bytes",
            "archive_array_bytes",
            "nonzero_seed",
            "nonzero_persistent_state",
            "provenance",
            "reference_source",
        },
        "archive metadata",
        optional={"backend_metadata"},
    )
    _contract(
        metadata["schema_version"] == ARCHIVE_SCHEMA_VERSION,
        "unsupported correctness archive schema",
    )
    backend = metadata["backend"]
    _contract(
        isinstance(backend, str) and backend in manifest["tolerances"],
        "archive backend is unknown",
    )
    if role == "reference":
        _contract(backend == "native", "reference archive backend must be native")
    if "backend_metadata" in metadata:
        _contract(
            isinstance(metadata["backend_metadata"], dict),
            "backend_metadata must be an object",
        )
    workload = metadata["workload"]
    _contract(isinstance(workload, dict), "workload must be an object")
    _contract(
        workload == find_case(manifest, workload.get("name")),
        "workload does not exactly match the manifest",
    )
    _contract(
        metadata["reference"] == _correctness_reference_contract(manifest["reference"]),
        "reference contract does not exactly match the manifest",
    )
    expected_steps = workload.get(
        "capture_steps", manifest["reference"]["capture_steps"]
    )
    _contract(
        isinstance(expected_steps, list)
        and expected_steps
        and all(type(step) is int and step > 0 for step in expected_steps)
        and expected_steps == sorted(set(expected_steps)),
        "capture step contract is invalid",
    )
    _contract(
        metadata["capture_steps"] == expected_steps,
        "capture_steps do not exactly match the manifest workload",
    )
    _contract(
        metadata["input_state"]
        == {
            "archive_prefix": "step/0",
            "precondition_steps": manifest["reference"]["precondition_steps"],
            "relative_capture_steps": True,
        },
        "input_state contract is invalid",
    )
    _validate_archive_provenance(metadata, manifest, role)

    maps = metadata["maps"]
    _exact_keys(maps, COMPONENT_NAMES, "maps")
    required_keys = set()
    field_shapes = {}
    map_record_keys = {
        "shape",
        "dtype",
        "active_cells",
        "material_regions",
        "underlying_regions",
    }
    for component in COMPONENT_NAMES:
        record = maps[component]
        _exact_keys(record, map_record_keys, f"maps.{component}")
        shape = record["shape"]
        _contract(
            isinstance(shape, list)
            and shape
            and all(type(length) is int and length > 0 for length in shape),
            f"maps.{component}.shape is invalid",
        )
        _contract(
            isinstance(record["dtype"], str)
            and np.issubdtype(np.dtype(record["dtype"]), np.number),
            f"maps.{component}.dtype is invalid",
        )
        _contract(
            all(
                type(record[name]) is int and record[name] >= 0
                for name in (
                    "active_cells",
                    "material_regions",
                    "underlying_regions",
                )
            ),
            f"maps.{component} counts are invalid",
        )
        field_shapes[component] = tuple(shape)
        material_key = f"map/{component}/material_ids"
        underlying_key = f"map/{component}/underlying_ids"
        for key in (material_key, underlying_key):
            _contract(key in archive.files, f"missing required array: {key}")
            required_keys.add(key)
        material_ids = archive[material_key]
        underlying_ids = archive[underlying_key]
        _contract(
            material_ids.shape == underlying_ids.shape
            and material_ids.shape == (int(np.prod(shape)),)
            and np.issubdtype(material_ids.dtype, np.integer)
            and np.issubdtype(underlying_ids.dtype, np.integer),
            f"map arrays are invalid for {component}",
        )
        _contract(
            record["active_cells"] == material_ids.size
            and record["material_regions"] == int(np.unique(material_ids).size),
            f"map metadata is inaccurate for {component}",
        )
        underlying = underlying_ids[underlying_ids >= 0]
        _contract(
            record["underlying_regions"] == int(np.unique(underlying).size),
            f"underlying map metadata is inaccurate for {component}",
        )

    steps = metadata["steps"]
    expected_step_names = {"0"} | {str(step) for step in expected_steps}
    _exact_keys(steps, expected_step_names, "steps")
    baseline_signatures = None
    all_material_records = []
    field_energies = {}
    for step in (0, *expected_steps):
        step_name = str(step)
        snapshot = steps[step_name]
        _exact_keys(
            snapshot,
            {"materials", "sources", "physical"},
            f"step/{step_name}",
        )
        fields = {}
        for component in COMPONENT_NAMES:
            key = f"step/{step_name}/field/{component}"
            _contract(key in archive.files, f"missing required array: {key}")
            required_keys.add(key)
            field = archive[key]
            _contract(
                field.shape == field_shapes[component]
                and str(field.dtype) == maps[component]["dtype"]
                and np.issubdtype(field.dtype, np.number)
                and bool(np.isfinite(field).all()),
                f"complete field array is invalid: {key}",
            )
            fields[component] = field
        time_key = f"step/{step_name}/time"
        _contract(time_key in archive.files, f"missing required array: {time_key}")
        required_keys.add(time_key)
        time = archive[time_key]
        expected_time_step = manifest["reference"]["precondition_steps"] + step
        _contract(
            time.shape == (3,)
            and np.issubdtype(time.dtype, np.number)
            and bool(np.isfinite(time).all())
            and float(time[2]) > 0
            and float(time[0]) == expected_time_step
            and _close_number(time[1], expected_time_step * time[2]),
            f"time array does not match relative step contract: {time_key}",
        )
        signatures = _validate_material_records(
            archive,
            snapshot["materials"],
            step_name,
            "state",
            field_shapes,
            required_keys,
        )
        _contract(signatures, f"step/{step_name} has no material state records")
        if baseline_signatures is None:
            baseline_signatures = signatures
        else:
            _contract(
                signatures == baseline_signatures,
                f"step/{step_name} material topology changed",
            )
        all_material_records.extend(snapshot["materials"])
        _validate_source_arrays(archive, snapshot["sources"], step_name, required_keys)
        field_energies[step_name] = _validate_physical_arrays(
            archive, snapshot["physical"], step_name, fields, required_keys
        )

    _validate_mixed_source_free_stability(workload, field_energies, expected_steps)

    geometry_and_coefficients = metadata["geometry_and_coefficients"]
    _contract(
        isinstance(geometry_and_coefficients, list) and geometry_and_coefficients,
        "geometry_and_coefficients must be a nonempty list",
    )
    for record in geometry_and_coefficients:
        _exact_keys(
            record,
            {"geometry", "material"},
            "geometry_and_coefficients",
        )
        _contract(
            isinstance(record["geometry"], str) and record["geometry"],
            "geometry_and_coefficients geometry is invalid",
        )
        _contract(
            _finite_json_value(record["material"]),
            "geometry_and_coefficients material is invalid",
        )
    final_records = steps[str(expected_steps[-1])]["materials"]
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
    for name, expected in totals.items():
        _contract(
            type(metadata[name]) is int and metadata[name] == expected,
            f"{name} byte or cell accounting is inaccurate",
        )
    _contract(metadata["nonzero_seed"] is True, "nonzero_seed must be true")
    expected_nonzero_state = all(
        record["state_values"] == 0 or record["state_nonzero_values"] > 0
        for record in all_material_records
    )
    _contract(
        metadata["nonzero_persistent_state"] is expected_nonzero_state
        and expected_nonzero_state,
        "nonzero_persistent_state is inaccurate",
    )
    actual_keys = set(archive.files) - {"metadata.json"}
    _contract(
        actual_keys == required_keys,
        "archive array keys are invalid: "
        f"missing={sorted(required_keys - actual_keys)}, "
        f"unexpected={sorted(actual_keys - required_keys)}",
    )
    archive_bytes = sum(archive[key].nbytes for key in required_keys)
    _contract(
        type(metadata["archive_array_bytes"]) is int
        and metadata["archive_array_bytes"] == archive_bytes,
        "archive_array_bytes is inaccurate",
    )
    return metadata


def compare_archives(reference_path, candidate_path, manifest):
    """Compare every complete array; raise with the precise failing key."""
    failures = []
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        reference_metadata = None
        candidate_metadata = None
        for role, archive in (("reference", reference), ("candidate", candidate)):
            try:
                metadata = _validate_archive(archive, manifest, role)
            except (
                KeyError,
                TypeError,
                ValueError,
                IndexError,
                OverflowError,
            ) as error:
                failures.append(
                    {
                        "key": f"{role}/archive-contract",
                        "error": str(error),
                    }
                )
            else:
                if role == "reference":
                    reference_metadata = metadata
                else:
                    candidate_metadata = metadata
        if reference_metadata is None or candidate_metadata is None:
            return {"passed": False, "failures": failures}
        if not _same_json_value(
            candidate_metadata["geometry_and_coefficients"],
            reference_metadata["geometry_and_coefficients"],
        ):
            failures.append(
                {
                    "key": "geometry_and_coefficients",
                    "error": "candidate geometry and coefficients differ from reference",
                }
            )
        backend = candidate_metadata["backend"]
        reference_keys = set(reference.files) - {"metadata.json"}
        candidate_keys = set(candidate.files) - {"metadata.json"}
        if reference_keys != candidate_keys:
            failures.append(
                {
                    "key": "archive",
                    "missing": sorted(reference_keys - candidate_keys),
                    "unexpected": sorted(candidate_keys - reference_keys),
                }
            )
        for key in sorted(reference_keys & candidate_keys):
            expected = reference[key]
            actual = candidate[key]
            tolerance = tolerance_for_key(manifest, backend, key, str(expected.dtype))
            same_shape = expected.shape == actual.shape
            if np.issubdtype(expected.dtype, np.number):
                equal = same_shape and np.allclose(
                    expected,
                    actual,
                    rtol=tolerance["rtol"],
                    atol=tolerance["atol"],
                    equal_nan=False,
                )
            else:
                equal = same_shape and np.array_equal(expected, actual)
            if not equal:
                difference = (
                    np.abs(expected - actual)
                    if expected.shape == actual.shape
                    else np.array([np.inf])
                )
                failures.append(
                    {
                        "key": key,
                        "expected_shape": list(expected.shape),
                        "actual_shape": list(actual.shape),
                        "max_abs_error": float(np.max(difference)),
                        "rtol": tolerance["rtol"],
                        "atol": tolerance["atol"],
                    }
                )
    return {"passed": not failures, "failures": failures}


def tolerance_for_key(manifest, backend, key, dtype):
    tolerances = manifest["tolerances"]
    if backend == "native":
        return tolerances["native"].get(dtype, {"rtol": 0.0, "atol": 0.0})
    backend_tolerances = tolerances[backend]
    normalized = key.lower()
    for model in (
        "dcp-plrc",
        "dcp-ade",
        "dcp-rc",
        "lorentz",
        "drude",
        "dm2",
        "pml",
        "dielectric",
    ):
        if model in normalized:
            if model == "dm2" and dtype == "complex128":
                dtype = "float64"
            return backend_tolerances[model].get(dtype, {"rtol": 0.0, "atol": 0.0})
    return backend_tolerances["mixed"].get(dtype, {"rtol": 0.0, "atol": 0.0})


def benchmark_case(spec, manifest, repeats, warmup, steps):
    """Reject native timing generation in the candidate checkout."""
    raise RuntimeError(
        "native benchmark execution is retired in this checkout; retain the "
        "manifest-pinned historical summary instead"
    )


def find_case(manifest, name):
    try:
        return next(
            case
            for group in ("correctness", "benchmarks", "physical_checks")
            for case in manifest.get(group, [])
            if case["name"] == name
        )
    except StopIteration as error:
        raise ValueError(f"unknown correctness workload: {name}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe = subparsers.add_parser("describe")
    describe.add_argument("--case")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "describe":
        value = find_case(manifest, args.case) if args.case else manifest
    elif args.command == "compare":
        value = compare_archives(args.reference, args.candidate, manifest)
        if not value["passed"]:
            print(json.dumps(value, indent=2, sort_keys=True))
            raise SystemExit(1)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
