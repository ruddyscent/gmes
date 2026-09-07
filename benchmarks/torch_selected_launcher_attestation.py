"""Fail-closed, opt-in host launcher-selection evidence for one pinned case.

This diagnostic observes live Inductor objects in a fresh process.  A valid
record means only that a mapped host launcher returned normally from its
Python call.  It does not attest CUDA device execution or completion.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCOPE = "pinned-torch-2.13-cuda-selected-launcher-host-call-v1"
PINNED_CASE = "all-material-2d"
PINNED_TORCH_VERSION = "2.13.0+cu130"
PINNED_TRITON_VERSION = "3.7.1"
PINNED_PLAN_IDENTITY = (
    "d04329bd2dc968c5adf12dfc6aa56b2040b8e1222f2baf6ffa31399436cf75db"
)
PINNED_COMPILE_CACHE_KEY = (
    "ec5b2e56eedd3b48f0166cb565ab731aa0130f9b078c4481db788a40a2671221"
)
PINNED_RUNTIME_SOURCES = MappingProxyType(
    {
        "async_compile.py": (
            "401933b518592b8958bb137ddd963c8dd623f4c63b9de215c933424ccc296c7d"
        ),
        "triton_heuristics.py": (
            "87a1937c8179f44b082736d6144b6f29e7cb56ec771a0fe973379aa99fcb3923"
        ),
        "static_triton_launcher.py": (
            "0510a7862f64f8fdab710cc03b4109490b1597b7c4548a79a0a025fc31179421"
        ),
    }
)
PINNED_WRAPPER_SHA256 = MappingProxyType(
    {
        "electric_half": (
            "cf88ea4cb9d493cda27ccb3c631140d236158334896e3365b29f81528d3ad61f"
        ),
        "magnetic_half": (
            "928087edd3d81a1674f6b7947bfcf252537930889c0d666c421169c3ba728f83"
        ),
    }
)
MAGNETIC_KERNEL = "triton_poi_fused_copy_fma_mul_slice_sub_13"
ELECTRIC_KERNEL = "triton_poi_fused_index_copy_view_38"
_MISSING = object()


class AttestationError(ValueError):
    """Raised when the narrow host-side attestation is not established."""


@dataclass(frozen=True)
class _ExpectedEvent:
    region: str
    ordinal: int
    kernel_name: str


PINNED_EVENTS = (
    _ExpectedEvent("magnetic_half", 0, MAGNETIC_KERNEL),
    _ExpectedEvent("electric_half", 0, ELECTRIC_KERNEL),
    _ExpectedEvent("electric_half", 1, ELECTRIC_KERNEL),
    _ExpectedEvent("electric_half", 2, ELECTRIC_KERNEL),
    _ExpectedEvent("electric_half", 3, ELECTRIC_KERNEL),
)


@dataclass(frozen=True)
class _Construction:
    callable_id: int
    kernel_name: str
    kernel_hash: str
    launcher_kind: str
    cache_hash: str
    config: Mapping[str, object]
    cubin_path: str
    cubin_sha256: str
    variant_directory: str
    globals_id: int
    code_id: int
    runner_id: int


@dataclass(frozen=True)
class _FastDerivation:
    parent_id: int
    derived_id: int
    cache_hash: str
    config: Mapping[str, object]
    globals_id: int
    code_id: int
    runner_id: int


@dataclass(frozen=True)
class _LoadedKernel:
    compile_result: object
    kernel: object
    name: str
    kernel_hash: str
    path: str
    sha256: str
    directory: str
    state: str


def _loaded_state(kernel: object) -> str:
    return json.dumps(
        _json_value(
            {
                name: getattr(kernel, name)
                for name in (
                    "module",
                    "function",
                    "num_warps",
                    "shared",
                    "arg_tys",
                    "has_global_scratch",
                    "has_profile_scratch",
                )
            }
        ),
        sort_keys=True,
    )


@dataclass(frozen=True)
class _RunSnapshot:
    benchmark_run: bool
    cached_launcher: object | None
    debug_active: bool
    kwargs_empty: bool
    plugins_empty: bool
    profiler_active: bool
    triton_interpret: bool

    @property
    def fast(self) -> bool:
        return (
            self.cached_launcher is not None
            and not self.benchmark_run
            and self.kwargs_empty
            and self.plugins_empty
            and not self.profiler_active
            and not self.debug_active
            and not self.triton_interpret
        )

    @property
    def slow_supported(self) -> bool:
        return (
            not self.benchmark_run
            and self.kwargs_empty
            and self.plugins_empty
            and not self.profiler_active
            and not self.debug_active
            and not self.triton_interpret
        )


@dataclass(frozen=True)
class _RuntimeAdapter:
    static_result_type: type
    regular_result_type: type
    autotuner_type: type
    profiler_active: Callable[[], bool]
    debug_active: Callable[[], bool]
    verify: Callable[[], None]
    cache_empty: Callable[[], bool]
    static_kernel_type: type
    cache_key: Callable[[str], str]


def _sha256_regular_file(path: Path) -> tuple[str, str, str]:
    """Return a strict CUBIN identity without following a symlink."""
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise AttestationError("CUBIN is not an absolute regular file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise AttestationError("CUBIN path is not canonical")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return str(path), digest.hexdigest(), str(path.parent)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AttestationError("configuration has a non-finite value")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise AttestationError("configuration has a non-string key")
        return {key: _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise AttestationError("configuration is not JSON-primitive")


def _canonical_config(config: object) -> Mapping[str, object]:
    try:
        values = vars(config)
    except TypeError as error:
        raise AttestationError(
            "launcher configuration has no instance state"
        ) from error
    if not values:
        raise AttestationError("launcher configuration is empty")
    encoded = _json_value(values)
    if not isinstance(encoded, dict):
        raise AttestationError("launcher configuration is malformed")
    return encoded


def _kernel_metadata(compile_result: object) -> tuple[str, str]:
    kernel = getattr(compile_result, "kernel")
    name = getattr(kernel, "name")
    kernel_hash = getattr(kernel, "hash")
    if not isinstance(name, str) or not name:
        raise AttestationError("compile-result kernel name is invalid")
    if not isinstance(kernel_hash, str) or not kernel_hash:
        raise AttestationError("compile-result kernel hash is invalid")
    return name, kernel_hash


def _callable_metadata(
    launcher: object,
) -> tuple[str, Mapping[str, object], int, int, int]:
    cache_hash = getattr(launcher, "cache_hash")
    if not isinstance(cache_hash, str) or not cache_hash:
        raise AttestationError("launcher cache hash is invalid")
    config = _canonical_config(getattr(launcher, "config"))
    code = getattr(launcher, "__code__")
    globals_map = getattr(launcher, "__globals__")
    if not isinstance(globals_map, dict):
        raise AttestationError("launcher globals are invalid")
    runner = globals_map.get("runner")
    if not callable(runner):
        raise AttestationError("launcher runner is invalid")
    return cache_hash, config, id(globals_map), id(code), id(runner)


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _pinned_runtime_adapter() -> _RuntimeAdapter:
    """Load only the pinned installed runtime and verify its source bytes."""
    import torch
    import triton
    from torch._inductor import async_compile
    from torch._inductor.runtime import static_triton_launcher, triton_heuristics

    def verify() -> None:
        if torch.__version__ != PINNED_TORCH_VERSION:
            raise AttestationError("Torch version is not pinned")
        if triton.__version__ != PINNED_TRITON_VERSION:
            raise AttestationError("Triton version is not pinned")
        paths = {
            "async_compile.py": inspect.getsourcefile(async_compile),
            "triton_heuristics.py": inspect.getsourcefile(triton_heuristics),
            "static_triton_launcher.py": inspect.getsourcefile(static_triton_launcher),
        }
        for name, expected in PINNED_RUNTIME_SOURCES.items():
            raw_path = paths.get(name)
            if raw_path is None:
                raise AttestationError(f"runtime source path is unknown: {name}")
            path = Path(raw_path)
            if path.is_symlink() or not path.is_file():
                raise AttestationError(f"runtime source is not a regular file: {name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise AttestationError(f"runtime source digest differs: {name}")

    def cache_empty() -> bool:
        cache = getattr(async_compile.CompiledTritonKernels, "_cache", None)
        if not isinstance(cache, dict):
            raise AttestationError("compiled Triton cache layout is unknown")
        return not cache

    return _RuntimeAdapter(
        static_result_type=triton_heuristics.StaticTritonCompileResult,
        regular_result_type=triton_heuristics.TritonCompileResult,
        autotuner_type=triton_heuristics.CachingAutotuner,
        profiler_active=lambda: bool(
            triton_heuristics.autograd_profiler._is_profiler_enabled
        ),
        debug_active=lambda: bool(triton_heuristics.get_active_debug_mode()),
        verify=verify,
        cache_empty=cache_empty,
        static_kernel_type=static_triton_launcher.StaticallyLaunchedCudaKernel,
        cache_key=triton_heuristics.triton_hash_to_path_key,
    )


class SelectedLauncherObserver(AbstractContextManager["SelectedLauncherObserver"]):
    """Observe one fresh, fixed five-call host dispatch without altering it."""

    def __init__(self, adapter: _RuntimeAdapter | None = None):
        self._adapter = adapter or _pinned_runtime_adapter()
        self._active_region: str | None = None
        self._attesting = False
        self._closed = False
        self._entered = False
        self._final: Mapping[str, object] | None = None
        self._patches: list[tuple[type, str, object]] = []
        self._constructors: dict[int, _Construction] = {}
        self._fast_derivations: dict[int, _FastDerivation] = {}
        self._live_callables: dict[int, object] = {}
        self._live_results: dict[int, object] = {}
        self._events: list[dict[str, object]] = []
        self._region_counts: Counter[str] = Counter()
        self._problems: list[str] = []
        self._expected = {(item.region, item.ordinal): item for item in PINNED_EVENTS}
        self._making: list[tuple[object, object]] = []
        self._loads: dict[int, _LoadedKernel] = {}
        self._call_join: Any = None

    def __enter__(self) -> SelectedLauncherObserver:
        if self._entered or self._closed:
            raise AttestationError("observer is single-use")
        self._adapter.verify()
        if not self._adapter.cache_empty():
            raise AttestationError("preexisting compiled Triton cache is unsupported")
        self._entered = True
        try:
            self._patch(
                self._adapter.static_kernel_type, "load_kernel", self._observe_load
            )
            self._patch(
                self._adapter.static_result_type,
                "make_launcher",
                self._observe_static_make_launcher,
            )
            self._patch(
                self._adapter.regular_result_type,
                "make_launcher",
                self._observe_regular_make_launcher,
            )
            self._patch(
                self._adapter.autotuner_type,
                "_build_fast_launcher",
                self._observe_build_fast_launcher,
            )
            self._patch(self._adapter.autotuner_type, "run", self._observe_run)
        except BaseException:
            self._restore()
            self._closed = True
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._restore()
        self._closed = True
        if exc_type is not None:
            self._problem("observed body raised")
        return False

    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        """Bind a wrapper region around its actual host dispatcher call."""
        if not self._entered or self._closed:
            raise AttestationError("region is outside the observer lifetime")
        if self._active_region is not None:
            self._problem("nested region binding is unsupported")
        if self._attesting and name not in PINNED_WRAPPER_SHA256:
            self._problem("unknown wrapper region")
        previous = self._active_region
        self._active_region = name
        try:
            yield
        finally:
            self._active_region = previous

    @contextmanager
    def attested_interval(self) -> Iterator[None]:
        """Limit target selection evidence to one explicit post-warmup interval."""
        if not self._entered or self._closed:
            raise AttestationError("attested interval is outside observer lifetime")
        if self._attesting:
            self._problem("nested attested interval is unsupported")
        previous = self._attesting
        self._attesting = True
        try:
            yield
        finally:
            self._attesting = previous

    def diagnostic(self) -> Mapping[str, object]:
        """Return a deeply immutable report only after a valid closed observation."""
        if not self._closed:
            raise AttestationError("observer has not completed")
        if self._final is not None:
            return self._final
        if self._problems:
            raise AttestationError("unverified observer: " + "; ".join(self._problems))
        if len(self._events) != len(PINNED_EVENTS):
            raise AttestationError("expected exactly five selected-launcher events")
        missing = set(self._expected) - {
            (str(event["region"]), int(event["ordinal"])) for event in self._events
        }
        if missing:
            raise AttestationError("selected-launcher event contexts are incomplete")
        report = {
            "scope": SCOPE,
            "status": "incomplete",
            "claim": "fresh-host-selected-launcher-normal-return-events",
            "does_not_attest": (
                "CUDA device execution, historical execution, or "
                "full-field/domain tensor identity-clone absence"
            ),
            "unverified": (
                "actual-wrapper-plan-cache-input-binding",
                "device-execution",
            ),
            "expected_context": {
                "case": PINNED_CASE,
                "compile_cache_key": PINNED_COMPILE_CACHE_KEY,
                "plan_identity": PINNED_PLAN_IDENTITY,
                "runtime_source_sha256": dict(PINNED_RUNTIME_SOURCES),
                "torch_version": PINNED_TORCH_VERSION,
                "triton_version": PINNED_TRITON_VERSION,
                "wrapper_sha256": dict(PINNED_WRAPPER_SHA256),
            },
            "events": self._events,
        }
        encoded = json.dumps(
            report, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        self._final = _freeze(json.loads(encoded))  # type: ignore[assignment]
        return self._final

    def diagnostic_json(self) -> bytes:
        """Return canonical immutable bytes after diagnostic validation."""
        report = self.diagnostic()
        return json.dumps(
            report, allow_nan=False, sort_keys=True, separators=(",", ":"), default=dict
        ).encode()

    def _patch(
        self, owner: type, name: str, replacement: Callable[..., object]
    ) -> None:
        original = getattr(owner, name)
        self._patches.append((owner, name, owner.__dict__.get(name, _MISSING)))

        def patched(instance: object, *args: object, **kwargs: object) -> object:
            return replacement(original, instance, *args, **kwargs)

        setattr(owner, name, patched)

    def _restore(self) -> None:
        while self._patches:
            owner, name, original = self._patches.pop()
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)

    def _observe_load(
        self,
        original: Callable[..., object],
        kernel: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        pending = None
        try:
            if getattr(kernel, "name") in {MAGNETIC_KERNEL, ELECTRIC_KERNEL}:
                if not self._making or self._making[-1][1] is not kernel:
                    raise AttestationError("static load has no matching compile result")
                compile_result, _ = self._making[-1]
                name, kernel_hash = _kernel_metadata(compile_result)
                if id(kernel) in self._loads or getattr(kernel, "function") is not None:
                    raise AttestationError(
                        "duplicate or preexisting static kernel load"
                    )
                path, digest, directory = _sha256_regular_file(Path(kernel.cubin_path))
                if (
                    not isinstance(kernel.cubin_raw, bytes)
                    or hashlib.sha256(kernel.cubin_raw).hexdigest() != digest
                ):
                    raise AttestationError(
                        "CUBIN bytes differ from compile-result kernel"
                    )
                if (
                    Path(directory).name != self._adapter.cache_key(kernel_hash)
                    or Path(path).name != f"{name}.cubin"
                ):
                    raise AttestationError(
                        "CUBIN path differs from actual kernel identity"
                    )
                pending = (compile_result, name, kernel_hash, path, digest, directory)
        except (AttributeError, TypeError, OSError, AttestationError) as error:
            self._problem(f"static load is unsupported: {error}")
        # The pinned load clears cubin_path/cubin_raw. Preserve the entry bytes,
        # then associate them with the same kernel's normally returned load.
        try:
            result = original(kernel, *args, **kwargs)
        except BaseException:
            self._problem("static load raised")
            raise
        if pending is not None:
            try:
                compile_result, name, kernel_hash, path, digest, directory = pending
                if (
                    compile_result.kernel is not kernel
                    or _kernel_metadata(compile_result) != (name, kernel_hash)
                    or kernel.function is None
                    or kernel.module is None
                    or kernel.cubin_path is not None
                    or kernel.cubin_raw is not None
                    or _sha256_regular_file(Path(path)) != (path, digest, directory)
                ):
                    raise AttestationError("static kernel changed during load")
                self._loads[id(kernel)] = _LoadedKernel(
                    compile_result,
                    kernel,
                    name,
                    kernel_hash,
                    path,
                    digest,
                    directory,
                    _loaded_state(kernel),
                )
            except (AttributeError, TypeError, OSError, AttestationError) as error:
                self._problem(f"static load is unsupported: {error}")
        return result

    def _observe_static_make_launcher(
        self,
        original: Callable[..., object],
        instance: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        kernel = getattr(instance, "kernel", None)
        self._making.append((instance, kernel))
        try:
            launcher = original(instance, *args, **kwargs)
        except BaseException:
            self._problem("static launcher construction raised")
            raise
        finally:
            self._making.pop()
        self._capture_construction(instance, launcher, static=True)
        return launcher

    def _observe_regular_make_launcher(
        self,
        original: Callable[..., object],
        instance: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        launcher = original(instance, *args, **kwargs)
        self._capture_construction(instance, launcher, static=False)
        return launcher

    def _observe_build_fast_launcher(
        self, original: Callable[..., object], instance: object, launcher: object
    ) -> object:
        derived = original(instance, launcher)
        if derived is not None:
            self._capture_fast_derivation(launcher, derived)
        return derived

    def _observe_run(
        self,
        original: Callable[..., object],
        instance: object,
        *args: object,
        stream: object,
        benchmark_run: bool = False,
        **kwargs: object,
    ) -> object:
        joined = None
        observe = self._attesting
        snapshot = self._snapshot(instance, benchmark_run, kwargs) if observe else None
        if observe and self._call_join is not None:
            frame = inspect.currentframe()
            try:
                joined = self._call_join.before_run(instance, frame.f_back.f_back)
                observe = joined is not None
                if not observe and not snapshot.slow_supported:
                    self._problem("joined-nontarget-state-unsupported")
            except AttestationError:
                self._problem("wrapper-call-join-rejected")
                observe = False
            finally:
                del frame
        if not observe:
            snapshot = None
        try:
            result = original(
                instance, *args, stream=stream, benchmark_run=benchmark_run, **kwargs
            )
        except BaseException:
            if self._attesting and self._call_join is not None:
                self._problem("joined-host-launch-raised")
            raise
        if snapshot is not None:
            count = len(self._events)
            self._capture_selected(instance, snapshot)
            if joined is not None and len(self._events) == count + 1:
                self._call_join.selected(joined, self._events[-1])
        return result

    def _capture_construction(
        self, compile_result: object, launcher: object, *, static: bool
    ) -> None:
        try:
            kernel_name, kernel_hash = _kernel_metadata(compile_result)
            if kernel_name not in {MAGNETIC_KERNEL, ELECTRIC_KERNEL}:
                return
            if not static or not bool(getattr(launcher, "_is_static", False)):
                raise AttestationError(
                    "selected target does not have a static launcher"
                )
            callable_id = id(launcher)
            if callable_id in self._constructors:
                raise AttestationError("duplicate selected launcher construction")
            cache_hash, config, globals_id, code_id, runner_id = _callable_metadata(
                launcher
            )
            loaded = self._verify_loaded(compile_result, launcher)
            path, cubin_sha256, variant_directory = (
                loaded.path,
                loaded.sha256,
                loaded.directory,
            )
            self._constructors[callable_id] = _Construction(
                callable_id=callable_id,
                kernel_name=kernel_name,
                kernel_hash=kernel_hash,
                launcher_kind="static-cubin",
                cache_hash=cache_hash,
                config=config,
                cubin_path=path,
                cubin_sha256=cubin_sha256,
                variant_directory=variant_directory,
                globals_id=globals_id,
                code_id=code_id,
                runner_id=runner_id,
            )
            self._live_callables[callable_id] = launcher
            self._live_results[callable_id] = compile_result
        except (AttributeError, TypeError, OSError, AttestationError) as error:
            self._problem(f"target construction is unsupported: {error}")

    def _capture_fast_derivation(self, parent: object, derived: object) -> None:
        try:
            parent_id = id(parent)
            derived_id = id(derived)
            record = self._constructors.get(parent_id)
            if record is None:
                return
            self._verify_parent(record)
            if parent_id == derived_id or derived_id in self._constructors:
                raise AttestationError("fast callable identity is malformed")
            if derived_id in self._fast_derivations:
                raise AttestationError("fast callable has multiple parent associations")
            cache_hash, config, globals_id, code_id, runner_id = _callable_metadata(
                derived
            )
            if cache_hash != record.cache_hash:
                raise AttestationError("fast callable cache hash differs from parent")
            if not bool(getattr(derived, "_is_static", False)):
                raise AttestationError("fast callable is not static")
            if config != record.config:
                raise AttestationError(
                    "fast callable configuration differs from parent"
                )
            self._fast_derivations[derived_id] = _FastDerivation(
                parent_id=parent_id,
                derived_id=derived_id,
                cache_hash=cache_hash,
                config=config,
                globals_id=globals_id,
                code_id=code_id,
                runner_id=runner_id,
            )
            self._live_callables[derived_id] = derived
        except (AttributeError, TypeError, OSError, AttestationError) as error:
            self._problem(f"fast callable association is unsupported: {error}")

    def _snapshot(
        self, autotuner: object, benchmark_run: bool, kwargs: Mapping[str, object]
    ) -> _RunSnapshot:
        try:
            plugins = getattr(autotuner, "_plugins")
            return _RunSnapshot(
                benchmark_run=benchmark_run,
                cached_launcher=getattr(autotuner, "_cached_launcher"),
                debug_active=bool(self._adapter.debug_active()),
                kwargs_empty=not kwargs,
                plugins_empty=not bool(plugins),
                profiler_active=bool(self._adapter.profiler_active()),
                triton_interpret=bool(getattr(autotuner, "triton_interpret")),
            )
        except (AttributeError, TypeError) as error:
            self._problem(f"pre-call state is unsupported: {error}")
            return _RunSnapshot(True, None, True, False, False, True, True)

    def _capture_selected(self, autotuner: object, snapshot: _RunSnapshot) -> None:
        try:
            post_cached = getattr(autotuner, "_cached_launcher")
            if snapshot.fast:
                selected = snapshot.cached_launcher
                if selected is None:
                    raise AttestationError("fast selection has no cached launcher")
                derivation = self._fast_derivations.get(id(selected))
                if derivation is None:
                    raise AttestationError(
                        "fast selection has no unique parent association"
                    )
                parent_id = derivation.parent_id
                record = self._constructors.get(parent_id)
                if record is None:
                    raise AttestationError("fast selection parent is not constructed")
                self._verify_fast_callable(selected, derivation)
                branch = "fast"
            else:
                if not snapshot.slow_supported:
                    raise AttestationError(
                        "slow selection has unsupported pre-call state"
                    )
                launchers = getattr(autotuner, "launchers")
                if not isinstance(launchers, Sequence) or len(launchers) != 1:
                    raise AttestationError(
                        "slow selection does not have exactly one launcher"
                    )
                selected = launchers[0]
                record = self._constructors.get(id(selected))
                if record is None:
                    raise AttestationError("slow selected launcher is not constructed")
                parent_id = None
                branch = "slow"
            self._verify_parent(record)
            region = self._active_region
            if region is None:
                raise AttestationError("selected launch has no wrapper region")
            ordinal = self._region_counts[region]
            expected = self._expected.get((region, ordinal))
            kernel_name = getattr(getattr(autotuner, "fn"), "__name__")
            if expected is None or kernel_name != expected.kernel_name:
                raise AttestationError(
                    "selected launch context differs from the fixed case"
                )
            if record.kernel_name != kernel_name:
                raise AttestationError(
                    "selected launcher kernel differs from autotuner"
                )
            self._events.append(
                {
                    "region": region,
                    "ordinal": ordinal,
                    "expected_wrapper_sha256": PINNED_WRAPPER_SHA256[region],
                    "kernel_name": kernel_name,
                    "kernel_hash": record.kernel_hash,
                    "launcher_kind": record.launcher_kind,
                    "branch": branch,
                    "pre_cached_callable_id": (
                        None
                        if snapshot.cached_launcher is None
                        else id(snapshot.cached_launcher)
                    ),
                    "post_cached_callable_id": (
                        None if post_cached is None else id(post_cached)
                    ),
                    "selected_callable_id": id(selected),
                    "parent_callable_id": parent_id,
                    "cache_hash": record.cache_hash,
                    "config": dict(record.config),
                    "cubin_path": record.cubin_path,
                    "cubin_sha256": record.cubin_sha256,
                    "variant_directory": record.variant_directory,
                    "host_returned_normally": True,
                }
            )
            self._region_counts[region] += 1
        except (AttributeError, TypeError, OSError, AttestationError) as error:
            self._problem(f"selected launch is unsupported: {error}")

    def _verify_loaded(self, compile_result: object, launcher: object) -> _LoadedKernel:
        kernel = getattr(compile_result, "kernel")
        loaded = self._loads.get(id(kernel))
        if (
            loaded is None
            or loaded.kernel is not kernel
            or loaded.compile_result is not compile_result
        ):
            raise AttestationError(
                "compile result has no matching observed static load"
            )
        runner = launcher.__globals__["runner"]
        if (
            getattr(runner, "__self__", None) is not kernel
            or getattr(runner, "__func__", None)
            is not self._adapter.static_kernel_type.run
            or _kernel_metadata(compile_result) != (loaded.name, loaded.kernel_hash)
            or _loaded_state(kernel) != loaded.state
            or kernel.cubin_path is not None
            or kernel.cubin_raw is not None
            or launcher.cache_hash != self._adapter.cache_key(loaded.kernel_hash)
            or _canonical_config(compile_result.config)
            != _canonical_config(launcher.config)
        ):
            raise AttestationError("loaded kernel or launcher metadata changed")
        if _sha256_regular_file(Path(loaded.path)) != (
            loaded.path,
            loaded.sha256,
            loaded.directory,
        ):
            raise AttestationError("selected launcher CUBIN changed after load")
        return loaded

    def _verify_parent(self, record: _Construction) -> None:
        launcher = self._live_callables.get(record.callable_id)
        compile_result = self._live_results.get(record.callable_id)
        if launcher is None or compile_result is None:
            raise AttestationError("selected parent is no longer live")
        kernel_name, kernel_hash = _kernel_metadata(compile_result)
        if kernel_name != record.kernel_name or kernel_hash != record.kernel_hash:
            raise AttestationError("selected parent kernel metadata changed")
        if not bool(getattr(launcher, "_is_static", False)):
            raise AttestationError("selected parent launcher kind changed")
        cache_hash, config, globals_id, code_id, runner_id = _callable_metadata(
            launcher
        )
        if (
            cache_hash != record.cache_hash
            or config != record.config
            or globals_id != record.globals_id
            or code_id != record.code_id
            or runner_id != record.runner_id
        ):
            raise AttestationError("selected parent callable metadata changed")
        loaded = self._verify_loaded(compile_result, launcher)
        path, digest, directory = loaded.path, loaded.sha256, loaded.directory
        if (
            path != record.cubin_path
            or digest != record.cubin_sha256
            or directory != record.variant_directory
        ):
            raise AttestationError("selected launcher CUBIN changed after construction")

    def _verify_fast_callable(
        self, launcher: object, derivation: _FastDerivation
    ) -> None:
        if self._live_callables.get(derivation.derived_id) is not launcher:
            raise AttestationError("selected fast callable identity changed")
        cache_hash, config, globals_id, code_id, runner_id = _callable_metadata(
            launcher
        )
        if (
            cache_hash != derivation.cache_hash
            or config != derivation.config
            or globals_id != derivation.globals_id
            or code_id != derivation.code_id
            or runner_id != derivation.runner_id
        ):
            raise AttestationError("selected fast callable metadata changed")

    def _problem(self, reason: str) -> None:
        if reason not in self._problems:
            self._problems.append(reason)


def pinned_observer(adapter: _RuntimeAdapter | None = None) -> SelectedLauncherObserver:
    """Create the fixed-scope observer before generated-wrapper execution."""
    return SelectedLauncherObserver(adapter=adapter)
