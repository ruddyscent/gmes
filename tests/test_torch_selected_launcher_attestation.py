"""CPU-only controls for pinned selected-launcher host evidence."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest

from benchmarks import torch_selected_launcher_attestation as attestation

PINNED_CUDA_PYTHON_ENV = "GMES_PINNED_CUDA_PYTHON"


class _Config:
    def __init__(self, value: int = 1):
        self.kwargs = {"BLOCK": value}
        self.num_stages = 1
        self.num_warps = 4


def _launcher(name: str, config: _Config, cache_hash: str):
    def runner(*args, stream):
        return ("parent", args, stream)

    scope = {"runner": runner}
    exec("def launcher(*args, stream):\n    return runner(*args, stream=stream)", scope)
    launch = scope["launcher"]
    launch.__name__ = "launcher"
    launch.cache_hash = cache_hash
    launch.config = config
    launch._is_static = True
    return launch


class _StaticKernel:
    def __init__(self, cubin_path, kernel_name, kernel_hash):
        self.cubin_path = str(cubin_path)
        self.cubin_raw = cubin_path.read_bytes()
        self.name = kernel_name
        self.hash = kernel_hash
        self.function = None
        self.module = None
        self.num_warps = 4
        self.shared = 0
        self.arg_tys = "O"
        self.has_global_scratch = False
        self.has_profile_scratch = False

    def load_kernel(self, device):
        if self.function is not None:
            return
        self.module, self.function = 101, 102
        self.cubin_path = None
        self.cubin_raw = None

    def run(self, *args, stream):
        return ("parent", args, stream)


class _StaticResult:
    def __init__(self, launcher, cubin_path: Path, kernel_name: str):
        self._launcher = launcher
        self.config = launcher.config
        self.kernel = _StaticKernel(cubin_path, kernel_name, launcher.cache_hash)

    def make_launcher(self):
        self.kernel.load_kernel(0)
        self._launcher.__globals__["runner"] = self.kernel.run
        return self._launcher


class _RegularResult:
    def make_launcher(self):
        raise AssertionError("regular result is not part of the positive fixture")


class _Autotuner:
    def __init__(self, kernel_name: str, launcher):
        self.fn = types.SimpleNamespace(__name__=kernel_name)
        self.launchers = [launcher]
        self._cached_launcher = None
        self._plugins = []
        self.triton_interpret = False

    def _build_fast_launcher(self, launcher):
        derived = types.FunctionType(
            launcher.__code__,
            dict(launcher.__globals__),
            launcher.__name__,
            launcher.__defaults__,
            launcher.__closure__,
        )
        derived.cache_hash = launcher.cache_hash
        derived.config = launcher.config
        derived._is_static = launcher._is_static
        return derived

    def run(self, *args, stream, benchmark_run=False, **kwargs):
        fast = self._cached_launcher
        if fast is not None and not benchmark_run and not kwargs:
            return fast(*args, stream=stream)
        launcher = self.launchers[0]
        result = launcher(*args, stream=stream)
        if self._cached_launcher is None and not benchmark_run:
            self._cached_launcher = self._build_fast_launcher(launcher) or launcher
        return result


class _FakeRuntime:
    def __init__(self):
        self.profiler = False
        self.debug = False
        self.cache_is_empty = True

    def adapter(self):
        return attestation._RuntimeAdapter(
            static_result_type=_StaticResult,
            regular_result_type=_RegularResult,
            autotuner_type=_Autotuner,
            profiler_active=lambda: self.profiler,
            debug_active=lambda: self.debug,
            verify=lambda: None,
            cache_empty=lambda: self.cache_is_empty,
            static_kernel_type=_StaticKernel,
            cache_key=lambda kernel_hash: kernel_hash,
        )


class TestSelectedLauncherObserver:
    def _result(self, directory: Path, name: str, *, value: int = 1):
        directory = directory.resolve(strict=True) / f"cache-{name}-{value}"
        directory.mkdir(exist_ok=True)
        cubin = directory / f"{name}.cubin"
        cubin.write_bytes(name.encode())
        return _StaticResult(
            _launcher(name, _Config(value), f"cache-{name}-{value}"), cubin, name
        )

    def _prepare(self, observer, result, kernel_name, *, fast: bool):
        parent = result.make_launcher()
        assert (result.kernel.cubin_path) is None
        assert (result.kernel.cubin_raw) is None
        tuner = _Autotuner(kernel_name, parent)
        if fast:
            tuner._cached_launcher = tuner._build_fast_launcher(parent)
        return tuner, parent

    def _five_events(self, observer, directory: Path):
        magnetic, parent = self._prepare(
            observer,
            self._result(directory, attestation.MAGNETIC_KERNEL),
            attestation.MAGNETIC_KERNEL,
            fast=False,
        )
        with observer.region("magnetic_half"):
            magnetic.run(stream="stream")
        for _ in range(4):
            electric, _ = self._prepare(
                observer,
                self._result(directory, attestation.ELECTRIC_KERNEL),
                attestation.ELECTRIC_KERNEL,
                fast=True,
            )
            with observer.region("electric_half"):
                electric.run(stream="stream")
        return parent

    def test_slow_and_fast_paths_bind_actual_parent_and_derived_callables(self):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as raw:
            with attestation.pinned_observer(runtime.adapter()) as observer:
                with observer.attested_interval():
                    parent = self._five_events(observer, Path(raw))
            report = observer.diagnostic()
        assert (len(report["events"])) == (5)
        assert (report["status"]) == ("incomplete")
        assert ("actual-wrapper-plan-cache-input-binding") in (report["unverified"])
        magnetic = report["events"][0]
        assert (magnetic["branch"]) == ("slow")
        assert (magnetic["selected_callable_id"]) == (id(parent))
        assert (magnetic["post_cached_callable_id"]) != (
            magnetic["selected_callable_id"]
        )
        electric = report["events"][1]
        assert (electric["branch"]) == ("fast")
        assert (electric["parent_callable_id"]) != (electric["selected_callable_id"])
        with pytest.raises(TypeError):
            report["events"] = ()  # type: ignore[index]
        assert (b'"events"') in (observer.diagnostic_json())

    def test_alias_root_is_canonicalized_for_positive_fixture_only(self):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            canonical = root / "canonical"
            canonical.mkdir()
            alias = root / "alias"
            alias.symlink_to(canonical, target_is_directory=True)
            with attestation.pinned_observer(runtime.adapter()) as observer:
                with observer.attested_interval():
                    self._five_events(observer, alias)
            assert (len(observer.diagnostic()["events"])) == (5)

            variant = canonical / "cache-parent-alias"
            variant.mkdir()
            cubin = alias / variant.name / f"{attestation.MAGNETIC_KERNEL}.cubin"
            cubin.write_bytes(b"parent alias")
            result = _StaticResult(
                _launcher(
                    attestation.MAGNETIC_KERNEL,
                    _Config(),
                    variant.name,
                ),
                cubin,
                attestation.MAGNETIC_KERNEL,
            )
            with attestation.pinned_observer(runtime.adapter()) as observer:
                result.make_launcher()
            with pytest.raises(
                attestation.AttestationError, match="CUBIN path is not canonical"
            ):
                observer.diagnostic()

    def test_restores_patched_methods_when_observed_body_raises(self):
        runtime = _FakeRuntime()
        original_static = _StaticResult.make_launcher
        original_regular = _RegularResult.make_launcher
        original_fast = _Autotuner._build_fast_launcher
        original_run = _Autotuner.run
        original_load = _StaticKernel.load_kernel
        with pytest.raises(RuntimeError, match="stop"):
            with attestation.pinned_observer(runtime.adapter()):
                raise RuntimeError("stop")
        assert (_StaticResult.make_launcher) is (original_static)
        assert (_RegularResult.make_launcher) is (original_regular)
        assert (_Autotuner._build_fast_launcher) is (original_fast)
        assert (_Autotuner.run) is (original_run)
        assert (_StaticKernel.load_kernel) is (original_load)

    @pytest.mark.parametrize(
        "mode_case",
        range(7),
        ids=(
            "kernel",
            "foreign-runner",
            "handle",
            "path",
            "artifact",
            "missing",
            "config",
        ),
    )
    def test_loaded_identity_and_artifact_mutations_reject_after_five_calls(
        self, mode_case
    ):
        mode_case_values = tuple(
            (
                "kernel",
                "foreign-runner",
                "handle",
                "path",
                "artifact",
                "missing",
                "config",
            )
        )
        assert len(mode_case_values) == 7
        mode = mode_case_values[mode_case]
        with tempfile.TemporaryDirectory() as raw:
            with attestation.pinned_observer(_FakeRuntime().adapter()) as observer:
                self._five_events_outside_interval(observer, Path(raw))
                result = observer._live_results[next(iter(observer._constructors))]
                parent = observer._live_callables[next(iter(observer._constructors))]
                loaded = observer._loads[id(result.kernel)]
                if mode == "kernel":
                    replacement = _StaticKernel(
                        Path(loaded.path), loaded.name, loaded.kernel_hash
                    )
                    replacement.__dict__.update(result.kernel.__dict__)
                    result.kernel = replacement
                elif mode == "foreign-runner":
                    foreign = _StaticKernel(
                        Path(loaded.path), loaded.name, loaded.kernel_hash
                    )
                    parent.__globals__["runner"] = foreign.run
                elif mode == "handle":
                    result.kernel.function += 1
                elif mode == "path":
                    result.kernel.cubin_path = loaded.path
                elif mode == "artifact":
                    Path(loaded.path).write_bytes(b"changed")
                elif mode == "missing":
                    Path(loaded.path).unlink()
                else:
                    result.config = _Config(999)
                with observer.attested_interval():
                    with observer.region("magnetic_half"):
                        _Autotuner(attestation.MAGNETIC_KERNEL, parent).run(
                            stream="stream"
                        )
                    for tuner in self._electric_tuners:
                        with observer.region("electric_half"):
                            tuner.run(stream="stream")
            with pytest.raises(attestation.AttestationError):
                observer.diagnostic()

    def _five_events_outside_interval(self, observer, directory):
        self._prepare(
            observer,
            self._result(directory, attestation.MAGNETIC_KERNEL),
            attestation.MAGNETIC_KERNEL,
            fast=False,
        )
        self._electric_tuners = [
            self._prepare(
                observer,
                self._result(directory, attestation.ELECTRIC_KERNEL),
                attestation.ELECTRIC_KERNEL,
                fast=True,
            )[0]
            for _ in range(4)
        ]

    @pytest.mark.parametrize(
        "mode_case",
        range(4),
        ids=("foreign", "foreign-bytes", "missing-raw", "preloaded"),
    )
    def test_foreign_artifact_and_preexisting_load_are_rejected(self, mode_case):
        mode_case_values = tuple(
            ("foreign", "foreign-bytes", "missing-raw", "preloaded")
        )
        assert len(mode_case_values) == 4
        mode = mode_case_values[mode_case]
        with tempfile.TemporaryDirectory() as raw:
            result = self._result(Path(raw), attestation.MAGNETIC_KERNEL)
            if mode == "preloaded":
                result.kernel.load_kernel(0)
            elif mode == "foreign-bytes":
                Path(result.kernel.cubin_path).write_bytes(b"foreign binary")
            elif mode == "missing-raw":
                result.kernel.cubin_raw = None
            else:
                foreign = Path(raw) / "foreign.cubin"
                foreign.write_bytes(b"foreign")
                result.kernel.cubin_path = str(foreign)
            with attestation.pinned_observer(_FakeRuntime().adapter()) as observer:
                result.make_launcher()
            assert observer._problems
            with pytest.raises(attestation.AttestationError):
                observer.diagnostic()

    def test_pre_call_fast_path_rejects_unknown_parent_association(self):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as raw:
            with attestation.pinned_observer(runtime.adapter()) as observer:
                result = self._result(Path(raw), attestation.MAGNETIC_KERNEL)
                parent = result.make_launcher()
                tuner = _Autotuner(attestation.MAGNETIC_KERNEL, parent)
                unknown = _launcher(
                    attestation.MAGNETIC_KERNEL, parent.config, parent.cache_hash
                )
                tuner._cached_launcher = unknown
                with observer.attested_interval():
                    with observer.region("magnetic_half"):
                        tuner.run(stream="stream")
            with pytest.raises(
                attestation.AttestationError, match="parent association"
            ):
                observer.diagnostic()

    def test_plugin_and_event_count_controls_fail_closed(self):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as raw:
            with attestation.pinned_observer(runtime.adapter()) as observer:
                result = self._result(Path(raw), attestation.MAGNETIC_KERNEL)
                parent = result.make_launcher()
                tuner = _Autotuner(attestation.MAGNETIC_KERNEL, parent)
                tuner._plugins = [object()]
                with observer.attested_interval():
                    with observer.region("magnetic_half"):
                        tuner.run(stream="stream")
            with pytest.raises(attestation.AttestationError):
                observer.diagnostic()

    @pytest.mark.parametrize(
        "mode_case", range(3), ids=("missing", "mutated", "symlink")
    )
    def test_missing_mutated_and_symlink_cubins_fail_closed(self, mode_case):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            mode_case_values = tuple(("missing", "mutated", "symlink"))
            assert len(mode_case_values) == 3
            mode = mode_case_values[mode_case]
            with attestation.pinned_observer(runtime.adapter()) as observer:
                variant = directory / f"cache-{mode}"
                variant.mkdir()
                cubin = variant / f"{attestation.MAGNETIC_KERNEL}.cubin"
                cubin.write_bytes(b"before")
                if mode == "symlink":
                    linked = directory / "linked.cubin"
                    linked.symlink_to(cubin)
                    cubin = linked
                result = _StaticResult(
                    _launcher(
                        attestation.MAGNETIC_KERNEL,
                        _Config(),
                        f"cache-{mode}",
                    ),
                    cubin,
                    attestation.MAGNETIC_KERNEL,
                )
                parent = result.make_launcher()
                tuner = _Autotuner(attestation.MAGNETIC_KERNEL, parent)
                if mode == "mutated":
                    cubin.write_bytes(b"after")
                elif mode == "missing":
                    cubin.unlink()
                with observer.attested_interval():
                    with observer.region("magnetic_half"):
                        tuner.run(stream="stream")
            with pytest.raises(attestation.AttestationError):
                observer.diagnostic()

    def test_preexisting_cache_is_rejected_before_patching(self):
        runtime = _FakeRuntime()
        runtime.cache_is_empty = False
        original = _Autotuner.run
        with pytest.raises(attestation.AttestationError, match="preexisting"):
            with attestation.pinned_observer(runtime.adapter()):
                pass
        assert (_Autotuner.run) is (original)

    def test_interval_ignores_warmup_and_rejects_unknown_target_scope(self):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with attestation.pinned_observer(runtime.adapter()) as observer:
                warmup = self._result(directory, "unrelated_kernel")
                parent = warmup.make_launcher()
                _Autotuner("unrelated_kernel", parent).run(stream="stream")
                with observer.attested_interval():
                    self._five_events(observer, directory)
            assert (len(observer.diagnostic()["events"])) == (5)

        with tempfile.TemporaryDirectory() as raw:
            with attestation.pinned_observer(runtime.adapter()) as observer:
                unknown = self._result(Path(raw), "unrelated_kernel")
                tuner = _Autotuner("unrelated_kernel", unknown.make_launcher())
                with observer.attested_interval():
                    with observer.region("magnetic_half"):
                        tuner.run(stream="stream")
            with pytest.raises(attestation.AttestationError):
                observer.diagnostic()

    @pytest.mark.parametrize(
        "mode_case",
        range(4),
        ids=("fast-config", "fast-cache", "fast-runner", "parent-runner"),
    )
    def test_revalidates_live_fast_and_parent_metadata(self, mode_case):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as raw:
            mode_case_values = tuple(
                ("fast-config", "fast-cache", "fast-runner", "parent-runner")
            )
            assert len(mode_case_values) == 4
            mode = mode_case_values[mode_case]
            with attestation.pinned_observer(runtime.adapter()) as observer:
                result = self._result(Path(raw), attestation.MAGNETIC_KERNEL)
                parent = result.make_launcher()
                tuner = _Autotuner(attestation.MAGNETIC_KERNEL, parent)
                derived = tuner._build_fast_launcher(parent)
                tuner._cached_launcher = derived
                if mode == "fast-config":
                    derived.config.kwargs["BLOCK"] = 999
                elif mode == "fast-cache":
                    derived.cache_hash = "mutated-cache"
                elif mode == "fast-runner":
                    derived.__globals__["runner"] = lambda *args, stream: None
                else:
                    parent.__globals__["runner"] = lambda *args, stream: None
                with observer.attested_interval():
                    with observer.region("magnetic_half"):
                        tuner.run(stream="stream")
            with pytest.raises(attestation.AttestationError, match="metadata changed"):
                observer.diagnostic()


class TestPinnedRuntimeSourceGate:
    @staticmethod
    def _pinned_runtime_interpreter() -> Path:
        configured = os.environ.get(PINNED_CUDA_PYTHON_ENV)
        if not configured:
            raise pytest.skip.Exception(
                f"{PINNED_CUDA_PYTHON_ENV} is required for pinned runtime integration"
            )
        interpreter = Path(configured)
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise pytest.skip.Exception(
                f"configured pinned runtime interpreter is unavailable: {configured}"
            )
        return interpreter

    def test_pinned_runtime_interpreter_requires_configured_regular_executable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(pytest.skip.Exception, match=PINNED_CUDA_PYTHON_ENV):
                self._pinned_runtime_interpreter()
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing-python"
            with mock.patch.dict(
                os.environ, {PINNED_CUDA_PYTHON_ENV: str(missing)}, clear=True
            ):
                with pytest.raises(pytest.skip.Exception, match="unavailable"):
                    self._pinned_runtime_interpreter()
        with mock.patch.dict(
            os.environ, {PINNED_CUDA_PYTHON_ENV: sys.executable}, clear=True
        ):
            assert (self._pinned_runtime_interpreter()) == (Path(sys.executable))

    def test_real_python_load_lifecycle_with_cpu_native_loader_stub(self):
        self._run_pinned_script(
            """
import tempfile
import types
from pathlib import Path
from benchmarks import torch_selected_launcher_attestation as a
from tests.test_torch_selected_launcher_attestation import _Config, _launcher
from torch._inductor.runtime import static_triton_launcher as s, triton_heuristics as h

owner = s.StaticallyLaunchedCudaKernel
original = owner.load_kernel
own_descriptor = owner.__dict__.get("load_kernel")
for fail in (False, True):
    with tempfile.TemporaryDirectory() as raw:
        kernel = object.__new__(owner)
        kernel.name = a.MAGNETIC_KERNEL
        kernel.hash = "a" * 64
        key = h.triton_hash_to_path_key(kernel.hash)
        directory = Path(raw) / key
        directory.mkdir()
        path = directory / (kernel.name + ".cubin")
        path.write_bytes(b"CPU fixture, not a device binary")
        kernel.cubin_path = str(path)
        kernel.cubin_raw = path.read_bytes()
        kernel.function = kernel.module = None
        kernel.num_warps = 4
        kernel.shared = 0
        kernel.arg_tys = "O"
        kernel.has_global_scratch = kernel.has_profile_scratch = False
        calls = []
        def native_stub(*args):
            calls.append(args)
            assert args == (str(path), kernel.name, 0, 0)
            if fail:
                raise RuntimeError("native-loader-stub")
            return (101, 102, 3, 0)
        kernel.C_impl = types.SimpleNamespace(_load_kernel=native_stub)
        result = types.SimpleNamespace(kernel=kernel, config=_Config())
        def make(result):
            result.kernel.load_kernel(0)
            launch = _launcher(kernel.name, result.config, key)
            launch.__globals__["runner"] = result.kernel.run
            return launch
        observer = a.pinned_observer()
        try:
            with observer:
                parent = observer._observe_static_make_launcher(make, result)
                assert kernel.cubin_path is None and kernel.cubin_raw is None
                record = observer._constructors[id(parent)]
                observer._verify_parent(record)
                assert record.cubin_path == str(path)
                assert not observer._problems, observer._problems
        except RuntimeError as error:
            assert fail and str(error) == "native-loader-stub"
        else:
            assert not fail
        assert len(calls) == 1
        assert owner.load_kernel is original
        assert owner.__dict__.get("load_kernel") is own_descriptor
        assert not observer._making
print("real-python-load-stub-restore-ok")
""",
            "real-python-load-stub-restore-ok",
        )

    def _run_pinned_script(self, script, expected):
        root = Path(__file__).resolve().parents[1]
        interpreter = self._pinned_runtime_interpreter()
        process = subprocess.run(
            [str(interpreter), "-c", script],
            cwd=root,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(root),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert (process.returncode) == (0)
        assert (process.stdout.strip()) == (expected)

    def test_pinned_cuda_runtime_source_gate_without_cuda_execution(self):
        interpreter = self._pinned_runtime_interpreter()
        root = Path(__file__).resolve().parents[1]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(root),
        }
        process = subprocess.run(
            [
                str(interpreter),
                "-c",
                (
                    "from benchmarks.torch_selected_launcher_attestation "
                    "import pinned_observer; "
                    "from torch._inductor.runtime import triton_heuristics as h; "
                    "before=(h.StaticTritonCompileResult.make_launcher,"
                    "h.TritonCompileResult.make_launcher,"
                    "h.CachingAutotuner._build_fast_launcher,h.CachingAutotuner.run); "
                    "observer=pinned_observer(); observer.__enter__(); "
                    "observer.__exit__(None,None,None); "
                    "assert before==(h.StaticTritonCompileResult.make_launcher,"
                    "h.TritonCompileResult.make_launcher,"
                    "h.CachingAutotuner._build_fast_launcher,h.CachingAutotuner.run); "
                    "print('entry-restore-ok')"
                ),
            ],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert (process.returncode) == (0)
        assert (process.stdout.strip()) == ("entry-restore-ok")
