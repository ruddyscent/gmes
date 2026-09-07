"""CPU-only controls for pinned selected-launcher host evidence."""

from __future__ import annotations

import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

from benchmarks import torch_selected_launcher_attestation as attestation


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


class _StaticResult:
    def __init__(self, launcher, cubin_path: Path, kernel_name: str):
        self._launcher = launcher
        self.kernel = types.SimpleNamespace(
            cubin_path=str(cubin_path),
            hash=f"hash-{kernel_name}",
            name=kernel_name,
        )

    def make_launcher(self):
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
        )


class SelectedLauncherObserverTest(unittest.TestCase):
    def _result(self, directory: Path, name: str, *, value: int = 1):
        cubin = directory / f"{name}.cubin"
        cubin.write_bytes(name.encode())
        return _StaticResult(
            _launcher(name, _Config(value), f"cache-{name}-{value}"), cubin, name
        )

    def _prepare(self, observer, result, kernel_name, *, fast: bool):
        parent = result.make_launcher()
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
        self.assertEqual(len(report["events"]), 5)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("actual-wrapper-plan-cache-input-binding", report["unverified"])
        magnetic = report["events"][0]
        self.assertEqual(magnetic["branch"], "slow")
        self.assertEqual(magnetic["selected_callable_id"], id(parent))
        self.assertNotEqual(
            magnetic["post_cached_callable_id"], magnetic["selected_callable_id"]
        )
        electric = report["events"][1]
        self.assertEqual(electric["branch"], "fast")
        self.assertNotEqual(
            electric["parent_callable_id"], electric["selected_callable_id"]
        )
        with self.assertRaises(TypeError):
            report["events"] = ()  # type: ignore[index]
        self.assertIn(b'"events"', observer.diagnostic_json())

    def test_restores_patched_methods_when_observed_body_raises(self):
        runtime = _FakeRuntime()
        original_static = _StaticResult.make_launcher
        original_regular = _RegularResult.make_launcher
        original_fast = _Autotuner._build_fast_launcher
        original_run = _Autotuner.run
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with attestation.pinned_observer(runtime.adapter()):
                raise RuntimeError("stop")
        self.assertIs(_StaticResult.make_launcher, original_static)
        self.assertIs(_RegularResult.make_launcher, original_regular)
        self.assertIs(_Autotuner._build_fast_launcher, original_fast)
        self.assertIs(_Autotuner.run, original_run)

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
            with self.assertRaisesRegex(
                attestation.AttestationError, "parent association"
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
            with self.assertRaises(attestation.AttestationError):
                observer.diagnostic()

    def test_missing_mutated_and_symlink_cubins_fail_closed(self):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            for mode in ("missing", "mutated", "symlink"):
                with self.subTest(mode=mode):
                    with attestation.pinned_observer(runtime.adapter()) as observer:
                        cubin = directory / f"{mode}.cubin"
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
                        with observer.attested_interval():
                            with observer.region("magnetic_half"):
                                tuner.run(stream="stream")
                    with self.assertRaises(attestation.AttestationError):
                        observer.diagnostic()

    def test_preexisting_cache_is_rejected_before_patching(self):
        runtime = _FakeRuntime()
        runtime.cache_is_empty = False
        original = _Autotuner.run
        with self.assertRaisesRegex(attestation.AttestationError, "preexisting"):
            with attestation.pinned_observer(runtime.adapter()):
                pass
        self.assertIs(_Autotuner.run, original)

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
            self.assertEqual(len(observer.diagnostic()["events"]), 5)

        with tempfile.TemporaryDirectory() as raw:
            with attestation.pinned_observer(runtime.adapter()) as observer:
                unknown = self._result(Path(raw), "unrelated_kernel")
                tuner = _Autotuner("unrelated_kernel", unknown.make_launcher())
                with observer.attested_interval():
                    with observer.region("magnetic_half"):
                        tuner.run(stream="stream")
            with self.assertRaises(attestation.AttestationError):
                observer.diagnostic()

    def test_revalidates_live_fast_and_parent_metadata(self):
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as raw:
            for mode in ("fast-config", "fast-cache", "fast-runner", "parent-runner"):
                with self.subTest(mode=mode):
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
                    with self.assertRaisesRegex(
                        attestation.AttestationError, "metadata changed"
                    ):
                        observer.diagnostic()


class PinnedRuntimeSourceGateTest(unittest.TestCase):
    def test_pinned_cuda_runtime_source_gate_without_cuda_execution(self):
        interpreter = Path("/tmp/gmes-issue-124-cuda-cu130-CUqJ9b/bin/python")
        self.assertTrue(interpreter.is_file())
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
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "entry-restore-ok")
