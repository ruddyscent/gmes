"""CPU-only structural coverage for pinned CUDA wrapper evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import torch_lowered_materialization as lowering


class _IdentityAsyncCompile:
    """Resolve fixture globals without compiling or launching a device kernel."""

    def triton(self, name, source):
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        namespace = {}
        exec(compile(source, "<CPU kernel fixture>", "exec"), namespace)
        tuner = object.__new__(CachingAutotuner)
        tuner.fn = namespace[name]
        return tuner


class _JoinAsyncCompile(_IdentityAsyncCompile):
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)

    def triton(self, name, source):
        from tests.test_torch_selected_launcher_attestation import (
            _Config,
            _launcher,
            _StaticResult,
        )

        tuner = super().triton(name, source)
        directory = self.directory / ("cache-" + name)
        directory.mkdir(exist_ok=True)
        path = directory / (name + ".cubin")
        path.write_bytes(b"CPU fixture, not a device binary")
        result = _StaticResult(_launcher(name, _Config(), directory.name), path, name)
        tuner.launchers = [result.make_launcher()]
        tuner._cached_launcher = None
        tuner._plugins = []
        tuner.triton_interpret = False
        return tuner


class WrapperCallJoinTest(unittest.TestCase):
    def _exercise(self, mode="normal"):
        from torch._inductor.runtime.compile_tasks import _reload_python_module
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        import gmes
        from benchmarks import torch_selected_launcher_attestation as a
        from tests.test_torch_selected_launcher_attestation import (
            _Autotuner,
            _FakeRuntime,
        )

        with (
            tempfile.TemporaryDirectory() as raw,
            mock.patch.object(CachingAutotuner, "run", _Autotuner.run),
            mock.patch.object(
                CachingAutotuner,
                "_build_fast_launcher",
                _Autotuner._build_fast_launcher,
            ),
        ):
            directory = Path(raw).resolve(strict=True)
            adapter = dataclasses.replace(
                _FakeRuntime().adapter(), autotuner_type=CachingAutotuner
            )
            observer = a.pinned_observer(adapter)
            evidence = lowering._ReturnedModuleEvidence()
            simulation = types.SimpleNamespace(_cuda_graphs={})
            dispatcher = types.MethodType(
                gmes.TorchSimulation._run_compute_region, simulation
            )
            join = lowering._WrapperCallJoin(evidence, observer, dispatcher)
            modules = {}
            with observer:
                extra = (
                    _JoinAsyncCompile(directory).triton(
                        a.MAGNETIC_KERNEL, f"def {a.MAGNETIC_KERNEL}(): pass"
                    )
                    if mode == "unowned-live"
                    else None
                )
                for region, target, count in (
                    ("electric_half", a.ELECTRIC_KERNEL, 4),
                    ("magnetic_half", a.MAGNETIC_KERNEL, 1),
                ):
                    source = f"""from tests.test_torch_lowered_materialization import _JoinAsyncCompile
async_compile = _JoinAsyncCompile({raw!r})
{target} = async_compile.triton({target!r}, 'def {target}(): pass')
unrelated = async_compile.triton('unrelated', 'def unrelated(): pass')
class Runner:
    def __init__(self, partitions):
        self.partitions = partitions
    def call(self, args):
        if args and args[0] == 'silent':
            return args
        {"args[0].run(stream='cpu')" if mode == "unowned-live" else "pass"}
        unrelated.run(stream='cpu')
        for _ in range({count}):
            {target}.run(stream='cpu')
        return args
runner = Runner(partitions=[])
call = runner.call
"""
                    path = directory / (region + ".py")
                    path.write_text(source)
                    module = _reload_python_module(
                        region, str(path), set_sys_modules=False
                    )
                    modules[region] = module

                    def function(module=module, target=target):
                        if mode == "nontarget-target":
                            module.unrelated._cached_launcher = getattr(
                                module, target
                            ).launchers[0]
                        if mode == "no-runner":
                            return getattr(module, target).run(stream="cpu")
                        result = module.call([extra])
                        if mode == "silent-second-invocation":
                            module.call(["silent"])
                        return result

                    setattr(simulation, "_" + region, function)
                    evidence.capture(
                        types.SimpleNamespace(cache_path=str(path), cache_key=region),
                        module,
                        [source],
                        (simulation, region, function),
                    )
                    with join.region(region, function):
                        dispatcher(region, function)
                if mode == "unknown-tuner":
                    modules["magnetic_half"].unrelated = _JoinAsyncCompile(
                        directory
                    ).triton("unrelated", "def unrelated(): pass")
                elif mode == "receiver":
                    module = modules["magnetic_half"]
                    module.call = module.Runner([]).call
                elif mode == "cross-region":
                    modules["magnetic_half"].unrelated = getattr(
                        modules["electric_half"], a.ELECTRIC_KERNEL
                    )
                elif mode == "exception":

                    def fail(*args, **kwargs):
                        raise RuntimeError("CPU fixture launch exception")

                    modules["magnetic_half"].unrelated._cached_launcher = fail
                try:
                    with join.advance():
                        for region in ("magnetic_half", "electric_half"):
                            function = getattr(simulation, "_" + region)
                            with join.region(region, function):
                                if mode == "missing-dispatch":
                                    function()
                                elif mode == "twice":
                                    dispatcher(region, function)
                                    dispatcher(region, function)
                                else:
                                    dispatcher(region, function)
                except a.AttestationError, RuntimeError:
                    self.assertNotEqual(mode, "normal")
            self.assertIsNone(observer._call_join)
            self.assertIsNone(join.active)
            report = join.diagnostic()
            self.assertEqual(report["status"], "incomplete")
            if mode == "normal":
                self.assertEqual(len(report["events"]), 5)
                self.assertEqual(
                    [item["region"] for item in report["events"]],
                    ["magnetic_half"] + ["electric_half"] * 4,
                )
                self.assertEqual(set(join.completed), set(lowering.COMPILED_REGIONS))
                self.assertTrue(
                    all(item["host_returned_normally"] for item in report["events"])
                )
                encoded = json.dumps(report)
                self.assertNotIn(raw, encoded)
                self.assertNotIn('_id"', encoded)
                self.assertIn(
                    "bound-method-alias-invocation-provenance", report["unverified"]
                )
                self.assertIn(
                    "silent-or-additional-runner-invocations", report["unverified"]
                )
                self.assertIn(
                    "total-runner-entry-exit-cardinality", report["unverified"]
                )
            elif mode != "silent-second-invocation":
                self.assertEqual(report["events"], [])
                self.assertTrue(report["reasons"])
            else:
                self.assertEqual(len(report["events"]), 5)
                self.assertIn(
                    "silent-or-additional-runner-invocations", report["unverified"]
                )
                self.assertIn(
                    "total-runner-entry-exit-cardinality", report["unverified"]
                )

    def test_silent_additional_runner_invocation_does_not_claim_cardinality(self):
        self._exercise("silent-second-invocation")

    def test_real_loader_dispatcher_frames_join_cpu_stub_launches(self):
        self._exercise()

    def test_unknown_cross_region_substitution_missing_duplicate_and_exception_reject(
        self,
    ):
        for mode in (
            "unknown-tuner",
            "receiver",
            "cross-region",
            "missing-dispatch",
            "twice",
            "exception",
            "unowned-live",
            "nontarget-target",
            "no-runner",
        ):
            with self.subTest(mode=mode):
                self._exercise(mode)

    def test_default_output_capture_does_not_install_module_or_join_hooks(self):
        from torch._inductor.graph import GraphLowering

        original = GraphLowering._compile_to_module_lines
        sources = []
        with lowering._capturing_output_code(sources.append):
            self.assertIs(GraphLowering._compile_to_module_lines, original)
            GraphLowering.save_output_code("source")
        self.assertEqual(sources, ["source"])


class CaseProducerReceiptTest(unittest.TestCase):
    def _construct_all_material(self, *, descriptor_name=None, courant_ratio=None):
        import gmes
        import gmes.torch_fdtd as torch_fdtd
        from benchmarks.torch_tuning import MANIFEST, _build_case, load_manifest

        spec, space, geometry, sources, bloch = _build_case(
            "all-material-2d", load_manifest(MANIFEST)
        )
        if descriptor_name is not None:
            spec = {**spec, "name": descriptor_name}
        sources = lowering._memory_sources(spec, sources)
        runtime = gmes.TorchRuntimeConfig(
            device="cpu",
            precision="float64",
            compile_policy="compile",
            cpu_threads=1,
            cpu_interop_threads=1,
        )
        originals = (
            torch_fdtd.TorchSimulation.__init__,
            torch_fdtd.TorchSimulationPlan.__init__,
            torch_fdtd.TorchSimulation._compute_plan_identity,
            torch_fdtd.TorchSimulation._compute_compile_cache_key,
            torch_fdtd._compile_fullgraph,
        )
        receipt = lowering._CaseProducerReceipt(
            spec, space, geometry, sources, bloch, runtime
        )
        with receipt:
            arguments = {
                "space": space,
                "geometry": geometry,
                "sources": sources,
                "bloch": bloch,
                "runtime": runtime,
            }
            if courant_ratio is not None:
                arguments["courant_ratio"] = courant_ratio
            simulation = gmes.TorchSimulation(
                **arguments,
            )
        receipt.finalize(simulation)
        self.assertEqual(
            originals,
            (
                torch_fdtd.TorchSimulation.__init__,
                torch_fdtd.TorchSimulationPlan.__init__,
                torch_fdtd.TorchSimulation._compute_plan_identity,
                torch_fdtd.TorchSimulation._compute_compile_cache_key,
                torch_fdtd._compile_fullgraph,
            ),
        )
        return simulation, receipt

    def test_real_lazy_cpu_constructor_records_all_material_producers(self):
        simulation, receipt = self._construct_all_material()
        self.assertTrue(receipt.valid_for(simulation), receipt.reasons)
        self.assertIs(receipt.plan, simulation.plan)
        self.assertIs(receipt.plan, simulation.state.plan)
        self.assertEqual(receipt.plan_digest, simulation.plan_identity)
        self.assertEqual(receipt.key, simulation.compile_cache_key)
        self.assertEqual(
            hashlib.sha256(receipt.preimage_bytes).hexdigest(),
            simulation.compile_cache_key,
        )
        self.assertEqual(set(receipt.halves), set(lowering.COMPILED_REGIONS))
        report = receipt.diagnostic()
        encoded = json.dumps(report)
        self.assertEqual(report["status"], "incomplete")
        self.assertNotIn("all-material-2d", encoded)
        self.assertNotIn("_compile_cache_key_preimage", encoded)
        self.assertNotIn("source-file-read-provenance", report["reasons"])
        self.assertIn("inductor-cache-input-authentication", report["unverified"])
        self.assertEqual(
            report["scope"],
            "constructor-object-plan-and-gmes-specialization-producers-v1",
        )
        self.assertIn("caller_descriptor_sha256", report)

    def test_changed_descriptor_and_effective_courant_stay_explicitly_unverified(self):
        simulation, receipt = self._construct_all_material(
            descriptor_name="report-only-impostor", courant_ratio=0.5
        )
        self.assertTrue(receipt.valid_for(simulation), receipt.reasons)
        self.assertEqual(receipt.plan.dt, simulation.plan.dt)
        for gap in (
            "descriptor-to-builder-output-binding",
            "effective-source-transform-provenance",
            "effective-constructor-parameter-binding",
            "producer-callback-cardinality-order",
        ):
            self.assertIn(gap, receipt.diagnostic()["unverified"])

    def test_cross_wiring_and_post_capture_preimage_replacement_reject(self):
        simulation, receipt = self._construct_all_material()
        self.assertFalse(receipt.valid_for(object()))
        simulation._compile_cache_key_preimage = ("foreign",)
        simulation.compile_cache_key = hashlib.sha256(
            repr(("foreign",)).encode()
        ).hexdigest()
        self.assertFalse(receipt.valid_for(simulation))
        self.assertIn("producer-specialization-digest-differs", receipt.reasons)

    def test_constructor_exception_restores_all_receipt_hooks(self):
        import gmes
        import gmes.torch_fdtd as torch_fdtd
        from benchmarks.torch_tuning import MANIFEST, _build_case, load_manifest

        spec, space, geometry, sources, bloch = _build_case(
            "all-material-2d", load_manifest(MANIFEST)
        )
        sources = lowering._memory_sources(spec, sources)
        runtime = gmes.TorchRuntimeConfig(device="cpu", precision="float64")
        receipt = lowering._CaseProducerReceipt(
            spec, space, geometry, sources, bloch, runtime
        )
        original = torch_fdtd.TorchSimulation.__init__
        with self.assertRaises(TypeError):
            with receipt:
                gmes.TorchSimulation(
                    space=space,
                    geometry=geometry,
                    sources=sources,
                    bloch=bloch,
                    runtime=object(),
                )
        self.assertIs(torch_fdtd.TorchSimulation.__init__, original)
        self.assertIn("producer-construction-raised", receipt.reasons)

    def _synthetic_reports(self):
        import gmes

        simulation, receipt = self._construct_all_material()
        evidence = lowering._ReturnedModuleEvidence(receipt)
        with tempfile.TemporaryDirectory() as raw:
            for region in lowering.COMPILED_REGIONS:
                graph, module = ReturnedModuleEvidenceTest()._load(Path(raw), region)
                evidence.capture(
                    graph,
                    module,
                    [_IDENTITY_SOURCE],
                    (simulation, region, getattr(simulation, "_" + region)),
                )
            observer = types.SimpleNamespace(diagnostic=lambda: None)
            dispatcher = types.MethodType(
                gmes.TorchSimulation._run_compute_region, simulation
            )
            join = lowering._WrapperCallJoin(evidence, observer, dispatcher)
            join.completed = list(lowering.COMPILED_REGIONS)
            for ordinal, region in enumerate(
                (
                    "magnetic_half",
                    "electric_half",
                    "electric_half",
                    "electric_half",
                    "electric_half",
                )
            ):
                entry = next(item for item in evidence.entries if item[4][1] == region)
                join.events.append(
                    (
                        (entry, entry[3]["kernels"][0], None),
                        {
                            "ordinal": ordinal,
                            "branch": "synthetic",
                            "cubin_sha256": "0" * 64,
                            "selected_callable_id": None,
                            "parent_callable_id": None,
                        },
                    )
                )
            yield simulation, receipt, evidence, join

    def test_receipt_revalidation_suppresses_stale_module_and_join_projection(self):
        for mode in ("normal", "preimage", "explicit", "plan", "modules-only"):
            with self.subTest(mode=mode):
                for simulation, receipt, evidence, join in self._synthetic_reports():
                    if mode == "preimage":
                        simulation._compile_cache_key_preimage = ("foreign",)
                        simulation.compile_cache_key = hashlib.sha256(
                            repr(("foreign",)).encode()
                        ).hexdigest()
                    elif mode == "explicit":
                        receipt.reasons.append("producer-specialization-digest-differs")
                    elif mode == "plan":
                        simulation.plan = object()
                    module_report = evidence.diagnostic()
                    if mode == "normal":
                        self.assertEqual(len(module_report["modules"]), 2)
                        self.assertEqual(len(join.diagnostic()["events"]), 5)
                    elif mode == "modules-only":
                        self.assertEqual(len(module_report["modules"]), 2)
                    else:
                        self.assertEqual(module_report["modules"], [])
                        self.assertIn(
                            "producer-receipt-revalidation-failed",
                            module_report["reasons"],
                        )
                        join_report = join.diagnostic()
                        self.assertEqual(join_report["modules"], [])
                        self.assertEqual(join_report["events"], [])
                        self.assertIn(
                            "producer-receipt-revalidation-failed",
                            join_report["reasons"],
                        )


_IDENTITY_SOURCE = """from tests.test_torch_lowered_materialization import _IdentityAsyncCompile
async_compile = _IdentityAsyncCompile()
kernel = async_compile.triton('kernel', 'def kernel(): pass')
def while_loop_body_graph_0(args):
    return args
class Runner:
    def __init__(self, partitions):
        self.partitions = partitions
    def call(self, args):
        return while_loop_body_graph_0(args)
runner = Runner(partitions=[])
call = runner.call
"""


class ReturnedModuleEvidenceTest(unittest.TestCase):
    def _load(self, directory, name="wrapper", source=_IDENTITY_SOURCE):
        from torch._inductor.runtime.compile_tasks import _reload_python_module

        directory = directory.resolve(strict=True)
        path = directory / (name + ".py")
        path.write_text(source)
        graph = types.SimpleNamespace(cache_path=str(path), cache_key=name)
        module = _reload_python_module(name, str(path), set_sys_modules=False)
        return graph, module

    def test_alias_root_fixture_is_canonicalized_and_lexical_alias_rejects(self):
        from torch._inductor.runtime.compile_tasks import _reload_python_module

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            canonical = root / "canonical"
            canonical.mkdir()
            alias = root / "alias"
            alias.symlink_to(canonical, target_is_directory=True)

            graph, module = self._load(alias, "canonical_wrapper")
            self.assertEqual(Path(graph.cache_path).parent, canonical)
            lowering._returned_module_identity(graph, module, _IDENTITY_SOURCE)

            lexical_path = alias / "lexical_wrapper.py"
            lexical_path.write_text(_IDENTITY_SOURCE)
            lexical_graph = types.SimpleNamespace(
                cache_path=str(lexical_path), cache_key="lexical_wrapper"
            )
            lexical_module = _reload_python_module(
                "lexical_wrapper", str(lexical_path), set_sys_modules=False
            )
            self.assertNotEqual(lexical_path.resolve(strict=True), lexical_path)
            self.assertEqual(lexical_module.__file__, str(lexical_path))
            with self.assertRaisesRegex(
                ValueError, "returned module source/cache identity differs"
            ):
                lowering._returned_module_identity(
                    lexical_graph, lexical_module, _IDENTITY_SOURCE
                )

    def _simulation(self):
        return types.SimpleNamespace(
            _electric_half=object(), _magnetic_half=object(), _cuda_graphs={}
        )

    def test_actual_python_module_loader_retains_bound_runner_and_tuner_objects(self):
        with tempfile.TemporaryDirectory() as raw:
            evidence = lowering._ReturnedModuleEvidence()
            simulation = self._simulation()
            for region in lowering.COMPILED_REGIONS:
                graph, module = self._load(Path(raw), region)
                evidence.capture(
                    graph,
                    module,
                    [_IDENTITY_SOURCE],
                    (simulation, region, getattr(simulation, "_" + region)),
                )
            report = evidence.diagnostic()
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(
                report["reasons"],
                [
                    "wrapper-pin-mismatch:electric_half",
                    "wrapper-pin-mismatch:magnetic_half",
                ],
            )
            record = evidence.entries[-1][3]
            self.assertEqual(record["call_id"], id(module.call))
            self.assertEqual(record["receiver_id"], id(module.runner))
            self.assertEqual(record["kernels"][0]["autotuner_id"], id(module.kernel))
            self.assertEqual(
                record["module_functions"][0]["code_id"],
                id(module.while_loop_body_graph_0.__code__),
            )
            report["modules"].clear()
            self.assertEqual(len(evidence.diagnostic()["modules"]), 2)

    def test_substitutions_and_artifact_mutation_invalidate_returned_evidence(self):
        for mode in (
            "source",
            "module-path",
            "cache-key",
            "receiver",
            "call-code",
            "tuner",
            "tuner-function",
            "loop-function",
            "dispatch",
            "replay",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as raw:
                graph, module = self._load(Path(raw))
                simulation = self._simulation()
                evidence = lowering._ReturnedModuleEvidence()
                evidence.capture(
                    graph,
                    module,
                    [_IDENTITY_SOURCE],
                    (simulation, "electric_half", simulation._electric_half),
                )
                self.assertEqual(
                    evidence.reasons,
                    ["wrapper-pin-mismatch:electric_half"],
                )
                if mode == "source":
                    Path(graph.cache_path).write_text(_IDENTITY_SOURCE + "# modified\n")
                elif mode == "module-path":
                    module.__file__ = graph.cache_path + ".foreign"
                elif mode == "cache-key":
                    graph.cache_key = "foreign"
                elif mode == "receiver":
                    module.call = module.Runner([]).call
                elif mode == "call-code":
                    module.Runner.call.__code__ = (lambda self, args: None).__code__
                elif mode == "tuner":
                    module.kernel = _IdentityAsyncCompile().triton(
                        "kernel", "def kernel(): pass"
                    )
                elif mode == "tuner-function":
                    module.kernel.fn = (
                        _IdentityAsyncCompile()
                        .triton("kernel", "def kernel(): pass")
                        .fn
                    )
                elif mode == "loop-function":
                    module.while_loop_body_graph_0 = lambda args: args
                elif mode == "dispatch":
                    simulation._electric_half = object()
                else:
                    simulation._cuda_graphs["electric_half"] = object()
                report = evidence.diagnostic()
                self.assertEqual(report["modules"], [])
                self.assertTrue(report["reasons"])

    def test_precapture_foreign_code_filenames_reject_including_nested_code(self):
        for mode in ("call", "module-function", "nested"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as raw:
                source = _IDENTITY_SOURCE
                if mode == "nested":
                    source = source.replace(
                        "return while_loop_body_graph_0(args)",
                        "def inner():\n            return args\n        return inner()",
                    )
                graph, module = self._load(Path(raw), source=source)
                function = (
                    module.while_loop_body_graph_0
                    if mode == "module-function"
                    else module.Runner.call
                )
                original = function.__code__
                if mode == "nested":
                    replacement = original.replace(
                        co_consts=tuple(
                            (
                                item.replace(co_filename="<foreign-private-file>")
                                if isinstance(item, types.CodeType)
                                else item
                            )
                            for item in original.co_consts
                        )
                    )
                else:
                    replacement = original.replace(co_filename="<foreign-private-file>")
                self.assertEqual(original, replacement)
                function.__code__ = replacement
                simulation = self._simulation()
                evidence = lowering._ReturnedModuleEvidence()
                evidence.capture(
                    graph,
                    module,
                    [source],
                    (simulation, "electric_half", simulation._electric_half),
                )
                self.assertEqual(evidence.diagnostic()["modules"], [])
                self.assertIn("returned-module-capture-rejected", evidence.reasons)

    def test_same_filename_code_and_foreign_tuner_remain_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as raw:
            graph, module = self._load(Path(raw))
            original_code = module.Runner.call.__code__
            module.Runner.call.__code__ = original_code.replace()
            self.assertIsNot(original_code, module.Runner.call.__code__)
            module.kernel = _IdentityAsyncCompile().triton(
                "kernel", "def kernel(): return 'different body'"
            )
            simulation = self._simulation()
            evidence = lowering._ReturnedModuleEvidence()
            evidence.capture(
                graph,
                module,
                [_IDENTITY_SOURCE],
                (simulation, "electric_half", simulation._electric_half),
            )
            report = evidence.diagnostic()
            self.assertEqual(len(report["modules"]), 1)
            self.assertEqual(
                evidence.entries[0][3]["kernels"][0]["autotuner_id"], id(module.kernel)
            )
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("structural-equivalence", report["python_code_matching"])
            for gap in (
                "code-object-loader-provenance",
                "pre-capture-substitution-absence",
                "source-declaration-to-autotuner-provenance",
                "actual-wrapper-invocation-and-selected-launcher-join",
                "device-execution",
            ):
                self.assertIn(gap, report["unverified"])

    def test_json_projection_hides_personal_paths_keys_ids_and_preserves_aliases(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw) / "PersonalName"
            directory.mkdir()
            simulation = self._simulation()
            evidence = lowering._ReturnedModuleEvidence()
            for region in lowering.COMPILED_REGIONS:
                graph, module = self._load(directory, "private-basename-" + region)
                graph.cache_key = module.key = "/private/arbitrary/cache-key"
                setattr(simulation, "_" + region, module.call)
                evidence.capture(
                    graph, module, [_IDENTITY_SOURCE], (simulation, region, module.call)
                )
            report = evidence.diagnostic()
            encoded = json.dumps(report)
            for private in (
                raw,
                "PersonalName",
                "private-basename",
                "/private/arbitrary",
                str(id(module)),
                '"cache_path"',
                '"module_cache_key"',
                '_id"',
            ):
                self.assertNotIn(private, encoded)
            self.assertEqual(report, evidence.diagnostic())
            electric, magnetic = report["modules"]
            self.assertEqual(
                electric["construction_context"]["simulation"],
                magnetic["construction_context"]["simulation"],
            )
            self.assertEqual(
                electric["call"], electric["construction_context"]["dispatch_function"]
            )
            self.assertEqual(
                evidence.entries[-1][3]["cache_path"],
                str(
                    directory.resolve(strict=True)
                    / ("private-basename-" + region + ".py")
                ),
            )
            with mock.patch.object(
                lowering,
                "_returned_module_identity",
                side_effect=FileNotFoundError(
                    2, "PersonalName error", "/private/error-path"
                ),
            ):
                failed = evidence.diagnostic()
                evidence.capture(
                    graph, module, [_IDENTITY_SOURCE], (simulation, region, module.call)
                )
            for private in ("PersonalName", "/private/error-path"):
                self.assertNotIn(private, json.dumps(failed))
                self.assertNotIn(private, json.dumps(evidence.diagnostic()))
            self.assertIn("returned-module-revalidation-failed", failed["reasons"])

    def test_missing_emission_unknown_context_and_duplicate_declarations_reject(self):
        for mode in ("missing", "multiple", "cross-region", "duplicate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as raw:
                source = _IDENTITY_SOURCE
                if mode == "duplicate":
                    source += "kernel = async_compile.triton('kernel', 'def kernel(): pass')\n"
                graph, module = self._load(Path(raw), source=source)
                simulation = self._simulation()
                emissions = (
                    []
                    if mode == "missing"
                    else [source] * (2 if mode == "multiple" else 1)
                )
                function = (
                    simulation._magnetic_half
                    if mode == "cross-region"
                    else simulation._electric_half
                )
                evidence = lowering._ReturnedModuleEvidence()
                evidence.capture(
                    graph, module, emissions, (simulation, "electric_half", function)
                )
                self.assertTrue(evidence.reasons)
                self.assertEqual(evidence.diagnostic()["modules"], [])

    def test_capture_hook_pairs_post_rewrite_emission_and_restores_on_exception(self):
        from torch._inductor.graph import GraphLowering

        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as raw:
                code = types.SimpleNamespace(value=_IDENTITY_SOURCE)
                graph = types.SimpleNamespace()

                def producer(instance, wrapper_code):
                    wrapper_code.value += "# runtime rewrite\n"
                    GraphLowering.save_output_code(wrapper_code.value)
                    if fail:
                        raise RuntimeError("compile fixture failed")
                    produced, module = self._load(Path(raw), source=wrapper_code.value)
                    instance.__dict__.update(produced.__dict__)
                    return module

                observations = []
                original_save = GraphLowering.save_output_code
                original_compile = GraphLowering._compile_to_module_lines
                with mock.patch.object(
                    GraphLowering, "_compile_to_module_lines", producer
                ):
                    try:
                        with lowering._capturing_output_code(
                            lambda source: None, lambda *args: observations.append(args)
                        ):
                            module = GraphLowering._compile_to_module_lines(graph, code)
                    except RuntimeError:
                        self.assertTrue(fail)
                    else:
                        self.assertFalse(fail)
                        self.assertIs(observations[0][1], module)
                        self.assertEqual(observations[0][2], [code.value])
                        lowering._returned_module_identity(graph, module, code.value)
                    self.assertIs(GraphLowering._compile_to_module_lines, producer)
                    self.assertIs(GraphLowering.save_output_code, original_save)
                self.assertIs(GraphLowering._compile_to_module_lines, original_compile)
                self.assertEqual(len(observations), 0 if fail else 1)

    def test_callback_exception_restores_both_hooks(self):
        from torch._inductor.graph import GraphLowering

        original_save = GraphLowering.save_output_code
        original_compile = GraphLowering._compile_to_module_lines

        def producer(graph, code):
            GraphLowering.save_output_code(code)
            return object()

        def reject(*args):
            raise RuntimeError("module callback failed")

        with mock.patch.object(GraphLowering, "_compile_to_module_lines", producer):
            with self.assertRaisesRegex(RuntimeError, "module callback failed"):
                with lowering._capturing_output_code(lambda source: None, reject):
                    GraphLowering._compile_to_module_lines(object(), "source")
            self.assertIs(GraphLowering._compile_to_module_lines, producer)
            self.assertIs(GraphLowering.save_output_code, original_save)
        self.assertIs(GraphLowering._compile_to_module_lines, original_compile)


def _contract(*, regions: list[str] | None = None) -> dict[str, object]:
    source_hashes = {
        "gmes/torch_fdtd.py": "a" * 64,
        "gmes/torch_plan.py": "b" * 64,
        "gmes/torch_source.py": "c" * 64,
    }
    return {
        "case": "all-material-2d",
        "case_descriptor_sha256": "d" * 64,
        "compile_cache_key": "e" * 64,
        "compiled_region_topology": lowering.PINNED_CUDA_TOPOLOGY,
        "device": "cuda:0",
        "field_buffers": [
            {
                "name": name,
                "dtype": "torch.float64",
                "shape": [8],
                "stride": [1],
            }
            for name in lowering.FIELD_NAMES
        ],
        "plan_identity": "f" * 64,
        "precision": "float64",
        "required_regions": ["electric"] if regions is None else regions,
        "runtime_source_files_sha256": source_hashes,
        "runtime_source_sha256": hashlib.sha256(
            b"gmes/torch_fdtd.py\0"
            + b"a" * 64
            + b"\0"
            + b"gmes/torch_plan.py\0"
            + b"b" * 64
            + b"\0"
            + b"gmes/torch_source.py\0"
            + b"c" * 64
            + b"\0"
        ).hexdigest(),
        "source_buffer_sha256": "0" * 64,
        "torch_version": "2.13.0+cu130",
    }


def _wrapper(
    *,
    allocation: str = "",
    output: str = "arg0_1",
    payload: str = "value = tl.load(in_ptr0)\n    tl.store(out_ptr0, value)",
) -> str:
    return f"""\
kernel = async_compile.triton('kernel', r'''\
import triton
import triton.language as tl
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel: tl.constexpr):
    {payload}
''')

def call(args):
    arg0_1, = args
    assert_size_stride(arg0_1, (8,), (1,))
    {allocation}
    kernel.run(arg0_1, {output}, 8, grid=grid(8), stream=stream0)
"""


def _arithmetic_wrapper(*, allocation: str, output: str = "buf0") -> str:
    return f"""\
kernel = async_compile.triton('kernel', r'''\
import triton
import triton.language as tl
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel: tl.constexpr):
    left = tl.load(in_ptr0)
    right = tl.load(in_ptr1)
    tl.store(out_ptr0, left - right)
''')

def call(args):
    arg0_1, arg1_1 = args
    assert_size_stride(arg0_1, (8,), (1,))
    assert_size_stride(arg1_1, (8,), (1,))
    {allocation}
    kernel.run(arg0_1, arg1_1, {output}, 8, grid=grid(8), stream=stream0)
"""


def _runner_wrapper(
    *,
    body: str = "",
    helpers: str = "",
    export: str = "runner = Runner()\ncall = runner.call\n",
) -> str:
    prefix, call = _wrapper().split("def call(args):\n", maxsplit=1)
    scoped_call = "\n".join(
        f"    {line}" if line else line for line in call.splitlines()
    )
    return f"{prefix}class Runner:\n    def call(self, args):\n{scoped_call}\n{body}{helpers}{export}"


class LoweredMaterializationTest(unittest.TestCase):
    def _audit(self, source: str, *, contract: dict[str, object] | None = None):
        return lowering.audit_compiled_wrapper_sources(
            sources=[{"region": "electric", "source": source}],
            contract=_contract() if contract is None else contract,
        )

    def test_direct_inplace_output_is_verified(self):
        audit = self._audit(_wrapper())
        self.assertEqual(audit["status"], "verified")
        self.assertTrue(audit["verified"])
        output = audit["wrappers"][0]["launches"][0]["outputs"][0]
        self.assertEqual(output["kind"], "direct-input")

    def test_runner_scope_is_diagnostic_only(self):
        audit = self._audit(_runner_wrapper())
        self.assertFalse(audit["verified"])
        self.assertIn("wrapper-execution-scope-incomplete", audit["reasons"])
        self.assertEqual(
            audit["wrappers"][0]["launches"][0]["outputs"][0]["kind"],
            "direct-input",
        )

    def test_runner_call_scope_excludes_helper_and_get_args_collisions(self):
        audit = self._audit(
            _runner_wrapper(
                body="""
def get_args():
    arg0_1 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float64)
    buf0 = empty_strided_cuda((8,), (1,), torch.float64)
    kernel.run(arg0_1, buf0, 8, grid=grid(8), stream=stream0)
""",
                helpers="""
def helper():
    arg0_1 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float64)
    kernel.run(arg0_1, arg0_1, 8, grid=grid(8), stream=stream0)
""",
            )
        )
        self.assertFalse(audit["verified"])
        self.assertIn("wrapper-execution-scope-incomplete", audit["reasons"])
        wrapper = audit["wrappers"][0]
        self.assertEqual(len(wrapper["launches"]), 1)
        self.assertEqual(wrapper["launches"][0]["outputs"][0]["kind"], "direct-input")
        self.assertEqual(audit["full_domain_output_candidates"], [])

    def test_runner_call_scope_excludes_nested_function_collisions(self):
        source = _runner_wrapper().replace(
            "        kernel.run(arg0_1, arg0_1, 8, grid=grid(8), stream=stream0)",
            """        def nested():
            arg0_1 = rand_strided((8,), (1,), device='cuda:0', dtype=torch.float64)
            kernel.run(arg0_1, arg0_1, 8, grid=grid(8), stream=stream0)
        kernel.run(arg0_1, arg0_1, 8, grid=grid(8), stream=stream0)""",
        )
        audit = self._audit(source)
        self.assertFalse(audit["verified"])
        self.assertIn("wrapper-execution-scope-incomplete", audit["reasons"])
        self.assertEqual(len(audit["wrappers"][0]["launches"]), 1)

    def test_runner_call_reachable_helper_and_closure_alias_fail_closed(self):
        helper = """
def hidden_copy(arg0_1):
    assert_size_stride(arg0_1, (8,), (1,))
    copied = empty_strided_cuda((8,), (1,), torch.float64)
    kernel.run(arg0_1, copied, 8, grid=grid(8), stream=stream0)
    return copied
"""
        launch = "        kernel.run(arg0_1, arg0_1, 8, grid=grid(8), stream=stream0)"
        closure = """        def hidden_copy():
            copied = empty_strided_cuda((8,), (1,), torch.float64)
            kernel.run(arg0_1, copied, 8, grid=grid(8), stream=stream0)
            return copied
        invoke = hidden_copy
        invoke()
"""
        for source in (
            _runner_wrapper(helpers=helper).replace(
                launch, launch + "\n        hidden_copy(arg0_1)"
            ),
            _runner_wrapper().replace(launch, closure + launch),
        ):
            with self.subTest(source=source):
                audit = self._audit(source)
                self.assertFalse(audit["verified"])
                self.assertIn("wrapper-execution-call-unresolved", audit["reasons"])

    def test_runner_call_requires_supported_export_binding(self):
        for export, reason in (
            ("", "wrapper-execution-entry-missing"),
            (
                "runner = Runner()\ncall = hidden_entry\n",
                "wrapper-execution-entry-unresolved",
            ),
            (
                "runner = Runner()\ncall = runner.call\ncall = hidden_entry\n",
                "wrapper-execution-entry-rebound",
            ),
        ):
            with self.subTest(export=export):
                audit = self._audit(_runner_wrapper(export=export))
                self.assertFalse(audit["verified"])
                self.assertIn(reason, audit["reasons"])
                self.assertIn("wrapper-execution-scope-incomplete", audit["reasons"])

    def test_runner_scope_guard_rejects_known_and_annotated_rebindings(self):
        helper = """
def hidden_copy(arg0_1):
    copied = empty_strided_cuda((8,), (1,), torch.float64)
    kernel.run(arg0_1, copied, 8, grid=grid(8), stream=stream0)
"""
        launch = "        kernel.run(arg0_1, arg0_1, 8, grid=grid(8), stream=stream0)"
        primitive_rebound = _runner_wrapper(helpers=helper).replace(
            launch,
            "        copy_if_misaligned = hidden_copy\n"
            "        copy_if_misaligned(arg0_1)\n" + launch,
        )
        annotated_export = _runner_wrapper(
            helpers=helper,
            export="runner = Runner()\ncall = runner.call\ncall: object = hidden_copy\n",
        )
        for source in (primitive_rebound, annotated_export):
            with self.subTest(source=source):
                audit = self._audit(source)
                self.assertFalse(audit["verified"])
                self.assertIn("wrapper-execution-scope-incomplete", audit["reasons"])

    def test_runner_call_local_reassignment_stays_unmapped(self):
        audit = self._audit(
            _runner_wrapper(
                body="""
""",
                helpers="""
""",
            ).replace(
                "assert_size_stride(arg0_1, (8,), (1,))\n        ",
                "assert_size_stride(arg0_1, (8,), (1,))\n        arg0_1 = copy_if_misaligned(arg0_1)\n        ",
            )
        )
        self.assertFalse(audit["verified"])
        self.assertIn("launch-output-unmapped:kernel:out_ptr0", audit["reasons"])

    def test_runner_call_list_loop_and_clone_outputs_stay_unmapped(self):
        for replacement in (
            """        value = arg0_1.clone()
        kernel.run(arg0_1, value, 8, grid=grid(8), stream=stream0)""",
            """        values = [arg0_1]
        for _ in range(1):
            values[0] = copy_if_misaligned(values[0])
        kernel.run(arg0_1, values[0], 8, grid=grid(8), stream=stream0)""",
        ):
            with self.subTest(replacement=replacement):
                source = _runner_wrapper().replace(
                    "        kernel.run(arg0_1, arg0_1, 8, grid=grid(8), stream=stream0)",
                    replacement,
                )
                audit = self._audit(source)
                self.assertFalse(audit["verified"])
                self.assertIn(
                    "launch-output-unmapped:kernel:out_ptr0", audit["reasons"]
                )

    def test_ambiguous_runner_entrypoints_fail_closed(self):
        source = (
            _runner_wrapper()
            + "\nclass Runner:\n    def call(self, args):\n        pass\n"
        )
        audit = self._audit(source)
        self.assertFalse(audit["verified"])
        self.assertIn("wrapper-execution-scope-ambiguous", audit["reasons"])
        self.assertIn("wrapper-execution-scope-incomplete", audit["reasons"])

    def test_arithmetic_field_workspaces_and_pool_reuse_are_diagnostic_only(self):
        for allocation in (
            "buf0 = empty_strided_cuda((8,), (1,), torch.float64)",
            "buf0 = alloc_from_pool((8,), (1,), torch.float64)",
        ):
            with self.subTest(allocation=allocation):
                audit = self._audit(_arithmetic_wrapper(allocation=allocation))
                self.assertEqual(audit["status"], "verified")
                self.assertTrue(audit["verified"])
                candidates = audit["full_domain_output_candidates"]
                self.assertEqual(len(candidates), 1)
                self.assertEqual(
                    candidates[0]["classification"],
                    "arithmetic-produced-field-workspace",
                )
                self.assertEqual(
                    candidates[0]["reused_storage"], "alloc_from_pool" in allocation
                )

    def test_identity_field_copy_and_unknown_payload_fail_closed(self):
        identity = self._audit(
            _wrapper(
                allocation="buf0 = empty_strided_cuda((8,), (1,), torch.float64)",
                output="buf0",
            )
        )
        self.assertFalse(identity["verified"])
        self.assertIn("field-identity-copy:kernel:out_ptr0", identity["reasons"])
        self.assertEqual(
            identity["full_domain_output_candidates"][0]["classification"],
            "field-identity-copy",
        )

        for payload in (
            "value = tl.load(in_ptr0) + 0\n    tl.store(out_ptr0, value)",
            "value = tl.load(in_ptr0)\n    value = tl.load(in_ptr0)\n    tl.store(out_ptr0, value)",
            "if xnumel:\n        value = tl.load(in_ptr0) + tl.load(in_ptr0)\n        tl.store(out_ptr0, value)",
        ):
            with self.subTest(payload=payload):
                unknown = self._audit(
                    _wrapper(
                        allocation="buf0 = alloc_from_pool((8,), (1,), torch.float64)",
                        output="buf0",
                        payload=payload,
                    )
                )
                self.assertFalse(unknown["verified"])
                self.assertIn(
                    "field-producer-unknown:kernel:out_ptr0", unknown["reasons"]
                )

    def test_selector_and_opaque_payloads_are_not_arithmetic_provenance(self):
        payloads = (
            "left = tl.load(in_ptr0)\n    right = tl.load(in_ptr1)\n"
            "    value = tl.where(right > 0, left, left)\n"
            "    tl.store(out_ptr0, value)",
            "left = tl.load(in_ptr0)\n    right = tl.load(in_ptr1)\n"
            "    value = tl.where(xnumel > 0, left, right)\n"
            "    tl.store(out_ptr0, value)",
            "left = tl.load(in_ptr0)\n    right = tl.load(in_ptr1)\n"
            "    value = left if xnumel > 0 else right\n"
            "    tl.store(out_ptr0, value)",
            "left = tl.load(in_ptr0)\n    right = tl.load(in_ptr1)\n"
            "    value = opaque_transform(left, right)\n"
            "    tl.store(out_ptr0, value)",
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                source = _arithmetic_wrapper(
                    allocation="buf0 = empty_strided_cuda((8,), (1,), torch.float64)"
                ).replace(
                    "left = tl.load(in_ptr0)\n    right = tl.load(in_ptr1)\n"
                    "    tl.store(out_ptr0, left - right)",
                    payload,
                )
                audit = self._audit(source)
                self.assertFalse(audit["verified"])
                self.assertIn(
                    "field-producer-unknown:kernel:out_ptr0", audit["reasons"]
                )
                self.assertEqual(
                    audit["full_domain_output_candidates"][0]["classification"],
                    "field-producer-unknown",
                )

    def test_larger_or_nonfield_outputs_are_not_clone_candidates(self):
        for allocation in (
            "buf0 = empty_strided_cuda((64,), (1,), torch.float64)",
            "buf0 = empty_strided_cuda((8,), (1,), torch.int64)",
            "buf0 = empty_strided_cuda((4, 2), (2, 1), torch.float64)",
            "buf0 = empty_strided_cuda((8,), (1,), torch.float32)",
            "buf0 = empty_strided_cuda((8,), (2,), torch.float64)",
        ):
            with self.subTest(allocation=allocation):
                audit = self._audit(_wrapper(allocation=allocation, output="buf0"))
                self.assertEqual(audit["status"], "verified")
                self.assertTrue(audit["verified"])
                self.assertEqual(audit["full_domain_output_candidates"], [])
                self.assertEqual(len(audit["non_field_output_exclusions"]), 1)

    def test_unknown_factory_mapping_and_opaque_paths_fail_closed(self):
        for source, reason in (
            (
                _wrapper(allocation="buf0 = mystery_alloc((8,))", output="buf0"),
                "allocation-factory-unknown:buf0",
            ),
            (_wrapper(output="buf0"), "launch-output-unmapped:kernel:out_ptr0"),
            ("extern_kernels.opaque()", "opaque-external-kernel"),
        ):
            with self.subTest(reason=reason):
                audit = self._audit(source)
                self.assertEqual(audit["status"], "unverified")
                self.assertIn(reason, audit["reasons"])

    def test_missing_required_region_fails_closed(self):
        audit = self._audit(
            _wrapper(), contract=_contract(regions=["electric", "magnetic"])
        )
        self.assertEqual(audit["status"], "unverified")
        self.assertIn("missing-region-coverage:magnetic", audit["reasons"])

    def test_both_required_halves_need_separate_payload_provenance(self):
        contract = _contract(regions=["electric", "magnetic"])
        audit = lowering.audit_compiled_wrapper_sources(
            sources=[
                {
                    "region": "electric",
                    "source": _arithmetic_wrapper(
                        allocation="buf0 = empty_strided_cuda((8,), (1,), torch.float64)"
                    ),
                },
                {
                    "region": "magnetic",
                    "source": _arithmetic_wrapper(
                        allocation="buf0 = alloc_from_pool((8,), (1,), torch.float64)"
                    ),
                },
            ],
            contract=contract,
        )
        self.assertTrue(audit["verified"])
        self.assertEqual(audit["region_coverage"]["missing"], [])
        self.assertEqual(len(audit["full_domain_output_candidates"]), 2)

    def test_malformed_field_layout_and_runtime_digest_fail_closed(self):
        contract = _contract()
        contract["field_buffers"][0]["dtype"] = "torch.float32"  # type: ignore[index]
        audit = self._audit(_wrapper(), contract=contract)
        self.assertEqual(audit["status"], "unverified")
        self.assertIn("contract-field-buffers", audit["reasons"])

        contract = _contract()
        contract["runtime_source_sha256"] = "1" * 64
        audit = self._audit(_wrapper(), contract=contract)
        self.assertEqual(audit["status"], "unverified")
        self.assertIn("contract-runtime-source-digest", audit["reasons"])

    def test_bundle_rehashes_sources_and_rejects_contract_mismatch(self):
        source = _wrapper()
        audit = self._audit(source)
        wrapper = audit["wrappers"][0]
        wrapper["_source_bytes"] = source.encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            lowering._write_bundle(root, audit)
            verified = lowering.verify_compiled_wrapper_audit(
                output_directory=root, expected_contract=_contract()
            )
            self.assertTrue(verified["verified"])
            descriptor = json.loads((root / "lowering-manifest.json").read_text())[
                "wrappers"
            ][0]
            (root / descriptor["path"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "digest or size"):
                lowering.verify_compiled_wrapper_audit(
                    output_directory=root, expected_contract=_contract()
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            audit = self._audit(source)
            audit["wrappers"][0]["_source_bytes"] = source.encode()
            lowering._write_bundle(root, audit)
            contract = _contract()
            contract["compile_cache_key"] = "9" * 64
            with self.assertRaisesRegex(ValueError, "contract differs"):
                lowering.verify_compiled_wrapper_audit(
                    output_directory=root, expected_contract=contract
                )


if __name__ == "__main__":
    unittest.main()
