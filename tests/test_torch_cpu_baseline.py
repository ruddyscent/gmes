import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from statistics import median
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "benchmarks" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TorchCpuBaselineTest(unittest.TestCase):
    _HOST_IDENTITY_SALT = (
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )

    @classmethod
    def setUpClass(cls):
        cls.baseline_module = load_script("torch_cpu_baseline.py")
        cls.manifest = json.loads(
            (ROOT / "benchmarks" / "native_oracle_workloads.json").read_text()
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _case_specs(self):
        acceptance = self.manifest["performance_gates"]["cpu_acceptance"]
        specs = {
            case["name"]: case
            for group in ("correctness", "benchmarks", "physical_checks")
            for case in self.manifest[group]
        }
        return [copy.deepcopy(specs[name]) for name in acceptance["cases"]]

    def _benchmark_contract(self):
        reference = self.manifest["reference"]
        return {
            "initializer": reference["field_initializer"],
            "seed": reference["seed"],
            "field_scale": reference["field_scale"],
            "warmup_steps": reference["performance_warmup_steps"],
            "steps_per_repeat": reference["performance_steps_per_repeat"],
            "repetitions": reference["performance_repetitions"],
            "profile_steps": reference["performance_profile_steps"],
            "timer": "time.perf_counter",
            "sample_start": "independently-restored-pre-warmup-state",
        }

    def _evidence(self):
        timing = self.manifest["performance_gates"]["cpu_acceptance"][
            "timing_reference"
        ]
        return {
            **copy.deepcopy(timing["legacy_evidence"]),
            "candidate_git_commit": timing["root_commit"],
            "candidate_git_status": "",
        }

    @staticmethod
    def _environment(threads):
        return {
            "hostname": "redacted",
            "platform": "Linux-baseline",
            "python": "3.14.0",
            "torch": "2.13.0+cpu",
            "cuda_runtime": None,
            "devices": [],
            "cpu_count": 8,
            "cpu_affinity": list(range(8)),
            "cpu_count_physical_affinity": 4,
            "cpu_topology": "0,0\n1,0\n2,0\n3,0",
            "cpu_model": f"baseline CPU\nCPU(s) scaling MHz: {90 + threads}%",
            "gpu_topology": None,
            "thread_environment": {
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
            },
        }

    @staticmethod
    def _advance(raw):
        values = [float(value) for value in raw]
        middle = median(values)
        relative_mad = median(abs(value - middle) for value in values) / middle
        return {
            "raw_seconds": values,
            "median_seconds": middle,
            "relative_mad": relative_mad,
            "repetitions": len(values),
            "steps_per_repeat": 100,
            "seconds_per_step": middle / 100,
        }

    def _raw_artifact(self, threads):
        environment = self._environment(threads)
        contract = self._benchmark_contract()
        cases = []
        for spec in self._case_specs():
            cases.append(
                {
                    "schema_version": 2,
                    "backend": "torch",
                    "workload": spec,
                    "benchmark_contract": copy.deepcopy(contract),
                    "runtime": {
                        "device": "cpu",
                        "precision": "float64",
                        "compile_policy": "compile",
                        "compile_mode": "default",
                        "explicit_cuda_graphs": False,
                        "execution_policy": "auto",
                        "experimental_dispersive_grouping": False,
                        "experimental_dispersive_grouping_scope": "combined",
                        "threads": threads,
                        "interop_threads": 1,
                        "cpu_affinity": copy.deepcopy(environment["cpu_affinity"]),
                        "cpu_count_physical_affinity": environment[
                            "cpu_count_physical_affinity"
                        ],
                        "cpu_topology": environment["cpu_topology"],
                        "compile_cache_key": ["legacy", spec["name"], threads],
                    },
                    "measurements": {"advance": self._advance([100.0] * 15)},
                    "memory": {"bounded": True},
                    "profiler": {
                        "chrome_trace": f"{spec['name']}-{threads}.json",
                        "positive_allocation_events": 15,
                        "allocated_bytes": 5560,
                        "freed_bytes": 5560,
                        "allocation_net_bytes": 0,
                        "max_allocation_bytes": 792,
                        "allocation_size_histogram": {"160": 10, "792": 5},
                    },
                    "acceptance": {
                        "compiler_clean": True,
                        "compiled_hot_path_complete": True,
                        "external_indexed_writes_only_sources": True,
                        "steady_state_transfers_zero": True,
                        "storage_stable": True,
                        "memory_bounded": True,
                        "recurring_allocations_zero": False,
                        "measurement_contract_matches_manifest": True,
                        "state_progressed": True,
                        "passed": False,
                    },
                }
            )
        return {
            "schema_version": 3,
            "kind": "cpu-acceptance-thread-slice",
            "evidence": self._evidence(),
            "environment": environment,
            "cases": cases,
        }

    def _artifact(self, threads):
        return self.baseline_module.sanitize_public_baseline_artifact(
            self._raw_artifact(threads), host_identity_salt=self._HOST_IDENTITY_SALT
        )

    def _write(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value, sort_keys=True))
        return path

    def _load(self, one=None, physical=None):
        one = self._artifact(1) if one is None else one
        physical = self._artifact(4) if physical is None else physical
        paths = [self._write("physical.json", physical), self._write("one.json", one)]
        manifest = copy.deepcopy(self.manifest)
        by_mode = {
            "one": {
                "thread_mode": "one",
                "threads": 1,
                "publication_url": self.baseline_module._release_asset_url("one"),
                "size_bytes": paths[1].stat().st_size,
                "sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
            },
            "physical": {
                "thread_mode": "physical",
                "threads": 4,
                "publication_url": self.baseline_module._release_asset_url("physical"),
                "size_bytes": paths[0].stat().st_size,
                "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            },
        }
        manifest["performance_gates"]["cpu_acceptance"]["timing_reference"][
            "slice_artifacts"
        ] = [by_mode["one"], by_mode["physical"]]
        with patch.object(self.baseline_module, "_FROZEN_SLICE_ARTIFACTS", by_mode):
            baseline = self.baseline_module.load_torch_cpu_baseline(paths, manifest)
        return baseline, paths

    def _candidate(self, ratio=1.0):
        candidate = copy.deepcopy(self._artifact(1)["cases"][0])
        candidate["measurements"]["advance"] = self._advance([100.0 * ratio] * 15)
        candidate["profiler"]["field_buffer_sizes_bytes"] = {
            "state.Ex": 4096,
            "state.Ey": 4096,
            "state.Ez": 4096,
            "state.Hx": 4096,
            "state.Hy": 4096,
            "state.Hz": 4096,
        }
        return candidate

    def test_loads_two_slices_and_records_exact_source_hashes(self):
        baseline, paths = self._load()
        self.assertEqual(baseline["kind"], "torch-cpu-baseline")
        self.assertEqual(
            [item["thread_mode"] for item in baseline["source_artifacts"]],
            ["one", "physical"],
        )
        self.assertEqual(
            [len(item["cases"]) for item in baseline["source_artifacts"]], [6, 6]
        )
        actual = {
            hashlib.sha256(path.read_bytes()).hexdigest(): path.stat().st_size
            for path in paths
        }
        self.assertEqual(
            {
                item["sha256"]: item["size_bytes"]
                for item in baseline["source_artifacts"]
            },
            actual,
        )
        for item in baseline["source_artifacts"]:
            self.assertNotIn("path", item)
            self.assertNotIn("raw_cpu_model", item)
            self.assertTrue(
                item["publication_url"].startswith(
                    "https://github.com/ruddyscent/gmes/releases/download/"
                )
            )
        self.assertEqual(baseline["source_artifacts"][0]["evidence"], self._evidence())
        self.assertNotIn("thread_environment", baseline["environment"])
        self.assertEqual(
            baseline["source_artifacts"][0]["thread_environment"],
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        self.assertEqual(
            baseline["environment"],
            {
                "hostname": "redacted",
                "host_identity": self.baseline_module._host_identity_token_from_raw_environment(
                    self._environment(1), self._HOST_IDENTITY_SALT
                ),
            },
        )
        json.dumps(baseline, allow_nan=False)

    def test_production_manifest_uses_external_release_assets(self):
        artifacts = self.manifest["performance_gates"]["cpu_acceptance"][
            "timing_reference"
        ]["slice_artifacts"]
        self.assertEqual(
            [
                (
                    artifact["thread_mode"],
                    artifact["threads"],
                    artifact["publication_url"],
                    artifact["size_bytes"],
                    artifact["sha256"],
                )
                for artifact in artifacts
            ],
            [
                (
                    "one",
                    1,
                    self.baseline_module._release_asset_url("one"),
                    18281,
                    "ea57620653b6e96a200ffc15ba8ca9cf2309a5ada2d8ee86a2945e4787431c79",
                ),
                (
                    "physical",
                    4,
                    self.baseline_module._release_asset_url("physical"),
                    18292,
                    "492b478211b5d1c32493197064393601008f1f2ca5683e261d9d103699b87ba6",
                ),
            ],
        )
        for name in (
            "torch-cpu-baseline-one.json",
            "torch-cpu-baseline-physical.json",
        ):
            self.assertFalse(
                (ROOT / "benchmarks" / "evidence" / "issue-123" / name).exists()
            )

    def test_production_manifest_and_loader_pin_the_same_artifact_bytes(self):
        artifacts = self.manifest["performance_gates"]["cpu_acceptance"][
            "timing_reference"
        ]["slice_artifacts"]
        self.assertEqual(
            {artifact["thread_mode"]: artifact for artifact in artifacts},
            self.baseline_module._FROZEN_SLICE_ARTIFACTS,
        )

    def test_rejects_incomplete_or_noncanonical_release_asset_schema(self):
        missing_size = copy.deepcopy(self.manifest)
        del missing_size["performance_gates"]["cpu_acceptance"]["timing_reference"][
            "slice_artifacts"
        ][0]["size_bytes"]
        with self.assertRaisesRegex(ValueError, "artifact schema"):
            self.baseline_module._manifest_contract(missing_size)

        for publication_url in (
            "http://github.com/ruddyscent/gmes/releases/download/"
            "issue-123-torch-cpu-baseline-v1/torch-cpu-baseline-one.json",
            "https://github.com/ruddyscent/gmes/releases/download/latest/"
            "torch-cpu-baseline-one.json",
            self.baseline_module._release_asset_url("one") + "?download=1",
        ):
            manifest = copy.deepcopy(self.manifest)
            manifest["performance_gates"]["cpu_acceptance"]["timing_reference"][
                "slice_artifacts"
            ][0]["publication_url"] = publication_url
            with (
                self.subTest(publication_url=publication_url),
                self.assertRaisesRegex(ValueError, "artifact pin"),
            ):
                self.baseline_module._manifest_contract(manifest)

    def test_rejects_artifact_byte_size_that_differs_from_the_manifest_pin(self):
        one_path = self._write("one.json", self._artifact(1))
        physical_path = self._write("physical.json", self._artifact(4))
        paths = [one_path, physical_path]
        artifacts = copy.deepcopy(
            self.manifest["performance_gates"]["cpu_acceptance"]["timing_reference"][
                "slice_artifacts"
            ]
        )
        artifacts[0]["size_bytes"] = one_path.stat().st_size
        artifacts[0]["sha256"] = hashlib.sha256(one_path.read_bytes()).hexdigest()
        artifacts[1]["size_bytes"] = physical_path.stat().st_size
        artifacts[1]["sha256"] = hashlib.sha256(physical_path.read_bytes()).hexdigest()
        artifacts[0]["size_bytes"] += 1
        manifest = copy.deepcopy(self.manifest)
        manifest["performance_gates"]["cpu_acceptance"]["timing_reference"][
            "slice_artifacts"
        ] = artifacts
        by_mode = {artifact["thread_mode"]: artifact for artifact in artifacts}
        with (
            patch.object(self.baseline_module, "_FROZEN_SLICE_ARTIFACTS", by_mode),
            self.assertRaisesRegex(ValueError, "byte size does not match"),
        ):
            self.baseline_module.load_torch_cpu_baseline(paths, manifest)

    def test_rejects_artifact_bytes_that_differ_from_the_manifest_pin(self):
        one_path = self._write("one.json", self._artifact(1))
        physical_path = self._write("physical.json", self._artifact(4))
        paths = [one_path, physical_path]
        manifest = copy.deepcopy(self.manifest)
        pins = {
            "one": {
                "thread_mode": "one",
                "threads": 1,
                "publication_url": self.baseline_module._release_asset_url("one"),
                "size_bytes": one_path.stat().st_size,
                "sha256": hashlib.sha256(one_path.read_bytes()).hexdigest(),
            },
            "physical": {
                "thread_mode": "physical",
                "threads": 4,
                "publication_url": self.baseline_module._release_asset_url("physical"),
                "size_bytes": physical_path.stat().st_size,
                "sha256": hashlib.sha256(physical_path.read_bytes()).hexdigest(),
            },
        }
        manifest["performance_gates"]["cpu_acceptance"]["timing_reference"][
            "slice_artifacts"
        ] = [pins["one"], pins["physical"]]
        raw = one_path.read_bytes()
        self.assertIn(b": ", raw)
        one_path.write_bytes(raw.replace(b": ", b":\t", 1))
        with patch.object(self.baseline_module, "_FROZEN_SLICE_ARTIFACTS", pins):
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                self.baseline_module.load_torch_cpu_baseline(paths, manifest)

    def test_rejects_nonexact_legacy_evidence(self):
        for mutation in (
            "candidate_git_commit",
            "candidate_git_status",
            "runner_sha256",
        ):
            one = self._artifact(1)
            one["evidence"][mutation] = "dirty" if mutation.endswith("status") else "0"
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(ValueError, "evidence"),
            ):
                self._load(one=one)

    def test_rejects_incomplete_or_different_host_identity(self):
        one = self._artifact(1)
        del one["environment"]["host_identity"]
        with self.assertRaisesRegex(ValueError, "public environment schema"):
            self._load(one=one)
        physical = self._artifact(4)
        physical["environment"]["hostname"] = "another-host"
        with self.assertRaisesRegex(ValueError, "hostname must be redacted"):
            self._load(physical=physical)
        physical = self._artifact(4)
        physical["environment"]["host_identity"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different host identity"):
            self._load(physical=physical)

    def test_rejects_unexpected_public_profiler_fields(self):
        one = self._artifact(1)
        one["cases"][0]["profiler"]["chrome_trace"] = "/tmp/trace.json"
        with self.assertRaisesRegex(ValueError, "profiler schema"):
            self._load(one=one)

    def test_rejects_wrong_case_order_or_workload_spec(self):
        one = self._artifact(1)
        one["cases"][0], one["cases"][1] = one["cases"][1], one["cases"][0]
        with self.assertRaisesRegex(ValueError, "wrong case order"):
            self._load(one=one)
        one = self._artifact(1)
        one["cases"][0]["workload"]["resolution"] += 1
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            self._load(one=one)

    def test_rejects_wrong_runtime_or_hard_control(self):
        mutations = (
            ("runtime", "interop_threads", 2),
            ("runtime", "compile_mode", "reduce-overhead"),
            ("runtime", "experimental_dispersive_grouping", True),
            ("acceptance", "state_progressed", False),
        )
        for group, name, value in mutations:
            one = self._artifact(1)
            one["cases"][0][group][name] = value
            with self.subTest(group=group, name=name), self.assertRaises(ValueError):
                self._load(one=one)
        one = self._artifact(1)
        one["cases"][0]["benchmark_contract"]["profile_steps"] = 6
        with self.assertRaisesRegex(ValueError, "benchmark contract"):
            self._load(one=one)
        one = self._artifact(1)
        one["cases"][0]["memory"]["bounded"] = False
        with self.assertRaisesRegex(ValueError, "not bounded"):
            self._load(one=one)

    def test_rejects_legacy_allocation_or_pass_flags_from_public_projection(self):
        one = self._artifact(1)
        one["cases"][0]["acceptance"]["recurring_allocations_zero"] = True
        one["cases"][0]["acceptance"]["passed"] = True
        with self.assertRaisesRegex(ValueError, "runtime controls"):
            self._load(one=one)

    def test_sanitizer_removes_host_process_trace_and_local_path_data(self):
        raw = self._raw_artifact(1)
        raw["environment"]["hostname"] = "owner-workstation"
        raw["cases"][0]["memory"]["cpu_rss_fresh_process"] = {
            "pid": 1234,
            "parent_pid": 99,
            "local_path": "/home/owner/results.json",
        }
        raw["cases"][0]["runtime"]["compile_cache_key"] = "owner-cache-key"
        raw["cases"][0]["profiler"]["chrome_trace"] = "/tmp/owner-trace.json"
        public = self.baseline_module.sanitize_public_baseline_artifact(
            raw, host_identity_salt=self._HOST_IDENTITY_SALT
        )
        rendered = json.dumps(public, sort_keys=True)
        for text in (
            "owner-workstation",
            "baseline CPU",
            "cpu_affinity",
            "cpu_topology",
            "gpu_topology",
            "compile_cache_key",
            "cpu_rss_fresh_process",
            "parent_pid",
            '"pid"',
            "/tmp/",
            "/home/",
            "chrome_trace",
        ):
            with self.subTest(text=text):
                self.assertNotIn(text, rendered)
        self.assertEqual(
            set(public["environment"]),
            {"hostname", "host_identity", "thread_environment"},
        )
        self.assertEqual(
            set(public["cases"][0]["profiler"]),
            set(self.baseline_module._PUBLIC_PROFILER_KEYS),
        )

    def test_host_identity_commitment_requires_matching_salt_and_host(self):
        environment = self._environment(1)
        baseline = self.baseline_module._host_identity_token_from_raw_environment(
            environment, self._HOST_IDENTITY_SALT
        )
        self.assertEqual(
            self.baseline_module.privacy_preserving_host_identity(
                environment, salt=self._HOST_IDENTITY_SALT
            ),
            baseline,
        )
        self.assertNotEqual(
            self.baseline_module._host_identity_token_from_raw_environment(
                environment, "f" * 64
            ),
            baseline,
        )
        environment["cpu_model"] = "different CPU"
        self.assertNotEqual(
            self.baseline_module._host_identity_token_from_raw_environment(
                environment, self._HOST_IDENTITY_SALT
            ),
            baseline,
        )

    def test_rejects_bad_raw_samples_and_reported_median(self):
        bad_values = ([100.0] * 14, [100.0] * 14 + [math.inf], [100.0] * 14 + [0])
        for values in bad_values:
            one = self._artifact(1)
            one["cases"][0]["measurements"]["advance"] = self._advance(values)
            with (
                self.subTest(values=values[-1:]),
                self.assertRaisesRegex(ValueError, "raw samples|15 raw samples"),
            ):
                self._load(one=one)
        one = self._artifact(1)
        one["cases"][0]["measurements"]["advance"]["median_seconds"] = 99.0
        with self.assertRaisesRegex(ValueError, "reported median"):
            self._load(one=one)

    def test_rejects_unstable_raw_samples(self):
        one = self._artifact(1)
        values = [float(value) for value in range(50, 200, 10)]
        one["cases"][0]["measurements"]["advance"] = self._advance(values)
        with self.assertRaisesRegex(ValueError, "relative-MAD limit"):
            self._load(one=one)

    def test_rejects_malformed_allocation_histogram(self):
        mutations = (
            ("positive_allocation_events", 16, "event histogram"),
            ("allocated_bytes", 5561, "byte histogram"),
            ("allocation_net_bytes", 1, "net bytes"),
        )
        for name, value, message in mutations:
            one = self._artifact(1)
            one["cases"][0]["profiler"][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                self._load(one=one)
        one = self._artifact(1)
        one["cases"][0]["profiler"]["allocation_size_histogram"]["792"] = 6
        with self.assertRaisesRegex(ValueError, "divisible by profile steps"):
            self._load(one=one)

    def test_finds_case_by_name_and_exact_thread_count(self):
        baseline, _paths = self._load()
        case = self.baseline_module.find_baseline_case(baseline, "cpu-crossover-2d", 4)
        self.assertEqual(case["name"], "cpu-crossover-2d")
        self.assertEqual(case["threads"], 4)
        with self.assertRaises(KeyError):
            self.baseline_module.find_baseline_case(baseline, "cpu-crossover-2d", 2)

    def test_compares_candidate_at_and_above_five_percent_boundary(self):
        baseline, _paths = self._load()
        boundary = self.baseline_module.compare_candidate_to_baseline(
            baseline, self._candidate(1.05)
        )
        self.assertTrue(boundary["comparison_valid"])
        self.assertTrue(boundary["within_five_percent"])
        self.assertAlmostEqual(boundary["candidate_to_torch_baseline_ratio"], 1.05)
        self.assertEqual(len(boundary["reference_raw_seconds_per_step"]), 15)
        self.assertEqual(len(boundary["candidate_raw_seconds_per_step"]), 15)
        reference = self.baseline_module.find_baseline_case(
            baseline, "cpu-crossover-2d", 1
        )
        self.assertEqual(
            boundary["reference_source_artifact_sha256"],
            reference["source_artifact_sha256"],
        )
        self.assertEqual(
            boundary["reference_root_commit"],
            self._evidence()["candidate_git_commit"],
        )

        slower = self.baseline_module.compare_candidate_to_baseline(
            baseline, self._candidate(1.050001)
        )
        self.assertTrue(slower["comparison_valid"])
        self.assertFalse(slower["within_five_percent"])

        wrong_contract = self._candidate()
        wrong_contract["runtime"]["execution_policy"] = "dense"
        invalid = self.baseline_module.compare_candidate_to_baseline(
            baseline, wrong_contract
        )
        self.assertFalse(invalid["comparison_valid"])
        self.assertTrue(invalid["contract_errors"])


if __name__ == "__main__":
    unittest.main()
