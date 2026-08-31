import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from benchmarks import issue123_completion as completion
from benchmarks import issue123_differential as differential
from benchmarks import native_oracle


class Issue123DifferentialEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = native_oracle.load_manifest(differential.DEFAULT_MANIFEST)
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": hashlib.sha256(
                differential.DEFAULT_MANIFEST.read_bytes()
            ).hexdigest(),
        }

    def descriptor(self, path):
        raw = path.read_bytes()
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "media_type": differential.MEDIA_TYPE_NPZ,
            "candidate_evidence": self.candidate,
        }

    @staticmethod
    def arrays(dtype):
        values = {
            name: np.asarray([1.0, -2.0, 3.0], dtype=dtype)
            for name in differential.FIELD_ARRAYS
        }
        values["persistent/state/indices"] = np.asarray([1, 7], dtype=np.int64)
        values["persistent/state/values"] = np.asarray([0.25, -0.5], dtype=dtype)
        return values

    def write_npz(self, name, arrays):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    def document(self, scope="single-gpu-cuda"):
        records = []
        for index, expected in enumerate(
            differential.expected_records(self.manifest, scope)
        ):
            dtype = (
                np.complex128
                if scope == "paired-real" and expected["device"] == "cpu"
                else np.complex64 if scope == "paired-real" else np.float32
            )
            reference = self.arrays(dtype)
            candidate = {name: value.copy() for name, value in reference.items()}
            tolerance = differential.frozen_tolerance(
                self.manifest, scope, expected["case"], expected["device"]
            )
            reference_path = self.write_npz(
                f"artifacts/{index}-reference.npz", reference
            )
            candidate_path = self.write_npz(
                f"artifacts/{index}-candidate.npz", candidate
            )
            maximum_abs, maximum_relative, passed = differential._compare_arrays(
                reference,
                candidate,
                tolerance["rtol"],
                tolerance["atol"],
            )
            records.append(
                {
                    **expected,
                    "reference": self.descriptor(reference_path),
                    "candidate": self.descriptor(candidate_path),
                    "field_arrays": list(differential.FIELD_ARRAYS),
                    "persistent_arrays": [
                        "persistent/state/indices",
                        "persistent/state/values",
                    ],
                    **tolerance,
                    "maximum_abs_error": maximum_abs,
                    "maximum_relative_error": maximum_relative,
                    "passed": passed,
                }
            )
        return {
            "schema_version": differential.INDEX_SCHEMA_VERSION,
            "kind": differential.INDEX_KIND,
            "scope": scope,
            "candidate_evidence": self.candidate,
            "required_cases": differential.expected_records(self.manifest, scope),
            "cases": records,
            "passed": True,
        }

    def write_index(self, document, name="index.json"):
        path = self.root / name
        path.write_text(
            json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def load(self, document):
        return differential.load_differential_evidence_index(
            self.write_index(document),
            self.manifest,
            self.candidate,
            descriptor_root=self.root,
            expected_scope=document["scope"],
        )

    def rewrite_candidate(self, document, index, arrays):
        path = self.root / document["cases"][index]["candidate"]["path"]
        path.unlink()
        np.savez_compressed(path, **arrays)
        document["cases"][index]["candidate"] = self.descriptor(path)

    def test_valid_index_recomputes_every_raw_projection(self):
        document = self.document()
        result = self.load(document)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["cases"]), 2)

        artifact = completion.LoadedArtifact(
            descriptor={"sha256": "0" * 64},
            path=self.root / "index.json",
            raw=b"",
            document=document,
        )
        completion._validate_differential(
            artifact,
            completion.ArtifactReader(self.root, self.candidate),
            self.manifest,
            self.candidate,
            scope="single-gpu-cuda",
        )

    def test_tolerance_is_manifest_frozen_not_artifact_selected(self):
        document = self.document()
        document["cases"][0]["rtol"] = 1.0
        document["cases"][0]["atol"] = 1.0
        with self.assertRaisesRegex(ValueError, "tolerance differs from the manifest"):
            self.load(document)

        artifact = completion.LoadedArtifact(
            descriptor={"sha256": "0" * 64},
            path=self.root / "index.json",
            raw=b"",
            document=document,
        )
        with self.assertRaisesRegex(
            completion.EvidenceError,
            "could not be independently recomputed",
        ):
            completion._validate_differential(
                artifact,
                completion.ArtifactReader(self.root, self.candidate),
                self.manifest,
                self.candidate,
                scope="single-gpu-cuda",
            )

    def test_descriptor_path_digest_and_candidate_are_exact(self):
        for mutation in ("path", "sha256", "candidate_evidence"):
            with self.subTest(mutation=mutation):
                document = self.document()
                descriptor = document["cases"][0]["candidate"]
                if mutation == "path":
                    descriptor[mutation] = "../escape.npz"
                elif mutation == "sha256":
                    descriptor[mutation] = "0" * 64
                else:
                    descriptor[mutation] = {
                        **self.candidate,
                        "candidate_git_commit": "b" * 40,
                    }
                with self.assertRaises(ValueError):
                    self.load(document)

    def test_missing_and_extra_npz_arrays_fail_closed(self):
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                document = self.document()
                arrays = self.arrays(np.float32)
                if mutation == "missing":
                    del arrays["Hz"]
                else:
                    arrays["persistent/unexpected"] = np.asarray(
                        [1.0], dtype=np.float32
                    )
                self.rewrite_candidate(document, 0, arrays)
                with self.assertRaisesRegex(ValueError, "NPZ array closure differs"):
                    self.load(document)

    def test_raw_numeric_failure_cannot_be_hidden_by_pass_flags(self):
        document = self.document()
        arrays = self.arrays(np.float32)
        arrays["Ex"] = arrays["Ex"] + np.float32(1.0)
        self.rewrite_candidate(document, 0, arrays)
        document["cases"][0]["maximum_abs_error"] = 1.0
        document["cases"][0]["maximum_relative_error"] = 1.0
        document["cases"][0]["passed"] = True
        document["passed"] = True
        with self.assertRaisesRegex(ValueError, "recomputed differential failed"):
            self.load(document)

    def test_integer_arrays_are_exact_even_under_float_tolerance(self):
        document = self.document()
        arrays = self.arrays(np.float32)
        arrays["persistent/state/indices"][0] += 1
        self.rewrite_candidate(document, 0, arrays)
        with self.assertRaisesRegex(ValueError, "recomputed differential failed"):
            self.load(document)

    def test_projection_paths_cannot_be_reused_across_cases(self):
        document = self.document()
        document["cases"][1]["reference"] = copy.deepcopy(
            document["cases"][0]["reference"]
        )
        with self.assertRaisesRegex(ValueError, "reuses an artifact path"):
            self.load(document)

    def test_paired_cpu_tolerance_uses_complex128_manifest_entry(self):
        tolerance = differential.frozen_tolerance(
            self.manifest,
            "paired-real",
            "dcp-ade-bloch",
            "cpu",
        )
        self.assertEqual(tolerance, {"rtol": 1e-11, "atol": 1e-12})

    def test_builder_output_round_trips_through_strict_loader(self):
        required = differential.expected_records(self.manifest, "single-gpu-cuda")
        source_directory = self.root / "sources"
        source_directory.mkdir()
        references = []
        candidates = []
        metadata = {}
        provenance = {
            "commit": self.candidate["candidate_git_commit"],
            "git_status": "",
            "clean": True,
        }
        for expected in required:
            case = expected["case"]
            reference = source_directory / f"{case}-reference.npz"
            candidate = source_directory / f"{case}-candidate.npz"
            reference.touch()
            candidate.touch()
            references.append(reference)
            candidates.append(candidate)
            workload = differential._workload(self.manifest, case)
            metadata[reference] = {"workload": workload}
            metadata[candidate] = {
                "workload": workload,
                "backend_metadata": {"device": "cuda:0"},
                "provenance": {
                    "source": provenance,
                    "controller": provenance,
                },
            }

        def archive_record(path, _manifest, _role):
            path = Path(path)
            return path, metadata[path]

        def projection(_reference, _candidate, _manifest, _scope, _expected):
            left = self.arrays(np.float32)
            right = {name: value.copy() for name, value in left.items()}
            return (
                left,
                right,
                self.candidate,
                ["persistent/state/indices", "persistent/state/values"],
            )

        with (
            mock.patch.object(
                differential.torch_correctness,
                "_archive_record",
                side_effect=archive_record,
            ),
            mock.patch.object(
                differential,
                "_projection_arrays",
                side_effect=projection,
            ),
        ):
            document = differential.build_differential_evidence(
                references,
                candidates,
                self.manifest,
                self.candidate,
                scope="single-gpu-cuda",
                descriptor_root=self.root,
                output_directory=self.root / "artifacts",
            )

        result = self.load(document)
        self.assertTrue(result["passed"])
        self.assertEqual(
            [record["case"] for record in result["cases"]],
            ["single-gpu-2d", "single-gpu-3d"],
        )

    def test_candidate_cli_writes_exact_three_key_binding(self):
        output = self.root / "candidate.json"
        argv = [
            "issue123_differential",
            "candidate",
            "--output",
            str(output),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                differential.host_contract,
                "candidate_evidence",
                return_value=self.candidate,
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(differential.main(), 0)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), self.candidate)


if __name__ == "__main__":
    unittest.main()
