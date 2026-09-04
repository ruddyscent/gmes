from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from benchmarks import issue123_completion as completion


class TwoGpuRawCorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": "b" * 64,
        }

    def descriptor(self, path: Path):
        raw = path.read_bytes()
        return {
            "path": path.relative_to(self.directory).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "media_type": completion.MEDIA_TYPE_NPZ,
            "candidate_evidence": self.candidate,
        }

    def write_npz(self, relative: str, arrays: dict[str, np.ndarray]):
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        path.write_bytes(buffer.getvalue())
        return self.descriptor(path)

    @staticmethod
    def storage_summary(names, initial, final, *, rank=None):
        def digest(values):
            hasher = hashlib.sha256()
            hasher.update(json.dumps(names, separators=(",", ":")).encode())
            hasher.update(values.tobytes(order="C"))
            return hasher.hexdigest()

        result = {
            "address_names": names,
            "address_count": len(names),
            "initial_sha256": digest(initial),
            "final_sha256": digest(final),
            "addresses_stable": bool(np.array_equal(initial, final)),
        }
        if rank is not None:
            result["rank"] = rank
        return result

    def case_record(self, case: str):
        whole_shape = completion._two_gpu_whole_shape(case)
        field_shapes = completion._two_gpu_field_shapes(whole_shape)
        dtype = np.complex128 if case.endswith("-bloch") else np.float64
        arrays = {}
        for step in completion.TWO_GPU_CAPTURE_STEPS:
            value = step * 1e-3
            for field in completion.FIELD_ARRAYS:
                for role in ("distributed", "serial"):
                    arrays[f"capture/{step}/{role}/{field}"] = np.full(
                        field_shapes[field], value, dtype=dtype
                    )
        for phase in ("expected", "replay", "serial"):
            for field in completion.FIELD_ARRAYS:
                arrays[f"checkpoint/{phase}/{field}"] = np.full(
                    field_shapes[field], 0.2, dtype=dtype
                )
        storage_names = ["state.Ex", "state.Hy"]
        rank_storage = []
        for rank, values in ((0, (100, 200)), (1, (300, 400))):
            initial = np.asarray(values, dtype=np.uint64)
            final = initial.copy()
            arrays[f"storage/rank/{rank}/initial"] = initial
            arrays[f"storage/rank/{rank}/final"] = final
            rank_storage.append(
                self.storage_summary(storage_names, initial, final, rank=rank)
            )
        serial_initial = np.asarray((500, 600), dtype=np.uint64)
        serial_final = serial_initial.copy()
        arrays["storage/serial/initial"] = serial_initial
        arrays["storage/serial/final"] = serial_final
        descriptor = self.write_npz(f"raw/{case}.npz", arrays)
        expected_axis = int(case[5]) if case.startswith("axis-") else 0
        return (
            {
                "name": case,
                "axis": expected_axis,
                "cut": 2,
                "capture_errors": {
                    str(step): 0.0 for step in completion.TWO_GPU_CAPTURE_STEPS
                },
                "checkpoint_determinism_error": 0.0,
                "checkpoint_reference_error": 0.0,
                "rank0_probe_count": 0,
                "checkpoint_replay_steps": 5,
                "checkpoint_replay_fields": list(completion.FIELD_ARRAYS),
                "rank_storage": rank_storage,
                "serial_storage": self.storage_summary(
                    storage_names, serial_initial, serial_final
                ),
                "raw_evidence": {
                    "artifact": descriptor,
                    "array_names": completion._two_gpu_case_raw_names(),
                    "field_shapes": field_shapes,
                },
            },
            arrays,
        )

    def document(self):
        cases = []
        raw_arrays = {}
        for name in completion.TWO_GPU_CORRECTNESS_CASES:
            record, arrays = self.case_record(name)
            cases.append(record)
            raw_arrays[name] = arrays
        field_shapes = completion._two_gpu_field_shapes((16, 12, 8))
        long_arrays = {}
        for phase, value in (
            ("initial", 1.0),
            ("distributed", 0.5),
            ("serial", 0.5),
        ):
            for field in completion.FIELD_ARRAYS:
                long_arrays[f"{phase}/{field}"] = np.full(
                    field_shapes[field], value, dtype=np.float64
                )
        long_descriptor = self.write_npz("raw/long.npz", long_arrays)
        initial_energy = sum(
            float(np.square(long_arrays[f"initial/{field}"]).sum())
            for field in completion.FIELD_ARRAYS
        )
        final_energy = sum(
            float(np.square(long_arrays[f"distributed/{field}"]).sum())
            for field in completion.FIELD_ARRAYS
        )
        checks = {
            "environment_complete": True,
            "case_closure_complete": True,
            "capture_steps_complete": True,
            "complete_field_replay": True,
            "rank_storage_stable": True,
            "raw_full_fields_bound": True,
            "long_stability_complete": True,
            "numerical_acceptance": True,
        }
        return (
            {
                "candidate_evidence": self.candidate,
                "environment": {},
                "schema_version": 3,
                "contract_id": "two-gpu-full-field-replay-v3",
                "capture_steps": list(completion.TWO_GPU_CAPTURE_STEPS),
                "capture_graphs": False,
                "execution_mode": "eager",
                "maximum_error": 0.0,
                "passed": True,
                "cases": cases,
                "long_stability": {
                    "steps": 1000,
                    "maximum_error": 0.0,
                    "finite": True,
                    "initial_energy": initial_energy,
                    "final_energy": final_energy,
                    "energy_ratio": final_energy / initial_energy,
                    "raw_evidence": {
                        "artifact": long_descriptor,
                        "array_names": completion._two_gpu_long_raw_names(),
                        "field_shapes": field_shapes,
                    },
                },
                "suite_acceptance": {
                    "required_cases": list(completion.TWO_GPU_CORRECTNESS_CASES),
                    "required_capture_steps": list(completion.TWO_GPU_CAPTURE_STEPS),
                    "required_long_steps": 1000,
                    "checks": checks,
                    "passed": True,
                },
            },
            raw_arrays,
            long_arrays,
        )

    def validate(self, document):
        artifact = completion.LoadedArtifact(
            {}, self.directory / "correctness.json", b"", document
        )
        reader = completion.ArtifactReader(self.directory, self.candidate)
        with mock.patch.object(
            completion,
            "_gpu_environment",
            return_value={"validated": True},
        ):
            result = completion._validate_two_gpu_correctness_v3(
                artifact, reader, self.candidate
            )
        return result, reader

    def test_full_raw_fields_and_storage_are_recomputed(self):
        document, _case_arrays, _long_arrays = self.document()

        result, reader = self.validate(document)

        self.assertEqual(result, (False, {"validated": True}))
        self.assertEqual(len(reader._seen), 19)

    def test_embedded_scalars_cannot_hide_raw_field_tampering(self):
        document, case_arrays, _long_arrays = self.document()
        case = completion.TWO_GPU_CORRECTNESS_CASES[0]
        arrays = copy.deepcopy(case_arrays[case])
        arrays["capture/1/distributed/Ex"] = arrays["capture/1/distributed/Ex"] + 1e-3
        document["cases"][0]["raw_evidence"]["artifact"] = self.write_npz(
            f"raw/{case}.npz", arrays
        )

        with self.assertRaisesRegex(completion.EvidenceError, "raw evidence"):
            self.validate(document)

    def test_storage_digest_and_long_field_tampering_fail_closed(self):
        document, case_arrays, _long_arrays = self.document()
        case = completion.TWO_GPU_CORRECTNESS_CASES[0]
        arrays = copy.deepcopy(case_arrays[case])
        arrays["storage/rank/0/final"] = np.asarray((100, 201), dtype=np.uint64)
        document["cases"][0]["raw_evidence"]["artifact"] = self.write_npz(
            f"raw/{case}.npz", arrays
        )
        with self.assertRaisesRegex(completion.EvidenceError, "storage digest"):
            self.validate(document)

        document, _case_arrays, long_arrays = self.document()
        long_arrays = copy.deepcopy(long_arrays)
        long_arrays["distributed/Ex"] = long_arrays["distributed/Ex"] + 1e-3
        document["long_stability"]["raw_evidence"]["artifact"] = self.write_npz(
            "raw/long.npz", long_arrays
        )
        with self.assertRaisesRegex(completion.EvidenceError, "raw evidence"):
            self.validate(document)

    def test_missing_extra_and_reused_raw_archives_fail_closed(self):
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                document, case_arrays, _long_arrays = self.document()
                case = completion.TWO_GPU_CORRECTNESS_CASES[0]
                arrays = copy.deepcopy(case_arrays[case])
                if mutation == "missing":
                    arrays.pop("capture/1/distributed/Ex")
                else:
                    arrays["unexpected"] = np.zeros(1, dtype=np.float64)
                document["cases"][0]["raw_evidence"]["artifact"] = self.write_npz(
                    f"raw/{case}.npz", arrays
                )
                with self.assertRaises(completion.EvidenceError):
                    self.validate(document)

        document, _case_arrays, _long_arrays = self.document()
        document["cases"][2]["raw_evidence"]["artifact"] = copy.deepcopy(
            document["cases"][0]["raw_evidence"]["artifact"]
        )
        with self.assertRaisesRegex(completion.EvidenceError, "reuses"):
            self.validate(document)


if __name__ == "__main__":
    unittest.main()
