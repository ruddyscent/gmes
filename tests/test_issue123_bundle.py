from __future__ import annotations

import copy
import hashlib
import io
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from benchmarks import issue123_completion as completion
from benchmarks import torch_correctness


class Issue123BundleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.manifest = self.directory / "manifest.json"
        self.manifest.write_bytes(b'{"reference": {}}\n')
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        }
        self.payload = self.directory / "input.json"
        self.payload.write_bytes(b'{"value": 1}\n')

    def specification(
        self,
        *,
        source_path: str | None = None,
        bundle_path: str = "evidence/input.json",
        media_type: str = completion.MEDIA_TYPE_JSON,
    ) -> Path:
        value = {
            "schema_version": completion.BUNDLE_SPEC_SCHEMA_VERSION,
            "kind": completion.BUNDLE_SPEC_KIND,
            "issue": 123,
            "candidate_evidence": self.candidate,
            "payloads": [
                {
                    "source_path": (
                        self.payload.name if source_path is None else source_path
                    ),
                    "bundle_path": bundle_path,
                    "media_type": media_type,
                }
            ],
            "artifacts": {
                "cpu": {"aggregate": bundle_path},
                "policy_paired_real": {},
                "single_gpu": {},
                "two_gpu": {},
                "macos": {},
                "operations": {},
            },
        }
        path = self.directory / "bundle-spec.json"
        path.write_text(
            json.dumps(value, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def assemble(self, name: str = "bundle") -> Path:
        return completion.assemble_evidence_bundle(
            self.specification(),
            self.directory / name,
            self.manifest,
        )

    def test_assembly_is_deterministic_and_relocatable(self):
        first_index = self.assemble("first")
        second_index = self.assemble("second")
        self.assertEqual(first_index.read_bytes(), second_index.read_bytes())

        relocated = self.directory / "relocated"
        first_index.parent.rename(relocated)
        index = completion._strict_json_bytes(
            (relocated / "completion-index.json").read_bytes(),
            "relocated index",
        )
        reader = completion.ArtifactReader(
            relocated,
            index["candidate_evidence"],
            index["payloads"],
        )
        loaded = reader.load(index["artifacts"]["cpu"]["aggregate"], "payload")
        self.assertEqual(loaded.document, {"value": 1})

    def test_assembly_rejects_noncanonical_bundle_paths(self):
        for index, bundle_path in enumerate(
            (
                "/absolute.json",
                "../escape.json",
                "a/./payload.json",
                "a//payload.json",
                "a\\payload.json",
                "C:/payload.json",
            )
        ):
            with self.subTest(bundle_path=bundle_path):
                spec = self.specification(bundle_path=bundle_path)
                with self.assertRaises(completion.EvidenceError):
                    completion.assemble_evidence_bundle(
                        spec,
                        self.directory / f"invalid-{index}",
                        self.manifest,
                    )

    def test_assembly_rejects_payload_symlinks(self):
        target = self.directory / "real.json"
        target.write_bytes(b"{}\n")
        link = self.directory / "linked.json"
        link.symlink_to(target)
        spec = self.specification(source_path=link.name)
        with self.assertRaisesRegex(completion.EvidenceError, "symlink"):
            completion.assemble_evidence_bundle(
                spec,
                self.directory / "symlink-bundle",
                self.manifest,
            )

    def test_assembly_rejects_symlinked_manifest_and_output_parent(self):
        manifest_link = self.directory / "manifest-link.json"
        manifest_link.symlink_to(self.manifest)
        with self.assertRaisesRegex(completion.EvidenceError, "symlink"):
            completion.assemble_evidence_bundle(
                self.specification(),
                self.directory / "manifest-link-bundle",
                manifest_link,
            )

        real_parent = self.directory / "real-parent"
        real_parent.mkdir()
        parent_link = self.directory / "parent-link"
        parent_link.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(completion.EvidenceError, "symlink"):
            completion.assemble_evidence_bundle(
                self.specification(),
                parent_link / "bundle",
                self.manifest,
            )
        self.assertEqual(list(real_parent.iterdir()), [])

    def test_path_audit_allows_only_configured_darwin_system_aliases(self):
        target = self.directory / "real-system-path"
        target.mkdir()
        payload = target / "payload.json"
        payload.write_bytes(b"{}\n")
        alias = self.directory / "system-alias"
        alias.symlink_to(target, target_is_directory=True)

        with (
            mock.patch.object(completion.platform, "system", return_value="Darwin"),
            mock.patch.dict(
                completion.DARWIN_SYSTEM_PATH_ALIASES,
                {str(alias): str(target)},
            ),
        ):
            checked, metadata = completion._path_without_symlinks(
                alias / payload.name, "configured alias"
            )
            output = completion._ensure_directory_without_symlinks(
                alias / "output", "configured alias output"
            )
        self.assertEqual(checked, payload.resolve(strict=True))
        self.assertTrue(stat.S_ISREG(metadata.st_mode))
        self.assertEqual(output, (target / "output").resolve(strict=True))

        with (
            mock.patch.object(completion.platform, "system", return_value="Darwin"),
            self.assertRaisesRegex(completion.EvidenceError, "symlink"),
        ):
            completion._path_without_symlinks(alias / payload.name, "unlisted alias")

    def test_assembly_never_overwrites_existing_or_dangling_symlink_destination(self):
        existing = self.directory / "existing"
        existing.mkdir()
        marker = existing / "marker"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(completion.EvidenceError, "already exists"):
            completion.assemble_evidence_bundle(
                self.specification(),
                existing,
                self.manifest,
            )
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

        dangling = self.directory / "dangling"
        dangling.symlink_to(self.directory / "absent", target_is_directory=True)
        with self.assertRaisesRegex(completion.EvidenceError, "already exists"):
            completion.assemble_evidence_bundle(
                self.specification(),
                dangling,
                self.manifest,
            )
        self.assertTrue(dangling.is_symlink())

    def test_reader_rejects_symlinked_artifact_after_relocation(self):
        index_path = self.assemble()
        index = completion._strict_json_bytes(index_path.read_bytes(), "index")
        descriptor = index["artifacts"]["cpu"]["aggregate"]
        payload_path = index_path.parent / descriptor["path"]
        external = self.directory / "external.json"
        external.write_bytes(payload_path.read_bytes())
        payload_path.unlink()
        payload_path.symlink_to(external)
        reader = completion.ArtifactReader(
            index_path.parent,
            index["candidate_evidence"],
            index["payloads"],
        )
        with self.assertRaisesRegex(completion.EvidenceError, "symlink"):
            reader.load(descriptor, "payload")

    def zip_bytes(self, members):
        stream = io.BytesIO()
        with zipfile.ZipFile(
            stream,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name, raw in members:
                archive.writestr(name, raw)
        return stream.getvalue()

    def test_zip_preflight_rejects_escape_duplicate_symlink_and_expansion_caps(self):
        with self.assertRaises(completion.EvidenceError):
            completion._preflight_zip(
                self.zip_bytes([("../escape", b"x")]),
                "escape",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            duplicate = self.zip_bytes([("same", b"x"), ("same", b"y")])
        with self.assertRaisesRegex(completion.EvidenceError, "repeats ZIP member"):
            completion._preflight_zip(duplicate, "duplicate")

        stream = io.BytesIO()
        with zipfile.ZipFile(stream, mode="w") as archive:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "target")
        with self.assertRaisesRegex(completion.EvidenceError, "symbolic link"):
            completion._preflight_zip(stream.getvalue(), "symlink")

        compressed = self.zip_bytes([("large", b"0" * 1024)])
        with self.assertRaisesRegex(completion.EvidenceError, "ZIP byte bound"):
            completion._preflight_zip(
                compressed,
                "bounded",
                max_total_bytes=32,
            )
        with mock.patch.object(completion, "MAX_ZIP_COMPRESSION_RATIO", 2.0):
            with self.assertRaisesRegex(completion.EvidenceError, "compression-ratio"):
                completion._preflight_zip(compressed, "ratio")

    def npz_artifact(self, arrays) -> completion.LoadedArtifact:
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        raw = stream.getvalue()
        return completion.LoadedArtifact(
            {
                "path": "arrays.npz",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "media_type": completion.MEDIA_TYPE_NPZ,
                "candidate_evidence": self.candidate,
            },
            self.directory / "arrays.npz",
            raw,
        )

    def test_npz_loader_enforces_closure_numeric_dtype_and_bounds(self):
        numeric = self.npz_artifact({"x": np.arange(4, dtype=np.float64)})
        arrays = completion._npz_arrays(numeric, ["x"], "numeric")
        np.testing.assert_array_equal(arrays["x"], np.arange(4, dtype=np.float64))
        with self.assertRaisesRegex(completion.EvidenceError, "closure"):
            completion._npz_arrays(numeric, ["x", "y"], "missing")

        object_array = self.npz_artifact(
            {"x": np.asarray([{"untrusted": True}], dtype=object)}
        )
        with self.assertRaises(completion.EvidenceError):
            completion._npz_arrays(object_array, ["x"], "object")

        structured = self.npz_artifact(
            {"x": np.asarray([(1, 2)], dtype=[("left", "i4"), ("right", "i4")])}
        )
        with self.assertRaisesRegex(completion.EvidenceError, "plain numeric"):
            completion._npz_arrays(structured, ["x"], "structured")

        with mock.patch.object(completion, "MAX_NPZ_ARRAY_BYTES", 1):
            with self.assertRaisesRegex(completion.EvidenceError, "bound"):
                completion._npz_arrays(numeric, ["x"], "bounded")

    def test_relocated_bundle_preloads_every_nested_correctness_npz(self):
        manifest_raw = completion.DEFAULT_MANIFEST.read_bytes()
        manifest = json.loads(manifest_raw)
        candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
        required = [
            case["name"]
            for group in ("correctness", "physical_checks")
            for case in manifest[group]
        ]
        sources = self.directory / "correctness-sources"
        sources.mkdir()
        payloads = []
        artifacts = []
        nested_paths = []
        for index, name in enumerate(required):
            record = {"case": name}
            for role in ("reference", "candidate"):
                source = sources / f"{index:02d}-{role}.npz"
                np.savez_compressed(source, value=np.asarray([index], dtype=np.int64))
                raw = source.read_bytes()
                bundle_path = f"correctness/{source.name}"
                descriptor = {
                    "path": bundle_path,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                    "media_type": completion.MEDIA_TYPE_NPZ,
                    "candidate_evidence": candidate,
                }
                record[role] = descriptor
                payloads.append(
                    {
                        "source_path": source.relative_to(self.directory).as_posix(),
                        "bundle_path": bundle_path,
                        "media_type": completion.MEDIA_TYPE_NPZ,
                    }
                )
                nested_paths.append(bundle_path)
            artifacts.append(record)
        document = {
            "schema_version": 1,
            "kind": "torch-correctness-evidence-index",
            "candidate_evidence": candidate,
            "manifest_contract_sha256": completion._canonical_sha256(manifest),
            "required_cases": required,
            "artifacts": artifacts,
            "suite_acceptance": {
                "correctness_case_count": len(manifest["correctness"]),
                "physical_check_case_count": len(manifest["physical_checks"]),
                "evaluated_case_count": len(required),
                "complete_fields": True,
                "persistent_state": True,
                "source_and_auxiliary_state": True,
                "physical_observables": True,
                "passed": True,
            },
        }
        source_index = self.directory / "correctness-index.json"
        source_index.write_bytes(completion._canonical_json_bytes(document))
        payloads.append(
            {
                "source_path": source_index.name,
                "bundle_path": "correctness/index.json",
                "media_type": completion.MEDIA_TYPE_JSON,
            }
        )
        specification = {
            "schema_version": completion.BUNDLE_SPEC_SCHEMA_VERSION,
            "kind": completion.BUNDLE_SPEC_KIND,
            "issue": 123,
            "candidate_evidence": candidate,
            "payloads": payloads,
            "artifacts": {
                "cpu": {"correctness_index": "correctness/index.json"},
                "policy_paired_real": {},
                "single_gpu": {},
                "two_gpu": {},
                "macos": {},
                "operations": {},
            },
        }
        specification_path = self.directory / "correctness-bundle-spec.json"
        specification_path.write_text(json.dumps(specification))
        top_path = completion.assemble_evidence_bundle(
            specification_path,
            self.directory / "correctness-bundle",
            completion.DEFAULT_MANIFEST,
        )
        relocated = self.directory / "correctness-relocated"
        top_path.parent.rename(relocated)
        top = completion._strict_json_bytes(
            (relocated / "completion-index.json").read_bytes(), "top index"
        )
        reader = completion.ArtifactReader(relocated, candidate, top["payloads"])
        reader.load(top["manifest"], "manifest")
        index_artifact = reader.load(
            top["artifacts"]["cpu"]["correctness_index"], "correctness index"
        )
        rebuilt = {"source_artifact": {"sha256": index_artifact.descriptor["sha256"]}}
        with mock.patch.object(
            torch_correctness,
            "load_correctness_evidence_index",
            return_value=rebuilt,
        ):
            self.assertIs(
                completion._validate_correctness_index(
                    index_artifact, manifest, candidate, reader
                ),
                rebuilt,
            )
        consumed = {path.relative_to(reader.base).as_posix() for path in reader._seen}
        self.assertTrue(set(nested_paths).issubset(consumed))

    def test_evaluator_returns_structured_false_for_untrusted_index(self):
        index_path = self.assemble()
        value = completion._strict_json_bytes(index_path.read_bytes(), "index")
        value["manifest"] = copy.deepcopy(value["manifest"])
        value["manifest"]["path"] = "../manifest.json"
        index_path.write_bytes(completion._canonical_json_bytes(value))
        result = completion.evaluate_completion(index_path)
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertEqual(len(result["cross_scope_errors"]), 1)
        error = result["cross_scope_errors"][0]
        self.assertEqual(error["code"], "invalid-evidence")
        self.assertEqual(error["phase"], "bundle-index")
        self.assertIsNone(error["scope"])
        self.assertIn("dot segment", error["message"])

    def test_evaluator_preflights_index_size_before_reading_json(self):
        index_path = self.assemble()
        with mock.patch.object(completion, "MAX_INDEX_BYTES", 1):
            result = completion.evaluate_completion(index_path)
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertEqual(result["cross_scope_errors"][0]["phase"], "bundle-index")
        self.assertIn("byte bound", result["cross_scope_errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()
