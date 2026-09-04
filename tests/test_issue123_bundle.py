from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from benchmarks import issue123_completion as completion
from benchmarks import issue123_privacy as privacy
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

    @staticmethod
    def _bundle_descriptor(path, root, candidate):
        raw = path.read_bytes()
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "media_type": completion.MEDIA_TYPE_JSON,
            "candidate_evidence": candidate,
        }

    def _authority_context(self, runtime_records, scope_artifacts):
        technical = {
            "candidate_evidence": self.candidate,
            "policy_sha256": "1" * 64,
            "scope_order": [
                "cpu",
                "policy_paired_real",
                "single_gpu",
                "two_gpu",
                "macos",
            ],
            "scope_artifacts": scope_artifacts,
            "runtime_receipts": runtime_records,
            "sources": [],
        }
        return {
            "candidate_evidence": self.candidate,
            "policy_sha256": "1" * 64,
            "source_specification_sha256": "2" * 64,
            "technical_inventory": technical,
            "technical_input_root": privacy.tagged_canonical_sha256(
                privacy.TECHNICAL_INPUT_INVENTORY_DOMAIN,
                technical,
            ),
            "public_projection_sha256": "3" * 64,
            "public_asset_ledger": [],
            "public_asset_ledger_sha256": privacy.tagged_canonical_sha256(
                privacy.PUBLIC_ASSET_LEDGER_DOMAIN,
                [],
            ),
        }

    def _write_authority_bundle(
        self,
        name,
        *,
        checked,
        updated_at,
        runtime_raw_by_role,
        issue_document=None,
    ):
        root = self.directory / name
        root.mkdir()
        payloads = []
        manifest_path = root / "manifest.json"
        manifest_path.write_bytes(self.manifest.read_bytes())
        manifest_descriptor = self._bundle_descriptor(
            manifest_path, root, self.candidate
        )
        payloads.append(manifest_descriptor)
        runtime_records = []
        for ordinal, role in enumerate(completion.RUNTIME_RECEIPT_ROLES):
            path = root / "runtime" / f"{ordinal}-{role}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(runtime_raw_by_role[role])
            descriptor = self._bundle_descriptor(path, root, self.candidate)
            payloads.append(descriptor)
            runtime_records.append(
                {
                    "role": role,
                    "bundle_path": descriptor["path"],
                    "sha256": descriptor["sha256"],
                    "size_bytes": descriptor["size_bytes"],
                }
            )
        mark = "x" if checked else " "
        issue = (
            {
                "body": (
                    "## Implementation work\n"
                    f"- [{mark}] publish the final bundle\n"
                    f"- [{mark}] complete the post-bundle checklist\n"
                ),
                "updated_at": updated_at,
            }
            if issue_document is None
            else copy.deepcopy(issue_document)
        )
        issue_path = root / "operations" / "issue-123.json"
        issue_path.parent.mkdir(exist_ok=True)
        issue_path.write_bytes(completion._compact_canonical_json_bytes(issue))
        issue_descriptor = self._bundle_descriptor(issue_path, root, self.candidate)
        payloads.append(issue_descriptor)
        operations_document = {
            "schema_version": 2,
            "responses": {
                "issue_123": {
                    "request": {},
                    "artifact": issue_descriptor,
                }
            },
        }
        operations_path = root / "operations" / "operations-index.json"
        operations_path.write_bytes(
            completion._canonical_json_bytes(operations_document)
        )
        operations_descriptor = self._bundle_descriptor(
            operations_path, root, self.candidate
        )
        payloads.append(operations_descriptor)
        scope_artifacts = {}
        for ordinal, scope in enumerate(
            ("cpu", "policy_paired_real", "single_gpu", "two_gpu", "macos")
        ):
            path = root / "technical" / f"{ordinal}-{scope}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(
                completion._compact_canonical_json_bytes(
                    {"scope": scope, "synthetic_fixture": True}
                )
            )
            descriptor = self._bundle_descriptor(path, root, self.candidate)
            payloads.append(descriptor)
            scope_artifacts[scope] = {"synthetic_fixture": descriptor}
        artifacts = {
            **scope_artifacts,
            "operations": {"index": operations_descriptor},
        }
        payloads.sort(key=lambda item: item["path"])
        index = {
            "schema_version": completion.INDEX_SCHEMA_VERSION,
            "kind": completion.INDEX_KIND,
            "issue": 123,
            "bundle": {
                "format": completion.BUNDLE_FORMAT,
                "path_contract": completion.PATH_CONTRACT,
                "artifact_count": len(payloads),
                "artifact_bytes": sum(item["size_bytes"] for item in payloads),
            },
            "candidate_evidence": self.candidate,
            "manifest": manifest_descriptor,
            "payloads": payloads,
            "artifacts": artifacts,
        }
        index_path = root / "completion-index.json"
        index_path.write_bytes(completion._canonical_json_bytes(index))
        return index_path, issue_path, runtime_records, scope_artifacts

    def _live_capture_fixture(self, name):
        runtime_raw_by_role = {
            role: completion._compact_canonical_json_bytes(
                {"ordinal": ordinal, "role": role}
            )
            for ordinal, role in enumerate(completion.RUNTIME_RECEIPT_ROLES)
        }
        source, _issue, runtime_records, scope_artifacts = self._write_authority_bundle(
            f"{name}-source",
            checked=True,
            updated_at="2026-01-02T00:00:00Z",
            runtime_raw_by_role=runtime_raw_by_role,
        )
        reopened_root = self.directory / f"{name}-reopened"
        shutil.copytree(source.parent, reopened_root)
        reopened = reopened_root / "completion-index.json"
        authority = self.directory / f"{name}-authority"
        authority.mkdir(mode=0o700)
        openings = privacy.PrivateOpenings(bytes(range(32)))
        openings._populated = True
        context = self._authority_context(runtime_records, scope_artifacts)
        openings_path = authority / "openings.json"
        openings_path.write_bytes(privacy.serialize_private_openings(openings, context))
        openings_path.chmod(0o600)
        detached_receipts = []
        for leaf in ("b0-reopen.json", "b1-reopen.json"):
            path = authority / leaf
            path.write_bytes(completion._compact_canonical_json_bytes({"leaf": leaf}))
            path.chmod(0o600)
            detached_receipts.append(path)
        external_runtime = []
        for ordinal, role in enumerate(completion.RUNTIME_RECEIPT_ROLES):
            path = authority / f"external-{ordinal}-{role}.json"
            path.write_bytes(runtime_raw_by_role[role])
            path.chmod(0o600)
            external_runtime.append(path)
        return {
            "source": source,
            "reopened": reopened,
            "openings": openings_path,
            "pre_ack": detached_receipts[0],
            "final": detached_receipts[1],
            "runtime": external_runtime,
        }

    def _capture_live_fixture(self, name):
        inputs = self._live_capture_fixture(name)
        snapshots = completion._capture_core_live_inputs(
            inputs["source"],
            inputs["reopened"],
            self.manifest,
            inputs["runtime"],
            inputs["openings"],
            inputs["pre_ack"],
            inputs["final"],
        )
        return inputs, snapshots

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

    def test_two_bundle_reopen_chain_is_finite_authenticated_and_fail_closed(self):
        runtime_raw_by_role = {
            role: completion._compact_canonical_json_bytes(
                {"ordinal": ordinal, "role": role}
            )
            for ordinal, role in enumerate(completion.RUNTIME_RECEIPT_ROLES)
        }
        (
            b0_source,
            b0_issue,
            runtime_records,
            frozen_scope_artifacts,
        ) = self._write_authority_bundle(
            "b0-source",
            checked=False,
            updated_at="2026-01-01T00:00:00Z",
            runtime_raw_by_role=runtime_raw_by_role,
        )
        b0_reopened_root = self.directory / "b0-reopened"
        shutil.copytree(b0_source.parent, b0_reopened_root)
        b0_reopened = b0_reopened_root / "completion-index.json"
        authority = self.directory / "authority"
        authority.mkdir(mode=0o700)
        openings = privacy.PrivateOpenings(bytes(range(32)))
        openings._populated = True
        context = self._authority_context(runtime_records, frozen_scope_artifacts)
        openings_path = authority / "openings.json"
        openings_path.write_bytes(privacy.serialize_private_openings(openings, context))
        openings_path.chmod(0o600)
        b0_receipt_path = authority / "b0-reopen.json"
        b0_raw = completion.record_bundle_reopen(
            source_index=b0_source,
            reopened_index=b0_reopened,
            stage="pre-acknowledgment",
            protected_openings=openings_path,
            pre_ack_response=b0_reopened_root / b0_issue.relative_to(b0_source.parent),
            output=b0_receipt_path,
        )
        b0_receipt = json.loads(b0_raw)
        self.assertEqual(b0_receipt["issue_response"]["state"], "unchecked")
        self.assertEqual(stat.S_IMODE(b0_receipt_path.stat().st_mode), 0o600)

        (
            b1_source,
            _b1_issue,
            b1_runtime_records,
            b1_scope_artifacts,
        ) = self._write_authority_bundle(
            "b1-source",
            checked=True,
            updated_at=b0_receipt["observed_at"],
            runtime_raw_by_role=runtime_raw_by_role,
        )
        self.assertEqual(runtime_records, b1_runtime_records)
        self.assertEqual(frozen_scope_artifacts, b1_scope_artifacts)
        b1_reopened_root = self.directory / "b1-reopened"
        shutil.copytree(b1_source.parent, b1_reopened_root)
        b1_reopened = b1_reopened_root / "completion-index.json"
        b1_receipt_path = authority / "b1-reopen.json"
        b1_raw = completion.record_bundle_reopen(
            source_index=b1_source,
            reopened_index=b1_reopened,
            stage="final",
            protected_openings=openings_path,
            pre_ack_receipt=b0_receipt_path,
            output=b1_receipt_path,
        )
        b1_receipt = json.loads(b1_raw)
        self.assertEqual(b1_receipt["issue_response"]["state"], "checked")
        self.assertEqual(
            b1_receipt["pre_acknowledgment_receipt_sha256"],
            hashlib.sha256(b0_raw).hexdigest(),
        )
        self.assertNotEqual(
            b0_receipt["source_bundle"]["inventory_root"],
            b1_receipt["source_bundle"]["inventory_root"],
        )
        self.assertEqual(
            completion.completion_bundle_inventory(b1_source, openings_path),
            completion.completion_bundle_inventory(b1_reopened, openings_path),
        )

        external_runtime = []
        for ordinal, role in enumerate(completion.RUNTIME_RECEIPT_ROLES):
            path = authority / f"external-{ordinal}-{role}.json"
            path.write_bytes(runtime_raw_by_role[role])
            path.chmod(0o600)
            external_runtime.append(path)
        snapshots = completion._capture_core_live_inputs(
            b1_source,
            b1_reopened,
            self.manifest,
            external_runtime,
            openings_path,
            b0_receipt_path,
            b1_receipt_path,
        )
        try:
            chain = completion._validate_final_bundle_reopen_chain(snapshots)
            expectation = chain["post_bundle_expectation"]
            self.assertEqual(
                expectation.o0_canonical_response_sha256,
                b0_receipt["issue_response"]["canonical_response_sha256"],
            )
            self.assertEqual(
                expectation.o1_canonical_response_sha256,
                b1_receipt["issue_response"]["canonical_response_sha256"],
            )
        finally:
            completion._close_retained_bundle_trees(
                (snapshots.reopened_bundle, snapshots.source_bundle)
            )

        legacy = copy.deepcopy(b1_receipt)
        legacy["schema_version"] = 0
        with self.assertRaisesRegex(completion.EvidenceError, "identity or stage"):
            completion.validate_bundle_reopen_receipt(
                privacy.binding_canonical_json_bytes(legacy),
                "final",
                openings_path,
            )
        with self.assertRaisesRegex(completion.EvidenceError, "stage"):
            completion.validate_bundle_reopen_receipt(
                b0_raw,
                "final",
                openings_path,
            )

        stale_source, _stale_issue, _records, _scopes = self._write_authority_bundle(
            "stale-b1-source",
            checked=True,
            updated_at="2026-01-01T00:00:00Z",
            runtime_raw_by_role=runtime_raw_by_role,
        )
        stale_reopened_root = self.directory / "stale-b1-reopened"
        shutil.copytree(stale_source.parent, stale_reopened_root)
        with self.assertRaisesRegex(completion.EvidenceError, "does not follow"):
            completion.record_bundle_reopen(
                source_index=stale_source,
                reopened_index=stale_reopened_root / "completion-index.json",
                stage="final",
                protected_openings=openings_path,
                pre_ack_receipt=b0_receipt_path,
            )

        (b1_reopened_root / "unexpected.bin").write_bytes(b"unexpected")
        with self.assertRaisesRegex(completion.EvidenceError, "missing or extra"):
            completion.completion_bundle_inventory(b1_reopened, openings_path)

    def test_retained_source_and_reopened_b1_detect_every_file_and_tree_mutation(self):
        attacks = (
            "append",
            "same-size-rewrite",
            "inode-replacement",
            "extra-file",
            "extra-directory",
        )
        for copy_name in ("source", "reopened"):
            for attack in attacks:
                suffix = f"retained-{copy_name}-{attack}"
                inputs, snapshots = self._capture_live_fixture(suffix)
                try:
                    root = inputs[copy_name].parent
                    target = root / "technical" / "0-cpu.json"
                    raw = target.read_bytes()
                    if attack == "append":
                        target.write_bytes(raw + b"x")
                    elif attack == "same-size-rewrite":
                        changed = bytearray(raw)
                        changed[0] ^= 1
                        target.write_bytes(bytes(changed))
                    elif attack == "inode-replacement":
                        replacement = root / "technical" / "replacement.json"
                        replacement.write_bytes(raw)
                        replacement.replace(target)
                    elif attack == "extra-file":
                        (root / "unexpected.bin").write_bytes(b"unexpected")
                    else:
                        (root / "unexpected-empty-directory").mkdir()
                    with (
                        self.subTest(copy=copy_name, attack=attack),
                        self.assertRaises(completion.EvidenceError),
                    ):
                        completion._require_core_live_inputs_unchanged(snapshots)
                finally:
                    completion._close_retained_bundle_trees(
                        (snapshots.reopened_bundle, snapshots.source_bundle)
                    )

    def test_core_capture_attempts_both_tree_closes_after_first_close_failure(self):
        inputs = self._live_capture_fixture("close-failure")
        real_close = completion._close_retained_bundle_tree
        closed_roots = []

        def close_then_fail_first(tree):
            closed_roots.append(tree.root)
            real_close(tree)
            if len(closed_roots) == 1:
                raise completion.EvidenceError(
                    "live B1 retained descriptors could not be closed"
                )

        primary = completion.EvidenceError("synthetic detached capture failed")
        with (
            mock.patch.object(
                completion,
                "_capture_detached_live_inputs",
                side_effect=primary,
            ),
            mock.patch.object(
                completion,
                "_close_retained_bundle_tree",
                side_effect=close_then_fail_first,
            ),
            self.assertRaisesRegex(
                completion.EvidenceError,
                "synthetic detached capture failed",
            ) as caught,
        ):
            completion._capture_core_live_inputs(
                inputs["source"],
                inputs["reopened"],
                self.manifest,
                inputs["runtime"],
                inputs["openings"],
                inputs["pre_ack"],
                inputs["final"],
            )
        self.assertIs(caught.exception, primary)
        self.assertEqual(
            closed_roots,
            [
                inputs["reopened"].parent.resolve(),
                inputs["source"].parent.resolve(),
            ],
        )
        self.assertIsNone(caught.exception.__context__)

    def test_retained_tree_close_matrix_preserves_primary_and_closes_once(self):
        for position in ("first", "middle", "last"):
            for body_fails in (False, True):
                suffix = f"close-matrix-{position}-{body_fails}"
                inputs, snapshots = self._capture_live_fixture(suffix)
                trees = (snapshots.reopened_bundle, snapshots.source_bundle)
                owners = []
                for tree in trees:
                    owners.extend(
                        (
                            *tree.payloads,
                            tree.index,
                            *reversed(tree.directories),
                            tree.root_directory,
                        )
                    )
                descriptors = [owner.fd for owner in owners]
                target = descriptors[
                    {"first": 0, "middle": len(descriptors) // 2, "last": -1}[position]
                ]
                calls = {descriptor: 0 for descriptor in descriptors}
                real_close = completion.os.close

                def fail_one_close(descriptor):
                    if descriptor in calls:
                        calls[descriptor] += 1
                    if descriptor == target:
                        raise OSError("synthetic-descriptor-close-canary")
                    return real_close(descriptor)

                primary = RuntimeError("synthetic-body-primary")
                manager = completion.open_authenticated_post_bundle_transition(
                    source_index=inputs["source"],
                    reopened_index=inputs["reopened"],
                    protected_openings=inputs["openings"],
                    pre_ack_bundle_reopen_receipt=inputs["pre_ack"],
                    final_bundle_reopen_receipt=inputs["final"],
                    manifest_path=self.manifest,
                    runtime_receipt_paths=inputs["runtime"],
                )
                try:
                    with (
                        self.subTest(position=position, body_fails=body_fails),
                        mock.patch.object(
                            completion,
                            "_capture_core_live_inputs",
                            return_value=snapshots,
                        ),
                        mock.patch.object(
                            completion,
                            "_validate_final_bundle_reopen_chain",
                            return_value={"post_bundle_expectation": object()},
                        ),
                        mock.patch.object(
                            completion.os,
                            "close",
                            side_effect=fail_one_close,
                        ),
                    ):
                        if body_fails:
                            with self.assertRaises(RuntimeError) as caught:
                                with manager:
                                    raise primary
                            self.assertIs(caught.exception, primary)
                            self.assertIsNone(caught.exception.__context__)
                        else:
                            with self.assertRaisesRegex(
                                completion.EvidenceError,
                                "retained descriptors could not be closed",
                            ) as caught:
                                with manager:
                                    pass
                            self.assertIsNone(caught.exception.__cause__)
                            self.assertIsNone(caught.exception.__context__)
                    self.assertTrue(all(count == 1 for count in calls.values()))
                finally:
                    try:
                        real_close(target)
                    except OSError:
                        pass

    def test_frozen_first_five_and_ordered_runtime_inventory_reject_drift(self):
        for copy_name in ("source", "reopened"):
            inputs = self._live_capture_fixture(f"frozen-{copy_name}")
            index_path = inputs[copy_name]
            document = completion._strict_json_bytes(
                index_path.read_bytes(),
                "synthetic frozen inventory index",
            )
            document["artifacts"]["cpu"]["synthetic_fixture"] = copy.deepcopy(
                document["artifacts"]["policy_paired_real"]["synthetic_fixture"]
            )
            index_path.write_bytes(completion._canonical_json_bytes(document))
            with (
                self.subTest(copy=copy_name),
                self.assertRaisesRegex(
                    completion.EvidenceError,
                    "first-five mappings differ",
                ),
            ):
                completion.completion_bundle_inventory(
                    index_path,
                    inputs["openings"],
                )

        inputs = self._live_capture_fixture("frozen-runtime")
        loaded_openings, context = privacy.load_private_openings(inputs["openings"])
        mutations = {
            "reordered": lambda records: records.reverse(),
            "role": lambda records: records[0].update(role="wrong-role"),
            "path": lambda records: records[0].update(
                bundle_path=records[1]["bundle_path"]
            ),
            "digest": lambda records: records[0].update(sha256="0" * 64),
            "size": lambda records: records[0].update(size_bytes=1),
        }
        for attack, mutate in mutations.items():
            changed = copy.deepcopy(context)
            records = changed["technical_inventory"]["runtime_receipts"]
            mutate(records)
            changed["technical_input_root"] = privacy.tagged_canonical_sha256(
                privacy.TECHNICAL_INPUT_INVENTORY_DOMAIN,
                changed["technical_inventory"],
            )
            attack_openings = privacy.PrivateOpenings(
                loaded_openings.salt_for_private_verification()
            )
            attack_openings._populated = True
            attack_path = inputs["openings"].parent / f"{attack}-openings.json"
            privacy.write_private_authority_file(
                attack_path,
                privacy.serialize_private_openings(attack_openings, changed),
                label="synthetic frozen inventory attack",
            )
            with (
                self.subTest(runtime_attack=attack),
                self.assertRaises(completion.EvidenceError),
            ):
                completion.completion_bundle_inventory(
                    inputs["source"],
                    attack_path,
                )

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
                {str(alias): str(target.resolve(strict=True))},
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

    def test_zip_preflight_rejects_unindexed_local_record_with_refreshed_bytes(self):
        original = self.zip_bytes([("indexed", b"reviewed")])
        hidden_archive = self.zip_bytes([("unindexed", b"not-reviewed")])
        original_eocd = original.rfind(b"PK\x05\x06")
        hidden_eocd = hidden_archive.rfind(b"PK\x05\x06")
        original_central = int.from_bytes(
            original[original_eocd + 16 : original_eocd + 20], "little"
        )
        hidden_central = int.from_bytes(
            hidden_archive[hidden_eocd + 16 : hidden_eocd + 20], "little"
        )
        hidden_local = hidden_archive[:hidden_central]
        mutated = bytearray(
            original[:original_central] + hidden_local + original[original_central:]
        )
        mutated_eocd = original_eocd + len(hidden_local)
        mutated[mutated_eocd + 16 : mutated_eocd + 20] = (
            original_central + len(hidden_local)
        ).to_bytes(4, "little")
        raw = bytes(mutated)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertEqual(archive.namelist(), ["indexed"])
        refreshed_descriptor = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        self.assertEqual(refreshed_descriptor["size_bytes"], len(raw))
        with self.assertRaisesRegex(
            completion.EvidenceError, "byte coverage differs|not contiguous"
        ):
            completion._preflight_zip(raw, "unindexed")

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
        required_pairs = [
            (group, case["name"])
            for group in ("correctness", "physical_checks")
            for case in manifest[group]
        ]
        required = [name for _group, name in required_pairs]
        correctness_evidence = completion._expected_correctness_candidate_evidence(
            manifest, candidate
        )
        sources = self.directory / "correctness-sources"
        sources.mkdir()
        payloads = []
        artifacts = []
        nested_paths = []
        for index, (group, name) in enumerate(required_pairs):
            record = {"case": name, "group": group}
            for role in ("reference", "candidate"):
                source = sources / f"{index:02d}-{role}.npz"
                marker = 2 * index + (role == "candidate")
                np.savez_compressed(source, value=np.asarray([marker], dtype=np.int64))
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
            record.update(
                {
                    "reference_observer_commit": manifest["reference"][
                        "observer_commit"
                    ],
                    "candidate_provenance": {
                        "commit": candidate["candidate_git_commit"],
                        "source_sha256": "1" * 64,
                        "controller_sha256": "2" * 64,
                    },
                    "comparison": {"passed": True, "failures": []},
                    "tolerance_results": [],
                }
            )
            artifacts.append(record)
        runtime_receipt = {
            "schema_version": 1,
            "kind": completion.RUNTIME_RECEIPT_KIND,
            "final_sha": candidate["candidate_git_commit"],
            "manifest_sha256": candidate["manifest_sha256"],
            "workflow": {
                "repository": "ruddyscent/gmes",
                "run_id": 100,
                "run_attempt": 1,
                "job_id": 200,
                "job_name": "issue123-cpu-correctness",
            },
            "profiler_witness": {
                "name": "cpu-profiler.json",
                "sha256": "f" * 64,
                "size_bytes": 1,
                "media_type": "application/json",
            },
            "runtime_mode": copy.deepcopy(completion.CPU_CORRECTNESS_RUNTIME_MODE),
            "candidate_archives": [
                {
                    "case": record["case"],
                    "sha256": record["candidate"]["sha256"],
                    "size_bytes": record["candidate"]["size_bytes"],
                }
                for record in artifacts
            ],
        }
        receipt_raw = completion._canonical_json_bytes(runtime_receipt)
        receipt_source = self.directory / "cpu-runtime-receipt.json"
        receipt_source.write_bytes(receipt_raw)
        receipt_bundle_path = "correctness/cpu-runtime-receipt.json"
        receipt_descriptor = {
            "path": receipt_bundle_path,
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "size_bytes": len(receipt_raw),
            "media_type": completion.MEDIA_TYPE_JSON,
        }
        document = {
            "schema_version": completion.CORRECTNESS_INDEX_SCHEMA_VERSION,
            "kind": completion.CORRECTNESS_INDEX_KIND,
            "contract_id": completion.CORRECTNESS_INDEX_CONTRACT_ID,
            "candidate_evidence": correctness_evidence,
            "manifest_contract_sha256": completion._canonical_sha256(manifest),
            "runtime_mode": {
                "device": "cpu",
                "precision": "float64",
                "graph_mode": "eager",
                "compile_policy": "eager",
                "compile_mode": "default",
            },
            "runtime_receipt": receipt_descriptor,
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
        payloads.append(
            {
                "source_path": receipt_source.name,
                "bundle_path": receipt_bundle_path,
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
                    index_artifact,
                    manifest,
                    candidate,
                    reader,
                    trusted_runtime_receipt=completion.LoadedArtifact(
                        receipt_descriptor,
                        receipt_source,
                        receipt_raw,
                        runtime_receipt,
                    ),
                ),
                rebuilt,
            )
            hardlink = self.directory / "cpu-runtime-receipt-hardlink.json"
            hardlink.hardlink_to(relocated / receipt_bundle_path)
            with self.assertRaisesRegex(
                completion.EvidenceError, "not independent from the external"
            ):
                completion._validate_correctness_index(
                    index_artifact,
                    manifest,
                    candidate,
                    reader,
                    trusted_runtime_receipt=completion.LoadedArtifact(
                        receipt_descriptor,
                        hardlink,
                        receipt_raw,
                        runtime_receipt,
                    ),
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
        self.assertEqual(error["message"], "evidence validation failed closed")
        self.assertNotIn("../manifest.json", repr(result))

    def test_evaluator_rejects_coherently_refreshed_untrusted_manifest(self):
        index_path = self.assemble("substituted-manifest")
        index = completion._strict_json_bytes(index_path.read_bytes(), "index")
        self.assertEqual(
            index["candidate_evidence"]["manifest_sha256"],
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        )
        result = completion.evaluate_completion(index_path, self.manifest)
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertEqual(result["cross_scope_errors"][0]["phase"], "bundle-index")
        self.assertEqual(
            result["cross_scope_errors"][0]["message"],
            "evidence validation failed closed",
        )
        self.assertNotIn(str(self.manifest), repr(result))

    def test_frozen_manifest_digest_matches_exact_repository_bytes(self):
        self.assertEqual(
            hashlib.sha256(completion.DEFAULT_MANIFEST.read_bytes()).hexdigest(),
            completion.TRUSTED_MANIFEST_SHA256,
        )

    def test_evaluator_preflights_index_size_before_reading_json(self):
        index_path = self.assemble()
        with mock.patch.object(completion, "MAX_INDEX_BYTES", 1):
            result = completion.evaluate_completion(index_path)
        self.assertFalse(result["issue_completion_satisfied"])
        self.assertEqual(result["cross_scope_errors"][0]["phase"], "bundle-index")
        self.assertEqual(
            result["cross_scope_errors"][0]["message"],
            "evidence validation failed closed",
        )
        self.assertNotIn(str(index_path), repr(result))


if __name__ == "__main__":
    unittest.main()
