from __future__ import annotations

import copy
import datetime as dt
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlencode

from benchmarks import issue123_completion as completion
from benchmarks import issue123_operations as operations

SYNTHETIC_OWNER_LOGIN = "fixture-owner.invalid"
SYNTHETIC_NON_OWNER_LOGIN = "fixture-non-owner.invalid"
SYNTHETIC_SUPERSEDED_IDS = {
    "SUPERSEDES_BASELINE_ISSUE_COMMENT": 910001,
    "SUPERSEDES_DM2_ISSUE_COMMENT": 910002,
    "SUPERSEDES_DM2_PR_COMMENT": 910003,
    "SUPERSEDES_SINGLE_GPU_ISSUE_COMMENT": 910004,
}


class _SyntheticBaselineLease:
    def __init__(self, validation):
        self.validation = validation
        self.require_count = 0
        self.closed = False

    def require_unchanged(self):
        if self.closed:
            raise operations.EvidenceError("synthetic baseline lease is closed")
        self.require_count += 1

    def close(self, *, primary_error=None):
        self.closed = True


class Issue123OperationsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.private_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.private_temporary.cleanup)
        self.private_root = Path(self.private_temporary.name)
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": "d" * 64,
        }
        self.base_sha = "b" * 40
        self.merge_sha = "c" * 40
        self.number = operations.PULL_REQUEST_NUMBER
        self.release_id = 30
        self.release_tag = "issue-123-technical-evidence-" + "a" * 40
        self.release_url = (
            f"https://github.com/{operations.REPOSITORY}/releases/tag/"
            + self.release_tag
        )
        self.asset_specs = {
            "technical_evidence": {
                "id": 401,
                "name": "issue-123-public-technical-evidence.zip",
                "size_bytes": 1001,
                "sha256": "1" * 64,
            },
            "technical_summary": {
                "id": 402,
                "name": "issue-123-technical-summary.json",
                "size_bytes": 1002,
                "sha256": "2" * 64,
            },
            "raw_timing": {
                "id": 403,
                "name": "issue-115-raw-timing.json",
                "size_bytes": 1003,
                "sha256": "3" * 64,
            },
            "event_profiler": {
                "id": 404,
                "name": "issue-115-event-level-profiler.json",
                "size_bytes": 1004,
                "sha256": "4" * 64,
            },
        }
        self.production_superseded_comments = operations.SUPERSEDED_OWNER_COMMENTS
        owner_patch = mock.patch.object(
            operations, "OWNER_LOGIN", SYNTHETIC_OWNER_LOGIN
        )
        owner_patch.start()
        self.addCleanup(owner_patch.stop)
        synthetic_baseline = {
            "BASELINE_V3_ROOT_COMMIT": "b" * 40,
            "BASELINE_RELEASE_TAG": "issue-123-synthetic-baseline-v3",
            "BASELINE_V3_ONE_URL": (
                f"https://github.com/{operations.REPOSITORY}/releases/download/"
                "issue-123-synthetic-baseline-v3/synthetic-one.json"
            ),
            "BASELINE_V3_ONE_SIZE": 101,
            "BASELINE_V3_ONE_SHA256": "8" * 64,
            "BASELINE_V3_PHYSICAL_URL": (
                f"https://github.com/{operations.REPOSITORY}/releases/download/"
                "issue-123-synthetic-baseline-v3/synthetic-physical.json"
            ),
            "BASELINE_V3_PHYSICAL_SIZE": 102,
            "BASELINE_V3_PHYSICAL_SHA256": "9" * 64,
            "BASELINE_V3_HOST_COMMITMENT": "7" * 64,
        }
        for name, value in synthetic_baseline.items():
            patcher = mock.patch.object(operations, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.superseded_comment_bodies = {}
        synthetic_specs = []
        for ordinal, specification in enumerate(
            self.production_superseded_comments, start=1
        ):
            changed = dict(specification)
            changed["id"] = SYNTHETIC_SUPERSEDED_IDS[changed["field"]]
            body = (
                f"Synthetic superseded amendment fixture {ordinal}; "
                "reserved test content only."
            )
            changed["body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
            changed["required_fragments"] = (
                "Synthetic superseded amendment",
                f"fixture {ordinal}",
            )
            self.superseded_comment_bodies[changed["id"]] = body
            synthetic_specs.append(changed)
        superseded_patch = mock.patch.object(
            operations, "SUPERSEDED_OWNER_COMMENTS", tuple(synthetic_specs)
        )
        superseded_patch.start()
        self.addCleanup(superseded_patch.stop)
        self.issue_fields = self._issue_contract_fields()
        self.pr_fields = self._pr_contract_fields()
        self.handoff_fields = self._handoff_contract_fields()
        self.responses = self._responses()
        self.publication_receipt_document = self._publication_receipt_document()
        self.publication_receipt_path = self.root / "publication-receipt.json"
        self.publication_receipt_path.write_bytes(
            operations._canonical_json_bytes(self.publication_receipt_document)
        )

    @staticmethod
    def _job(job_id, run_id, name):
        return {
            "id": job_id,
            "run_id": run_id,
            "run_attempt": 1,
            "name": name,
            "status": "completed",
            "conclusion": "success",
            "started_at": "2026-09-02T23:00:00Z",
            "completed_at": "2026-09-02T23:10:00Z",
        }

    @staticmethod
    def _contract(marker, fields):
        return "\n".join([marker, *(f"{key}={value}" for key, value in fields.items())])

    def _asset_url(self, name):
        return (
            "https://github.com/ruddyscent/gmes/releases/download/"
            f"{self.release_tag}/{name}"
        )

    def _asset_contract_fields(self, prefix, role):
        asset = self.asset_specs[role]
        return {
            f"{prefix}_ASSET_NAME": asset["name"],
            f"{prefix}_ASSET_URL": self._asset_url(asset["name"]),
            f"{prefix}_ASSET_SIZE_BYTES": str(asset["size_bytes"]),
            f"{prefix}_ASSET_SHA256": asset["sha256"],
        }

    def _issue_contract_fields(self):
        return {
            "FINAL_SHA": "a" * 40,
            "PR": "167",
            "TARGET_ISSUE": "123",
            "TECHNICAL_RELEASE_URL": self.release_url,
            "BASELINE_V3_ROOT_COMMIT": operations.BASELINE_V3_ROOT_COMMIT,
            "BASELINE_V3_ONE_URL": operations.BASELINE_V3_ONE_URL,
            "BASELINE_V3_ONE_SIZE_BYTES": str(operations.BASELINE_V3_ONE_SIZE),
            "BASELINE_V3_ONE_SHA256": operations.BASELINE_V3_ONE_SHA256,
            "BASELINE_V3_PHYSICAL_URL": operations.BASELINE_V3_PHYSICAL_URL,
            "BASELINE_V3_PHYSICAL_SIZE_BYTES": str(
                operations.BASELINE_V3_PHYSICAL_SIZE
            ),
            "BASELINE_V3_PHYSICAL_SHA256": operations.BASELINE_V3_PHYSICAL_SHA256,
            "BASELINE_V3_HOSTNAME": "redacted",
            "BASELINE_V3_HOST_IDENTITY_SCHEMA": "torch-cpu-host-identity-v2",
            "BASELINE_V3_HOST_COMMITMENT_SHA256": (
                operations.BASELINE_V3_HOST_COMMITMENT
            ),
            "BASELINE_V3_DISPOSITION": ("authoritative-published-privacy-sanitized"),
            **{
                field: str(identifier)
                for field, identifier in SYNTHETIC_SUPERSEDED_IDS.items()
            },
            "PRIOR_CONTRACT_DISPOSITION": "superseded-by-this-amendment",
            "SOLVER_ABI": "torch-fdtd-regions-v15",
            "EXECUTION_REPRESENTATION": (
                "external-no-inner-cudagraph-regions+dm2-raw-fixed-masked-v1"
            ),
            "DIFFERENTIAL_SCHEMA_VERSION": "5",
            "DIFFERENTIAL_EARLY_STEPS": "0,1,2,5",
            "DIFFERENTIAL_EARLY_CONTRACT": "manifest-strict-elementwise",
            "DIFFERENTIAL_LATE_STEPS": "20,100",
            "DIFFERENTIAL_LATE_CONTRACT": "normalized-linf-l2-at-most-1e-6",
            "SINGLE_GPU_3D_CASE": "single-gpu-3d",
            "SINGLE_GPU_3D_PRECISION": "float64",
            "SINGLE_GPU_3D_LATE_STEPS": "20,100",
            "SINGLE_GPU_3D_LATE_RESIDUAL_CONTRACT": ("normalized-linf-l2-at-most-1e-6"),
            "SINGLE_GPU_3D_RESIDUAL_DENOMINATOR_FLOOR": "2e-12",
            "SINGLE_GPU_3D_L2_DENOMINATOR_SCALE": "sqrt(N)",
            "SINGLE_GPU_3D_ZERO_REFERENCE_CONTRACT": "exact",
            "PUBLIC_TRACE_DISPOSITION": ("published-event-complete-privacy-normalized"),
            "CORRECTNESS_ARRAY_DISPOSITION": "private",
            "CORRECTNESS_COMMITMENT_DISPOSITION": ("published-in-technical-evidence"),
        }

    def _pr_contract_fields(self):
        return {
            "FINAL_SHA": "a" * 40,
            "PR": "167",
            "TARGET_ISSUE": "123",
            "FINAL_COMMIT_URL": (
                f"https://github.com/{operations.REPOSITORY}/commit/" + "a" * 40
            ),
            "FINAL_COMMIT_VERIFICATION": "verified:valid",
            "CI_RUN_URL": (
                f"https://github.com/{operations.REPOSITORY}/actions/runs/10"
            ),
            "CODEQL_RUN_URL": (
                f"https://github.com/{operations.REPOSITORY}/actions/runs/20"
            ),
            "TEST_SUMMARY": "required-ci-and-regression-tests-pass",
            "EVIDENCE_SUMMARY": (
                "five-technical-scopes-pass-private-arrays-commitment-published"
            ),
            "TECHNICAL_RELEASE_URL": self.release_url,
            **self._asset_contract_fields("TECHNICAL_EVIDENCE", "technical_evidence"),
            **self._asset_contract_fields("TECHNICAL_SUMMARY", "technical_summary"),
        }

    def _handoff_contract_fields(self):
        return {
            "FINAL_SHA": "a" * 40,
            "PR": "167",
            "TARGET_ISSUE": "123",
            "HANDOFF_ISSUE": "115",
            "TECHNICAL_RELEASE_URL": self.release_url,
            "RAW_TIMING_CONTRACT": "torch-utils-benchmark-fixed-workloads",
            **self._asset_contract_fields("RAW_TIMING", "raw_timing"),
            "EVENT_PROFILER_CONTRACT": "event-level-profiler-fixed-workloads",
            **self._asset_contract_fields("EVENT_PROFILER", "event_profiler"),
            "HANDOFF_DISPOSITION": "complete",
        }

    def _comment(
        self,
        identifier,
        issue_number,
        body,
        *,
        owner=True,
        html_kind="issues",
        created_at="2026-09-03T01:00:00Z",
        updated_at=None,
    ):
        return {
            "id": identifier,
            "url": (
                f"https://api.github.com/repos/{operations.REPOSITORY}/issues/comments/"
                f"{identifier}"
            ),
            "issue_url": (
                f"https://api.github.com/repos/{operations.REPOSITORY}/issues/"
                f"{issue_number}"
            ),
            "html_url": (
                f"https://github.com/{operations.REPOSITORY}/{html_kind}/{issue_number}"
                f"#issuecomment-{identifier}"
            ),
            "user": {
                "login": (SYNTHETIC_OWNER_LOGIN if owner else SYNTHETIC_NON_OWNER_LOGIN)
            },
            "author_association": "OWNER" if owner else "CONTRIBUTOR",
            "created_at": created_at,
            "updated_at": updated_at or created_at,
            "body": body,
        }

    def _release_asset(self, role):
        asset = self.asset_specs[role]
        return {
            "id": asset["id"],
            "name": asset["name"],
            "url": (
                f"https://api.github.com/repos/{operations.REPOSITORY}/releases/assets/"
                f"{asset['id']}"
            ),
            "browser_download_url": self._asset_url(asset["name"]),
            "state": "uploaded",
            "size": asset["size_bytes"],
            "digest": f"sha256:{asset['sha256']}",
            "uploader": {"login": SYNTHETIC_OWNER_LOGIN},
            "created_at": "2026-09-02T23:20:00Z",
            "updated_at": "2026-09-02T23:25:00Z",
        }

    def _publication_bindings(self):
        return {
            "final_sha": self.candidate["candidate_git_commit"],
            "manifest_sha256": self.candidate["manifest_sha256"],
            "jobs": [
                {
                    "name": operations.PUBLICATION_JOB_NAMES[0],
                    "run_id": 10,
                    "run_attempt": 1,
                    "job_id": 11,
                },
                {
                    "name": operations.PUBLICATION_JOB_NAMES[1],
                    "run_id": 10,
                    "run_attempt": 1,
                    "job_id": 12,
                },
                {
                    "name": operations.PUBLICATION_JOB_NAMES[2],
                    "run_id": 20,
                    "run_attempt": 1,
                    "job_id": 21,
                },
                {
                    "name": operations.PUBLICATION_JOB_NAMES[3],
                    "run_id": 20,
                    "run_attempt": 1,
                    "job_id": 22,
                },
            ],
        }

    def _publication_receipt_document(self):
        bindings = self._publication_bindings()
        ledger = [
            {
                "role": role,
                "name": self.asset_specs[role]["name"],
                "size_bytes": self.asset_specs[role]["size_bytes"],
                "sha256": self.asset_specs[role]["sha256"],
            }
            for role in operations.TECHNICAL_RELEASE_ASSETS
        ]
        release_assets = [
            {
                "role": role,
                "asset_id": self.asset_specs[role]["id"],
                "release_id": self.release_id,
                "name": self.asset_specs[role]["name"],
                "api_url": (
                    "https://api.github.com/repos/ruddyscent/gmes/releases/assets/"
                    f"{self.asset_specs[role]['id']}"
                ),
                "browser_download_url": self._asset_url(self.asset_specs[role]["name"]),
                "state": "uploaded",
                "size_bytes": self.asset_specs[role]["size_bytes"],
                "sha256": self.asset_specs[role]["sha256"],
            }
            for role in operations.TECHNICAL_RELEASE_ASSETS
        ]
        api_root = f"https://api.github.com/repos/{operations.REPOSITORY}"
        release_capture = {
            "schema_version": 1,
            "kind": operations.PUBLICATION_RELEASE_CAPTURE_KIND,
            "repository": operations.REPOSITORY,
            "release_id": self.release_id,
            "tag_name": self.release_tag,
            "target_commitish": self.candidate["candidate_git_commit"],
            "api_url": f"{api_root}/releases/{self.release_id}",
            "html_url": self.release_url,
            "immutable": True,
            "draft": False,
            "prerelease": False,
            "tag_ref": {
                "ref": f"refs/tags/{self.release_tag}",
                "api_url": f"{api_root}/git/refs/tags/{self.release_tag}",
                "object_type": "commit",
                "object_sha": self.candidate["candidate_git_commit"],
                "object_url": (
                    f"{api_root}/git/commits/{self.candidate['candidate_git_commit']}"
                ),
            },
            "assets": release_assets,
        }
        claims = []
        for index, (claim_name, scope) in enumerate(
            operations.PUBLICATION_EXECUTION_CLAIMS, start=1
        ):
            claims.append(
                {
                    "claim": claim_name,
                    "scope": scope,
                    "trace_name": f"trace-{scope}",
                    "validation_workflow": "CI",
                    "validator_job": bindings["jobs"][0],
                    "event_count": index,
                    "semantic_inventory_sha256": f"{index + 4:x}" * 64,
                    "normalized_trace_sha256": f"{index + 7:x}" * 64,
                }
            )
        witness = {
            "schema_version": 1,
            "kind": operations.PUBLICATION_EXECUTION_WITNESS_KIND,
            "bindings": bindings,
            "claims": claims,
        }
        witness_raw = operations._canonical_json_bytes(witness)
        return {
            "schema_version": 1,
            "kind": operations.PUBLICATION_RECEIPT_KIND,
            "bindings": bindings,
            "asset_order": list(operations.TECHNICAL_RELEASE_ASSETS),
            "asset_ledger": ledger,
            "release_capture": release_capture,
            "execution_witness": witness,
            "execution_witness_member": {
                "path": operations.PUBLICATION_EXECUTION_WITNESS_PATH,
                "media_type": operations.MEDIA_TYPE_JSON,
                "size_bytes": len(witness_raw),
                "sha256": hashlib.sha256(witness_raw).hexdigest(),
            },
            "hashes": {
                "release_capture_sha256": hashlib.sha256(
                    operations._canonical_json_bytes(release_capture)
                ).hexdigest(),
                "execution_witness_member_sha256": hashlib.sha256(
                    witness_raw
                ).hexdigest(),
                "trusted_policy_sha256": "e" * 64,
                "asset_ledger_sha256": hashlib.sha256(
                    operations._canonical_json_bytes(ledger)
                ).hexdigest(),
            },
        }

    def _responses(self):
        candidate = self.candidate["candidate_git_commit"]
        merge_ref = f"refs/pull/{self.number}/merge"
        superseded = {
            specification["id"]: self._comment(
                specification["id"],
                specification["issue_number"],
                self.superseded_comment_bodies[specification["id"]],
                html_kind=specification["html_kind"],
                created_at=specification["created_at"],
                updated_at=specification["updated_at"],
            )
            for specification in operations.SUPERSEDED_OWNER_COMMENTS
        }
        release_assets = [
            self._release_asset(role)
            for role in (
                "technical_evidence",
                "technical_summary",
                "raw_timing",
                "event_profiler",
            )
        ]
        analysis = lambda category, identifier: {
            "id": identifier,
            "language": "python" if category.endswith("python") else "cpp",
            "analysis_key": ".github/workflows/codeql.yml:analyze",
            "category": category,
            "commit_sha": self.merge_sha,
            "ref": merge_ref,
            "created_at": "2026-09-02T23:05:00Z",
            "results_count": 7,
            "rules_count": 100,
            "error": "",
            "warning": "",
            "tool": {"name": "CodeQL", "version": "2.23.0"},
        }
        return {
            "technical_release": {
                "id": self.release_id,
                "url": (
                    "https://api.github.com/repos/ruddyscent/gmes/releases/"
                    f"{self.release_id}"
                ),
                "assets_url": (
                    "https://api.github.com/repos/ruddyscent/gmes/releases/"
                    f"{self.release_id}/assets"
                ),
                "upload_url": (
                    "https://uploads.github.com/repos/ruddyscent/gmes/releases/"
                    f"{self.release_id}/assets{{?name,label}}"
                ),
                "html_url": self.release_url,
                "tag_name": self.release_tag,
                "target_commitish": candidate,
                "draft": False,
                "prerelease": False,
                "immutable": True,
                "created_at": "2026-09-02T23:15:00Z",
                "published_at": "2026-09-02T23:30:00Z",
                "updated_at": "2026-09-02T23:40:00Z",
                "author": {"login": SYNTHETIC_OWNER_LOGIN},
                "assets": copy.deepcopy(release_assets),
            },
            "technical_release_assets": [copy.deepcopy(release_assets)],
            "technical_release_tag": {
                "ref": f"refs/tags/{self.release_tag}",
                "url": (
                    "https://api.github.com/repos/ruddyscent/gmes/git/refs/tags/"
                    f"{self.release_tag}"
                ),
                "object": {
                    "type": "commit",
                    "sha": candidate,
                    "url": (
                        "https://api.github.com/repos/ruddyscent/gmes/git/commits/"
                        f"{candidate}"
                    ),
                },
            },
            "issue_123": {
                "number": operations.TARGET_ISSUE_NUMBER,
                "state": "open",
                "closed_at": None,
                "repository_url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}"
                ),
                "url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}/issues/"
                    f"{operations.TARGET_ISSUE_NUMBER}"
                ),
                "comments": 4,
                "created_at": "2026-08-30T20:00:00Z",
                "updated_at": "2026-09-03T01:30:00Z",
                "body": (
                    "## Implementation work\n"
                    "- [ ] publish the final bundle\n"
                    "- [ ] complete the post-bundle checklist\n"
                ),
            },
            "issue_123_comments": [
                [
                    superseded[910001],
                    superseded[910002],
                    superseded[910004],
                    self._comment(
                        800,
                        operations.TARGET_ISSUE_NUMBER,
                        self._contract(
                            "GMES_ISSUE_123_FINAL_CONTRACT_AMENDMENT_V2",
                            self.issue_fields,
                        ),
                        created_at="2026-09-03T01:30:00Z",
                    ),
                ]
            ],
            "issue_115": {
                "number": operations.HANDOFF_ISSUE_NUMBER,
                "state": "closed",
                "state_reason": "completed",
                "closed_at": "2026-08-25T09:01:33Z",
                "repository_url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}"
                ),
                "url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}/issues/"
                    f"{operations.HANDOFF_ISSUE_NUMBER}"
                ),
                "comments": 1,
                "created_at": "2026-08-24T00:00:00Z",
                "updated_at": "2026-09-02T22:40:00Z",
                "body": (
                    "## Implementation work\n"
                    "- [x] Add runtime-aware repeated measurement using "
                    "`torch.utils.benchmark` once Torch is present, while preserving "
                    "a native reference runner.\n"
                    "- [x] Add optional profiler capture for graph breaks, kernel "
                    "launches, device copies, and allocator behavior.\n"
                ),
            },
            "issue_115_comments": [
                [
                    self._comment(
                        900,
                        operations.HANDOFF_ISSUE_NUMBER,
                        self._contract(
                            "GMES_ISSUE_115_FINAL_RUNTIME_HANDOFF_V2",
                            self.handoff_fields,
                        ),
                        created_at="2026-09-03T01:00:00Z",
                    )
                ]
            ],
            "pull_request": {
                "number": self.number,
                "url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}/pulls/"
                    f"{self.number}"
                ),
                "issue_url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}/issues/"
                    f"{self.number}"
                ),
                "comments": 2,
                "created_at": "2026-08-30T20:00:00Z",
                "updated_at": "2026-09-03T01:20:00Z",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "merge_commit_sha": self.merge_sha,
                "base": {
                    "ref": "master",
                    "sha": self.base_sha,
                    "repo": {"full_name": operations.REPOSITORY},
                },
                "head": {
                    "sha": candidate,
                    "repo": {"full_name": operations.REPOSITORY},
                },
            },
            "pull_request_comments": [
                [
                    superseded[910003],
                    self._comment(
                        950,
                        self.number,
                        self._contract(
                            "GMES_PR_167_FINAL_CANDIDATE_INSIGHT_V2",
                            self.pr_fields,
                        ),
                        html_kind="pull",
                        created_at="2026-09-03T01:20:00Z",
                    ),
                ]
            ],
            "candidate_commit": {
                "sha": candidate,
                "url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}/commits/"
                    f"{candidate}"
                ),
                "html_url": (
                    f"https://github.com/{operations.REPOSITORY}/commit/{candidate}"
                ),
                "commit": {
                    "verification": {
                        "verified": True,
                        "reason": "valid",
                        "verified_at": "2026-09-02T23:00:00Z",
                    }
                },
            },
            "base_compare": {
                "url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}/compare/"
                    f"{self.base_sha}...{candidate}"
                ),
                "status": "ahead",
                "ahead_by": 1,
                "behind_by": 0,
                "total_commits": 1,
                "base_commit": {"sha": self.base_sha},
                "merge_base_commit": {"sha": self.base_sha},
                "commits": [{"sha": candidate}],
            },
            "ci_run": {
                "id": 10,
                "url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}/actions/"
                    "runs/10"
                ),
                "html_url": "https://github.com/ruddyscent/gmes/actions/runs/10",
                "repository": {"full_name": operations.REPOSITORY},
                "head_repository": {"full_name": operations.REPOSITORY},
                "pull_requests": [
                    {
                        "number": self.number,
                        "url": (
                            f"https://api.github.com/repos/{operations.REPOSITORY}/"
                            f"pulls/{self.number}"
                        ),
                        "base": {
                            "ref": "master",
                            "sha": self.base_sha,
                            "repo": {
                                "url": f"https://api.github.com/repos/{operations.REPOSITORY}"
                            },
                        },
                        "head": {
                            "sha": candidate,
                            "repo": {
                                "url": f"https://api.github.com/repos/{operations.REPOSITORY}"
                            },
                        },
                    }
                ],
                "name": "CI",
                "event": "pull_request",
                "head_sha": candidate,
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "created_at": "2026-09-02T22:50:00Z",
                "updated_at": "2026-09-02T23:10:00Z",
            },
            "ci_jobs": [
                {
                    "total_count": 2,
                    "jobs": [
                        self._job(11, 10, operations.REQUIRED_STATUS_CONTEXTS[0]),
                        self._job(12, 10, operations.REQUIRED_STATUS_CONTEXTS[1]),
                    ],
                }
            ],
            "codeql_run": {
                "id": 20,
                "url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}/actions/"
                    "runs/20"
                ),
                "html_url": "https://github.com/ruddyscent/gmes/actions/runs/20",
                "repository": {"full_name": operations.REPOSITORY},
                "head_repository": {"full_name": operations.REPOSITORY},
                "pull_requests": [
                    {
                        "number": self.number,
                        "url": (
                            f"https://api.github.com/repos/{operations.REPOSITORY}/"
                            f"pulls/{self.number}"
                        ),
                        "base": {
                            "ref": "master",
                            "sha": self.base_sha,
                            "repo": {
                                "url": f"https://api.github.com/repos/{operations.REPOSITORY}"
                            },
                        },
                        "head": {
                            "sha": candidate,
                            "repo": {
                                "url": f"https://api.github.com/repos/{operations.REPOSITORY}"
                            },
                        },
                    }
                ],
                "name": "CodeQL",
                "event": "pull_request",
                "head_sha": candidate,
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "created_at": "2026-09-02T22:50:00Z",
                "updated_at": "2026-09-02T23:10:00Z",
            },
            "codeql_jobs": [
                {
                    "total_count": 2,
                    "jobs": [
                        self._job(21, 20, operations.REQUIRED_CODEQL_JOBS[0]),
                        self._job(22, 20, operations.REQUIRED_CODEQL_JOBS[1]),
                    ],
                }
            ],
            "codeql_analyses": [
                [
                    analysis("/language:python", 31),
                    analysis("/language:c-cpp", 32),
                ]
            ],
            "codeql_alerts": [[]],
            "ruleset": {
                "id": operations.RULESET_ID,
                "name": "Protect master",
                "target": "branch",
                "source_type": "Repository",
                "source": operations.REPOSITORY,
                "enforcement": "active",
                "conditions": {
                    "ref_name": {"exclude": [], "include": ["~DEFAULT_BRANCH"]}
                },
                "rules": [
                    {"type": "deletion"},
                    {"type": "non_fast_forward"},
                    {"type": "required_linear_history"},
                    {
                        "type": "pull_request",
                        "parameters": {
                            "required_approving_review_count": 0,
                            "required_review_thread_resolution": True,
                            "require_extra_approval_for_unattributed_changes": True,
                            "allowed_merge_methods": ["squash"],
                        },
                    },
                    {
                        "type": "required_status_checks",
                        "parameters": {
                            "strict_required_status_checks_policy": True,
                            "do_not_enforce_on_create": False,
                            "required_status_checks": [
                                {"context": name, "integration_id": 15368}
                                for name in operations.REQUIRED_STATUS_CONTEXTS
                            ],
                        },
                    },
                    {
                        "type": "code_scanning",
                        "parameters": {
                            "code_scanning_tools": [
                                {
                                    "tool": "CodeQL",
                                    "security_alerts_threshold": "high_or_higher",
                                    "alerts_threshold": "errors",
                                }
                            ]
                        },
                    },
                ],
                "bypass_actors": [],
                "current_user_can_bypass": "never",
                "created_at": "2026-08-21T00:00:00Z",
                "updated_at": "2026-08-22T00:00:00Z",
            },
            "check_runs": [
                {
                    "total_count": 2,
                    "check_runs": [
                        {
                            "name": name,
                            "head_sha": candidate,
                            "status": "completed",
                            "conclusion": "success",
                            "app": {"slug": "github-actions"},
                        }
                        for name in operations.REQUIRED_STATUS_CONTEXTS
                    ],
                }
            ],
            "reviews": [[]],
            "requested_reviewers": {"users": [], "teams": []},
            "review_threads": [
                {
                    "data": {
                        "repository": {
                            "nameWithOwner": operations.REPOSITORY,
                            "pullRequest": {
                                "number": self.number,
                                "headRefOid": candidate,
                                "closingIssuesReferences": {
                                    "totalCount": 1,
                                    "nodes": [
                                        {
                                            "number": operations.TARGET_ISSUE_NUMBER,
                                            "repository": {
                                                "nameWithOwner": operations.REPOSITORY
                                            },
                                        }
                                    ],
                                },
                                "reviews": {"totalCount": 0},
                                "reviewThreads": {
                                    "totalCount": 1,
                                    "pageInfo": {
                                        "hasNextPage": False,
                                        "endCursor": "thread-cursor",
                                    },
                                    "nodes": [{"id": "thread", "isResolved": True}],
                                },
                            },
                        }
                    }
                }
            ],
        }

    def _fixture_capture(self, request, value):
        pages = value if request["paginated"] else [value]
        frames = []
        for index, page in enumerate(pages):
            headers = {
                "content-type": "application/json; charset=utf-8",
                "etag": f'"{hashlib.sha256(operations._canonical_json_bytes(page)).hexdigest()}"',
                "last-modified": "Wed, 02 Sep 2026 23:40:00 GMT",
                "x-github-api-version-selected": "2022-11-28",
                "x-github-media-type": "github.v3; format=json",
                "x-oauth-scopes": "repo",
            }
            if request["graphql"]:
                headers.pop("x-github-api-version-selected")
                headers["x-github-media-type"] = "github.v4; format=json"
            if not request["graphql"] and index < len(pages) - 1:
                query = {**request["parameters"], "page": str(index + 2)}
                next_url = (
                    f"https://api.github.com/{request['endpoint']}?"
                    f"{urlencode(sorted(query.items()))}"
                )
                headers["link"] = f'<{next_url}>; rel="next"'
            frames.append({"status": 200, "headers": headers, "value": page})
        return operations._response_capture_from_frames(
            frames,
            paginated=request["paginated"],
            graphql=request["graphql"],
            label="fixture response",
        )

    def _coherent_document(self, document, raw):
        changed = copy.deepcopy(document)
        changed["response_captures"] = {
            role: self._fixture_capture(
                changed["responses"][role]["request"], raw[role]
            )[1]
            for role in operations.RESPONSE_ROLE_ORDER
        }
        return changed

    def _evaluate(self, document, raw):
        return operations.evaluate_operations(
            self._coherent_document(document, raw), raw, self.candidate
        )

    def _use_checked_final_issue(self):
        issue = self.responses["issue_123"]
        issue["body"] = issue["body"].replace("- [ ] publish", "- [x] publish")
        issue["body"] = issue["body"].replace("- [ ] complete", "- [x] complete")
        issue["updated_at"] = "2026-09-03T01:31:00Z"

    @staticmethod
    def _post_bundle_expectation(document, responses):
        issue = responses["issue_123"]
        observation = operations._validate_post_bundle_checklist(issue, "checked")
        o1_sha256 = document["response_captures"]["issue_123"][
            "canonical_response_sha256"
        ]
        return operations.AuthenticatedPostBundleExpectation(
            checked_lines=operations.FINAL_CHECKLIST_CHECKED,
            o0_canonical_response_sha256="0" * 64,
            o1_canonical_response_sha256=o1_sha256,
            o1_body_sha256=observation["body_sha256"],
            o1_updated_at=observation["updated_at"],
            b0_inventory_root="1" * 64,
            b0_reopen_receipt_sha256="2" * 64,
            b0_reopened_at="2026-09-03T01:30:00Z",
            checklist_transition_sha256=operations.checklist_transition_sha256(
                issue, "checked"
            ),
        )

    @staticmethod
    def _acknowledgment(expectation):
        return {
            "checked_lines": list(expectation.checked_lines),
            "o0_canonical_response_sha256": (expectation.o0_canonical_response_sha256),
            "o1_canonical_response_sha256": (expectation.o1_canonical_response_sha256),
            "o1_body_sha256": expectation.o1_body_sha256,
            "o1_updated_at": expectation.o1_updated_at,
            "b0_inventory_root": expectation.b0_inventory_root,
            "b0_reopen_receipt_sha256": (expectation.b0_reopen_receipt_sha256),
            "b0_reopened_at": expectation.b0_reopened_at,
            "fresh_response_equal": True,
        }

    @staticmethod
    def _baseline_validation():
        observed_at = (
            dt.datetime.now(dt.UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        release_id = 42
        api_root = f"https://api.github.com/repos/{operations.REPOSITORY}"
        web_root = f"https://github.com/{operations.REPOSITORY}"
        body = {
            "release_identity": {
                "repository": operations.REPOSITORY,
                "release_id": release_id,
                "tag_name": operations.BASELINE_RELEASE_TAG,
                "api_url": f"{api_root}/releases/{release_id}",
                "html_url": (
                    f"{web_root}/releases/tag/{operations.BASELINE_RELEASE_TAG}"
                ),
                "tag_ref": {
                    "ref": f"refs/tags/{operations.BASELINE_RELEASE_TAG}",
                    "object_type": "commit",
                    "object_sha": operations.BASELINE_V3_ROOT_COMMIT,
                    "object_url": (
                        f"{api_root}/git/commits/"
                        f"{operations.BASELINE_V3_ROOT_COMMIT}"
                    ),
                },
            },
            "asset_ledger": [
                {
                    "thread_mode": mode,
                    "name": name,
                    "asset_id": ordinal,
                    "release_id": release_id,
                    "api_url": f"{api_root}/releases/assets/{ordinal}",
                    "browser_download_url": (
                        f"{web_root}/releases/download/"
                        f"{operations.BASELINE_RELEASE_TAG}/{name}"
                    ),
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
                for ordinal, (mode, name, size_bytes, sha256) in enumerate(
                    (
                        (
                            "one",
                            "torch-cpu-baseline-one.json",
                            operations.BASELINE_V3_ONE_SIZE,
                            operations.BASELINE_V3_ONE_SHA256,
                        ),
                        (
                            "physical",
                            "torch-cpu-baseline-physical.json",
                            operations.BASELINE_V3_PHYSICAL_SIZE,
                            operations.BASELINE_V3_PHYSICAL_SHA256,
                        ),
                    ),
                    start=101,
                )
            ],
            "observed_at": observed_at,
            "api_observations": [
                {
                    "endpoint": endpoint,
                    "canonical_response_sha256": "3" * 64,
                    "canonical_response_size_bytes": 1,
                    "page_ledger_sha256": "4" * 64,
                }
                for endpoint in (
                    f"repos/{operations.REPOSITORY}/releases/tags/"
                    f"{operations.BASELINE_RELEASE_TAG}",
                    f"repos/{operations.REPOSITORY}/git/ref/tags/"
                    f"{operations.BASELINE_RELEASE_TAG}",
                )
            ],
        }
        from benchmarks import issue123_privacy as privacy

        return {
            **body,
            "authority_sha256": privacy.tagged_canonical_sha256(
                operations.BASELINE_AUTHORITY_DOMAIN,
                body,
            ),
        }

    @staticmethod
    def _replace_receipt_document(document, receipt):
        changed = copy.deepcopy(document)
        raw = operations._canonical_json_bytes(receipt)
        changed["publication_receipt"] = {
            "media_type": operations.MEDIA_TYPE_JSON,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "document": receipt,
        }
        return changed

    @staticmethod
    def _comment_with_marker(raw, role, marker):
        return next(
            comment
            for page in raw[role]
            for comment in page
            if marker in comment["body"]
        )

    def _capture(self, output_name="operations", *, before_capture=None):
        def capture(endpoint, **kwargs):
            if before_capture is not None:
                before_capture(endpoint)
            role_by_suffix = {
                f"/releases/tags/{self.release_tag}": "technical_release",
                f"/releases/{self.release_id}/assets": "technical_release_assets",
                f"/git/ref/tags/{self.release_tag}": "technical_release_tag",
                f"/issues/{operations.TARGET_ISSUE_NUMBER}": "issue_123",
                f"/issues/{operations.TARGET_ISSUE_NUMBER}/comments": (
                    "issue_123_comments"
                ),
                f"/issues/{operations.HANDOFF_ISSUE_NUMBER}": "issue_115",
                f"/issues/{operations.HANDOFF_ISSUE_NUMBER}/comments": (
                    "issue_115_comments"
                ),
                f"/pulls/{self.number}": "pull_request",
                f"/issues/{self.number}/comments": "pull_request_comments",
                f"/commits/{self.candidate['candidate_git_commit']}": (
                    "candidate_commit"
                ),
                f"compare/{self.base_sha}...{self.candidate['candidate_git_commit']}": "base_compare",
                "/actions/runs/10": "ci_run",
                "/actions/runs/10/jobs": "ci_jobs",
                "/actions/runs/20": "codeql_run",
                "/actions/runs/20/jobs": "codeql_jobs",
                "/code-scanning/analyses": "codeql_analyses",
                "/code-scanning/alerts": "codeql_alerts",
                f"/rulesets/{operations.RULESET_ID}": "ruleset",
                f"/commits/{self.candidate['candidate_git_commit']}/check-runs": "check_runs",
                f"/pulls/{self.number}/reviews": "reviews",
                f"/pulls/{self.number}/requested_reviewers": "requested_reviewers",
            }
            if endpoint == "graphql":
                role = "review_threads"
            else:
                matches = [
                    role
                    for suffix, role in role_by_suffix.items()
                    if endpoint.endswith(suffix)
                ]
                self.assertEqual(len(matches), 1, endpoint)
                role = matches[0]
            graphql_variables = kwargs.get("graphql_variables")
            request = {
                "endpoint": endpoint,
                "method": "POST" if graphql_variables is not None else "GET",
                "headers": operations.GITHUB_API_HEADERS,
                "parameters": kwargs.get("parameters") or {},
                "paginated": kwargs.get("paginated", False),
                "jq": "." if kwargs.get("paginated", False) else None,
                "graphql": graphql_variables is not None,
                "query": (
                    operations.PULL_REQUEST_CONTEXT_QUERY
                    if graphql_variables is not None
                    else None
                ),
                "variables": graphql_variables or {},
            }
            return self._fixture_capture(request, self.responses[role])

        output = self.root / output_name
        with (
            mock.patch.object(
                operations, "candidate_evidence", return_value=self.candidate
            ),
            mock.patch.object(operations, "_github_api_capture", side_effect=capture),
            mock.patch.object(operations.subprocess, "run") as run,
        ):
            index, scope = operations.capture_operations(
                repository=operations.REPOSITORY,
                pull_request_number=self.number,
                ci_run_id=10,
                codeql_run_id=20,
                technical_release_tag=self.release_tag,
                publication_receipt=self.publication_receipt_path,
                output_directory=output,
            )
        run.assert_not_called()
        return output, index, scope

    def test_capture_and_completion_recompute_raw_api_evidence(self):
        output, index_path, scope_path = self._capture()
        expected_output = output.resolve()
        self.assertEqual(index_path.parent, expected_output)
        self.assertEqual(scope_path.parent, expected_output)
        self.assertFalse(index_path.parent.name.startswith("."))
        document = json.loads(index_path.read_text())
        scope = json.loads(scope_path.read_text())
        result = completion._validate_operations_scope(
            scope,
            completion.ArtifactReader(output, self.candidate),
            self.candidate,
        )
        self.assertEqual(result["pull_request"]["head_sha"], "a" * 40)
        self.assertEqual(
            result["codeql_analyses"]["/language:python"]["results_count"], 7
        )
        self.assertEqual(result["codeql_quality_blockers"], [])
        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(
            (
                document["repository"],
                document["target_issue_number"],
                document["handoff_issue_number"],
                document["pull_request_number"],
            ),
            (operations.REPOSITORY, 123, 115, 167),
        )
        self.assertEqual(result["closing_issue_numbers"], [123])
        self.assertEqual(result["candidate_commit_verification"]["reason"], "valid")
        self.assertEqual(document["technical_release_tag"], self.release_tag)
        self.assertEqual(document["technical_release_id"], self.release_id)
        self.assertEqual(len(document["responses"]), 22)
        self.assertEqual(set(document["responses"]), operations.RESPONSE_ROLES)
        self.assertEqual(set(document["response_captures"]), operations.RESPONSE_ROLES)
        self.assertTrue(
            all(
                set(record) == {"request", "artifact"}
                for record in document["responses"].values()
            )
        )
        self.assertEqual(
            set(result["technical_release"]["assets"]),
            {
                "technical_evidence",
                "technical_summary",
                "raw_timing",
                "event_profiler",
            },
        )
        self.assertEqual(result["issue_contract_amendment_comment_id"], 800)
        self.assertEqual(result["pr_candidate_insight_comment_id"], 950)
        self.assertEqual(result["graphql_review_total"], 0)
        self.assertFalse(result["final_acceptance"])
        self.assertEqual(
            result["final_acceptance_authority"],
            "same-process-live-verification-required",
        )
        self.assertFalse(result["publication"]["offline_final_acceptance"])
        self.assertEqual(
            [record["id"] for record in result["superseded_owner_comments"]],
            [910001, 910002, 910003, 910004],
        )

    def test_raw_tampering_cannot_hide_operational_blockers(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        cases = []
        high_alert = copy.deepcopy(self.responses)
        high_alert["codeql_alerts"] = [
            [
                {
                    "number": 4,
                    "state": "open",
                    "rule": {"severity": "warning", "security_severity_level": "high"},
                    "most_recent_instance": {
                        "ref": f"refs/pull/{self.number}/merge",
                        "commit_sha": self.merge_sha,
                    },
                }
            ]
        ]
        cases.append(("security", high_alert))
        error_analysis = copy.deepcopy(self.responses)
        error_analysis["codeql_analyses"][0][0]["error"] = "incomplete upload"
        cases.append(("analysis", error_analysis))
        wrong_check = copy.deepcopy(self.responses)
        wrong_check["check_runs"][0]["check_runs"][0]["head_sha"] = self.merge_sha
        cases.append(("check", wrong_check))
        wrong_workflow_pr = copy.deepcopy(self.responses)
        wrong_workflow_pr["ci_run"]["pull_requests"][0]["number"] = 999
        cases.append(("workflow-pr", wrong_workflow_pr))
        forged_identical = copy.deepcopy(self.responses)
        forged_identical["base_compare"].update(
            {"status": "identical", "ahead_by": 0, "total_commits": 0, "commits": []}
        )
        cases.append(("compare-identity", forged_identical))
        masked_change_request = copy.deepcopy(self.responses)
        masked_change_request["reviews"] = [
            [
                {
                    "id": 1,
                    "pull_request_url": (
                        f"https://api.github.com/repos/{operations.REPOSITORY}/"
                        f"pulls/{self.number}"
                    ),
                    "user": {"login": "reviewer"},
                    "state": "CHANGES_REQUESTED",
                    "submitted_at": "2026-08-31T00:01:00Z",
                },
                {
                    "id": 2,
                    "pull_request_url": (
                        f"https://api.github.com/repos/{operations.REPOSITORY}/"
                        f"pulls/{self.number}"
                    ),
                    "user": {"login": "reviewer"},
                    "state": "COMMENTED",
                    "submitted_at": "2026-08-31T00:02:00Z",
                },
            ]
        ]
        cases.append(("comment-masks-change-request", masked_change_request))
        unresolved = copy.deepcopy(self.responses)
        unresolved["review_threads"][0]["data"]["repository"]["pullRequest"][
            "reviewThreads"
        ]["nodes"][0]["isResolved"] = False
        cases.append(("review", unresolved))
        for label, raw in cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(document, raw, self.candidate)

    def test_capture_rejects_any_pull_request_other_than_167(self):
        with self.assertRaisesRegex(operations.EvidenceError, "must be #167"):
            operations.capture_operations(
                repository=operations.REPOSITORY,
                pull_request_number=166,
                ci_run_id=10,
                codeql_run_id=20,
                technical_release_tag=self.release_tag,
                publication_receipt=self.publication_receipt_path,
                output_directory=self.root / "wrong-pr",
            )

    def test_capture_rejects_obsolete_candidate_before_github_requests(self):
        rejected = copy.deepcopy(self.candidate)
        rejected["candidate_git_commit"] = operations.REJECTED_CANDIDATE_SHA
        with (
            mock.patch.object(operations, "candidate_evidence", return_value=rejected),
            mock.patch.object(operations, "_github_api_capture") as github_api,
            self.assertRaisesRegex(operations.EvidenceError, "rejected candidate"),
        ):
            operations.capture_operations(
                repository=operations.REPOSITORY,
                pull_request_number=operations.PULL_REQUEST_NUMBER,
                ci_run_id=10,
                codeql_run_id=20,
                technical_release_tag=(
                    operations.TECHNICAL_RELEASE_TAG_PREFIX
                    + operations.REJECTED_CANDIDATE_SHA
                ),
                publication_receipt=self.publication_receipt_path,
                output_directory=self.root / "obsolete",
            )
        github_api.assert_not_called()

    def test_capture_rejects_noncanonical_release_tag_before_github_requests(self):
        with (
            mock.patch.object(
                operations, "candidate_evidence", return_value=self.candidate
            ),
            mock.patch.object(operations, "_github_api_capture") as github_api,
            self.assertRaisesRegex(operations.EvidenceError, "non-v FINAL_SHA"),
        ):
            operations.capture_operations(
                repository=operations.REPOSITORY,
                pull_request_number=operations.PULL_REQUEST_NUMBER,
                ci_run_id=10,
                codeql_run_id=20,
                technical_release_tag="v1.0.0",
                publication_receipt=self.publication_receipt_path,
                output_directory=self.root / "version-tag",
            )
        github_api.assert_not_called()

    def test_wrong_pr_or_issue_identity_fails_closed(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        cases = []
        for key, wrong in (
            ("repository", "someone/gmes"),
            ("target_issue_number", 124),
            ("handoff_issue_number", 116),
            ("pull_request_number", 168),
        ):
            changed = copy.deepcopy(document)
            changed[key] = wrong
            cases.append((f"index-{key}", changed, self.responses))
        wrong_target = copy.deepcopy(self.responses)
        wrong_target["issue_123"]["number"] = 124
        cases.append(("target-response", document, wrong_target))
        pull_shaped_target = copy.deepcopy(self.responses)
        pull_shaped_target["issue_123"]["pull_request"] = {
            "url": "https://api.github.com/repos/ruddyscent/gmes/pulls/123"
        }
        cases.append(("target-is-pull-request", document, pull_shaped_target))
        wrong_handoff = copy.deepcopy(self.responses)
        wrong_handoff["issue_115"]["number"] = 116
        cases.append(("handoff-response", document, wrong_handoff))
        wrong_pull = copy.deepcopy(self.responses)
        wrong_pull["pull_request"]["number"] = 168
        cases.append(("pull-response", document, wrong_pull))
        for label, changed_document, raw in cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(changed_document, raw, self.candidate)

    def test_closing_references_require_exactly_issue_123(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        cases = []
        missing = copy.deepcopy(self.responses)
        connection = missing["review_threads"][0]["data"]["repository"]["pullRequest"][
            "closingIssuesReferences"
        ]
        connection["totalCount"] = 0
        connection["nodes"] = []
        cases.append(("missing", missing))
        extra = copy.deepcopy(self.responses)
        connection = extra["review_threads"][0]["data"]["repository"]["pullRequest"][
            "closingIssuesReferences"
        ]
        connection["totalCount"] = 2
        connection["nodes"].append(
            {
                "number": 124,
                "repository": {"nameWithOwner": operations.REPOSITORY},
            }
        )
        cases.append(("extra", extra))
        wrong = copy.deepcopy(self.responses)
        wrong["review_threads"][0]["data"]["repository"]["pullRequest"][
            "closingIssuesReferences"
        ]["nodes"][0]["number"] = 124
        cases.append(("wrong-target", wrong))
        wrong_repository = copy.deepcopy(self.responses)
        wrong_repository["review_threads"][0]["data"]["repository"]["pullRequest"][
            "closingIssuesReferences"
        ]["nodes"][0]["repository"]["nameWithOwner"] = "someone/gmes"
        cases.append(("wrong-repository", wrong_repository))
        for label, raw in cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(document, raw, self.candidate)

    def test_candidate_commit_must_match_final_sha_and_be_verified(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        cases = []
        unverified = copy.deepcopy(self.responses)
        unverified["candidate_commit"]["commit"]["verification"]["verified"] = False
        cases.append(("unverified", unverified))
        invalid_reason = copy.deepcopy(self.responses)
        invalid_reason["candidate_commit"]["commit"]["verification"][
            "reason"
        ] = "unsigned"
        cases.append(("reason", invalid_reason))
        mismatched = copy.deepcopy(self.responses)
        mismatched["candidate_commit"]["sha"] = "e" * 40
        cases.append(("sha", mismatched))
        wrong_url = copy.deepcopy(self.responses)
        wrong_url["candidate_commit"]["url"] = (
            "https://api.github.com/repos/someone/gmes/commits/" + "a" * 40
        )
        cases.append(("url", wrong_url))
        wrong_html_url = copy.deepcopy(self.responses)
        wrong_html_url["candidate_commit"]["html_url"] = (
            "https://github.com/someone/gmes/commit/" + "a" * 40
        )
        cases.append(("html-url", wrong_html_url))
        missing_verification_time = copy.deepcopy(self.responses)
        missing_verification_time["candidate_commit"]["commit"]["verification"][
            "verified_at"
        ] = None
        cases.append(("verified-at", missing_verification_time))
        for label, raw in cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(document, raw, self.candidate)

    def test_issue_115_requires_completed_state_reason(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        cases = []
        for value in (None, "not_planned", "COMPLETED"):
            changed = copy.deepcopy(self.responses)
            changed["issue_115"]["state_reason"] = value
            cases.append((str(value), changed))
        missing = copy.deepcopy(self.responses)
        del missing["issue_115"]["state_reason"]
        cases.append(("missing", missing))
        deleted_checklist_item = copy.deepcopy(self.responses)
        deleted_checklist_item["issue_115"]["body"] = "\n".join(
            line
            for line in deleted_checklist_item["issue_115"]["body"].splitlines()
            if "torch.utils.benchmark" not in line
        )
        cases.append(("deleted-runtime-item", deleted_checklist_item))
        for label, raw in cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(document, raw, self.candidate)

    def test_technical_release_and_asset_ledger_fail_closed(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        cases = []

        wrong_target = copy.deepcopy(self.responses)
        wrong_target["technical_release"]["target_commitish"] = "e" * 40
        cases.append(("release-target", wrong_target))

        annotated_tag = copy.deepcopy(self.responses)
        annotated_tag["technical_release_tag"]["object"]["type"] = "tag"
        cases.append(("annotated-tag", annotated_tag))

        wrong_tag_target = copy.deepcopy(self.responses)
        wrong_tag_target["technical_release_tag"]["object"]["sha"] = "e" * 40
        cases.append(("tag-target", wrong_tag_target))

        draft = copy.deepcopy(self.responses)
        draft["technical_release"]["draft"] = True
        cases.append(("draft", draft))

        prerelease = copy.deepcopy(self.responses)
        prerelease["technical_release"]["prerelease"] = True
        cases.append(("prerelease", prerelease))

        mutable = copy.deepcopy(self.responses)
        mutable["technical_release"]["immutable"] = False
        cases.append(("mutable", mutable))

        wrong_author = copy.deepcopy(self.responses)
        wrong_author["technical_release"]["author"]["login"] = SYNTHETIC_NON_OWNER_LOGIN
        cases.append(("release-author", wrong_author))

        published_after_comments = copy.deepcopy(self.responses)
        published_after_comments["technical_release"][
            "published_at"
        ] = "2026-08-31T02:00:00Z"
        cases.append(("release-after-comments", published_after_comments))

        missing_asset = copy.deepcopy(self.responses)
        missing_asset["technical_release"]["assets"].pop()
        missing_asset["technical_release_assets"][0].pop()
        cases.append(("missing-asset", missing_asset))

        duplicate_name = copy.deepcopy(self.responses)
        duplicate_name["technical_release_assets"][0][1]["name"] = duplicate_name[
            "technical_release_assets"
        ][0][0]["name"]
        cases.append(("duplicate-name", duplicate_name))

        duplicate_id = copy.deepcopy(self.responses)
        duplicate_id["technical_release_assets"][0][1]["id"] = duplicate_id[
            "technical_release_assets"
        ][0][0]["id"]
        cases.append(("duplicate-id", duplicate_id))

        zero_size = copy.deepcopy(self.responses)
        zero_size["technical_release_assets"][0][0]["size"] = 0
        cases.append(("zero-size", zero_size))

        bad_digest = copy.deepcopy(self.responses)
        bad_digest["technical_release_assets"][0][0]["digest"] = "sha512:" + "1" * 64
        cases.append(("digest-algorithm", bad_digest))

        wrong_uploader = copy.deepcopy(self.responses)
        wrong_uploader["technical_release_assets"][0][0]["uploader"][
            "login"
        ] = SYNTHETIC_NON_OWNER_LOGIN
        cases.append(("uploader", wrong_uploader))

        ledger_drift = copy.deepcopy(self.responses)
        ledger_drift["technical_release_assets"][0][0]["digest"] = "sha256:" + "f" * 64
        cases.append(("embedded-ledger-drift", ledger_drift))

        off_domain = copy.deepcopy(self.responses)
        evidence_name = "issue-123-public-technical-evidence.zip"
        expected_url = self._asset_url(evidence_name)
        evil_url = "https://github.com.evil/ruddyscent/gmes/" + evidence_name
        for collection in (
            off_domain["technical_release"]["assets"],
            off_domain["technical_release_assets"][0],
        ):
            record = next(
                asset for asset in collection if asset["name"] == evidence_name
            )
            record["browser_download_url"] = evil_url
        off_domain["pull_request_comments"][0][0]["body"] = off_domain[
            "pull_request_comments"
        ][0][0]["body"].replace(
            f"TECHNICAL_EVIDENCE_ASSET_URL={expected_url}",
            f"TECHNICAL_EVIDENCE_ASSET_URL={evil_url}",
        )
        cases.append(("synchronized-off-domain-url", off_domain))

        mismatched_comment_url = copy.deepcopy(self.responses)
        timing_url = self._asset_url("issue-115-raw-timing.json")
        profiler_url = self._asset_url("issue-115-event-level-profiler.json")
        mismatched_comment_url["issue_115_comments"][0][0][
            "body"
        ] = mismatched_comment_url["issue_115_comments"][0][0]["body"].replace(
            f"RAW_TIMING_ASSET_URL={timing_url}",
            f"RAW_TIMING_ASSET_URL={profiler_url}",
        )
        cases.append(("cross-asset-url", mismatched_comment_url))

        for label, raw in cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(document, raw, self.candidate)

    def test_structured_owner_contracts_reject_field_and_authorship_attacks(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        specifications = (
            (
                "issue_123_comments",
                "issue_123",
                "GMES_ISSUE_123_FINAL_CONTRACT_AMENDMENT_V2",
                "SOLVER_ABI",
                "torch-fdtd-regions-v14",
                123,
                "issues",
            ),
            (
                "pull_request_comments",
                "pull_request",
                "GMES_PR_167_FINAL_CANDIDATE_INSIGHT_V2",
                "CI_RUN_URL",
                "https://github.com/ruddyscent/gmes/actions/runs/11",
                167,
                "pull",
            ),
            (
                "issue_115_comments",
                "issue_115",
                "GMES_ISSUE_115_FINAL_RUNTIME_HANDOFF_V2",
                "RAW_TIMING_ASSET_SHA256",
                "f" * 64,
                115,
                "issues",
            ),
        )
        cases = []
        for (
            role,
            parent,
            marker,
            field,
            wrong_value,
            issue_number,
            html_kind,
        ) in specifications:
            absent = copy.deepcopy(self.responses)
            absent_comment = self._comment_with_marker(absent, role, marker)
            absent_comment["body"] = absent_comment["body"].replace(
                marker, marker.removesuffix("V2") + "V1"
            )
            cases.append((f"{role}-absent", absent))

            omitted = copy.deepcopy(self.responses)
            omitted_comment = self._comment_with_marker(omitted, role, marker)
            omitted_comment["body"] = "\n".join(
                line
                for line in omitted_comment["body"].splitlines()
                if not line.startswith(f"{field}=")
            )
            cases.append((f"{role}-omitted-field", omitted))

            wrong = copy.deepcopy(self.responses)
            wrong_comment = self._comment_with_marker(wrong, role, marker)
            wrong_comment["body"] = "\n".join(
                f"{field}={wrong_value}" if line.startswith(f"{field}=") else line
                for line in wrong_comment["body"].splitlines()
            )
            cases.append((f"{role}-wrong-field", wrong))

            duplicate_field = copy.deepcopy(self.responses)
            duplicate_field_comment = self._comment_with_marker(
                duplicate_field, role, marker
            )
            original = next(
                line
                for line in duplicate_field_comment["body"].splitlines()
                if line.startswith(f"{field}=")
            )
            duplicate_field_comment["body"] += f"\n{original}"
            cases.append((f"{role}-duplicate-field", duplicate_field))

            duplicate = copy.deepcopy(self.responses)
            duplicate_comment = self._comment_with_marker(duplicate, role, marker)
            second = self._comment(
                duplicate_comment["id"] + 1000,
                issue_number,
                duplicate_comment["body"],
                owner=False,
                html_kind=html_kind,
            )
            duplicate[role][0].append(second)
            duplicate[parent]["comments"] += 1
            cases.append((f"{role}-duplicate", duplicate))

            non_owner = copy.deepcopy(self.responses)
            self._comment_with_marker(non_owner, role, marker)[
                "author_association"
            ] = "CONTRIBUTOR"
            cases.append((f"{role}-non-owner", non_owner))

            wrong_login = copy.deepcopy(self.responses)
            self._comment_with_marker(wrong_login, role, marker)["user"] = {
                "login": SYNTHETIC_NON_OWNER_LOGIN
            }
            cases.append((f"{role}-wrong-owner-login", wrong_login))

            repeated_literal = copy.deepcopy(self.responses)
            self._comment_with_marker(repeated_literal, role, marker)[
                "body"
            ] += f"\n{marker}"
            cases.append((f"{role}-repeated-literal", repeated_literal))

            repeated_sha = copy.deepcopy(self.responses)
            self._comment_with_marker(repeated_sha, role, marker)[
                "body"
            ] += f"\nFINAL_SHA={self.candidate['candidate_git_commit']}"
            cases.append((f"{role}-repeated-sha", repeated_sha))

            quoted_template = copy.deepcopy(self.responses)
            quoted_comment = self._comment_with_marker(quoted_template, role, marker)
            quoted_comment["body"] = "```text\n" + quoted_comment["body"] + "\n```"
            cases.append((f"{role}-quoted-template", quoted_template))

            split = copy.deepcopy(self.responses)
            split_comment = self._comment_with_marker(split, role, marker)
            lines = split_comment["body"].splitlines()
            midpoint = 1 + (len(lines) - 1) // 2
            split_comment["body"] = "\n".join(lines[:midpoint])
            split[role][0].append(
                self._comment(
                    split_comment["id"] + 2000,
                    issue_number,
                    "\n".join(lines[midpoint:]),
                    html_kind=html_kind,
                )
            )
            split[parent]["comments"] += 1
            cases.append((f"{role}-cross-comment-fields", split))

        cross_substitution = copy.deepcopy(self.responses)
        self._comment_with_marker(
            cross_substitution,
            "issue_123_comments",
            operations.ISSUE_CONTRACT_AMENDMENT_MARKER,
        )["body"] = self._comment_with_marker(
            self.responses,
            "pull_request_comments",
            operations.PR_CANDIDATE_INSIGHT_MARKER,
        )[
            "body"
        ]
        cases.append(("PR-contract-substituted-for-amendment", cross_substitution))
        cross_substitution = copy.deepcopy(self.responses)
        self._comment_with_marker(
            cross_substitution,
            "pull_request_comments",
            operations.PR_CANDIDATE_INSIGHT_MARKER,
        )["body"] = self._comment_with_marker(
            self.responses, "issue_115_comments", operations.HANDOFF_MARKER
        )[
            "body"
        ]
        cases.append(("handoff-substituted-for-PR-contract", cross_substitution))
        cross_substitution = copy.deepcopy(self.responses)
        self._comment_with_marker(
            cross_substitution, "issue_115_comments", operations.HANDOFF_MARKER
        )["body"] = self._comment_with_marker(
            self.responses,
            "issue_123_comments",
            operations.ISSUE_CONTRACT_AMENDMENT_MARKER,
        )[
            "body"
        ]
        cases.append(("amendment-substituted-for-handoff", cross_substitution))

        reused_comment_id = copy.deepcopy(self.responses)
        reused = self._comment_with_marker(
            reused_comment_id,
            "issue_123_comments",
            operations.ISSUE_CONTRACT_AMENDMENT_MARKER,
        )
        reused["id"] = 950
        reused["url"] = (
            "https://api.github.com/repos/ruddyscent/gmes/issues/comments/950"
        )
        reused["html_url"] = (
            "https://github.com/ruddyscent/gmes/issues/123#issuecomment-950"
        )
        cases.append(("cross-stream-comment-id", reused_comment_id))

        obsolete_baselines = copy.deepcopy(self.responses)
        obsolete_comment = self._comment_with_marker(
            obsolete_baselines,
            "issue_123_comments",
            operations.ISSUE_CONTRACT_AMENDMENT_MARKER,
        )
        obsolete_comment["body"] = (
            obsolete_comment["body"]
            .replace(
                operations.BASELINE_V3_ONE_SHA256,
                "a" * 64,
            )
            .replace(
                operations.BASELINE_V3_PHYSICAL_SHA256,
                "c" * 64,
            )
        )
        cases.append(("obsolete-owner-baseline-pair", obsolete_baselines))

        for label, raw in cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(document, raw, self.candidate)

    def test_superseded_owner_comment_provenance_is_exact_and_complete(self):
        self.assertEqual(
            tuple(
                specification["field"]
                for specification in self.production_superseded_comments
            ),
            tuple(SYNTHETIC_SUPERSEDED_IDS),
        )
        self.assertEqual(
            tuple(
                specification["id"]
                for specification in operations.SUPERSEDED_OWNER_COMMENTS
            ),
            tuple(SYNTHETIC_SUPERSEDED_IDS.values()),
        )
        self.assertTrue(
            all(
                specification["id"] not in SYNTHETIC_SUPERSEDED_IDS.values()
                and len(specification["body_sha256"]) == 64
                and specification["required_fragments"]
                for specification in self.production_superseded_comments
            )
        )
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        result = operations.evaluate_operations(
            document, self.responses, self.candidate
        )
        records = {
            record["id"]: record for record in result["superseded_owner_comments"]
        }
        self.assertEqual(set(records), set(self.superseded_comment_bodies))
        for specification in operations.SUPERSEDED_OWNER_COMMENTS:
            identifier = specification["id"]
            record = records[identifier]
            self.assertEqual(record["role"], specification["role"])
            self.assertEqual(record["stream"], specification["stream"])
            self.assertEqual(record["owner_login"], SYNTHETIC_OWNER_LOGIN)
            self.assertEqual(
                record["body_sha256"],
                hashlib.sha256(
                    self.superseded_comment_bodies[identifier].encode()
                ).hexdigest(),
            )

    def test_superseded_owner_comments_cannot_be_absent_or_substituted(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        parent_by_stream = {
            "issue_123_comments": "issue_123",
            "pull_request_comments": "pull_request",
        }
        for specification in operations.SUPERSEDED_OWNER_COMMENTS:
            stream = specification["stream"]
            identifier = specification["id"]
            absent = copy.deepcopy(self.responses)
            absent[stream][0] = [
                comment for comment in absent[stream][0] if comment["id"] != identifier
            ]
            absent[parent_by_stream[stream]]["comments"] -= 1
            with (
                self.subTest(identifier=identifier, attack="absent"),
                self.assertRaises(operations.EvidenceError),
            ):
                self._evaluate(document, absent)

            substituted = copy.deepcopy(self.responses)
            comment = next(
                item
                for page in substituted[stream]
                for item in page
                if item["id"] == identifier
            )
            comment["body"] += " substituted"
            with (
                self.subTest(identifier=identifier, attack="content"),
                self.assertRaises(operations.EvidenceError),
            ):
                self._evaluate(document, substituted)

            non_owner = copy.deepcopy(self.responses)
            comment = next(
                item
                for page in non_owner[stream]
                for item in page
                if item["id"] == identifier
            )
            comment["author_association"] = "CONTRIBUTOR"
            with (
                self.subTest(identifier=identifier, attack="association"),
                self.assertRaises(operations.EvidenceError),
            ):
                self._evaluate(document, non_owner)

    def test_operational_chronology_fails_closed(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        cases = []

        closed_after_handoff = copy.deepcopy(self.responses)
        closed_after_handoff["issue_115"]["closed_at"] = "2026-09-03T01:00:01Z"
        closed_after_handoff["issue_115"]["updated_at"] = "2026-09-03T01:00:02Z"
        cases.append(("issue-115-handoff", closed_after_handoff))

        unverified_insight = copy.deepcopy(self.responses)
        unverified_insight["candidate_commit"]["commit"]["verification"][
            "verified_at"
        ] = "2026-09-03T01:20:01Z"
        cases.append(("commit-verification", unverified_insight))

        incomplete_run_insight = copy.deepcopy(self.responses)
        incomplete_run_insight["ci_run"]["updated_at"] = "2026-09-03T01:20:01Z"
        cases.append(("CI-run-completion", incomplete_run_insight))

        incomplete_job_insight = copy.deepcopy(self.responses)
        incomplete_job_insight["codeql_jobs"][0]["jobs"][0][
            "completed_at"
        ] = "2026-09-03T01:20:01Z"
        cases.append(("CodeQL-job-completion", incomplete_job_insight))

        reversed_creation = copy.deepcopy(self.responses)
        reversed_creation["ruleset"]["created_at"] = "2026-08-23T00:00:00Z"
        cases.append(("created-after-updated", reversed_creation))

        reversed_comment = copy.deepcopy(self.responses)
        handoff = self._comment_with_marker(
            reversed_comment, "issue_115_comments", operations.HANDOFF_MARKER
        )
        handoff["updated_at"] = "2026-09-03T00:59:59Z"
        cases.append(("comment-created-after-updated", reversed_comment))

        for label, raw in cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                self._evaluate(document, raw)

    def test_publication_receipt_identity_substitution_fails_closed(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        cases = []

        wrong_candidate = copy.deepcopy(self.publication_receipt_document)
        wrong_candidate["bindings"]["final_sha"] = "f" * 40
        cases.append(("FINAL_SHA", wrong_candidate))

        wrong_asset_ledger = copy.deepcopy(self.publication_receipt_document)
        wrong_asset_ledger["asset_ledger"][0]["sha256"] = "f" * 64
        wrong_asset_ledger["hashes"]["asset_ledger_sha256"] = hashlib.sha256(
            operations._canonical_json_bytes(wrong_asset_ledger["asset_ledger"])
        ).hexdigest()
        cases.append(("external-byte-ledger", wrong_asset_ledger))

        wrong_release = copy.deepcopy(self.publication_receipt_document)
        wrong_release["release_capture"]["release_id"] = 31
        wrong_release["release_capture"][
            "api_url"
        ] = "https://api.github.com/repos/ruddyscent/gmes/releases/31"
        for asset in wrong_release["release_capture"]["assets"]:
            asset["release_id"] = 31
        wrong_release["hashes"]["release_capture_sha256"] = hashlib.sha256(
            operations._canonical_json_bytes(wrong_release["release_capture"])
        ).hexdigest()
        cases.append(("release-id", wrong_release))

        boolean_schema = copy.deepcopy(self.publication_receipt_document)
        boolean_schema["schema_version"] = True
        cases.append(("boolean-schema", boolean_schema))

        for label, receipt in cases:
            changed = self._replace_receipt_document(document, receipt)
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(changed, self.responses, self.candidate)

    def test_complete_review_thread_pagination_is_recomputed(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        raw = copy.deepcopy(self.responses)
        first = raw["review_threads"][0]
        connection = first["data"]["repository"]["pullRequest"]["reviewThreads"]
        connection["totalCount"] = 101
        connection["pageInfo"] = {"hasNextPage": True, "endCursor": "page-1"}
        connection["nodes"] = [
            {"id": f"thread-{index}", "isResolved": True} for index in range(100)
        ]
        second = copy.deepcopy(first)
        second_connection = second["data"]["repository"]["pullRequest"]["reviewThreads"]
        second_connection["pageInfo"] = {
            "hasNextPage": False,
            "endCursor": "page-2",
        }
        second_connection["nodes"] = [{"id": "thread-100", "isResolved": True}]
        raw["review_threads"].append(second)
        result = self._evaluate(document, raw)
        self.assertEqual(result["review_threads"], 101)

    def test_complete_issue_comment_pagination_is_recomputed(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        raw = copy.deepcopy(self.responses)
        final_page = raw["issue_123_comments"][0]
        marker = self._comment_with_marker(
            raw,
            "issue_123_comments",
            operations.ISSUE_CONTRACT_AMENDMENT_MARKER,
        )
        ordinary = [
            self._comment(
                1000 + index,
                operations.TARGET_ISSUE_NUMBER,
                f"ordinary comment {index}",
                owner=False,
            )
            for index in range(100)
        ]
        raw["issue_123_comments"] = [ordinary, final_page]
        raw["issue_123"]["comments"] = 100 + len(final_page)
        result = self._evaluate(document, raw)
        self.assertEqual(result["issue_contract_amendment_comment_id"], marker["id"])

    def test_deleted_terminal_rest_pages_fail_closed(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())

        def review(identifier):
            return {
                "id": identifier,
                "pull_request_url": (
                    f"https://api.github.com/repos/{operations.REPOSITORY}/pulls/"
                    f"{self.number}"
                ),
                "user": {"login": f"reviewer-{identifier}"},
                "state": "COMMENTED",
                "submitted_at": "2026-09-02T23:00:00Z",
            }

        def alert(identifier):
            return {
                "number": identifier,
                "state": "open",
                "rule": {
                    "severity": "warning",
                    "security_severity_level": "medium",
                },
                "most_recent_instance": {
                    "ref": f"refs/pull/{self.number}/merge",
                    "commit_sha": self.merge_sha,
                },
                "created_at": "2026-09-02T23:00:00Z",
                "updated_at": "2026-09-02T23:01:00Z",
            }

        cases = {}
        reviews = copy.deepcopy(self.responses)
        review_records = [review(10000 + index) for index in range(101)]
        reviews["reviews"] = [review_records[:100], review_records[100:]]
        reviews["review_threads"][0]["data"]["repository"]["pullRequest"]["reviews"][
            "totalCount"
        ] = 101
        cases["reviews"] = reviews

        analyses = copy.deepcopy(self.responses)
        analysis_records = [*analyses["codeql_analyses"][0]]
        analysis_records.extend({"ignored": index} for index in range(99))
        analyses["codeql_analyses"] = [
            analysis_records[:100],
            analysis_records[100:],
        ]
        cases["codeql_analyses"] = analyses

        alerts = copy.deepcopy(self.responses)
        alert_records = [alert(20000 + index) for index in range(101)]
        alerts["codeql_alerts"] = [alert_records[:100], alert_records[100:]]
        cases["codeql_alerts"] = alerts

        for role, raw in cases.items():
            with self.subTest(role=role):
                complete_document = self._coherent_document(document, raw)
                operations.evaluate_operations(complete_document, raw, self.candidate)
                deleted_raw = copy.deepcopy(raw)
                deleted_raw[role].pop()
                deleted_document = copy.deepcopy(complete_document)
                capture = deleted_document["response_captures"][role]
                capture["pages"].pop()
                canonical = operations._canonical_json_bytes(deleted_raw[role])
                capture["canonical_response_size_bytes"] = len(canonical)
                capture["canonical_response_sha256"] = hashlib.sha256(
                    canonical
                ).hexdigest()
                with self.assertRaisesRegex(
                    operations.EvidenceError, "body ledger|final-page/no-next"
                ):
                    operations.evaluate_operations(
                        deleted_document, deleted_raw, self.candidate
                    )

    def test_rest_page_ledger_rejects_route_filter_and_last_substitution(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        raw = copy.deepcopy(self.responses)
        ordinary = [
            self._comment(
                30000 + index,
                operations.TARGET_ISSUE_NUMBER,
                f"ordinary comment {index}",
                owner=False,
            )
            for index in range(100)
        ]
        final_page = raw["issue_123_comments"][0]
        raw["issue_123_comments"] = [ordinary, final_page]
        raw["issue_123"]["comments"] = 100 + len(final_page)
        complete = self._coherent_document(document, raw)
        operations.evaluate_operations(complete, raw, self.candidate)
        role = "issue_123_comments"
        original = complete["response_captures"][role]["pages"][0]
        endpoint = complete["responses"][role]["request"]["endpoint"]
        attacks = {
            "suffix-route": (
                "https://api.github.com/evil/issues/123/comments?page=2&per_page=100"
            ),
            "extra-filter": (
                f"https://api.github.com/{endpoint}?page=2&per_page=100&since=now"
            ),
        }
        for label, url in attacks.items():
            changed = copy.deepcopy(complete)
            page = changed["response_captures"][role]["pages"][0]
            page["headers"]["link"]["next"] = url
            page["next"]["value"] = url
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(changed, raw, self.candidate)

        wrong_last = copy.deepcopy(complete)
        wrong_last_page = wrong_last["response_captures"][role]["pages"][0]
        wrong_last_page["headers"]["link"][
            "last"
        ] = f"https://api.github.com/{endpoint}?page=3&per_page=100"
        self.assertTrue(original["has_next"])
        with self.assertRaisesRegex(operations.EvidenceError, "final page"):
            operations.evaluate_operations(wrong_last, raw, self.candidate)

    def test_malformed_pagination_and_request_metadata_fail_closed(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        raw_cases = []
        flat_comments = copy.deepcopy(self.responses)
        flat_comments["issue_123_comments"] = flat_comments["issue_123_comments"][0]
        raw_cases.append(("flat-comments", flat_comments))
        trailing_page = copy.deepcopy(self.responses)
        marker = trailing_page["pull_request_comments"][0][0]
        trailing_page["pull_request_comments"] = [
            [
                marker,
                *[
                    self._comment(
                        2000 + index,
                        self.number,
                        f"ordinary PR comment {index}",
                        owner=False,
                    )
                    for index in range(99)
                ],
            ],
            [],
        ]
        trailing_page["pull_request"]["comments"] = operations.PAGE_SIZE
        raw_cases.append(("trailing-page", trailing_page))
        wrong_total = copy.deepcopy(self.responses)
        wrong_total["ci_jobs"][0]["total_count"] = 3
        raw_cases.append(("object-total", wrong_total))
        graphql_object = copy.deepcopy(self.responses)
        graphql_object["review_threads"] = graphql_object["review_threads"][0]
        raw_cases.append(("graphql-not-pages", graphql_object))
        open_graphql_page = copy.deepcopy(self.responses)
        open_graphql_page["review_threads"][0]["data"]["repository"]["pullRequest"][
            "reviewThreads"
        ]["pageInfo"]["hasNextPage"] = True
        raw_cases.append(("graphql-open-page", open_graphql_page))
        missing_cursor = copy.deepcopy(self.responses)
        missing_cursor["review_threads"][0]["data"]["repository"]["pullRequest"][
            "reviewThreads"
        ]["pageInfo"]["endCursor"] = None
        raw_cases.append(("graphql-missing-cursor", missing_cursor))
        missing_response = copy.deepcopy(self.responses)
        del missing_response["candidate_commit"]
        raw_cases.append(("raw-response-closure", missing_response))
        for label, raw in raw_cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(document, raw, self.candidate)

        document_cases = []
        wrong_pagination = copy.deepcopy(document)
        wrong_pagination["responses"]["issue_123_comments"]["request"][
            "paginated"
        ] = False
        document_cases.append(("request-pagination", wrong_pagination))
        wrong_jq = copy.deepcopy(document)
        wrong_jq["responses"]["issue_123_comments"]["request"]["jq"] = None
        document_cases.append(("request-jq", wrong_jq))
        wrong_endpoint = copy.deepcopy(document)
        wrong_endpoint["responses"]["issue_123"]["request"][
            "endpoint"
        ] = "repos/ruddyscent/gmes/issues/124"
        document_cases.append(("request-endpoint", wrong_endpoint))
        wrong_method = copy.deepcopy(document)
        wrong_method["responses"]["candidate_commit"]["request"]["method"] = "POST"
        document_cases.append(("request-method", wrong_method))
        wrong_headers = copy.deepcopy(document)
        wrong_headers["responses"]["technical_release"]["request"]["headers"][
            "X-GitHub-Api-Version"
        ] = "2020-01-01"
        document_cases.append(("request-headers", wrong_headers))
        wrong_parameters = copy.deepcopy(document)
        wrong_parameters["responses"]["issue_123_comments"]["request"]["parameters"] = {
            "per_page": "99"
        }
        document_cases.append(("request-parameters", wrong_parameters))
        wrong_release_endpoint = copy.deepcopy(document)
        wrong_release_endpoint["responses"]["technical_release_assets"]["request"][
            "endpoint"
        ] = "repos/ruddyscent/gmes/releases/31/assets"
        document_cases.append(("release-request-endpoint", wrong_release_endpoint))
        wrong_release_parameters = copy.deepcopy(document)
        wrong_release_parameters["responses"]["technical_release_assets"]["request"][
            "parameters"
        ] = {"per_page": "99"}
        document_cases.append(("release-request-parameters", wrong_release_parameters))
        wrong_graphql_flag = copy.deepcopy(document)
        wrong_graphql_flag["responses"]["review_threads"]["request"]["graphql"] = False
        document_cases.append(("request-graphql", wrong_graphql_flag))
        wrong_query = copy.deepcopy(document)
        wrong_query["responses"]["review_threads"]["request"]["query"] += "\n# drift"
        document_cases.append(("graphql-query", wrong_query))
        wrong_variables = copy.deepcopy(document)
        wrong_variables["responses"]["review_threads"]["request"]["variables"][
            "number"
        ] = str(self.number)
        document_cases.append(("graphql-variable-type", wrong_variables))
        wrong_run = copy.deepcopy(document)
        wrong_run["ci_run_id"] = 11
        document_cases.append(("run-id", wrong_run))
        wrong_release_id = copy.deepcopy(document)
        wrong_release_id["technical_release_id"] = 31
        document_cases.append(("release-id", wrong_release_id))
        wrong_release_tag = copy.deepcopy(document)
        wrong_release_tag["technical_release_tag"] = "v1.0.0"
        document_cases.append(("release-tag", wrong_release_tag))
        extra_response = copy.deepcopy(document)
        extra_response["responses"]["unexpected"] = copy.deepcopy(
            extra_response["responses"]["issue_123"]
        )
        document_cases.append(("response-closure", extra_response))
        for label, changed in document_cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(changed, self.responses, self.candidate)

    def test_offline_operations_receipt_cannot_grant_final_acceptance(self):
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        result = operations.evaluate_operations(
            document, self.responses, self.candidate
        )
        self.assertFalse(result["final_acceptance"])
        receipt_output = self.root / "offline-must-not-authorize.json"
        with (
            mock.patch.object(
                operations, "candidate_evidence", return_value=self.candidate
            ),
            self.assertRaisesRegex(operations.EvidenceError, "not authenticated"),
        ):
            operations.verify_operations_live(
                index_path=index_path,
                manifest=operations.DEFAULT_MANIFEST,
                publication_policy=self.root / "policy.json",
                publication_policy_sha256="e" * 64,
                publication_assets={
                    role: self.root / name
                    for role, name in operations.TECHNICAL_RELEASE_ASSETS.items()
                },
                receipt_output=receipt_output,
                post_bundle_lease={},
            )
        self.assertFalse(receipt_output.exists())

    def test_live_receipt_preflight_uses_authenticated_b1_roots_not_cli_aliases(self):
        source = self.root / "retained-source-b1"
        reopened = self.root / "retained-reopened-b1"
        index_root = self.root / "operations-index-root"
        public_root = self.root / "public-assets"
        for directory in (source, reopened, index_root, public_root):
            directory.mkdir()
        expectation = operations.AuthenticatedPostBundleExpectation(
            checked_lines=operations.FINAL_CHECKLIST_CHECKED,
            o0_canonical_response_sha256="0" * 64,
            o1_canonical_response_sha256="1" * 64,
            o1_body_sha256="2" * 64,
            o1_updated_at="2026-09-03T01:31:00Z",
            b0_inventory_root="3" * 64,
            b0_reopen_receipt_sha256="4" * 64,
            b0_reopened_at="2026-09-03T01:30:00Z",
            checklist_transition_sha256="5" * 64,
        )
        lease = object.__new__(completion.AuthenticatedPostBundleLease)
        lease._snapshots = SimpleNamespace(
            source_bundle=SimpleNamespace(root=source.resolve()),
            reopened_bundle=SimpleNamespace(root=reopened.resolve()),
        )
        lease._chain = {"post_bundle_expectation": expectation}
        lease.expectation = expectation
        lease._closed = False
        alias = self.root / "retained-source-alias"
        alias.symlink_to(source, target_is_directory=True)
        candidates = (
            source / "receipt.json",
            reopened / "receipt.json",
            alias / "receipt.json",
            source / ".." / source.name / "receipt.json",
        )
        with (
            mock.patch.object(
                completion.AuthenticatedPostBundleLease,
                "require_unchanged",
                return_value=None,
            ),
            mock.patch.object(
                completion.AuthenticatedPostBundleLease,
                "_baseline_authority_set",
                return_value=operations.PRODUCTION_BASELINE_AUTHORITY_SET,
            ),
            mock.patch.object(
                operations,
                "_capture_production_baseline_authority",
            ) as baseline,
        ):
            for candidate in candidates:
                with (
                    self.subTest(candidate=candidate.name),
                    self.assertRaisesRegex(
                        operations.EvidenceError,
                        "overlaps protected evidence",
                    ),
                ):
                    with operations.open_verified_operations_live(
                        index_path=index_root / "index.json",
                        manifest=operations.DEFAULT_MANIFEST,
                        publication_policy=self.root / "policy.json",
                        publication_policy_sha256="e" * 64,
                        publication_assets={
                            role: public_root / name
                            for role, name in operations.TECHNICAL_RELEASE_ASSETS.items()
                        },
                        receipt_output=candidate,
                        post_bundle_lease=lease,
                    ):
                        self.fail("overlapping receipt path was accepted")
        baseline.assert_not_called()
        self.assertEqual(list(source.iterdir()), [])
        self.assertEqual(list(reopened.iterdir()), [])

    def test_live_initial_authority_failure_leaves_no_receipt(self):
        source = self.root / "initial-authority-source-b1"
        reopened = self.root / "initial-authority-reopened-b1"
        index_root = self.root / "initial-authority-index-root"
        public_root = self.root / "initial-authority-public-assets"
        for directory in (source, reopened, index_root, public_root):
            directory.mkdir()
        expectation = operations.AuthenticatedPostBundleExpectation(
            checked_lines=operations.FINAL_CHECKLIST_CHECKED,
            o0_canonical_response_sha256="0" * 64,
            o1_canonical_response_sha256="1" * 64,
            o1_body_sha256="2" * 64,
            o1_updated_at="2026-09-03T01:31:00Z",
            b0_inventory_root="3" * 64,
            b0_reopen_receipt_sha256="4" * 64,
            b0_reopened_at="2026-09-03T01:30:00Z",
            checklist_transition_sha256="5" * 64,
        )
        lease = object.__new__(completion.AuthenticatedPostBundleLease)
        lease._snapshots = SimpleNamespace(
            source_bundle=SimpleNamespace(root=source.resolve()),
            reopened_bundle=SimpleNamespace(root=reopened.resolve()),
        )
        lease._chain = {"post_bundle_expectation": expectation}
        lease.expectation = expectation
        lease._closed = False
        receipt_output = self.private_root / "initial-authority-failure.json"
        with (
            mock.patch.object(
                completion.AuthenticatedPostBundleLease,
                "require_unchanged",
                return_value=None,
            ),
            mock.patch.object(
                completion.AuthenticatedPostBundleLease,
                "_baseline_authority_set",
                return_value=operations.PRODUCTION_BASELINE_AUTHORITY_SET,
            ),
            mock.patch.object(
                operations,
                "_capture_production_baseline_authority",
                side_effect=operations.EvidenceError("GitHub API request failed"),
            ),
            self.assertRaisesRegex(
                operations.EvidenceError, "GitHub API request failed"
            ),
        ):
            with operations.open_verified_operations_live(
                index_path=index_root / "index.json",
                manifest=operations.DEFAULT_MANIFEST,
                publication_policy=self.root / "policy.json",
                publication_policy_sha256="e" * 64,
                publication_assets={
                    role: public_root / name
                    for role, name in operations.TECHNICAL_RELEASE_ASSETS.items()
                },
                receipt_output=receipt_output,
                post_bundle_lease=lease,
            ):
                self.fail("initial authority failure was accepted")
        self.assertFalse(receipt_output.exists())

    def test_live_verification_is_same_process_authority_and_canonical_provenance(self):
        self._use_checked_final_issue()
        _output, index_path, _scope = self._capture()
        renamed_index = index_path.with_name("secret=password-hunter2.json")
        renamed_index.write_bytes(index_path.read_bytes())
        document = json.loads(index_path.read_text())
        fresh_responses = copy.deepcopy(self.responses)
        fresh_captures = copy.deepcopy(document["response_captures"])
        receipt_output = self.private_root / "live-verification.json"
        publication_result = {
            "strict_four_byte_validator": "same-process-invoked",
            "receipt_sha256": document["publication_receipt"]["sha256"],
            "trusted_policy_sha256": "e" * 64,
            "asset_ledger": document["publication_receipt"]["document"]["asset_ledger"],
            "release_identity_anchor": document["publication_receipt"]["document"][
                "release_capture"
            ],
            "bindings": document["publication_receipt"]["document"]["bindings"],
            "execution_claims": [],
            "event_profiler": {
                "contract_id": "fixture",
                "record_count": 1,
                "inventory_sha256": "f" * 64,
                "asset_sha256": "4" * 64,
            },
        }
        post_bundle_expectation = self._post_bundle_expectation(
            document, self.responses
        )
        baseline_validation = self._baseline_validation()
        with (
            mock.patch.object(
                operations, "candidate_evidence", return_value=self.candidate
            ),
            mock.patch.object(
                operations,
                "_fresh_github_capture",
                return_value=(fresh_responses, fresh_captures),
            ) as fresh,
            mock.patch.object(
                operations,
                "_validate_downloaded_publication",
                return_value=publication_result,
            ) as publication_check,
        ):
            baseline_lease = _SyntheticBaselineLease(baseline_validation)
            source_bundle_root = self.root / "source-b1"
            reopened_bundle_root = self.root / "reopened-b1"
            source_bundle_root.mkdir()
            reopened_bundle_root.mkdir()
            receipt = operations._verify_operations_live_with_baseline(
                index_path=renamed_index,
                manifest=operations.DEFAULT_MANIFEST,
                publication_policy=self.root / "policy.json",
                publication_policy_sha256="e" * 64,
                publication_assets={
                    role: self.root / name
                    for role, name in operations.TECHNICAL_RELEASE_ASSETS.items()
                },
                receipt_output=receipt_output,
                post_bundle_expectation=post_bundle_expectation,
                source_bundle_root=source_bundle_root,
                reopened_bundle_root=reopened_bundle_root,
                baseline_lease=baseline_lease,
            )
        fresh.assert_called_once_with(document)
        publication_check.assert_called_once()
        self.assertGreaterEqual(baseline_lease.require_count, 4)
        self.assertEqual(
            receipt["schema_version"],
            operations.LIVE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
        )
        self.assertEqual(
            receipt["post_bundle_acknowledgment"],
            self._acknowledgment(post_bundle_expectation),
        )
        self.assertEqual(receipt["baseline_validation"], baseline_validation)
        self.assertTrue(receipt["same_process_live_accepted"])
        self.assertFalse(receipt["receipt_replay_authority"])
        self.assertEqual(
            receipt["authority"], "same-process-authenticated-gh-live-verification"
        )
        self.assertEqual(len(receipt["queries"]), len(operations.RESPONSE_ROLE_ORDER))
        self.assertEqual(
            set(receipt["operations_index"]),
            {"size_bytes", "sha256"},
        )
        self.assertEqual(
            receipt_output.read_bytes(), operations._canonical_json_bytes(receipt)
        )
        serialized = receipt_output.read_text().lower()
        for forbidden in (
            "authorization",
            "cookie",
            "oauth",
            "token",
            "secret",
            "password",
            "hunter2",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_live_verification_rejects_stale_values_and_page_metadata(self):
        self._use_checked_final_issue()
        _output, index_path, _scope = self._capture()
        document = json.loads(index_path.read_text())
        post_bundle_expectation = self._post_bundle_expectation(
            document, self.responses
        )
        attacks = []
        stale_responses = copy.deepcopy(self.responses)
        stale_responses["ruleset"]["name"] = "stale ruleset"
        attacks.append(
            (
                "value",
                stale_responses,
                self._coherent_document(document, stale_responses)["response_captures"],
            )
        )
        stale_captures = copy.deepcopy(document["response_captures"])
        stale_captures["technical_release"]["pages"][0]["headers"][
            "etag"
        ] = '"stale-etag"'
        attacks.append(("metadata", copy.deepcopy(self.responses), stale_captures))
        for label, responses, captures in attacks:
            receipt_output = self.private_root / f"stale-{label}.json"
            with (
                mock.patch.object(
                    operations, "candidate_evidence", return_value=self.candidate
                ),
                mock.patch.object(
                    operations,
                    "_fresh_github_capture",
                    return_value=(responses, captures),
                ),
                mock.patch.object(
                    operations, "_validate_downloaded_publication"
                ) as publication_check,
                self.subTest(label=label),
                self.assertRaisesRegex(
                    operations.EvidenceError, "stale or substituted"
                ),
            ):
                source_bundle_root = self.root / f"stale-{label}-source-b1"
                reopened_bundle_root = self.root / f"stale-{label}-reopened-b1"
                source_bundle_root.mkdir()
                reopened_bundle_root.mkdir()
                operations._verify_operations_live_with_baseline(
                    index_path=index_path,
                    manifest=operations.DEFAULT_MANIFEST,
                    publication_policy=self.root / "policy.json",
                    publication_policy_sha256="e" * 64,
                    publication_assets={
                        role: self.root / name
                        for role, name in operations.TECHNICAL_RELEASE_ASSETS.items()
                    },
                    receipt_output=receipt_output,
                    post_bundle_expectation=post_bundle_expectation,
                    source_bundle_root=source_bundle_root,
                    reopened_bundle_root=reopened_bundle_root,
                    baseline_lease=_SyntheticBaselineLease(self._baseline_validation()),
                )
            publication_check.assert_not_called()
            self.assertFalse(receipt_output.exists())

    def test_strict_publication_validator_rejects_recomputed_forged_witness(self):
        from benchmarks import issue123_publication as publication
        from tests.test_issue123_publication import Issue123PublicationTest

        fixture = Issue123PublicationTest(
            "test_offline_release_capture_and_receipt_are_independently_reopenable"
        )
        fixture.setUp()
        release_anchor = fixture._release_capture()
        receipt_raw = publication.finalize_publication(
            fixture.assets,
            release_anchor,
            expected_policy=fixture.policy,
            expected_release_identity=fixture.release_identity,
            expected_bindings=fixture.bindings,
            expected_assets=fixture.ledger,
        )
        receipt = json.loads(receipt_raw)
        policy_path = self.root / "strict-publication-policy.json"
        policy_raw = publication.canonical_json_bytes(fixture.policy)
        policy_path.write_bytes(policy_raw)
        policy_sha256 = hashlib.sha256(policy_raw).hexdigest()
        asset_paths = {}
        for role, name in publication.ASSET_ORDER:
            path = self.root / name
            path.write_bytes(fixture.assets[name])
            asset_paths[role] = path

        def envelope(value):
            raw = publication.canonical_json_bytes(value)
            return {
                "media_type": operations.MEDIA_TYPE_JSON,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "document": value,
            }

        asset_ledger = [
            {"role": role, **fixture.ledger[role]}
            for role, _name in publication.ASSET_ORDER
        ]
        execution_claims = [
            {
                key: claim[key]
                for key in (
                    "claim",
                    "scope",
                    "event_count",
                    "semantic_inventory_sha256",
                    "normalized_trace_sha256",
                )
            }
            for claim in receipt["execution_witness"]["claims"]
        ]
        operations_result = {
            "technical_release": {"assets": copy.deepcopy(fixture.ledger)},
            "publication": {
                "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
                "receipt_size_bytes": len(receipt_raw),
                "bindings": copy.deepcopy(fixture.bindings),
                "release_id": release_anchor["release_id"],
                "release_tag": release_anchor["tag_name"],
                "asset_ids": {
                    record["role"]: record["asset_id"]
                    for record in release_anchor["assets"]
                },
                "asset_ledger": asset_ledger,
                "release_identity_anchor": release_anchor,
                "execution_claims": execution_claims,
            },
        }
        validated = operations._validate_downloaded_publication(
            envelope(receipt),
            policy_path,
            policy_sha256,
            asset_paths,
            operations_result,
        )
        self.assertEqual(
            validated["strict_four_byte_validator"], "same-process-invoked"
        )
        self.assertEqual(validated["release_identity_anchor"], release_anchor)

        forged = copy.deepcopy(receipt)
        for index, claim in enumerate(forged["execution_witness"]["claims"], start=1):
            claim["trace_name"] = f"invented-trace-{index}"
            claim["event_count"] = 900 + index
            claim["semantic_inventory_sha256"] = "a" * 64
            claim["normalized_trace_sha256"] = "b" * 64
        witness_raw = publication.canonical_json_bytes(forged["execution_witness"])
        forged["execution_witness_member"]["size_bytes"] = len(witness_raw)
        forged["execution_witness_member"]["sha256"] = hashlib.sha256(
            witness_raw
        ).hexdigest()
        forged["hashes"]["execution_witness_member_sha256"] = forged[
            "execution_witness_member"
        ]["sha256"]
        forged_envelope = envelope(forged)
        forged_result = copy.deepcopy(operations_result)
        forged_result["publication"]["receipt_sha256"] = forged_envelope["sha256"]
        forged_result["publication"]["receipt_size_bytes"] = forged_envelope[
            "size_bytes"
        ]
        with (
            mock.patch.object(
                publication,
                "validate_publication_receipt",
                wraps=publication.validate_publication_receipt,
            ) as strict_validator,
            self.assertRaises(operations.EvidenceError),
        ):
            operations._validate_downloaded_publication(
                forged_envelope,
                policy_path,
                policy_sha256,
                asset_paths,
                forged_result,
            )
        strict_validator.assert_called_once()
        strict_assets = strict_validator.call_args.args[1]
        self.assertEqual(set(strict_assets), set(fixture.assets))
        self.assertTrue(
            all(strict_assets[name] == fixture.assets[name] for name in fixture.assets)
        )
        self.assertEqual(
            strict_validator.call_args.kwargs,
            {
                "expected_policy": fixture.policy,
                "expected_release_identity": fixture.release_identity,
                "expected_bindings": fixture.bindings,
                "expected_assets": fixture.ledger,
            },
        )

        forged_policy = copy.deepcopy(forged)
        forged_policy["hashes"]["trusted_policy_sha256"] = "f" * 64
        forged_policy_envelope = envelope(forged_policy)
        forged_policy_result = copy.deepcopy(operations_result)
        forged_policy_result["publication"]["receipt_sha256"] = forged_policy_envelope[
            "sha256"
        ]
        forged_policy_result["publication"]["receipt_size_bytes"] = (
            forged_policy_envelope["size_bytes"]
        )
        with self.assertRaisesRegex(operations.EvidenceError, "policy digest"):
            operations._validate_downloaded_publication(
                forged_policy_envelope,
                policy_path,
                policy_sha256,
                asset_paths,
                forged_policy_result,
            )

    def test_github_api_pagination_normalizes_concatenated_json_pages(self):
        next_url = (
            "https://api.github.com/repos/ruddyscent/gmes/issues/123/comments"
            "?page=2&per_page=100"
        )
        completed = mock.Mock(
            stdout=(
                "HTTP/2.0 200 OK\r\n"
                "content-type: application/json; charset=utf-8\r\n"
                'etag: "page-one"\r\n'
                "x-github-api-version-selected: 2022-11-28\r\n"
                "x-github-media-type: github.v3; format=json\r\n"
                "authorization: Bearer must-not-survive\r\n"
                "x-oauth-scopes: repo\r\n"
                f'link: <{next_url}>; rel="next"\r\n\r\n'
                '[{"id":1}]\n'
                "HTTP/2.0 200 OK\r\n"
                "content-type: application/json; charset=utf-8\r\n"
                'etag: "page-two"\r\n'
                "x-github-api-version-selected: 2022-11-28\r\n"
                "x-github-media-type: github.v3; format=json\r\n"
                "set-cookie: secret=no\r\n\r\n"
                '[{"id":2}]\n'
            ).encode()
        )
        with mock.patch.object(
            operations.subprocess, "run", return_value=completed
        ) as run:
            raw, capture = operations._github_api_capture(
                "repos/ruddyscent/gmes/issues/123/comments",
                parameters={"per_page": "100"},
                paginated=True,
            )
        self.assertEqual(json.loads(raw), [[{"id": 1}], [{"id": 2}]])
        self.assertTrue(capture["pages"][0]["has_next"])
        self.assertFalse(capture["pages"][1]["has_next"])
        self.assertIsNone(capture["pages"][1]["next"])
        self.assertEqual(
            set(capture["pages"][0]["headers"]),
            set(operations.SAFE_RESPONSE_HEADERS),
        )
        self.assertNotIn("authorization", json.dumps(capture).lower())
        self.assertNotIn("oauth", json.dumps(capture).lower())
        self.assertNotIn("cookie", json.dumps(capture).lower())
        command = run.call_args.args[0]
        self.assertEqual(
            command[:5],
            [
                "gh",
                "api",
                "--hostname",
                "github.com",
                "repos/ruddyscent/gmes/issues/123/comments",
            ],
        )
        self.assertIn("--include", command)
        self.assertIn("--paginate", command)
        self.assertIn("--jq", command)
        self.assertIn("Accept: application/vnd.github+json", command)
        self.assertIn("X-GitHub-Api-Version: 2022-11-28", command)
        self.assertEqual(command[command.index("--jq") + 1], ".")
        self.assertNotIn("--slurp", command)

    def test_github_api_direct_success_never_spawns_auth_status(self):
        completed = mock.Mock(
            stdout=(
                "HTTP/2.0 200 OK\r\n"
                "content-type: application/json; charset=utf-8\r\n"
                'etag: "single-page"\r\n'
                "x-github-api-version-selected: 2022-11-28\r\n"
                "x-github-media-type: github.v3; format=json\r\n\r\n"
                '{"id":1}\n'
            ).encode()
        )
        with mock.patch.object(
            operations.subprocess,
            "run",
            return_value=completed,
        ) as run:
            raw, _capture = operations._github_api_capture(
                "repos/ruddyscent/gmes/issues/123"
            )
        self.assertEqual(json.loads(raw), {"id": 1})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:2], ["gh", "api"])

    def test_github_api_failure_preserves_original_cause_without_status(self):
        api_error = operations.subprocess.CalledProcessError(
            1,
            ["gh", "api"],
            stderr=b"gh: Bad credentials (HTTP 401)",
        )
        endpoint = "repos/ruddyscent/gmes/issues/123"
        with (
            mock.patch.object(
                operations.subprocess,
                "run",
                side_effect=api_error,
            ) as run,
            self.assertRaisesRegex(
                operations.EvidenceError, "GitHub API request failed"
            ) as raised,
        ):
            operations._github_api_capture(endpoint)
        self.assertIs(raised.exception.__cause__, api_error)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args_list[0].args[0][:2], ["gh", "api"])
        self.assertEqual(
            run.call_args_list[0].kwargs,
            {"check": True, "capture_output": True},
        )

    def test_capture_initial_api_failure_leaves_no_partial_output(self):
        output = self.root / "not-created" / "initial-api-failure-output"
        api_error = operations.subprocess.CalledProcessError(
            1,
            ["gh", "api"],
            stderr=b"gh: Bad credentials (HTTP 401)",
        )
        with (
            mock.patch.object(
                operations, "candidate_evidence", return_value=self.candidate
            ),
            mock.patch.object(
                operations.subprocess,
                "run",
                side_effect=api_error,
            ) as run,
            self.assertRaisesRegex(
                operations.EvidenceError, "GitHub API request failed"
            ),
        ):
            operations.capture_operations(
                repository=operations.REPOSITORY,
                pull_request_number=self.number,
                ci_run_id=10,
                codeql_run_id=20,
                technical_release_tag=self.release_tag,
                publication_receipt=self.publication_receipt_path,
                output_directory=output,
            )
        self.assertEqual(run.call_count, 1)
        self.assertFalse(output.exists())
        self.assertFalse(output.parent.exists())

    def test_capture_late_api_failure_cleans_staging_and_allows_retry(self):
        output_name = "late-api-failure-output"
        output = self.root / output_name

        def fail_ci_jobs(endpoint):
            if endpoint.endswith(f"/actions/runs/10/jobs"):
                raise operations.EvidenceError("synthetic ci_jobs failure")

        with self.assertRaisesRegex(operations.EvidenceError, "synthetic ci_jobs"):
            self._capture(output_name, before_capture=fail_ci_jobs)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(f".{output_name}.*")), [])
        retry_output, index_path, scope_path = self._capture(output_name)
        self.assertEqual(retry_output, output)
        self.assertTrue(index_path.is_file())
        self.assertTrue(scope_path.is_file())

    def test_capture_rejects_destination_appearing_during_assembly(self):
        output_name = "appeared-during-assembly"
        output = self.root / output_name
        sentinel = output / "sentinel"

        def create_foreign_destination(endpoint):
            if endpoint == "graphql":
                output.mkdir()
                sentinel.write_text("foreign")

        with self.assertRaisesRegex(
            operations.EvidenceError, "appeared during assembly"
        ):
            self._capture(output_name, before_capture=create_foreign_destination)
        self.assertEqual(sentinel.read_text(), "foreign")
        self.assertEqual(list(self.root.glob(f".{output_name}.*")), [])

    def test_capture_rejects_dangling_destination_symlink_during_assembly(self):
        output_name = "dangling-output"
        output = self.root / output_name
        target = self.root / "absent-target"

        def create_dangling_destination(endpoint):
            if endpoint == "graphql":
                output.symlink_to(target)

        with self.assertRaisesRegex(
            operations.EvidenceError, "appeared during assembly"
        ):
            self._capture(output_name, before_capture=create_dangling_destination)
        self.assertTrue(output.is_symlink())
        self.assertEqual(output.readlink(), target)
        self.assertEqual(list(self.root.glob(f".{output_name}.*")), [])

    def test_github_api_graphql_command_pins_query_and_typed_pr(self):
        response = {
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "pageInfo": {
                                "hasNextPage": False,
                                "endCursor": "final-cursor",
                            }
                        }
                    }
                }
            }
        }
        completed = mock.Mock(
            stdout=(
                "HTTP/2.0 200 OK\n"
                "content-type: application/json; charset=utf-8\n"
                'etag: "graphql-page"\n'
                "x-github-media-type: github.v4; format=json\n\n"
            ).encode()
            + operations._canonical_json_bytes(response)
        )
        with mock.patch.object(
            operations.subprocess, "run", return_value=completed
        ) as run:
            raw, capture = operations._github_api_capture(
                "graphql",
                paginated=True,
                graphql_variables={
                    "owner": "ruddyscent",
                    "name": "gmes",
                    "number": operations.PULL_REQUEST_NUMBER,
                },
            )
        self.assertEqual(json.loads(raw), [response])
        self.assertIsNone(
            capture["pages"][0]["headers"]["x-github-api-version-selected"]
        )
        self.assertEqual(capture["pages"][0]["headers"]["link"], {})
        command = run.call_args.args[0]
        self.assertEqual(
            command[:5],
            ["gh", "api", "--hostname", "github.com", "graphql"],
        )
        self.assertIn("--include", command)
        self.assertIn("--paginate", command)
        self.assertIn("--jq", command)
        self.assertIn("Accept: application/vnd.github+json", command)
        self.assertIn("X-GitHub-Api-Version: 2022-11-28", command)
        query_argument = next(item for item in command if item.startswith("query="))
        query = query_argument.removeprefix("query=")
        self.assertIn("closingIssuesReferences(first:2)", query)
        self.assertIn("reviews(first:1,states:", query)
        self.assertIn("reviewThreads(first:100,after:$endCursor)", query)
        number_argument = f"number={operations.PULL_REQUEST_NUMBER}"
        self.assertEqual(command[command.index(number_argument) - 1], "-F")
        self.assertNotIn("--slurp", command)

    def test_final_checklist_requires_exact_state_order_and_section(self):
        checked = {
            "body": "\n".join(
                (
                    operations.FINAL_CHECKLIST_SECTION,
                    *operations.FINAL_CHECKLIST_CHECKED,
                )
            ),
            "updated_at": "2026-09-03T01:31:00Z",
        }
        observation = operations._validate_post_bundle_checklist(checked, "checked")
        self.assertEqual(observation["state"], "checked")
        unchecked = {
            **checked,
            "body": checked["body"].replace("[x]", "[ ]"),
            "updated_at": "2026-09-03T01:30:00Z",
        }
        self.assertEqual(
            operations.checklist_transition_sha256(unchecked, "unchecked"),
            operations.checklist_transition_sha256(checked, "checked"),
        )
        for label, changed_body in (
            ("unrelated-text", checked["body"] + "synthetic trailing line\n"),
            (
                "heading",
                checked["body"].replace(
                    operations.FINAL_CHECKLIST_SECTION,
                    "## Implementation work changed",
                ),
            ),
            ("line-endings", checked["body"].replace("\n", "\r\n")),
            (
                "whitespace",
                checked["body"].replace(
                    "## Implementation work\n",
                    "## Implementation work \n",
                ),
            ),
        ):
            changed = {**checked, "body": changed_body}
            with self.subTest(transition=label):
                try:
                    changed_transition = operations.checklist_transition_sha256(
                        changed, "checked"
                    )
                except operations.EvidenceError:
                    continue
                self.assertNotEqual(
                    changed_transition,
                    operations.checklist_transition_sha256(unchecked, "unchecked"),
                )
        attacks = {
            "missing": checked["body"].replace(
                operations.FINAL_CHECKLIST_CHECKED[1], ""
            ),
            "substituted": checked["body"].replace(
                "post-bundle checklist", "different checklist"
            ),
            "wrong-section": checked["body"].replace(
                operations.FINAL_CHECKLIST_SECTION, "## Synthetic other section"
            ),
            "unchecked": checked["body"].replace("[x]", "[ ]"),
            "reordered": "\n".join(
                (
                    operations.FINAL_CHECKLIST_SECTION,
                    *reversed(operations.FINAL_CHECKLIST_CHECKED),
                )
            ),
            "duplicated": checked["body"]
            + "\n"
            + operations.FINAL_CHECKLIST_CHECKED[0],
        }
        for label, body in attacks.items():
            with (
                self.subTest(label=label),
                self.assertRaises(operations.EvidenceError),
            ):
                operations._validate_post_bundle_checklist(
                    {**checked, "body": body}, "checked"
                )

    def test_checklist_transition_commits_complete_response_except_contract_fields(
        self,
    ):
        unchecked = copy.deepcopy(self.responses["issue_123"])
        checked = copy.deepcopy(unchecked)
        checked["body"] = checked["body"].replace("- [ ]", "- [x]")
        checked["updated_at"] = "2026-09-03T01:31:00Z"
        unchecked_digest = operations.checklist_transition_sha256(
            unchecked,
            "unchecked",
        )
        checked_digest = operations.checklist_transition_sha256(
            checked,
            "checked",
        )
        self.assertEqual(unchecked_digest, checked_digest)
        self.assertNotEqual(
            hashlib.sha256(operations._canonical_json_bytes(unchecked)).hexdigest(),
            hashlib.sha256(operations._canonical_json_bytes(checked)).hexdigest(),
        )
        attacks = {
            "number": lambda value: value.update(number=124),
            "state": lambda value: value.update(state="closed"),
            "body": lambda value: value.update(body="unrelated\n" + value["body"]),
            "nested-timestamp": lambda value: value.update(
                fixture={"updated_at": "2026-09-03T01:32:00Z"}
            ),
            "array-order": lambda value: value.update(labels=["b", "a"]),
            "extra": lambda value: value.update(extra_contract_value=True),
        }
        for label, mutate in attacks.items():
            changed = copy.deepcopy(checked)
            mutate(changed)
            with self.subTest(label=label):
                self.assertNotEqual(
                    operations.checklist_transition_sha256(changed, "checked"),
                    unchecked_digest,
                )

    def test_baseline_live_authority_rejects_missing_or_mutated_exact_bytes(self):
        authority, download_by_id = operations._synthetic_baseline_authority_fixture()
        api_root = f"https://api.github.com/repos/{operations.REPOSITORY}"
        web_root = f"https://github.com/{operations.REPOSITORY}"
        release_id = 77
        release_assets = [
            {
                "id": 701 + ordinal,
                "name": asset.name,
                "url": f"{api_root}/releases/assets/{701 + ordinal}",
                "browser_download_url": asset.publication_url,
                "state": "uploaded",
                "size": asset.size_bytes,
            }
            for ordinal, asset in enumerate(authority.assets)
        ]
        release = {
            "id": release_id,
            "tag_name": operations.BASELINE_RELEASE_TAG,
            "url": f"{api_root}/releases/{release_id}",
            "html_url": (f"{web_root}/releases/tag/{operations.BASELINE_RELEASE_TAG}"),
            "draft": False,
            "prerelease": False,
            "assets": release_assets,
        }
        tag = {
            "ref": f"refs/tags/{operations.BASELINE_RELEASE_TAG}",
            "object": {
                "type": "commit",
                "sha": authority.root_commit,
                "url": f"{api_root}/git/commits/{authority.root_commit}",
            },
        }
        capture_values = {
            f"repos/{operations.REPOSITORY}/releases/tags/"
            f"{operations.BASELINE_RELEASE_TAG}": (
                operations._canonical_json_bytes(release),
                {"fixture": "release"},
            ),
            f"repos/{operations.REPOSITORY}/git/ref/tags/"
            f"{operations.BASELINE_RELEASE_TAG}": (
                operations._canonical_json_bytes(tag),
                {"fixture": "tag"},
            ),
        }

        def capture(endpoint):
            return capture_values[endpoint]

        with operations._open_baseline_authority_core(
            code_authority=authority,
            manifest_authority=authority,
            b1_authority=authority,
            api_capture=capture,
            asset_download=lambda identifier: download_by_id[identifier],
            observed_at="2026-09-03T02:00:00Z",
        ) as lease:
            result = lease.validation
            lease.require_unchanged()
        self.assertEqual(len(result["asset_ledger"]), 2)
        self.assertEqual(
            [item["name"] for item in result["asset_ledger"]],
            [asset.name for asset in authority.assets],
        )
        self.assertNotIn("salt", json.dumps(result).lower())
        self.assertNotIn("hostname", json.dumps(result).lower())

        reordered_release = copy.deepcopy(release)
        reordered_release["assets"].reverse()
        reordered_capture = dict(capture_values)
        reordered_capture[next(iter(reordered_capture))] = (
            operations._canonical_json_bytes(reordered_release),
            {"fixture": "release"},
        )
        reordered_download = mock.Mock(
            side_effect=lambda identifier: download_by_id[identifier]
        )
        with self.assertRaisesRegex(
            operations.EvidenceError,
            "closure or order",
        ):
            operations._capture_baseline_authority(
                code_authority=authority,
                manifest_authority=authority,
                b1_authority=authority,
                api_capture=lambda endpoint: reordered_capture[endpoint],
                asset_download=reordered_download,
            )
        reordered_download.assert_not_called()

        wrong_b1 = operations.BaselineAuthoritySet(
            root_commit="e" * 40,
            assets=authority.assets,
        )
        authority_capture = mock.Mock()
        with self.assertRaisesRegex(
            operations.EvidenceError,
            "code, manifest, and authenticated B1",
        ):
            operations._capture_baseline_authority(
                code_authority=authority,
                manifest_authority=authority,
                b1_authority=wrong_b1,
                api_capture=authority_capture,
                asset_download=mock.Mock(),
            )
        authority_capture.assert_not_called()

        for attack in (
            "append",
            "same-size",
            "inode-replacement",
            "symlink",
            "mode",
            "extra-file",
        ):
            with self.subTest(retained_attack=attack):
                lease = operations._capture_baseline_authority(
                    code_authority=authority,
                    manifest_authority=authority,
                    b1_authority=authority,
                    api_capture=capture,
                    asset_download=lambda identifier: download_by_id[identifier],
                    observed_at="2026-09-03T02:00:00Z",
                )
                first = lease._root / authority.assets[0].name
                original = first.read_bytes()
                if attack == "append":
                    first.write_bytes(original + b"x")
                elif attack == "same-size":
                    changed = bytearray(original)
                    changed[0] ^= 1
                    first.write_bytes(changed)
                elif attack == "inode-replacement":
                    replacement = first.with_name("replacement")
                    replacement.write_bytes(original)
                    replacement.replace(first)
                elif attack == "symlink":
                    first.unlink()
                    first.symlink_to(authority.assets[1].name)
                elif attack == "mode":
                    first.chmod(0o640)
                else:
                    (lease._root / "unexpected-third-entry").write_bytes(b"x")
                with self.assertRaises(operations.EvidenceError):
                    lease.require_unchanged()
                lease.close()

        missing_release = copy.deepcopy(release)
        missing_release["assets"] = missing_release["assets"][:-1]
        download = mock.Mock(side_effect=lambda identifier: download_by_id[identifier])
        missing_capture = dict(capture_values)
        missing_capture[next(iter(missing_capture))] = (
            operations._canonical_json_bytes(missing_release),
            {"fixture": "release"},
        )
        with self.assertRaisesRegex(
            operations.EvidenceError,
            "closure or order",
        ):
            operations._capture_baseline_authority(
                code_authority=authority,
                manifest_authority=authority,
                b1_authority=authority,
                api_capture=lambda endpoint: missing_capture[endpoint],
                asset_download=download,
            )
        download.assert_not_called()

        mutated = dict(download_by_id)
        mutated[release_assets[1]["id"]] += b"mutated"
        with self.assertRaisesRegex(operations.EvidenceError, "bytes differ"):
            operations._capture_baseline_authority(
                code_authority=authority,
                manifest_authority=authority,
                b1_authority=authority,
                api_capture=capture,
                asset_download=lambda identifier: mutated[identifier],
            )
        with self.assertRaisesRegex(operations.EvidenceError, "unsupported"):
            operations._capture_production_baseline_authority(
                operations.DEFAULT_MANIFEST,
                authority="immutable-mirror",
                b1_authority=operations.PRODUCTION_BASELINE_AUTHORITY_SET,
            )

    def test_live_receipt_privacy_scan_is_recursive(self):
        operations._assert_provenance_receipt_safe(
            {
                "safe_commitment_sha256": "a" * 64,
                "nested": [{"role": "synthetic-role", "size_bytes": 1}],
            }
        )
        for unsafe in (
            {"nested": [{"private_path": "/private/fixture"}]},
            {"nested": [{"value": "github_pat_" + "syntheticinvalid" * 2}]},
        ):
            with self.assertRaises(operations.EvidenceError):
                operations._assert_provenance_receipt_safe(unsafe)

    def test_verify_live_cli_uses_authenticated_b1_lease_not_caller_json(self):
        from benchmarks import issue123_privacy as privacy
        from tests.test_issue123_bundle import Issue123BundleTest

        bundle = Issue123BundleTest()
        bundle.setUp()
        self.addCleanup(bundle.doCleanups)
        bundle.candidate = copy.deepcopy(self.candidate)
        runtime_raw_by_role = {
            role: completion._compact_canonical_json_bytes(
                {"ordinal": ordinal, "role": role}
            )
            for ordinal, role in enumerate(completion.RUNTIME_RECEIPT_ROLES)
        }
        unchecked_issue = copy.deepcopy(self.responses["issue_123"])
        unchecked_issue["updated_at"] = "2026-09-02T22:45:00Z"
        (
            b0_source,
            b0_issue,
            runtime_records,
            scope_artifacts,
        ) = bundle._write_authority_bundle(
            "operations-cli-b0-source",
            checked=False,
            updated_at=unchecked_issue["updated_at"],
            runtime_raw_by_role=runtime_raw_by_role,
            issue_document=unchecked_issue,
        )
        b0_reopened_root = bundle.directory / "operations-cli-b0-reopened"
        shutil.copytree(b0_source.parent, b0_reopened_root)
        b0_reopened = b0_reopened_root / "completion-index.json"
        authority = bundle.directory / "operations-cli-authority"
        authority.mkdir(mode=0o700)
        openings = privacy.PrivateOpenings(bytes(range(32)))
        openings._populated = True
        context = bundle._authority_context(runtime_records, scope_artifacts)
        openings_path = authority / "openings.json"
        privacy.write_private_authority_file(
            openings_path,
            privacy.serialize_private_openings(openings, context),
            label="synthetic protected openings",
        )
        b0_receipt_path = authority / "b0-reopen.json"
        b0_raw = completion.record_bundle_reopen(
            source_index=b0_source,
            reopened_index=b0_reopened,
            stage="pre-acknowledgment",
            protected_openings=openings_path,
            pre_ack_response=(
                b0_reopened_root / b0_issue.relative_to(b0_source.parent)
            ),
            output=b0_receipt_path,
        )
        b0_receipt = json.loads(b0_raw)

        self._use_checked_final_issue()
        self.responses["issue_123"]["updated_at"] = b0_receipt["observed_at"]
        _operations_root, index_path, _scope = self._capture("operations-cli-capture")
        document = json.loads(index_path.read_text())
        (
            b1_source,
            _b1_issue,
            b1_runtime_records,
            b1_scope_artifacts,
        ) = bundle._write_authority_bundle(
            "operations-cli-b1-source",
            checked=True,
            updated_at=self.responses["issue_123"]["updated_at"],
            runtime_raw_by_role=runtime_raw_by_role,
            issue_document=self.responses["issue_123"],
        )
        self.assertEqual(b1_runtime_records, runtime_records)
        self.assertEqual(b1_scope_artifacts, scope_artifacts)
        b1_reopened_root = bundle.directory / "operations-cli-b1-reopened"
        shutil.copytree(b1_source.parent, b1_reopened_root)
        b1_reopened = b1_reopened_root / "completion-index.json"
        b1_receipt_path = authority / "b1-reopen.json"
        completion.record_bundle_reopen(
            source_index=b1_source,
            reopened_index=b1_reopened,
            stage="final",
            protected_openings=openings_path,
            pre_ack_receipt=b0_receipt_path,
            output=b1_receipt_path,
        )
        runtime_paths = []
        for ordinal, role in enumerate(completion.RUNTIME_RECEIPT_ROLES):
            path = authority / f"external-{ordinal}-{role}.json"
            path.write_bytes(runtime_raw_by_role[role])
            path.chmod(0o600)
            runtime_paths.append(path)

        policy = self.root / "trusted-policy.json"
        policy.write_bytes(operations._canonical_json_bytes({"fixture": True}))
        asset_paths = {}
        for role, name in operations.TECHNICAL_RELEASE_ASSETS.items():
            path = self.root / name
            path.write_bytes(f"synthetic-{role}".encode())
            asset_paths[role] = path
        receipt_path = self.private_root / "operations-cli-receipt.json"
        publication_result = {
            "strict_four_byte_validator": "same-process-invoked",
            "receipt_sha256": document["publication_receipt"]["sha256"],
            "trusted_policy_sha256": "e" * 64,
            "asset_ledger": document["publication_receipt"]["document"]["asset_ledger"],
            "release_identity_anchor": document["publication_receipt"]["document"][
                "release_capture"
            ],
            "bindings": document["publication_receipt"]["document"]["bindings"],
            "execution_claims": [],
            "event_profiler": {
                "contract_id": "synthetic-fixture",
                "record_count": 1,
                "inventory_sha256": "f" * 64,
                "asset_sha256": "4" * 64,
            },
        }
        arguments = [
            "verify-live",
            "--index",
            str(index_path),
            "--manifest",
            str(operations.DEFAULT_MANIFEST),
            "--publication-policy",
            str(policy),
            "--publication-policy-sha256",
            "e" * 64,
            "--technical-evidence-asset",
            str(asset_paths["technical_evidence"]),
            "--technical-summary-asset",
            str(asset_paths["technical_summary"]),
            "--raw-timing-asset",
            str(asset_paths["raw_timing"]),
            "--event-profiler-asset",
            str(asset_paths["event_profiler"]),
            "--source-index",
            str(b1_source),
            "--reopened-index",
            str(b1_reopened),
            "--private-openings",
            str(openings_path),
            "--pre-ack-bundle-reopen-receipt",
            str(b0_receipt_path),
            "--final-bundle-reopen-receipt",
            str(b1_receipt_path),
            "--runtime-receipts",
            *(str(path) for path in runtime_paths),
            "--baseline-authority",
            "live-release",
            "--receipt-output",
            str(receipt_path),
        ]
        with (
            mock.patch.object(
                operations, "candidate_evidence", return_value=self.candidate
            ),
            mock.patch.object(
                operations,
                "_fresh_github_capture",
                return_value=(
                    copy.deepcopy(self.responses),
                    copy.deepcopy(document["response_captures"]),
                ),
            ),
            mock.patch.object(
                operations,
                "_validate_downloaded_publication",
                return_value=publication_result,
            ),
            mock.patch.object(
                completion,
                "_validate_final_b1_baseline_descriptors",
                return_value=operations.PRODUCTION_BASELINE_AUTHORITY_SET,
            ),
            mock.patch.object(
                operations,
                "_capture_production_baseline_authority",
                return_value=_SyntheticBaselineLease(self._baseline_validation()),
            ),
        ):
            status = operations.main(arguments)
        self.assertEqual(status, 0)
        receipt = json.loads(receipt_path.read_bytes())
        self.assertEqual(
            receipt["schema_version"],
            operations.LIVE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
        )
        self.assertEqual(
            receipt["baseline_validation"]["release_identity"]["tag_name"],
            operations.BASELINE_RELEASE_TAG,
        )
        self.assertTrue(receipt["same_process_live_accepted"])

        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            mock.patch("sys.stderr", new=stderr),
            mock.patch("sys.stdout", new=stdout),
        ):
            old_status = operations._cli(
                [*arguments, "--post-bundle-expectation", "forbidden.json"]
            )
        self.assertEqual(old_status, 2)
        self.assertEqual(stderr.getvalue(), "issue123-operations-usage-failed\n")
        self.assertEqual(stdout.getvalue(), "")

    def test_operations_cli_failure_tokens_never_render_private_text(self):
        marker = (
            "/tmp/synthetic-private.invalid/identity "
            + "salt="
            + "ab" * 32
            + " hmac="
            + "cd" * 32
            + " raw-body=fixture-private-value"
        )
        for command, token in (
            ("capture", "issue123-operations-capture-failed\n"),
            ("verify-live", "issue123-operations-verify-live-failed\n"),
        ):
            for boundary in (operations.main, operations._cli):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    self.subTest(command=command, boundary=boundary.__name__),
                    mock.patch.object(
                        operations,
                        "_main",
                        side_effect=operations.EvidenceError(marker),
                    ),
                    mock.patch("sys.stdout", new=stdout),
                    mock.patch("sys.stderr", new=stderr),
                ):
                    status = boundary([command])
                self.assertEqual(status, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(stderr.getvalue(), token)
                rendered = stdout.getvalue() + stderr.getvalue()
                self.assertNotIn("Traceback", rendered)
                self.assertNotIn(marker, rendered)


if __name__ == "__main__":
    unittest.main()
