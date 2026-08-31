from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import issue123_completion as completion
from benchmarks import issue123_operations as operations


class Issue123OperationsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.candidate = {
            "candidate_git_commit": "a" * 40,
            "candidate_git_status": "",
            "manifest_sha256": "d" * 64,
        }
        self.base_sha = "b" * 40
        self.merge_sha = "c" * 40
        self.number = 166
        self.responses = self._responses()

    @staticmethod
    def _job(job_id, run_id, name):
        return {
            "id": job_id,
            "run_id": run_id,
            "run_attempt": 1,
            "name": name,
            "status": "completed",
            "conclusion": "success",
            "started_at": "2026-08-31T00:00:00Z",
            "completed_at": "2026-08-31T00:10:00Z",
        }

    def _responses(self):
        candidate = self.candidate["candidate_git_commit"]
        merge_ref = f"refs/pull/{self.number}/merge"
        analysis = lambda category, identifier: {
            "id": identifier,
            "language": "python" if category.endswith("python") else "cpp",
            "analysis_key": ".github/workflows/codeql.yml:analyze",
            "category": category,
            "commit_sha": self.merge_sha,
            "ref": merge_ref,
            "created_at": "2026-08-31T00:05:00Z",
            "results_count": 7,
            "rules_count": 100,
            "error": "",
            "warning": "",
            "tool": {"name": "CodeQL", "version": "2.23.0"},
        }
        return {
            "issue_115": {
                "number": 115,
                "state": "closed",
                "closed_at": "2026-08-25T09:01:33Z",
                "body": "## Implementation work\n- [x] runtime\n- [x] profiler\n",
            },
            "issue_115_comments": [
                [
                    {
                        "id": 900,
                        "user": {"login": "ruddyscent"},
                        "author_association": "OWNER",
                        "body": (
                            "Issue #123 final runtime-observation handoff\n"
                            f"PR #{self.number} candidate {candidate}\n"
                            "torch.utils.benchmark output and profiler artifacts linked."
                        ),
                    }
                ]
            ],
            "pull_request": {
                "number": self.number,
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "merge_commit_sha": self.merge_sha,
                "base": {"ref": "master", "sha": self.base_sha},
                "head": {
                    "sha": candidate,
                    "repo": {"full_name": operations.REPOSITORY},
                },
            },
            "base_compare": {
                "status": "ahead",
                "ahead_by": 1,
                "behind_by": 0,
                "base_commit": {"sha": self.base_sha},
                "merge_base_commit": {"sha": self.base_sha},
                "commits": [{"sha": candidate}],
            },
            "ci_run": {
                "id": 10,
                "name": "CI",
                "event": "pull_request",
                "head_sha": candidate,
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
            },
            "ci_jobs": [
                {
                    "jobs": [
                        self._job(11, 10, operations.REQUIRED_STATUS_CONTEXTS[0]),
                        self._job(12, 10, operations.REQUIRED_STATUS_CONTEXTS[1]),
                    ]
                }
            ],
            "codeql_run": {
                "id": 20,
                "name": "CodeQL",
                "event": "pull_request",
                "head_sha": candidate,
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
            },
            "codeql_jobs": [
                {
                    "jobs": [
                        self._job(21, 20, operations.REQUIRED_CODEQL_JOBS[0]),
                        self._job(22, 20, operations.REQUIRED_CODEQL_JOBS[1]),
                    ]
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
            },
            "check_runs": [
                {
                    "check_runs": [
                        {
                            "name": name,
                            "head_sha": candidate,
                            "status": "completed",
                            "conclusion": "success",
                            "app": {"slug": "github-actions"},
                        }
                        for name in operations.REQUIRED_STATUS_CONTEXTS
                    ]
                }
            ],
            "reviews": [[]],
            "requested_reviewers": {"users": [], "teams": []},
            "review_threads": {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "totalCount": 1,
                                "pageInfo": {"hasNextPage": False, "endCursor": "x"},
                                "nodes": [{"id": "thread", "isResolved": True}],
                            }
                        }
                    }
                }
            },
        }

    def _capture(self):
        def raw(endpoint, **_kwargs):
            role_by_suffix = {
                "/issues/115": "issue_115",
                "/issues/115/comments": "issue_115_comments",
                f"/pulls/{self.number}": "pull_request",
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
            return json.dumps(self.responses[role], separators=(",", ":")).encode()

        output = self.root / "operations"
        with (
            mock.patch.object(
                operations, "candidate_evidence", return_value=self.candidate
            ),
            mock.patch.object(operations, "_github_api_raw", side_effect=raw),
        ):
            index, scope = operations.capture_operations(
                repository=operations.REPOSITORY,
                pull_request_number=self.number,
                ci_run_id=10,
                codeql_run_id=20,
                output_directory=output,
            )
        return output, index, scope

    def test_capture_and_completion_recompute_raw_api_evidence(self):
        output, _index, scope_path = self._capture()
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
        unresolved = copy.deepcopy(self.responses)
        unresolved["review_threads"]["data"]["repository"]["pullRequest"][
            "reviewThreads"
        ]["nodes"][0]["isResolved"] = False
        cases.append(("review", unresolved))
        for label, raw in cases:
            with self.subTest(label=label), self.assertRaises(operations.EvidenceError):
                operations.evaluate_operations(document, raw, self.candidate)


if __name__ == "__main__":
    unittest.main()
