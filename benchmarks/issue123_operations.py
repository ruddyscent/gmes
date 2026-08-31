#!/usr/bin/env python3
"""Capture raw GitHub operational evidence required to close issue #123."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.host_contract import candidate_evidence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "native_oracle_workloads.json"
INDEX_KIND = "issue-123-operations-evidence-index"
MEDIA_TYPE_JSON = "application/json"
REPOSITORY = "ruddyscent/gmes"
RULESET_ID = 21130311
REVIEW_THREADS_QUERY = """
query($owner:String!,$name:String!,$number:Int!){
  repository(owner:$owner,name:$name){
    pullRequest(number:$number){
      reviewThreads(first:100){
        totalCount
        pageInfo{hasNextPage endCursor}
        nodes{id isResolved}
      }
    }
  }
}
""".strip()
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceError(ValueError):
    """The operational evidence capture is incomplete or ambiguous."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _strict_json(raw: bytes, label: str) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise EvidenceError(f"{label} repeats JSON key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceError(f"{label} contains {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not strict UTF-8 JSON") from error


def _github_api_raw(
    endpoint: str,
    *,
    parameters: dict[str, str] | None = None,
    paginated: bool = False,
    graphql_variables: dict[str, str] | None = None,
) -> bytes:
    command = ["gh", "api", endpoint]
    if paginated:
        command.extend(("--paginate", "--slurp"))
    if parameters:
        command.extend(("-X", "GET"))
        for key, value in sorted(parameters.items()):
            command.extend(("-f", f"{key}={value}"))
    if graphql_variables is not None:
        command.extend(("-f", f"query={REVIEW_THREADS_QUERY}"))
        for key, value in sorted(graphql_variables.items()):
            flag = "-F" if key == "number" else "-f"
            command.extend((flag, f"{key}={value}"))
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError(f"GitHub API request failed: {endpoint}") from error
    _strict_json(completed.stdout, f"GitHub API response {endpoint}")
    return completed.stdout


def _descriptor(path: Path, base: Path, candidate: dict[str, str]) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": MEDIA_TYPE_JSON,
        "candidate_evidence": candidate,
    }


def capture_operations(
    *,
    repository: str,
    pull_request_number: int,
    ci_run_id: int,
    codeql_run_id: int,
    output_directory: Path,
    manifest: Path = DEFAULT_MANIFEST,
) -> tuple[Path, Path]:
    """Capture exact raw API bytes and emit a descriptor-only operations index."""

    _require(repository == REPOSITORY, "operations repository differs")
    _require(
        all(
            type(value) is int and value > 0
            for value in (
                pull_request_number,
                ci_run_id,
                codeql_run_id,
            )
        ),
        "PR and workflow run ids must be positive integers",
    )
    output_directory = output_directory.resolve()
    _require(not output_directory.exists(), "operations output already exists")
    output_directory.mkdir(parents=True)
    raw_directory = output_directory / "raw"
    raw_directory.mkdir()
    candidate = candidate_evidence(manifest.resolve(strict=True))

    records: dict[str, dict[str, Any]] = {}

    def capture(
        role: str,
        endpoint: str,
        *,
        parameters: dict[str, str] | None = None,
        paginated: bool = False,
        graphql_variables: dict[str, str] | None = None,
    ) -> Any:
        raw = _github_api_raw(
            endpoint,
            parameters=parameters,
            paginated=paginated,
            graphql_variables=graphql_variables,
        )
        path = raw_directory / f"{role}.json"
        path.write_bytes(raw)
        records[role] = {
            "request": {
                "endpoint": endpoint,
                "parameters": parameters or {},
                "paginated": paginated,
                "graphql": graphql_variables is not None,
                "variables": graphql_variables or {},
            },
            "artifact": _descriptor(path, output_directory, candidate),
        }
        return _strict_json(raw, role)

    capture("issue_115", f"repos/{repository}/issues/115")
    capture(
        "issue_115_comments",
        f"repos/{repository}/issues/115/comments",
        parameters={"per_page": "100"},
        paginated=True,
    )
    pull = capture("pull_request", f"repos/{repository}/pulls/{pull_request_number}")
    _require(isinstance(pull, dict), "pull request response differs")
    base = pull.get("base")
    head = pull.get("head")
    _require(isinstance(base, dict) and isinstance(head, dict), "PR refs differ")
    base_sha = base.get("sha")
    head_sha = head.get("sha")
    merge_sha = pull.get("merge_commit_sha")
    _require(
        all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
            for value in (base_sha, head_sha, merge_sha)
        ),
        "PR commits are incomplete",
    )
    _require(
        head_sha == candidate["candidate_git_commit"],
        "PR head does not match the clean candidate",
    )
    capture(
        "base_compare",
        f"repos/{repository}/compare/{base_sha}...{head_sha}",
    )
    capture("ci_run", f"repos/{repository}/actions/runs/{ci_run_id}")
    capture(
        "ci_jobs",
        f"repos/{repository}/actions/runs/{ci_run_id}/jobs",
        parameters={"per_page": "100"},
        paginated=True,
    )
    capture("codeql_run", f"repos/{repository}/actions/runs/{codeql_run_id}")
    capture(
        "codeql_jobs",
        f"repos/{repository}/actions/runs/{codeql_run_id}/jobs",
        parameters={"per_page": "100"},
        paginated=True,
    )
    merge_ref = f"refs/pull/{pull_request_number}/merge"
    capture(
        "codeql_analyses",
        f"repos/{repository}/code-scanning/analyses",
        parameters={"per_page": "100", "ref": merge_ref},
        paginated=True,
    )
    capture(
        "codeql_alerts",
        f"repos/{repository}/code-scanning/alerts",
        parameters={"per_page": "100", "ref": merge_ref, "state": "open"},
        paginated=True,
    )
    capture("ruleset", f"repos/{repository}/rulesets/{RULESET_ID}")
    capture(
        "check_runs",
        f"repos/{repository}/commits/{head_sha}/check-runs",
        parameters={"per_page": "100"},
        paginated=True,
    )
    capture(
        "reviews",
        f"repos/{repository}/pulls/{pull_request_number}/reviews",
        parameters={"per_page": "100"},
        paginated=True,
    )
    capture(
        "requested_reviewers",
        f"repos/{repository}/pulls/{pull_request_number}/requested_reviewers",
    )
    owner, name = repository.split("/", 1)
    capture(
        "review_threads",
        "graphql",
        graphql_variables={
            "owner": owner,
            "name": name,
            "number": str(pull_request_number),
        },
    )
    index = {
        "schema_version": 1,
        "kind": INDEX_KIND,
        "candidate_evidence": candidate,
        "repository": repository,
        "issue_number": 115,
        "pull_request_number": pull_request_number,
        "responses": records,
    }
    index_path = output_directory / "operations-index.json"
    index_path.write_text(
        json.dumps(index, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scope_path = output_directory / "scope.json"
    scope_path.write_text(
        json.dumps(
            {"index": _descriptor(index_path, output_directory, candidate)},
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return index_path, scope_path


RESPONSE_ROLES = frozenset(
    {
        "issue_115",
        "issue_115_comments",
        "pull_request",
        "base_compare",
        "ci_run",
        "ci_jobs",
        "codeql_run",
        "codeql_jobs",
        "codeql_analyses",
        "codeql_alerts",
        "ruleset",
        "check_runs",
        "reviews",
        "requested_reviewers",
        "review_threads",
    }
)
REQUIRED_STATUS_CONTEXTS = (
    "Python 3.14 / ubuntu-latest",
    "Python 3.14 / macos-latest",
)
REQUIRED_CODEQL_JOBS = ("CodeQL / python", "CodeQL / c-cpp")


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(set(value) == expected, f"{label} schema differs")


def _timestamp(value: Any, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise EvidenceError(f"{label} is not an ISO timestamp") from error
    _require(parsed.tzinfo is not None, f"{label} has no timezone")
    return parsed


def _sha40(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _list_pages(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list) and bool(value), f"{label} pages differ")
    result = []
    for index, page in enumerate(value):
        _require(isinstance(page, list), f"{label} page {index} differs")
        result.extend(page)
    return result


def _object_pages(value: Any, key: str, label: str) -> list[Any]:
    _require(isinstance(value, list) and bool(value), f"{label} pages differ")
    result = []
    for index, page in enumerate(value):
        _require(
            isinstance(page, dict) and isinstance(page.get(key), list),
            f"{label} page {index} differs",
        )
        result.extend(page[key])
    return result


def _validate_request(
    record: Any,
    endpoint: str,
    *,
    parameters: dict[str, str] | None = None,
    paginated: bool = False,
    variables: dict[str, str] | None = None,
) -> None:
    _exact_keys(
        record,
        {"endpoint", "parameters", "paginated", "graphql", "variables"},
        f"operations request {endpoint}",
    )
    _require(
        record["endpoint"] == endpoint
        and record["parameters"] == (parameters or {})
        and record["paginated"] is paginated
        and record["graphql"] is (variables is not None)
        and record["variables"] == (variables or {}),
        f"operations request provenance differs: {endpoint}",
    )


def _validate_workflow_run(
    value: Any,
    *,
    expected_name: str,
    candidate_sha: str,
    label: str,
) -> tuple[int, int]:
    _require(isinstance(value, dict), f"{label} response differs")
    _require(
        type(value.get("id")) is int
        and value["id"] > 0
        and value.get("name") == expected_name
        and value.get("event") == "pull_request"
        and value.get("head_sha") == candidate_sha
        and value.get("status") == "completed"
        and value.get("conclusion") == "success"
        and type(value.get("run_attempt")) is int
        and value["run_attempt"] > 0,
        f"{label} is incomplete, failing, or not candidate-bound",
    )
    return value["id"], value["run_attempt"]


def _validate_jobs(
    values: list[Any],
    names: tuple[str, ...],
    run_id: int,
    run_attempt: int,
    label: str,
) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for name in names:
        selected = [
            item
            for item in values
            if isinstance(item, dict) and item.get("name") == name
        ]
        _require(len(selected) == 1, f"{label} {name!r} closure differs")
        item = selected[0]
        started = _timestamp(item.get("started_at"), f"{label} {name} start")
        completed = _timestamp(item.get("completed_at"), f"{label} {name} completion")
        _require(
            type(item.get("id")) is int
            and item["id"] > 0
            and item.get("run_id") == run_id
            and item.get("run_attempt") == run_attempt
            and item.get("status") == "completed"
            and item.get("conclusion") == "success"
            and started <= completed,
            f"{label} {name!r} is incomplete or failing",
        )
        matches[name] = item
    return matches


def _validate_ruleset(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "master ruleset response differs")
    _require(
        value.get("id") == RULESET_ID
        and value.get("name") == "Protect master"
        and value.get("target") == "branch"
        and value.get("source_type") == "Repository"
        and value.get("source") == REPOSITORY
        and value.get("enforcement") == "active"
        and value.get("bypass_actors") == []
        and value.get("current_user_can_bypass") == "never",
        "master ruleset identity or enforcement differs",
    )
    conditions = value.get("conditions")
    _require(
        isinstance(conditions, dict)
        and conditions.get("ref_name")
        == {"exclude": [], "include": ["~DEFAULT_BRANCH"]},
        "master ruleset target differs",
    )
    rules = value.get("rules")
    _require(isinstance(rules, list), "master ruleset rules differ")
    by_type = {
        item.get("type"): item
        for item in rules
        if isinstance(item, dict) and isinstance(item.get("type"), str)
    }
    required_types = {
        "deletion",
        "non_fast_forward",
        "required_linear_history",
        "pull_request",
        "required_status_checks",
        "code_scanning",
    }
    _require(
        set(by_type) == required_types and len(rules) == len(required_types),
        "master ruleset rule closure differs",
    )
    pull = by_type["pull_request"].get("parameters")
    _require(
        isinstance(pull, dict)
        and pull.get("required_approving_review_count") == 0
        and pull.get("required_review_thread_resolution") is True
        and pull.get("require_extra_approval_for_unattributed_changes") is True
        and pull.get("allowed_merge_methods") == ["squash"],
        "master pull-request rule differs",
    )
    status = by_type["required_status_checks"].get("parameters")
    _require(
        isinstance(status, dict)
        and status.get("strict_required_status_checks_policy") is True
        and status.get("do_not_enforce_on_create") is False,
        "master status-check policy differs",
    )
    contexts = status.get("required_status_checks")
    _require(isinstance(contexts, list), "required status contexts differ")
    context_names = [item.get("context") for item in contexts if isinstance(item, dict)]
    _require(
        context_names == list(REQUIRED_STATUS_CONTEXTS)
        and all(
            type(item.get("integration_id")) is int and item["integration_id"] > 0
            for item in contexts
        ),
        "required status context closure differs",
    )
    scanning = by_type["code_scanning"].get("parameters")
    tools = scanning.get("code_scanning_tools") if isinstance(scanning, dict) else None
    _require(
        tools
        == [
            {
                "tool": "CodeQL",
                "security_alerts_threshold": "high_or_higher",
                "alerts_threshold": "errors",
            }
        ],
        "CodeQL ruleset threshold differs",
    )
    return {
        "id": RULESET_ID,
        "required_status_contexts": list(REQUIRED_STATUS_CONTEXTS),
        "quality_threshold": "errors",
        "security_threshold": "high_or_higher",
    }


def evaluate_operations(
    document: Any,
    raw_responses: dict[str, Any],
    expected_candidate: dict[str, str],
) -> dict[str, Any]:
    """Recompute all operational gates from strict raw GitHub API responses."""

    _exact_keys(
        document,
        {
            "schema_version",
            "kind",
            "candidate_evidence",
            "repository",
            "issue_number",
            "pull_request_number",
            "responses",
        },
        "operations index",
    )
    _require(
        type(document["schema_version"]) is int
        and document["schema_version"] == 1
        and document["kind"] == INDEX_KIND
        and document["candidate_evidence"] == expected_candidate
        and document["repository"] == REPOSITORY
        and type(document["issue_number"]) is int
        and document["issue_number"] == 115
        and type(document["pull_request_number"]) is int
        and document["pull_request_number"] > 0,
        "operations index identity differs",
    )
    records = document["responses"]
    _require(
        isinstance(records, dict) and set(records) == RESPONSE_ROLES,
        "operations response closure differs",
    )
    _require(
        set(raw_responses) == RESPONSE_ROLES, "operations raw response closure differs"
    )
    for role, record in records.items():
        _exact_keys(record, {"request", "artifact"}, f"operations response {role}")

    repository = document["repository"]
    number = document["pull_request_number"]
    candidate_sha = expected_candidate["candidate_git_commit"]
    pull = raw_responses["pull_request"]
    _require(isinstance(pull, dict), "pull request response differs")
    base = pull.get("base")
    head = pull.get("head")
    _require(
        isinstance(base, dict) and isinstance(head, dict), "pull request refs differ"
    )
    base_sha = base.get("sha")
    head_sha = head.get("sha")
    merge_sha = pull.get("merge_commit_sha")
    merge_ref = f"refs/pull/{number}/merge"
    _require(
        pull.get("number") == number
        and pull.get("state") == "open"
        and pull.get("draft") is False
        and pull.get("mergeable") is True
        and pull.get("mergeable_state") == "clean"
        and base.get("ref") == "master"
        and _sha40(base_sha)
        and head_sha == candidate_sha
        and _sha40(merge_sha)
        and isinstance(head.get("repo"), dict)
        and head["repo"].get("full_name") == repository,
        "pull request is not clean, mergeable, master-targeted, or candidate-bound",
    )
    expected_requests = {
        "issue_115": (f"repos/{repository}/issues/115", {}, False),
        "issue_115_comments": (
            f"repos/{repository}/issues/115/comments",
            {"per_page": "100"},
            True,
        ),
        "pull_request": (f"repos/{repository}/pulls/{number}", {}, False),
        "base_compare": (
            f"repos/{repository}/compare/{base_sha}...{head_sha}",
            {},
            False,
        ),
        "ci_run": (
            f"repos/{repository}/actions/runs/{raw_responses['ci_run'].get('id')}",
            {},
            False,
        ),
        "ci_jobs": (
            f"repos/{repository}/actions/runs/{raw_responses['ci_run'].get('id')}/jobs",
            {"per_page": "100"},
            True,
        ),
        "codeql_run": (
            f"repos/{repository}/actions/runs/{raw_responses['codeql_run'].get('id')}",
            {},
            False,
        ),
        "codeql_jobs": (
            f"repos/{repository}/actions/runs/{raw_responses['codeql_run'].get('id')}/jobs",
            {"per_page": "100"},
            True,
        ),
        "codeql_analyses": (
            f"repos/{repository}/code-scanning/analyses",
            {"per_page": "100", "ref": merge_ref},
            True,
        ),
        "codeql_alerts": (
            f"repos/{repository}/code-scanning/alerts",
            {"per_page": "100", "ref": merge_ref, "state": "open"},
            True,
        ),
        "ruleset": (f"repos/{repository}/rulesets/{RULESET_ID}", {}, False),
        "check_runs": (
            f"repos/{repository}/commits/{head_sha}/check-runs",
            {"per_page": "100"},
            True,
        ),
        "reviews": (
            f"repos/{repository}/pulls/{number}/reviews",
            {"per_page": "100"},
            True,
        ),
        "requested_reviewers": (
            f"repos/{repository}/pulls/{number}/requested_reviewers",
            {},
            False,
        ),
    }
    for role, (endpoint, parameters, paginated) in expected_requests.items():
        _validate_request(
            records[role]["request"],
            endpoint,
            parameters=parameters,
            paginated=paginated,
        )
    owner, name = repository.split("/", 1)
    _validate_request(
        records["review_threads"]["request"],
        "graphql",
        variables={"owner": owner, "name": name, "number": str(number)},
    )

    issue = raw_responses["issue_115"]
    _require(isinstance(issue, dict), "issue #115 response differs")
    body = issue.get("body")
    _require(
        issue.get("number") == 115
        and issue.get("state") == "closed"
        and isinstance(issue.get("closed_at"), str)
        and isinstance(body, str)
        and "## Implementation work" in body
        and "- [ ]" not in body,
        "issue #115 is not closed with its runtime checklist complete",
    )
    _timestamp(issue["closed_at"], "issue #115 closure")
    comments = _list_pages(raw_responses["issue_115_comments"], "issue #115 comments")
    handoffs = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        text = comment.get("body")
        user = comment.get("user")
        if (
            isinstance(text, str)
            and "issue #123 final runtime-observation handoff" in text.lower()
            and candidate_sha in text
            and f"PR #{number}" in text
            and "torch.utils.benchmark" in text
            and "profiler" in text.lower()
            and isinstance(user, dict)
            and user.get("login") == "ruddyscent"
            and comment.get("author_association") == "OWNER"
        ):
            handoffs.append(comment)
    _require(
        len(handoffs) == 1, "issue #115 final runtime handoff comment closure differs"
    )
    handoff = handoffs[0]
    _require(
        type(handoff.get("id")) is int and handoff["id"] > 0,
        "issue #115 handoff id differs",
    )

    comparison = raw_responses["base_compare"]
    _require(
        isinstance(comparison, dict)
        and comparison.get("status") in {"ahead", "identical"}
        and comparison.get("behind_by") == 0
        and type(comparison.get("ahead_by")) is int
        and comparison["ahead_by"] >= 0
        and isinstance(comparison.get("base_commit"), dict)
        and comparison["base_commit"].get("sha") == base_sha
        and isinstance(comparison.get("merge_base_commit"), dict)
        and comparison["merge_base_commit"].get("sha") == base_sha,
        "candidate branch is not up to date with the exact PR base",
    )
    commits = comparison.get("commits")
    _require(
        comparison["ahead_by"] == 0
        or isinstance(commits, list)
        and commits
        and commits[-1].get("sha") == head_sha,
        "compare response does not terminate at the candidate",
    )

    ci_id, ci_attempt = _validate_workflow_run(
        raw_responses["ci_run"],
        expected_name="CI",
        candidate_sha=candidate_sha,
        label="CI run",
    )
    codeql_id, codeql_attempt = _validate_workflow_run(
        raw_responses["codeql_run"],
        expected_name="CodeQL",
        candidate_sha=candidate_sha,
        label="CodeQL run",
    )
    ci_jobs = _validate_jobs(
        _object_pages(raw_responses["ci_jobs"], "jobs", "CI jobs"),
        REQUIRED_STATUS_CONTEXTS,
        ci_id,
        ci_attempt,
        "CI jobs",
    )
    codeql_jobs = _validate_jobs(
        _object_pages(raw_responses["codeql_jobs"], "jobs", "CodeQL jobs"),
        REQUIRED_CODEQL_JOBS,
        codeql_id,
        codeql_attempt,
        "CodeQL jobs",
    )
    ruleset = _validate_ruleset(raw_responses["ruleset"])

    check_runs = _object_pages(raw_responses["check_runs"], "check_runs", "check runs")
    for context in REQUIRED_STATUS_CONTEXTS:
        selected = [
            item
            for item in check_runs
            if isinstance(item, dict) and item.get("name") == context
        ]
        _require(len(selected) == 1, f"required check {context!r} closure differs")
        check = selected[0]
        _require(
            check.get("head_sha") == head_sha
            and check.get("status") == "completed"
            and check.get("conclusion") == "success"
            and isinstance(check.get("app"), dict)
            and check["app"].get("slug") == "github-actions",
            f"required check {context!r} is incomplete or failing",
        )

    analyses = _list_pages(raw_responses["codeql_analyses"], "CodeQL analyses")
    analysis_details = {}
    for category, job_name in (
        ("/language:python", "CodeQL / python"),
        ("/language:c-cpp", "CodeQL / c-cpp"),
    ):
        selected = [
            item
            for item in analyses
            if isinstance(item, dict)
            and item.get("category") == category
            and item.get("commit_sha") == merge_sha
            and item.get("ref") == merge_ref
        ]
        _require(bool(selected), f"CodeQL analysis {category} is missing")
        selected.sort(
            key=lambda item: _timestamp(
                item.get("created_at"), f"CodeQL {category} creation"
            )
        )
        analysis = selected[-1]
        created = _timestamp(analysis.get("created_at"), f"CodeQL {category} creation")
        job = codeql_jobs[job_name]
        _require(
            _timestamp(job["started_at"], f"{job_name} start")
            <= created
            <= _timestamp(job["completed_at"], f"{job_name} completion")
            and analysis.get("analysis_key") == ".github/workflows/codeql.yml:analyze"
            and analysis.get("error") == ""
            and analysis.get("warning") == ""
            and type(analysis.get("rules_count")) is int
            and analysis["rules_count"] > 0
            and isinstance(analysis.get("tool"), dict)
            and analysis["tool"].get("name") == "CodeQL",
            f"CodeQL analysis {category} is incomplete or failing",
        )
        analysis_details[category] = {
            "id": analysis.get("id"),
            "results_count": analysis.get("results_count"),
            "rules_count": analysis["rules_count"],
        }

    alerts = _list_pages(raw_responses["codeql_alerts"], "CodeQL alerts")
    quality_blockers = []
    security_blockers = []
    for alert in alerts:
        _require(isinstance(alert, dict), "CodeQL alert record differs")
        rule = alert.get("rule")
        instance = alert.get("most_recent_instance")
        _require(
            alert.get("state") == "open"
            and isinstance(rule, dict)
            and isinstance(instance, dict)
            and instance.get("ref") == merge_ref
            and instance.get("commit_sha") == merge_sha,
            "CodeQL alert is not exact-merge-bound",
        )
        if rule.get("severity") == "error":
            quality_blockers.append(alert.get("number"))
        if rule.get("security_severity_level") in {"high", "critical"}:
            security_blockers.append(alert.get("number"))
    _require(
        not quality_blockers and not security_blockers,
        "CodeQL has error-level quality or high-or-higher security blockers",
    )

    requested = raw_responses["requested_reviewers"]
    _require(
        isinstance(requested, dict)
        and requested.get("users") == []
        and requested.get("teams") == [],
        "GitHub still requests additional reviewers",
    )
    reviews = _list_pages(raw_responses["reviews"], "pull-request reviews")
    latest_reviews = {}
    for review in reviews:
        _require(
            isinstance(review, dict) and isinstance(review.get("user"), dict),
            "review record differs",
        )
        login = review["user"].get("login")
        _require(isinstance(login, str) and bool(login), "review author differs")
        key = (
            _timestamp(review.get("submitted_at"), f"review by {login}"),
            review.get("id", 0),
        )
        if login not in latest_reviews or key > latest_reviews[login][0]:
            latest_reviews[login] = (key, review)
    _require(
        all(
            record[1].get("state") != "CHANGES_REQUESTED"
            for record in latest_reviews.values()
        ),
        "a latest pull-request review still requests changes",
    )
    threads_root = raw_responses["review_threads"]
    try:
        threads = threads_root["data"]["repository"]["pullRequest"]["reviewThreads"]
    except (KeyError, TypeError) as error:
        raise EvidenceError("review-thread GraphQL response differs") from error
    _require(
        isinstance(threads, dict)
        and isinstance(threads.get("nodes"), list)
        and type(threads.get("totalCount")) is int
        and threads["totalCount"] == len(threads["nodes"])
        and isinstance(threads.get("pageInfo"), dict)
        and threads["pageInfo"].get("hasNextPage") is False
        and all(
            isinstance(item, dict) and item.get("isResolved") is True
            for item in threads["nodes"]
        ),
        "pull-request review conversations are unresolved or incomplete",
    )

    macos_job = ci_jobs["Python 3.14 / macos-latest"]
    return {
        "candidate_evidence": expected_candidate,
        "repository": repository,
        "pull_request": {
            "number": number,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "merge_sha": merge_sha,
            "merge_ref": merge_ref,
        },
        "ruleset": ruleset,
        "ci_run_id": ci_id,
        "codeql_run_id": codeql_id,
        "macos_job": {
            "run_id": ci_id,
            "started_at": macos_job["started_at"],
            "completed_at": macos_job["completed_at"],
        },
        "codeql_analyses": analysis_details,
        "codeql_quality_blockers": quality_blockers,
        "codeql_security_blockers": security_blockers,
        "latest_review_states": {
            login: record[1].get("state") for login, record in latest_reviews.items()
        },
        "review_threads": threads["totalCount"],
        "handoff_comment_id": handoff["id"],
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--pull-request", type=int, required=True)
    parser.add_argument("--ci-run-id", type=int, required=True)
    parser.add_argument("--codeql-run-id", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    capture_operations(
        repository=args.repository,
        pull_request_number=args.pull_request,
        ci_run_id=args.ci_run_id,
        codeql_run_id=args.codeql_run_id,
        output_directory=args.output_directory,
        manifest=args.manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
