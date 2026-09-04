#!/usr/bin/env python3
"""Capture raw GitHub operational evidence required to close issue #123."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from benchmarks.host_contract import candidate_evidence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "native_oracle_workloads.json"
INDEX_KIND = "issue-123-operations-evidence-index"
MEDIA_TYPE_JSON = "application/json"
REPOSITORY = "ruddyscent/gmes"
TARGET_ISSUE_NUMBER = 123
HANDOFF_ISSUE_NUMBER = 115
PULL_REQUEST_NUMBER = 167
RULESET_ID = 21130311
PAGE_SIZE = 100
MAX_PAGES = 1000
MAX_OPERATIONS_INDEX_BYTES = 16 * 1024 * 1024
MAX_GITHUB_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_PUBLICATION_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_PUBLICATION_POLICY_BYTES = 64 * 1024 * 1024
MAX_PUBLICATION_ASSET_BYTES = 256 * 1024 * 1024
GITHUB_API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
REJECTED_CANDIDATE_SHA = "74e8c6947012814f29618eb9acd59d487534e996"
OWNER_LOGIN = "ruddyscent"
ISSUE_CONTRACT_AMENDMENT_MARKER = "GMES_ISSUE_123_FINAL_CONTRACT_AMENDMENT_V2"
PR_CANDIDATE_INSIGHT_MARKER = "GMES_PR_167_FINAL_CANDIDATE_INSIGHT_V2"
HANDOFF_MARKER = "GMES_ISSUE_115_FINAL_RUNTIME_HANDOFF_V2"
TECHNICAL_RELEASE_TAG_PREFIX = "issue-123-technical-evidence-"
TECHNICAL_RELEASE_ASSETS = {
    "technical_evidence": "issue-123-public-technical-evidence.zip",
    "technical_summary": "issue-123-technical-summary.json",
    "raw_timing": "issue-115-raw-timing.json",
    "event_profiler": "issue-115-event-level-profiler.json",
}
BASELINE_V3_ROOT_COMMIT = "821c075b9328e02c3f3e5d16488a44b64ff08c04"
BASELINE_V3_ONE_URL = (
    "https://github.com/ruddyscent/gmes/releases/download/"
    "issue-123-torch-cpu-baseline-v3/torch-cpu-baseline-one.json"
)
BASELINE_V3_ONE_SIZE = 18281
BASELINE_V3_ONE_SHA256 = (
    "c8eba3c17ccae5ba744a8fbc90b89d72a77dcf0624339cda1deb4d7f594395ed"
)
BASELINE_V3_PHYSICAL_URL = (
    "https://github.com/ruddyscent/gmes/releases/download/"
    "issue-123-torch-cpu-baseline-v3/torch-cpu-baseline-physical.json"
)
BASELINE_V3_PHYSICAL_SIZE = 18292
BASELINE_V3_PHYSICAL_SHA256 = (
    "b1a3c82a069c2475560468a7b8d0a237db89e857bdce96f8fc812449b5c35602"
)
BASELINE_V3_HOST_COMMITMENT = (
    "f7b3b1b0eb13531682ea0381698c60aa9a97c7a3a0dfffc5344b828772f67a56"
)
HANDOFF_CHECKLIST_ITEMS = (
    "- [x] Add runtime-aware repeated measurement using `torch.utils.benchmark` "
    "once Torch is present, while preserving a native reference runner.",
    "- [x] Add optional profiler capture for graph breaks, kernel launches, "
    "device copies, and allocator behavior.",
)
PUBLICATION_RECEIPT_KIND = "issue-123-publication-receipt"
PUBLICATION_EXECUTION_WITNESS_KIND = "issue-123-public-execution-witness"
PUBLICATION_RELEASE_CAPTURE_KIND = "issue-123-public-release-capture"
PUBLICATION_EXECUTION_WITNESS_PATH = "execution/witness.json"
PUBLICATION_JOB_NAMES = (
    "Python 3.14 / ubuntu-latest",
    "Python 3.14 / macos-latest",
    "CodeQL / python",
    "CodeQL / c-cpp",
)
PUBLICATION_EXECUTION_CLAIMS = (
    ("cpu-eager", "cpu"),
    ("cuda-eager", "single_gpu"),
    ("cuda-graph", "two_gpu"),
)
LIVE_VERIFICATION_RECEIPT_KIND = "issue-123-live-operations-verification-receipt"
LIVE_VERIFICATION_RECEIPT_SCHEMA_VERSION = 3
BASELINE_AUTHORITY_DOMAIN = "gmes.issue123.baseline-authority.v1"
CHECKLIST_TRANSITION_DOMAIN = "gmes.issue123.checklist-complete-response-transition.v1"
CHECKLIST_TRANSITION_UPDATED_AT_SENTINEL = "<contract-permitted-updated-at>"
BASELINE_RELEASE_TAG = "issue-123-torch-cpu-baseline-v3"
FINAL_CHECKLIST_SECTION = "## Implementation work"
FINAL_CHECKLIST_CHECKED = (
    "- [x] publish the final bundle",
    "- [x] complete the post-bundle checklist",
)
FINAL_CHECKLIST_UNCHECKED = tuple(
    line.replace("[x]", "[ ]") for line in FINAL_CHECKLIST_CHECKED
)


@dataclass(frozen=True, slots=True)
class AuthenticatedPostBundleExpectation:
    checked_lines: tuple[str, str]
    o0_canonical_response_sha256: str
    o1_canonical_response_sha256: str
    o1_body_sha256: str
    o1_updated_at: str
    b0_inventory_root: str
    b0_reopen_receipt_sha256: str
    b0_reopened_at: str
    checklist_transition_sha256: str


@dataclass(frozen=True, slots=True)
class BaselineAssetExpectation:
    ordinal: int
    thread_mode: str
    name: str
    publication_url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BaselineAuthoritySet:
    root_commit: str
    assets: tuple[BaselineAssetExpectation, BaselineAssetExpectation]


PRODUCTION_BASELINE_AUTHORITY_SET = BaselineAuthoritySet(
    root_commit=BASELINE_V3_ROOT_COMMIT,
    assets=(
        BaselineAssetExpectation(
            ordinal=0,
            thread_mode="one",
            name="torch-cpu-baseline-one.json",
            publication_url=BASELINE_V3_ONE_URL,
            size_bytes=BASELINE_V3_ONE_SIZE,
            sha256=BASELINE_V3_ONE_SHA256,
        ),
        BaselineAssetExpectation(
            ordinal=1,
            thread_mode="physical",
            name="torch-cpu-baseline-physical.json",
            publication_url=BASELINE_V3_PHYSICAL_URL,
            size_bytes=BASELINE_V3_PHYSICAL_SIZE,
            sha256=BASELINE_V3_PHYSICAL_SHA256,
        ),
    ),
)


SAFE_RESPONSE_HEADERS = (
    "content-type",
    "etag",
    "last-modified",
    "link",
    "x-github-api-version-selected",
    "x-github-media-type",
)
SUPERSEDED_OWNER_COMMENTS = (
    {
        "field": "SUPERSEDES_BASELINE_ISSUE_COMMENT",
        "id": 5471826009,
        "stream": "issue_123_comments",
        "issue_number": 123,
        "html_kind": "issues",
        "role": "cpu-acceptance-v2-baseline-closure-plan",
        "created_at": "2026-08-30T23:06:54Z",
        "updated_at": "2026-08-30T23:06:54Z",
        "body_sha256": (
            "fc0fa360bf8ab23b5def748cd89b82b3a28bac995d753a522e222cc04c20e929"
        ),
        "required_fragments": ("CPU acceptance v2", "#123"),
    },
    {
        "field": "SUPERSEDES_DM2_ISSUE_COMMENT",
        "id": 5501920996,
        "stream": "issue_123_comments",
        "issue_number": 123,
        "html_kind": "issues",
        "role": "dm2-float32-numerical-contract-amendment",
        "created_at": "2026-09-01T23:26:25Z",
        "updated_at": "2026-09-01T23:26:25Z",
        "body_sha256": (
            "70e567d3d900f64405b85d82e4f3d9b321a6276b345de8e4a9fdf87f8483803f"
        ),
        "required_fragments": ("DM2 float32", "rtol=6e-4", "atol=3e-6"),
    },
    {
        "field": "SUPERSEDES_DM2_PR_COMMENT",
        "id": 5501929771,
        "stream": "pull_request_comments",
        "issue_number": 167,
        "html_kind": "pull",
        "role": "dm2-cuda-graph-insight-and-evidence-reset",
        "created_at": "2026-09-01T23:27:02Z",
        "updated_at": "2026-09-01T23:27:02Z",
        "body_sha256": (
            "3b6660ba2922218672f5e199be7a2ce8aca9f1f65a9ce7ed5315cf388b38888d"
        ),
        "required_fragments": ("DM2 CUDA Graph", "evidence reset"),
    },
    {
        "field": "SUPERSEDES_SINGLE_GPU_ISSUE_COMMENT",
        "id": 5504326509,
        "stream": "issue_123_comments",
        "issue_number": 123,
        "html_kind": "issues",
        "role": "single-gpu-3d-float64-late-residual-amendment",
        "created_at": "2026-09-02T04:21:02Z",
        "updated_at": "2026-09-02T04:21:02Z",
        "body_sha256": (
            "91c3570c6146b9e551c6b9b2bb65b681173a0d4478a04c2654ae3c079219226d"
        ),
        "required_fragments": (
            "single-gpu-3d",
            "float64",
            "captures 20 and 100",
            "L∞",
            "L2",
            "≤ 1e-6",
            "2e-12",
            "all-zero reference arrays require exact equality",
            "integer, boolean, topology, shape, dtype, and finiteness",
            "point-source-semantic-v1",
        ),
    },
)
# `gh api --paginate` supports one GraphQL cursor, so only reviewThreads exposes
# pageInfo; totalCount == 1 closes the smaller closingIssuesReferences connection.
PULL_REQUEST_CONTEXT_QUERY = """
query($owner:String!,$name:String!,$number:Int!,$endCursor:String){
  repository(owner:$owner,name:$name){
    nameWithOwner
    pullRequest(number:$number){
      number
      headRefOid
      closingIssuesReferences(first:2){
        totalCount
        nodes{number repository{nameWithOwner}}
      }
      reviews(first:1,states:[APPROVED,CHANGES_REQUESTED,COMMENTED,DISMISSED]){
        totalCount
      }
      reviewThreads(first:100,after:$endCursor){
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


def _typed_json_equal(value: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(value, dict)
            and set(value) == set(expected)
            and all(
                _typed_json_equal(value[key], item) for key, item in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(
                _typed_json_equal(actual, item)
                for actual, item in zip(value, expected, strict=True)
            )
        )
    return type(value) is type(expected) and value == expected


def _strict_json_values(raw: bytes, label: str) -> list[Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise EvidenceError(f"{label} repeats JSON key {key!r}")
            value[key] = item
        return value

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{label} is not strict UTF-8 JSON") from error
    decoder = json.JSONDecoder(
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            EvidenceError(f"{label} contains {value}")
        ),
    )
    values = []
    position = 0
    while position < len(text):
        whitespace = re.match(r"\s*", text[position:])
        assert whitespace is not None
        position += whitespace.end()
        if position == len(text):
            break
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"{label} is not strict UTF-8 JSON") from error
        values.append(value)
    _require(bool(values), f"{label} is empty")
    return values


def _strict_json(raw: bytes, label: str) -> Any:
    values = _strict_json_values(raw, label)
    _require(len(values) == 1, f"{label} contains multiple JSON values")
    return values[0]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise EvidenceError("JSON value is not canonicalizable") from error
    return (rendered + "\n").encode("utf-8")


def _canonical_api_url(value: str, label: str) -> str:
    try:
        parsed = urlsplit(value)
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as error:
        raise EvidenceError(f"{label} is malformed") from error
    _require(
        parsed.scheme == "https"
        and parsed.hostname == "api.github.com"
        and parsed.username is None
        and parsed.password is None
        and parsed.port is None
        and not parsed.fragment,
        f"{label} is not a canonical GitHub API URL",
    )
    return urlunsplit(
        (
            "https",
            "api.github.com",
            parsed.path,
            urlencode(sorted(query)),
            "",
        )
    )


def _link_relations(value: str | None, label: str) -> dict[str, str]:
    if value is None:
        return {}
    _require(isinstance(value, str) and bool(value), f"{label} differs")
    result: dict[str, str] = {}
    for item in value.split(","):
        match = re.fullmatch(r'\s*<([^>]+)>\s*;\s*rel="([^"]+)"\s*', item)
        _require(match is not None, f"{label} differs")
        url, relation = match.groups()
        _require(
            relation in {"first", "prev", "next", "last"} and relation not in result,
            f"{label} relation closure differs",
        )
        result[relation] = _canonical_api_url(url, f"{label} {relation}")
    return {key: result[key] for key in sorted(result)}


def _included_response_frames(raw: bytes, label: str) -> list[dict[str, Any]]:
    _require(
        type(raw) is bytes and 0 < len(raw) <= MAX_GITHUB_RESPONSE_BYTES,
        f"{label} byte size differs",
    )
    starts = [
        match.start()
        for match in re.finditer(
            rb"(?m)^HTTP/[0-9.]+ [0-9]{3}(?: [^\r\n]*)?\r?$",
            raw,
        )
    ]
    _require(starts and starts[0] == 0, f"{label} omits HTTP response headers")
    starts.append(len(raw))
    frames = []
    for index in range(len(starts) - 1):
        framed = raw[starts[index] : starts[index + 1]]
        separator = re.search(rb"\r?\n\r?\n", framed)
        _require(separator is not None, f"{label} page {index + 1} is malformed")
        header_raw = framed[: separator.start()]
        body_raw = framed[separator.end() :]
        lines = re.split(rb"\r?\n", header_raw)
        try:
            status_line = lines[0].decode("ascii").rstrip("\r")
        except UnicodeDecodeError as error:
            raise EvidenceError(f"{label} page {index + 1} status differs") from error
        status = re.fullmatch(r"HTTP/[0-9.]+ ([0-9]{3})(?: [^\r\n]*)?", status_line)
        _require(status is not None, f"{label} page {index + 1} status differs")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            _require(b":" in line, f"{label} page {index + 1} header differs")
            name_raw, value_raw = line.split(b":", 1)
            try:
                name = name_raw.decode("ascii").strip().lower()
                header_value = value_raw.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise EvidenceError(
                    f"{label} page {index + 1} header differs"
                ) from error
            _require(
                re.fullmatch(r"[a-z0-9-]+", name) is not None and name not in headers,
                f"{label} page {index + 1} header closure differs",
            )
            headers[name] = header_value
        value = _strict_json(body_raw, f"{label} page {index + 1} body")
        frames.append(
            {
                "status": int(status.group(1)),
                "headers": headers,
                "value": value,
            }
        )
    _require(len(frames) <= MAX_PAGES, f"{label} has too many pages")
    return frames


def _safe_response_headers(
    headers: Mapping[str, str], label: str, *, graphql: bool
) -> dict[str, Any]:
    content_type = headers.get("content-type")
    api_version = headers.get("x-github-api-version-selected")
    media_type = headers.get("x-github-media-type")
    expected_media_type = (
        "github.v4; format=json" if graphql else "github.v3; format=json"
    )
    _require(
        isinstance(content_type, str)
        and content_type.lower().startswith("application/json")
        and (
            (
                graphql
                and api_version in {None, GITHUB_API_HEADERS["X-GitHub-Api-Version"]}
            )
            or (
                not graphql
                and api_version == GITHUB_API_HEADERS["X-GitHub-Api-Version"]
            )
        )
        and media_type == expected_media_type,
        f"{label} required response headers differ",
    )
    return {
        "content-type": content_type,
        "etag": headers.get("etag"),
        "last-modified": headers.get("last-modified"),
        "link": _link_relations(headers.get("link"), f"{label} Link"),
        "x-github-api-version-selected": api_version,
        "x-github-media-type": media_type,
    }


def _page_item_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and type(value.get("total_count")) is int:
        for key in ("jobs", "check_runs", "items"):
            if isinstance(value.get(key), list):
                return len(value[key])
    return None


def _graphql_next(value: Any, label: str) -> tuple[bool, str | None]:
    try:
        page_info = value["data"]["repository"]["pullRequest"]["reviewThreads"][
            "pageInfo"
        ]
    except (KeyError, TypeError) as error:
        raise EvidenceError(f"{label} GraphQL pageInfo differs") from error
    _require(
        isinstance(page_info, dict) and type(page_info.get("hasNextPage")) is bool,
        f"{label} GraphQL pageInfo differs",
    )
    has_next = page_info["hasNextPage"]
    cursor = page_info.get("endCursor") if has_next else None
    _require(
        not has_next or (isinstance(cursor, str) and bool(cursor)),
        f"{label} GraphQL next cursor differs",
    )
    return has_next, cursor


def _response_capture_from_frames(
    frames: list[dict[str, Any]],
    *,
    paginated: bool,
    graphql: bool,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    _require(bool(frames), f"{label} is empty")
    _require(paginated or len(frames) == 1, f"{label} unexpectedly paginates")
    pages = []
    values = []
    for index, frame in enumerate(frames):
        ordinal = index + 1
        value = frame["value"]
        body = _canonical_json_bytes(value)
        safe_headers = _safe_response_headers(
            frame["headers"], f"{label} page {ordinal}", graphql=graphql
        )
        if graphql:
            has_next, next_value = _graphql_next(value, label)
            next_page = (
                {"kind": "graphql-cursor", "value": next_value} if has_next else None
            )
        else:
            next_value = safe_headers["link"].get("next")
            has_next = next_value is not None
            next_page = {"kind": "rest-link", "value": next_value} if has_next else None
        _require(
            has_next is (index < len(frames) - 1),
            f"{label} final-page/no-next relationship differs",
        )
        pages.append(
            {
                "ordinal": ordinal,
                "status": frame["status"],
                "headers": safe_headers,
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "body_size_bytes": len(body),
                "item_count": _page_item_count(value),
                "has_next": has_next,
                "next": next_page,
            }
        )
        values.append(value)
    stored = values if paginated else values[0]
    canonical = _canonical_json_bytes(stored)
    return canonical, {
        "canonical_response_sha256": hashlib.sha256(canonical).hexdigest(),
        "canonical_response_size_bytes": len(canonical),
        "pages": pages,
    }


def _github_api_capture(
    endpoint: str,
    *,
    parameters: dict[str, str] | None = None,
    paginated: bool = False,
    graphql_variables: dict[str, str | int] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    command = ["gh", "api", "--hostname", "github.com", endpoint]
    for name, value in GITHUB_API_HEADERS.items():
        command.extend(("-H", f"{name}: {value}"))
    command.append("--include")
    if paginated:
        command.extend(("--paginate", "--jq", "."))
    if parameters:
        command.extend(("-X", "GET"))
        for key, value in sorted(parameters.items()):
            command.extend(("-f", f"{key}={value}"))
    if graphql_variables is not None:
        command.extend(("-f", f"query={PULL_REQUEST_CONTEXT_QUERY}"))
        for key, value in sorted(graphql_variables.items()):
            flag = "-F" if key == "number" else "-f"
            command.extend((flag, f"{key}={value}"))
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError(f"GitHub API request failed: {endpoint}") from error
    label = f"GitHub API response {endpoint}"
    frames = _included_response_frames(completed.stdout, label)
    return _response_capture_from_frames(
        frames,
        paginated=paginated,
        graphql=graphql_variables is not None,
        label=label,
    )


def _github_api_raw(
    endpoint: str,
    *,
    parameters: dict[str, str] | None = None,
    paginated: bool = False,
    graphql_variables: dict[str, str | int] | None = None,
) -> bytes:
    raw, _capture = _github_api_capture(
        endpoint,
        parameters=parameters,
        paginated=paginated,
        graphql_variables=graphql_variables,
    )
    return raw


def _descriptor(path: Path, base: Path, candidate: dict[str, str]) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": MEDIA_TYPE_JSON,
        "candidate_evidence": candidate,
    }


def _publication_receipt_envelope(
    path: Path,
    candidate: Mapping[str, str],
) -> dict[str, Any]:
    try:
        raw = path.resolve(strict=True).read_bytes()
    except OSError as error:
        raise EvidenceError("publication receipt is unavailable") from error
    _require(
        0 < len(raw) <= MAX_PUBLICATION_RECEIPT_BYTES,
        "publication receipt byte size differs",
    )
    document = _strict_json(raw, "publication receipt")
    canonical = _canonical_json_bytes(document)
    _require(raw == canonical, "publication receipt bytes are not canonical")
    _require(isinstance(document, dict), "publication receipt differs")
    bindings = document.get("bindings")
    _require(
        type(document.get("schema_version")) is int
        and document["schema_version"] == 1
        and document.get("kind") == PUBLICATION_RECEIPT_KIND
        and isinstance(bindings, dict)
        and bindings.get("final_sha") == candidate.get("candidate_git_commit")
        and bindings.get("manifest_sha256") == candidate.get("manifest_sha256"),
        "publication receipt candidate or manifest binding differs",
    )
    return {
        "media_type": MEDIA_TYPE_JSON,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "document": document,
    }


def capture_operations(
    *,
    repository: str,
    pull_request_number: int,
    ci_run_id: int,
    codeql_run_id: int,
    technical_release_tag: str,
    publication_receipt: Path,
    output_directory: Path,
    manifest: Path = DEFAULT_MANIFEST,
) -> tuple[Path, Path]:
    """Capture strict API response pages and emit a descriptor-only index."""

    _require(repository == REPOSITORY, "operations repository differs")
    _require(
        type(pull_request_number) is int and pull_request_number == PULL_REQUEST_NUMBER,
        f"operations pull request must be #{PULL_REQUEST_NUMBER}",
    )
    _require(
        all(type(value) is int and value > 0 for value in (ci_run_id, codeql_run_id))
        and ci_run_id != codeql_run_id,
        "workflow run ids must be distinct positive integers",
    )
    candidate = candidate_evidence(manifest.resolve(strict=True))
    candidate_sha = candidate.get("candidate_git_commit")
    _require(
        isinstance(candidate_sha, str)
        and re.fullmatch(r"[0-9a-f]{40}", candidate_sha)
        and candidate_sha != REJECTED_CANDIDATE_SHA,
        "FINAL_SHA is malformed or is the rejected candidate",
    )
    _require(
        technical_release_tag == f"{TECHNICAL_RELEASE_TAG_PREFIX}{candidate_sha}"
        and not technical_release_tag.startswith("v"),
        "technical release tag is not the non-v FINAL_SHA-bound tag",
    )
    publication_receipt_record = _publication_receipt_envelope(
        publication_receipt, candidate
    )
    _require_authenticated_gh()
    output_directory = output_directory.resolve()
    _require(not output_directory.exists(), "operations output already exists")
    output_directory.mkdir(parents=True)
    raw_directory = output_directory / "raw"
    raw_directory.mkdir()
    records: dict[str, dict[str, Any]] = {}
    response_captures: dict[str, dict[str, Any]] = {}

    def capture(
        role: str,
        endpoint: str,
        *,
        parameters: dict[str, str] | None = None,
        paginated: bool = False,
        graphql_variables: dict[str, str | int] | None = None,
    ) -> Any:
        raw, response_capture = _github_api_capture(
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
                "method": "POST" if graphql_variables is not None else "GET",
                "headers": GITHUB_API_HEADERS,
                "parameters": parameters or {},
                "paginated": paginated,
                "jq": "." if paginated else None,
                "graphql": graphql_variables is not None,
                "query": (
                    PULL_REQUEST_CONTEXT_QUERY
                    if graphql_variables is not None
                    else None
                ),
                "variables": graphql_variables or {},
            },
            "artifact": _descriptor(path, output_directory, candidate),
        }
        response_captures[role] = response_capture
        return _strict_json(raw, role)

    technical_release = capture(
        "technical_release",
        f"repos/{repository}/releases/tags/{technical_release_tag}",
    )
    _require(isinstance(technical_release, dict), "technical release response differs")
    technical_release_id = technical_release.get("id")
    _require(
        type(technical_release_id) is int and technical_release_id > 0,
        "technical release id differs",
    )
    capture(
        "technical_release_assets",
        f"repos/{repository}/releases/{technical_release_id}/assets",
        parameters={"per_page": str(PAGE_SIZE)},
        paginated=True,
    )
    capture(
        "technical_release_tag",
        f"repos/{repository}/git/ref/tags/{technical_release_tag}",
    )
    capture(
        "issue_123",
        f"repos/{repository}/issues/{TARGET_ISSUE_NUMBER}",
    )
    capture(
        "issue_123_comments",
        f"repos/{repository}/issues/{TARGET_ISSUE_NUMBER}/comments",
        parameters={"per_page": str(PAGE_SIZE)},
        paginated=True,
    )
    capture("issue_115", f"repos/{repository}/issues/{HANDOFF_ISSUE_NUMBER}")
    capture(
        "issue_115_comments",
        f"repos/{repository}/issues/{HANDOFF_ISSUE_NUMBER}/comments",
        parameters={"per_page": str(PAGE_SIZE)},
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
        pull.get("number") == PULL_REQUEST_NUMBER and head_sha == candidate_sha,
        "PR head does not match the clean candidate",
    )
    capture(
        "pull_request_comments",
        f"repos/{repository}/issues/{PULL_REQUEST_NUMBER}/comments",
        parameters={"per_page": str(PAGE_SIZE)},
        paginated=True,
    )
    capture("candidate_commit", f"repos/{repository}/commits/{candidate_sha}")
    capture(
        "base_compare",
        f"repos/{repository}/compare/{base_sha}...{head_sha}",
    )
    capture("ci_run", f"repos/{repository}/actions/runs/{ci_run_id}")
    capture(
        "ci_jobs",
        f"repos/{repository}/actions/runs/{ci_run_id}/jobs",
        parameters={"per_page": str(PAGE_SIZE)},
        paginated=True,
    )
    capture("codeql_run", f"repos/{repository}/actions/runs/{codeql_run_id}")
    capture(
        "codeql_jobs",
        f"repos/{repository}/actions/runs/{codeql_run_id}/jobs",
        parameters={"per_page": str(PAGE_SIZE)},
        paginated=True,
    )
    merge_ref = f"refs/pull/{pull_request_number}/merge"
    capture(
        "codeql_analyses",
        f"repos/{repository}/code-scanning/analyses",
        parameters={"per_page": str(PAGE_SIZE), "ref": merge_ref},
        paginated=True,
    )
    capture(
        "codeql_alerts",
        f"repos/{repository}/code-scanning/alerts",
        parameters={
            "per_page": str(PAGE_SIZE),
            "ref": merge_ref,
            "state": "open",
        },
        paginated=True,
    )
    capture("ruleset", f"repos/{repository}/rulesets/{RULESET_ID}")
    capture(
        "check_runs",
        f"repos/{repository}/commits/{head_sha}/check-runs",
        parameters={"per_page": str(PAGE_SIZE)},
        paginated=True,
    )
    capture(
        "reviews",
        f"repos/{repository}/pulls/{pull_request_number}/reviews",
        parameters={"per_page": str(PAGE_SIZE)},
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
            "number": pull_request_number,
        },
        paginated=True,
    )
    index = {
        "schema_version": 2,
        "kind": INDEX_KIND,
        "candidate_evidence": candidate,
        "repository": repository,
        "target_issue_number": TARGET_ISSUE_NUMBER,
        "handoff_issue_number": HANDOFF_ISSUE_NUMBER,
        "pull_request_number": pull_request_number,
        "ci_run_id": ci_run_id,
        "codeql_run_id": codeql_run_id,
        "technical_release_tag": technical_release_tag,
        "technical_release_id": technical_release_id,
        "publication_receipt": publication_receipt_record,
        "responses": records,
        "response_captures": response_captures,
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


RESPONSE_ROLE_ORDER = (
    "technical_release",
    "technical_release_assets",
    "technical_release_tag",
    "issue_123",
    "issue_123_comments",
    "issue_115",
    "issue_115_comments",
    "pull_request",
    "pull_request_comments",
    "candidate_commit",
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
)
RESPONSE_ROLES = frozenset(RESPONSE_ROLE_ORDER)
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


def _validate_post_bundle_checklist(
    issue: Mapping[str, Any],
    expected_state: str | None = None,
) -> dict[str, Any]:
    """Validate the exact two-line O0/O1 checklist state."""

    _require(
        expected_state in {None, "unchecked", "checked"},
        "post-bundle checklist expected state differs",
    )
    body = issue.get("body")
    updated_at = issue.get("updated_at")
    _require(
        type(body) is str and type(updated_at) is str,
        "post-bundle checklist response differs",
    )
    _timestamp(updated_at, "post-bundle checklist update")
    lines = body.splitlines()
    _require(
        lines.count(FINAL_CHECKLIST_SECTION) == 1,
        "post-bundle checklist must use one Implementation work section",
    )
    section_start = lines.index(FINAL_CHECKLIST_SECTION) + 1
    section_stop = next(
        (
            ordinal
            for ordinal in range(section_start, len(lines))
            if lines[ordinal].startswith("## ")
        ),
        len(lines),
    )
    section = lines[section_start:section_stop]
    states = {
        "unchecked": FINAL_CHECKLIST_UNCHECKED,
        "checked": FINAL_CHECKLIST_CHECKED,
    }
    matched = []
    for state, required in states.items():
        forbidden = states["checked" if state == "unchecked" else "unchecked"]
        if (
            all(
                lines.count(line) == 1 and section.count(line) == 1 for line in required
            )
            and all(line not in lines for line in forbidden)
            and section.index(required[0]) < section.index(required[1])
        ):
            matched.append(state)
    _require(
        len(matched) == 1 and (expected_state is None or matched[0] == expected_state),
        "post-bundle checklist lines, state, order, or section differ",
    )
    state = matched[0]
    return {
        "state": state,
        "lines": list(states[state]),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "updated_at": updated_at,
    }


def checklist_transition_sha256(issue: Mapping[str, Any], expected_state: str) -> str:
    """Commit to a full issue response with only contract changes neutralized."""

    observation = _validate_post_bundle_checklist(issue, expected_state)
    originals = (
        FINAL_CHECKLIST_UNCHECKED
        if observation["state"] == "unchecked"
        else FINAL_CHECKLIST_CHECKED
    )
    neutral = tuple(
        line.replace("[ ]", "[~]").replace("[x]", "[~]") for line in originals
    )
    replaced = []
    counts = [0, 0]
    for line in issue["body"].splitlines(keepends=True):
        ending = ""
        content = line
        if line.endswith("\r\n"):
            content, ending = line[:-2], "\r\n"
        elif line.endswith("\n") or line.endswith("\r"):
            content, ending = line[:-1], line[-1:]
        for ordinal, original in enumerate(originals):
            if content == original:
                content = neutral[ordinal]
                counts[ordinal] += 1
                break
        replaced.append(content + ending)
    _require(counts == [1, 1], "checklist transition marker closure differs")
    try:
        from benchmarks import issue123_privacy as privacy

        projection = dict(issue)
        projection["body"] = "".join(replaced)
        projection["updated_at"] = CHECKLIST_TRANSITION_UPDATED_AT_SENTINEL
        return privacy.tagged_canonical_sha256(
            CHECKLIST_TRANSITION_DOMAIN,
            projection,
        )
    except ImportError, TypeError, ValueError:
        raise EvidenceError("checklist transition digest is unavailable") from None


def _creation_update_window(
    value: Mapping[str, Any], label: str
) -> tuple[dt.datetime, dt.datetime]:
    created = _timestamp(value.get("created_at"), f"{label} creation")
    updated = _timestamp(value.get("updated_at"), f"{label} update")
    _require(created <= updated, f"{label} creation is after its update")
    return created, updated


def _sha40(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _validate_rest_link(
    url: str,
    relation: str,
    ordinal: int,
    request: Mapping[str, Any],
    label: str,
) -> int:
    canonical = _canonical_api_url(url, label)
    parsed = urlsplit(canonical)
    pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    _require(
        len(pairs) == len({key for key, _value in pairs}),
        f"{label} repeats a query parameter",
    )
    query = dict(pairs)
    endpoint_parts = request["endpoint"].split("/")
    suffix = "/" + "/".join(endpoint_parts[3:])
    named_route = "/" + request["endpoint"]
    numeric_route = re.fullmatch(
        rf"/repositories/[1-9][0-9]*{re.escape(suffix)}", parsed.path
    )
    _require(
        parsed.path == named_route or numeric_route is not None,
        f"{label} endpoint differs",
    )
    _require(
        set(query) == {*request["parameters"], "page"},
        f"{label} query parameter closure differs",
    )
    for key, value in request["parameters"].items():
        _require(query.get(key) == value, f"{label} query differs")
    expected_page = {
        "first": 1,
        "prev": ordinal - 1,
        "next": ordinal + 1,
    }.get(relation)
    try:
        page = int(query["page"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"{label} page differs") from error
    _require(page > 0, f"{label} page differs")
    if expected_page is not None:
        _require(page == expected_page, f"{label} page differs")
    if relation == "last":
        _require(page >= ordinal, f"{label} final page differs")
    return page


def _validate_response_capture(
    value: Any,
    raw_response: Any,
    request: Mapping[str, Any],
    label: str,
) -> None:
    _exact_keys(
        value,
        {
            "canonical_response_sha256",
            "canonical_response_size_bytes",
            "pages",
        },
        label,
    )
    canonical = _canonical_json_bytes(raw_response)
    _require(
        type(value["canonical_response_size_bytes"]) is int
        and value["canonical_response_sha256"] == hashlib.sha256(canonical).hexdigest()
        and value["canonical_response_size_bytes"] == len(canonical),
        f"{label} canonical response bytes differ",
    )
    paginated = request["paginated"]
    graphql = request["graphql"]
    response_pages = raw_response if paginated else [raw_response]
    pages = value["pages"]
    _require(
        isinstance(response_pages, list)
        and isinstance(pages, list)
        and bool(pages)
        and len(pages) == len(response_pages)
        and len(pages) <= MAX_PAGES,
        f"{label} page closure differs",
    )
    declared_last_pages: set[int] = set()
    for index, (page, response_page) in enumerate(
        zip(pages, response_pages, strict=True)
    ):
        ordinal = index + 1
        _exact_keys(
            page,
            {
                "ordinal",
                "status",
                "headers",
                "body_sha256",
                "body_size_bytes",
                "item_count",
                "has_next",
                "next",
            },
            f"{label} page {ordinal}",
        )
        headers = page["headers"]
        _exact_keys(headers, set(SAFE_RESPONSE_HEADERS), f"{label} page headers")
        _require(
            type(page["ordinal"]) is int
            and page["ordinal"] == ordinal
            and type(page["status"]) is int
            and page["status"] == 200
            and isinstance(headers["content-type"], str)
            and headers["content-type"].lower().startswith("application/json")
            and (
                headers["etag"] is None
                or (isinstance(headers["etag"], str) and bool(headers["etag"]))
            )
            and (
                headers["last-modified"] is None
                or (
                    isinstance(headers["last-modified"], str)
                    and bool(headers["last-modified"])
                )
            )
            and (
                (
                    graphql
                    and headers["x-github-api-version-selected"]
                    in {None, GITHUB_API_HEADERS["X-GitHub-Api-Version"]}
                )
                or (
                    not graphql
                    and headers["x-github-api-version-selected"]
                    == GITHUB_API_HEADERS["X-GitHub-Api-Version"]
                )
            )
            and headers["x-github-media-type"]
            == ("github.v4; format=json" if graphql else "github.v3; format=json")
            and isinstance(headers["link"], dict)
            and set(headers["link"]) <= {"first", "prev", "next", "last"},
            f"{label} page {ordinal} metadata differs",
        )
        page_body = _canonical_json_bytes(response_page)
        has_next = index < len(pages) - 1
        _require(
            type(page["body_size_bytes"]) is int
            and (
                page["item_count"] is None
                if _page_item_count(response_page) is None
                else type(page["item_count"]) is int
            )
            and page["body_sha256"] == hashlib.sha256(page_body).hexdigest()
            and page["body_size_bytes"] == len(page_body)
            and page["item_count"] == _page_item_count(response_page)
            and page["has_next"] is has_next,
            f"{label} page {ordinal} body ledger differs",
        )
        if graphql:
            _require(headers["link"] == {}, f"{label} GraphQL Link header differs")
            graphql_next, cursor = _graphql_next(response_page, label)
            expected_next = (
                {"kind": "graphql-cursor", "value": cursor} if graphql_next else None
            )
            _require(
                graphql_next is has_next and page["next"] == expected_next,
                f"{label} GraphQL final-page/no-next relationship differs",
            )
        else:
            for relation, url in headers["link"].items():
                linked_page = _validate_rest_link(
                    url,
                    relation,
                    ordinal,
                    request,
                    f"{label} page {ordinal} {relation} Link",
                )
                if relation == "last":
                    declared_last_pages.add(linked_page)
            link_next = headers["link"].get("next")
            expected_next = (
                {"kind": "rest-link", "value": link_next}
                if link_next is not None
                else None
            )
            _require(
                (link_next is not None) is has_next and page["next"] == expected_next,
                f"{label} REST final-page/no-next relationship differs",
            )
    if not graphql:
        _require(
            not declared_last_pages or declared_last_pages == {len(pages)},
            f"{label} declared final page differs from the captured ledger",
        )


def _list_pages(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list) and bool(value), f"{label} pages differ")
    result = []
    for index, page in enumerate(value):
        _require(isinstance(page, list), f"{label} page {index} differs")
        _require(len(page) <= PAGE_SIZE, f"{label} page {index} is oversized")
        if index < len(value) - 1:
            _require(
                len(page) == PAGE_SIZE,
                f"{label} page {index} terminates pagination early",
            )
        elif len(value) > 1:
            _require(bool(page), f"{label} has a trailing empty page")
        result.extend(page)
    return result


def _object_pages(value: Any, key: str, label: str) -> list[Any]:
    _require(isinstance(value, list) and bool(value), f"{label} pages differ")
    result = []
    total_count = None
    for index, page in enumerate(value):
        _exact_keys(page, {"total_count", key}, f"{label} page {index}")
        _require(
            type(page["total_count"]) is int
            and page["total_count"] >= 0
            and isinstance(page[key], list),
            f"{label} page {index} differs",
        )
        if total_count is None:
            total_count = page["total_count"]
        _require(
            page["total_count"] == total_count,
            f"{label} total count changes between pages",
        )
        _require(
            len(page[key]) <= PAGE_SIZE,
            f"{label} page {index} is oversized",
        )
        if index < len(value) - 1:
            _require(
                len(page[key]) == PAGE_SIZE,
                f"{label} page {index} terminates pagination early",
            )
        elif len(value) > 1:
            _require(bool(page[key]), f"{label} has a trailing empty page")
        result.extend(page[key])
    _require(total_count == len(result), f"{label} total count differs")
    return result


def _validate_request(
    record: Any,
    endpoint: str,
    *,
    parameters: dict[str, str] | None = None,
    paginated: bool = False,
    variables: dict[str, str | int] | None = None,
) -> None:
    _exact_keys(
        record,
        {
            "endpoint",
            "method",
            "headers",
            "parameters",
            "paginated",
            "jq",
            "graphql",
            "query",
            "variables",
        },
        f"operations request {endpoint}",
    )
    graphql = variables is not None
    _require(
        record["endpoint"] == endpoint
        and record["method"] == ("POST" if graphql else "GET")
        and record["headers"] == GITHUB_API_HEADERS
        and record["parameters"] == (parameters or {})
        and record["paginated"] is paginated
        and record["jq"] == ("." if paginated else None)
        and record["graphql"] is graphql
        and record["query"] == (PULL_REQUEST_CONTEXT_QUERY if graphql else None)
        and record["variables"] == (variables or {}),
        f"operations request provenance differs: {endpoint}",
    )


def _validate_issue_identity(
    value: Any,
    number: int,
    state: str,
    label: str,
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} response differs")
    api_root = f"https://api.github.com/repos/{REPOSITORY}"
    _require(
        value.get("number") == number
        and value.get("state") == state
        and value.get("repository_url") == api_root
        and value.get("url") == f"{api_root}/issues/{number}"
        and type(value.get("comments")) is int
        and value["comments"] >= 0
        and isinstance(value.get("body"), str)
        and "pull_request" not in value,
        f"{label} identity differs",
    )
    _creation_update_window(value, label)
    return value


def _validate_issue_comments(
    value: Any,
    issue_number: int,
    expected_count: int,
    label: str,
    *,
    html_kind: str = "issues",
) -> list[dict[str, Any]]:
    comments = _list_pages(value, label)
    _require(len(comments) == expected_count, f"{label} count differs")
    issue_url = f"https://api.github.com/repos/{REPOSITORY}/issues/{issue_number}"
    result = []
    ids = set()
    for comment in comments:
        identifier = comment.get("id") if isinstance(comment, dict) else None
        _require(
            isinstance(comment, dict)
            and type(identifier) is int
            and identifier > 0
            and identifier not in ids
            and comment.get("url")
            == f"https://api.github.com/repos/{REPOSITORY}/issues/comments/{identifier}"
            and comment.get("issue_url") == issue_url
            and comment.get("html_url")
            == f"https://github.com/{REPOSITORY}/{html_kind}/{issue_number}"
            f"#issuecomment-{identifier}"
            and isinstance(comment.get("body"), str)
            and isinstance(comment.get("user"), dict)
            and isinstance(comment["user"].get("login"), str)
            and bool(comment["user"]["login"])
            and isinstance(comment.get("author_association"), str),
            f"{label} record differs",
        )
        _creation_update_window(comment, f"{label} comment {identifier}")
        ids.add(identifier)
        result.append(comment)
    return result


def _parse_owner_contract(
    comments: list[dict[str, Any]],
    marker: str,
    expected_fields: dict[str, str],
    label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    marked = [comment for comment in comments if marker in comment["body"]]
    _require(len(marked) == 1, f"{label} marker closure differs")
    comment = marked[0]
    body = comment["body"]
    lines = body.splitlines()
    parsed: dict[str, str] = {}
    keys = []
    for line in lines[1:]:
        _require(
            line == line.strip() and "=" in line,
            f"{label} contains a malformed structured field",
        )
        key, value = line.split("=", 1)
        _require(
            re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is not None
            and bool(value)
            and key not in parsed,
            f"{label} repeats or malforms a structured field",
        )
        parsed[key] = value
        keys.append(key)
    _require(
        body.count(marker) == 1
        and lines.count(marker) == 1
        and bool(lines)
        and lines[0] == marker
        and len(lines) == len(expected_fields) + 1
        and keys == list(expected_fields)
        and parsed == expected_fields
        and comment["user"].get("login") == OWNER_LOGIN
        and comment.get("author_association") == "OWNER",
        f"{label} is not the exact OWNER-authored structured contract",
    )
    return comment, parsed


def _resolve_superseded_owner_comments(
    streams: Mapping[str, list[dict[str, Any]]],
    final_amendment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    final_created = _timestamp(
        final_amendment.get("created_at"),
        "final issue #123 contract amendment creation",
    )
    resolved = []
    for specification in SUPERSEDED_OWNER_COMMENTS:
        stream = specification["stream"]
        comments = streams.get(stream)
        _require(isinstance(comments, list), "superseded comment stream differs")
        matches = [
            comment for comment in comments if comment.get("id") == specification["id"]
        ]
        _require(
            len(matches) == 1,
            f"superseded OWNER comment {specification['id']} is absent or duplicated",
        )
        comment = matches[0]
        identifier = specification["id"]
        api_url = (
            f"https://api.github.com/repos/{REPOSITORY}/issues/comments/{identifier}"
        )
        issue_url = (
            f"https://api.github.com/repos/{REPOSITORY}/issues/"
            f"{specification['issue_number']}"
        )
        html_url = (
            f"https://github.com/{REPOSITORY}/{specification['html_kind']}/"
            f"{specification['issue_number']}#issuecomment-{identifier}"
        )
        body = comment.get("body")
        _require(
            comment.get("url") == api_url
            and comment.get("issue_url") == issue_url
            and comment.get("html_url") == html_url
            and comment.get("user", {}).get("login") == OWNER_LOGIN
            and comment.get("author_association") == "OWNER"
            and comment.get("created_at") == specification["created_at"]
            and comment.get("updated_at") == specification["updated_at"]
            and isinstance(body, str)
            and hashlib.sha256(body.encode("utf-8")).hexdigest()
            == specification["body_sha256"]
            and all(
                fragment in body for fragment in specification["required_fragments"]
            ),
            f"superseded OWNER comment {identifier} provenance or content differs",
        )
        _require(
            _timestamp(comment["updated_at"], f"superseded comment {identifier} update")
            <= final_created,
            f"superseded OWNER comment {identifier} postdates the final amendment",
        )
        resolved.append(
            {
                "id": identifier,
                "stream": stream,
                "role": specification["role"],
                "api_url": api_url,
                "html_url": html_url,
                "owner_login": OWNER_LOGIN,
                "author_association": "OWNER",
                "created_at": comment["created_at"],
                "updated_at": comment["updated_at"],
                "body_sha256": specification["body_sha256"],
            }
        )
    return resolved


def _technical_release_url(tag: str) -> str:
    return f"https://github.com/{REPOSITORY}/releases/tag/{tag}"


def _technical_asset_url(tag: str, name: str) -> str:
    return f"https://github.com/{REPOSITORY}/releases/download/{tag}/{name}"


def _validate_release_asset(value: Any, release_id: int, tag: str) -> dict[str, Any]:
    _require(isinstance(value, dict), "technical release asset record differs")
    identifier = value.get("id")
    name = value.get("name")
    digest = value.get("digest")
    uploader = value.get("uploader")
    _creation_update_window(value, f"technical release asset {identifier}")
    _require(
        type(identifier) is int
        and identifier > 0
        and isinstance(name, str)
        and name in set(TECHNICAL_RELEASE_ASSETS.values())
        and value.get("url")
        == f"https://api.github.com/repos/{REPOSITORY}/releases/assets/{identifier}"
        and value.get("browser_download_url") == _technical_asset_url(tag, name)
        and value.get("state") == "uploaded"
        and isinstance(uploader, dict)
        and uploader.get("login") == OWNER_LOGIN
        and type(value.get("size")) is int
        and value["size"] > 0
        and isinstance(digest, str)
        and digest.startswith("sha256:")
        and SHA256_RE.fullmatch(digest.removeprefix("sha256:")) is not None,
        "technical release asset identity, size, or digest differs",
    )
    return {
        "id": identifier,
        "name": name,
        "api_url": value["url"],
        "url": value["browser_download_url"],
        "state": value["state"],
        "size_bytes": value["size"],
        "sha256": digest.removeprefix("sha256:"),
        "release_id": release_id,
    }


def _validate_technical_release(
    release: Any,
    tag_ref: Any,
    asset_pages: Any,
    *,
    release_id: int,
    tag: str,
    candidate_sha: str,
) -> tuple[str, dt.datetime, dict[str, dict[str, Any]]]:
    api_root = f"https://api.github.com/repos/{REPOSITORY}"
    _require(isinstance(release, dict), "technical release response differs")
    author = release.get("author")
    release_created_at, release_updated_at = _creation_update_window(
        release, "technical release"
    )
    published_at = _timestamp(
        release.get("published_at"), "technical release publication"
    )
    _require(
        release.get("id") == release_id
        and release.get("url") == f"{api_root}/releases/{release_id}"
        and release.get("assets_url") == f"{api_root}/releases/{release_id}/assets"
        and release.get("upload_url")
        == f"https://uploads.github.com/repos/{REPOSITORY}/releases/"
        f"{release_id}/assets{{?name,label}}"
        and release.get("html_url") == _technical_release_url(tag)
        and release.get("tag_name") == tag
        and release.get("target_commitish") == candidate_sha
        and release.get("draft") is False
        and release.get("prerelease") is False
        and release.get("immutable") is True
        and isinstance(author, dict)
        and author.get("login") == OWNER_LOGIN
        and isinstance(release.get("assets"), list),
        "technical release is unpublished or not FINAL_SHA-bound",
    )
    _require(
        release_created_at <= published_at <= release_updated_at,
        "technical release publication chronology differs",
    )
    _require(
        tag == f"{TECHNICAL_RELEASE_TAG_PREFIX}{candidate_sha}"
        and not tag.startswith("v"),
        "technical release tag is not the exact non-v FINAL_SHA-bound tag",
    )
    _require(isinstance(tag_ref, dict), "technical release tag response differs")
    target = tag_ref.get("object")
    _require(
        tag_ref.get("ref") == f"refs/tags/{tag}"
        and tag_ref.get("url") == f"{api_root}/git/refs/tags/{tag}"
        and isinstance(target, dict)
        and target.get("type") == "commit"
        and target.get("sha") == candidate_sha
        and target.get("url") == f"{api_root}/git/commits/{candidate_sha}",
        "technical release lightweight tag does not target FINAL_SHA",
    )

    assets = _list_pages(asset_pages, "technical release assets")
    _require(
        len(assets) == len(TECHNICAL_RELEASE_ASSETS),
        "technical release asset closure differs",
    )
    validated = [_validate_release_asset(asset, release_id, tag) for asset in assets]
    names = [asset["name"] for asset in validated]
    identifiers = [asset["id"] for asset in validated]
    digests = [asset["sha256"] for asset in validated]
    _require(
        set(names) == set(TECHNICAL_RELEASE_ASSETS.values())
        and len(names) == len(set(names))
        and len(identifiers) == len(set(identifiers))
        and len(digests) == len(set(digests)),
        "technical release assets are missing, duplicated, or substituted",
    )
    embedded = [
        _validate_release_asset(asset, release_id, tag) for asset in release["assets"]
    ]
    _require(
        sorted(embedded, key=lambda asset: asset["name"])
        == sorted(validated, key=lambda asset: asset["name"]),
        "technical release embedded asset ledger differs",
    )
    by_role = {
        role: next(asset for asset in validated if asset["name"] == name)
        for role, name in TECHNICAL_RELEASE_ASSETS.items()
    }
    return release["html_url"], published_at, by_role


def _validate_issue_contract_amendment(
    comments: list[dict[str, Any]],
    candidate_sha: str,
    release_url: str,
) -> dict[str, Any]:
    expected = {
        "FINAL_SHA": candidate_sha,
        "PR": str(PULL_REQUEST_NUMBER),
        "TARGET_ISSUE": str(TARGET_ISSUE_NUMBER),
        "TECHNICAL_RELEASE_URL": release_url,
        "BASELINE_V3_ROOT_COMMIT": BASELINE_V3_ROOT_COMMIT,
        "BASELINE_V3_ONE_URL": BASELINE_V3_ONE_URL,
        "BASELINE_V3_ONE_SIZE_BYTES": str(BASELINE_V3_ONE_SIZE),
        "BASELINE_V3_ONE_SHA256": BASELINE_V3_ONE_SHA256,
        "BASELINE_V3_PHYSICAL_URL": BASELINE_V3_PHYSICAL_URL,
        "BASELINE_V3_PHYSICAL_SIZE_BYTES": str(BASELINE_V3_PHYSICAL_SIZE),
        "BASELINE_V3_PHYSICAL_SHA256": BASELINE_V3_PHYSICAL_SHA256,
        "BASELINE_V3_HOSTNAME": "redacted",
        "BASELINE_V3_HOST_IDENTITY_SCHEMA": "torch-cpu-host-identity-v2",
        "BASELINE_V3_HOST_COMMITMENT_SHA256": BASELINE_V3_HOST_COMMITMENT,
        "BASELINE_V3_DISPOSITION": "authoritative-published-privacy-sanitized",
        **{
            specification["field"]: str(specification["id"])
            for specification in SUPERSEDED_OWNER_COMMENTS
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
        "SINGLE_GPU_3D_LATE_RESIDUAL_CONTRACT": "normalized-linf-l2-at-most-1e-6",
        "SINGLE_GPU_3D_RESIDUAL_DENOMINATOR_FLOOR": "2e-12",
        "SINGLE_GPU_3D_L2_DENOMINATOR_SCALE": "sqrt(N)",
        "SINGLE_GPU_3D_ZERO_REFERENCE_CONTRACT": "exact",
        "PUBLIC_TRACE_DISPOSITION": "published-event-complete-privacy-normalized",
        "CORRECTNESS_ARRAY_DISPOSITION": "private",
        "CORRECTNESS_COMMITMENT_DISPOSITION": "published-in-technical-evidence",
    }
    comment, _fields = _parse_owner_contract(
        comments,
        ISSUE_CONTRACT_AMENDMENT_MARKER,
        expected,
        "final issue #123 contract amendment",
    )
    return comment


def _asset_contract_fields(prefix: str, asset: dict[str, Any]) -> dict[str, str]:
    return {
        f"{prefix}_ASSET_NAME": asset["name"],
        f"{prefix}_ASSET_URL": asset["url"],
        f"{prefix}_ASSET_SIZE_BYTES": str(asset["size_bytes"]),
        f"{prefix}_ASSET_SHA256": asset["sha256"],
    }


def _validate_pr_candidate_insight(
    comments: list[dict[str, Any]],
    *,
    candidate_sha: str,
    commit_url: str,
    ci_run_url: str,
    codeql_run_url: str,
    release_url: str,
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = {
        "FINAL_SHA": candidate_sha,
        "PR": str(PULL_REQUEST_NUMBER),
        "TARGET_ISSUE": str(TARGET_ISSUE_NUMBER),
        "FINAL_COMMIT_URL": commit_url,
        "FINAL_COMMIT_VERIFICATION": "verified:valid",
        "CI_RUN_URL": ci_run_url,
        "CODEQL_RUN_URL": codeql_run_url,
        "TEST_SUMMARY": "required-ci-and-regression-tests-pass",
        "EVIDENCE_SUMMARY": (
            "five-technical-scopes-pass-private-arrays-commitment-published"
        ),
        "TECHNICAL_RELEASE_URL": release_url,
        **_asset_contract_fields("TECHNICAL_EVIDENCE", assets["technical_evidence"]),
        **_asset_contract_fields("TECHNICAL_SUMMARY", assets["technical_summary"]),
    }
    comment, _fields = _parse_owner_contract(
        comments,
        PR_CANDIDATE_INSIGHT_MARKER,
        expected,
        "final PR #167 candidate insight",
    )
    return comment


def _validate_handoff_contract(
    comments: list[dict[str, Any]],
    *,
    candidate_sha: str,
    release_url: str,
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = {
        "FINAL_SHA": candidate_sha,
        "PR": str(PULL_REQUEST_NUMBER),
        "TARGET_ISSUE": str(TARGET_ISSUE_NUMBER),
        "HANDOFF_ISSUE": str(HANDOFF_ISSUE_NUMBER),
        "TECHNICAL_RELEASE_URL": release_url,
        "RAW_TIMING_CONTRACT": "torch-utils-benchmark-fixed-workloads",
        **_asset_contract_fields("RAW_TIMING", assets["raw_timing"]),
        "EVENT_PROFILER_CONTRACT": "event-level-profiler-fixed-workloads",
        **_asset_contract_fields("EVENT_PROFILER", assets["event_profiler"]),
        "HANDOFF_DISPOSITION": "complete",
    }
    comment, _fields = _parse_owner_contract(
        comments,
        HANDOFF_MARKER,
        expected,
        "final issue #115 runtime handoff",
    )
    _require(
        assets["raw_timing"]["url"] != assets["event_profiler"]["url"]
        and assets["raw_timing"]["sha256"] != assets["event_profiler"]["sha256"],
        "issue #115 timing and profiler assets are not distinct",
    )
    return comment


def _validate_candidate_commit(value: Any, candidate_sha: str) -> dict[str, Any]:
    _require(isinstance(value, dict), "candidate commit response differs")
    commit = value.get("commit")
    verification = commit.get("verification") if isinstance(commit, dict) else None
    api_url = f"https://api.github.com/repos/{REPOSITORY}/commits/{candidate_sha}"
    _require(
        value.get("sha") == candidate_sha
        and value.get("url") == api_url
        and value.get("html_url")
        == f"https://github.com/{REPOSITORY}/commit/{candidate_sha}"
        and isinstance(verification, dict)
        and verification.get("verified") is True
        and verification.get("reason") == "valid",
        "candidate commit is unverified, invalid, or not FINAL_SHA-bound",
    )
    _timestamp(verification.get("verified_at"), "candidate commit verification")
    return {
        "sha": candidate_sha,
        "verified": True,
        "reason": "valid",
        "verified_at": verification["verified_at"],
    }


def _validate_pull_request_context(
    value: Any,
    candidate_sha: str,
) -> tuple[int, int]:
    _require(
        isinstance(value, list) and bool(value),
        "pull-request GraphQL pages differ",
    )
    thread_total = None
    review_total = None
    thread_ids = set()
    thread_cursors = set()
    for page_index, page in enumerate(value):
        _exact_keys(page, {"data"}, f"pull-request GraphQL page {page_index}")
        data = page["data"]
        _exact_keys(data, {"repository"}, f"pull-request GraphQL data {page_index}")
        repository = data["repository"]
        _exact_keys(
            repository,
            {"nameWithOwner", "pullRequest"},
            f"pull-request GraphQL repository {page_index}",
        )
        pull = repository["pullRequest"]
        _exact_keys(
            pull,
            {
                "number",
                "headRefOid",
                "closingIssuesReferences",
                "reviews",
                "reviewThreads",
            },
            f"pull-request GraphQL pull request {page_index}",
        )
        _require(
            repository["nameWithOwner"] == REPOSITORY
            and pull["number"] == PULL_REQUEST_NUMBER
            and pull["headRefOid"] == candidate_sha,
            "pull-request GraphQL identity differs",
        )

        closing = pull["closingIssuesReferences"]
        _exact_keys(
            closing,
            {"totalCount", "nodes"},
            f"closing-issue connection {page_index}",
        )
        nodes = closing["nodes"]
        _require(
            type(closing["totalCount"]) is int
            and closing["totalCount"] == 1
            and isinstance(nodes, list)
            and len(nodes) == 1,
            "pull request must close exactly one issue",
        )
        node = nodes[0]
        _exact_keys(node, {"number", "repository"}, "closing-issue node")
        _exact_keys(
            node["repository"],
            {"nameWithOwner"},
            "closing-issue repository",
        )
        _require(
            node["number"] == TARGET_ISSUE_NUMBER
            and node["repository"]["nameWithOwner"] == REPOSITORY,
            f"PR #{PULL_REQUEST_NUMBER} does not close exactly target issue "
            f"#{TARGET_ISSUE_NUMBER}",
        )

        reviews = pull["reviews"]
        _exact_keys(reviews, {"totalCount"}, f"review connection {page_index}")
        _require(
            type(reviews["totalCount"]) is int and reviews["totalCount"] >= 0,
            f"review connection {page_index} differs",
        )
        if review_total is None:
            review_total = reviews["totalCount"]
        _require(
            reviews["totalCount"] == review_total,
            "review total count changes between GraphQL pages",
        )

        threads = pull["reviewThreads"]
        _exact_keys(
            threads,
            {"totalCount", "pageInfo", "nodes"},
            f"review-thread connection {page_index}",
        )
        page_info = threads["pageInfo"]
        _exact_keys(
            page_info,
            {"hasNextPage", "endCursor"},
            f"review-thread page info {page_index}",
        )
        nodes = threads["nodes"]
        _require(
            type(threads["totalCount"]) is int
            and threads["totalCount"] >= 0
            and isinstance(nodes, list)
            and len(nodes) <= PAGE_SIZE,
            f"review-thread page {page_index} differs",
        )
        if thread_total is None:
            thread_total = threads["totalCount"]
        _require(
            threads["totalCount"] == thread_total,
            "review-thread total count changes between pages",
        )
        has_next = page_index < len(value) - 1
        _require(
            page_info["hasNextPage"] is has_next,
            "review-thread pagination closure differs",
        )
        _require(
            (
                bool(nodes)
                and isinstance(page_info["endCursor"], str)
                and bool(page_info["endCursor"])
            )
            or (not nodes and page_info["endCursor"] is None),
            "review-thread page cursor differs",
        )
        if page_info["endCursor"] is not None:
            _require(
                page_info["endCursor"] not in thread_cursors,
                "review-thread page cursor repeats",
            )
            thread_cursors.add(page_info["endCursor"])
        if has_next:
            _require(
                len(nodes) == PAGE_SIZE
                and isinstance(page_info["endCursor"], str)
                and bool(page_info["endCursor"]),
                "review-thread pagination ends early",
            )
        elif len(value) > 1:
            _require(bool(nodes), "review threads have a trailing empty page")
        for thread in nodes:
            _exact_keys(thread, {"id", "isResolved"}, "review-thread node")
            _require(
                isinstance(thread["id"], str)
                and bool(thread["id"])
                and thread["id"] not in thread_ids
                and thread["isResolved"] is True,
                "pull-request review conversations are unresolved or duplicated",
            )
            thread_ids.add(thread["id"])
    _require(thread_total == len(thread_ids), "review-thread total count differs")
    assert review_total is not None
    return len(thread_ids), review_total


def _validate_workflow_run(
    value: Any,
    *,
    expected_id: int,
    expected_name: str,
    candidate_sha: str,
    base_sha: str,
    label: str,
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} response differs")
    repository = value.get("repository")
    head_repository = value.get("head_repository")
    pull_requests = value.get("pull_requests")
    pull = (
        pull_requests[0] if isinstance(pull_requests, list) and pull_requests else None
    )
    pull_base = pull.get("base") if isinstance(pull, dict) else None
    pull_head = pull.get("head") if isinstance(pull, dict) else None
    api_root = f"https://api.github.com/repos/{REPOSITORY}"
    created_at, updated_at = _creation_update_window(value, label)
    _require(
        type(value.get("id")) is int
        and value["id"] == expected_id
        and value.get("url") == f"{api_root}/actions/runs/{expected_id}"
        and value.get("html_url")
        == f"https://github.com/{REPOSITORY}/actions/runs/{expected_id}"
        and isinstance(repository, dict)
        and repository.get("full_name") == REPOSITORY
        and isinstance(head_repository, dict)
        and head_repository.get("full_name") == REPOSITORY
        and value.get("name") == expected_name
        and value.get("event") == "pull_request"
        and value.get("head_sha") == candidate_sha
        and isinstance(pull_requests, list)
        and len(pull_requests) == 1
        and isinstance(pull, dict)
        and pull.get("number") == PULL_REQUEST_NUMBER
        and pull.get("url") == f"{api_root}/pulls/{PULL_REQUEST_NUMBER}"
        and isinstance(pull_base, dict)
        and pull_base.get("ref") == "master"
        and pull_base.get("sha") == base_sha
        and isinstance(pull_base.get("repo"), dict)
        and pull_base["repo"].get("url") == api_root
        and isinstance(pull_head, dict)
        and pull_head.get("sha") == candidate_sha
        and isinstance(pull_head.get("repo"), dict)
        and pull_head["repo"].get("url") == api_root
        and value.get("status") == "completed"
        and value.get("conclusion") == "success"
        and type(value.get("run_attempt")) is int
        and value["run_attempt"] > 0,
        f"{label} is incomplete, failing, or not candidate-bound",
    )
    _require(created_at <= updated_at, f"{label} chronology differs")
    return {
        "id": value["id"],
        "run_attempt": value["run_attempt"],
        "created_at": value["created_at"],
        "completed_at": value["updated_at"],
    }


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
            and type(item.get("run_id")) is int
            and item["run_id"] == run_id
            and type(item.get("run_attempt")) is int
            and item["run_attempt"] == run_attempt
            and item.get("status") == "completed"
            and item.get("conclusion") == "success"
            and started <= completed,
            f"{label} {name!r} is incomplete or failing",
        )
        matches[name] = item
    return matches


def _expected_publication_bindings(
    candidate: Mapping[str, str],
    ci_run: Mapping[str, Any],
    codeql_run: Mapping[str, Any],
    ci_jobs: Mapping[str, Mapping[str, Any]],
    codeql_jobs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    jobs = []
    for name in PUBLICATION_JOB_NAMES:
        run = ci_run if name in REQUIRED_STATUS_CONTEXTS else codeql_run
        selected = ci_jobs if name in REQUIRED_STATUS_CONTEXTS else codeql_jobs
        job = selected[name]
        jobs.append(
            {
                "name": name,
                "run_id": run["id"],
                "run_attempt": run["run_attempt"],
                "job_id": job["id"],
            }
        )
    return {
        "final_sha": candidate["candidate_git_commit"],
        "manifest_sha256": candidate["manifest_sha256"],
        "jobs": jobs,
    }


def _publication_release_identity_anchor(
    *,
    candidate_sha: str,
    technical_release_id: int,
    technical_release_tag: str,
    technical_release_url: str,
    release_assets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the publication identity anchor only from validated GitHub data."""

    api_root = f"https://api.github.com/repos/{REPOSITORY}"
    return {
        "schema_version": 1,
        "kind": PUBLICATION_RELEASE_CAPTURE_KIND,
        "repository": REPOSITORY,
        "release_id": technical_release_id,
        "tag_name": technical_release_tag,
        "target_commitish": candidate_sha,
        "api_url": f"{api_root}/releases/{technical_release_id}",
        "html_url": technical_release_url,
        "immutable": True,
        "draft": False,
        "prerelease": False,
        "tag_ref": {
            "ref": f"refs/tags/{technical_release_tag}",
            "api_url": f"{api_root}/git/refs/tags/{technical_release_tag}",
            "object_type": "commit",
            "object_sha": candidate_sha,
            "object_url": f"{api_root}/git/commits/{candidate_sha}",
        },
        "assets": [
            {
                "role": role,
                "asset_id": release_assets[role]["id"],
                "release_id": technical_release_id,
                "name": release_assets[role]["name"],
                "api_url": release_assets[role]["api_url"],
                "browser_download_url": release_assets[role]["url"],
                "state": release_assets[role]["state"],
                "size_bytes": release_assets[role]["size_bytes"],
                "sha256": release_assets[role]["sha256"],
            }
            for role in TECHNICAL_RELEASE_ASSETS
        ],
    }


def _validate_publication_receipt_adapter(
    envelope: Any,
    *,
    expected_candidate: Mapping[str, str],
    technical_release_id: int,
    technical_release_tag: str,
    technical_release_url: str,
    release_assets: Mapping[str, Mapping[str, Any]],
    ci_run: Mapping[str, Any],
    codeql_run: Mapping[str, Any],
    ci_jobs: Mapping[str, Mapping[str, Any]],
    codeql_jobs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-bind the moving publication-v1 interface at one narrow boundary."""

    _exact_keys(
        envelope,
        {"media_type", "size_bytes", "sha256", "document"},
        "publication receipt envelope",
    )
    receipt = envelope["document"]
    _require(isinstance(receipt, dict), "publication receipt differs")
    raw = _canonical_json_bytes(receipt)
    _require(
        envelope["media_type"] == MEDIA_TYPE_JSON
        and type(envelope["size_bytes"]) is int
        and 0 < envelope["size_bytes"] <= MAX_PUBLICATION_RECEIPT_BYTES
        and envelope["size_bytes"] == len(raw)
        and envelope["sha256"] == hashlib.sha256(raw).hexdigest(),
        "publication receipt canonical bytes differ",
    )
    _exact_keys(
        receipt,
        {
            "schema_version",
            "kind",
            "bindings",
            "asset_order",
            "asset_ledger",
            "release_capture",
            "execution_witness",
            "execution_witness_member",
            "hashes",
        },
        "publication receipt",
    )
    expected_bindings = _expected_publication_bindings(
        expected_candidate, ci_run, codeql_run, ci_jobs, codeql_jobs
    )
    _require(
        type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["kind"] == PUBLICATION_RECEIPT_KIND
        and _typed_json_equal(receipt["bindings"], expected_bindings),
        "publication receipt FINAL_SHA, manifest, or workflow binding differs",
    )
    asset_order = list(TECHNICAL_RELEASE_ASSETS)
    expected_ledger = [
        {
            "role": role,
            "name": release_assets[role]["name"],
            "size_bytes": release_assets[role]["size_bytes"],
            "sha256": release_assets[role]["sha256"],
        }
        for role in asset_order
    ]
    _require(
        _typed_json_equal(receipt["asset_order"], asset_order)
        and _typed_json_equal(receipt["asset_ledger"], expected_ledger),
        "publication receipt downloaded-byte asset ledger differs",
    )
    release_capture = receipt["release_capture"]
    expected_release_capture = _publication_release_identity_anchor(
        candidate_sha=expected_candidate["candidate_git_commit"],
        technical_release_id=technical_release_id,
        technical_release_tag=technical_release_tag,
        technical_release_url=technical_release_url,
        release_assets=release_assets,
    )
    _require(
        _typed_json_equal(release_capture, expected_release_capture),
        "publication receipt release, tag, or asset identity differs",
    )
    witness = receipt["execution_witness"]
    _exact_keys(
        witness,
        {"schema_version", "kind", "bindings", "claims"},
        "publication execution witness",
    )
    _require(
        type(witness["schema_version"]) is int
        and witness["schema_version"] == 1
        and witness["kind"] == PUBLICATION_EXECUTION_WITNESS_KIND
        and _typed_json_equal(witness["bindings"], expected_bindings)
        and isinstance(witness["claims"], list)
        and len(witness["claims"]) == len(PUBLICATION_EXECUTION_CLAIMS),
        "publication execution witness identity or closure differs",
    )
    for index, ((claim_name, scope), claim) in enumerate(
        zip(PUBLICATION_EXECUTION_CLAIMS, witness["claims"], strict=True)
    ):
        _exact_keys(
            claim,
            {
                "claim",
                "scope",
                "trace_name",
                "validation_workflow",
                "validator_job",
                "event_count",
                "semantic_inventory_sha256",
                "normalized_trace_sha256",
            },
            f"publication execution witness claim {index}",
        )
        _require(
            claim["claim"] == claim_name
            and claim["scope"] == scope
            and isinstance(claim["trace_name"], str)
            and bool(claim["trace_name"])
            and claim["validation_workflow"] == "CI"
            and _typed_json_equal(claim["validator_job"], expected_bindings["jobs"][0])
            and type(claim["event_count"]) is int
            and claim["event_count"] > 0
            and isinstance(claim["semantic_inventory_sha256"], str)
            and SHA256_RE.fullmatch(claim["semantic_inventory_sha256"]) is not None
            and isinstance(claim["normalized_trace_sha256"], str)
            and SHA256_RE.fullmatch(claim["normalized_trace_sha256"]) is not None,
            f"publication execution witness claim {claim_name} differs",
        )
    witness_raw = _canonical_json_bytes(witness)
    expected_member = {
        "path": PUBLICATION_EXECUTION_WITNESS_PATH,
        "media_type": MEDIA_TYPE_JSON,
        "size_bytes": len(witness_raw),
        "sha256": hashlib.sha256(witness_raw).hexdigest(),
    }
    _require(
        _typed_json_equal(receipt["execution_witness_member"], expected_member),
        "publication execution witness member binding differs",
    )
    hashes = receipt["hashes"]
    _exact_keys(
        hashes,
        {
            "release_capture_sha256",
            "execution_witness_member_sha256",
            "trusted_policy_sha256",
            "asset_ledger_sha256",
        },
        "publication receipt hashes",
    )
    _require(
        hashes["release_capture_sha256"]
        == hashlib.sha256(_canonical_json_bytes(release_capture)).hexdigest()
        and hashes["execution_witness_member_sha256"] == expected_member["sha256"]
        and isinstance(hashes["trusted_policy_sha256"], str)
        and SHA256_RE.fullmatch(hashes["trusted_policy_sha256"]) is not None
        and hashes["asset_ledger_sha256"]
        == hashlib.sha256(_canonical_json_bytes(expected_ledger)).hexdigest(),
        "publication receipt internal hash binding differs",
    )
    return {
        "receipt_sha256": envelope["sha256"],
        "receipt_size_bytes": envelope["size_bytes"],
        "bindings": expected_bindings,
        "release_id": technical_release_id,
        "release_tag": technical_release_tag,
        "asset_ids": {
            role: release_assets[role]["id"] for role in TECHNICAL_RELEASE_ASSETS
        },
        "asset_ledger": expected_ledger,
        "release_identity_anchor": expected_release_capture,
        "execution_witness_member": expected_member,
        "execution_claims": [
            {
                "claim": claim["claim"],
                "scope": claim["scope"],
                "event_count": claim["event_count"],
                "semantic_inventory_sha256": claim["semantic_inventory_sha256"],
                "normalized_trace_sha256": claim["normalized_trace_sha256"],
            }
            for claim in witness["claims"]
        ],
        "event_profiler_asset": expected_ledger[-1],
        "trusted_policy_sha256": hashes["trusted_policy_sha256"],
        "offline_final_acceptance": False,
    }


def _validate_ruleset(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "master ruleset response differs")
    _creation_update_window(value, "master ruleset")
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
            "target_issue_number",
            "handoff_issue_number",
            "pull_request_number",
            "ci_run_id",
            "codeql_run_id",
            "technical_release_tag",
            "technical_release_id",
            "publication_receipt",
            "responses",
            "response_captures",
        },
        "operations index",
    )
    _require(
        type(document["schema_version"]) is int
        and document["schema_version"] == 2
        and document["kind"] == INDEX_KIND
        and document["candidate_evidence"] == expected_candidate
        and document["repository"] == REPOSITORY
        and type(document["target_issue_number"]) is int
        and document["target_issue_number"] == TARGET_ISSUE_NUMBER
        and type(document["handoff_issue_number"]) is int
        and document["handoff_issue_number"] == HANDOFF_ISSUE_NUMBER
        and type(document["pull_request_number"]) is int
        and document["pull_request_number"] == PULL_REQUEST_NUMBER
        and type(document["ci_run_id"]) is int
        and document["ci_run_id"] > 0
        and type(document["codeql_run_id"]) is int
        and document["codeql_run_id"] > 0
        and document["ci_run_id"] != document["codeql_run_id"]
        and isinstance(document["technical_release_tag"], str)
        and type(document["technical_release_id"]) is int
        and document["technical_release_id"] > 0,
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
    response_captures = document["response_captures"]
    _require(
        isinstance(response_captures, dict)
        and set(response_captures) == RESPONSE_ROLES,
        "operations response-capture closure differs",
    )

    repository = document["repository"]
    number = document["pull_request_number"]
    candidate_sha = expected_candidate.get("candidate_git_commit")
    _require(
        _sha40(candidate_sha) and candidate_sha != REJECTED_CANDIDATE_SHA,
        "FINAL_SHA is malformed or is the rejected candidate",
    )
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
    technical_release_tag = document["technical_release_tag"]
    technical_release_id = document["technical_release_id"]
    _require(
        pull.get("number") == number
        and pull.get("url")
        == f"https://api.github.com/repos/{repository}/pulls/{number}"
        and pull.get("issue_url")
        == f"https://api.github.com/repos/{repository}/issues/{number}"
        and type(pull.get("comments")) is int
        and pull["comments"] >= 0
        and pull.get("state") == "open"
        and pull.get("draft") is False
        and pull.get("mergeable") is True
        and pull.get("mergeable_state") == "clean"
        and base.get("ref") == "master"
        and isinstance(base.get("repo"), dict)
        and base["repo"].get("full_name") == repository
        and _sha40(base_sha)
        and head_sha == candidate_sha
        and _sha40(merge_sha)
        and isinstance(head.get("repo"), dict)
        and head["repo"].get("full_name") == repository,
        "pull request is not clean, mergeable, master-targeted, or candidate-bound",
    )
    _creation_update_window(pull, "pull request")
    expected_requests = {
        "technical_release": (
            f"repos/{repository}/releases/tags/{technical_release_tag}",
            {},
            False,
        ),
        "technical_release_assets": (
            f"repos/{repository}/releases/{technical_release_id}/assets",
            {"per_page": str(PAGE_SIZE)},
            True,
        ),
        "technical_release_tag": (
            f"repos/{repository}/git/ref/tags/{technical_release_tag}",
            {},
            False,
        ),
        "issue_123": (
            f"repos/{repository}/issues/{TARGET_ISSUE_NUMBER}",
            {},
            False,
        ),
        "issue_123_comments": (
            f"repos/{repository}/issues/{TARGET_ISSUE_NUMBER}/comments",
            {"per_page": str(PAGE_SIZE)},
            True,
        ),
        "issue_115": (
            f"repos/{repository}/issues/{HANDOFF_ISSUE_NUMBER}",
            {},
            False,
        ),
        "issue_115_comments": (
            f"repos/{repository}/issues/{HANDOFF_ISSUE_NUMBER}/comments",
            {"per_page": str(PAGE_SIZE)},
            True,
        ),
        "pull_request": (f"repos/{repository}/pulls/{number}", {}, False),
        "pull_request_comments": (
            f"repos/{repository}/issues/{number}/comments",
            {"per_page": str(PAGE_SIZE)},
            True,
        ),
        "candidate_commit": (
            f"repos/{repository}/commits/{candidate_sha}",
            {},
            False,
        ),
        "base_compare": (
            f"repos/{repository}/compare/{base_sha}...{head_sha}",
            {},
            False,
        ),
        "ci_run": (
            f"repos/{repository}/actions/runs/{document['ci_run_id']}",
            {},
            False,
        ),
        "ci_jobs": (
            f"repos/{repository}/actions/runs/{document['ci_run_id']}/jobs",
            {"per_page": str(PAGE_SIZE)},
            True,
        ),
        "codeql_run": (
            f"repos/{repository}/actions/runs/{document['codeql_run_id']}",
            {},
            False,
        ),
        "codeql_jobs": (
            f"repos/{repository}/actions/runs/{document['codeql_run_id']}/jobs",
            {"per_page": str(PAGE_SIZE)},
            True,
        ),
        "codeql_analyses": (
            f"repos/{repository}/code-scanning/analyses",
            {"per_page": str(PAGE_SIZE), "ref": merge_ref},
            True,
        ),
        "codeql_alerts": (
            f"repos/{repository}/code-scanning/alerts",
            {
                "per_page": str(PAGE_SIZE),
                "ref": merge_ref,
                "state": "open",
            },
            True,
        ),
        "ruleset": (f"repos/{repository}/rulesets/{RULESET_ID}", {}, False),
        "check_runs": (
            f"repos/{repository}/commits/{head_sha}/check-runs",
            {"per_page": str(PAGE_SIZE)},
            True,
        ),
        "reviews": (
            f"repos/{repository}/pulls/{number}/reviews",
            {"per_page": str(PAGE_SIZE)},
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
        paginated=True,
        variables={"owner": owner, "name": name, "number": number},
    )
    for role in RESPONSE_ROLES:
        _validate_response_capture(
            response_captures[role],
            raw_responses[role],
            records[role]["request"],
            f"operations response capture {role}",
        )

    commit_verification = _validate_candidate_commit(
        raw_responses["candidate_commit"], candidate_sha
    )
    release_url, release_published_at, release_assets = _validate_technical_release(
        raw_responses["technical_release"],
        raw_responses["technical_release_tag"],
        raw_responses["technical_release_assets"],
        release_id=technical_release_id,
        tag=technical_release_tag,
        candidate_sha=candidate_sha,
    )

    target_issue = _validate_issue_identity(
        raw_responses["issue_123"],
        TARGET_ISSUE_NUMBER,
        "open",
        f"issue #{TARGET_ISSUE_NUMBER}",
    )
    _require(
        target_issue.get("closed_at") is None,
        f"target issue #{TARGET_ISSUE_NUMBER} is unexpectedly closed",
    )
    target_checklist = _validate_post_bundle_checklist(target_issue)
    target_comments = _validate_issue_comments(
        raw_responses["issue_123_comments"],
        TARGET_ISSUE_NUMBER,
        target_issue["comments"],
        f"issue #{TARGET_ISSUE_NUMBER} comments",
    )
    issue_contract = _validate_issue_contract_amendment(
        target_comments, candidate_sha, release_url
    )
    _require(
        release_published_at
        <= _timestamp(issue_contract["created_at"], "issue contract creation"),
        "issue #123 contract amendment predates the technical release",
    )

    issue = _validate_issue_identity(
        raw_responses["issue_115"],
        HANDOFF_ISSUE_NUMBER,
        "closed",
        f"issue #{HANDOFF_ISSUE_NUMBER}",
    )
    body = issue["body"]
    _require(
        isinstance(issue.get("closed_at"), str)
        and issue.get("state_reason") == "completed"
        and "## Implementation work" in body
        and "- [ ]" not in body,
        "issue #115 is not completed with its runtime checklist complete",
    )
    _require(
        all(body.splitlines().count(item) == 1 for item in HANDOFF_CHECKLIST_ITEMS),
        "issue #115 required runtime checklist items are absent or duplicated",
    )
    issue_closed_at = _timestamp(issue["closed_at"], "issue #115 closure")
    comments = _validate_issue_comments(
        raw_responses["issue_115_comments"],
        HANDOFF_ISSUE_NUMBER,
        issue["comments"],
        f"issue #{HANDOFF_ISSUE_NUMBER} comments",
    )
    handoff = _validate_handoff_contract(
        comments,
        candidate_sha=candidate_sha,
        release_url=release_url,
        assets=release_assets,
    )
    handoff_created_at = _timestamp(
        handoff["created_at"], "issue #115 handoff creation"
    )
    issue_created_at, issue_updated_at = _creation_update_window(issue, "issue #115")
    _require(
        issue_created_at <= issue_closed_at <= issue_updated_at <= handoff_created_at
        and release_published_at <= handoff_created_at,
        "issue #115 closure/checklist completion does not precede its handoff",
    )

    pull_comments = _validate_issue_comments(
        raw_responses["pull_request_comments"],
        PULL_REQUEST_NUMBER,
        pull["comments"],
        f"PR #{PULL_REQUEST_NUMBER} issue comments",
        html_kind="pull",
    )
    all_contract_comments = [*target_comments, *comments, *pull_comments]
    _require(
        len({comment["id"] for comment in all_contract_comments})
        == len(all_contract_comments)
        and len({comment["url"] for comment in all_contract_comments})
        == len(all_contract_comments),
        "issue and PR comment streams overlap",
    )
    superseded_owner_comments = _resolve_superseded_owner_comments(
        {
            "issue_123_comments": target_comments,
            "pull_request_comments": pull_comments,
        },
        issue_contract,
    )

    comparison = raw_responses["base_compare"]
    commits = comparison.get("commits") if isinstance(comparison, dict) else None
    _require(
        isinstance(comparison, dict)
        and comparison.get("url")
        == f"https://api.github.com/repos/{repository}/compare/{base_sha}...{head_sha}"
        and comparison.get("behind_by") == 0
        and type(comparison.get("ahead_by")) is int
        and comparison["ahead_by"] >= 0
        and type(comparison.get("total_commits")) is int
        and comparison["total_commits"] == comparison["ahead_by"]
        and isinstance(comparison.get("base_commit"), dict)
        and comparison["base_commit"].get("sha") == base_sha
        and isinstance(comparison.get("merge_base_commit"), dict)
        and comparison["merge_base_commit"].get("sha") == base_sha
        and isinstance(commits, list)
        and len(commits) == comparison["ahead_by"]
        and all(
            isinstance(commit, dict) and _sha40(commit.get("sha")) for commit in commits
        )
        and len({commit["sha"] for commit in commits}) == len(commits),
        "candidate branch is not up to date with the exact PR base",
    )
    _require(
        (
            comparison.get("status") == "identical"
            and comparison["ahead_by"] == 0
            and head_sha == base_sha
            and commits == []
        )
        or (
            comparison.get("status") == "ahead"
            and comparison["ahead_by"] > 0
            and commits[-1].get("sha") == head_sha
        ),
        "compare response does not terminate at the candidate",
    )

    ci_run = _validate_workflow_run(
        raw_responses["ci_run"],
        expected_id=document["ci_run_id"],
        expected_name="CI",
        candidate_sha=candidate_sha,
        base_sha=base_sha,
        label="CI run",
    )
    codeql_run = _validate_workflow_run(
        raw_responses["codeql_run"],
        expected_id=document["codeql_run_id"],
        expected_name="CodeQL",
        candidate_sha=candidate_sha,
        base_sha=base_sha,
        label="CodeQL run",
    )
    ci_id = ci_run["id"]
    ci_attempt = ci_run["run_attempt"]
    codeql_id = codeql_run["id"]
    codeql_attempt = codeql_run["run_attempt"]
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
    candidate_insight = _validate_pr_candidate_insight(
        pull_comments,
        candidate_sha=candidate_sha,
        commit_url=raw_responses["candidate_commit"]["html_url"],
        ci_run_url=raw_responses["ci_run"]["html_url"],
        codeql_run_url=raw_responses["codeql_run"]["html_url"],
        release_url=release_url,
        assets=release_assets,
    )
    insight_created_at = _timestamp(
        candidate_insight["created_at"], "PR insight creation"
    )
    prerequisite_times = [
        release_published_at,
        _timestamp(commit_verification["verified_at"], "candidate commit verification"),
        _timestamp(ci_run["completed_at"], "CI run completion"),
        _timestamp(codeql_run["completed_at"], "CodeQL run completion"),
        *(
            _timestamp(job["completed_at"], f"CI job {name} completion")
            for name, job in ci_jobs.items()
        ),
        *(
            _timestamp(job["completed_at"], f"CodeQL job {name} completion")
            for name, job in codeql_jobs.items()
        ),
    ]
    _require(
        all(completed_at <= insight_created_at for completed_at in prerequisite_times),
        "PR #167 insight predates commit verification or selected CI/CodeQL completion",
    )
    publication = _validate_publication_receipt_adapter(
        document["publication_receipt"],
        expected_candidate=expected_candidate,
        technical_release_id=technical_release_id,
        technical_release_tag=technical_release_tag,
        technical_release_url=release_url,
        release_assets=release_assets,
        ci_run=ci_run,
        codeql_run=codeql_run,
        ci_jobs=ci_jobs,
        codeql_jobs=codeql_jobs,
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
        _creation_update_window(alert, "CodeQL alert")
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
    latest_decisive_reviews = {}
    review_ids = set()
    for review in reviews:
        _require(
            isinstance(review, dict)
            and type(review.get("id")) is int
            and review["id"] > 0
            and review["id"] not in review_ids
            and review.get("pull_request_url")
            == f"https://api.github.com/repos/{repository}/pulls/{number}"
            and review.get("state")
            in {"APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED"}
            and isinstance(review.get("user"), dict),
            "review record differs",
        )
        review_ids.add(review["id"])
        login = review["user"].get("login")
        _require(isinstance(login, str) and bool(login), "review author differs")
        key = (
            _timestamp(review.get("submitted_at"), f"review by {login}"),
            review["id"],
        )
        if login not in latest_reviews or key > latest_reviews[login][0]:
            latest_reviews[login] = (key, review)
        if review["state"] != "COMMENTED" and (
            login not in latest_decisive_reviews
            or key > latest_decisive_reviews[login][0]
        ):
            latest_decisive_reviews[login] = (key, review)
    _require(
        all(
            record[1].get("state") != "CHANGES_REQUESTED"
            for record in latest_decisive_reviews.values()
        ),
        "a latest pull-request review still requests changes",
    )
    review_thread_count, graphql_review_total = _validate_pull_request_context(
        raw_responses["review_threads"], candidate_sha
    )
    _require(
        graphql_review_total == len(review_ids),
        "REST review total differs from the GraphQL review total",
    )

    macos_job = ci_jobs["Python 3.14 / macos-latest"]
    return {
        "candidate_evidence": expected_candidate,
        "repository": repository,
        "target_issue_number": TARGET_ISSUE_NUMBER,
        "handoff_issue_number": HANDOFF_ISSUE_NUMBER,
        "pull_request": {
            "number": number,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "merge_sha": merge_sha,
            "merge_ref": merge_ref,
        },
        "candidate_commit_verification": commit_verification,
        "technical_release": {
            "id": technical_release_id,
            "tag": technical_release_tag,
            "url": release_url,
            "assets": {
                role: {
                    "name": asset["name"],
                    "url": asset["url"],
                    "size_bytes": asset["size_bytes"],
                    "sha256": asset["sha256"],
                }
                for role, asset in release_assets.items()
            },
        },
        "closing_issue_numbers": [TARGET_ISSUE_NUMBER],
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
        "graphql_review_total": graphql_review_total,
        "review_threads": review_thread_count,
        "publication": publication,
        "superseded_owner_comments": superseded_owner_comments,
        "issue_contract_amendment_comment_id": issue_contract["id"],
        "pr_candidate_insight_comment_id": candidate_insight["id"],
        "handoff_comment_id": handoff["id"],
        "post_bundle_checklist": target_checklist,
        "final_acceptance": False,
        "final_acceptance_authority": "same-process-live-verification-required",
    }


def _bounded_file_bytes(path: Path, label: str, limit: int) -> tuple[Path, bytes]:
    try:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        raw = resolved.read_bytes()
        after = resolved.stat()
    except OSError as error:
        raise EvidenceError(f"{label} is unavailable") from error
    _require(
        resolved.is_file()
        and 0 < len(raw) <= limit
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        and before.st_size == len(raw),
        f"{label} file identity or byte size differs",
    )
    return resolved, raw


def _load_response_artifact(
    descriptor: Any,
    base: Path,
    candidate: Mapping[str, str],
    label: str,
) -> Any:
    _exact_keys(
        descriptor,
        {"path", "sha256", "size_bytes", "media_type", "candidate_evidence"},
        f"{label} descriptor",
    )
    path_value = descriptor["path"]
    _require(
        isinstance(path_value, str) and bool(path_value) and "\\" not in path_value,
        f"{label} path differs",
    )
    relative = PurePosixPath(path_value)
    _require(
        not relative.is_absolute()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{label} path is not canonical",
    )
    path, raw = _bounded_file_bytes(
        base.joinpath(*relative.parts), label, MAX_GITHUB_RESPONSE_BYTES
    )
    _require(
        path.is_relative_to(base)
        and descriptor["media_type"] == MEDIA_TYPE_JSON
        and descriptor["candidate_evidence"] == candidate
        and descriptor["size_bytes"] == len(raw)
        and descriptor["sha256"] == hashlib.sha256(raw).hexdigest(),
        f"{label} descriptor or candidate binding differs",
    )
    return _strict_json(raw, label)


def _load_operations_capture(
    index_path: Path,
    manifest: Path,
) -> tuple[Path, bytes, dict[str, Any], dict[str, Any], dict[str, str]]:
    _manifest_path, _manifest_raw = _bounded_file_bytes(
        manifest, "trusted manifest", MAX_PUBLICATION_POLICY_BYTES
    )
    candidate = candidate_evidence(manifest.resolve(strict=True))
    index, index_raw = _bounded_file_bytes(
        index_path, "operations evidence index", MAX_OPERATIONS_INDEX_BYTES
    )
    document = _strict_json(index_raw, "operations evidence index")
    _require(isinstance(document, dict), "operations evidence index differs")
    records = document.get("responses")
    _require(
        isinstance(records, dict) and set(records) == RESPONSE_ROLES,
        "operations response index differs",
    )
    base = index.parent.resolve()
    raw_responses = {
        role: _load_response_artifact(
            records[role].get("artifact") if isinstance(records[role], dict) else None,
            base,
            candidate,
            f"operations raw response {role}",
        )
        for role in RESPONSE_ROLE_ORDER
    }
    evaluate_operations(document, raw_responses, candidate)
    return index, index_raw, document, raw_responses, candidate


def _require_authenticated_gh() -> None:
    try:
        subprocess.run(
            ["gh", "auth", "status", "--hostname", "github.com"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError("authenticated GitHub CLI access is required") from error


def _fresh_github_capture(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    responses: dict[str, Any] = {}
    captures: dict[str, Any] = {}
    records = document["responses"]
    for role in RESPONSE_ROLE_ORDER:
        request = records[role]["request"]
        raw, capture = _github_api_capture(
            request["endpoint"],
            parameters=request["parameters"] or None,
            paginated=request["paginated"],
            graphql_variables=(request["variables"] if request["graphql"] else None),
        )
        responses[role] = _strict_json(raw, f"fresh GitHub response {role}")
        captures[role] = capture
    return responses, captures


def _compare_live_capture(
    captured_responses: Mapping[str, Any],
    captured_ledgers: Mapping[str, Any],
    live_responses: Mapping[str, Any],
    live_ledgers: Mapping[str, Any],
) -> None:
    _require(
        set(captured_responses)
        == set(captured_ledgers)
        == set(live_responses)
        == set(live_ledgers)
        == RESPONSE_ROLES,
        "captured/live GitHub response closure differs",
    )
    for role in RESPONSE_ROLE_ORDER:
        _require(
            _canonical_json_bytes(captured_responses[role])
            == _canonical_json_bytes(live_responses[role]),
            f"fresh GitHub response {role} is stale or substituted",
        )
        _require(
            _canonical_json_bytes(captured_ledgers[role])
            == _canonical_json_bytes(live_ledgers[role]),
            f"fresh GitHub response metadata {role} is stale or substituted",
        )


def _validate_downloaded_publication(
    publication_receipt: Mapping[str, Any],
    publication_policy_path: Path,
    publication_policy_sha256: str,
    publication_asset_paths: Mapping[str, Path],
    operations_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate publication bytes through the public, separately owned adapter."""

    try:
        from benchmarks import issue123_publication as publication
    except (ImportError, OSError) as error:
        raise EvidenceError(
            "publication validation interface is unavailable"
        ) from error
    _require(
        publication.PUBLICATION_RECEIPT_KIND == PUBLICATION_RECEIPT_KIND
        and publication.EXECUTION_WITNESS_KIND == PUBLICATION_EXECUTION_WITNESS_KIND
        and publication.RELEASE_CAPTURE_KIND == PUBLICATION_RELEASE_CAPTURE_KIND
        and publication.EXECUTION_WITNESS_PATH == PUBLICATION_EXECUTION_WITNESS_PATH
        and tuple(publication.ASSET_ORDER) == tuple(TECHNICAL_RELEASE_ASSETS.items()),
        "publication validation interface is incompatible",
    )
    publication_result = operations_result.get("publication")
    _require(
        isinstance(publication_result, Mapping)
        and publication_receipt.get("sha256")
        == publication_result.get("receipt_sha256")
        and publication_receipt.get("size_bytes")
        == publication_result.get("receipt_size_bytes"),
        "publication receipt envelope differs from the live operations binding",
    )
    identity_anchor = publication_result.get("release_identity_anchor")
    _require(
        isinstance(identity_anchor, dict)
        and identity_anchor.get("release_id") == publication_result.get("release_id")
        and identity_anchor.get("tag_name") == publication_result.get("release_tag")
        and isinstance(identity_anchor.get("assets"), list)
        and {
            record.get("role"): record.get("asset_id")
            for record in identity_anchor["assets"]
            if isinstance(record, dict)
        }
        == publication_result.get("asset_ids"),
        "caller-owned publication release identity anchor differs",
    )
    strict_release_identity = {
        key: identity_anchor[key]
        for key in (
            "repository",
            "release_id",
            "tag_name",
            "target_commitish",
            "api_url",
            "html_url",
            "tag_ref",
        )
    }
    strict_release_identity["assets"] = [
        {
            key: record[key]
            for key in (
                "role",
                "asset_id",
                "release_id",
                "name",
                "api_url",
                "browser_download_url",
            )
        }
        for record in identity_anchor["assets"]
    ]
    _policy_path, policy_raw = _bounded_file_bytes(
        publication_policy_path,
        "trusted publication policy",
        MAX_PUBLICATION_POLICY_BYTES,
    )
    policy = _strict_json(policy_raw, "trusted publication policy")
    _require(
        isinstance(publication_policy_sha256, str)
        and SHA256_RE.fullmatch(publication_policy_sha256) is not None,
        "caller-owned publication policy digest anchor differs",
    )
    _require(
        isinstance(publication_asset_paths, Mapping)
        and set(publication_asset_paths) == set(TECHNICAL_RELEASE_ASSETS),
        "publication asset path closure differs",
    )
    assets: dict[str, bytes] = {}
    downloaded_ledger = []
    for role, name in TECHNICAL_RELEASE_ASSETS.items():
        _asset_path, raw = _bounded_file_bytes(
            publication_asset_paths[role],
            f"downloaded publication asset {role}",
            MAX_PUBLICATION_ASSET_BYTES,
        )
        assets[name] = raw
        downloaded_ledger.append(
            {
                "role": role,
                "name": name,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    expected_assets = {
        role: {
            "name": asset["name"],
            "size_bytes": asset["size_bytes"],
            "sha256": asset["sha256"],
        }
        for role, asset in operations_result["technical_release"]["assets"].items()
    }
    _require(
        downloaded_ledger
        == publication_result.get("asset_ledger")
        == publication_receipt["document"].get("asset_ledger")
        and expected_assets
        == {
            record["role"]: {
                "name": record["name"],
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
            for record in downloaded_ledger
        },
        "downloaded publication bytes differ from the external byte ledger",
    )
    bindings = publication_result["bindings"]
    receipt_document = publication_receipt["document"]
    policy_sha256 = hashlib.sha256(_canonical_json_bytes(policy)).hexdigest()
    _require(
        policy_sha256
        == publication_policy_sha256
        == receipt_document["hashes"].get("trusted_policy_sha256"),
        "publication receipt or bytes differ from the caller-owned policy digest",
    )
    receipt_raw = _canonical_json_bytes(receipt_document)
    try:
        reopened = publication.validate_publication_receipt(
            receipt_raw,
            assets,
            expected_policy=policy,
            expected_release_identity=strict_release_identity,
            expected_bindings=bindings,
            expected_assets=expected_assets,
        )
        validated = publication.validate_publication_assets(
            assets,
            expected_policy=policy,
            expected_bindings=bindings,
            expected_assets=expected_assets,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise EvidenceError(
            "downloaded publication receipt/assets/policy validation failed"
        ) from error
    _require(
        reopened == receipt_document
        and validated.get("execution_witness") == receipt_document["execution_witness"]
        and reopened.get("release_capture") == identity_anchor,
        "publication validator result or caller-owned release anchor differs",
    )
    profiler = validated.get("event_profiler")
    _require(
        isinstance(profiler, dict)
        and profiler.get("contract_id") == publication.EVENT_PROFILER_CONTRACT_ID
        and profiler.get("bindings") == bindings
        and isinstance(profiler.get("records"), list)
        and isinstance(profiler.get("closure"), dict)
        and profiler["closure"].get("record_count") == len(profiler["records"]),
        "downloaded event-profiler claims differ",
    )
    return {
        "strict_four_byte_validator": "same-process-invoked",
        "receipt_sha256": publication_receipt["sha256"],
        "trusted_policy_sha256": policy_sha256,
        "asset_ledger": downloaded_ledger,
        "release_identity_anchor": identity_anchor,
        "bindings": bindings,
        "execution_claims": publication_result["execution_claims"],
        "event_profiler": {
            "contract_id": profiler["contract_id"],
            "record_count": profiler["closure"]["record_count"],
            "inventory_sha256": profiler["closure"]["inventory_sha256"],
            "asset_sha256": expected_assets["event_profiler"]["sha256"],
        },
    }


def _validate_post_bundle_acknowledgment(
    expectation: AuthenticatedPostBundleExpectation,
    captured_issue: Mapping[str, Any],
    captured_capture: Mapping[str, Any],
    live_issue: Mapping[str, Any],
    live_capture: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        type(expectation) is AuthenticatedPostBundleExpectation,
        "post-bundle expectation is not authenticated",
    )
    captured = _validate_post_bundle_checklist(captured_issue, "checked")
    live = _validate_post_bundle_checklist(live_issue, "checked")
    captured_transition = checklist_transition_sha256(captured_issue, "checked")
    live_transition = checklist_transition_sha256(live_issue, "checked")
    captured_digest = captured_capture.get("canonical_response_sha256")
    live_digest = live_capture.get("canonical_response_sha256")
    expected = {
        "checked_lines": list(expectation.checked_lines),
        "o0_canonical_response_sha256": expectation.o0_canonical_response_sha256,
        "o1_canonical_response_sha256": expectation.o1_canonical_response_sha256,
        "o1_body_sha256": expectation.o1_body_sha256,
        "o1_updated_at": expectation.o1_updated_at,
        "b0_inventory_root": expectation.b0_inventory_root,
        "b0_reopen_receipt_sha256": expectation.b0_reopen_receipt_sha256,
        "b0_reopened_at": expectation.b0_reopened_at,
    }
    digest_keys = (
        "o0_canonical_response_sha256",
        "o1_canonical_response_sha256",
        "o1_body_sha256",
        "b0_inventory_root",
        "b0_reopen_receipt_sha256",
    )
    _require(
        expectation.checked_lines == FINAL_CHECKLIST_CHECKED
        and all(
            isinstance(expected[key], str)
            and re.fullmatch(r"[0-9a-f]{64}", expected[key]) is not None
            for key in digest_keys
        )
        and isinstance(expectation.checklist_transition_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", expectation.checklist_transition_sha256)
        is not None
        and expectation.o0_canonical_response_sha256
        != expectation.o1_canonical_response_sha256
        and captured_digest == live_digest == expectation.o1_canonical_response_sha256
        and hashlib.sha256(_canonical_json_bytes(captured_issue)).hexdigest()
        == captured_digest
        and hashlib.sha256(_canonical_json_bytes(live_issue)).hexdigest() == live_digest
        and captured == live
        and captured_transition
        == live_transition
        == expectation.checklist_transition_sha256
        and captured["body_sha256"] == expectation.o1_body_sha256
        and captured["updated_at"] == expectation.o1_updated_at
        and _timestamp(expectation.o1_updated_at, "O1 update time")
        >= _timestamp(expectation.b0_reopened_at, "B0 reopen time"),
        "checked O1 response does not match the B0/B1 acknowledgment expectation",
    )
    return {
        **expected,
        "fresh_response_equal": True,
    }


def _assert_provenance_receipt_safe(value: Any) -> None:
    forbidden_keys = (
        "authorization",
        "cookie",
        "oauth",
        "token",
        "secret",
        "salt",
        "opening",
        "hostname",
        "username",
        "user_name",
        "account_id",
        "environment",
        "device_id",
        "device_uuid",
        "serial_number",
        "source_path",
        "private_path",
        "absolute_path",
    )

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                _require(
                    isinstance(key, str)
                    and not any(word in key.lower() for word in forbidden_keys),
                    "live-verification receipt contains sensitive metadata",
                )
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            _require(
                re.search(
                    r"(?i)(?:bearer\s+|github_pat_|gh[opusr]_[A-Za-z0-9]{12,})",
                    item,
                )
                is None,
                "live-verification receipt contains credential material",
            )

    visit(value)
    try:
        from benchmarks import issue123_privacy as privacy

        privacy.scan_public_bytes(
            _canonical_json_bytes(value),
            label="live operations receipt",
        )
    except (ImportError, TypeError, ValueError) as error:
        raise EvidenceError(
            "live-verification receipt failed privacy policy"
        ) from error


def _validate_baseline_authority_set(
    value: BaselineAuthoritySet,
    label: str,
) -> BaselineAuthoritySet:
    _require(
        type(value) is BaselineAuthoritySet
        and isinstance(value.root_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", value.root_commit) is not None
        and type(value.assets) is tuple
        and len(value.assets) == 2,
        f"{label} differs",
    )
    seen_names: set[str] = set()
    for ordinal, asset in enumerate(value.assets):
        _require(
            type(asset) is BaselineAssetExpectation
            and asset.ordinal == ordinal
            and asset.thread_mode == ("one", "physical")[ordinal]
            and isinstance(asset.name, str)
            and PurePosixPath(asset.name).name == asset.name
            and asset.name not in seen_names
            and isinstance(asset.publication_url, str)
            and PurePosixPath(urlsplit(asset.publication_url).path).name == asset.name
            and type(asset.size_bytes) is int
            and asset.size_bytes > 0
            and isinstance(asset.sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", asset.sha256) is not None,
            f"{label} asset {ordinal} differs",
        )
        seen_names.add(asset.name)
    return value


def _require_equal_baseline_authorities(
    code_authority: BaselineAuthoritySet,
    manifest_authority: BaselineAuthoritySet,
    b1_authority: BaselineAuthoritySet,
) -> BaselineAuthoritySet:
    code = _validate_baseline_authority_set(code_authority, "code-owned baseline")
    manifest = _validate_baseline_authority_set(
        manifest_authority,
        "trusted-manifest baseline",
    )
    b1 = _validate_baseline_authority_set(b1_authority, "authenticated B1 baseline")
    _require(
        code == manifest == b1,
        "baseline code, manifest, and authenticated B1 authority differ",
    )
    return code


def _baseline_manifest_expectation(manifest_path: Path) -> BaselineAuthoritySet:
    _path, raw = _bounded_file_bytes(
        manifest_path,
        "trusted baseline manifest",
        MAX_PUBLICATION_POLICY_BYTES,
    )
    manifest = _strict_json(raw, "trusted baseline manifest")
    try:
        timing = manifest["performance_gates"]["cpu_acceptance"]["timing_reference"]
        pins = timing["slice_artifacts"]
    except (KeyError, TypeError) as error:
        raise EvidenceError("trusted baseline manifest pins are absent") from error
    _require(
        timing.get("root_commit") == PRODUCTION_BASELINE_AUTHORITY_SET.root_commit
        and isinstance(pins, list)
        and len(pins) == len(PRODUCTION_BASELINE_AUTHORITY_SET.assets),
        "trusted baseline manifest identity differs",
    )
    assets: list[BaselineAssetExpectation] = []
    for ordinal, (pin, constant) in enumerate(
        zip(pins, PRODUCTION_BASELINE_AUTHORITY_SET.assets, strict=True)
    ):
        _require(
            isinstance(pin, dict)
            and pin.get("thread_mode") == constant.thread_mode
            and pin.get("publication_url") == constant.publication_url
            and pin.get("size_bytes") == constant.size_bytes
            and pin.get("sha256") == constant.sha256,
            f"trusted baseline manifest asset {ordinal} differs from constants",
        )
        assets.append(
            BaselineAssetExpectation(
                ordinal=ordinal,
                thread_mode=pin["thread_mode"],
                name=PurePosixPath(urlsplit(pin["publication_url"]).path).name,
                publication_url=pin["publication_url"],
                size_bytes=pin["size_bytes"],
                sha256=pin["sha256"],
            )
        )
    return _validate_baseline_authority_set(
        BaselineAuthoritySet(
            root_commit=timing["root_commit"],
            assets=tuple(assets),
        ),
        "trusted-manifest baseline",
    )


def _download_baseline_release_asset(asset_id: int) -> bytes:
    command = [
        "gh",
        "api",
        "--hostname",
        "github.com",
        f"repos/{REPOSITORY}/releases/assets/{asset_id}",
        "-H",
        "Accept: application/octet-stream",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvidenceError("baseline release asset download failed") from error
    return bytes(completed.stdout)


@dataclass(slots=True)
class _BaselineAssetLease:
    expectation: BaselineAssetExpectation
    descriptor: int
    identity: tuple[int, int, int, int, int]


class BaselineAuthorityLease:
    """Own exact ordered baseline bytes until final authority is durable."""

    __slots__ = (
        "_temporary",
        "_root",
        "_root_descriptor",
        "_root_identity",
        "_assets",
        "validation",
        "_closed",
    )

    def __init__(
        self,
        temporary: tempfile.TemporaryDirectory[str],
        root: Path,
        root_descriptor: int,
        root_identity: tuple[int, int, int],
        assets: tuple[_BaselineAssetLease, ...],
        validation: dict[str, Any],
    ) -> None:
        self._temporary = temporary
        self._root = root
        self._root_descriptor = root_descriptor
        self._root_identity = root_identity
        self._assets = assets
        self.validation = validation
        self._closed = False

    @staticmethod
    def _read_descriptor(descriptor: int, size_bytes: int) -> bytes:
        chunks: list[bytes] = []
        offset = 0
        while offset < size_bytes:
            try:
                chunk = os.pread(
                    descriptor, min(size_bytes - offset, 1024 * 1024), offset
                )
            except OSError:
                raise EvidenceError("retained baseline bytes are unavailable") from None
            _require(bool(chunk), "retained baseline bytes changed")
            chunks.append(chunk)
            offset += len(chunk)
        try:
            extra = os.pread(descriptor, 1, size_bytes)
        except OSError:
            raise EvidenceError("retained baseline bytes are unavailable") from None
        _require(not extra, "retained baseline bytes changed")
        return b"".join(chunks)

    def require_unchanged(self) -> None:
        _require(not self._closed, "retained baseline authority is closed")
        try:
            root_metadata = os.fstat(self._root_descriptor)
            named_root = self._root.lstat()
            entries = os.listdir(self._root_descriptor)
        except OSError:
            raise EvidenceError("retained baseline directory changed") from None
        _require(
            stat.S_ISDIR(root_metadata.st_mode)
            and stat.S_IMODE(root_metadata.st_mode) == 0o700
            and (
                root_metadata.st_dev,
                root_metadata.st_ino,
                root_metadata.st_mode,
            )
            == self._root_identity
            and (named_root.st_dev, named_root.st_ino, named_root.st_mode)
            == self._root_identity
            and sorted(entries)
            == sorted(asset.expectation.name for asset in self._assets),
            "retained baseline directory changed",
        )
        for lease in self._assets:
            try:
                retained = os.fstat(lease.descriptor)
            except OSError:
                raise EvidenceError("retained baseline asset changed") from None
            identity = (
                retained.st_dev,
                retained.st_ino,
                retained.st_mode,
                retained.st_size,
                retained.st_mtime_ns,
            )
            _require(
                identity == lease.identity
                and stat.S_ISREG(retained.st_mode)
                and stat.S_IMODE(retained.st_mode) == 0o600,
                "retained baseline asset changed",
            )
            raw = self._read_descriptor(
                lease.descriptor,
                lease.expectation.size_bytes,
            )
            _require(
                hashlib.sha256(raw).hexdigest() == lease.expectation.sha256,
                "retained baseline asset changed",
            )
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            try:
                reopened = os.open(
                    lease.expectation.name,
                    flags,
                    dir_fd=self._root_descriptor,
                )
            except OSError:
                raise EvidenceError("retained baseline asset changed") from None
            reopen_failure: EvidenceError | None = None
            try:
                reopened_metadata = os.fstat(reopened)
                reopened_raw = self._read_descriptor(
                    reopened,
                    lease.expectation.size_bytes,
                )
                if (
                    reopened_metadata.st_dev,
                    reopened_metadata.st_ino,
                    reopened_metadata.st_mode,
                    reopened_metadata.st_size,
                    reopened_metadata.st_mtime_ns,
                ) != lease.identity or reopened_raw != raw:
                    reopen_failure = EvidenceError("retained baseline asset changed")
            except OSError, EvidenceError:
                reopen_failure = EvidenceError("retained baseline asset changed")
            try:
                os.close(reopened)
            except OSError:
                reopen_failure = EvidenceError("retained baseline asset changed")
            if reopen_failure is not None:
                raise reopen_failure from None

    def close(self, *, primary_error: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup_failed = False
        for lease in self._assets:
            descriptor = lease.descriptor
            lease.descriptor = -1
            try:
                os.close(descriptor)
            except OSError:
                cleanup_failed = True
        root_descriptor = self._root_descriptor
        self._root_descriptor = -1
        try:
            os.close(root_descriptor)
        except OSError:
            cleanup_failed = True
        try:
            self._temporary.cleanup()
        except OSError:
            cleanup_failed = True
        if cleanup_failed and primary_error is None:
            raise EvidenceError(
                "retained baseline authority could not be closed"
            ) from None


def _synthetic_baseline_authority_fixture() -> (
    tuple[BaselineAuthoritySet, dict[int, bytes]]
):
    """Return a code-owned raw-edge fixture; production never selects it."""

    raw_values = (
        b"issue123-synthetic-baseline-one-v1\n",
        b"issue123-synthetic-baseline-physical-v1\n",
    )
    assets = tuple(
        BaselineAssetExpectation(
            ordinal=ordinal,
            thread_mode=("one", "physical")[ordinal],
            name=f"issue123-synthetic-baseline-{ordinal}.json",
            publication_url=(
                f"https://github.com/{REPOSITORY}/releases/download/"
                f"{BASELINE_RELEASE_TAG}/issue123-synthetic-baseline-{ordinal}.json"
            ),
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        for ordinal, raw in enumerate(raw_values)
    )
    return (
        BaselineAuthoritySet(root_commit="f" * 40, assets=assets),
        {701 + ordinal: raw for ordinal, raw in enumerate(raw_values)},
    )


def _capture_baseline_authority(
    *,
    code_authority: BaselineAuthoritySet,
    manifest_authority: BaselineAuthoritySet,
    b1_authority: BaselineAuthoritySet,
    api_capture: Any,
    asset_download: Any,
    observed_at: str | None = None,
) -> BaselineAuthorityLease:
    expected = _require_equal_baseline_authorities(
        code_authority,
        manifest_authority,
        b1_authority,
    )
    release_endpoint = f"repos/{REPOSITORY}/releases/tags/{BASELINE_RELEASE_TAG}"
    tag_endpoint = f"repos/{REPOSITORY}/git/ref/tags/{BASELINE_RELEASE_TAG}"
    release_raw, release_capture = api_capture(release_endpoint)
    tag_raw, tag_capture = api_capture(tag_endpoint)
    release = _strict_json(release_raw, "fresh baseline release")
    tag = _strict_json(tag_raw, "fresh baseline tag")
    api_root = f"https://api.github.com/repos/{REPOSITORY}"
    web_root = f"https://github.com/{REPOSITORY}"
    release_id = release.get("id") if isinstance(release, dict) else None
    release_assets = release.get("assets") if isinstance(release, dict) else None
    tag_object = tag.get("object") if isinstance(tag, dict) else None
    _require(
        type(release_id) is int
        and release_id > 0
        and release.get("tag_name") == BASELINE_RELEASE_TAG
        and release.get("url") == f"{api_root}/releases/{release_id}"
        and release.get("html_url") == f"{web_root}/releases/tag/{BASELINE_RELEASE_TAG}"
        and release.get("draft") is False
        and release.get("prerelease") is False
        and isinstance(release_assets, list)
        and tag.get("ref") == f"refs/tags/{BASELINE_RELEASE_TAG}"
        and isinstance(tag_object, dict)
        and tag_object.get("type") == "commit"
        and tag_object.get("sha") == expected.root_commit,
        "fresh baseline release/tag identity differs",
    )
    _require(
        len(release_assets) == len(expected.assets),
        "fresh baseline release asset closure or order differs",
    )
    for ordinal, (asset, expected_asset) in enumerate(
        zip(release_assets, expected.assets, strict=True)
    ):
        _require(
            isinstance(asset, dict) and asset.get("name") == expected_asset.name,
            f"fresh baseline release asset {ordinal} closure or order differs",
        )
    temporary_owner = tempfile.TemporaryDirectory(prefix="issue123-baseline-")
    temporary = Path(temporary_owner.name)
    root_descriptor = -1
    asset_leases: list[_BaselineAssetLease] = []
    try:
        os.chmod(temporary, 0o700)
        root_descriptor = os.open(
            temporary,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        root_metadata = os.fstat(root_descriptor)
        _require(
            stat.S_ISDIR(root_metadata.st_mode)
            and stat.S_IMODE(root_metadata.st_mode) == 0o700,
            "baseline temporary directory is not private",
        )
        ledger = []
        for ordinal, (asset, expected_asset) in enumerate(
            zip(release_assets, expected.assets, strict=True)
        ):
            asset_id = asset.get("id")
            _require(
                type(asset_id) is int
                and asset_id > 0
                and asset.get("url") == f"{api_root}/releases/assets/{asset_id}"
                and asset.get("browser_download_url") == expected_asset.publication_url
                and asset.get("state") == "uploaded"
                and asset.get("size") == expected_asset.size_bytes,
                f"fresh baseline release asset {ordinal} identity differs",
            )
            raw = asset_download(asset_id)
            _require(
                type(raw) is bytes
                and len(raw) == expected_asset.size_bytes
                and hashlib.sha256(raw).hexdigest() == expected_asset.sha256,
                f"fresh baseline release asset {ordinal} bytes differ",
            )
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    expected_asset.name,
                    flags,
                    0o600,
                    dir_fd=root_descriptor,
                )
                os.fchmod(descriptor, 0o600)
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    _require(written > 0, "baseline release asset write failed")
                    view = view[written:]
                os.fsync(descriptor)
                metadata = os.fstat(descriptor)
            except OSError:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                raise EvidenceError(
                    "baseline release asset could not be staged"
                ) from None
            identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
            _require(
                stat.S_ISREG(metadata.st_mode)
                and stat.S_IMODE(metadata.st_mode) == 0o600
                and metadata.st_size == expected_asset.size_bytes,
                f"fresh baseline release asset {ordinal} bytes differ",
            )
            asset_leases.append(
                _BaselineAssetLease(expected_asset, descriptor, identity)
            )
            ledger.append(
                {
                    "thread_mode": expected_asset.thread_mode,
                    "name": expected_asset.name,
                    "asset_id": asset_id,
                    "release_id": release_id,
                    "api_url": asset["url"],
                    "browser_download_url": asset["browser_download_url"],
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        _require(
            sorted(os.listdir(root_descriptor))
            == sorted(asset.name for asset in expected.assets),
            "fresh baseline release asset closure or order differs",
        )
        if observed_at is None:
            observed_at = (
                dt.datetime.now(dt.UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        else:
            _require(
                isinstance(observed_at, str) and observed_at.endswith("Z"),
                "baseline observation time differs",
            )
        body = {
            "release_identity": {
                "repository": REPOSITORY,
                "release_id": release_id,
                "tag_name": BASELINE_RELEASE_TAG,
                "api_url": release["url"],
                "html_url": release["html_url"],
                "tag_ref": {
                    "ref": tag["ref"],
                    "object_type": tag_object["type"],
                    "object_sha": tag_object["sha"],
                    "object_url": tag_object.get("url"),
                },
            },
            "asset_ledger": ledger,
            "observed_at": observed_at,
            "api_observations": [
                {
                    "endpoint": endpoint,
                    "canonical_response_sha256": hashlib.sha256(raw).hexdigest(),
                    "canonical_response_size_bytes": len(raw),
                    "page_ledger_sha256": hashlib.sha256(
                        _canonical_json_bytes(capture)
                    ).hexdigest(),
                }
                for endpoint, raw, capture in (
                    (release_endpoint, release_raw, release_capture),
                    (tag_endpoint, tag_raw, tag_capture),
                )
            ],
        }
        try:
            from benchmarks import issue123_privacy as privacy

            authority_sha256 = privacy.tagged_canonical_sha256(
                BASELINE_AUTHORITY_DOMAIN,
                body,
            )
        except ImportError, TypeError, ValueError:
            raise EvidenceError(
                "baseline authority digest could not be computed"
            ) from None
        lease = BaselineAuthorityLease(
            temporary_owner,
            temporary,
            root_descriptor,
            (root_metadata.st_dev, root_metadata.st_ino, root_metadata.st_mode),
            tuple(asset_leases),
            {**body, "authority_sha256": authority_sha256},
        )
        root_descriptor = -1
        asset_leases = []
        lease.require_unchanged()
        return lease
    except BaseException as primary_error:
        for asset_lease in asset_leases:
            try:
                os.close(asset_lease.descriptor)
            except OSError:
                pass
        if root_descriptor >= 0:
            try:
                os.close(root_descriptor)
            except OSError:
                pass
        try:
            temporary_owner.cleanup()
        except OSError:
            pass
        raise


@contextmanager
def _open_baseline_authority_core(
    *,
    code_authority: BaselineAuthoritySet,
    manifest_authority: BaselineAuthoritySet,
    b1_authority: BaselineAuthoritySet,
    api_capture: Any,
    asset_download: Any,
    observed_at: str | None = None,
):
    lease = _capture_baseline_authority(
        code_authority=code_authority,
        manifest_authority=manifest_authority,
        b1_authority=b1_authority,
        api_capture=api_capture,
        asset_download=asset_download,
        observed_at=observed_at,
    )
    primary_error: BaseException | None = None
    try:
        yield lease
    except BaseException as error:
        primary_error = error
        raise
    finally:
        lease.close(primary_error=primary_error)


def _open_production_baseline_authority(
    manifest_path: Path,
    *,
    authority: str,
    b1_authority: BaselineAuthoritySet,
):
    _require(authority == "live-release", "baseline authority mode is unsupported")
    manifest_authority = _baseline_manifest_expectation(manifest_path)
    return _open_baseline_authority_core(
        code_authority=PRODUCTION_BASELINE_AUTHORITY_SET,
        manifest_authority=manifest_authority,
        b1_authority=b1_authority,
        api_capture=_github_api_capture,
        asset_download=_download_baseline_release_asset,
    )


def _capture_production_baseline_authority(
    manifest_path: Path,
    *,
    authority: str,
    b1_authority: BaselineAuthoritySet,
) -> BaselineAuthorityLease:
    _require(authority == "live-release", "baseline authority mode is unsupported")
    return _capture_baseline_authority(
        code_authority=PRODUCTION_BASELINE_AUTHORITY_SET,
        manifest_authority=_baseline_manifest_expectation(manifest_path),
        b1_authority=b1_authority,
        api_capture=_github_api_capture,
        asset_download=_download_baseline_release_asset,
    )


def _write_private_receipt(
    path: Path,
    raw: bytes,
    *,
    forbidden_roots: tuple[Path, ...],
) -> None:
    try:
        from benchmarks import issue123_privacy as privacy

        privacy.write_private_authority_file(
            path,
            raw,
            label="live-verification receipt",
            forbidden_roots=forbidden_roots,
        )
    except ImportError, OSError, TypeError, ValueError:
        raise EvidenceError(
            "live-verification receipt could not be committed"
        ) from None


@dataclass(slots=True)
class _RetainedReceipt:
    path: Path
    descriptor: int
    identity: tuple[int, int, int, int, int]
    raw: bytes

    def require_unchanged(self) -> None:
        _require(self.descriptor >= 0, "retained live receipt is closed")
        try:
            metadata = os.fstat(self.descriptor)
            named = self.path.lstat()
            current = BaselineAuthorityLease._read_descriptor(
                self.descriptor,
                len(self.raw),
            )
        except OSError:
            raise EvidenceError("retained live receipt changed") from None
        _require(
            (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
            == self.identity
            and (
                named.st_dev,
                named.st_ino,
                named.st_mode,
                named.st_size,
                named.st_mtime_ns,
            )
            == self.identity
            and stat.S_ISREG(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == 0o600
            and current == self.raw,
            "retained live receipt changed",
        )

    def close(self) -> bool:
        if self.descriptor < 0:
            return False
        descriptor = self.descriptor
        self.descriptor = -1
        try:
            os.close(descriptor)
        except OSError:
            return True
        return False


def _retain_live_receipt(path: Path, expected_raw: bytes) -> _RetainedReceipt:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise EvidenceError("live-verification receipt could not be retained") from None
    try:
        metadata = os.fstat(descriptor)
        raw = BaselineAuthorityLease._read_descriptor(descriptor, len(expected_raw))
        named = path.lstat()
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        _require(
            stat.S_ISREG(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == 0o600
            and metadata.st_size == len(expected_raw)
            and raw == expected_raw
            and (
                named.st_dev,
                named.st_ino,
                named.st_mode,
                named.st_size,
                named.st_mtime_ns,
            )
            == identity,
            "live-verification receipt durable bytes differ",
        )
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return _RetainedReceipt(path, descriptor, identity, raw)


class OperationsAuthorityLease:
    """Retain baseline and receipt authority through the completion link."""

    __slots__ = ("receipt", "_baseline", "_receipt", "_closed")

    def __init__(
        self,
        receipt: dict[str, Any],
        baseline: BaselineAuthorityLease,
        retained_receipt: _RetainedReceipt,
    ) -> None:
        self.receipt = receipt
        self._baseline = baseline
        self._receipt = retained_receipt
        self._closed = False

    def require_unchanged(self) -> None:
        _require(not self._closed, "operations authority lease is closed")
        self._baseline.require_unchanged()
        self._receipt.require_unchanged()

    def close(self, *, primary_error: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup_failed = self._receipt.close()
        try:
            self._baseline.close(
                primary_error=(
                    primary_error
                    if primary_error is not None
                    else (
                        EvidenceError("receipt cleanup failed")
                        if cleanup_failed
                        else None
                    )
                )
            )
        except EvidenceError:
            cleanup_failed = True
        if cleanup_failed and primary_error is None:
            raise EvidenceError("operations authority could not be closed") from None


def _verify_operations_live_with_baseline(
    *,
    index_path: Path,
    manifest: Path,
    publication_policy: Path,
    publication_policy_sha256: str,
    publication_assets: Mapping[str, Path],
    receipt_output: Path,
    post_bundle_expectation: AuthenticatedPostBundleExpectation,
    source_bundle_root: Path,
    reopened_bundle_root: Path,
    baseline_lease: BaselineAuthorityLease,
) -> dict[str, Any]:
    """Perform the authoritative live check and emit a provenance-only receipt."""

    _require(
        type(post_bundle_expectation) is AuthenticatedPostBundleExpectation,
        "post-bundle expectation is not authenticated",
    )
    forbidden_roots = (
        source_bundle_root,
        reopened_bundle_root,
        Path(index_path).parent,
        *(Path(path).parent for path in publication_assets.values()),
    )
    try:
        from benchmarks import issue123_privacy as privacy

        privacy.preflight_private_output_path(
            receipt_output,
            label="live-verification receipt",
            forbidden_roots=forbidden_roots,
        )
    except ImportError, OSError, TypeError, ValueError:
        failure = EvidenceError("live-verification receipt overlaps protected evidence")
    else:
        failure = None
    if failure is not None:
        raise failure from None
    (
        _resolved_index,
        index_raw,
        document,
        captured_responses,
        candidate,
    ) = _load_operations_capture(index_path, manifest)
    baseline_lease.require_unchanged()
    captured_result = evaluate_operations(document, captured_responses, candidate)
    baseline_lease.require_unchanged()
    _require(
        captured_result["final_acceptance"] is False,
        "offline operations evidence cannot authorize final acceptance",
    )
    live_responses, live_captures = _fresh_github_capture(document)
    _compare_live_capture(
        captured_responses,
        document["response_captures"],
        live_responses,
        live_captures,
    )
    post_bundle_acknowledgment = _validate_post_bundle_acknowledgment(
        post_bundle_expectation,
        captured_responses["issue_123"],
        document["response_captures"]["issue_123"],
        live_responses["issue_123"],
        live_captures["issue_123"],
    )
    live_document = {**document, "response_captures": live_captures}
    live_result = evaluate_operations(live_document, live_responses, candidate)
    baseline_lease.require_unchanged()
    publication = _validate_downloaded_publication(
        document["publication_receipt"],
        publication_policy,
        publication_policy_sha256,
        publication_assets,
        live_result,
    )
    baseline_validation = baseline_lease.validation
    verified_at = (
        dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    queries = []
    for role in RESPONSE_ROLE_ORDER:
        capture = live_captures[role]
        queries.append(
            {
                "role": role,
                "canonical_response_sha256": capture["canonical_response_sha256"],
                "canonical_response_size_bytes": capture[
                    "canonical_response_size_bytes"
                ],
                "page_count": len(capture["pages"]),
                "page_ledger_sha256": hashlib.sha256(
                    _canonical_json_bytes(capture)
                ).hexdigest(),
            }
        )
    receipt = {
        "schema_version": LIVE_VERIFICATION_RECEIPT_SCHEMA_VERSION,
        "kind": LIVE_VERIFICATION_RECEIPT_KIND,
        "authority": "same-process-authenticated-gh-live-verification",
        "receipt_replay_authority": False,
        "verified_at": verified_at,
        "candidate_evidence": candidate,
        "repository": REPOSITORY,
        "pull_request_number": PULL_REQUEST_NUMBER,
        "operations_index": {
            "size_bytes": len(index_raw),
            "sha256": hashlib.sha256(index_raw).hexdigest(),
        },
        "publication_validation": publication,
        "post_bundle_acknowledgment": post_bundle_acknowledgment,
        "baseline_validation": baseline_validation,
        "queries": queries,
        "same_process_live_accepted": True,
    }
    _assert_provenance_receipt_safe(receipt)
    baseline_lease.require_unchanged()
    raw = _canonical_json_bytes(receipt)
    _write_private_receipt(
        receipt_output,
        raw,
        forbidden_roots=forbidden_roots,
    )
    baseline_lease.require_unchanged()
    return receipt


@contextmanager
def open_verified_operations_live(
    *,
    index_path: Path,
    manifest: Path,
    publication_policy: Path,
    publication_policy_sha256: str,
    publication_assets: Mapping[str, Path],
    receipt_output: Path,
    post_bundle_lease: Any,
    baseline_authority: str = "live-release",
):
    """Create v3 authority while retaining its B1, baseline, and receipt inputs."""

    from benchmarks import issue123_completion as completion

    _require(
        type(post_bundle_lease) is completion.AuthenticatedPostBundleLease,
        "post-bundle lease is not authenticated",
    )
    post_bundle_lease.require_unchanged()
    source_bundle_root, reopened_bundle_root = post_bundle_lease._private_writer_roots()
    b1_authority = post_bundle_lease._baseline_authority_set(sys.modules[__name__])
    forbidden_roots = (
        source_bundle_root,
        reopened_bundle_root,
        Path(index_path).parent,
        *(Path(path).parent for path in publication_assets.values()),
    )
    try:
        from benchmarks import issue123_privacy as privacy

        privacy.preflight_private_output_path(
            receipt_output,
            label="live-verification receipt",
            forbidden_roots=forbidden_roots,
        )
    except ImportError, OSError, TypeError, ValueError:
        raise EvidenceError(
            "live-verification receipt overlaps protected evidence"
        ) from None
    _require_authenticated_gh()
    baseline = _capture_production_baseline_authority(
        manifest,
        authority=baseline_authority,
        b1_authority=b1_authority,
    )
    authority_lease: OperationsAuthorityLease | None = None
    primary_error: BaseException | None = None
    try:
        receipt = _verify_operations_live_with_baseline(
            index_path=index_path,
            manifest=manifest,
            publication_policy=publication_policy,
            publication_policy_sha256=publication_policy_sha256,
            publication_assets=publication_assets,
            receipt_output=receipt_output,
            post_bundle_expectation=post_bundle_lease.expectation,
            source_bundle_root=source_bundle_root,
            reopened_bundle_root=reopened_bundle_root,
            baseline_lease=baseline,
        )
        receipt_raw = _canonical_json_bytes(receipt)
        retained_receipt = _retain_live_receipt(Path(receipt_output), receipt_raw)
        authority_lease = OperationsAuthorityLease(
            receipt,
            baseline,
            retained_receipt,
        )
        authority_lease.require_unchanged()
        post_bundle_lease.require_unchanged()
        yield authority_lease
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if authority_lease is None:
            baseline.close(primary_error=primary_error)
        else:
            authority_lease.close(primary_error=primary_error)


def verify_operations_live(
    *,
    index_path: Path,
    manifest: Path,
    publication_policy: Path,
    publication_policy_sha256: str,
    publication_assets: Mapping[str, Path],
    receipt_output: Path,
    post_bundle_lease: Any,
    baseline_authority: str = "live-release",
) -> dict[str, Any]:
    """Compatibility wrapper retaining authority through receipt revalidation."""

    with open_verified_operations_live(
        index_path=index_path,
        manifest=manifest,
        publication_policy=publication_policy,
        publication_policy_sha256=publication_policy_sha256,
        publication_assets=publication_assets,
        receipt_output=receipt_output,
        post_bundle_lease=post_bundle_lease,
        baseline_authority=baseline_authority,
    ) as lease:
        lease.require_unchanged()
        return lease.receipt


class _CliUsageError(ValueError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _CliUsageError("operations CLI usage differs") from None


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _SafeArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--repository", default=REPOSITORY)
    capture.add_argument(
        "--pull-request",
        type=int,
        choices=(PULL_REQUEST_NUMBER,),
        required=True,
    )
    capture.add_argument("--ci-run-id", type=int, required=True)
    capture.add_argument("--codeql-run-id", type=int, required=True)
    capture.add_argument("--technical-release-tag", required=True)
    capture.add_argument("--publication-receipt", type=Path, required=True)
    capture.add_argument("--output-directory", type=Path, required=True)
    capture.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    verify = subparsers.add_parser("verify-live")
    verify.add_argument("--index", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--publication-policy", type=Path, required=True)
    verify.add_argument("--publication-policy-sha256", required=True)
    verify.add_argument("--technical-evidence-asset", type=Path, required=True)
    verify.add_argument("--technical-summary-asset", type=Path, required=True)
    verify.add_argument("--raw-timing-asset", type=Path, required=True)
    verify.add_argument("--event-profiler-asset", type=Path, required=True)
    verify.add_argument("--source-index", type=Path, required=True)
    verify.add_argument("--reopened-index", type=Path, required=True)
    verify.add_argument("--private-openings", type=Path, required=True)
    verify.add_argument("--pre-ack-bundle-reopen-receipt", type=Path, required=True)
    verify.add_argument("--final-bundle-reopen-receipt", type=Path, required=True)
    verify.add_argument(
        "--runtime-receipts",
        type=Path,
        nargs=5,
        metavar=(
            "CPU",
            "CUDA_EAGER",
            "CUDA_GRAPH",
            "SINGLE_GPU_2D",
            "SINGLE_GPU_3D",
        ),
        required=True,
    )
    verify.add_argument(
        "--baseline-authority",
        choices=("live-release",),
        default="live-release",
    )
    verify.add_argument("--receipt-output", type=Path, required=True)

    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] not in {"capture", "verify-live"}:
        values.insert(0, "capture")
    return parser.parse_args(values)


def _main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.command == "capture":
        capture_operations(
            repository=args.repository,
            pull_request_number=args.pull_request,
            ci_run_id=args.ci_run_id,
            codeql_run_id=args.codeql_run_id,
            technical_release_tag=args.technical_release_tag,
            publication_receipt=args.publication_receipt,
            output_directory=args.output_directory,
            manifest=args.manifest,
        )
    else:
        from benchmarks import issue123_completion as completion

        with completion.open_authenticated_post_bundle_transition(
            source_index=args.source_index,
            reopened_index=args.reopened_index,
            protected_openings=args.private_openings,
            pre_ack_bundle_reopen_receipt=(args.pre_ack_bundle_reopen_receipt),
            final_bundle_reopen_receipt=args.final_bundle_reopen_receipt,
            manifest_path=args.manifest,
            runtime_receipt_paths=args.runtime_receipts,
        ) as lease:
            lease.require_unchanged()
            with open_verified_operations_live(
                index_path=args.index,
                manifest=args.manifest,
                publication_policy=args.publication_policy,
                publication_policy_sha256=args.publication_policy_sha256,
                publication_assets={
                    "technical_evidence": args.technical_evidence_asset,
                    "technical_summary": args.technical_summary_asset,
                    "raw_timing": args.raw_timing_asset,
                    "event_profiler": args.event_profiler_asset,
                },
                receipt_output=args.receipt_output,
                post_bundle_lease=lease,
                baseline_authority=args.baseline_authority,
            ) as operations_lease:
                receipt = operations_lease.receipt
                _receipt_path, receipt_raw = _bounded_file_bytes(
                    args.receipt_output,
                    "live-verification receipt",
                    MAX_PUBLICATION_RECEIPT_BYTES,
                )
                _require(
                    receipt_raw == _canonical_json_bytes(receipt)
                    and stat.S_IMODE(_receipt_path.stat().st_mode) == 0o600,
                    "live-verification receipt durable bytes differ",
                )
                operations_lease.require_unchanged()
                lease.require_unchanged()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the fixed-token operations command boundary."""

    values = list(sys.argv[1:] if argv is None else argv)
    command = (
        values[0] if values and values[0] in {"capture", "verify-live"} else "capture"
    )
    try:
        return _main(values)
    except _CliUsageError:
        print("issue123-operations-usage-failed", file=sys.stderr)
        return 2
    except ImportError, OSError, EvidenceError, TypeError, ValueError:
        print(f"issue123-operations-{command}-failed", file=sys.stderr)
        return 2


def _cli(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(_cli())
