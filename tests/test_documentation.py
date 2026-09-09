"""Regression tests for executable development documentation."""

import re
from pathlib import Path

import pytest


@pytest.fixture(scope="class", autouse=True)
def documentation_files(request):
    project_root = Path(__file__).resolve().parents[1]
    request.cls.primary_documents = {
        name: (project_root / name).read_text()
        for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md")
    }
    request.cls.benchmarks_readme = (
        project_root / "benchmarks" / "README.md"
    ).read_text()
    request.cls.examples_readme = (project_root / "examples" / "README").read_text()


class TestDevelopmentDocumentation:

    @pytest.mark.parametrize("document", ("README.md", "CONTRIBUTING.md", "AGENTS.md"))
    def test_primary_documents_share_setup_test_and_build_commands(self, document):
        canonical_workflow = """uv python install 3.14
uv sync --locked --extra torch-cpu --extra hdf5
uv run --no-sync python -m pytest -v
uv build"""
        assert canonical_workflow in self.primary_documents[document]

    def test_examples_use_the_uv_environment_without_mpi_requirements(self):
        readme = self.primary_documents["README.md"]
        assert ("Torch runtime") in (readme)
        assert ("torchrun") in (readme)
        assert ("mpiexec") not in (readme)
        assert ("uv run --no-sync python examples/air2d.py --no-plot --steps 2") in (
            readme
        )
        assert ("uv run --no-sync python examples/<example file name>") in (
            self.examples_readme
        )
        assert ("$ python examples/") not in (self.examples_readme)

    def test_lock_migration_and_pure_torch_prerequisites_are_documented(self):
        readme = self.primary_documents["README.md"]
        contributing = self.primary_documents["CONTRIBUTING.md"]
        agents = self.primary_documents["AGENTS.md"]
        assert ("uv lock --upgrade") in (readme)
        assert ("PEP 735") in (readme)
        assert ("PyTorch `>=2.13,<2.14`") in (readme)
        assert ("py3-none-any") in (readme)
        assert ("No compiler, SWIG, Cython, OpenMP runtime, or MPI") in (contributing)
        assert ("pure-Python PyTorch package") in (agents)
        assert ("tests/test_packaging.py") in (contributing)
        assert ("uv lock --upgrade") in (agents)

    def test_issue123_global_correctness_topology_is_mandatory(self):
        readme = " ".join(self.benchmarks_readme.split())
        assert (
            "one identical ordered 34-reference set across CPU, CUDA eager, and CUDA "
            "graph"
        ) in (readme)
        assert (
            "every reference record identical by case, path, SHA-256, size, media "
            "type, and payload identity"
        ) in (readme)
        assert (
            "CPU, CUDA eager, and CUDA graph must each use a distinct 34-candidate set"
        ) in (readme)
        assert (
            "All three candidate sets must be mutually disjoint and disjoint from "
            "the shared references by path, digest, and payload identity"
        ) in (readme)
        assert (
            "204 descriptor occurrences therefore resolve to exactly 34 shared "
            "references plus 102 candidates, for 136 globally unique archives"
        ) in (readme)
        assert ("An operator may reuse") not in (readme)

    @pytest.mark.parametrize(
        "deferred_item_case",
        range(6),
        ids=(
            "asset-generation",
            "byte-schema",
            "immutable-release",
            "release-metadata",
            "authority-reopen",
            "publication-cutover",
        ),
    )
    def test_issue123_authority_clis_and_fixed_point_are_documented(
        self, deferred_item_case
    ):
        self._check_issue123_authority_clis_and_fixed_point_are_documented_phase(
            phase="deferred_items", deferred_item_case=deferred_item_case
        )

    @pytest.mark.parametrize(
        "forbidden_case",
        range(5),
        ids=(
            "completion-claim",
            "authority-waiver",
            "runtime-fallback",
            "final-acceptance",
            "issue-completion",
        ),
    )
    def test_issue123_authority_clis_and_fixed_point_are_documented_forbidden_literals(
        self, forbidden_case
    ):
        self._check_issue123_authority_clis_and_fixed_point_are_documented_phase(
            phase="forbidden_literals", forbidden_case=forbidden_case
        )

    def _check_issue123_authority_clis_and_fixed_point_are_documented_phase(
        self, *, phase, deferred_item_case=None, forbidden_case=None
    ):
        readme = self.benchmarks_readme
        normalized = " ".join(readme.split())
        plain = normalized.replace("`", "")
        assert ("python -m benchmarks.issue123_publication prepare") in (readme)
        assert ("python -m benchmarks.issue123_publication finalize") in (readme)
        assert ("--completion-index") in (readme)
        assert ("--private-openings-output") in (readme)
        assert ("issue123-publication-prepare-ok") in (readme)
        assert ("issue123-publication-finalize-ok") in (readme)
        assert (
            "O0/B0 reopen -> authorized two-line acknowledgment -> O1 recapture "
            "-> B1 reopen -> offline evaluate -> live verify"
        ) in (normalized)
        assert ("python -m benchmarks.issue123_completion record-reopen") in (readme)
        assert ("--reopened-index") in (readme)
        assert ("--private-openings") in (readme)
        assert ("--pre-ack-bundle-reopen-receipt") in (readme)
        assert ("--final-bundle-reopen-receipt") in (readme)
        assert ("--baseline-authority live-release") in (readme)
        assert ("Only `--baseline-authority live-release` is implemented") in (readme)
        assert ("Production binding readiness is currently fail-closed") in (readme)
        assert (
            "The final-SHA publication and release-dependent operations steps in "
            "this section are the six-item chain governed by the"
        ) in (normalized)
        assert (
            "[Recommendation A OWNER amendment](https://github.com/ruddyscent/"
            "gmes/issues/123#issuecomment-5523144396)"
        ) in (readme)
        assert (
            "deferred to open [#169](https://github.com/ruddyscent/gmes/issues/169)"
        ) in (normalized)
        assert ("production literal binding registry remains empty") in (normalized)
        assert (
            "commands below document the executable fail-closed interface; they do "
            "not claim present production readiness"
        ) in (normalized)
        follow_up = "#169"
        follow_up_url = "https://github.com/ruddyscent/gmes/issues/169"
        assert (
            f"Production evaluator-binding authority is intentionally deferred to "
            f"[{follow_up}]({follow_up_url})"
        ) in (normalized)
        assert (follow_up_url) in (readme)
        assert ("That follow-up owns the six deferred items") in (normalized)
        if phase == "deferred_items":
            deferred_item_case_values = tuple(
                (
                    "production-bound final-SHA generation of the four public assets",
                    "actual-public-byte schema, cardinality, commitment, digest",
                    "final-SHA immutable release, four OWNER uploads",
                    "release link/tag/ID/URL/size/hash fields",
                    "release-dependent O0/B0/ack/O1/B1",
                    "production publication, cutover, a nonempty production registry",
                )
            )
            assert len(deferred_item_case_values) == 6
            deferred_item = deferred_item_case_values[deferred_item_case]
            assert (deferred_item) in (normalized)
        assert (
            "These six items are deferred, unperformed, unsatisfied, still required, "
            "and owned by open #169"
        ) in (normalized)
        assert ("These deferrals change no runtime authority") in (normalized)
        assert ("CODE_OWNED_LITERAL_TARGET_BINDINGS remains empty") in (plain)
        assert (
            "completion live verification cannot set final_acceptance or "
            "issue_completion_satisfied through this path"
        ) in (plain)
        assert (
            "Issue #123 may accept and close for its technical work only after every "
            "retained, non-deferred performance, correctness, evidence, operations, "
            "privacy, security, CI, CodeQL, review, and clean-candidate gate passes"
        ) in (normalized)
        assert (
            "technical acceptance remains distinct from the deferred "
            "production-publication chain"
        ) in (normalized)
        assert (
            "Its closure record must state that all six items above remain deferred, "
            "unperformed, unsatisfied, and still required by open #169"
        ) in (normalized)
        assert ("none is a #123 closure prerequisite") in (normalized)
        assert ("The technical release must precede the issue #123 amendment") not in (
            normalized
        )
        assert (
            "Production evaluator authority remains deferred, and production "
            "publication and cutover remain blocked on #169"
        ) in (normalized)
        boundary_start = normalized.index(
            "Production binding readiness is currently fail-closed"
        )
        boundary = normalized[
            boundary_start : normalized.index("~~~sh", boundary_start)
        ]
        if phase == "forbidden_literals":
            forbidden_case_values = tuple(
                (
                    "M1 is complete",
                    "binding authority is waived",
                    "runtime fallback",
                    "final_acceptance may be true",
                    "issue_completion_satisfied may be true",
                )
            )
            assert len(forbidden_case_values) == 5
            forbidden = forbidden_case_values[forbidden_case]
            assert (forbidden) not in (boundary)
        assert ("complete canonical O0/O1 response projection") in (normalized)
        assert (
            "Both protected B1 roots come from that retained authenticated lease"
        ) in (normalized)
        assert (
            "documented `main(argv)` boundary and the module process entry use the "
            "same fixed, path-free success and failure tokens"
        ) in (normalized)
        assert ("atomic no-replace link") in (normalized)
        assert ("sole authority linearization point") in (normalized)
        assert (
            "exact two-asset order, and retains the two downloaded baseline-v3 file "
            "identities and exact bytes"
        ) in (normalized)
        assert ("--post-bundle-expectation") not in (readme)
        assert ("it has no command-line wrapper") not in (readme)
        assert ("no receipt-input option exists") not in (readme)

    def test_issue123_operations_capture_authentication_is_direct_api_first(self):
        readme = self.benchmarks_readme
        start = readme.index("Capture operations only after that receipt")
        stop = readme.index("The schema-v2 producer", start)
        capture_section = readme[start:stop]
        assert ("gh auth status") not in (capture_section)
        assert ("repeats the authenticated") not in (capture_section)
        assert (
            "`capture` sends each fixed `gh api --hostname github.com` request "
            "directly"
        ) in (capture_section)
        assert ("rerun\n`capture` with a nonexistent output directory") in (
            capture_section
        )

    @pytest.mark.parametrize(
        "forbidden_case",
        range(4),
        ids=("host-salt", "salt-length", "linux-home", "macos-home"),
    )
    def test_issue123_public_privacy_contract_is_documented_without_literals(
        self, forbidden_case
    ):
        self._check_issue123_public_privacy_contract_is_documented_without_literals_phase(
            phase="forbidden_literals", forbidden_case=forbidden_case
        )

    @pytest.mark.parametrize(
        "variant_extra_reference_case",
        range(10),
        ids=(
            "anchor",
            "api",
            "api-relative",
            "api-command-token",
            "prose",
            "prose-colon",
            "prose-hash",
            "duplicate",
            "prefix",
            "suffix",
        ),
    )
    def test_issue123_public_privacy_contract_is_documented_without_literals_variants(
        self, variant_extra_reference_case
    ):
        self._check_issue123_public_privacy_contract_is_documented_without_literals_phase(
            phase="variants", variant_extra_reference_case=variant_extra_reference_case
        )

    def _check_issue123_public_privacy_contract_is_documented_without_literals_phase(
        self, *, phase, forbidden_case=None, variant_extra_reference_case=None
    ):
        readme = self.benchmarks_readme
        normalized = " ".join(readme.split())
        assert (
            "Public documentation and artifacts disclose only that safe commitment"
        ) in (normalized)
        assert (
            "Salts, openings, raw arrays, private paths, host/device identities, "
            "and source identities exist only in protected inputs or memory"
        ) in (normalized)
        assert ("serialize no private paths, raw identities, keys, or openings") in (
            normalized
        )
        if phase == "forbidden_literals":
            forbidden_case_values = tuple(
                (
                    "BASELINE_V3_HOST_SALT",
                    "32-byte-salted",
                    "/home/",
                    "/Users/",
                )
            )
            assert len(forbidden_case_values) == 4
            forbidden = forbidden_case_values[forbidden_case]
            assert (forbidden) not in (readme)
        allowed_comment_url = (
            "https://github.com/ruddyscent/gmes/issues/123#issuecomment-5523144396"
        )
        comment_ref_pattern = (
            r"[^\s()<>\[\]]*issuecomment-\d+[^\s()<>\[\]]*"
            r"|[^\s()<>\[\]]*/issues/comments/\d+[^\s()<>\[\]]*"
            r"|\bissue[ \t]+comment(?:[ \t]+|[ \t]*[:#][ \t]*)\d{6,}\b"
        )
        comment_refs = re.findall(
            comment_ref_pattern,
            readme,
            flags=re.IGNORECASE,
        )
        assert (comment_refs) == ([allowed_comment_url])
        if phase == "variants":
            variant_extra_reference_case_values = tuple(
                (
                    (
                        "anchor",
                        "https://github.com/synthetic-owner/synthetic-repository/"
                        "issues/123#issuecomment-111111",
                    ),
                    (
                        "api",
                        "https://api.github.com/repos/synthetic-owner/"
                        "synthetic-repository/issues/comments/111111",
                    ),
                    ("api-relative", "/issues/comments/111111"),
                    (
                        "api-command-token",
                        "repos/synthetic-owner/synthetic-repository/"
                        "issues/comments/111111",
                    ),
                    ("prose", "Issue comment 111111"),
                    ("prose-colon", "Issue comment: 111111"),
                    ("prose-hash", "Issue comment #111111"),
                    ("duplicate", allowed_comment_url),
                    ("prefix", f"prefix{allowed_comment_url}"),
                    ("suffix", f"{allowed_comment_url}?copy=1"),
                )
            )
            assert len(variant_extra_reference_case_values) == 10
            variant, extra_reference = variant_extra_reference_case_values[
                variant_extra_reference_case
            ]
            assert (
                re.findall(
                    comment_ref_pattern,
                    f"{allowed_comment_url} {extra_reference}",
                    flags=re.IGNORECASE,
                )
            ) != ([allowed_comment_url])
        assert (
            re.findall(
                comment_ref_pattern,
                f"{allowed_comment_url} ordinary issue #123456",
                flags=re.IGNORECASE,
            )
        ) == ([allowed_comment_url])
        assert re.search(r"(?m)^[A-Z_]*COMMENT(?:_ID)?=\d{6,}$", readme) is None

    @pytest.mark.parametrize(
        "contract_case",
        range(7),
        ids=(
            "projection-v1",
            "bundle-v1",
            "completion-v2",
            "four-assets",
            "five-receipts",
            "operations-roles",
            "live-v3",
        ),
    )
    def test_issue123_authority_versions_and_cardinalities_are_pinned(
        self, contract_case
    ):
        normalized = " ".join(self.benchmarks_readme.split())
        contract_case_values = tuple(
            (
                "public projection/publication schema v1",
                "bundle specification v1",
                "completion index v2",
                "exactly four ordered public assets",
                "exactly five ordered runtime receipts",
                "exactly 22 operations roles",
                "completion live output and private operations live receipt advance to v3",
            )
        )
        assert len(contract_case_values) == 7
        contract = contract_case_values[contract_case]
        assert (contract) in (normalized)
