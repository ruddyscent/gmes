"""Regression tests for executable development documentation."""

import re
import unittest
from pathlib import Path


class DevelopmentDocumentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        cls.primary_documents = {
            name: (project_root / name).read_text()
            for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md")
        }
        cls.benchmarks_readme = (project_root / "benchmarks" / "README.md").read_text()
        cls.examples_readme = (project_root / "examples" / "README").read_text()

    def test_primary_documents_share_setup_test_and_build_commands(self):
        canonical_workflow = """uv python install 3.14
uv sync --locked --extra torch-cpu --extra hdf5
uv run --no-sync python -m unittest discover -v
uv build"""
        for name, contents in self.primary_documents.items():
            with self.subTest(document=name):
                self.assertIn(canonical_workflow, contents)

    def test_examples_and_mpi_use_the_uv_environment(self):
        readme = self.primary_documents["README.md"]
        self.assertIn("uv run --no-sync python examples/air2d.py", readme)
        self.assertIn("uv run --no-sync mpiexec", readme)
        self.assertIn(
            "uv run --no-sync python examples/<example file name>", self.examples_readme
        )
        self.assertIn("uv run --no-sync mpiexec", self.examples_readme)
        self.assertNotIn("$ python examples/", self.examples_readme)

    def test_lock_migration_and_native_prerequisites_are_documented(self):
        readme = self.primary_documents["README.md"]
        contributing = self.primary_documents["CONTRIBUTING.md"]
        agents = self.primary_documents["AGENTS.md"]
        self.assertIn("uv lock --upgrade", readme)
        self.assertIn("PEP 735", readme)
        self.assertIn("build-essential", readme)
        self.assertIn("xcode-select --install", readme)
        self.assertIn("libopenmpi-dev openmpi-bin", readme)
        self.assertIn("brew install open-mpi", readme)
        self.assertIn("contiguous-indexing fallback", readme)
        self.assertIn("tests/test_packaging.py", contributing)
        self.assertIn("uv lock --upgrade", agents)

    def test_issue123_global_correctness_topology_is_mandatory(self):
        readme = " ".join(self.benchmarks_readme.split())
        self.assertIn(
            "one identical ordered 34-reference set across CPU, CUDA eager, and CUDA "
            "graph",
            readme,
        )
        self.assertIn(
            "every reference record identical by case, path, SHA-256, size, media "
            "type, and payload identity",
            readme,
        )
        self.assertIn(
            "CPU, CUDA eager, and CUDA graph must each use a distinct 34-candidate set",
            readme,
        )
        self.assertIn(
            "All three candidate sets must be mutually disjoint and disjoint from "
            "the shared references by path, digest, and payload identity",
            readme,
        )
        self.assertIn(
            "204 descriptor occurrences therefore resolve to exactly 34 shared "
            "references plus 102 candidates, for 136 globally unique archives",
            readme,
        )
        self.assertNotIn("An operator may reuse", readme)

    def test_issue123_authority_clis_and_fixed_point_are_documented(self):
        readme = self.benchmarks_readme
        normalized = " ".join(readme.split())
        plain = normalized.replace("`", "")
        self.assertIn("python -m benchmarks.issue123_publication prepare", readme)
        self.assertIn("python -m benchmarks.issue123_publication finalize", readme)
        self.assertIn("--completion-index", readme)
        self.assertIn("--private-openings-output", readme)
        self.assertIn("issue123-publication-prepare-ok", readme)
        self.assertIn("issue123-publication-finalize-ok", readme)
        self.assertIn(
            "O0/B0 reopen -> authorized two-line acknowledgment -> O1 recapture "
            "-> B1 reopen -> offline evaluate -> live verify",
            normalized,
        )
        self.assertIn("python -m benchmarks.issue123_completion record-reopen", readme)
        self.assertIn("--reopened-index", readme)
        self.assertIn("--private-openings", readme)
        self.assertIn("--pre-ack-bundle-reopen-receipt", readme)
        self.assertIn("--final-bundle-reopen-receipt", readme)
        self.assertIn("--baseline-authority live-release", readme)
        self.assertIn("Only `--baseline-authority live-release` is implemented", readme)
        self.assertIn("Production binding readiness is currently fail-closed", readme)
        self.assertIn(
            "The final-SHA publication and release-dependent operations steps in "
            "this section are the six-item chain governed by the",
            normalized,
        )
        self.assertIn(
            "[Recommendation A OWNER amendment](https://github.com/ruddyscent/"
            "gmes/issues/123#issuecomment-5523144396)",
            readme,
        )
        self.assertIn(
            "deferred to open [#169](https://github.com/ruddyscent/gmes/issues/169)",
            normalized,
        )
        self.assertIn("production literal binding registry remains empty", normalized)
        self.assertIn(
            "commands below document the executable fail-closed interface; they do "
            "not claim present production readiness",
            normalized,
        )
        follow_up = "#169"
        follow_up_url = "https://github.com/ruddyscent/gmes/issues/169"
        self.assertIn(
            f"Production evaluator-binding authority is intentionally deferred to "
            f"[{follow_up}]({follow_up_url})",
            normalized,
        )
        self.assertIn(follow_up_url, readme)
        self.assertIn("That follow-up owns the six deferred items", normalized)
        for deferred_item in (
            "production-bound final-SHA generation of the four public assets",
            "actual-public-byte schema, cardinality, commitment, digest",
            "final-SHA immutable release, four OWNER uploads",
            "release link/tag/ID/URL/size/hash fields",
            "release-dependent O0/B0/ack/O1/B1",
            "production publication, cutover, a nonempty production registry",
        ):
            with self.subTest(deferred_item=deferred_item):
                self.assertIn(deferred_item, normalized)
        self.assertIn(
            "These six items are deferred, unperformed, unsatisfied, still required, "
            "and owned by open #169",
            normalized,
        )
        self.assertIn("These deferrals change no runtime authority", normalized)
        self.assertIn("CODE_OWNED_LITERAL_TARGET_BINDINGS remains empty", plain)
        self.assertIn(
            "completion live verification cannot set final_acceptance or "
            "issue_completion_satisfied through this path",
            plain,
        )
        self.assertIn(
            "Issue #123 may accept and close for its technical work only after every "
            "retained, non-deferred performance, correctness, evidence, operations, "
            "privacy, security, CI, CodeQL, review, and clean-candidate gate passes",
            normalized,
        )
        self.assertIn(
            "technical acceptance remains distinct from the deferred "
            "production-publication chain",
            normalized,
        )
        self.assertIn(
            "Its closure record must state that all six items above remain deferred, "
            "unperformed, unsatisfied, and still required by open #169",
            normalized,
        )
        self.assertIn("none is a #123 closure prerequisite", normalized)
        self.assertNotIn(
            "The technical release must precede the issue #123 amendment",
            normalized,
        )
        self.assertIn(
            "Production evaluator authority remains deferred, and production "
            "publication and cutover remain blocked on #169",
            normalized,
        )
        boundary_start = normalized.index(
            "Production binding readiness is currently fail-closed"
        )
        boundary = normalized[
            boundary_start : normalized.index("~~~sh", boundary_start)
        ]
        for forbidden in (
            "M1 is complete",
            "binding authority is waived",
            "runtime fallback",
            "final_acceptance may be true",
            "issue_completion_satisfied may be true",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, boundary)
        self.assertIn("complete canonical O0/O1 response projection", normalized)
        self.assertIn(
            "Both protected B1 roots come from that retained authenticated lease",
            normalized,
        )
        self.assertIn(
            "documented `main(argv)` boundary and the module process entry use the "
            "same fixed, path-free success and failure tokens",
            normalized,
        )
        self.assertIn("atomic no-replace link", normalized)
        self.assertIn("sole authority linearization point", normalized)
        self.assertIn(
            "exact two-asset order, and retains the two downloaded baseline-v3 file "
            "identities and exact bytes",
            normalized,
        )
        self.assertNotIn("--post-bundle-expectation", readme)
        self.assertNotIn("it has no command-line wrapper", readme)
        self.assertNotIn("no receipt-input option exists", readme)

    def test_issue123_operations_capture_authentication_is_direct_api_first(self):
        readme = self.benchmarks_readme
        start = readme.index("Capture operations only after that receipt")
        stop = readme.index("The schema-v2 producer", start)
        capture_section = readme[start:stop]
        self.assertNotIn("gh auth status", capture_section)
        self.assertNotIn("repeats the authenticated", capture_section)
        self.assertIn(
            "`capture` sends each fixed `gh api --hostname github.com` request "
            "directly",
            capture_section,
        )
        self.assertIn(
            "rerun\n`capture` with a nonexistent output directory", capture_section
        )

    def test_issue123_public_privacy_contract_is_documented_without_literals(self):
        readme = self.benchmarks_readme
        normalized = " ".join(readme.split())
        self.assertIn(
            "Public documentation and artifacts disclose only that safe commitment",
            normalized,
        )
        self.assertIn(
            "Salts, openings, raw arrays, private paths, host/device identities, "
            "and source identities exist only in protected inputs or memory",
            normalized,
        )
        self.assertIn(
            "serialize no private paths, raw identities, keys, or openings", normalized
        )
        for forbidden in (
            "BASELINE_V3_HOST_SALT",
            "32-byte-salted",
            "/home/",
            "/Users/",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, readme)
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
        self.assertEqual(comment_refs, [allowed_comment_url])
        for variant, extra_reference in (
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
                "repos/synthetic-owner/synthetic-repository/" "issues/comments/111111",
            ),
            ("prose", "Issue comment 111111"),
            ("prose-colon", "Issue comment: 111111"),
            ("prose-hash", "Issue comment #111111"),
            ("duplicate", allowed_comment_url),
            ("prefix", f"prefix{allowed_comment_url}"),
            ("suffix", f"{allowed_comment_url}?copy=1"),
        ):
            with self.subTest(comment_reference_variant=variant):
                self.assertNotEqual(
                    re.findall(
                        comment_ref_pattern,
                        f"{allowed_comment_url} {extra_reference}",
                        flags=re.IGNORECASE,
                    ),
                    [allowed_comment_url],
                )
        self.assertEqual(
            re.findall(
                comment_ref_pattern,
                f"{allowed_comment_url} ordinary issue #123456",
                flags=re.IGNORECASE,
            ),
            [allowed_comment_url],
        )
        self.assertNotRegex(
            readme,
            r"(?m)^[A-Z_]*COMMENT(?:_ID)?=\d{6,}$",
        )

    def test_issue123_authority_versions_and_cardinalities_are_pinned(self):
        normalized = " ".join(self.benchmarks_readme.split())
        for contract in (
            "public projection/publication schema v1",
            "bundle specification v1",
            "completion index v2",
            "exactly four ordered public assets",
            "exactly five ordered runtime receipts",
            "exactly 22 operations roles",
            "completion live output and private operations live receipt advance to v3",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, normalized)


if __name__ == "__main__":
    unittest.main()
