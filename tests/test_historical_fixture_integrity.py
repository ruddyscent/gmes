"""Preserve the byte-pinned V5.4 vendored evidence snapshot."""

import ast
import hashlib
import json
import re
import tomllib
import unittest
from pathlib import Path

from benchmarks import historical_probes

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "issue124"

# These are fixed C1 identities copied from the accepted frozen repository,
# never derived from candidate contents or numerical probe values.
V5_4_TRUST = {
    "expected_manifest_sha256": "cce4820ccc0e8050db6baf47d18fe646ee2c59bb64c89c0279cd38be1ba17d41",
    "expected_manifest_bytes": 28865,
    "expected_fixture_sha256": "9fa7d54d63c4e0c6bceca37ed2c9c392d398f4ebe5ae94748de3b25b73b6b289",
    "expected_fixture_bytes": 202209,
}
STYLE_EXEMPT_IDENTITIES = {
    "tests/fixtures/issue124/generator.py": (
        32587,
        "e5c678e8b3cce853ece2c6548619b46f6b6bb5ffa744db20e1375ef325d32e55",
    ),
    "tests/fixtures/issue124/historical_probe_loader.py": (
        52326,
        "4a6c7e0dd7da0467db7ce66b6b48895ed7c9c124655ba41ac4d193d68117a3a4",
    ),
    "tests/fixtures/issue124/package_integration_assertions.py": (
        5322,
        "8359d17614d89e24243541591d59d075103bf3df5e95dd86aa6cf536d254db60",
    ),
    "tests/fixtures/issue124/self_test.py": (
        47386,
        "805cf5a85ac8db23d9d1248355e071d9cee990ad6c6154c44fbd3e65fcdcd7f7",
    ),
    "benchmarks/historical_probes.py": (
        52326,
        "4a6c7e0dd7da0467db7ce66b6b48895ed7c9c124655ba41ac4d193d68117a3a4",
    ),
}
PROFILE_FILES = {
    "profiles/canonical-dm2-0766dbf93288.json",
    "profiles/long-dummy-345c284da712.json",
    "profiles/long-gaussian-2b721c7d152f.json",
    "profiles/long-tfsf-e6289d035614.json",
    "profiles/small-dcp-d7781a0517f3.json",
    "profiles/small-drude1-e00d2eae1726.json",
    "profiles/small-gaussian-095e5d4cdf76.json",
    "profiles/small-index-matrix-5f1106504c56.json",
    "profiles/small-initial-eb42a7849f3b.json",
    "profiles/small-main-fields-5dec520153b5.json",
    "profiles/small-material-matrix-5355ca306f08.json",
    "profiles/small-mixed-26ebbf7ce60d.json",
    "profiles/small-stability-c3d3ea9a05fc.json",
    "profiles/small-tfsf-808b0f4710a6.json",
}
MANIFEST_CLOSURE = {
    "generator.py",
    "historical_probe_loader.py",
    "package_integration_assertions.py",
    "README.md",
    "REPRODUCTION.md",
    "schema.json",
    "inputs/native_oracle_workloads.json",
    "pre-cutover-native-numeric-probes-v5-4.json.gz",
    *PROFILE_FILES,
}
COMPLETE_CLOSURE = MANIFEST_CLOSURE | {
    "BUNDLE-MANIFEST.json",
    "BUNDLE-MANIFEST.sha256",
    "pre-cutover-native-numeric-probes-v5-4.sha256",
    "self_test.py",
}


class HistoricalFixtureIntegrityTest(unittest.TestCase):
    @staticmethod
    def _identity(path):
        data = path.read_bytes()
        return len(data), hashlib.sha256(data).hexdigest()

    def test_literal_v54_fixture_and_manifest_anchors(self):
        self.assertEqual(
            self._identity(FIXTURE_ROOT / "BUNDLE-MANIFEST.json"),
            (
                V5_4_TRUST["expected_manifest_bytes"],
                V5_4_TRUST["expected_manifest_sha256"],
            ),
        )
        self.assertEqual(
            self._identity(
                FIXTURE_ROOT / "pre-cutover-native-numeric-probes-v5-4.json.gz"
            ),
            (
                V5_4_TRUST["expected_fixture_bytes"],
                V5_4_TRUST["expected_fixture_sha256"],
            ),
        )
        self.assertEqual(
            (FIXTURE_ROOT / "BUNDLE-MANIFEST.sha256").read_text(),
            f"{V5_4_TRUST['expected_manifest_sha256']}  BUNDLE-MANIFEST.json\n",
        )

    def test_torch_correctness_retains_the_same_literal_trust_anchor(self):
        module = ast.parse((ROOT / "tests" / "test_torch_correctness.py").read_text())
        anchor = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PROBE_TRUST"
                for target in node.targets
            )
        )
        self.assertEqual(ast.literal_eval(anchor.value), V5_4_TRUST)

    def test_loader_validates_the_complete_immutable_closure(self):
        side = json.loads((FIXTURE_ROOT / "BUNDLE-MANIFEST.json").read_text())
        referenced = {
            side["generator"]["file"],
            side["generator"]["loader_file"],
            side["inputs"]["source_manifest"]["file"],
            side["fixture"]["file"],
            *(item["file"] for item in side["support"].values()),
            *(item["file"] for item in side["profiles"].values()),
        }
        self.assertEqual(referenced, MANIFEST_CLOSURE)
        files = {
            path.relative_to(FIXTURE_ROOT).as_posix()
            for path in FIXTURE_ROOT.rglob("*")
            if path.is_file()
        }
        self.assertEqual(files, COMPLETE_CLOSURE)
        bundle = historical_probes.load_bundle(FIXTURE_ROOT, **V5_4_TRUST)
        self.assertEqual(
            bundle.side["generator"]["loader_sha256"],
            STYLE_EXEMPT_IDENTITIES["benchmarks/historical_probes.py"][1],
        )
        self.assertEqual(
            set(bundle.profiles),
            {
                path.removeprefix("profiles/").removesuffix(".json")
                for path in PROFILE_FILES
            },
        )

    def test_frozen_style_exempt_files_and_loader_mirror_are_original_bytes(self):
        for relative, expected in STYLE_EXEMPT_IDENTITIES.items():
            with self.subTest(relative=relative):
                self.assertEqual(self._identity(ROOT / relative), expected)
        self.assertEqual(
            (ROOT / "benchmarks" / "historical_probes.py").read_bytes(),
            (FIXTURE_ROOT / "historical_probe_loader.py").read_bytes(),
        )
        self.assertNotIn(
            "self_test.py",
            json.loads((FIXTURE_ROOT / "BUNDLE-MANIFEST.json").read_text())["support"],
        )

    def test_formatter_exemption_matches_only_the_five_pinned_paths(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        expected = set(STYLE_EXEMPT_IDENTITIES)
        black = re.compile(config["tool"]["black"]["force-exclude"], re.VERBOSE)
        self.assertEqual(set(config["tool"]["isort"]["extend_skip"]), expected)
        for path in expected:
            self.assertIsNotNone(black.search(f"/{path}"), path)
        for path in (
            "tests/fixtures/issue124/README.md",
            "tests/fixtures/issue124/unpinned.py",
            "benchmarks/torch_correctness.py",
        ):
            self.assertIsNone(black.search(f"/{path}"), path)


if __name__ == "__main__":
    unittest.main()
