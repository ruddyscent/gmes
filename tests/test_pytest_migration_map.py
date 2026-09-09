"""Validate the durable issue-179 unittest-to-pytest migration inventory."""

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.pytest_failure_collector import FailureCollector

MAP_PATH = Path(__file__).with_name("pytest-migration-map.json")
RELATIONS_DIGEST_PATH = Path(__file__).with_name(
    "pytest-migration-map-relations.sha256"
)
BASE_REVISION = "8c1ea5aba7b11d96bd85541f95241c0539b280f9"
BASELINE_LOG_SHA256 = "c6fb3f7c2f1c61cb8544d20fc80de481dbe6a6c999dc703688c6d0d0857b2234"
# Independently verified against BASE_REVISION: all 913 method identities and
# 328 subTest call-site identities. Keep this anchor outside the editable map;
# validation must also work in source distributions that do not contain .git.
BASELINE_IDENTITIES_SHA256 = (
    "d6a9c2fb6fce47f105239c83b69ea973d0ac8e49192db79d68fab8861b16b7bb"
)
MAP_VALIDATION_NODE = "tests/test_pytest_migration_map.py::test_pytest_migration_map_is_complete_and_auditable"
FAILURE_COLLECTOR_NODE = "tests/test_pytest_migration_map.py::test_failure_collector_preserves_sequential_continuation"
MAP_VALIDATION_NODES = {MAP_VALIDATION_NODE, FAILURE_COLLECTOR_NODE}
# The sidecar gives reviewers a compact relation-only diff; this independent
# constant prevents the editable JSON and sidecar from authenticating each other.
MAPPING_RELATIONS_SHA256 = (
    "ceda6bd42b92433a531ea2101058758bb6a3d4ed2976351e63b5c5dbce77af56"
)


def _normalized_current_test_source(source_node):
    path, class_name, method = source_node.split("::")[:3]
    method = method.split("[", 1)[0]
    source = (MAP_PATH.parent.parent / path).read_text()
    tree = ast.parse(source, filename=path)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    return " ".join(ast.get_source_segment(source, function).split())


def _normalized_current_test_decorators(node_pattern):
    path, class_name, method = node_pattern.split("::")
    source = (MAP_PATH.parent.parent / path).read_text()
    tree = ast.parse(source, filename=path)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    return " ".join(
        " ".join(ast.get_source_segment(source, decorator).split())
        for decorator in function.decorator_list
    )


def _current_converted_test_patterns():
    patterns = set()
    for path in sorted(MAP_PATH.parent.glob("test_*.py")):
        relative = path.relative_to(MAP_PATH.parent.parent).as_posix()
        tree = ast.parse(path.read_text(), filename=relative)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                patterns.add(f"{relative}::{node.name}")
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                patterns.update(
                    f"{relative}::{node.name}::{function.name}"
                    for function in node.body
                    if isinstance(function, ast.FunctionDef)
                    and function.name.startswith("test_")
                )
    return patterns


def _collected_test_counts():
    # A separate collection-only process includes the full suite even when this
    # validator was selected alone. It never executes this test recursively.
    environment = dict(os.environ)
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--color=no", "tests"],
        cwd=MAP_PATH.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    assert nodes, result.stdout
    assert len(nodes) == len(set(nodes)), "duplicate collected node IDs"
    return Counter(node.split("[", 1)[0] for node in nodes)


def _derived_module_summary(methods, subtests):
    summaries = {}
    for item in methods:
        path = item["old"].split("::", 1)[0]
        summary = summaries.setdefault(path, Counter())
        summary["old_methods"] += 1
        summary["methods_mapped"] += 1
    for item in subtests:
        path = item["old"].split("::", 1)[0]
        summary = summaries.setdefault(path, Counter())
        summary["old_subtests"] += 1
        summary[f"subtests_{item['kind']}"] += 1
    return [
        {"path": path, **dict(sorted(summary.items()))}
        for path, summary in sorted(summaries.items())
    ]


def _normalized_mapping_relations(migration_map):
    def normalized(items, keys):
        return [
            {key: item.get(key) for key in keys}
            for item in sorted(items, key=lambda item: item["old"])
        ]

    return {
        "helper_owned_subtests": normalized(
            migration_map["helper_owned_subtests"],
            ("old", "new", "parameter_tokens"),
        ),
        "methods": normalized(migration_map["method_mappings"], ("old", "line", "new")),
        "semantic_audits": {
            "methods": normalized(
                migration_map["semantic_audits"]["methods"], ("old", "new")
            ),
            "subtests": normalized(
                migration_map["semantic_audits"]["subtests"],
                ("old", "node_pattern", "count", "parameter_tokens"),
            ),
        },
        "subtests": normalized(
            migration_map["subtest_mappings"],
            ("old", "line", "args", "kind", "new", "reason"),
        ),
    }


def _mapping_relations_digest(migration_map):
    relations = _normalized_mapping_relations(migration_map)
    encoded = json.dumps(
        relations, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _assert_mapping_relations_digest(migration_map):
    digest_file = f"{MAPPING_RELATIONS_SHA256}  pytest-migration-map.relations.v1\n"
    assert RELATIONS_DIGEST_PATH.read_text() == digest_file
    assert _mapping_relations_digest(migration_map) == MAPPING_RELATIONS_SHA256


def _conventional_method_pattern(old):
    path, member = old.split("::")
    class_name, method = member.rsplit(".", 1)
    stem = class_name.removesuffix("Tests").removesuffix("Test")
    return f"{path}::Test{stem}::{method}"


def _assert_mapping_tamper_regressions(migration_map):
    methods = copy.deepcopy(migration_map)
    methods["method_mappings"][0]["new"], methods["method_mappings"][1]["new"] = (
        methods["method_mappings"][1]["new"],
        methods["method_mappings"][0]["new"],
    )
    with pytest.raises(AssertionError):
        _assert_mapping_relations_digest(methods)

    subtests = copy.deepcopy(migration_map)
    subtests["subtest_mappings"][0]["new"], subtests["subtest_mappings"][1]["new"] = (
        subtests["subtest_mappings"][1]["new"],
        subtests["subtest_mappings"][0]["new"],
    )
    with pytest.raises(AssertionError):
        _assert_mapping_relations_digest(subtests)

    phases = copy.deepcopy(migration_map)
    by_parent = {}
    for subtest in phases["subtest_mappings"]:
        by_parent.setdefault(subtest["old"].rsplit("::subTest@", 1)[0], []).append(
            subtest
        )
    first, second = next(
        pair
        for pair in by_parent.values()
        if len(pair) > 1 and pair[0]["args"] != pair[1]["args"]
    )
    first["args"] = second["args"]
    first["new"] = second["new"]
    with pytest.raises(AssertionError):
        _assert_mapping_relations_digest(phases)

    lines = copy.deepcopy(migration_map)
    original_line = lines["method_mappings"][0]["line"]
    lines["method_mappings"][0]["line"] = 1 if original_line != 1 else 2
    with pytest.raises(AssertionError):
        _assert_mapping_relations_digest(lines)


def test_pytest_migration_map_is_complete_and_auditable():
    migration_map = json.loads(MAP_PATH.read_text())

    assert migration_map["schema_version"] == 4
    assert migration_map["base_revision"] == BASE_REVISION
    assert migration_map["baseline"] == {
        "log_sha256": BASELINE_LOG_SHA256,
        "passed": 896,
        "skip_breakdown": {
            "cuda_unavailable": 11,
            "issue169_quarantine": 3,
            "pinned_cuda_python_required": 2,
            "plot_extra_missing": 1,
        },
        "skipped": 17,
        "total": 913,
    }
    assert migration_map["old_counts"] == {"methods": 913, "subtests": 328}
    assert set(migration_map["classification_counts"]) == {
        "parametrized",
        "retained-sequential",
    }
    assert all(not gap for gap in migration_map["gaps"].values())

    methods = migration_map["method_mappings"]
    subtests = migration_map["subtest_mappings"]
    assert len(methods) == 913
    assert len(subtests) == 328
    assert len({item["old"] for item in methods}) == len(methods)
    assert len({item["old"] for item in subtests}) == len(subtests)
    baseline_identities = {
        "methods": sorted(item["old"] for item in methods),
        "subtests": sorted(item["old"] for item in subtests),
    }
    identities_json = json.dumps(
        baseline_identities, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    assert (
        hashlib.sha256(identities_json.encode("utf-8")).hexdigest()
        == BASELINE_IDENTITIES_SHA256
    ), "mapped old identities do not match the immutable baseline inventory"
    assert migration_map["classification_counts"] == dict(
        Counter(item["kind"] for item in subtests)
    )
    assert migration_map["module_summary"] == _derived_module_summary(methods, subtests)
    _assert_mapping_relations_digest(migration_map)
    _assert_mapping_tamper_regressions(migration_map)
    collected_counts = _collected_test_counts()
    assert migration_map["current_collection"]["node_count"] == sum(
        collected_counts.values()
    )

    for item in [*methods, *subtests]:
        assert item["line"] > 0
        assert item["new"]
        assert all(
            node["node_pattern"].startswith("tests/") and node["count"] > 0
            for node in item["new"]
        )
        patterns = [node["node_pattern"] for node in item["new"]]
        assert len(patterns) == len(set(patterns)), item["old"]
        for node in item["new"]:
            assert node["node_pattern"] in collected_counts, node
            assert type(node["count"]) is int
            assert node["count"] == collected_counts[node["node_pattern"]], node

    mapped_patterns = {node["node_pattern"] for item in methods for node in item["new"]}
    mapped_occurrences = Counter(
        node["node_pattern"] for item in methods for node in item["new"]
    )
    assert all(count == 1 for count in mapped_occurrences.values()), mapped_occurrences
    assert migration_map["coverage_exclusions"] == [
        {
            "node_pattern": FAILURE_COLLECTOR_NODE,
            "reason": "tests the migration-only failure collector",
        },
        {
            "node_pattern": MAP_VALIDATION_NODE,
            "reason": "validates the migration map itself",
        },
    ]
    expected_patterns = mapped_patterns | MAP_VALIDATION_NODES
    assert _current_converted_test_patterns() == expected_patterns
    assert set(collected_counts) == expected_patterns
    assert all(collected_counts[node] == 1 for node in MAP_VALIDATION_NODES)

    retained = [item for item in subtests if item["kind"] == "retained-sequential"]
    assert (
        len(retained) == migration_map["classification_counts"]["retained-sequential"]
    )
    for item in retained:
        reason = item["reason"]
        assert reason["category"] in {
            "cross-case comparison",
            "failure aggregation",
            "ordered control",
            "shared evolving state",
            "shared fixture/artifact",
        }
        assert reason["source_node"].startswith("tests/")
        assert reason["evidence_token"]
        assert reason["evidence_token"] in _normalized_current_test_source(
            reason["source_node"]
        )

    subtests_by_old = {item["old"]: item for item in subtests}
    methods_by_old = {item["old"]: item for item in methods}
    helper_audits = {
        item["old"]: item for item in migration_map["helper_owned_subtests"]
    }
    helper_owned = {
        item["old"]
        for item in subtests
        if item["old"].rsplit("::subTest@", 1)[0] not in methods_by_old
    }
    assert set(helper_audits) == helper_owned
    assert len(helper_audits) == 2
    for old_id, audit in helper_audits.items():
        subtest = subtests_by_old[old_id]
        assert subtest["new"] == [
            {"node_pattern": node["node_pattern"], "count": node["count"]}
            for node in audit["new"]
        ]
        assert all(node["old_case_count"] > 0 for node in audit["new"])
        assert audit["parameter_tokens"]
        for node in audit["new"]:
            decorators = _normalized_current_test_decorators(node["node_pattern"])
            assert all(token in decorators for token in audit["parameter_tokens"])

    expected_helper_destinations = {
        "tests/test_docstrings.py::DocstringCoverageTest._assert_public_members_documented::subTest@131": {
            "tests/test_docstrings.py::TestDocstringCoverage::test_public_exports_have_docstrings": (
                483,
                344,
            ),
            "tests/test_docstrings.py::TestDocstringCoverage::test_supported_extension_hooks_have_docstrings": (
                14,
                11,
            ),
        },
        "tests/test_material_bulk.py::MaterialMappingFastPathTest.assert_planner_maps_match_pointwise_geometry::subTest@66": {
            "tests/test_material_bulk.py::TestMaterialMappingFastPath::test_batched_mapping_matches_pointwise_for_all_components_and_clipping": (
                12,
                12,
            ),
        },
    }
    assert {
        old_id: {
            node["node_pattern"]: (node["count"], node["old_case_count"])
            for node in audit["new"]
        }
        for old_id, audit in helper_audits.items()
    } == expected_helper_destinations

    for subtest in subtests:
        parent = subtest["old"].rsplit("::subTest@", 1)[0]
        if parent in methods_by_old:
            parent_targets = {
                node["node_pattern"] for node in methods_by_old[parent]["new"]
            }
            assert {node["node_pattern"] for node in subtest["new"]} <= parent_targets

    semantic_subtests = {
        item["old"]: item for item in migration_map["semantic_audits"]["subtests"]
    }
    ambiguous_subtests = {
        subtest["old"]
        for subtest in subtests
        if subtest["old"].rsplit("::subTest@", 1)[0] in methods_by_old
        and len(methods_by_old[subtest["old"].rsplit("::subTest@", 1)[0]]["new"]) > 1
    }
    assert set(semantic_subtests) == ambiguous_subtests
    for old_id, audit in semantic_subtests.items():
        subtest = subtests_by_old[old_id]
        assert subtest["kind"] == "parametrized"
        assert subtest["new"] == [
            {"node_pattern": audit["node_pattern"], "count": audit["count"]}
        ]
        decorators = _normalized_current_test_decorators(audit["node_pattern"])
        assert audit["parameter_tokens"]
        assert all(token in decorators for token in audit["parameter_tokens"])

    expected_semantic_subtests = {
        "tests/test_macos_ci_evidence.py::MacOSCiEvidenceTest.test_inductor_import_filter_ignores_only_the_exact_warning::subTest@438": (
            "unrelated-message",
            "wrong-module",
        ),
        "tests/test_torch_dispersive.py::DispersiveOracleTest.test_scalar_recurrences_match_explicit_independent_equations::subTest@335": (
            "dcp-plrc",
            "dcp-rc",
        ),
        "tests/test_torch_fdtd.py::TorchStateTest.test_direct_mutation_views_match_the_solver_slices::subTest@539": (
            "low",
            "high",
            "x",
            "y",
            "z",
        ),
        "tests/test_torch_pml.py::TorchPmlStorageTest.test_cpml_cuda_direct_view_metadata_uses_actual_plan_buckets::subTest@860": (
            "unfused",
            "cpu",
            "upml",
        ),
    }
    for old_id, tokens in expected_semantic_subtests.items():
        assert tuple(semantic_subtests[old_id]["parameter_tokens"]) == tokens

    semantic_methods = {
        item["old"]: item for item in migration_map["semantic_audits"]["methods"]
    }
    nonconventional_methods = {
        item["old"]
        for item in methods
        if [node["node_pattern"] for node in item["new"]]
        != [_conventional_method_pattern(item["old"])]
    }
    assert set(semantic_methods) == nonconventional_methods
    for old_id, audit in semantic_methods.items():
        assert audit["new"] == methods_by_old[old_id]["new"]


def test_failure_collector_preserves_sequential_continuation():
    collector = FailureCollector()
    completed = []
    with collector.case("first"):
        completed.append("first")
        assert False, "first failure"
    with collector.case("second"):
        completed.append("second")
        assert False, "second failure"
    with collector.case("third"):
        completed.append("third")
    assert completed == ["first", "second", "third"]
    with pytest.raises(ExceptionGroup) as caught:
        collector.raise_if_any()
    assert len(caught.value.exceptions) == 2
    assert all(
        "retained sequential subcase" in " ".join(error.__notes__)
        for error in caught.value.exceptions
    )

    with pytest.raises(ExceptionGroup) as compound:
        with FailureCollector() as collector:
            with collector.case("pending"):
                assert False, "pending assertion"
            raise KeyError("later ordinary exception")
    assert {type(error) for error in compound.value.exceptions} == {
        AssertionError,
        KeyError,
    }
