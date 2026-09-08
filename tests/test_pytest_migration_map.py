"""Validate the durable issue-179 unittest-to-pytest migration inventory."""

import ast
import json
from pathlib import Path

MAP_PATH = Path(__file__).with_name("pytest-migration-map.json")
BASE_REVISION = "8c1ea5aba7b11d96bd85541f95241c0539b280f9"
BASELINE_LOG_SHA256 = "c6fb3f7c2f1c61cb8544d20fc80de481dbe6a6c999dc703688c6d0d0857b2234"
MAP_VALIDATION_NODE = "tests/test_pytest_migration_map.py::test_pytest_migration_map_is_complete_and_auditable"


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


def test_pytest_migration_map_is_complete_and_auditable():
    migration_map = json.loads(MAP_PATH.read_text())

    assert migration_map["schema_version"] == 3
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
    assert migration_map["classification_counts"] == {
        "parametrized": 144,
        "retained-sequential": 184,
    }
    assert migration_map["current_collection"]["node_count"] >= 913
    assert all(not gap for gap in migration_map["gaps"].values())

    methods = migration_map["method_mappings"]
    subtests = migration_map["subtest_mappings"]
    assert len(methods) == 913
    assert len(subtests) == 328
    assert len({item["old"] for item in methods}) == len(methods)
    assert len({item["old"] for item in subtests}) == len(subtests)

    for item in [*methods, *subtests]:
        assert item["line"] > 0
        assert item["new"]
        assert all(
            node["node_pattern"].startswith("tests/") and node["count"] > 0
            for node in item["new"]
        )

    mapped_patterns = {node["node_pattern"] for item in methods for node in item["new"]}
    assert migration_map["coverage_exclusions"] == [
        {
            "node_pattern": MAP_VALIDATION_NODE,
            "reason": "validates the migration map itself",
        }
    ]
    assert _current_converted_test_patterns() - mapped_patterns == {MAP_VALIDATION_NODE}

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

    semantic_subtests = {
        item["old"]: item for item in migration_map["semantic_audits"]["subtests"]
    }
    expected_semantic_subtests = {
        "tests/test_macos_ci_evidence.py::MacOSCiEvidenceTest.test_inductor_import_filter_ignores_only_the_exact_warning::subTest@438": {
            "node_pattern": "tests/test_macos_ci_evidence.py::TestMacOSCiEvidence::test_inductor_import_filter_propagates_nonmatching_warnings",
            "count": 2,
            "parameter_tokens": ("unrelated-message", "wrong-module"),
        },
        "tests/test_torch_dispersive.py::DispersiveOracleTest.test_scalar_recurrences_match_explicit_independent_equations::subTest@335": {
            "node_pattern": "tests/test_torch_dispersive.py::TestDispersiveOracle::test_scalar_convolution_recurrences_match_explicit_independent_equations",
            "count": 2,
            "parameter_tokens": ("dcp-plrc", "dcp-rc"),
        },
        "tests/test_torch_fdtd.py::TorchStateTest.test_direct_mutation_views_match_the_solver_slices::subTest@539": {
            "node_pattern": "tests/test_torch_fdtd.py::TestTorchState::test_boundary_planes_match_solver_slices",
            "count": 6,
            "parameter_tokens": ("low", "high", "x", "y", "z"),
        },
        "tests/test_torch_pml.py::TorchPmlStorageTest.test_cpml_cuda_direct_view_metadata_uses_actual_plan_buckets::subTest@860": {
            "node_pattern": "tests/test_torch_pml.py::TestTorchPmlStorage::test_cpml_cuda_direct_views_reject_unsupported_execution",
            "count": 3,
            "parameter_tokens": ("unfused", "cpu", "upml"),
        },
    }
    assert set(semantic_subtests) == set(expected_semantic_subtests)
    subtests_by_old = {item["old"]: item for item in subtests}
    for old_id, expected in expected_semantic_subtests.items():
        audit = semantic_subtests[old_id]
        assert audit["count"] == expected["count"]
        assert audit["node_pattern"] == expected["node_pattern"]
        assert tuple(audit["parameter_tokens"]) == expected["parameter_tokens"]
        assert subtests_by_old[old_id]["kind"] == "parametrized"
        assert subtests_by_old[old_id]["new"] == [
            {"node_pattern": audit["node_pattern"], "count": audit["count"]}
        ]
        assert all(
            token in _normalized_current_test_decorators(audit["node_pattern"])
            for token in audit["parameter_tokens"]
        )

    expected_semantic_methods = {
        "tests/test_annotations.py::AnnotationCoverageTest.test_public_annotations_are_complete_and_resolvable": {
            "tests/test_annotations.py::TestAnnotationCoverage::test_public_annotation_inventory_is_complete": 1,
            "tests/test_annotations.py::TestAnnotationCoverage::test_public_annotations_are_complete_and_resolvable": 272,
        },
        "tests/test_issue123_privacy.py::Issue123PrivacyProjectionTest.test_numeric_boolean_and_null_identity_leaves_are_scanned": {
            "tests/test_issue123_privacy.py::TestIssue123PrivacyProjection::test_numeric_boolean_and_null_identity_leaves_are_scanned": 4,
            "tests/test_issue123_privacy.py::TestIssue123PrivacyProjection::test_short_numeric_identity_is_rejected_before_scanning": 1,
        },
        "tests/test_macos_ci_evidence.py::MacOSCiEvidenceTest.test_inductor_import_filter_ignores_only_the_exact_warning": {
            "tests/test_macos_ci_evidence.py::TestMacOSCiEvidence::test_inductor_import_filter_ignores_only_the_exact_warning": 1,
            "tests/test_macos_ci_evidence.py::TestMacOSCiEvidence::test_inductor_import_filter_propagates_nonmatching_warnings": 2,
        },
        "tests/test_torch_dispersive.py::DispersiveOracleTest.test_scalar_recurrences_match_explicit_independent_equations": {
            "tests/test_torch_dispersive.py::TestDispersiveOracle::test_scalar_ade_recurrence_matches_explicit_independent_equations": 1,
            "tests/test_torch_dispersive.py::TestDispersiveOracle::test_scalar_convolution_recurrences_match_explicit_independent_equations": 2,
            "tests/test_torch_dispersive.py::TestDispersiveOracle::test_scalar_recurrences_match_explicit_independent_equations": 2,
        },
        "tests/test_torch_dispersive.py::DispersiveOracleTest.test_paired_real_complex_recurrences_match_dense_reference": {
            "tests/test_torch_dispersive.py::TestDispersiveOracle::test_paired_real_complex_recurrences_match_dense_reference": 5,
            "tests/test_torch_dispersive.py::TestDispersiveOracle::test_paired_real_mixed_grouping_matches_dense_reference": 1,
        },
        "tests/test_torch_fdtd.py::TorchStateTest.test_direct_mutation_views_match_the_solver_slices": {
            "tests/test_torch_fdtd.py::TestTorchState::test_boundary_planes_match_solver_slices": 6,
            "tests/test_torch_fdtd.py::TestTorchState::test_direct_mutation_views_match_the_solver_slices": 6,
        },
        "tests/test_torch_pml.py::TorchPmlStorageTest.test_cpml_cuda_direct_view_metadata_uses_actual_plan_buckets": {
            "tests/test_torch_pml.py::TestTorchPmlStorage::test_cpml_cuda_direct_view_metadata_uses_actual_plan_buckets": 3,
            "tests/test_torch_pml.py::TestTorchPmlStorage::test_cpml_cuda_direct_views_reject_unsupported_execution": 3,
        },
    }
    semantic_methods = {
        item["old"]: item for item in migration_map["semantic_audits"]["methods"]
    }
    assert set(semantic_methods) == set(expected_semantic_methods)
    methods_by_old = {item["old"]: item for item in methods}
    for old_id, expected in expected_semantic_methods.items():
        actual = {
            item["node_pattern"]: item["count"]
            for item in semantic_methods[old_id]["new"]
        }
        assert actual == expected
        assert {
            item["node_pattern"]: item["count"]
            for item in methods_by_old[old_id]["new"]
        } == expected
