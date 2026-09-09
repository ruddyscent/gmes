import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

_spec = importlib.util.spec_from_file_location(
    "torch_correctness", Path(__file__).parents[1] / "benchmarks/torch_correctness.py"
)
torch_correctness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(torch_correctness)


class TestTorchCorrectnessCli:
    @pytest.mark.parametrize(
        "command_case", range(3), ids=("capture", "index", "validate-index")
    )
    def test_main_emits_fixed_token_for_each_command(self, tmp_path, command_case):
        output_path = tmp_path / "out.json"
        index_path = tmp_path / "index.json"
        cases = {
            "capture": (
                "torch-correctness-capture-ok",
                [
                    "--reference",
                    "/private/reference.json",
                    "--output",
                    str(output_path),
                ],
            ),
            "index": (
                "torch-correctness-index-ok",
                [
                    "--references",
                    "/private/reference.json",
                    "--candidates",
                    "/private/candidate.json",
                    "--candidate-evidence",
                    "/private/evidence.json",
                    "--descriptor-root",
                    "/private/descriptors",
                    "--runtime-receipt",
                    "/private/receipt.json",
                    "--output",
                    str(index_path),
                ],
            ),
            "validate-index": (
                "torch-correctness-validate-index-ok",
                [
                    "--index",
                    "/private/index.json",
                    "--candidate-evidence",
                    "/private/evidence.json",
                    "--descriptor-root",
                    "/private/descriptors",
                    "--runtime-receipt",
                    "/private/receipt.json",
                ],
            ),
        }
        command_case_values = tuple(cases.items())
        assert len(command_case_values) == 3
        command, (token, arguments) = command_case_values[command_case]
        with (
            mock.patch.object(sys, "argv", ["torch_correctness", command, *arguments]),
            mock.patch.object(
                torch_correctness,
                "_load_trusted_manifest",
                return_value=({}, "manifest"),
            ),
            mock.patch.object(
                torch_correctness,
                "_load_candidate_evidence",
                return_value={
                    "manifest_sha256": "manifest",
                    "sentinel": "/private/secret",
                },
            ) as evidence,
            mock.patch.object(
                torch_correctness,
                "capture_torch_candidate",
                return_value={"path": "/private/secret"},
            ) as capture,
            mock.patch.object(
                torch_correctness,
                "build_correctness_evidence_index",
                return_value={"path": "/private/secret"},
            ) as build,
            mock.patch.object(
                torch_correctness,
                "load_correctness_evidence_index",
                return_value={"path": "/private/secret"},
            ) as load,
            redirect_stdout(io.StringIO()) as output,
        ):
            assert (0) == (torch_correctness.main())
        assert (token + "\n") == (output.getvalue())
        assert ("/private/secret") not in (output.getvalue())
        if command == "capture":
            capture.assert_called_once()
        elif command == "index":
            build.assert_called_once()
            assert (1) == (evidence.call_count)
        else:
            load.assert_called_once()
            assert (1) == (evidence.call_count)
