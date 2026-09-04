import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

_spec = importlib.util.spec_from_file_location(
    "torch_correctness", Path(__file__).parents[1] / "benchmarks/torch_correctness.py"
)
torch_correctness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(torch_correctness)


class TorchCorrectnessCliTests(unittest.TestCase):
    def test_main_emits_fixed_token_for_each_command(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        output_path = Path(temporary_directory.name) / "out.json"
        index_path = Path(temporary_directory.name) / "index.json"
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
        for command, (token, arguments) in cases.items():
            with (
                self.subTest(command=command),
                mock.patch.object(
                    sys, "argv", ["torch_correctness", command, *arguments]
                ),
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
                self.assertEqual(0, torch_correctness.main())
            self.assertEqual(token + "\n", output.getvalue())
            self.assertNotIn("/private/secret", output.getvalue())
            if command == "capture":
                capture.assert_called_once()
            elif command == "index":
                build.assert_called_once()
                self.assertEqual(1, evidence.call_count)
            else:
                load.assert_called_once()
                self.assertEqual(1, evidence.call_count)


if __name__ == "__main__":
    unittest.main()
