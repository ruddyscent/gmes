"""Verify compilation skips without masking eager tests or runtime failures."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("capability", ("supported", "unsupported", "error"))
def test_compile_capability_gate_preserves_test_outcomes(pytester, capability):
    conftest = Path(__file__).with_name("conftest.py").read_text()
    probe = (
        'raise RuntimeError("capability probe failed")'
        if capability == "error"
        else f'return {capability == "supported"!r}'
    )
    pytester.makeconftest(
        conftest
        + f"\ndef probe():\n    {probe}\n"
        + "\ntorch._dynamo.is_dynamo_supported = probe\n"
    )
    pytester.makeini(
        "[pytest]\nmarkers =\n"
        "    requires_torch_compile: requires real TorchDynamo compilation support\n"
    )
    pytester.makepyfile("""
        import pytest
        import torch

        @pytest.mark.parametrize("policy", (
            "eager",
            pytest.param("compile", marks=pytest.mark.requires_torch_compile),
        ))
        def test_execution(policy):
            if policy == "compile":
                raise RuntimeError("unrelated compilation failure")
            assert torch.add(torch.tensor(1), 2).item() == 3

        def test_mocked_compile(monkeypatch):
            monkeypatch.setattr(torch, "compile", lambda fn: fn)
            assert torch.compile(lambda: 42)() == 42

        def test_conditional_compile(request):
            request.getfixturevalue("requires_torch_compile")
            raise RuntimeError("unrelated compilation failure")
        """)
    result = pytester.runpytest_subprocess("-q", "-rs")
    if capability == "unsupported":
        result.assert_outcomes(passed=2, skipped=2)
        result.stdout.fnmatch_lines(["*torch.compile unsupported by torch*"])
    elif capability == "supported":
        result.assert_outcomes(passed=2, failed=2)
        result.stdout.fnmatch_lines(["*RuntimeError: unrelated compilation failure*"])
    else:
        # A fixture failure during setup is an error; a dynamic request fails
        # in the test body. Neither may turn into an unsupported-runtime skip.
        result.assert_outcomes(passed=2, errors=1, failed=1)
        result.stdout.fnmatch_lines(["*RuntimeError: capability probe failed*"])


@pytest.mark.parametrize(
    "operation",
    (
        "import",
        pytest.param("compile", marks=pytest.mark.requires_torch_compile),
    ),
)
def test_inductor_import_does_not_warn_about_torchscript(operation):
    """Exercise the first import/compile in a fresh process without suppression."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import warnings

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter("always")
    if sys.argv[1] == "import":
        import torch.utils.mkldnn
        import torch._inductor.fx_passes.pre_grad
    else:
        x = torch.arange(8, device="cpu", dtype=torch.float64)
        compiled = torch.compile(lambda x: x.sin() + 1, fullgraph=True, mode="default")
        torch.testing.assert_close(compiled(x), x.sin() + 1)
incidental = [str(w.message) for w in recorded if "torch.jit.script_method" in str(w.message)]
assert not incidental, (torch.__version__, incidental)
""",
            operation,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
