"""Gate real compilation tests on the installed TorchDynamo capability."""

import sys

import pytest
import torch


@pytest.fixture
def requires_torch_compile():
    """Skip only when TorchDynamo declares this interpreter unsupported."""
    if not torch._dynamo.is_dynamo_supported():
        pytest.skip(
            f"torch.compile unsupported by torch {torch.__version__} "
            f"on Python {sys.version.split()[0]} (TorchDynamo capability check)"
        )


@pytest.fixture(autouse=True)
def _check_torch_compile_marker(request):
    if request.node.get_closest_marker("requires_torch_compile") is not None:
        request.getfixturevalue("requires_torch_compile")
