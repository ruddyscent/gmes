"""CPU-only structural coverage for pinned CUDA wrapper evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks import torch_lowered_materialization as lowering


def _contract(*, regions: list[str] | None = None) -> dict[str, object]:
    source_hashes = {
        "gmes/torch_fdtd.py": "a" * 64,
        "gmes/torch_plan.py": "b" * 64,
        "gmes/torch_source.py": "c" * 64,
    }
    return {
        "case": "all-material-2d",
        "case_descriptor_sha256": "d" * 64,
        "compile_cache_key": "e" * 64,
        "compiled_region_topology": lowering.PINNED_CUDA_TOPOLOGY,
        "device": "cuda:0",
        "field_buffers": [
            {
                "name": name,
                "dtype": "torch.float64",
                "shape": [8],
                "stride": [1],
            }
            for name in lowering.FIELD_NAMES
        ],
        "plan_identity": "f" * 64,
        "precision": "float64",
        "required_regions": ["electric"] if regions is None else regions,
        "runtime_source_files_sha256": source_hashes,
        "runtime_source_sha256": hashlib.sha256(
            b"gmes/torch_fdtd.py\0"
            + b"a" * 64
            + b"\0"
            + b"gmes/torch_plan.py\0"
            + b"b" * 64
            + b"\0"
            + b"gmes/torch_source.py\0"
            + b"c" * 64
            + b"\0"
        ).hexdigest(),
        "source_buffer_sha256": "0" * 64,
        "torch_version": "2.13.0+cu130",
    }


def _wrapper(*, allocation: str = "", output: str = "arg0_1") -> str:
    return f"""\
kernel = async_compile.triton('kernel', r'''\
import triton
import triton.language as tl
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel: tl.constexpr):
    pass
''')

def call(args):
    arg0_1, = args
    assert_size_stride(arg0_1, (8,), (1,))
    {allocation}
    kernel.run(arg0_1, {output}, 8, grid=grid(8), stream=stream0)
"""


class LoweredMaterializationTest(unittest.TestCase):
    def _audit(self, source: str, *, contract: dict[str, object] | None = None):
        return lowering.audit_compiled_wrapper_sources(
            sources=[{"region": "electric", "source": source}],
            contract=_contract() if contract is None else contract,
        )

    def test_direct_inplace_output_is_verified(self):
        audit = self._audit(_wrapper())
        self.assertEqual(audit["status"], "verified")
        self.assertTrue(audit["verified"])
        output = audit["wrappers"][0]["launches"][0]["outputs"][0]
        self.assertEqual(output["kind"], "direct-input")

    def test_exact_field_layout_outputs_are_diagnostic_candidates(self):
        for allocation in (
            "buf0 = empty_strided_cuda((8,), (1,), torch.float64)",
            "buf0 = alloc_from_pool((8,), (1,), torch.float64)",
        ):
            with self.subTest(allocation=allocation):
                audit = self._audit(_wrapper(allocation=allocation, output="buf0"))
                self.assertEqual(audit["status"], "verified")
                self.assertTrue(audit["verified"])
                candidates = audit["full_domain_output_candidates"]
                self.assertEqual(len(candidates), 1)
                self.assertEqual(
                    candidates[0]["classification"],
                    "exact-field-layout-output-candidate",
                )

    def test_larger_or_nonfield_outputs_are_not_clone_candidates(self):
        for allocation in (
            "buf0 = empty_strided_cuda((64,), (1,), torch.float64)",
            "buf0 = empty_strided_cuda((8,), (1,), torch.int64)",
            "buf0 = empty_strided_cuda((4, 2), (2, 1), torch.float64)",
            "buf0 = empty_strided_cuda((8,), (1,), torch.float32)",
            "buf0 = empty_strided_cuda((8,), (2,), torch.float64)",
        ):
            with self.subTest(allocation=allocation):
                audit = self._audit(_wrapper(allocation=allocation, output="buf0"))
                self.assertEqual(audit["status"], "verified")
                self.assertTrue(audit["verified"])
                self.assertEqual(audit["full_domain_output_candidates"], [])

    def test_unknown_factory_mapping_and_opaque_paths_fail_closed(self):
        for source, reason in (
            (
                _wrapper(allocation="buf0 = mystery_alloc((8,))", output="buf0"),
                "allocation-factory-unknown:buf0",
            ),
            (_wrapper(output="buf0"), "launch-output-unmapped:kernel:out_ptr0"),
            ("extern_kernels.opaque()", "opaque-external-kernel"),
        ):
            with self.subTest(reason=reason):
                audit = self._audit(source)
                self.assertEqual(audit["status"], "unverified")
                self.assertIn(reason, audit["reasons"])

    def test_missing_required_region_fails_closed(self):
        audit = self._audit(
            _wrapper(), contract=_contract(regions=["electric", "magnetic"])
        )
        self.assertEqual(audit["status"], "unverified")
        self.assertIn("missing-region-coverage:magnetic", audit["reasons"])

    def test_malformed_field_layout_and_runtime_digest_fail_closed(self):
        contract = _contract()
        contract["field_buffers"][0]["dtype"] = "torch.float32"  # type: ignore[index]
        audit = self._audit(_wrapper(), contract=contract)
        self.assertEqual(audit["status"], "unverified")
        self.assertIn("contract-field-buffers", audit["reasons"])

        contract = _contract()
        contract["runtime_source_sha256"] = "1" * 64
        audit = self._audit(_wrapper(), contract=contract)
        self.assertEqual(audit["status"], "unverified")
        self.assertIn("contract-runtime-source-digest", audit["reasons"])

    def test_bundle_rehashes_sources_and_rejects_contract_mismatch(self):
        source = _wrapper()
        audit = self._audit(source)
        wrapper = audit["wrappers"][0]
        wrapper["_source_bytes"] = source.encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            lowering._write_bundle(root, audit)
            verified = lowering.verify_compiled_wrapper_audit(
                output_directory=root, expected_contract=_contract()
            )
            self.assertTrue(verified["verified"])
            descriptor = json.loads((root / "lowering-manifest.json").read_text())[
                "wrappers"
            ][0]
            (root / descriptor["path"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "digest or size"):
                lowering.verify_compiled_wrapper_audit(
                    output_directory=root, expected_contract=_contract()
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            audit = self._audit(source)
            audit["wrappers"][0]["_source_bytes"] = source.encode()
            lowering._write_bundle(root, audit)
            contract = _contract()
            contract["compile_cache_key"] = "9" * 64
            with self.assertRaisesRegex(ValueError, "contract differs"):
                lowering.verify_compiled_wrapper_audit(
                    output_directory=root, expected_contract=contract
                )


if __name__ == "__main__":
    unittest.main()
