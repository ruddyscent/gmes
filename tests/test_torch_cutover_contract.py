"""Atomic pure-Torch cutover contracts staged before legacy module deletion."""

import importlib
import os
import pickle
import subprocess
import sys

import pytest

import gmes

EXPECTED_EXPORTS = frozenset(
    (
        "geometry",
        "constant",
        "source",
        "material",
        "Cartesian",
        "DefaultMedium",
        "Cone",
        "Cylinder",
        "Block",
        "Ellipsoid",
        "Sphere",
        "Shell",
        "Ex",
        "Ey",
        "Ez",
        "Hx",
        "Hy",
        "Hz",
        "Jx",
        "Jy",
        "Jz",
        "Mx",
        "My",
        "Mz",
        "X",
        "Y",
        "Z",
        "PlusX",
        "MinusX",
        "PlusY",
        "MinusY",
        "PlusZ",
        "MinusZ",
        "Continuous",
        "Bandpass",
        "DifferentiatedGaussian",
        "PointSource",
        "TotalFieldScatteredField",
        "GaussianBeam",
        "Dummy",
        "Const",
        "Dielectric",
        "Upml",
        "Cpml",
        "DrudePole",
        "LorentzPole",
        "CriticalPoint",
        "DcpAde",
        "DcpPlrc",
        "DcpRc",
        "Drude",
        "Lorentz",
        "Dm2",
        "ComponentPlan",
        "DistributedLaunch",
        "ExecutionSignature",
        "FlattenedStencilTerm",
        "MaterialBucketPlan",
        "TorchConfigurationError",
        "TorchDistributedError",
        "TorchDistributedSimulation",
        "TorchHaloExchange",
        "TorchExecutionPlanner",
        "TorchPointSourceRecord",
        "TorchProbeSamples",
        "TorchProbeSpec",
        "TorchProbeSpectrum",
        "TorchRuntimeConfig",
        "TorchSimulation",
        "TorchSimulationPlan",
        "TorchSimulationState",
        "TorchSourceLoweringContext",
        "TwoGpuDecomposition",
        "pi",
        "c0",
        "mu0",
        "eps0",
        "Z0",
        "PETA",
        "TERA",
        "GIGA",
        "MEGA",
        "KILO",
        "MILLI",
        "MICRO",
        "NANO",
        "PICO",
        "FEMTO",
        "ATTO",
        "inf",
        "choose_two_gpu_decomposition",
        "distributed_launch_from_environment",
        "probe_spectrum",
        "rank_local_space",
        "read_torch_checkpoint",
        "torch_runtime_diagnostics",
        "write_probe_text",
        "write_torch_checkpoint",
    )
)


class TestTorchCutoverContract:
    """Verify the supported public boundary independently of legacy adapters."""

    def test_exact_root_exports_and_canonical_constant_identity(self):
        assert (len(gmes.__all__)) == (98)
        assert (set(gmes.__all__)) == (EXPECTED_EXPORTS)
        assert (len(gmes.__all__)) == (len(set(gmes.__all__)))
        assert (gmes.Ex) is (gmes.constant.Ex)

        old_ex = gmes.constant.Ex
        old_vector = gmes.constant.PlusX.vector
        old_pickle = pickle.dumps(old_ex)
        reloaded = importlib.reload(gmes.constant)
        assert (reloaded.Ex) is (old_ex)
        assert (reloaded.PlusX.vector) is (old_vector)
        assert (pickle.loads(old_pickle)) is (old_ex)
        assert (pickle.loads(pickle.dumps(reloaded.Ex))) is (old_ex)
        markers = (
            reloaded.Component,
            reloaded.Electric,
            reloaded.Magnetic,
            reloaded.Ex,
            reloaded.Ey,
            reloaded.Ez,
            reloaded.Hx,
            reloaded.Hy,
            reloaded.Hz,
            reloaded.ElectricCurrent,
            reloaded.Jx,
            reloaded.Jy,
            reloaded.Jz,
            reloaded.MagneticCurrent,
            reloaded.Mx,
            reloaded.My,
            reloaded.Mz,
            reloaded.Directional,
            reloaded.X,
            reloaded.Y,
            reloaded.Z,
            reloaded.PlusX,
            reloaded.MinusX,
            reloaded.PlusY,
            reloaded.MinusY,
            reloaded.PlusZ,
            reloaded.MinusZ,
        )
        assert ({marker.tag for marker in markers}) == (set(range(27)))

    def test_root_import_does_not_load_matplotlib_or_a_gmes_native_module(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, gmes; "
                "assert not any(name.startswith('matplotlib') for name in sys.modules); "
                "assert not any(name.startswith('gmes._') for name in sys.modules)",
            ],
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
        )
        assert (result.returncode) == (0)

    def test_parallel_cartesian_rejects_before_runtime_allocation(self):
        with pytest.raises(NotImplementedError, match="parallel=True.*unsupported"):
            gmes.Cartesian((1, 1, 1), parallel=True)

    @pytest.mark.parametrize(
        "module_name_case", range(4), ids=("fdtd", "show", "pw-source", "pw-material")
    )
    def test_retired_modules_are_absent_after_atomic_cutover(self, module_name_case):
        module_name_case_values = tuple(
            (
                "gmes.fdtd",
                "gmes.show",
                "gmes.pw_source",
                "gmes.pw_material",
            )
        )
        assert len(module_name_case_values) == 4
        module_name = module_name_case_values[module_name_case]
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
