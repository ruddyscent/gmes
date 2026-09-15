"""Public geometry opt-in contracts at bounded lowering boundaries."""

import inspect
import pickle
from copy import deepcopy

import numpy as np
import pytest

import gmes
from gmes.pygeom import GeomBoxTree, GeometricObject
from tests.test_torch_fdtd import restore_torch_runtime as restore_torch_runtime


@gmes.vectorized_geometry
class PublicSphere(gmes.Sphere):
    """A pickleable parameter-only extension of the built-in sphere."""


class ScalarSphere(gmes.Sphere):
    """A sphere that retains the default custom-class scalar fallback."""


class LegacySphere(gmes.Sphere):
    """Keep the existing private opt-in available."""

    _gmes_vectorized_geometry = True


def _tree(sphere_type=PublicSphere):
    space = gmes.Cartesian((2, 2, 2), 4)
    space.dt = 0.05
    geometries = (
        gmes.DefaultMedium(gmes.Dielectric()),
        sphere_type(gmes.Dielectric(2), radius=0.25),
    )
    for geometry in geometries:
        geometry.init(space)
    return GeomBoxTree(geometries)


def _axes():
    return tuple(
        np.array(values, dtype=np.float64) for values in ((-0.5, 0, 0.5), (0,), (0,))
    )


def _assert_scalar_map(tree):
    lowered = tree.lower_grid(*_axes())
    geometries = tree.root.geom_list
    expected = [
        geometries.index(tree.object_of_point((x, 0, 0))[0]) for x in _axes()[0]
    ]
    np.testing.assert_array_equal(lowered.material_ids, expected)
    return lowered


def test_public_exports_identity_and_introspection():
    from gmes.geometry import vectorized_geometry
    from gmes.pygeom import vectorized_geometry as implementation

    assert vectorized_geometry is implementation is gmes.vectorized_geometry
    methods = PublicSphere.in_object, PublicSphere._contains_points
    signature = inspect.signature(PublicSphere)
    metadata = PublicSphere.__name__, PublicSphere.__qualname__, PublicSphere.__module__
    assert gmes.vectorized_geometry(PublicSphere) is PublicSphere
    assert gmes.vectorized_geometry(PublicSphere) is PublicSphere
    assert inspect.signature(PublicSphere) == signature
    assert metadata == (
        PublicSphere.__name__,
        PublicSphere.__qualname__,
        PublicSphere.__module__,
    )
    assert methods == (PublicSphere.in_object, PublicSphere._contains_points)
    assert PublicSphere.__bases__ == (gmes.Sphere,)


@pytest.mark.parametrize(
    "initialized", (False, True), ids=("uninitialized", "initialized")
)
@pytest.mark.parametrize(
    "copy_object",
    (deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))),
    ids=("deepcopy", "pickle"),
)
def test_serialization_preserves_class_parameters_and_opt_in(initialized, copy_object):
    original = PublicSphere(gmes.Dielectric(2.5), radius=0.25, center=(0.1, 0.2, 0.3))
    if initialized:
        space = gmes.Cartesian((2, 2, 2), 4)
        space.dt = 0.05
        original.init(space)
    restored = copy_object(original)
    assert type(restored) is PublicSphere
    assert restored.radius == original.radius
    np.testing.assert_array_equal(restored.center, original.center)
    assert restored.material.eps_inf == original.material.eps_inf
    assert GeomBoxTree._uses_vectorized_predicate(restored)
    if initialized:
        np.testing.assert_array_equal(restored.box.low, original.box.low)
        np.testing.assert_array_equal(restored.box.high, original.box.high)
    else:
        assert restored.box is None
    coordinates = np.array((0, 0.25, 0.5), dtype=np.float64)
    np.testing.assert_array_equal(
        restored._contains_points(coordinates, coordinates, coordinates),
        original._contains_points(coordinates, coordinates, coordinates),
    )


@pytest.mark.parametrize(
    "sphere_type, expected",
    (
        (gmes.Sphere, True),
        (PublicSphere, True),
        (ScalarSphere, False),
        (LegacySphere, True),
    ),
    ids=("builtin", "public", "bare-subclass-fallback", "legacy"),
)
def test_opt_in_and_fallback(sphere_type, expected):
    tree = _tree(sphere_type)
    assert tree.supports_bulk_lowering() is expected
    np.testing.assert_array_equal(_assert_scalar_map(tree).material_ids, (0, 1, 0))


def test_parameter_only_subclass_inherits_public_contract():
    class SmallSphere(PublicSphere):
        def __init__(self, material, radius=1):
            super().__init__(material, radius=radius / 2)

    tree = _tree(SmallSphere)
    assert tree.supports_bulk_lowering()
    _assert_scalar_map(tree)


def test_classmethod_implementations_preserve_inheritance_and_detect_overrides():
    @gmes.vectorized_geometry
    class ClassSphere(gmes.Sphere):
        extent = 0.25

        @classmethod
        def in_object(cls, point):
            return bool(sum(value * value for value in point) <= cls.extent**2)

        @classmethod
        def _contains_points(cls, x, y, z):
            return x * x + y * y + z * z <= cls.extent**2

    class Child(ClassSphere):
        extent = 0.125

    class Grandchild(Child):
        pass

    for sphere_type in (ClassSphere, Child, Grandchild):
        tree = _tree(sphere_type)
        assert tree.supports_bulk_lowering()
        _assert_scalar_map(tree)

    class ChangedScalar(Grandchild):
        @classmethod
        def in_object(cls, point):
            return False

    class ChangedArray(Grandchild):
        @classmethod
        def _contains_points(cls, x, y, z):
            return np.zeros(x.shape, dtype=np.bool_)

    class ChangedBoth(ChangedScalar, ChangedArray):
        pass

    for sphere_type in (ChangedScalar, ChangedArray, ChangedBoth):
        tree = _tree(sphere_type)
        assert not tree.supports_bulk_lowering()
        _assert_scalar_map(tree)
    assert gmes.vectorized_geometry(ChangedBoth) is ChangedBoth
    assert _tree(ChangedBoth).supports_bulk_lowering()
    np.testing.assert_array_equal(
        _assert_scalar_map(_tree(ChangedBoth)).material_ids, (0, 0, 0)
    )


@pytest.mark.parametrize("override", ("scalar", "array", "both", "mixin"))
def test_containment_changes_require_renewed_public_opt_in(override):
    class EmptyMixin:
        def in_object(self, point):
            return False

    if override == "mixin":

        class Changed(EmptyMixin, PublicSphere):
            pass

    else:

        class Changed(PublicSphere):
            pass

        if override in ("scalar", "both"):
            Changed.in_object = lambda self, point: False
        if override in ("array", "both"):
            Changed._contains_points = lambda self, x, y, z: np.zeros(
                x.shape, dtype=np.bool_
            )
    tree = _tree(Changed)
    assert not tree.supports_bulk_lowering()
    _assert_scalar_map(tree)
    if override == "both":
        assert gmes.vectorized_geometry(Changed) is Changed
        assert tree.supports_bulk_lowering()
        np.testing.assert_array_equal(_assert_scalar_map(tree).material_ids, (0, 0, 0))


def test_legacy_inheritance_and_explicit_opt_out_remain_compatible():
    class ChangedLegacy(LegacySphere):
        def in_object(self, point):
            return False

    tree = _tree(ChangedLegacy)
    assert tree.supports_bulk_lowering()
    # Legacy opt-in trusts the author's semantic-equivalence promise unchanged.
    np.testing.assert_array_equal(tree.lower_grid(*_axes()).material_ids, (0, 1, 0))
    assert tree.object_of_point((0, 0, 0))[0] is tree.root.geom_list[0]

    class OptedOut(ChangedLegacy):
        _gmes_vectorized_geometry = False

    opted_out = _tree(OptedOut)
    assert not opted_out.supports_bulk_lowering()
    np.testing.assert_array_equal(_assert_scalar_map(opted_out).material_ids, (0, 0, 0))


def test_public_contract_takes_precedence_over_legacy_marker():
    @gmes.vectorized_geometry
    class PublicLegacy(LegacySphere):
        pass

    class Changed(PublicLegacy):
        def in_object(self, point):
            return False

    assert not _tree(Changed).supports_bulk_lowering()
    _assert_scalar_map(_tree(Changed))

    class ParameterOnly(PublicLegacy):
        _gmes_vectorized_geometry = False

    assert _tree(ParameterOnly).supports_bulk_lowering()


@pytest.mark.parametrize(
    "invalid",
    (object, object(), gmes.Sphere(None), GeometricObject),
    ids=("non-geometry-class", "non-class", "instance", "base-loop"),
)
def test_invalid_decorator_usage_fails(invalid):
    with pytest.raises(TypeError, match="vectorized_geometry requires"):
        gmes.vectorized_geometry(invalid)


@pytest.mark.parametrize("method", ("in_object", "_contains_points"))
def test_noncallable_predicates_are_rejected(method):
    class Invalid(gmes.Sphere):
        pass

    setattr(Invalid, method, None)
    with pytest.raises(TypeError, match="scalar and array predicates"):
        gmes.vectorized_geometry(Invalid)


@pytest.mark.parametrize("opt_in", ("public", "legacy"))
@pytest.mark.parametrize(
    "result, error, message",
    (
        (lambda x: np.ones(x.shape, dtype=np.int64), TypeError, "Boolean NumPy array"),
        (
            lambda x: np.ones(x.shape, dtype=np.float64),
            TypeError,
            "Boolean NumPy array",
        ),
        (lambda x: [True] * len(x), TypeError, "Boolean NumPy array"),
        (lambda x: True, TypeError, "Boolean NumPy array"),
        (lambda x: np.array(True), ValueError, "same shape"),
        (lambda x: np.ones((len(x), 1), dtype=np.bool_), ValueError, "same shape"),
        (lambda x: np.ones(len(x) + 1, dtype=np.bool_), ValueError, "same shape"),
    ),
    ids=(
        "integer-mask",
        "float-mask",
        "list",
        "scalar-bool",
        "zero-dimensional",
        "column",
        "wrong-length",
    ),
)
def test_invalid_predicate_results_fail_before_indexing(opt_in, result, error, message):
    class Invalid(gmes.Sphere):
        def _contains_points(self, x, y, z):
            return result(x)

    if opt_in == "public":
        gmes.vectorized_geometry(Invalid)
    else:
        Invalid._gmes_vectorized_geometry = True
    with pytest.raises(error, match=message):
        _tree(Invalid).lower_grid(*_axes())


@pytest.mark.parametrize(
    "empty_axis", (None, 0, 1, 2), ids=("empty-tile", "empty-x", "empty-y", "empty-z")
)
def test_empty_lowering_does_not_call_predicates(empty_axis):
    @gmes.vectorized_geometry
    class NoCalls(gmes.Sphere):
        def _contains_points(self, x, y, z):
            raise AssertionError("empty lowering must not invoke containment")

    axes = list(_axes())
    if empty_axis is not None:
        axes[empty_axis] = np.array((), dtype=np.float64)
    with np.errstate(all="raise"):
        lowered = _tree(NoCalls).lower_grid(*axes, stop=0)
    assert lowered.material_ids.shape == lowered.underlying_ids.shape == (0,)
    assert lowered.material_ids.dtype == lowered.underlying_ids.dtype == np.int32


def test_predicate_direct_empty_contract():
    empty = np.array((), dtype=np.float64)
    result = PublicSphere(None)._contains_points(empty, empty, empty)
    assert result.dtype == np.bool_
    assert result.shape == (0,)


def _simulation(sphere_type, *, compile_policy="eager"):
    return gmes.TorchSimulation(
        space=gmes.Cartesian((1, 1, 1), 3),
        geometry=[
            gmes.DefaultMedium(gmes.Dielectric()),
            sphere_type(gmes.Dielectric(2), radius=0.6),
            sphere_type(gmes.Dielectric(3), radius=0.3),
            gmes.Shell(gmes.Cpml()),
        ],
        sources=[gmes.PointSource(gmes.Continuous(0.8, width=0.5), (0, 0, 0), gmes.Ez)],
        runtime=gmes.TorchRuntimeConfig(
            device="cpu",
            precision="float64",
            cpu_threads=1,
            execution_policy="dense",
            compile_policy=compile_policy,
        ),
    )


def test_public_scalar_overlap_maps_and_multistep_fields_are_equal():
    vectorized = _simulation(PublicSphere)
    scalar = _simulation(ScalarSphere)
    for name, expected in scalar.plan.components.items():
        actual = vectorized.plan.components[name]
        np.testing.assert_array_equal(actual.material_ids, expected.material_ids)
        np.testing.assert_array_equal(actual.underlying_ids, expected.underlying_ids)
    assert np.any(vectorized.plan.components["Ex"].underlying_ids == 2)
    vectorized.advance(3)
    scalar.advance(3)
    assert vectorized.state.step_count.item() == scalar.state.step_count.item() == 3
    for name, expected in scalar.host_snapshot().items():
        np.testing.assert_array_equal(vectorized.host_snapshot()[name], expected)


def test_planner_bounds_predicate_inputs_and_preserves_maps():
    from gmes.torch_fdtd import _field_shapes
    from gmes.torch_plan import TorchExecutionPlanner

    calls = []

    @gmes.vectorized_geometry
    class BoundedSphere(gmes.Sphere):
        def _contains_points(self, x, y, z):
            assert x.ndim == y.ndim == z.ndim == 1
            assert x.dtype == y.dtype == z.dtype == np.float64
            assert x.shape == y.shape == z.shape
            assert 0 < len(x) <= 7
            calls.append(len(x))
            return super()._contains_points(x, y, z)

    space = gmes.Cartesian((2, 2, 2), 3)
    plans = []
    for sphere_type in (BoundedSphere, ScalarSphere):
        tree = _tree(sphere_type)
        plans.append(
            TorchExecutionPlanner(
                geom_tree=tree,
                space=space,
                shapes=_field_shapes(space),
                precision="float64",
                device_type="cpu",
                policy="dense",
                material_tile_size=7,
            ).build()
        )
    assert len(calls) > 6
    for actual, expected in zip(*plans, strict=True):
        np.testing.assert_array_equal(actual.material_ids, expected.material_ids)
        np.testing.assert_array_equal(actual.underlying_ids, expected.underlying_ids)


def test_distributed_cost_lowering_matches_scalar_geometry():
    space = gmes.Cartesian((4, 2, 2), 2)
    decompositions = []
    for sphere_type in (PublicSphere, ScalarSphere):
        geometry = [
            gmes.DefaultMedium(gmes.Dielectric()),
            sphere_type(gmes.Dm2(omega=(1.0,), n_atom=(1.0,)), radius=0.6),
        ]
        decompositions.append(
            gmes.choose_two_gpu_decomposition(space, geometry, split_axis=0)
        )
    assert decompositions[0] == decompositions[1]
    background = gmes.choose_two_gpu_decomposition(
        space, [gmes.DefaultMedium(gmes.Dielectric())], split_axis=0
    )
    assert sum(decompositions[0].rank_costs) > sum(background.rank_costs)


def test_distributed_cost_lowering_rejects_invalid_array_mask():
    @gmes.vectorized_geometry
    class InvalidSphere(gmes.Sphere):
        def _contains_points(self, x, y, z):
            return super()._contains_points(x, y, z).astype(np.int64)

    with pytest.raises(TypeError, match="Boolean NumPy array"):
        gmes.choose_two_gpu_decomposition(
            gmes.Cartesian((4, 2, 2), 2),
            [
                gmes.DefaultMedium(gmes.Dielectric()),
                InvalidSphere(gmes.Dm2(omega=(1.0,), n_atom=(1.0,)), radius=0.6),
            ],
            split_axis=0,
        )


@pytest.mark.requires_torch_compile
def test_compiled_steps_do_not_reenter_geometry_predicates():
    import torch

    calls = []

    @gmes.vectorized_geometry
    class CountedSphere(gmes.Sphere):
        def _contains_points(self, x, y, z):
            calls.append(len(x))
            return super()._contains_points(x, y, z)

    simulation = gmes.TorchSimulation(
        space=gmes.Cartesian((1, 1, 0), 2),
        geometry=[
            gmes.DefaultMedium(gmes.Dielectric()),
            CountedSphere(gmes.Dielectric(2), radius=0.3),
        ],
        sources=[gmes.PointSource(gmes.Continuous(0.8), (0, 0, 0), gmes.Ez)],
        runtime=gmes.TorchRuntimeConfig(
            device="cpu",
            precision="float64",
            cpu_threads=1,
            execution_policy="dense",
            compile_policy="compile",
        ),
    )
    after_construction = calls.copy()
    assert after_construction
    simulation.advance(2)
    assert calls == after_construction
    assert simulation.state.step_count.item() == 2
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    assert all(
        np.isfinite(field).all() for field in simulation.host_snapshot().values()
    )
