"""Regress finite-point containment and unbounded geometric extents."""

import numpy as np
import pytest

from gmes import Block, DefaultMedium, Dielectric, Ellipsoid, Sphere
from gmes.pygeom import GeomBoxTree


@pytest.mark.parametrize("shape_class", (Block, Ellipsoid), ids=("block", "ellipsoid"))
@pytest.mark.parametrize(
    ("e1", "e2"),
    (((1, 0, 0), (0, 1, 0)), ((1, 1, 0), (-1, 1, 0)), ((1, 0, 0), (1, 1, 0))),
    ids=("axis-aligned", "rotated", "skew"),
)
@pytest.mark.parametrize(
    "size",
    ((2, 4, 6), (np.inf, 4, 6), (np.inf,) * 3),
    ids=("bounded", "one-unbounded-axis", "all-unbounded"),
)
@pytest.mark.parametrize(
    "point",
    (
        (np.inf,) * 3,
        (-np.inf,) * 3,
        (np.inf, 0, 0),
        (0, -np.inf, 0),
        (0, 0, np.inf),
        (np.nan, 0, 0),
    ),
    ids=(
        "background-sentinel",
        "negative-infinities",
        "positive-x",
        "negative-y",
        "positive-z",
        "nan",
    ),
)
def test_nonfinite_queries_are_outside(shape_class, e1, e2, size, point):
    shape = shape_class(Dielectric(4), e1=e1, e2=e2, size=size)
    with np.errstate(all="raise"):
        assert shape.in_object(point) is False
        # Mixed batches must retain the finite entries, including the center.
        points = np.array(((0, 0, 0), point, (0, 0, 4)), dtype=np.float64)
        expected = (True, False, size[2] == np.inf)
        np.testing.assert_array_equal(shape._contains_points(*points.T), expected)
        np.testing.assert_array_equal(
            shape._contains_points(*np.array((point,), dtype=np.float64).T), (False,)
        )


@pytest.mark.parametrize("shape_class", (Block, Ellipsoid), ids=("block", "ellipsoid"))
@pytest.mark.parametrize(
    ("e1", "e2", "size", "half_size"),
    (
        ((1, 0, 0), (0, 1, 0), (2, 4, 6), (1, 2, 3)),
        ((1, 0, 0), (0, 1, 0), (np.inf, 4, 6), (np.inf, 2, 3)),
        ((0, -1, 0), (1, 0, 0), (np.inf, 4, 6), (2, np.inf, 3)),
        ((1, 1, 0), (-1, 1, 0), (2 * np.sqrt(2), 2 * np.sqrt(2), 6), (2, 2, 3)),
        ((1, 1, 0), (-1, 1, 0), (np.inf, 4, 6), (np.inf, np.inf, 3)),
        ((1, 0, 0), (1, 1, 0), (np.inf, 2 * np.sqrt(2), 6), (np.inf, 1, 3)),
        ((1, 0, 0), (1, 1, 0), (2, np.inf, 6), (np.inf, np.inf, 3)),
        ((1, 0, 0), (0, 1, 0), (np.inf,) * 3, (np.inf,) * 3),
    ),
    ids=(
        "finite",
        "unbounded-x",
        "permuted-negative-axis",
        "finite-rotated",
        "unbounded-rotated",
        "unbounded-skew-x",
        "unbounded-skew-diagonal",
        "all-unbounded",
    ),
)
def test_bounds_enclose_unbounded_edges(shape_class, e1, e2, size, half_size):
    center = np.array((0.25, -0.5, 0.75))
    shape = shape_class(Dielectric(4), center=center, e1=e1, e2=e2, size=size)
    default = DefaultMedium(Dielectric(1))
    with np.errstate(all="raise"):
        shape.init(None)
        default.init(None)
        np.testing.assert_allclose(shape.box.low, center - half_size)
        np.testing.assert_allclose(shape.box.high, center + half_size)
        tree = GeomBoxTree((default, shape))
        assert tree.object_of_point(tuple(center))[0] is shape
        assert tree.object_of_point((np.inf,) * 3)[0] is default


@pytest.mark.parametrize("shape_class", (Block, Ellipsoid), ids=("block", "ellipsoid"))
@pytest.mark.parametrize(
    ("e1", "e2", "size", "offset", "expected"),
    (
        ((1, 0, 0), (0, 1, 0), (2, 4, 6), (1, 0, 0), True),
        ((1, 0, 0), (0, 1, 0), (2, 4, 6), (1.01, 0, 0), False),
        ((1, 0, 0), (0, 1, 0), (np.inf, 4, 6), (1e6, 2, 0), True),
        ((1, 0, 0), (0, 1, 0), (np.inf, 4, 6), (-1e6, 2.01, 0), False),
        # A 45-degree rotation gives a diamond for Block and a circle for
        # Ellipsoid in the xy plane. These points are interior/exterior to both.
        ((1, 1, 0), (-1, 1, 0), (2 * np.sqrt(2), 2 * np.sqrt(2), 2), (1, 0, 0), True),
        ((1, 1, 0), (-1, 1, 0), (2 * np.sqrt(2), 2 * np.sqrt(2), 2), (3, 0, 0), False),
        # Making the diagonal axis infinite gives the strip |y - x| <= 2.
        ((1, 1, 0), (-1, 1, 0), (np.inf, 2 * np.sqrt(2), 2), (100, 101, 0), True),
        ((1, 1, 0), (-1, 1, 0), (np.inf, 2 * np.sqrt(2), 2), (100, 103, 0), False),
        # The skew diagonal's y extent is one; its unbounded x partner
        # must not change that transverse constraint.
        ((1, 0, 0), (1, 1, 0), (np.inf, 2 * np.sqrt(2), 2), (-100, 1, 0), True),
        ((1, 0, 0), (1, 1, 0), (np.inf, 2 * np.sqrt(2), 2), (-100, 1.01, 0), False),
        ((1, 1, 0), (-1, 1, 0), (np.inf,) * 3, (1e6, -1e6, 1e6), True),
    ),
    ids=(
        "finite-boundary",
        "finite-outside",
        "unbounded-boundary",
        "unbounded-outside",
        "rotated-inside",
        "rotated-outside",
        "unbounded-rotated-inside",
        "unbounded-rotated-outside",
        "skew-boundary",
        "skew-outside",
        "all-unbounded-finite-point",
    ),
)
def test_finite_containment_matches_known_geometry(
    shape_class, e1, e2, size, offset, expected
):
    center = np.array((0.25, -0.5, 0.75))
    shape = shape_class(Dielectric(4), center=center, e1=e1, e2=e2, size=size)
    point = center + offset
    default = DefaultMedium(Dielectric(1))
    with np.errstate(all="raise"):
        assert shape.in_object(point) is expected
        np.testing.assert_array_equal(
            shape._contains_points(*point[:, None]), (expected,)
        )
        shape.init(None)
        default.init(None)
        tree = GeomBoxTree((default, shape))
        assert tree.object_of_point(point)[0] is (shape if expected else default)
        materials, _ = tree.material_of_grid(*(np.array((value,)) for value in point))
        assert materials == [(shape if expected else default).material]


def test_unbounded_block_lookup_with_unrelated_partition_candidate():
    default = DefaultMedium(Dielectric(1))
    block = Block(Dielectric(4), size=(np.inf, 2, 2))
    sphere = Sphere(Dielectric(9), center=(10, 0, 0))
    with np.errstate(all="raise"):
        for shape in (default, block, sphere):
            shape.init(None)
        tree = GeomBoxTree((default, block, sphere))
        # With the old truncated bounds, the distant sphere caused a partition
        # that omitted the block at x=5. Correct bounds may avoid that split.
        assert block.in_object((5, 0, 0)) is True
        assert tree.object_of_point((5, 0, 0))[0] is block
        materials, _ = tree.material_of_grid(
            np.array((5, 10), dtype=np.float64),
            np.array((0,), dtype=np.float64),
            np.array((0,), dtype=np.float64),
        )
        assert materials == [block.material, sphere.material]


@pytest.mark.parametrize("shape_class", (Block, Ellipsoid), ids=("block", "ellipsoid"))
@pytest.mark.parametrize(
    ("y", "expected"),
    (((0,), (True, False)), (((0,), (3,)), ((True, False), (False, False)))),
    ids=("one-dimensional", "two-dimensional"),
)
def test_nonfinite_batch_queries_preserve_broadcasting(shape_class, y, expected):
    shape = shape_class(Dielectric(), size=(2, 2, 2))
    with np.errstate(all="raise"):
        matches = shape._contains_points(
            np.array((0, np.inf), dtype=np.float64),
            np.array(y, dtype=np.float64),
            np.array((0,), dtype=np.float64),
        )
        np.testing.assert_array_equal(matches, expected)
