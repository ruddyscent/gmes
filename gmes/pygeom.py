# -*- coding: utf-8 -*-

"""Define geometric primitives and accelerate material lookup on Yee grids."""

from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

type Index3 = tuple[int, int, int]
type Vector3 = Sequence[float] | NDArray[np.float64]
type RealArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
# Pickle state is an intentionally dynamic compatibility boundary.
type PickleState = dict[str, Any]


class _MaterialSpace(Protocol):
    """Grid values required by material initialization."""

    @property
    def dt(self) -> float:
        """Return the simulation time step."""

    @property
    def dr(self) -> RealArray:
        """Return the three spatial step sizes."""

    @property
    def half_size(self) -> RealArray:
        """Return half the simulation-domain extent."""


class _NativeMaterial(Protocol):
    """Minimal native updater returned by material lowering hooks."""

    def attach(self, idx: Index3, parameter: Any) -> None:
        """Attach one native update record."""


def norm(p: Vector3) -> float:
    """Return the Euclidean norm of a three-component vector."""

    return sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)


class Material(object):
    """A base class for material types."""

    def __init__(self, eps_inf: float = 1, mu_inf: float = 1) -> None:
        self.eps_inf = float(eps_inf)
        self.mu_inf = float(mu_inf)

    @property
    def eps_inf(self) -> float:
        """Return the infinite-frequency relative permittivity."""

        return self._eps_inf

    @eps_inf.setter
    def eps_inf(self, value: float) -> None:
        self._eps_inf = float(value)

    @property
    def mu_inf(self) -> float:
        """Return the infinite-frequency relative permeability."""

        return self._mu_inf

    @mu_inf.setter
    def mu_inf(self, value: float) -> None:
        self._mu_inf = float(value)

    def __getstate__(self) -> PickleState:
        d: PickleState = {}
        d["eps_inf"] = self.eps_inf
        d["mu_inf"] = self.mu_inf
        return d

    def __reduce__(self) -> tuple[object, ...]:
        return self.__class__, (), self.__getstate__()

    def __setstate__(self, d: PickleState) -> None:
        self.eps_inf = d["eps_inf"]
        self.mu_inf = d["mu_inf"]

    def display_info(self, indent: int = 0) -> None:
        """Display the parameter values."""
        raise NotImplementedError

    def get_pw_material_ex(
        self,
        idx: Index3,
        coords: Vector3,
        underneath: Material | None = None,
        cmplx: bool = False,
    ) -> _NativeMaterial | None:
        """Return an ElectricParam structure of the given point.

        Arguments:
            idx -- (local) array index of the target point
            coords -- (global) space coordinate of the target point
            complex -- whether the EM field has complex value. Default is False.
            underneath -- underneath material object of the target point.

        """
        raise NotImplementedError

    def get_pw_material_ey(
        self,
        idx: Index3,
        coords: Vector3,
        underneath: Material | None = None,
        cmplx: bool = False,
    ) -> _NativeMaterial | None:
        """Return an ElectricParam structure of the given point.

        Arguments:
            idx -- (local) array index of the target point
            coords -- (global) space coordinate of the target point
            complex -- whether the EM field has complex value. Default is False.
            underneath -- underneath material object of the target point.

        """
        raise NotImplementedError

    def get_pw_material_ez(
        self,
        idx: Index3,
        coords: Vector3,
        underneath: Material | None = None,
        cmplx: bool = False,
    ) -> _NativeMaterial | None:
        """Return an ElectricParam structure of the given point.

        Arguments:
            idx -- (local) array index of the target point
            coords -- (global) space coordinate of the target point
            complex -- whether the EM field has complex value. Default is False.
            underneath -- underneath material object of the target point.

        """
        raise NotImplementedError

    def get_pw_material_hx(
        self,
        idx: Index3,
        coords: Vector3,
        underneath: Material | None = None,
        cmplx: bool = False,
    ) -> _NativeMaterial | None:
        """Return a MagneticParam structure of the given point.

        Arguments:
            idx -- (local) array index of the target point
            coords -- (global) space coordinate of the target point
            complex -- whether the EM field has complex value. Default is False.
            underneath -- underneath material object of the target point.

        """
        raise NotImplementedError

    def get_pw_material_hy(
        self,
        idx: Index3,
        coords: Vector3,
        underneath: Material | None = None,
        cmplx: bool = False,
    ) -> _NativeMaterial | None:
        """Return a MagneticParam structure of the given point.

        Arguments:
            idx -- (local) array index of the target point
            coords -- (global) space coordinate of the target point
            complex -- whether the EM field has complex value. Default is False.
            underneath -- underneath material object of the target point.

        """
        raise NotImplementedError

    def get_pw_material_hz(
        self,
        idx: Index3,
        coords: Vector3,
        underneath: Material | None = None,
        cmplx: bool = False,
    ) -> _NativeMaterial | None:
        """Return a MagneticParam structure of the given point.

        Arguments:
            idx -- (local) array index of the target point
            coords -- (global) space coordinate of the target point
            complex -- whether the EM field has complex value. Default is False.
            underneath -- underneath material object of the target point.

        """

    def init(self, space: _MaterialSpace, param: object | None = None) -> None:
        """Initialize material coefficients for a Cartesian simulation space."""

        raise NotImplementedError


class Compound(object):
    """Marker for materials that refer to an underlying medium."""

    pass


####################################################################
#                                                                  #
#                      Fast geometry routines                      #
#                                                                  #
# Using the geometry list is way too slow, especially when there   #
# are lots of objects to test.                                     #
#                                                                  #
# The basic idea here is twofold. (1) Compute bounding boxes for   #
# each geometric object, for which inclusion tests can be computed #
# quickly. (2) Build a tree that recursively breaks down the unit  #
# cell in half, allowing us to perform searches in logarithmic     #
# time.                                                            #
#                                                                  #
####################################################################


def find_object(
    point: Vector3, geom_list: Sequence[GeometricObject]
) -> tuple[GeometricObject, int]:
    """Find the last object including point in geom_list.

    find_object returns (object, array index). If no object includes
    the given point it returns (geom_list[0], 0).

    """
    i = len(geom_list) - 1
    while i > 0 and geom_list[i].in_object(point) is False:
        i -= 1

    return geom_list[i], i


class GeomBox(object):
    """A bounding box of a geometric object.

    Attributes:
    low -- the coordinates of the lowest vertex
    high -- the coordinates of the highest vertex

    """

    def __init__(self, low: Vector3 = (0, 0, 0), high: Vector3 = (0, 0, 0)) -> None:
        self.low = np.array(low, np.double)
        self.high = np.array(high, np.double)

    def __getstate__(self) -> PickleState:
        d: PickleState = {}
        d["low"] = self.low
        d["high"] = self.high
        return d

    def __reduce__(self) -> tuple[object, ...]:
        return self.__class__, (), self.__getstate__()

    def __setstate__(self, d: PickleState) -> None:
        self.low.setfield(d["low"], np.double)
        self.high.setfield(d["high"], np.double)

    def union(self, box: GeomBox) -> None:
        """Enlarge the box to include the given box."""
        self.low.setfield([min(a, b) for a, b in zip(self.low, box.low)], np.double)
        self.high.setfield([max(a, b) for a, b in zip(self.high, box.high)], np.double)

    def intersection(self, box: GeomBox) -> None:
        """Reduce the box to intersect volume with the given box."""
        self.low.setfield([max(a, b) for a, b in zip(self.low, box.low)], np.double)
        self.high.setfield([min(a, b) for a, b in zip(self.high, box.high)], np.double)

    def add_point(self, point: Vector3) -> None:
        """Enlarge the box to include the given point."""
        self.low.setfield([min(a, b) for a, b in zip(self.low, point)], np.double)
        self.high.setfield([max(a, b) for a, b in zip(self.high, point)], np.double)

    def between(self, x: float, low: float, high: float) -> bool:
        """Return truth of low <= x <= high."""
        return bool(low <= x <= high)

    def in_box(self, point: Vector3) -> bool:
        """Check whether the given point is in this box."""
        truth = (
            self.between(point[0], self.low[0], self.high[0])
            and self.between(point[1], self.low[1], self.high[1])
            and self.between(point[2], self.low[2], self.high[2])
        )

        return bool(truth)

    def _contains_points(self, x: RealArray, y: RealArray, z: RealArray) -> BoolArray:
        return cast(
            BoolArray,
            (self.low[0] <= x)
            & (x <= self.high[0])
            & (self.low[1] <= y)
            & (y <= self.high[1])
            & (self.low[2] <= z)
            & (z <= self.high[2]),
        )

    def overlap(self, box: GeomBox) -> bool:
        """Check whether the given box intersect with the box."""
        truth = (
            (
                self.between(box.low[0], self.low[0], self.high[0])
                or self.between(box.high[0], self.low[0], self.high[0])
                or self.between(self.low[0], box.low[0], box.high[0])
            )
            and (
                self.between(box.low[1], self.low[1], self.high[1])
                or self.between(box.high[1], self.low[1], self.high[1])
                or self.between(self.low[1], box.low[1], box.high[1])
            )
            and (
                self.between(box.low[2], self.low[2], self.high[2])
                or self.between(box.high[2], self.low[2], self.high[2])
                or self.between(self.low[2], box.low[2], box.high[2])
            )
        )

        return bool(truth)

    def divide(self, axis: int, x: float) -> tuple[GeomBox, GeomBox]:
        """Split this box at one coordinate and return the adjacent boxes."""

        high1 = list(self.high)
        high1[axis] = x

        low2 = list(self.low)
        low2[axis] = x

        return GeomBox(self.low, high1), GeomBox(low2, self.high)

    def display_info(self, indent: int = 0) -> None:
        """Print this box's lower and upper coordinates."""

        print(" " * indent, "geom box:", end=" ")
        print("low:", self.low, "high:", self.high)

    def __str__(self) -> str:
        return "low: " + self.low.__str__() + " high: " + self.high.__str__()


class GeomBoxNode(object):
    """Node class which makes up a binary search tree.

    Attributes:
    box -- a bounding box enclosing the volume of this node
    t1 -- left branch from this node
    t2 -- right branch from this node
    geom_list -- a geometric object list overlapping the volume of this node.
    depth -- depth from the root of this binary search tree

    """

    box: GeomBox
    t1: GeomBoxNode | None
    t2: GeomBoxNode | None
    geom_list: tuple[GeometricObject, ...]
    depth: int

    def __init__(
        self, box: GeomBox, geom_list: Sequence[GeometricObject], depth: int
    ) -> None:
        self.box = box
        self.t1, self.t2 = None, None
        self.geom_list = tuple(geom_list)
        self.depth = depth

    def __getstate__(self) -> PickleState:
        d: PickleState = {}
        d["box"] = self.box
        d["t1"] = self.t1
        d["t2"] = self.t2
        d["geom_list"] = self.geom_list
        d["depth"] = self.depth
        return d

    def __reduce__(self) -> tuple[object, ...]:
        return self.__class__, (None, (), 0), self.__getstate__()

    def __setstate__(self, d: PickleState) -> None:
        self.box = deepcopy(d["box"])
        self.t1 = deepcopy(d["t1"])
        self.t2 = deepcopy(d["t2"])
        self.geom_list = deepcopy(d["geom_list"])
        self.depth = d["depth"]


@dataclass(frozen=True)
class GeometryMap:
    """A bounded, backend-neutral material map for one rectilinear tile."""

    material_ids: NDArray[np.int32]
    underlying_ids: NDArray[np.int32]
    geometries: tuple[GeometricObject, ...]
    shape: tuple[int, int, int]
    start: int
    stop: int
    component: object | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.geometries, tuple)
            or not self.geometries
            or len(self.shape) != 3
            or any(not isinstance(length, int) or length < 0 for length in self.shape)
        ):
            raise ValueError("geometry table and three-dimensional shape are required")
        total = int(np.prod(self.shape))
        if (
            not isinstance(self.start, int)
            or not isinstance(self.stop, int)
            or self.start < 0
            or self.stop < self.start
            or self.stop > total
        ):
            raise ValueError("geometry-map tile is out of bounds")
        size = self.stop - self.start
        for name in ("material_ids", "underlying_ids"):
            values = getattr(self, name)
            if (
                not isinstance(values, np.ndarray)
                or values.dtype != np.int32
                or values.ndim != 1
                or not values.flags.c_contiguous
                or len(values) != size
            ):
                raise ValueError(f"{name} must be a contiguous int32 tile array")
        if size and (
            self.material_ids.min() < 0
            or self.material_ids.max() >= len(self.geometries)
            or self.underlying_ids.min() < -1
            or self.underlying_ids.max() >= len(self.geometries)
        ):
            raise ValueError("geometry-map region ID is outside the geometry table")

    @property
    def materials(self) -> tuple[Material, ...]:
        """Return the stable region-ID-to-material table."""
        return tuple(geometry.material for geometry in self.geometries)


class GeomBoxTree(object):
    """A tree for the fast inclusion test of geometric objects within them.

    The tree recursively partitions the unit cell, allowing us to perform
    binary searches for the object containing a give point.

    Attributes:
    root -- root node of the binary search tree

    """

    root: GeomBoxNode

    def __init__(self, geom_list: Sequence[GeometricObject]) -> None:
        box = GeomBox((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))
        self.root = GeomBoxNode(box, geom_list, 0)
        self.branch_out(self.root)

    def __getstate__(self) -> PickleState:
        d: PickleState = {}
        d["root"] = self.root
        return d

    def __reduce__(self) -> tuple[object, ...]:
        return self.__class__, ((),), self.__getstate__()

    def __setstate__(self, d: PickleState) -> None:
        self.root = deepcopy(d["root"])

    def find_best_partition(
        self, node: GeomBoxNode, divide_axis: int
    ) -> tuple[float | None, int, int]:
        """Find the most even object partition along one axis.

        Find the best place to "cut" along the axis divide_axis in
        order to maximally divide the objects between the partitions.
        Upon return, n1 and n2 are the number of objects below and
        above the partition, respectively.

        """
        small = 1e-7
        best_partition = None

        n1 = n2 = len(node.geom_list)

        # Search for the best partition, by checking all possible partitions
        # either just above the high end of an object or just below the low
        # end of an object.

        for geometry in node.geom_list:
            geometry_box = cast(GeomBox, geometry.box)
            curPartition = geometry_box.high[divide_axis] + small
            curN1 = curN2 = 0
            for candidate in node.geom_list:
                candidate_box = cast(GeomBox, candidate.box)
                if candidate_box.low[divide_axis] <= curPartition:
                    curN1 += 1
                if candidate_box.high[divide_axis] >= curPartition:
                    curN2 += 1
            if max(curN1, curN2) < max(n1, n2):
                best_partition = curPartition
                n1 = curN1
                n2 = curN2

        for geometry in node.geom_list:
            geometry_box = cast(GeomBox, geometry.box)
            curPartition = geometry_box.low[divide_axis] - small
            curN1 = curN2 = 0
            for candidate in node.geom_list:
                candidate_box = cast(GeomBox, candidate.box)
                if candidate_box.low[divide_axis] <= curPartition:
                    curN1 += 1
                if candidate_box.high[divide_axis] >= curPartition:
                    curN2 += 1
            if max(curN1, curN2) < max(n1, n2):
                best_partition = curPartition
                n1 = curN1
                n2 = curN2

        return best_partition, n1, n2

    def divide_geom_box_tree(
        self, node: GeomBoxNode
    ) -> tuple[GeomBoxNode | None, GeomBoxNode | None]:
        """Divide box in two, along the axis that maximally partitions the boxes."""
        # Try partitioning along each dimension, counting the
        # number of objects in the partitioned boxes and finding
        # the best partition.
        best = 0
        division: list[tuple[float | None, int, int]] = []
        for i in range(3):
            partition, n1, n2 = self.find_best_partition(node, i)
            division.append((partition, n1, n2))
            if max(division[i][1], division[i][2]) < max(
                division[best][1], division[best][2]
            ):
                best = i

        # Don't do anything if division makes the worst case worse or if
        # it fails to improve the best case:
        if division[best][0] is None:
            return None, None

        partition = cast(float, division[best][0])
        box1, box2 = node.box.divide(best, partition)
        b1GeomList = []
        b2GeomList = []

        for geometry in node.geom_list:
            geometry_box = cast(GeomBox, geometry.box)
            if box1.overlap(geometry_box):
                b1GeomList.append(geometry)
            if box2.overlap(geometry_box):
                b2GeomList.append(geometry)

        return GeomBoxNode(box1, b1GeomList, node.depth + 1), GeomBoxNode(
            box2, b2GeomList, node.depth + 1
        )

    def branch_out(self, node: GeomBoxNode) -> None:
        """Recursively split a geometry-tree node while partitions improve."""

        node.t1, node.t2 = self.divide_geom_box_tree(node)

        if node.t1 is not None and node.t2 is not None:
            self.branch_out(node.t1)
            self.branch_out(node.t2)

    def tree_search(self, node: GeomBoxNode, point: Vector3) -> GeomBoxNode | None:
        """Return the leaf node containing a physical point."""

        if node.box.in_box(point) == False:
            return None
        else:
            if not (node.t1 and node.t2):
                return node
            else:
                if node.t1.box.in_box(point):
                    return self.tree_search(node.t1, point)

                if node.t2.box.in_box(point):
                    return self.tree_search(node.t2, point)

        return None

    def object_of_point(
        self, point: Vector3
    ) -> tuple[GeometricObject, GeometricObject | None]:
        """Return the topmost object and underlying object at a point."""

        leaf = cast(GeomBoxNode, self.tree_search(self.root, point))
        geom_obj, idx = find_object(point, leaf.geom_list)

        # eps_inf and mu_inf of compound material
        # refer to the underneath material.
        if isinstance(geom_obj.material, Compound):
            aux_geom_list = leaf.geom_list[:idx]
            underneath_obj, trash = find_object(point, aux_geom_list)
        else:
            underneath_obj = None

        return geom_obj, underneath_obj

    def material_of_point(self, point: Vector3) -> tuple[Material, Material | None]:
        """Return the topmost material and underlying material at a point."""

        geom_obj, underneath_obj = self.object_of_point(point)

        if underneath_obj:
            underneath_material = underneath_obj.material
        else:
            underneath_material = None

        return geom_obj.material, underneath_material

    def material_of_grid(
        self,
        x_axis: RealArray,
        y_axis: RealArray,
        z_axis: RealArray,
        start: int = 0,
        stop: int | None = None,
    ) -> tuple[list[Material], list[Material | None]]:
        """Return materials for a bounded C-order tile of a rectilinear grid."""
        geometry_map = self.lower_grid(x_axis, y_axis, z_axis, start, stop)
        geometries = geometry_map.geometries
        materials = [geometries[index].material for index in geometry_map.material_ids]
        underlying = [
            None if index < 0 else geometries[index].material
            for index in geometry_map.underlying_ids
        ]
        return materials, underlying

    def lower_grid(
        self,
        x_axis: RealArray,
        y_axis: RealArray,
        z_axis: RealArray,
        start: int = 0,
        stop: int | None = None,
        *,
        component: object | None = None,
    ) -> GeometryMap:
        """Lower a bounded C-order tile to integer geometry identifiers."""
        axes = x_axis, y_axis, z_axis
        for axis in axes:
            if (
                not isinstance(axis, np.ndarray)
                or axis.dtype != np.double
                or axis.ndim != 1
            ):
                raise TypeError("grid axes must be one-dimensional float64 arrays")

        nx = x_axis.shape[0]
        ny = y_axis.shape[0]
        nz = z_axis.shape[0]
        plane = ny * nz
        total = nx * plane

        if stop is None:
            end = total
        else:
            end = stop
        if start < 0 or end < start or end > total:
            raise IndexError("grid tile is out of bounds")

        geometries = self.root.geom_list
        if not geometries:
            raise ValueError("geometry list must contain a default medium")

        linear = np.arange(start, end, dtype=np.intp)
        i, remainder = np.divmod(linear, plane)
        j, k = np.divmod(remainder, nz)
        x = x_axis[i]
        y = y_axis[j]
        z = z_axis[k]
        material_ids = np.full(end - start, -1, dtype=np.int32)
        underlying_ids = np.full(end - start, -1, dtype=np.int32)
        geometry_ids = {
            id(geometry): index for index, geometry in enumerate(geometries)
        }
        all_positions = np.arange(end - start, dtype=np.intp)
        for leaf, leaf_positions in self._leaf_tiles(self.root, x, y, z, all_positions):
            leaf_geometries = leaf.geom_list
            material_ids[leaf_positions] = geometry_ids[id(leaf_geometries[0])]
            for geometry in leaf_geometries[1:]:
                if self._uses_vectorized_predicate(geometry):
                    matches = geometry._contains_points(
                        x[leaf_positions], y[leaf_positions], z[leaf_positions]
                    )
                else:
                    matches = np.fromiter(
                        (
                            geometry.in_object((x[pos], y[pos], z[pos]))
                            for pos in leaf_positions
                        ),
                        dtype=np.bool_,
                        count=len(leaf_positions),
                    )
                matched_positions = leaf_positions[matches]
                if not matched_positions.size:
                    continue

                if isinstance(geometry.material, Compound):
                    underlying_ids[matched_positions] = material_ids[matched_positions]
                else:
                    underlying_ids[matched_positions] = -1
                material_ids[matched_positions] = geometry_ids[id(geometry)]

        return GeometryMap(
            material_ids,
            underlying_ids,
            geometries,
            (nx, ny, nz),
            start,
            end,
            component,
        )

    @staticmethod
    def _uses_vectorized_predicate(geometry: GeometricObject) -> bool:
        geometry_type = type(geometry)
        return geometry_type in _BUILTIN_VECTOR_GEOMETRY_TYPES or (
            getattr(geometry_type, "_gmes_vectorized_geometry", False) is True
        )

    def supports_bulk_lowering(self) -> bool:
        """Return whether every geometry opted into bounded array predicates."""
        return all(
            self._uses_vectorized_predicate(geometry)
            for geometry in self.root.geom_list
        )

    def _leaf_tiles(
        self,
        node: GeomBoxNode,
        x: RealArray,
        y: RealArray,
        z: RealArray,
        positions: NDArray[np.intp],
    ) -> Iterator[tuple[GeomBoxNode, NDArray[np.intp]]]:
        if not positions.size:
            return
        if node.t1 is None or node.t2 is None:
            yield node, positions
            return

        in_first = node.t1.box._contains_points(
            x[positions], y[positions], z[positions]
        )
        first_positions = positions[in_first]
        yield from self._leaf_tiles(node.t1, x, y, z, first_positions)

        remaining = positions[~in_first]
        in_second = node.t2.box._contains_points(
            x[remaining], y[remaining], z[remaining]
        )
        yield from self._leaf_tiles(node.t2, x, y, z, remaining[in_second])

    def display_info(self, node: GeomBoxNode | None = None, indent: int = 0) -> None:
        """Print the geometry tree recursively from a node."""

        if not node:
            node = self.root

        print(" " * indent, "depth:", node.depth, node.box)

        if node.t1 is None or node.t2 is None:
            for i in node.geom_list:
                print(" " * (indent + 5), "bounding box:", i.box)
                i.display_info(indent + 5)

        if node.t1:
            self.display_info(node.t1, indent + 5)
        if node.t2:
            self.display_info(node.t2, indent + 5)


####################################################################
#                                                                  #
#                       Geometric primitives                       #
#                                                                  #
####################################################################


class GeometricObject(object):
    """Base class for geometric object types.

    This class and its descendants are used to specify the solid
    geometric objects that form the structure being simulated. One
    normally does not create objects of type geometric-object directly,
    however; instead, you use one of the subclasses. Recall that
    subclasses inherit the properties of their superclass, so these
    subclasses automatically have the material property (which must be
    specified, since they have no default values). In a two- or one-
    dimensional calculation, only the intersections of the objects with
    the simulation plane or line are considered.

    Attributes:
    material -- Filling up material.
    box -- bounding box enclosing this geometric object

    """

    # Custom classes may explicitly set this to True when their parameters
    # preserve or override `_contains_points()` with equivalent array semantics.
    # The opt-in is inherited by parameter-only subclasses.
    _gmes_vectorized_geometry = False

    center: RealArray
    box: GeomBox | None
    _material: Material | None

    def __init__(self, material: Material | None) -> None:
        """Initialize a geometric object with its filling material.

        Args:
            material: Material filling the object.
        """
        self.material = material
        self.box = None

    @property
    def material(self) -> Material:
        """Return the material filling this object."""

        return cast(Material, self._material)

    @material.setter
    def material(self, value: Material | None) -> None:
        if value is not None and not isinstance(value, Material):
            raise TypeError("material must be a Material instance")
        self._material = value

    def __getstate__(self) -> PickleState:
        d: PickleState = {}
        d["material"] = self.material
        d["box"] = self.box
        return d

    def __reduce__(self) -> tuple[object, ...]:
        return self.__class__, (None,), self.__getstate__()

    def __setstate__(self, d: PickleState) -> None:
        self.material = deepcopy(d["material"])
        self.box = deepcopy(d["box"])

    def init(self, space: _MaterialSpace) -> None:
        """Initialize the material and cache this object's bounding box."""

        self.material.init(space)
        self.box = self.geom_box()

    def geom_box(self) -> GeomBox:
        """Return a bounding box enclosing this geometric object.

        The derived classes should override this method.

        """
        raise NotImplementedError

    def in_object(self, point: Vector3) -> bool:
        """Return whether or not the point is inside.

        Return whether or not the point (in the lattice basis) is
        inside this geometric object. This method additionally requires
        that fixObject has been called on this object (if the lattice
        basis is non-orthogonal).

        """
        raise NotImplementedError

    def _contains_points(self, x: RealArray, y: RealArray, z: RealArray) -> BoolArray:
        return np.fromiter(
            (self.in_object(point) for point in zip(x, y, z, strict=True)),
            dtype=np.bool_,
            count=len(x),
        )

    def display_info(self, indent: int = 0) -> None:
        """Display some information about this geometric object."""
        print(" " * indent, "geometric object")
        print(" " * indent, "center:", self.center)
        if self.material:
            self.material.display_info(indent + 5)


class DefaultMedium(GeometricObject):
    """A geometric object expanding the whole space."""

    def __init__(self, material: Material | None) -> None:
        GeometricObject.__init__(self, material)

    def in_object(self, point: Vector3) -> bool:
        """Return whether a point lies in the default infinite medium."""

        return cast(GeomBox, self.box).in_box(point)

    def _contains_points(self, x: RealArray, y: RealArray, z: RealArray) -> BoolArray:
        return cast(GeomBox, self.box)._contains_points(x, y, z)

    def geom_box(self) -> GeomBox:
        """Return an unbounded box spanning the simulation space."""
        return GeomBox((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))

    def display_info(self, indent: int = 0) -> None:
        """Print the default-medium material summary."""
        print(" " * indent, "default medium")
        if self.material:
            self.material.display_info(indent + 5)


class Cone(GeometricObject):
    """Form a cone or possibly a truncated cone.

    Attributes:
    center -- coordinates of the center of this geometric object
    axis -- unit vector of axis
    height -- length of axis
    box -- bounding box

    """

    def __init__(
        self,
        material: Material,
        center: Vector3 = (0, 0, 0),
        radius2: float = 0,
        axis: Vector3 = (1, 0, 0),
        radius: float = 1,
        height: float = 1,
    ) -> None:
        """Initialize a cone or truncated cone.

        Keyword arguments:
        radius2 -- Radius of the tip of the cone (i.e. the end of the
            cone pointed to by the axis vector). Defaults to zero
            (a "sharp" cone).

        """
        GeometricObject.__init__(self, material)

        if radius < 0:
            msg = "radius must be non-negative."
            raise ValueError(msg)
        else:
            self.radius = float(radius)  # low side radius

        if radius2 < 0:
            msg = "radius2 must be non-negative."
            raise ValueError(msg)
        else:
            self.radius2 = float(radius2)  # high side radius

        self.center = np.array(center, np.double)
        self.axis = np.array(axis, np.double) / norm(axis)
        self.height = float(height)

    def __getstate__(self) -> PickleState:
        d = GeometricObject.__getstate__(self)
        d["radius"] = self.radius
        d["radius2"] = self.radius2
        d["height"] = self.height
        d["center"] = self.center
        d["axis"] = self.axis
        return d

    def __setstate__(self, d: PickleState) -> None:
        GeometricObject.__setstate__(self, d)
        self.radius = d["radius"]
        self.radius2 = d["radius2"]
        self.height = d["height"]
        self.center.setfield(d["center"], np.double)
        self.axis.setfield(d["axis"], np.double)

    def in_object(self, point: Vector3) -> bool:
        """Check whether the given point is in this Cone."""
        rx = point[0] - self.center[0]
        ry = point[1] - self.center[1]
        rz = point[2] - self.center[2]
        proj = self.axis[0] * rx + self.axis[1] * ry + self.axis[2] * rz
        if abs(proj) <= 0.5 * self.height:
            if self.radius2 == self.radius == np.inf:
                truth = True
            else:
                radius = self.radius
                radius += (proj / self.height + 0.5) * (self.radius2 - radius)
                perpendicular_squared = rx * rx + ry * ry + rz * rz - proj * proj
                if perpendicular_squared < 0:
                    perpendicular_squared = 0
                truth = sqrt(perpendicular_squared) <= abs(radius)
        else:
            truth = False

        return bool(truth)

    def _contains_points(self, x: RealArray, y: RealArray, z: RealArray) -> BoolArray:
        rx = x - self.center[0]
        ry = y - self.center[1]
        rz = z - self.center[2]
        projection = self.axis[0] * rx + self.axis[1] * ry + self.axis[2] * rz
        axial = np.abs(projection) <= 0.5 * self.height
        if self.radius2 == self.radius == np.inf:
            return cast(BoolArray, axial)

        radius = self.radius + (projection / self.height + 0.5) * (
            self.radius2 - self.radius
        )
        perpendicular_squared = np.maximum(
            0, rx * rx + ry * ry + rz * rz - projection * projection
        )
        return cast(
            BoolArray, axial & (np.sqrt(perpendicular_squared) <= np.abs(radius))
        )

    def display_info(self, indent: int = 0) -> None:
        """Print this cone's geometry and material."""
        print(" " * indent, "cone")
        print(" " * indent, end=" ")
        print("center:", self.center, end=" ")
        print("radius:", self.radius, end=" ")
        print("height:", self.height, end=" ")
        print("axis:", self.axis, end=" ")
        print("radius2:", self.radius2)
        if self.material:
            self.material.display_info(indent + 5)

    def geom_box(self) -> GeomBox:
        """Return an axis-aligned box enclosing the cone."""
        h = 0.5 * self.height

        # Project a radius perpendicular to the axis onto each Cartesian
        # direction. Clamp roundoff before taking the square root.
        radial = np.sqrt(np.maximum(0, 1 - self.axis * self.axis))
        extent1 = np.zeros((3,), np.double)
        extent2 = np.zeros((3,), np.double)
        np.multiply(self.radius, radial, out=extent1, where=radial != 0)
        np.multiply(self.radius2, radial, out=extent2, where=radial != 0)

        # bounding box for -h*axis cylinder end
        tmpBox1 = GeomBox(low=self.center, high=self.center)
        tmpBox1.low -= h * self.axis + extent1
        tmpBox1.high -= h * self.axis - extent1

        # bounding box for +h*axis cylinder end
        tmpBox2 = GeomBox(low=self.center, high=self.center)
        tmpBox2.low += h * self.axis - extent2
        tmpBox2.high += h * self.axis + extent2

        tmpBox1.union(tmpBox2)

        return tmpBox1


class Cylinder(Cone):
    """Form a cylinder."""

    def __init__(
        self,
        material: Material,
        center: Vector3 = (0, 0, 0),
        axis: Vector3 = (0, 0, 1),
        radius: float = 1,
        height: float = 1,
    ) -> None:
        """Initialize a finite cylinder.

        Keyword arguments:
            material -- The material that the object is made of.
                No default.
            center -- Center point of the object. Default is (0,0,0).
            axis -- Direction of the cylinder's axis; the length of
                this vector is ignored. Defaults to point parallel to
                the z axis i.e., (0,0,1).
            radius -- Radius of the cylinder's cross-section. Default is 1.
            height -- Length of the cylinder along its axis. Default is 1.

        """
        Cone.__init__(self, material, center, radius, axis, radius, height)

    def display_info(self, indent: int = 0) -> None:
        """Display information of this cylinder."""
        print(" " * indent, "cylinder")
        print(" " * indent, end=" ")
        print("center:", self.center, end=" ")
        print("radius:", self.radius, end=" ")
        print("height:", self.height, end=" ")
        print("axis:", self.axis)
        if self.material:
            self.material.display_info(indent + 5)


class Block(GeometricObject):
    """Form a parallelpiped(i.e., a brick, possibly with non-orthogonal axes.)"""

    def __init__(
        self,
        material: Material,
        center: Vector3 = (0, 0, 0),
        e1: Vector3 = (1, 0, 0),
        e2: Vector3 = (0, 1, 0),
        e3: Vector3 = (0, 0, 1),
        size: Vector3 = (1, 1, 1),
    ) -> None:
        """Initialize a parallelepiped from its center, axes, and edge lengths.

        Keyword arguments:
            center -- center location. Default is (0, 0, 0).
            e1, e2, e3 -- The directions of the axes of the block; the
                lengths of these vectors are ignored. Must be linearly
                independent. They default to the three Cartesian axis.
            size -- The lengths of the block edges along each of its
                three axes. Default is (1, 1, 1).

        """
        GeometricObject.__init__(self, material)

        self.center = np.array(center, np.double)

        self.e1 = np.array(e1, np.double) / norm(e1)
        self.e2 = np.array(e2, np.double) / norm(e2)
        self.e3 = np.array(e3, np.double) / norm(e3)
        self.size = np.array(size, np.double)

        basis = np.column_stack((self.e1, self.e2, self.e3))
        self.projection_matrix = np.linalg.inv(basis)

    def __getstate__(self) -> PickleState:
        d = GeometricObject.__getstate__(self)
        d["center"] = self.center
        d["e1"] = self.e1
        d["e2"] = self.e2
        d["e3"] = self.e3
        d["size"] = self.size
        d["pm"] = self.projection_matrix
        return d

    def __setstate__(self, d: PickleState) -> None:
        GeometricObject.__setstate__(self, d)
        self.center.setfield(d["center"], np.double)
        self.e1.setfield(d["e1"], np.double)
        self.e2.setfield(d["e2"], np.double)
        self.e3.setfield(d["e3"], np.double)
        self.size.setfield(d["size"], np.double)
        basis = np.column_stack((self.e1, self.e2, self.e3))
        self.projection_matrix.setfield(np.linalg.inv(basis), np.double)

    def in_object(self, point: Vector3) -> bool:
        """Check whether the given point is in this block."""
        rx = point[0] - self.center[0]
        ry = point[1] - self.center[1]
        rz = point[2] - self.center[2]
        proj0 = (
            self.projection_matrix[0, 0] * rx
            + self.projection_matrix[0, 1] * ry
            + self.projection_matrix[0, 2] * rz
        )
        proj1 = (
            self.projection_matrix[1, 0] * rx
            + self.projection_matrix[1, 1] * ry
            + self.projection_matrix[1, 2] * rz
        )
        proj2 = (
            self.projection_matrix[2, 0] * rx
            + self.projection_matrix[2, 1] * ry
            + self.projection_matrix[2, 2] * rz
        )
        truth = (
            abs(proj0) <= 0.5 * self.size[0]
            and abs(proj1) <= 0.5 * self.size[1]
            and abs(proj2) <= 0.5 * self.size[2]
        )

        return bool(truth)

    def _project_points(
        self, x: RealArray, y: RealArray, z: RealArray
    ) -> tuple[RealArray, RealArray, RealArray]:
        rx = x - self.center[0]
        ry = y - self.center[1]
        rz = z - self.center[2]
        return tuple(
            self.projection_matrix[row, 0] * rx
            + self.projection_matrix[row, 1] * ry
            + self.projection_matrix[row, 2] * rz
            for row in range(3)
        )

    def _contains_points(self, x: RealArray, y: RealArray, z: RealArray) -> BoolArray:
        projection = self._project_points(x, y, z)
        return cast(
            BoolArray,
            (np.abs(projection[0]) <= 0.5 * self.size[0])
            & (np.abs(projection[1]) <= 0.5 * self.size[1])
            & (np.abs(projection[2]) <= 0.5 * self.size[2]),
        )

    def geom_box(self) -> GeomBox:
        """Return a GeomBox for this block."""
        tmpBox = GeomBox(low=self.center, high=self.center)
        # enlarge the box to be big enough to contain all 8 corners
        # of the block.
        s1 = self.size[0] * self.e1
        s2 = self.size[1] * self.e2
        s3 = self.size[2] * self.e3

        corner = self.center - 0.5 * (s1 + s2 + s3)

        tmpBox.add_point(corner)
        tmpBox.add_point(corner + s1)
        tmpBox.add_point(corner + s2)
        tmpBox.add_point(corner + s3)
        tmpBox.add_point(corner + s1 + s2)
        tmpBox.add_point(corner + s1 + s3)
        tmpBox.add_point(corner + s3 + s2)
        tmpBox.add_point(corner + s1 + s2 + s3)

        return tmpBox

    def display_info(self, indent: int = 0) -> None:
        """Display information of this block."""
        print(" " * indent, "block")
        print(" " * indent, end=" ")
        print("center:", self.center, end=" ")
        print("size:", self.size, end=" ")
        print("axes:", self.e1, self.e2, self.e3)
        if self.material:
            self.material.display_info(indent + 5)


class Ellipsoid(Block):
    """Form an ellipsoid."""

    def __init__(
        self,
        material: Material,
        center: Vector3 = (0, 0, 0),
        e1: Vector3 = (1, 0, 0),
        e2: Vector3 = (0, 1, 0),
        e3: Vector3 = (0, 0, 1),
        size: Vector3 = (1, 1, 1),
    ) -> None:
        """Initialize an ellipsoid from its center, principal axes, and diameters."""

        Block.__init__(self, material, center, e1, e2, e3, size)
        self.inverse_semi_axes = 2 / np.array(size, np.double)

    def __getstate__(self) -> PickleState:
        d = Block.__getstate__(self)
        d["isa"] = self.inverse_semi_axes
        return d

    def __setstate__(self, d: PickleState) -> None:
        Block.__setstate__(self, d)
        self.inverse_semi_axes = d["isa"]

    def in_object(self, point: Vector3) -> bool:
        """Check whether the given point is in this ellipsoid."""
        rx = point[0] - self.center[0]
        ry = point[1] - self.center[1]
        rz = point[2] - self.center[2]
        q0 = self.inverse_semi_axes[0] * (
            self.projection_matrix[0, 0] * rx
            + self.projection_matrix[0, 1] * ry
            + self.projection_matrix[0, 2] * rz
        )
        q1 = self.inverse_semi_axes[1] * (
            self.projection_matrix[1, 0] * rx
            + self.projection_matrix[1, 1] * ry
            + self.projection_matrix[1, 2] * rz
        )
        q2 = self.inverse_semi_axes[2] * (
            self.projection_matrix[2, 0] * rx
            + self.projection_matrix[2, 1] * ry
            + self.projection_matrix[2, 2] * rz
        )
        truth = q0 * q0 + q1 * q1 + q2 * q2 <= 1

        return bool(truth)

    def _contains_points(self, x: RealArray, y: RealArray, z: RealArray) -> BoolArray:
        projection = self._project_points(x, y, z)
        q0 = self.inverse_semi_axes[0] * projection[0]
        q1 = self.inverse_semi_axes[1] * projection[1]
        q2 = self.inverse_semi_axes[2] * projection[2]
        return cast(BoolArray, q0 * q0 + q1 * q1 + q2 * q2 <= 1)

    def display_info(self, indent: int = 0) -> None:
        """Display information of this ellipsoid."""
        print(" " * indent, "ellipsoid")
        print(" " * indent, end=" ")
        print("center:", self.center, end=" ")
        print("size:", self.size, end=" ")
        print("axis:", self.e1, self.e2, self.e3)
        if self.material:
            self.material.display_info(indent + 5)


class Sphere(GeometricObject):
    """Form a sphere.

    Attributes:
    radius -- Radius of the sphere.

    """

    def __init__(
        self, material: Material, center: Vector3 = (0, 0, 0), radius: float = 1
    ) -> None:
        """Initialize a sphere.

        Keyword arguments:
        radius -- Radius of the sphere. Default is 1.
        material -- The material that the object is made of.
            No default.
        center -- Center point of the object. Default is (0,0,0).

        """
        GeometricObject.__init__(self, material)

        if radius < 0:
            msg = "radius must be non-negative."
            raise ValueError(msg)
        else:
            self.radius = float(radius)

        self.center = np.array(center, np.double)

    def __getstate__(self) -> PickleState:
        d = GeometricObject.__getstate__(self)
        d["radius"] = self.radius
        d["center"] = self.center
        return d

    def __setstate__(self, d: PickleState) -> None:
        GeometricObject.__setstate__(self, d)
        self.radius = d["radius"]
        self.center.setfield(d["center"], np.double)

    def geom_box(self) -> GeomBox:
        """Return GeomBox for the sphere."""
        box = GeomBox(low=self.center, high=self.center)

        box.low -= self.radius
        box.high += self.radius

        return box

    def in_object(self, point: Vector3) -> bool:
        """Check whether the given point is in the sphere."""
        rx = point[0] - self.center[0]
        ry = point[1] - self.center[1]
        rz = point[2] - self.center[2]
        truth = rx * rx + ry * ry + rz * rz <= self.radius * self.radius

        return bool(truth)

    def _contains_points(self, x: RealArray, y: RealArray, z: RealArray) -> BoolArray:
        rx = x - self.center[0]
        ry = y - self.center[1]
        rz = z - self.center[2]
        return cast(BoolArray, rx * rx + ry * ry + rz * rz <= self.radius * self.radius)

    def display_info(self, indent: int = 0) -> None:
        """Display information of the sphere."""
        print(" " * indent, "sphere")
        print(" " * indent, end=" ")
        print("center:", self.center, end=" ")
        print("radius:", self.radius)
        if self.material:
            self.material.display_info(indent + 5)


class Shell(GeometricObject):
    """Form a boundary."""

    def __init__(
        self,
        material: Material,
        center: Vector3 = (0, 0, 0),
        size: Vector3 | None = None,
        thickness: float = 1,
        plus_x: bool = True,
        minus_x: bool = True,
        plus_y: bool = True,
        minus_y: bool = True,
        plus_z: bool = True,
        minus_z: bool = True,
    ) -> None:
        """Initialize selected faces of a rectangular boundary shell.

        Keyword arguments:
            material -- The filling material
            center -- coordinates of the center of this geometric object. Default is (0,0,0).
            size -- size of ot the shell. Default is None.
            thickness -- The spatial thickness of the Shell (the
                distance between inner and outer surface)
                Default value is 1.
            plus_x -- Specify whether the high of the boundary in
                direction x is set. Default is True.
            minus_x -- Specify whether the low of the boundary in
                direction x is set. Default is True.
            plus_y -- Specify whether the high of the boundary in
                direction y is set. Default is True.
            minus_y -- Specify whether the low of the boundary in
                direction y is set. Default is True.
            plus_z -- Specify whether the high of the boundary in
                direction z is set. Default is True.
            minus_z -- Specify whether the low of the boundary in
                direction z is set. Default is True.

        """
        GeometricObject.__init__(self, material)

        self.center = np.array(center, np.double)
        self.d = float(thickness)

        self.box_list: list[GeomBox] = []

        self.minus_x, self.plus_x = minus_x, plus_x
        self.minus_y, self.plus_y = minus_y, plus_y
        self.minus_z, self.plus_z = minus_z, plus_z

        if size is None:
            self.half_size = np.zeros((3,), np.double)
            self.boundary = True
        else:
            self.half_size = np.array([0.5 * value for value in size], np.double)
            self.boundary = False

        # do someting for the PML derived class?

    def __getstate__(self) -> PickleState:
        d = GeometricObject.__getstate__(self)
        d["center"] = self.center
        d["half_size"] = self.half_size
        d["d"] = self.d
        d["box_list"] = self.box_list
        d["minus_x"] = self.minus_x
        d["plus_x"] = self.plus_x
        d["minus_y"] = self.minus_y
        d["plus_y"] = self.plus_y
        d["minus_z"] = self.minus_z
        d["plus_z"] = self.plus_z
        d["boundary"] = self.boundary
        return d

    def __setstate__(self, d: PickleState) -> None:
        GeometricObject.__setstate__(self, d)
        self.center.setfield(d["center"], np.double)
        self.half_size.setfield(d["half_size"], np.double)
        self.d = d["d"]
        self.box_list = deepcopy(d["box_list"])
        self.minus_x = d["minus_x"]
        self.plus_x = d["plus_x"]
        self.minus_y = d["minus_y"]
        self.plus_y = d["plus_y"]
        self.minus_z = d["minus_z"]
        self.plus_z = d["plus_z"]
        self.boundary = d["boundary"]

    def init(self, space: _MaterialSpace) -> None:
        if self.boundary:
            self.half_size.setfield(space.half_size, np.double)

        for i in range(3):
            if 2 * self.half_size[i] < space.dr[i]:
                self.half_size[i] = 0.5 * space.dr[i]

        if 2 * self.half_size[0] <= space.dr[0]:
            self.plus_x = False
            self.minus_x = False

        if 2 * self.half_size[1] <= space.dr[1]:
            self.plus_y = False
            self.minus_y = False

        if 2 * self.half_size[2] <= space.dr[2]:
            self.plus_z = False
            self.minus_z = False

        low: Vector3
        high: Vector3
        if self.plus_x:
            low = (
                self.center[0] + self.half_size[0] - self.d,
                self.center[1] - self.half_size[1],
                self.center[2] - self.half_size[2],
            )
            high = self.center + self.half_size
            self.box_list.append(GeomBox(low, high))

        if self.minus_x:
            low = self.center - self.half_size
            high = (
                self.center[0] - self.half_size[0] + self.d,
                self.center[1] + self.half_size[1],
                self.center[2] + self.half_size[2],
            )
            self.box_list.append(GeomBox(low, high))

        if self.plus_y:
            low = (
                self.center[0] - self.half_size[0],
                self.center[1] + self.half_size[1] - self.d,
                self.center[2] - self.half_size[2],
            )
            high = self.center + self.half_size
            self.box_list.append(GeomBox(low, high))

        if self.minus_y:
            low = self.center - self.half_size
            high = (
                self.center[0] + self.half_size[0],
                self.center[1] - self.half_size[1] + self.d,
                self.center[2] + self.half_size[2],
            )
            self.box_list.append(GeomBox(low, high))

        if self.plus_z:
            low = (
                self.center[0] - self.half_size[0],
                self.center[1] - self.half_size[1],
                self.center[2] + self.half_size[2] - self.d,
            )
            high = self.center + self.half_size
            self.box_list.append(GeomBox(low, high))

        if self.minus_z:
            low = self.center - self.half_size
            high = (
                self.center[0] + self.half_size[0],
                self.center[1] + self.half_size[1],
                self.center[2] - self.half_size[2] + self.d,
            )
            self.box_list.append(GeomBox(low, high))

        self.material.init(space, (self.center, self.half_size, self.d))
        self.box = self.geom_box()

    def in_object(self, point: Vector3) -> bool:
        for box in self.box_list:
            if box.in_box(point):
                return True
        return False

    def _contains_points(self, x: RealArray, y: RealArray, z: RealArray) -> BoolArray:
        matches = np.zeros(len(x), dtype=np.bool_)
        for box in self.box_list:
            matches |= box._contains_points(x, y, z)
        return matches

    def geom_box(self) -> GeomBox:
        return GeomBox(self.center - self.half_size, self.center + self.half_size)

    def display_info(self, indent: int = 0) -> None:
        print(" " * indent, "shell")
        print(" " * indent, "center:", self.center)
        print(" " * indent, "size:", 2 * self.half_size)
        print(" " * indent, end=" ")
        print("+x:", self.plus_x, "-x:", self.minus_x, end=" ")
        print("+y:", self.plus_y, "-y:", self.minus_y, end=" ")
        print("+z:", self.plus_z, "-z:", self.minus_z)
        if self.material:
            self.material.display_info(indent + 5)


_BUILTIN_VECTOR_GEOMETRY_TYPES = (
    DefaultMedium,
    Cone,
    Cylinder,
    Block,
    Ellipsoid,
    Sphere,
    Shell,
)
