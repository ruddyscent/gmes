#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define Cartesian simulation grids and map physical coordinates to Yee cells."""

# This code is based on libctl 3.0.2.

from collections.abc import Collection, Sequence
from copy import deepcopy
from typing import Any, Protocol, TypeVar, cast

import numpy as np
from numpy import array, dot, empty, inf, zeros
from numpy.typing import NDArray

type Index3 = tuple[int, int, int]
type Shape3 = tuple[int, int, int]
type CoordinateScalar = float | np.float64
type Coordinate3 = tuple[CoordinateScalar, CoordinateScalar, CoordinateScalar]
type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]
type FieldArray = RealArray | ComplexArray
type IndexArray = NDArray[np.intp]
type IndexLike = Sequence[int] | IndexArray
type Resolution = float | Sequence[float] | NDArray[np.float64]

_T = TypeVar("_T")


class _CartComm(Protocol):
    """Cartesian metadata and local communicator operations."""

    def Get_topo(self) -> tuple[Index3, Index3, Index3]:
        """Return Cartesian dimensions, periods, and coordinates."""

    def Get_size(self) -> int:
        """Return the communicator size."""

    def Shift(self, direction: int, disp: int) -> tuple[int, int]:
        """Return source and destination ranks for a Cartesian shift."""

    def sendrecv(
        self,
        sendbuf: _T,
        dest: int = 0,
        sendtag: int = 0,
        recvbuf: object | None = None,
        source: int = 0,
        recvtag: int = 0,
        status: object | None = None,
    ) -> _T | None:
        """Exchange an object with a neighboring rank."""

    def bcast(self, obj: _T | None = None, root: int = 0) -> _T | None:
        """Broadcast an object from the root rank."""

    def allgather(self, obj: _T | None = None) -> list[_T | None]:
        """Gather an object from every rank."""


# GMES modules
from . import constant as const
from .pygeom import *

type ComponentType = type[const.Component]

_BUILTIN_GEOMETRY_TYPES = (
    DefaultMedium,
    Cone,
    Cylinder,
    Block,
    Ellipsoid,
    Sphere,
    Shell,
)


class AuxiCartComm(object):
    """Auxiliary MPI Cartesian communicator for the absence of MPI implementation.

    Make an instance with default parameters only.

    Attributes:
    dims -- size of mpi Cartesian communicator
    ndims -- dimensionality of this Cartesian topology

    """

    def __init__(
        self,
        dims: Index3 = (1, 1, 1),
        periods: Index3 | None = None,
        reorder: Index3 = (0, 0, 0),
    ) -> None:
        """Constructor.

        Keyword arguments:
        dims -- dimensions of the communicator (default (1,1,1))
        periods -- type: 3-int tuple, default (1,1,1)
        reorder -- default (0,0,0)

        """
        self.rank = 0
        self.dim = 3
        if periods:
            cyclic = tuple(map(int, periods))
        else:
            cyclic = (0, 0, 0)
        self.topo = ((1, 1, 1), cyclic, (0, 0, 0))

    def Get_cart_rank(self, coords: Sequence[int]) -> int:
        """Return the sole rank for a Cartesian coordinate.

        Args:
            coords: Cartesian process coordinate, ignored by the serial fallback.

        Returns:
            The integer rank 0.
        """

        return 0

    def Get_coords(self, rank: int) -> Index3:
        """Get local or remote grid coordinate.

        Keyword arguments:
        rank -- local rank

        """
        return 0, 0, 0

    def Get_dim(self) -> int:
        """Return the number of Cartesian process dimensions."""

        return self.dim

    def Get_topo(self) -> tuple[Index3, Index3, Index3]:
        """Return MPI-compatible dimensions, periods, and local coordinates."""

        return cast(tuple[Index3, Index3, Index3], self.topo)

    def Get_size(self) -> int:
        """Return the serial communicator size of one."""

        return 1

    def Shift(self, direction: int, disp: int) -> tuple[int, int]:
        """Get source/destination with specified shift.

        Keyword arguments:
        direction -- 0 <= Dimension to move < ndims
        displacement -- steps to take in that dimension

        """
        if self.topo[1][direction]:
            return 0, 0
        else:
            return -1, -1

    def sendrecv(
        self,
        sendbuf: _T,
        dest: int = 0,
        sendtag: int = 0,
        recvbuf: object | None = None,
        source: int = 0,
        recvtag: int = 0,
        status: object | None = None,
    ) -> _T | None:
        """Mimic Sendrecv method.

        All arguments except message are ignored.

        """
        if dest == -1 or source == -1:
            return None
        else:
            return sendbuf

    def reduce(self, value: _T, root: int = 0, op: object | None = None) -> _T:
        """Mimic reduce method."""
        return value

    def bcast(self, obj: _T | None = None, root: int = 0) -> _T | None:
        """Mimic bcast method."""
        return obj

    def allgather(self, obj: _T | None = None) -> list[_T | None]:
        """Mimic allgather method."""
        return [obj]

    def recv(self, source: int | None = None) -> None:
        """Return no message in the single-process fallback."""
        return None

    def send(self, obj: object, dest: int = 0) -> None:
        """Accept a no-op send in the single-process fallback."""


class Cartesian(object):
    """Define the calculation space with Cartesian coordinates.

    Attributes:
    half_size -- the half size of whole calculation volume
    res -- number of sections of one unit length
    dx, dy, dz -- the space differentials
    dt -- the time differential
    whole_field_size -- the total array size for the each component of
        the electromagnetic field except the communication buffers
    my_id -- local rank (0 for a serial grid)
    numprocs -- number of ranks (1 for a serial grid)
    cart_comm -- Cartesian topology metadata and local communicator operations
    my_cart_idx -- the coordinates of this rank in the Cartesian topology
    general_field_size -- the general array size for the each component of
        the electromagnetic field except the communication buffers
    my_field_size -- the specific array size for the each component of
        the electromagnetic field of this node except the communication buffers

    """

    cart_comm: _CartComm
    dt: float

    def __init__(
        self, size: Vector3, resolution: Resolution = 15, parallel: bool = False
    ) -> None:
        """Constructor

        Keyword arguments:
        size -- a length three sequence consists of non-negative numbers
        resolution -- number of sections of one unit. scalar or 3-tuple
            (default 15)
        parallel -- must be False; use TorchDistributedSimulation for
            distributed execution

        """
        if parallel:
            raise NotImplementedError(
                "Cartesian(parallel=True) is unsupported; "
                "use TorchDistributedSimulation for distributed execution"
            )

        try:
            if len(cast(Sequence[float], resolution)) == 3:
                self.res = array(resolution, np.double)
        except TypeError:
            self.res = array((resolution,) * 3, np.double)

        self.dr = array(1 / self.res, np.double)

        self.half_size = 0.5 * array(size, np.double)

        for i, v in enumerate(self.dr):
            if self.half_size[i] == 0:
                self.half_size[i] = 0.5 * v

        # the size of the whole field arrays
        self.whole_field_size = array((2 * self.half_size * self.res).round(), np.intp)

        self.my_id = 0
        self.numprocs = 1
        self.cart_comm = AuxiCartComm((1, 1, 1), (1, 1, 1))

        self.my_cart_idx = self.cart_comm.Get_topo()[2]

        # Usually the my_field_size is general_field_size,
        # except the last node in each dimension.
        self.general_field_size = self.whole_field_size // self.cart_comm.Get_topo()[0]

        # my_field_size may be different than general_field_size at the last
        # node in each dimension.
        self.my_field_size = self.get_my_field_size()
        self.global_field_offset = self.general_field_size * self.my_cart_idx

    def bcast(self, obj: _T | None = None, root: int | None = None) -> _T | None:
        """Return the local object; multi-rank object broadcasts are unsupported."""
        if self.numprocs != 1 or self.cart_comm.Get_size() != 1:
            raise NotImplementedError(
                "Cartesian.bcast() supports serial grids only; "
                "use TorchDistributedSimulation for distributed execution"
            )

        return obj

    def get_my_field_size(self) -> IndexArray:
        """Return the field size of this node.

        This method depends on
        self.general_field_size
        self.whole_field_size
        self.my_cart_idx

        """
        field_size = empty(3, np.intp)
        dims = self.cart_comm.Get_topo()[0]

        for i in range(3):
            # At the last node of that dimension.
            if self.my_cart_idx[i] == dims[i] - 1:
                field_size[i] = self.whole_field_size[i] - (
                    self.my_cart_idx[i] * self.general_field_size[i]
                )
            else:
                field_size[i] = self.general_field_size[i]

        return field_size

    def find_best_deploy(self) -> Index3:
        """Return the minimum load deploy of the nodes.

        This method depends on
        self.numprocs

        """
        best_partition: tuple[int, int, int] = (1, 1, 1)
        min_load = inf

        factors = [i for i in range(1, self.numprocs + 1) if self.numprocs % i == 0]
        for l in factors:
            for m in factors:
                if l * m > self.numprocs:
                    break
                n = self.numprocs // (l * m)
                if l * m * n == self.numprocs:
                    tmp_load = self.load_metric(l, m, n)
                    if tmp_load < min_load:
                        best_partition = l, m, n
                        min_load = tmp_load

        return best_partition

    def load_metric(self, l: int, m: int, n: int) -> float:
        """Estimate the load on a node.

        Keyword arguments:
        l, m, n -- the number of node in each direction

        This method depends on
        self.whole_field_size

        """
        # network load ratio compared to CPU
        R = 1000

        cpu_load = (
            self.whole_field_size[0]
            * self.whole_field_size[1]
            * self.whole_field_size[2]
        ) / (l * m * n)
        net_load = (
            4
            * R
            * (
                self.whole_field_size[0] / l * self.whole_field_size[1] / m
                + self.whole_field_size[1] / m * self.whole_field_size[2] / n
                + self.whole_field_size[2] / n * self.whole_field_size[0] / l
            )
        )

        return cast(float, cpu_load + net_load)

    def _get_em_field_storage(self, shape: Sequence[int], cmplx: bool) -> FieldArray:
        if cmplx:
            return zeros(shape, complex)
        else:
            return zeros(shape, np.double)

    def component_coordinate_axes(
        self, component: ComponentType, shape: Sequence[int]
    ) -> tuple[RealArray, RealArray, RealArray]:
        """Return global coordinate axes for a local Yee-grid field."""
        offsets = {
            const.Ex: (0.5, 0.0, 0.0),
            const.Ey: (0.0, 0.5, 0.0),
            const.Ez: (0.0, 0.0, 0.5),
            const.Hx: (0.0, -0.5, -0.5),
            const.Hy: (-0.5, 0.0, -0.5),
            const.Hz: (-0.5, -0.5, 0.0),
        }
        try:
            component_offsets = offsets[component]
        except KeyError as error:
            raise ValueError("unknown Yee-grid component") from error

        global_origin = self.global_field_offset
        return tuple(
            (
                np.arange(length, dtype=np.double)
                + global_origin[axis]
                + component_offsets[axis]
            )
            * self.dr[axis]
            - self.half_size[axis]
            for axis, length in enumerate(shape)
        )

    def get_ex_storage(
        self, field_compnt: Collection[ComponentType], cmplx: bool = False
    ) -> FieldArray:
        """Return an initialized array for Ex field component."""
        if const.Ex in field_compnt:
            shape = (
                self.my_field_size[0],
                self.my_field_size[1] + 1,
                self.my_field_size[2] + 1,
            )
        else:
            shape = (1, 1, 1)

        return self._get_em_field_storage(shape, cmplx)

    def get_ey_storage(
        self, field_compnt: Collection[ComponentType], cmplx: bool = False
    ) -> FieldArray:
        """Return an initialized array for Ey field component."""
        if const.Ey in field_compnt:
            shape = (
                self.my_field_size[0] + 1,
                self.my_field_size[1],
                self.my_field_size[2] + 1,
            )
        else:
            shape = (1, 1, 1)

        return self._get_em_field_storage(shape, cmplx)

    def get_ez_storage(
        self, field_compnt: Collection[ComponentType], cmplx: bool = False
    ) -> FieldArray:
        """Return an initialized array for Ez field component."""
        if const.Ez in field_compnt:
            shape = (
                self.my_field_size[0] + 1,
                self.my_field_size[1] + 1,
                self.my_field_size[2],
            )
        else:
            shape = (1, 1, 1)

        return self._get_em_field_storage(shape, cmplx)

    def get_hx_storage(
        self, field_compnt: Collection[ComponentType], cmplx: bool = False
    ) -> FieldArray:
        """Return an initialized array for Hx field component."""
        if const.Hx in field_compnt:
            shape = (
                self.my_field_size[0],
                self.my_field_size[1] + 1,
                self.my_field_size[2] + 1,
            )
        else:
            shape = (1, 1, 1)

        return self._get_em_field_storage(shape, cmplx)

    def get_hy_storage(
        self, field_compnt: Collection[ComponentType], cmplx: bool = False
    ) -> FieldArray:
        """Return an initialized array for Hy field component."""
        if const.Hy in field_compnt:
            shape = (
                self.my_field_size[0] + 1,
                self.my_field_size[1],
                self.my_field_size[2] + 1,
            )
        else:
            shape = (1, 1, 1)

        return self._get_em_field_storage(shape, cmplx)

    def get_hz_storage(
        self, field_compnt: Collection[ComponentType], cmplx: bool = False
    ) -> FieldArray:
        """Return an initialized array for Hz field component."""
        if const.Hz in field_compnt:
            shape = (
                self.my_field_size[0] + 1,
                self.my_field_size[1] + 1,
                self.my_field_size[2],
            )
        else:
            shape = (1, 1, 1)

        return self._get_em_field_storage(shape, cmplx)

    def ex_index_to_space(self, i: int, j: int, k: int) -> Coordinate3:
        """Return space coordinate of the given index.

        This method returns the (global) space coordinates corresponding to
        the given (local) index of Ex mesh point.

        Keyword arguments:
        i, j, k -- array index

        """
        idx = array((i, j, k), np.intp)
        global_idx = idx + self.global_field_offset

        spc_0 = (global_idx[0] + 0.5) * self.dr[0] - self.half_size[0]
        spc_1 = global_idx[1] * self.dr[1] - self.half_size[1]
        spc_2 = global_idx[2] * self.dr[2] - self.half_size[2]

        return spc_0, spc_1, spc_2

    def spc_to_exact_ex_idx(self, x: float, y: float, z: float) -> Coordinate3:
        """Return the exact mesh point of the given space coordinate.

        This method returns the (local) position, in index dimension,
        of the nearest Ex mesh point of the given (global) space
        coordinate. The return index could be out-of-range.

        Keyword arguments:
            x, y, z -- (global) space coordinate

        """
        coords = array((x, y, z), np.double)

        global_idx = empty(3, np.double)
        global_idx[0] = (coords[0] + self.half_size[0]) / self.dr[0] - 0.5
        global_idx[1] = (coords[1] + self.half_size[1]) / self.dr[1]
        global_idx[2] = (coords[2] + self.half_size[2]) / self.dr[2]

        idx = empty(3, np.double)
        for i in range(3):
            if self.whole_field_size[i] == 1:
                idx[i] = 0
            else:
                idx[i] = global_idx[i] - self.global_field_offset[i]

        return cast(Coordinate3, tuple(idx))

    def space_to_ex_index(self, x: float, y: float, z: float) -> Index3:
        """Return the nearest mesh point of the given space coordinate.

        This method returns the (local) index of the nearest Ex mesh point of
        the given (global) space coordinate. The return index could be
        out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        exact_idx = self.spc_to_exact_ex_idx(x, y, z)
        return cast(Index3, tuple(np.floor(array(exact_idx) + 0.5).astype(np.intp)))

    def ey_index_to_space(self, i: int, j: int, k: int) -> Coordinate3:
        """Return space coordinate of the given index.

        This method returns the (global) space coordinates corresponding to
        the given (local) index of Ey mesh point.

        Keyword arguments:
        i, j, k -- array index

        """
        idx = array((i, j, k), np.intp)

        global_idx = idx + self.global_field_offset

        coords_0 = global_idx[0] * self.dr[0] - self.half_size[0]
        coords_1 = (global_idx[1] + 0.5) * self.dr[1] - self.half_size[1]
        coords_2 = global_idx[2] * self.dr[2] - self.half_size[2]

        return coords_0, coords_1, coords_2

    def spc_to_exact_ey_idx(self, x: float, y: float, z: float) -> Coordinate3:
        """Return the exact mesh point of the given space coordinate.

        This method returns the (local) position, in index dimension,
        of the nearest Ey mesh point of the given (global) space
        coordinate. The return index could be out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        coords = array((x, y, z), np.double)

        global_idx = empty(3, np.double)
        global_idx[0] = (coords[0] + self.half_size[0]) / self.dr[0]
        global_idx[1] = (coords[1] + self.half_size[1]) / self.dr[1] - 0.5
        global_idx[2] = (coords[2] + self.half_size[2]) / self.dr[2]

        idx = empty(3, np.double)
        for i in range(3):
            if self.whole_field_size[i] == 1:
                idx[i] = 0
            else:
                idx[i] = global_idx[i] - self.global_field_offset[i]

        return cast(Coordinate3, tuple(idx))

    def space_to_ey_index(self, x: float, y: float, z: float) -> Index3:
        """Return the nearest mesh point of the given space coordinate.

        This method returns the (local) index of the nearest Ey mesh point of
        the given (global) space coordinate. The return index could be
        out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        exact_idx = self.spc_to_exact_ey_idx(x, y, z)
        return cast(Index3, tuple(np.floor(array(exact_idx) + 0.5).astype(np.intp)))

    def ez_index_to_space(self, i: int, j: int, k: int) -> Coordinate3:
        """Return space coordinate of the given index.

        This method returns the (global) space coordinates corresponding to
        the given (local) index of Ez mesh point.

        Keyword arguments:
        i, j, k -- array index

        """
        idx = array((i, j, k), np.intp)

        global_idx = idx + self.global_field_offset

        coords_0 = global_idx[0] * self.dr[0] - self.half_size[0]
        coords_1 = global_idx[1] * self.dr[1] - self.half_size[1]
        coords_2 = (global_idx[2] + 0.5) * self.dr[2] - self.half_size[2]

        return coords_0, coords_1, coords_2

    def spc_to_exact_ez_idx(self, x: float, y: float, z: float) -> Coordinate3:
        """Return the exact mesh point of the given space coordinate.

        This method returns the (local) position, in index dimension,
        of the nearest Ez mesh point of the given (global) space
        coordinate. The return index could be out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        coords = array((x, y, z), np.double)

        global_idx = empty(3, np.double)
        global_idx[0] = (coords[0] + self.half_size[0]) / self.dr[0]
        global_idx[1] = (coords[1] + self.half_size[1]) / self.dr[1]
        global_idx[2] = (coords[2] + self.half_size[2]) / self.dr[2] - 0.5

        idx = empty(3, np.double)
        for i in range(3):
            if self.whole_field_size[i] == 1:
                idx[i] = 0
            else:
                idx[i] = global_idx[i] - self.global_field_offset[i]

        return cast(Coordinate3, tuple(idx))

    def space_to_ez_index(self, x: float, y: float, z: float) -> Index3:
        """Return the nearest mesh point of the given space coordinate.

        This method returns the (local) index of the nearest Ez mesh point of
        the given (global) space coordinate. The return index could be
        out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        exact_idx = self.spc_to_exact_ez_idx(x, y, z)
        return cast(Index3, tuple(np.floor(array(exact_idx) + 0.5).astype(np.intp)))

    def hx_index_to_space(self, i: int, j: int, k: int) -> Coordinate3:
        """Return space coordinate of the given index.

        This method returns the (global) space coordinates corresponding to
        the given (local) index of Hx mesh point.

        Keyword arguments:
        i, j, k -- array index

        """
        idx = array((i, j, k), np.intp)

        global_idx = idx + self.global_field_offset

        coords_0 = global_idx[0] * self.dr[0] - self.half_size[0]
        coords_1 = (global_idx[1] - 0.5) * self.dr[1] - self.half_size[1]
        coords_2 = (global_idx[2] - 0.5) * self.dr[2] - self.half_size[2]

        return coords_0, coords_1, coords_2

    def spc_to_exact_hx_idx(self, x: float, y: float, z: float) -> Coordinate3:
        """Return the exact mesh point of the given space coordinate.

        This method returns the (local) position, in index dimension,
        of the nearest Hx mesh point of the given (global) space
        coordinate. The return index could be out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        coords = array((x, y, z), np.double)

        global_idx = empty(3, np.double)
        global_idx[0] = (coords[0] + self.half_size[0]) / self.dr[0]
        global_idx[1] = (coords[1] + self.half_size[1]) / self.dr[1] + 0.5
        global_idx[2] = (coords[2] + self.half_size[2]) / self.dr[2] + 0.5

        idx = global_idx - self.my_cart_idx * self.general_field_size
        if self.whole_field_size[0] == 1:
            idx[0] = 0
        if self.whole_field_size[1] == 1:
            idx[1] = 1
        if self.whole_field_size[2] == 1:
            idx[2] = 1

        return cast(Coordinate3, tuple(idx))

    def space_to_hx_index(self, x: float, y: float, z: float) -> Index3:
        """Return the nearest mesh point of the given space coordinate.

        This method returns the (local) index of the nearest Hx mesh point of
        the given (global) space coordinate. The return index could be
        out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        exact_idx = self.spc_to_exact_hx_idx(x, y, z)
        return cast(Index3, tuple(np.floor(array(exact_idx) + 0.5).astype(np.intp)))

    def hy_index_to_space(self, i: int, j: int, k: int) -> Coordinate3:
        """Return space coordinate of the given index.

        This method returns the (global) space coordinates corresponding to
        the given (local) index of Hy mesh point.

        Keyword arguments:
        i, j, k -- array index

        """
        idx = array((i, j, k), np.intp)

        global_idx = idx + self.global_field_offset

        coords_0 = (global_idx[0] - 0.5) * self.dr[0] - self.half_size[0]
        coords_1 = global_idx[1] * self.dr[1] - self.half_size[1]
        coords_2 = (global_idx[2] - 0.5) * self.dr[2] - self.half_size[2]

        return coords_0, coords_1, coords_2

    def spc_to_exact_hy_idx(self, x: float, y: float, z: float) -> Coordinate3:
        """Return the exact mesh point of the given space coordinate.

        This method returns the (local) position, in index dimension,
        of the nearest Hy mesh point of the given (global) space
        coordinate. The return index could be out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        coords = array((x, y, z), np.double)

        global_idx = empty(3, np.double)
        global_idx[0] = (coords[0] + self.half_size[0]) / self.dr[0] + 0.5
        global_idx[1] = (coords[1] + self.half_size[1]) / self.dr[1]
        global_idx[2] = (coords[2] + self.half_size[2]) / self.dr[2] + 0.5

        idx = global_idx - self.my_cart_idx * self.general_field_size
        if self.whole_field_size[0] == 1:
            idx[0] = 1
        if self.whole_field_size[1] == 1:
            idx[1] = 0
        if self.whole_field_size[2] == 1:
            idx[2] = 1

        return cast(Coordinate3, tuple(idx))

    def space_to_hy_index(self, x: float, y: float, z: float) -> Index3:
        """Return the nearest mesh point of the given space coordinate.

        This method returns the (local) index of the nearest Hy mesh point of
        the given (global) space coordinate. The return index could be
        out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        exact_idx = self.spc_to_exact_hy_idx(x, y, z)
        return cast(Index3, tuple(np.floor(array(exact_idx) + 0.5).astype(np.intp)))

    def hz_index_to_space(self, i: int, j: int, k: int) -> Coordinate3:
        """Return space coordinate of the given index.

        This method returns the (global) space coordinates corresponding to
        the given (local) index of Hz mesh point.

        Keyword arguments:
        i, j, k -- array index

        """
        idx = array((i, j, k), np.intp)

        global_idx = idx + self.global_field_offset

        coords_0 = (global_idx[0] - 0.5) * self.dr[0] - self.half_size[0]
        coords_1 = (global_idx[1] - 0.5) * self.dr[1] - self.half_size[1]
        coords_2 = global_idx[2] * self.dr[2] - self.half_size[2]

        return coords_0, coords_1, coords_2

    def spc_to_exact_hz_idx(self, x: float, y: float, z: float) -> Coordinate3:
        """Return the exact mesh point of the given space coordinate.

        This method returns the (local) position, in index dimension,
        of the nearest Hz mesh point of the given (global) space
        coordinate. The return index could be out-of-range.

        Keyword arguments:
        x, y, z -- (global) space coordinate

        """
        coords = array((x, y, z), np.double)

        global_idx = empty(3, np.double)
        global_idx[0] = (coords[0] + self.half_size[0]) / self.dr[0] + 0.5
        global_idx[1] = (coords[1] + self.half_size[1]) / self.dr[1] + 0.5
        global_idx[2] = (coords[2] + self.half_size[2]) / self.dr[2]

        idx = global_idx - self.my_cart_idx * self.general_field_size
        if self.whole_field_size[0] == 1:
            idx[0] = 1
        if self.whole_field_size[1] == 1:
            idx[1] = 1
        if self.whole_field_size[2] == 1:
            idx[2] = 0

        return cast(Coordinate3, tuple(idx))

    def space_to_hz_index(self, x: float, y: float, z: float) -> Index3:
        """Return the nearest mesh point of the given space coordinate.

        This method returns the (local) index of the nearest Hy mesh point of
        the given (global) space coordinate. The return index could be
        out-of-range.

        Keyword arguments:
         x, y, z -- (global) space coordinate

        """
        exact_idx = self.spc_to_exact_hz_idx(x, y, z)
        return cast(Index3, tuple(np.floor(array(exact_idx) + 0.5).astype(np.intp)))

    def display_info(self, indent: int = 0) -> None:
        """Print a human-readable summary of the local Cartesian grid.

        Args:
            indent: Number of leading spaces for every output line.
        """

        print(" " * indent, "Cartesian space")

        print(" " * indent, "MPI topology:", end=" ")
        print(self.my_id, "of", self.numprocs)

        print(" " * indent, end=" ")
        print("size:", 2 * self.half_size, end=" ")
        print("resolution:", self.res)

        print(" " * indent, end=" ")
        print("dx:", self.dr[0], "dy:", self.dr[1], "dz:", self.dr[2])

        print(" " * indent, end=" ")
        print("number of participating nodes:", self.numprocs)


def in_range(idx: IndexLike, shape: Sequence[int], component: ComponentType) -> bool:
    """Perform bounds checking.


    Keyword arguments:
        idx -- index of an array
        shape -- shape of the array to be checked
        component -- specify field component

    """
    if component is const.Ex:
        if idx[0] < 0 or idx[0] >= shape[0]:
            return False
        if idx[1] < 0 or idx[1] >= shape[1] - 1:
            return False
        if idx[2] < 0 or idx[2] >= shape[2] - 1:
            return False

    elif component is const.Ey:
        if idx[0] < 0 or idx[0] >= shape[0] - 1:
            return False
        if idx[1] < 0 or idx[1] >= shape[1]:
            return False
        if idx[2] < 0 or idx[2] >= shape[2] - 1:
            return False

    elif component is const.Ez:
        if idx[0] < 0 or idx[0] >= shape[0] - 1:
            return False
        if idx[1] < 0 or idx[1] >= shape[1] - 1:
            return False
        if idx[2] < 0 or idx[2] >= shape[2]:
            return False

    elif component is const.Hx:
        if idx[0] < 0 or idx[0] >= shape[0]:
            return False
        if idx[1] <= 0 or idx[1] >= shape[1]:
            return False
        if idx[2] <= 0 or idx[2] >= shape[2]:
            return False

    elif component is const.Hy:
        if idx[0] <= 0 or idx[0] >= shape[0]:
            return False
        if idx[1] < 0 or idx[1] >= shape[1]:
            return False
        if idx[2] <= 0 or idx[2] >= shape[2]:
            return False

    elif component is const.Hz:
        if idx[0] <= 0 or idx[0] >= shape[0]:
            return False
        if idx[1] <= 0 or idx[1] >= shape[1]:
            return False
        if idx[2] < 0 or idx[2] >= shape[2]:
            return False

    else:
        raise ValueError

    return True
