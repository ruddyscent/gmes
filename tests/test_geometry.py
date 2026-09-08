import numpy as np
import pytest

from gmes.constant import Ex, Ey, Ez, Hx, Hy, Hz
from gmes.geometry import Cartesian


class TestCartesianGrid:
    def test_serial_grid_sizes_are_integral(self):
        space = Cartesian(size=(4, 6, 0), resolution=(2, 3, 4))

        np.testing.assert_array_equal(space.whole_field_size, (8, 18, 1))
        np.testing.assert_array_equal(space.general_field_size, (8, 18, 1))
        np.testing.assert_array_equal(space.my_field_size, (8, 18, 1))
        assert space.whole_field_size.dtype.kind == "i"
        assert space.general_field_size.dtype.kind == "i"
        assert space.my_field_size.dtype.kind == "i"

    @pytest.mark.parametrize(
        "process_count",
        (2, 3, 5, 8, 10, 12, 16, 18, 25, 32, 64, 127, 256),
        ids=lambda value: f"processes-{value}",
    )
    def test_partition_contains_every_process(self, process_count):
        space = Cartesian(size=(8, 6, 4), resolution=2)
        space.numprocs = process_count
        partition = space.find_best_deploy()

        assert len(partition) == 3
        assert np.prod(partition) == process_count
        assert all(isinstance(value, int) for value in partition)

    @pytest.mark.parametrize(
        "component", ("ex", "ey", "ez", "hx", "hy", "hz"), ids=str.upper
    )
    def test_component_coordinate_round_trips(self, component):
        space = Cartesian(size=(4, 6, 2), resolution=2)

        index_to_space = getattr(space, component + "_index_to_space")
        space_to_index = getattr(space, "space_to_" + component + "_index")
        point = index_to_space(2, 3, 1)

        assert space_to_index(*point) == (2, 3, 1)

    @pytest.mark.parametrize(
        "component", (Ex, Ey, Ez, Hx, Hy, Hz), ids=lambda value: value.__name__
    )
    def test_component_coordinate_axes_match_mpi_local_point_conversions(
        self, component
    ):
        space = Cartesian(size=(4, 0, 2), resolution=(2, 3, 4))
        space.general_field_size = np.array((5, 1, 3), dtype=np.intp)
        space.my_cart_idx = (2, 4, 3)
        shape = (3, 2, 4)

        axes = space.component_coordinate_axes(component, shape)
        index_to_space = getattr(space, f"{component.__name__.lower()}_index_to_space")
        for index in np.ndindex(shape):
            expected = index_to_space(*index)
            actual = tuple(axis[value] for axis, value in zip(axes, index))
            np.testing.assert_array_equal(actual, expected)

    def test_component_coordinate_axes_reject_unknown_components(self):
        space = Cartesian(size=(2, 2, 2), resolution=2)

        with pytest.raises(ValueError, match="unknown Yee-grid component"):
            space.component_coordinate_axes(object(), (1, 1, 1))

    @pytest.mark.parametrize(
        "component", ("ex", "ey", "ez", "hx", "hy", "hz"), ids=str.upper
    )
    def test_component_indices_floor_negative_nearest_grid_values(self, component):
        space = Cartesian(size=(2, 2, 2), resolution=2)

        assert space.space_to_ex_index(-1.3, 0, 0)[0] == -1

        index_to_space = getattr(space, component + "_index_to_space")
        space_to_index = getattr(space, "space_to_" + component + "_index")
        first_point = np.array(index_to_space(0, 0, 0))
        below_first = first_point - 0.6 * space.dr

        assert space_to_index(*below_first) == (-1, -1, -1)

    @pytest.mark.parametrize(
        ("component", "axis"),
        tuple(
            (component, axis)
            for component in ("ex", "ey", "ez", "hx", "hy", "hz")
            for axis in range(3)
        ),
        ids=tuple(
            f"{component.upper()}-axis-{axis}"
            for component in ("ex", "ey", "ez", "hx", "hy", "hz")
            for axis in range(3)
        ),
    )
    @pytest.mark.parametrize(
        ("offset", "expected_delta"),
        (
            (-0.500001, -1),
            (-0.5, 0),
            (-0.499999, 0),
            (0.499999, 0),
            (0.5, 1),
            (0.500001, 1),
        ),
        ids=(
            "below-lower-half",
            "at-lower-half",
            "above-lower-half",
            "below-upper-half",
            "at-upper-half",
            "above-upper-half",
        ),
    )
    def test_component_indices_handle_both_nearest_grid_boundaries(
        self, component, axis, offset, expected_delta
    ):
        space = Cartesian(size=(4, 4, 4), resolution=2)
        base_index = np.array((2, 2, 2))

        index_to_space = getattr(space, component + "_index_to_space")
        space_to_index = getattr(space, "space_to_" + component + "_index")
        base_point = np.array(index_to_space(*base_index))

        point = base_point.copy()
        point[axis] += offset * space.dr[axis]
        expected = base_index.copy()
        expected[axis] += expected_delta

        assert space_to_index(*point) == tuple(expected)
