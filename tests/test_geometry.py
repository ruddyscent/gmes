import unittest

import numpy as np

from gmes.geometry import Cartesian


class CartesianGridTest(unittest.TestCase):
    def test_serial_grid_sizes_are_integral(self):
        space = Cartesian(size=(4, 6, 0), resolution=(2, 3, 4))

        np.testing.assert_array_equal(space.whole_field_size, (8, 18, 1))
        np.testing.assert_array_equal(space.general_field_size, (8, 18, 1))
        np.testing.assert_array_equal(space.my_field_size, (8, 18, 1))
        self.assertEqual(space.whole_field_size.dtype.kind, "i")
        self.assertEqual(space.general_field_size.dtype.kind, "i")
        self.assertEqual(space.my_field_size.dtype.kind, "i")

    def test_partition_contains_every_process(self):
        space = Cartesian(size=(8, 6, 4), resolution=2)
        process_counts = (2, 3, 5, 8, 10, 12, 16, 18, 25, 32, 64, 127, 256)

        for process_count in process_counts:
            with self.subTest(process_count=process_count):
                space.numprocs = process_count
                partition = space.find_best_deploy()

                self.assertEqual(len(partition), 3)
                self.assertEqual(np.prod(partition), process_count)
                self.assertTrue(all(isinstance(value, int) for value in partition))

    def test_component_coordinate_round_trips(self):
        space = Cartesian(size=(4, 6, 2), resolution=2)

        for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
            index_to_space = getattr(space, component + "_index_to_space")
            space_to_index = getattr(space, "space_to_" + component + "_index")
            point = index_to_space(2, 3, 1)

            self.assertEqual(space_to_index(*point), (2, 3, 1))

    def test_component_indices_floor_negative_nearest_grid_values(self):
        space = Cartesian(size=(2, 2, 2), resolution=2)

        self.assertEqual(space.space_to_ex_index(-1.3, 0, 0)[0], -1)

        for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
            with self.subTest(component=component):
                index_to_space = getattr(space, component + "_index_to_space")
                space_to_index = getattr(space, "space_to_" + component + "_index")
                first_point = np.array(index_to_space(0, 0, 0))
                below_first = first_point - 0.6 * space.dr

                self.assertEqual(space_to_index(*below_first), (-1, -1, -1))

    def test_component_indices_handle_both_nearest_grid_boundaries(self):
        space = Cartesian(size=(4, 4, 4), resolution=2)
        base_index = np.array((2, 2, 2))
        boundaries = (
            (-0.500001, -1),
            (-0.5, 0),
            (-0.499999, 0),
            (0.499999, 0),
            (0.5, 1),
            (0.500001, 1),
        )

        for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
            index_to_space = getattr(space, component + "_index_to_space")
            space_to_index = getattr(space, "space_to_" + component + "_index")
            base_point = np.array(index_to_space(*base_index))

            for axis in range(3):
                for offset, expected_delta in boundaries:
                    with self.subTest(component=component, axis=axis, offset=offset):
                        point = base_point.copy()
                        point[axis] += offset * space.dr[axis]
                        expected = base_index.copy()
                        expected[axis] += expected_delta

                        self.assertEqual(space_to_index(*point), tuple(expected))


if __name__ == "__main__":
    unittest.main()
