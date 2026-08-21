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


if __name__ == "__main__":
    unittest.main()
