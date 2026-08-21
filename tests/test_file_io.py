import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gmes.file_io import Probe, write_hdf5


class ProbeTest(unittest.TestCase):
    def test_probe_writes_metadata_and_sample(self):
        field = np.arange(8).reshape((2, 2, 2))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'probe.txt'
            probe = Probe((1, 0, 1), field, output)
            probe.write_header((0.5, 0.0, -0.5), 0.25)
            probe.write(3)
            probe.close()

            self.assertEqual(
                output.read_text(),
                '# location=(0.5, 0.0, -0.5)\n# dt=0.25\n3 5\n',
            )

    @unittest.skipUnless(importlib.util.find_spec('tables'),
                         'PyTables is not installed')
    def test_hdf5_writer_uses_modern_pytables_api(self):
        from tables import open_file

        data = np.arange(27).reshape((3, 3, 3))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'field'
            write_hdf5(data, str(output), (0, 1, 1), (2, 3, 3))

            with open_file(str(output) + '.h5') as h5file:
                np.testing.assert_array_equal(
                    h5file.root.field.read(),
                    data[0:2, 1:3, 1:3],
                )


if __name__ == "__main__":
    unittest.main()
