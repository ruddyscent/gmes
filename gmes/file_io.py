#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Record field probes and export simulation snapshots."""

from os.path import exists
from pathlib import Path
from sys import stderr


class Probe(object):
    """Record one field-array cell to a text file as the simulation advances."""

    def __init__(self, idx, field, filename):
        """Open a probe file and retain a live view of a field array.

        Args:
            idx: Three-dimensional array index of the sampled field cell.
            field: NumPy-compatible field array. The probe does not copy it.
            filename: Destination text-file path, which is replaced if it exists.
        """
        self.idx = tuple(idx)

        self.field = field

        f_name = str(filename)
        if exists(f_name):
            stderr.write("Warning: " + f_name + " already exists.\n")
        try:
            self.f = open(f_name, "w")
        except IOError:
            self.f = None
            print(("Warning: Can't open file " + f_name + ".\n"))

    def __del__(self):
        self.close()

    def close(self):
        """Close the destination file if it is still open."""

        if self.f is not None and not self.f.closed:
            self.f.close()

    def write_header(self, p, dt):
        """Write the probe location and time-step metadata.

        Args:
            p: Three-dimensional physical coordinates of the sampled cell.
            dt: Simulation time-step size.
        """
        self.f.write("# location=" + str(p) + "\n")
        self.f.write("# dt=" + str(dt) + "\n")

    def write(self, n):
        """Append the current step and sampled field value to the file.

        Args:
            n: Current simulation step, including half steps when applicable.
        """

        self.f.write(str(n) + " " + str(self.field[self.idx]) + "\n")


def write_hdf5(data, name, low_index, high_index):
    """Write a half-open three-dimensional array selection to an HDF5 file.

    Args:
        data: Array-like simulation data indexed in three dimensions.
        name: Output path without the .h5 suffix and HDF5 node name source.
        low_index: Inclusive lower index for each dimension.
        high_index: Exclusive upper index for each dimension.

    Note:
        This function requires the optional hdf5 dependency.
    """

    from tables import open_file

    node_name = Path(name).name
    selection = data[
        low_index[0] : high_index[0],
        low_index[1] : high_index[1],
        low_index[2] : high_index[2],
    ]
    with open_file(name + ".h5", mode="w") as h5file:
        h5file.create_array("/", node_name, selection)


def snapshot(data, filename, title):
    """Render two-dimensional data to an image file with Matplotlib.

    Args:
        data: Two-dimensional array-like values to render.
        filename: Destination image path; its suffix selects the image format.
        title: Figure title.

    Note:
        This function requires the optional plot dependency.
    """

    from matplotlib import pyplot

    pyplot.title(title)
    pyplot.imshow(data, origin="lower")
    pyplot.savefig(filename)
