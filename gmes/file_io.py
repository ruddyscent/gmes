#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stderr
from os.path import exists
from pathlib import Path

class Probe(object):
    def __init__(self, idx, field, filename):
        """
        idx: index of probing point. type: tuple-3
        field: field to probe. type: numpy.array
        filename: recording file name. type: str

        """
        self.idx = tuple(idx)

        self.field = field

        f_name = str(filename)
        if exists(f_name):
            stderr.write('Warning: ' + f_name + ' already exists.\n')
        try:
            self.f = open(f_name, 'w')
        except IOError:
            self.f = None
            print(('Warning: Can\'t open file ' + f_name + '.\n'))

    def __del__(self):
        self.close()

    def close(self):
        if self.f is not None and not self.f.closed:
            self.f.close()

    def write_header(self, p, dt):
        """Write some meta-data on the header of the recording file.

        p: space coordinates. type: tuple-3
        dt: time-step. type: float

        """
        self.f.write('# location=' + str(p) + '\n')
        self.f.write('# dt=' + str(dt) + '\n')

    def write(self, n):
        self.f.write(str(n) + ' ' + str(self.field[self.idx]) + '\n')

def write_hdf5(data, name, low_index, high_index):
    from tables import open_file

    node_name = Path(name).name
    selection = data[low_index[0]:high_index[0],
                     low_index[1]:high_index[1],
                     low_index[2]:high_index[2]]
    with open_file(name + '.h5', mode='w') as h5file:
        h5file.create_array('/', node_name, selection)

def snapshot(data, filename, title):
    from matplotlib import pyplot

    pyplot.title(title)
    pyplot.imshow(data, origin="lower")
    pyplot.savefig(filename)
