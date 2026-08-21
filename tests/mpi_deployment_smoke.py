"""Exercise Cartesian deployment through an actual MPI communicator."""

from math import prod

from gmes.geometry import Cartesian

space = Cartesian(size=(10, 10, 10), resolution=1, parallel=True)
partition = space.cart_comm.Get_topo()[0]

assert prod(partition) == space.numprocs
assert space.cart_comm.Get_size() == space.numprocs
