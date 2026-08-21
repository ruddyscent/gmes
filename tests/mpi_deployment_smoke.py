"""Exercise Cartesian deployment through an actual MPI communicator."""

from math import prod

from gmes import FDTD, DefaultMedium, Dielectric, Ez
from gmes.geometry import Cartesian, in_range

space = Cartesian(size=(10, 10, 10), resolution=1, parallel=True)
partition = space.cart_comm.Get_topo()[0]

assert prod(partition) == space.numprocs
assert space.cart_comm.Get_size() == space.numprocs

simulation = FDTD(
    space=space,
    geom_list=[DefaultMedium(material=Dielectric())],
    src_list=[],
    verbose=False,
)
simulation.init()

remote_point = (4, 4, 4)
remote_idx = space.space_to_ez_index(*remote_point)
local_owner = space.my_id if in_range(remote_idx, simulation.ez.shape, Ez) else None
owners = [
    owner for owner in space.cart_comm.allgather(local_owner) if owner is not None
]
assert len(owners) == 1
assert owners[0] != 0
if space.my_id == owners[0]:
    simulation.ez[remote_idx] = 1

simulation.step_while_zero(Ez, remote_point)
assert simulation.time_step.n == 1

try:
    simulation.step_while_zero(Ez, (100, 100, 100))
except ValueError:
    pass
else:
    raise AssertionError("an unowned observation point must raise ValueError")

assert simulation.time_step.n == 1
