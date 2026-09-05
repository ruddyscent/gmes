"""Physical constants and immutable marker classes used by GMES."""

from math import atan, sqrt

import numpy as np

# ``importlib.reload()`` retains the module dictionary while re-executing this
# file.  Keep the marker classes (and their class-owned direction vectors)
# canonical across that operation: they are public tokens used as dictionary
# keys and pickle globals, not merely equivalent type definitions.
_previous_markers = {
    name: globals().get(name)
    for name in (
        "Component",
        "Electric",
        "Ex",
        "Ey",
        "Ez",
        "Magnetic",
        "Hx",
        "Hy",
        "Hz",
        "ElectricCurrent",
        "Jx",
        "Jy",
        "Jz",
        "MagneticCurrent",
        "Mx",
        "My",
        "Mz",
        "Directional",
        "X",
        "Y",
        "Z",
        "PlusX",
        "MinusX",
        "PlusY",
        "MinusY",
        "PlusZ",
        "MinusZ",
    )
}


pi = 4.0 * atan(1.0)
h = 6.6260695729e-34
hbar = h / (2.0 * pi)
c0 = 299792458.0
mu0 = 4.0 * pi * 1e-7
eps0 = 1.0 / (c0 * c0 * mu0)
Z0 = sqrt(mu0 / eps0)

YOTTA = 1e24
ZETTA = 1e21
EXA = 1e18
PETA = 1e15
TERA = 1e12
GIGA = 1e9
MEGA = 1e6
KILO = 1e3
MILLI = 1e-3
MICRO = 1e-6
NANO = 1e-9
PICO = 1e-12
FEMTO = 1e-15
ATTO = 1e-18
ZEPTO = 1e-21
YOCTO = 1e-24


class Component:
    """Base marker for stable electromagnetic component tokens."""

    tag = 0

    @classmethod
    def get_tag(cls) -> int:
        """Return this marker's stable integer tag."""
        return cls.tag


class Electric(Component):
    """Marker base for electric-field component tokens."""

    tag = 1


class Ex(Electric):
    """Marker token for the x-directed electric field."""

    tag = 3

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "ex"


class Ey(Electric):
    """Marker token for the y-directed electric field."""

    tag = 4

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "ey"


class Ez(Electric):
    """Marker token for the z-directed electric field."""

    tag = 5

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "ez"


class Magnetic(Component):
    """Marker base for magnetic-field component tokens."""

    tag = 2


class Hx(Magnetic):
    """Marker token for the x-directed magnetic field."""

    tag = 6

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "hx"


class Hy(Magnetic):
    """Marker token for the y-directed magnetic field."""

    tag = 7

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "hy"


class Hz(Magnetic):
    """Marker token for the z-directed magnetic field."""

    tag = 8

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "hz"


class ElectricCurrent(Component):
    """Marker base for electric-current component tokens."""

    tag = 9


class Jx(ElectricCurrent):
    """Marker token for the x-directed electric current."""

    tag = 10

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "jx"


class Jy(ElectricCurrent):
    """Marker token for the y-directed electric current."""

    tag = 11

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "jy"


class Jz(ElectricCurrent):
    """Marker token for the z-directed electric current."""

    tag = 12

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "jz"


class MagneticCurrent(Component):
    """Marker base for magnetic-current component tokens."""

    tag = 13


class Mx(MagneticCurrent):
    """Marker token for the x-directed magnetic current."""

    tag = 14

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "mx"


class My(MagneticCurrent):
    """Marker token for the y-directed magnetic current."""

    tag = 15

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "my"


class Mz(MagneticCurrent):
    """Marker token for the z-directed magnetic current."""

    tag = 16

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase component spelling."""
        return "mz"


class Directional:
    """Base marker for stable Cartesian direction tokens."""

    tag = 17

    @classmethod
    def get_tag(cls) -> int:
        """Return this direction marker's stable integer tag."""
        return cls.tag


class X(Directional):
    """Marker token for the x direction."""

    tag = 18

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase direction spelling."""
        return "x"


class Y(Directional):
    """Marker token for the y direction."""

    tag = 19

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase direction spelling."""
        return "y"


class Z(Directional):
    """Marker token for the z direction."""

    tag = 20

    @classmethod
    def str(cls) -> str:
        """Return the stable lowercase direction spelling."""
        return "z"


class PlusX(X):
    """Marker token for the positive x direction."""

    tag = 21
    vector = np.array((1.0, 0.0, 0.0), dtype=np.double)

    @classmethod
    def str(cls) -> str:
        """Return the stable signed direction spelling."""
        return "+x"


class MinusX(X):
    """Marker token for the negative x direction."""

    tag = 22
    vector = np.array((-1.0, 0.0, 0.0), dtype=np.double)

    @classmethod
    def str(cls) -> str:
        """Return the stable signed direction spelling."""
        return "-x"


class PlusY(Y):
    """Marker token for the positive y direction."""

    tag = 23
    vector = np.array((0.0, 1.0, 0.0), dtype=np.double)

    @classmethod
    def str(cls) -> str:
        """Return the stable signed direction spelling."""
        return "+y"


class MinusY(Y):
    """Marker token for the negative y direction."""

    tag = 24
    vector = np.array((0.0, -1.0, 0.0), dtype=np.double)

    @classmethod
    def str(cls) -> str:
        """Return the stable signed direction spelling."""
        return "-y"


class PlusZ(Z):
    """Marker token for the positive z direction."""

    tag = 25
    vector = np.array((0.0, 0.0, 1.0), dtype=np.double)

    @classmethod
    def str(cls) -> str:
        """Return the stable signed direction spelling."""
        return "+z"


class MinusZ(Z):
    """Marker token for the negative z direction."""

    tag = 26
    vector = np.array((0.0, 0.0, -1.0), dtype=np.double)

    @classmethod
    def str(cls) -> str:
        """Return the stable signed direction spelling."""
        return "-z"


for _marker_name, _previous_marker in _previous_markers.items():
    if _previous_marker is not None:
        globals()[_marker_name] = _previous_marker


__all__ = [
    "pi",
    "h",
    "hbar",
    "c0",
    "mu0",
    "eps0",
    "Z0",
    "YOTTA",
    "ZETTA",
    "EXA",
    "PETA",
    "TERA",
    "GIGA",
    "MEGA",
    "KILO",
    "MILLI",
    "MICRO",
    "NANO",
    "PICO",
    "FEMTO",
    "ATTO",
    "ZEPTO",
    "YOCTO",
    "Component",
    "Electric",
    "Ex",
    "Ey",
    "Ez",
    "Magnetic",
    "Hx",
    "Hy",
    "Hz",
    "ElectricCurrent",
    "Jx",
    "Jy",
    "Jz",
    "MagneticCurrent",
    "Mx",
    "My",
    "Mz",
    "Directional",
    "X",
    "Y",
    "Z",
    "PlusX",
    "MinusX",
    "PlusY",
    "MinusY",
    "PlusZ",
    "MinusZ",
]
