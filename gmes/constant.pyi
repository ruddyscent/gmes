from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

pi: float
h: float
hbar: float
c0: float
mu0: float
eps0: float
Z0: float
YOTTA: float
ZETTA: float
EXA: float
PETA: float
TERA: float
GIGA: float
MEGA: float
KILO: float
MILLI: float
MICRO: float
NANO: float
PICO: float
FEMTO: float
ATTO: float
ZEPTO: float
YOCTO: float

class Component:
    tag: ClassVar[int]
    @classmethod
    def get_tag(cls) -> int: ...

class Electric(Component): ...

class Ex(Electric):
    @classmethod
    def str(cls) -> str: ...

class Ey(Electric):
    @classmethod
    def str(cls) -> str: ...

class Ez(Electric):
    @classmethod
    def str(cls) -> str: ...

class Magnetic(Component): ...

class Hx(Magnetic):
    @classmethod
    def str(cls) -> str: ...

class Hy(Magnetic):
    @classmethod
    def str(cls) -> str: ...

class Hz(Magnetic):
    @classmethod
    def str(cls) -> str: ...

class ElectricCurrent(Component): ...

class Jx(ElectricCurrent):
    @classmethod
    def str(cls) -> str: ...

class Jy(ElectricCurrent):
    @classmethod
    def str(cls) -> str: ...

class Jz(ElectricCurrent):
    @classmethod
    def str(cls) -> str: ...

class MagneticCurrent(Component): ...

class Mx(MagneticCurrent):
    @classmethod
    def str(cls) -> str: ...

class My(MagneticCurrent):
    @classmethod
    def str(cls) -> str: ...

class Mz(MagneticCurrent):
    @classmethod
    def str(cls) -> str: ...

class Directional:
    tag: ClassVar[int]
    @classmethod
    def get_tag(cls) -> int: ...

class X(Directional):
    @classmethod
    def str(cls) -> str: ...

class Y(Directional):
    @classmethod
    def str(cls) -> str: ...

class Z(Directional):
    @classmethod
    def str(cls) -> str: ...

class PlusX(X):
    vector: ClassVar[NDArray[np.float64]]
    @classmethod
    def str(cls) -> str: ...

class MinusX(X):
    vector: ClassVar[NDArray[np.float64]]
    @classmethod
    def str(cls) -> str: ...

class PlusY(Y):
    vector: ClassVar[NDArray[np.float64]]
    @classmethod
    def str(cls) -> str: ...

class MinusY(Y):
    vector: ClassVar[NDArray[np.float64]]
    @classmethod
    def str(cls) -> str: ...

class PlusZ(Z):
    vector: ClassVar[NDArray[np.float64]]
    @classmethod
    def str(cls) -> str: ...

class MinusZ(Z):
    vector: ClassVar[NDArray[np.float64]]
    @classmethod
    def str(cls) -> str: ...

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
