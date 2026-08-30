from _typeshed import Incomplete

class _SwigNonDynamicMeta(type):
    __setattr__: Incomplete

class _component:
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

cvar: Incomplete
pi: Incomplete
h: Incomplete
hbar: Incomplete
c0: Incomplete
mu0: Incomplete
eps0: Incomplete
Z0: Incomplete
YOTTA: Incomplete
ZETTA: Incomplete
EXA: Incomplete
PETA: Incomplete
TERA: Incomplete
GIGA: Incomplete
MEGA: Incomplete
KILO: Incomplete
MILLI: Incomplete
MICRO: Incomplete
NANO: Incomplete
PICO: Incomplete
FEMTO: Incomplete
ATTO: Incomplete
ZEPTO: Incomplete
YOCTO: Incomplete

class _electric(_component):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _ex(_electric):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _ey(_electric):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _ez(_electric):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _magnetic(_component):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _hx(_magnetic):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _hy(_magnetic):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _hz(_magnetic):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _electriccurrent(_component):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _jx(_electriccurrent):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _jy(_electriccurrent):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _jz(_electriccurrent):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _magneticcurrent(_component):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _mx(_magneticcurrent):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _my(_magneticcurrent):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _mz(_magneticcurrent):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _directional:
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _x(_directional):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _y(_directional):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _z(_directional):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    __swig_destroy__: Incomplete

class _plus_x(_x):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    @staticmethod
    def get_vector(): ...
    __swig_destroy__: Incomplete

class _minus_x(_x):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    @staticmethod
    def get_vector(): ...
    __swig_destroy__: Incomplete

class _plus_y(_y):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    @staticmethod
    def get_vector(): ...
    __swig_destroy__: Incomplete

class _minus_y(_y):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    @staticmethod
    def get_vector(): ...
    __swig_destroy__: Incomplete

class _plus_z(_z):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    @staticmethod
    def get_vector(): ...
    __swig_destroy__: Incomplete

class _minus_z(_z):
    thisown: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_tag(): ...
    @staticmethod
    def get_vector(): ...
    __swig_destroy__: Incomplete

class Component(_component):
    tag: Incomplete

class Electric(Component):
    tag: Incomplete

class Ex(Electric):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Ey(Electric):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Ez(Electric):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Magnetic(Component):
    tag: Incomplete

class Hx(Magnetic):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Hy(Magnetic):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Hz(Magnetic):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class ElectricCurrent(Component):
    tag: Incomplete

class Jx(ElectricCurrent):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Jy(ElectricCurrent):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Jz(ElectricCurrent):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class MagneticCurrent(Component):
    tag: Incomplete

class Mx(MagneticCurrent):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class My(MagneticCurrent):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Mz(MagneticCurrent):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Directional(_directional):
    tag: Incomplete

class X(Directional):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Y(Directional):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class Z(Directional):
    tag: Incomplete
    @classmethod
    def str(cls) -> str: ...

class PlusX(X):
    tag: Incomplete
    vector: Incomplete
    @classmethod
    def str(cls) -> str: ...

class MinusX(X):
    tag: Incomplete
    vector: Incomplete
    @classmethod
    def str(cls) -> str: ...

class PlusY(Y):
    tag: Incomplete
    vector: Incomplete
    @classmethod
    def str(cls) -> str: ...

class MinusY(Y):
    tag: Incomplete
    vector: Incomplete
    @classmethod
    def str(cls) -> str: ...

class PlusZ(Z):
    tag: Incomplete
    vector: Incomplete
    @classmethod
    def str(cls) -> str: ...

class MinusZ(Z):
    tag: Incomplete
    vector: Incomplete
    @classmethod
    def str(cls) -> str: ...
