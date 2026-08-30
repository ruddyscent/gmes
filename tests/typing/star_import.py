"""Checker-only coverage for the established root star-import API."""

from typing import assert_type

from gmes import *

space = Cartesian(size=(2.0, 2.0, 0.0), resolution=2)
dielectric = Dielectric()
geometry_entry = DefaultMedium(material=dielectric)
component = Ex

assert_type(space, Cartesian)
assert_type(dielectric, Dielectric)
assert_type(geometry_entry, DefaultMedium)
assert_type(component, type[Ex])
