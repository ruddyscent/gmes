#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Aggregate per-cell source parameters into legacy field-update kernels."""

import sys
from math import cos, floor, pi, sin
from typing import Any

import numpy as np
from numpy import inf

# GMES modules
from . import constant as const


class PwSourceParam(object):
    """Base marker for parameters attached to one point-wise source cell."""

    pass


class PwSource(object):
    """Aggregate source parameters by field index and update them in place."""

    def __init__(self) -> None:
        self._param: dict[tuple[int, ...], Any] = {}

    def name(self) -> Any:
        """Return the descriptive source-kernel name."""

        raise NotImplementedError

    def attach(self, idx: Any, parameter: Any) -> Any:
        """Attach source parameters to an array index, replacing duplicates."""

        key = tuple(idx)
        if key in self._param:
            sys.stderr.write("Overwriting the existing index.\n")
        self._param[tuple(idx)] = parameter

    def merge(self, ps: Any) -> Any:
        """Merge another compatible point-wise source into this aggregate."""

        for idx, param in ps._param.items():
            if idx in self._param and isinstance(self._param[idx], TransparentParam):
                self._param[idx].merge(param)
            else:
                self._param[idx] = param

    def idx_size(self) -> Any:
        """Return the number of indexed source updates."""

        return len(self._param)

    def update_all(
        self,
        inplace_field: Any,
        in_field1: Any,
        in_field2: Any,
        d1: Any,
        d2: Any,
        dt: Any,
        n: Any,
    ) -> Any:
        """Apply every indexed source update to the destination field in place."""

        for idx, param in self._param.items():
            self._update(inplace_field, in_field1, in_field2, d1, d2, dt, n, idx, param)

    def _update(
        self,
        inplace_field: Any,
        in_field1: Any,
        in_field2: Any,
        d1: Any,
        d2: Any,
        dt: Any,
        n: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        raise NotImplementedError


class PointSourceParam(PwSourceParam):
    """Store one point source waveform, amplitude, medium, and optional output."""

    def __init__(
        self,
        src_time: Any = None,
        amp: Any = 1,
        comp: Any = None,
        eps_inf: Any = 1,
        mu_inf: Any = 1,
        filename: Any = None,
    ) -> None:
        self.src_time = src_time
        self.amp = float(amp)
        self.comp = comp
        self.eps_inf = float(eps_inf)
        self.mu_inf = float(mu_inf)
        self.f = None
        if filename:
            self.f = open(filename, "w")


def _record_source_value(output: Any, time: Any, value: Any) -> Any:
    if np.iscomplexobj(value):
        output.write("%f\t%f\t%f\n" % (time, value.real, value.imag))
    else:
        output.write("%f\t%f\n" % (time, value))


class PointSourceElectric(PwSource):
    """Apply electric-field or electric-current point sources."""

    def name(self) -> Any:
        """Return the electric point-source kernel name."""

        return "PointSourceElectric"

    def _update(
        self,
        e: Any,
        h1: Any,
        h2: Any,
        dr1: Any,
        dr2: Any,
        dt: Any,
        n: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        """Apply one electric source after its material update."""
        src_t = param.amp * param.src_time.oscillator(dt * n)
        if param.f:
            _record_source_value(param.f, dt * n, src_t)

        if issubclass(param.comp, const.Electric):
            e[idx] = src_t
        elif issubclass(param.comp, const.ElectricCurrent):
            e[idx] -= dt * src_t / param.eps_inf


class PointSourceEx(PointSourceElectric):
    """Apply an x-directed electric point source."""

    pass


class PointSourceEy(PointSourceElectric):
    """Apply a y-directed electric point source."""

    pass


class PointSourceEz(PointSourceElectric):
    """Apply a z-directed electric point source."""

    pass


class PointSourceMagnetic(PwSource):
    """Apply magnetic-field or magnetic-current point sources."""

    def name(self) -> Any:
        """Return the magnetic point-source kernel name."""

        return "PointSourceMagnetic"

    def _update(
        self,
        h: Any,
        e1: Any,
        e2: Any,
        dr1: Any,
        dr2: Any,
        dt: Any,
        n: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        """Apply one magnetic source after its material update."""
        src_t = param.amp * param.src_time.oscillator(dt * n)
        if param.f:
            _record_source_value(param.f, dt * n, src_t)

        if issubclass(param.comp, const.Magnetic):
            h[idx] = src_t
        elif issubclass(param.comp, const.MagneticCurrent):
            h[idx] -= dt * src_t / param.mu_inf


class PointSourceHx(PointSourceMagnetic):
    """Apply an x-directed magnetic point source."""

    pass


class PointSourceHy(PointSourceMagnetic):
    """Apply a y-directed magnetic point source."""

    pass


class PointSourceHz(PointSourceMagnetic):
    """Apply a z-directed magnetic point source."""

    pass


class TransparentParam(PwSourceParam):
    """Store incident-field amplitudes for one or more interface faces."""

    def __init__(self, amp: Any, aux_fdtd: Any, directional: Any) -> None:
        self.aux_fdtd = aux_fdtd

        self.face_list = [directional]
        self.amp = {directional: amp}

    def merge(self, param: Any) -> Any:
        """Merge compatible face parameters from the same auxiliary FDTD."""

        if type(self) is not type(param) or self.aux_fdtd is not param.aux_fdtd:
            raise ValueError("incompatible transparent source parameters")

        for face in param.face_list:
            if face not in self.face_list:
                self.face_list.append(face)
        self.amp.update(param.amp)


class TransparentElectricParam(TransparentParam):
    """Store interpolated auxiliary magnetic samples for an electric update."""

    def __init__(
        self, eps_inf: Any, amp: Any, aux_fdtd: Any, samp_pnt: Any, directional: Any
    ) -> None:
        TransparentParam.__init__(self, amp, aux_fdtd, directional)

        self.eps_inf = float(eps_inf)

        samp_idx = aux_fdtd.space.spc_to_exact_hy_idx(*samp_pnt)
        low_idx = np.floor(samp_idx).astype(np.intp)
        self.samp_idx0 = {directional: tuple(low_idx)}
        self.samp_idx1 = {directional: tuple(low_idx + (0, 0, 1))}

        r1_value = samp_idx[2] - floor(samp_idx[2])
        self.r1 = {directional: r1_value}
        self.r0 = {directional: 1 - r1_value}

    def merge(self, param: Any) -> Any:
        """Merge compatible electric-interface interpolation parameters."""

        super().merge(param)
        self.samp_idx0.update(param.samp_idx0)
        self.samp_idx1.update(param.samp_idx1)
        self.r0.update(param.r0)
        self.r1.update(param.r1)


class TransparentMagneticParam(TransparentParam):
    """Store interpolated auxiliary electric samples for a magnetic update."""

    def __init__(
        self, mu_inf: Any, amp: Any, aux_fdtd: Any, samp_pnt: Any, directional: Any
    ) -> None:
        TransparentParam.__init__(self, amp, aux_fdtd, directional)

        self.mu_inf = float(mu_inf)

        samp_idx = aux_fdtd.space.spc_to_exact_ex_idx(*samp_pnt)
        low_idx = np.floor(samp_idx).astype(np.intp)
        self.samp_idx0 = {directional: tuple(low_idx)}
        self.samp_idx1 = {directional: tuple(low_idx + (0, 0, 1))}

        r1_value = samp_idx[2] - floor(samp_idx[2])
        self.r1 = {directional: r1_value}
        self.r0 = {directional: 1 - r1_value}

    def merge(self, param: Any) -> Any:
        """Merge compatible magnetic-interface interpolation parameters."""

        super().merge(param)
        self.samp_idx0.update(param.samp_idx0)
        self.samp_idx1.update(param.samp_idx1)
        self.r0.update(param.r0)
        self.r1.update(param.r1)


class TransparentElectric(PwSource):
    """Base aggregate for total-field electric boundary corrections."""

    def name(self) -> Any:
        """Return the transparent electric-source kernel name."""

        return "TransparentElectric"


class TransparentEx(TransparentElectric):
    """Correct Ex values on transverse total-field interface faces."""

    def __init__(self) -> None:
        PwSource.__init__(self)
        self._consist_cond = {
            const.MinusY: self._consistency_minus_y,
            const.MinusZ: self._consistency_minus_z,
            const.PlusY: self._consistency_plus_y,
            const.PlusZ: self._consistency_plus_z,
        }

    def _update(
        self,
        ex: Any,
        hz: Any,
        hy: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        n: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        for face in param.face_list:
            self._consist_cond[face](ex, hz, hy, dy, dz, dt, face, idx, param)

    def _consistency_minus_y(
        self,
        ex: Any,
        hz: Any,
        hy: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hz = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ex[idx] -= dt / (param.eps_inf * dy) * param.amp[face] * incident_hz

    def _consistency_plus_y(
        self,
        ex: Any,
        hz: Any,
        hy: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hz = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ex[idx] += dt / (param.eps_inf * dy) * param.amp[face] * incident_hz

    def _consistency_minus_z(
        self,
        ex: Any,
        hz: Any,
        hy: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hy = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ex[idx] += dt / (param.eps_inf * dz) * param.amp[face] * incident_hy

    def _consistency_plus_z(
        self,
        ex: Any,
        hz: Any,
        hy: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hy = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ex[idx] -= dt / (param.eps_inf * dz) * param.amp[face] * incident_hy


class TransparentEy(TransparentElectric):
    """Correct Ey values on transverse total-field interface faces."""

    def __init__(self) -> None:
        PwSource.__init__(self)
        self._consist_cond = {
            const.MinusZ: self._consistency_minus_z,
            const.MinusX: self._consistency_minus_x,
            const.PlusZ: self._consistency_plus_z,
            const.PlusX: self._consistency_plus_x,
        }

    def _update(
        self,
        ey: Any,
        hx: Any,
        hz: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        n: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        for face in param.face_list:
            self._consist_cond[face](ey, hx, hz, dz, dx, dt, face, idx, param)

    def _consistency_minus_z(
        self,
        ey: Any,
        hx: Any,
        hz: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hx = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ey[idx] -= dt / (param.eps_inf * dz) * param.amp[face] * incident_hx

    def _consistency_minus_x(
        self,
        ey: Any,
        hx: Any,
        hz: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hz = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ey[idx] += dt / (param.eps_inf * dx) * param.amp[face] * incident_hz

    def _consistency_plus_z(
        self,
        ey: Any,
        hx: Any,
        hz: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hx = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ey[idx] += dt / (param.eps_inf * dz) * param.amp[face] * incident_hx

    def _consistency_plus_x(
        self,
        ey: Any,
        hx: Any,
        hz: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hz = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ey[idx] -= dt / (param.eps_inf * dx) * param.amp[face] * incident_hz


class TransparentEz(TransparentElectric):
    """Correct Ez values on transverse total-field interface faces."""

    def __init__(self) -> None:
        PwSource.__init__(self)
        self._consist_cond = {
            const.MinusX: self._consistency_minus_x,
            const.MinusY: self._consistency_minus_y,
            const.PlusX: self._consistency_plus_x,
            const.PlusY: self._consistency_plus_y,
        }

    def _update(
        self,
        ez: Any,
        hy: Any,
        hx: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        n: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        for face in param.face_list:
            self._consist_cond[face](ez, hy, hx, dx, dy, dt, face, idx, param)

    def _consistency_minus_x(
        self,
        ez: Any,
        hy: Any,
        hx: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hy = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ez[idx] -= dt / (param.eps_inf * dx) * param.amp[face] * incident_hy

    def _consistency_minus_y(
        self,
        ez: Any,
        hy: Any,
        hx: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hx = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ez[idx] += dt / (param.eps_inf * dy) * param.amp[face] * incident_hx

    def _consistency_plus_x(
        self,
        ez: Any,
        hy: Any,
        hx: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hy = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ez[idx] += dt / (param.eps_inf * dx) * param.amp[face] * incident_hy

    def _consistency_plus_y(
        self,
        ez: Any,
        hy: Any,
        hx: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_hx = (
            param.r0[face] * param.aux_fdtd.hy[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.hy[param.samp_idx1[face]]
        )

        ez[idx] -= dt / (param.eps_inf * dy) * param.amp[face] * incident_hx


class TransparentMagnetic(PwSource):
    """Base aggregate for total-field magnetic boundary corrections."""

    def name(self) -> Any:
        """Return the transparent magnetic-source kernel name."""

        return "TransparentMagnetic"


class TransparentHx(TransparentMagnetic):
    """Correct Hx values on transverse total-field interface faces."""

    def __init__(self) -> None:
        PwSource.__init__(self)
        self._consist_cond = {
            const.MinusY: self._consistency_minus_y,
            const.MinusZ: self._consistency_minus_z,
            const.PlusY: self._consistency_plus_y,
            const.PlusZ: self._consistency_plus_z,
        }

    def _update(
        self,
        hx: Any,
        ez: Any,
        ey: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        n: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        for face in param.face_list:
            self._consist_cond[face](hx, ez, ey, dy, dz, dt, face, idx, param)

    def _consistency_minus_y(
        self,
        hx: Any,
        ez: Any,
        ey: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ez = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hx[idx] += dt / (param.mu_inf * dy) * param.amp[face] * incident_ez

    def _consistency_plus_y(
        self,
        hx: Any,
        ez: Any,
        ey: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ez = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hx[idx] -= dt / (param.mu_inf * dy) * param.amp[face] * incident_ez

    def _consistency_minus_z(
        self,
        hx: Any,
        ez: Any,
        ey: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ey = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hx[idx] -= dt / (param.mu_inf * dz) * param.amp[face] * incident_ey

    def _consistency_plus_z(
        self,
        hx: Any,
        ez: Any,
        ey: Any,
        dy: Any,
        dz: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ey = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hx[idx] += dt / (param.mu_inf * dz) * param.amp[face] * incident_ey


class TransparentHy(TransparentMagnetic):
    """Correct Hy values on transverse total-field interface faces."""

    def __init__(self) -> None:
        PwSource.__init__(self)
        self._consist_cond = {
            const.MinusZ: self._consistency_minus_z,
            const.MinusX: self._consistency_minus_x,
            const.PlusZ: self._consistency_plus_z,
            const.PlusX: self._consistency_plus_x,
        }

    def _update(
        self,
        hy: Any,
        ex: Any,
        ez: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        n: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        for face in param.face_list:
            self._consist_cond[face](hy, ex, ez, dz, dx, dt, face, idx, param)

    def _consistency_minus_z(
        self,
        hy: Any,
        ex: Any,
        ez: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ex = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hy[idx] += dt / (param.mu_inf * dz) * param.amp[face] * incident_ex

    def _consistency_minus_x(
        self,
        hy: Any,
        ex: Any,
        ez: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ez = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hy[idx] -= dt / (param.mu_inf * dx) * param.amp[face] * incident_ez

    def _consistency_plus_z(
        self,
        hy: Any,
        ex: Any,
        ez: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ex = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hy[idx] -= dt / (param.mu_inf * dz) * param.amp[face] * incident_ex

    def _consistency_plus_x(
        self,
        hy: Any,
        ex: Any,
        ez: Any,
        dz: Any,
        dx: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ez = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hy[idx] += dt / (param.mu_inf * dx) * param.amp[face] * incident_ez


class TransparentHz(TransparentMagnetic):
    """Correct Hz values on transverse total-field interface faces."""

    def __init__(self) -> None:
        PwSource.__init__(self)
        self._consist_cond = {
            const.MinusX: self._consistency_minus_x,
            const.MinusY: self._consistency_minus_y,
            const.PlusX: self._consistency_plus_x,
            const.PlusY: self._consistency_plus_y,
        }

    def _update(
        self,
        hz: Any,
        ey: Any,
        ex: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        n: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        for face in param.face_list:
            self._consist_cond[face](hz, ey, ex, dx, dy, dt, face, idx, param)

    def _consistency_minus_x(
        self,
        hz: Any,
        ey: Any,
        ex: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ey = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hz[idx] += dt / (param.mu_inf * dx) * param.amp[face] * incident_ey

    def _consistency_minus_y(
        self,
        hz: Any,
        ey: Any,
        ex: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ex = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hz[idx] -= dt / (param.mu_inf * dy) * param.amp[face] * incident_ex

    def _consistency_plus_x(
        self,
        hz: Any,
        ey: Any,
        ex: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ey = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hz[idx] -= dt / (param.mu_inf * dx) * param.amp[face] * incident_ey

    def _consistency_plus_y(
        self,
        hz: Any,
        ey: Any,
        ex: Any,
        dx: Any,
        dy: Any,
        dt: Any,
        face: Any,
        idx: Any,
        param: Any,
    ) -> Any:
        incident_ex = (
            param.r0[face] * param.aux_fdtd.ex[param.samp_idx0[face]]
            + param.r1[face] * param.aux_fdtd.ex[param.samp_idx1[face]]
        )

        hz[idx] += dt / (param.mu_inf * dy) * param.amp[face] * incident_ex
