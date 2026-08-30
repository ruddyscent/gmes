"""Device-resident Maxwell--Bloch updates for Torch DM2 material buckets."""

from dataclasses import dataclass

import torch
from torch import nn

DM2_MAX_ITERATIONS = 100
DM2_ITERATIONS_PER_CHUNK = 10


@dataclass(frozen=True)
class Dm2BucketMetadata:
    """Static identity and storage offsets for one exact-width DM2 bucket."""

    component: str
    bucket_index: int
    transition_count: int
    target_count: int
    prefix: str
    source_component: str
    status_offset: int


class TorchDm2BucketState(nn.Module):
    """Contiguous mutable state and fixed scratch for one DM2 bucket."""

    metadata: Dm2BucketMetadata
    u: torch.Tensor
    _status: torch.Tensor
    _iterations: torch.Tensor
    _u_new: torch.Tensor
    _u_previous: torch.Tensor
    _u_candidate: torch.Tensor
    _a: torch.Tensor
    _b: torch.Tensor
    _transition: torch.Tensor
    _e_old: torch.Tensor
    _e_base: torch.Tensor
    _e_new: torch.Tensor
    _e_previous: torch.Tensor
    _e_candidate: torch.Tensor
    _source_positive: torch.Tensor
    _source_negative: torch.Tensor
    _c_plus: torch.Tensor
    _c_minus: torch.Tensor
    _d: torch.Tensor
    _error: torch.Tensor
    _error_candidate: torch.Tensor
    _cell0: torch.Tensor
    _cell1: torch.Tensor
    _cell2: torch.Tensor
    _active: torch.Tensor
    _invalid: torch.Tensor
    _mask: torch.Tensor
    _mask2: torch.Tensor
    _time: torch.Tensor

    def __init__(
        self,
        metadata: Dm2BucketMetadata,
        *,
        status: torch.Tensor,
        iterations: torch.Tensor,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self._status = status
        self._iterations = iterations
        cells = metadata.target_count
        transitions = metadata.transition_count
        state_shape = (3, cells, transitions)
        transition_shape = (cells, transitions)

        self.register_buffer("u", torch.zeros(state_shape, device=device, dtype=dtype))
        for name in ("u_new", "u_previous", "u_candidate"):
            self.register_buffer(
                f"_{name}",
                torch.zeros(state_shape, device=device, dtype=dtype),
                persistent=False,
            )
        for name in ("a", "b", "transition"):
            self.register_buffer(
                f"_{name}",
                torch.zeros(transition_shape, device=device, dtype=dtype),
                persistent=False,
            )
        for name in (
            "e_old",
            "e_base",
            "e_new",
            "e_previous",
            "e_candidate",
            "source_positive",
            "source_negative",
            "c_plus",
            "c_minus",
            "d",
            "error",
            "error_candidate",
            "cell0",
            "cell1",
            "cell2",
        ):
            self.register_buffer(
                f"_{name}",
                torch.zeros(cells, device=device, dtype=dtype),
                persistent=False,
            )
        for name in ("active", "invalid", "mask", "mask2"):
            self.register_buffer(
                f"_{name}",
                torch.zeros(cells, device=device, dtype=torch.bool),
                persistent=False,
            )
        self.register_buffer(
            "_time", torch.zeros((), device=device, dtype=dtype), persistent=False
        )
        self.requires_grad_(False)

    def prepare(
        self,
        field: torch.Tensor,
        source: torch.Tensor,
        step_count: torch.Tensor,
        time_step: torch.Tensor,
        targets: torch.Tensor,
        source_positive_indices: torch.Tensor,
        source_negative_indices: torch.Tensor,
        rho30: torch.Tensor,
        gamma: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        hbar: torch.Tensor,
        omega: torch.Tensor,
        n_atom: torch.Tensor,
        curl_scale: torch.Tensor,
    ) -> None:
        """Prepare time-dependent coefficients and initial corrector state."""
        self._time.copy_(step_count).add_(1).mul_(time_step)
        self._cell0.copy_(self._time).div_(t2).neg_().exp_()
        self._a.copy_(n_atom).mul_(gamma[:, None]).div_(t2[:, None])
        self._a.mul_(self._cell0[:, None])
        self._b.copy_(n_atom).mul_(gamma[:, None]).mul_(omega)
        self._b.mul_(self._cell0[:, None])

        self._cell1.copy_(t1).reciprocal_()
        self._cell2.copy_(t2).reciprocal_()
        self._cell1.sub_(self._cell2).mul_(self._time).neg_().exp_()
        self._c_plus.copy_(gamma).mul_(2).div_(hbar).mul_(self._cell1)
        self._cell1.copy_(t2).reciprocal_()
        self._cell2.copy_(t1).reciprocal_()
        self._cell1.sub_(self._cell2).mul_(self._time).neg_().exp_()
        self._c_minus.copy_(gamma).mul_(2).div_(hbar).mul_(self._cell1)
        self._cell1.copy_(self._time).div_(t2).exp_()
        self._d.copy_(gamma).mul_(rho30).mul_(2).div_(hbar).mul_(self._cell1)

        flat_field = field.reshape(-1)
        flat_source = source.reshape(-1)
        torch.index_select(flat_field, 0, targets, out=self._e_old)
        torch.index_select(
            flat_source, 0, source_positive_indices, out=self._source_positive
        )
        torch.index_select(
            flat_source, 0, source_negative_indices, out=self._source_negative
        )
        self._e_base.copy_(self._source_positive).sub_(self._source_negative)
        self._e_base.mul_(curl_scale).add_(self._e_old)

        self._e_new.copy_(self._e_old)
        self._u_new.copy_(self.u)
        self._active.fill_(True)
        self._invalid.zero_()
        self._error.zero_()
        self._status.zero_()
        self._iterations.zero_()

    def iterate(
        self,
        half_dt: float,
        quarter_dt: float,
        rtol: torch.Tensor,
        omega: torch.Tensor,
    ) -> None:
        """Advance one fixed device-side masked corrector chunk."""
        for _ in range(DM2_ITERATIONS_PER_CHUNK):
            self._e_previous.copy_(self._e_new)
            self._u_previous.copy_(self._u_new)

            self._e_candidate.copy_(self._e_base)
            torch.add(self._u_new[0], self.u[0], out=self._transition)
            self._transition.mul_(self._a)
            torch.sum(self._transition, dim=1, out=self._cell0)
            self._e_candidate.add_(self._cell0, alpha=-half_dt)
            torch.add(self._u_new[1], self.u[1], out=self._transition)
            self._transition.mul_(self._b)
            torch.sum(self._transition, dim=1, out=self._cell0)
            self._e_candidate.add_(self._cell0, alpha=half_dt)
            torch.where(self._active, self._e_candidate, self._e_new, out=self._e_new)

            torch.add(self._u_new[1], self.u[1], out=self._transition)
            self._transition.mul_(omega).mul_(half_dt)
            torch.add(
                self.u[0],
                self._transition,
                out=self._u_candidate[0],
            )

            self._u_candidate[1].copy_(self.u[1])
            torch.add(
                self._u_candidate[0],
                self.u[0],
                out=self._transition,
            )
            self._transition.mul_(omega).mul_(-half_dt)
            self._u_candidate[1].add_(self._transition)
            torch.add(self._u_new[2], self.u[2], out=self._transition)
            self._transition.mul_(self._c_plus[:, None])
            self._cell0.copy_(self._e_new).add_(self._e_old)
            self._transition.mul_(self._cell0[:, None]).mul_(quarter_dt)
            self._u_candidate[1].add_(self._transition)
            self._cell0.mul_(self._d).mul_(half_dt)
            self._u_candidate[1].add_(self._cell0[:, None])

            torch.add(
                self._u_candidate[1],
                self.u[1],
                out=self._transition,
            )
            self._transition.mul_(self._c_minus[:, None])
            self._cell0.copy_(self._e_new).add_(self._e_old)
            self._transition.mul_(self._cell0[:, None]).mul_(-quarter_dt)
            torch.add(
                self.u[2],
                self._transition,
                out=self._u_candidate[2],
            )
            torch.where(
                self._active[None, :, None],
                self._u_candidate,
                self._u_new,
                out=self._u_new,
            )

            torch.sub(self._e_new, self._e_previous, out=self._cell0)
            self._cell0.square_()
            torch.sub(self._u_new, self._u_previous, out=self._u_candidate)
            self._u_candidate.square_()
            torch.sum(self._u_candidate, dim=(0, 2), out=self._cell2)
            self._cell0.add_(self._cell2).sqrt_()

            self._cell1.copy_(self._e_previous).square_()
            self._u_previous.square_()
            torch.sum(self._u_previous, dim=(0, 2), out=self._cell2)
            self._cell1.add_(self._cell2).sqrt_()
            torch.div(self._cell0, self._cell1, out=self._error_candidate)
            torch.eq(self._cell0, 0, out=self._mask)
            torch.eq(self._cell1, 0, out=self._mask2)
            torch.logical_and(self._mask, self._mask2, out=self._mask)
            self._error_candidate.masked_fill_(self._mask, 0)
            torch.where(
                self._active,
                self._error_candidate,
                self._error,
                out=self._error,
            )
            torch.add(self._iterations, self._active, out=self._iterations)

            torch.ne(self._error_candidate, self._error_candidate, out=self._mask)
            torch.logical_and(self._mask, self._active, out=self._mask)
            torch.logical_or(self._invalid, self._mask, out=self._invalid)
            torch.gt(self._error, rtol, out=self._mask)
            torch.logical_and(self._active, self._mask, out=self._active)
            torch.logical_not(self._invalid, out=self._mask)
            torch.logical_and(self._active, self._mask, out=self._active)

    def finalize(self, field: torch.Tensor, targets: torch.Tensor) -> None:
        """Commit converged targets and retain failed targets' prior state."""
        flat_field = field.reshape(-1)
        self._status.masked_fill_(self._invalid, 1)
        self._status.masked_fill_(self._active, 2)
        torch.logical_not(self._invalid, out=self._mask)
        torch.logical_not(self._active, out=self._mask2)
        torch.logical_and(self._mask, self._mask2, out=self._mask)
        torch.where(self._mask, self._e_new, self._e_old, out=self._e_new)
        flat_field.index_copy_(0, targets, self._e_new)
        torch.where(self._mask[None, :, None], self._u_new, self.u, out=self.u)


__all__ = [
    "DM2_ITERATIONS_PER_CHUNK",
    "DM2_MAX_ITERATIONS",
    "Dm2BucketMetadata",
    "TorchDm2BucketState",
]
