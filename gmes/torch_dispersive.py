"""Device-resident exact-width tensor buckets for linear dispersive media."""

from dataclasses import dataclass

import numpy as np
import torch

DISPERSIVE_MODELS = frozenset(("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"))


@dataclass(frozen=True)
class DispersiveBucket:
    """Fixed runtime identity for one exact-width dispersive bucket."""

    component: str
    prefix: str
    model: str
    pole_count: int
    point_count: int
    target_count: int
    state_width: int


def _tensor(values, *, dtype, device):
    return torch.tensor(values, dtype=dtype, device=device).contiguous()


def _target_rows(bucket):
    rows = bucket.region_coefficient_indices[bucket.target_region_indices]
    return bucket.coefficient_table[rows]


def _coefficient_columns(bucket, target_rows, names):
    indices = {name: index for index, name in enumerate(bucket.coefficient_names)}
    return np.stack([target_rows[:, indices[name]] for name in names])


def _real_recurrence(bucket, target_rows, stem, width, terms):
    if width == 0:
        return np.empty((terms, 0, len(target_rows)), dtype=np.float64)
    names = tuple(
        f"{stem}{item}_{term}" for term in range(terms) for item in range(width)
    )
    return _coefficient_columns(bucket, target_rows, names).reshape(
        terms, width, len(target_rows)
    )


def _complex_recurrence(bucket, target_rows, stem, width, terms):
    values = np.empty((terms, width, len(target_rows), 2), dtype=np.float64)
    indices = {name: index for index, name in enumerate(bucket.coefficient_names)}
    for term in range(terms):
        for item in range(width):
            values[term, item, :, 0] = target_rows[
                :, indices[f"{stem}{item}_{term}_real"]
            ]
            values[term, item, :, 1] = target_rows[
                :, indices[f"{stem}{item}_{term}_imag"]
            ]
    return values


def register_plan_buffers(module, bucket, component, prefix, *, dtype, device):
    """Finalize one host bucket as contiguous device-side SoA tensors."""
    model = bucket.signature.model
    if model not in DISPERSIVE_MODELS:
        return None

    targets = bucket.targets
    target_rows = _target_rows(bucket)
    if not hasattr(module, f"{prefix}_targets"):
        module.register_buffer(
            f"{prefix}_targets", _tensor(targets, dtype=torch.int64, device=device)
        )
    coordinates = np.stack(np.unravel_index(targets, component.shape), axis=1).astype(
        np.int64, copy=False
    )
    for term_index, term in enumerate(component.stencil):
        strides = np.asarray(term.source_strides, dtype=np.int64)
        base = coordinates @ strides
        module.register_buffer(
            f"{prefix}_source_{term_index}_positive",
            _tensor(base + term.positive_offset, dtype=torch.int64, device=device),
        )
        module.register_buffer(
            f"{prefix}_source_{term_index}_negative",
            _tensor(base + term.negative_offset, dtype=torch.int64, device=device),
        )

    if model in {"drude", "lorentz"}:
        poles = bucket.signature.state_shape[0]
        points = 0
        recurrence_a = _real_recurrence(bucket, target_rows, "a", poles, 3)
        recurrence_c = _coefficient_columns(bucket, target_rows, ("c0", "c1", "c2"))
        module.register_buffer(
            f"{prefix}_a", _tensor(recurrence_a, dtype=dtype, device=device)
        )
        module.register_buffer(
            f"{prefix}_c", _tensor(recurrence_c, dtype=dtype, device=device)
        )
    elif model == "dcp-ade":
        poles, points = bucket.signature.state_shape
        recurrence_a = _real_recurrence(bucket, target_rows, "a", poles, 3)
        recurrence_b = _real_recurrence(bucket, target_rows, "b", points, 5)
        recurrence_c = _coefficient_columns(
            bucket, target_rows, ("c0", "c1", "c2", "c3")
        )
        module.register_buffer(
            f"{prefix}_a", _tensor(recurrence_a, dtype=dtype, device=device)
        )
        module.register_buffer(
            f"{prefix}_b", _tensor(recurrence_b, dtype=dtype, device=device)
        )
        module.register_buffer(
            f"{prefix}_c", _tensor(recurrence_c, dtype=dtype, device=device)
        )
    else:
        poles, points = bucket.signature.state_shape
        recurrence_a = _real_recurrence(bucket, target_rows, "a", poles, 3)
        recurrence_b = _complex_recurrence(bucket, target_rows, "b", points, 3)
        recurrence_c = _coefficient_columns(bucket, target_rows, ("c0", "c1", "c2"))
        module.register_buffer(
            f"{prefix}_a", _tensor(recurrence_a, dtype=dtype, device=device)
        )
        module.register_buffer(
            f"{prefix}_b", _tensor(recurrence_b, dtype=dtype, device=device)
        )
        module.register_buffer(
            f"{prefix}_c", _tensor(recurrence_c, dtype=dtype, device=device)
        )

    return DispersiveBucket(
        component=component.name,
        prefix=prefix,
        model=model,
        pole_count=poles,
        point_count=points,
        target_count=len(targets),
        state_width=bucket.state_width,
    )


def register_state_buffers(module, descriptors, *, paired_real, dtype, device):
    """Allocate exact-width mutable state and fixed scratch storage."""
    channels = 2 if paired_real else 1

    def zeros(name, shape, *, persistent=True):
        module.register_buffer(
            name,
            torch.zeros(shape, dtype=dtype, device=device),
            persistent=persistent,
        )

    for descriptor in descriptors:
        prefix = descriptor.prefix
        count = descriptor.target_count
        common = (count, channels)
        for suffix in (
            "field_now",
            "field_new",
            "curl",
            "gather_a",
            "gather_b",
            "response",
        ):
            zeros(f"{prefix}_{suffix}", common, persistent=False)

        poles = descriptor.pole_count
        points = descriptor.point_count
        if descriptor.model in {"drude", "lorentz"}:
            shape = (poles, count, channels)
            zeros(f"{prefix}_previous", shape)
            zeros(f"{prefix}_current", shape)
            zeros(f"{prefix}_pole_work", shape, persistent=False)
            zeros(f"{prefix}_pole_delta", shape, persistent=False)
        elif descriptor.model == "dcp-ade":
            pole_shape = (poles, count, channels)
            point_shape = (points, count, channels)
            zeros(f"{prefix}_field_old", common)
            zeros(f"{prefix}_pole_old", pole_shape)
            zeros(f"{prefix}_pole_now", pole_shape)
            zeros(f"{prefix}_point_old", point_shape)
            zeros(f"{prefix}_point_now", point_shape)
            zeros(f"{prefix}_pole_work", pole_shape, persistent=False)
            zeros(f"{prefix}_point_work", point_shape, persistent=False)
        else:
            pole_shape = (poles, count, channels)
            point_shape = (points, count, channels, 2)
            zeros(f"{prefix}_pole_state", pole_shape)
            zeros(f"{prefix}_point_state", point_shape)
            zeros(f"{prefix}_pole_work", pole_shape, persistent=False)
            zeros(f"{prefix}_point_work", point_shape, persistent=False)


def _prepare_curl(plan, state, descriptor):
    prefix = descriptor.prefix
    channels = 2 if state.paired_real else 1
    field = state.field(descriptor.component).reshape(-1, channels)
    targets = getattr(plan, f"{prefix}_targets")
    field_now = getattr(state, f"{prefix}_field_now")
    torch.index_select(field, 0, targets, out=field_now)

    curl = getattr(state, f"{prefix}_curl")
    gather_a = getattr(state, f"{prefix}_gather_a")
    gather_b = getattr(state, f"{prefix}_gather_b")
    component = plan.components[descriptor.component]
    for term_index, term in enumerate(component.stencil):
        source = state.field(term.source).reshape(-1, channels)
        positive = getattr(plan, f"{prefix}_source_{term_index}_positive")
        negative = getattr(plan, f"{prefix}_source_{term_index}_negative")
        torch.index_select(source, 0, positive, out=gather_a)
        torch.index_select(source, 0, negative, out=gather_b)
        torch.sub(gather_a, gather_b, out=gather_a)
        gather_a.div_(plan.dr[term.scale_axis])
        if term_index == 0:
            curl.copy_(gather_a)
            if term.sign < 0:
                curl.neg_()
        elif term.sign > 0:
            curl.add_(gather_a)
        else:
            curl.sub_(gather_a)
    return field, targets


def _update_two_level(plan, state, descriptor):
    prefix = descriptor.prefix
    a = getattr(plan, f"{prefix}_a")
    c = getattr(plan, f"{prefix}_c")
    previous = getattr(state, f"{prefix}_previous")
    current = getattr(state, f"{prefix}_current")
    pole_work = getattr(state, f"{prefix}_pole_work")
    pole_delta = getattr(state, f"{prefix}_pole_delta")
    field_now = getattr(state, f"{prefix}_field_now")
    field_new = getattr(state, f"{prefix}_field_new")
    curl = getattr(state, f"{prefix}_curl")
    response = getattr(state, f"{prefix}_response")

    pole_work.copy_(previous).mul_(a[0].unsqueeze(-1))
    pole_work.addcmul_(a[1].unsqueeze(-1), current)
    pole_work.addcmul_(a[2].unsqueeze(-1), field_now.unsqueeze(0))
    pole_delta.copy_(pole_work).sub_(current)
    torch.sum(pole_delta, dim=0, out=response)

    field_new.copy_(curl).mul_(c[0].unsqueeze(-1))
    field_new.addcmul_(c[1].unsqueeze(-1), response)
    field_new.addcmul_(c[2].unsqueeze(-1), field_now)
    previous.copy_(current)
    current.copy_(pole_work)


def _update_dcp_ade(plan, state, descriptor):
    prefix = descriptor.prefix
    a = getattr(plan, f"{prefix}_a")
    b = getattr(plan, f"{prefix}_b")
    c = getattr(plan, f"{prefix}_c")
    field_old = getattr(state, f"{prefix}_field_old")
    field_now = getattr(state, f"{prefix}_field_now")
    field_new = getattr(state, f"{prefix}_field_new")
    curl = getattr(state, f"{prefix}_curl")
    response = getattr(state, f"{prefix}_response")
    combo = getattr(state, f"{prefix}_gather_a")
    pole_old = getattr(state, f"{prefix}_pole_old")
    pole_now = getattr(state, f"{prefix}_pole_now")
    point_old = getattr(state, f"{prefix}_point_old")
    point_now = getattr(state, f"{prefix}_point_now")
    pole_work = getattr(state, f"{prefix}_pole_work")
    point_work = getattr(state, f"{prefix}_point_work")

    pole_work.copy_(pole_now)
    pole_work.addcmul_(a[1].unsqueeze(-1), pole_now, value=-1)
    pole_work.addcmul_(a[0].unsqueeze(-1), pole_old, value=-1)
    torch.sum(pole_work, dim=0, out=response)
    point_work.copy_(point_now)
    point_work.addcmul_(b[1].unsqueeze(-1), point_now, value=-1)
    point_work.addcmul_(b[0].unsqueeze(-1), point_old, value=-1)
    torch.sum(point_work, dim=0, out=combo)
    response.add_(combo)

    field_new.copy_(curl).mul_(c[0].unsqueeze(-1))
    field_new.addcmul_(c[1].unsqueeze(-1), response)
    field_new.addcmul_(c[2].unsqueeze(-1), field_old)
    field_new.addcmul_(c[3].unsqueeze(-1), field_now)

    combo.copy_(field_old).add_(field_now, alpha=2).add_(field_new)
    pole_work.copy_(pole_old).mul_(a[0].unsqueeze(-1))
    pole_work.addcmul_(a[1].unsqueeze(-1), pole_now)
    pole_work.addcmul_(a[2].unsqueeze(-1), combo.unsqueeze(0))
    point_work.copy_(point_old).mul_(b[0].unsqueeze(-1))
    point_work.addcmul_(b[1].unsqueeze(-1), point_now)
    point_work.addcmul_(b[2].unsqueeze(-1), field_old.unsqueeze(0))
    point_work.addcmul_(b[3].unsqueeze(-1), field_now.unsqueeze(0))
    point_work.addcmul_(b[4].unsqueeze(-1), field_new.unsqueeze(0))

    pole_old.copy_(pole_now)
    pole_now.copy_(pole_work)
    point_old.copy_(point_now)
    point_now.copy_(point_work)
    field_old.copy_(field_now)


def _update_dcp_convolution(plan, state, descriptor):
    prefix = descriptor.prefix
    a = getattr(plan, f"{prefix}_a")
    b = getattr(plan, f"{prefix}_b")
    c = getattr(plan, f"{prefix}_c")
    field_now = getattr(state, f"{prefix}_field_now")
    field_new = getattr(state, f"{prefix}_field_new")
    curl = getattr(state, f"{prefix}_curl")
    response = getattr(state, f"{prefix}_response")
    point_response = getattr(state, f"{prefix}_gather_a")
    pole_state = getattr(state, f"{prefix}_pole_state")
    point_state = getattr(state, f"{prefix}_point_state")
    pole_work = getattr(state, f"{prefix}_pole_work")
    point_work = getattr(state, f"{prefix}_point_work")

    torch.sum(pole_state, dim=0, out=response)
    torch.sum(point_state[..., 0], dim=0, out=point_response)
    response.add_(point_response)
    field_new.copy_(curl).mul_(c[0].unsqueeze(-1))
    field_new.addcmul_(c[1].unsqueeze(-1), field_now)
    field_new.addcmul_(c[2].unsqueeze(-1), response)

    pole_work.copy_(field_new.unsqueeze(0)).mul_(a[0].unsqueeze(-1))
    pole_work.addcmul_(a[1].unsqueeze(-1), field_now.unsqueeze(0))
    pole_work.addcmul_(a[2].unsqueeze(-1), pole_state)

    b0_real = b[0, ..., 0].unsqueeze(-1)
    b0_imag = b[0, ..., 1].unsqueeze(-1)
    b1_real = b[1, ..., 0].unsqueeze(-1)
    b1_imag = b[1, ..., 1].unsqueeze(-1)
    b2_real = b[2, ..., 0].unsqueeze(-1)
    b2_imag = b[2, ..., 1].unsqueeze(-1)
    point_work[..., 0].copy_(field_new.unsqueeze(0)).mul_(b0_real)
    point_work[..., 0].addcmul_(b1_real, field_now.unsqueeze(0))
    point_work[..., 0].addcmul_(b2_real, point_state[..., 0])
    point_work[..., 0].addcmul_(b2_imag, point_state[..., 1], value=-1)
    point_work[..., 1].copy_(field_new.unsqueeze(0)).mul_(b0_imag)
    point_work[..., 1].addcmul_(b1_imag, field_now.unsqueeze(0))
    point_work[..., 1].addcmul_(b2_real, point_state[..., 1])
    point_work[..., 1].addcmul_(b2_imag, point_state[..., 0])

    pole_state.copy_(pole_work)
    point_state.copy_(point_work)


def update_bucket(plan, state, descriptor):
    """Apply one exact-width bucket with unique indexed destinations."""
    field, targets = _prepare_curl(plan, state, descriptor)
    if descriptor.model in {"drude", "lorentz"}:
        _update_two_level(plan, state, descriptor)
    elif descriptor.model == "dcp-ade":
        _update_dcp_ade(plan, state, descriptor)
    else:
        _update_dcp_convolution(plan, state, descriptor)
    field_new = getattr(state, f"{descriptor.prefix}_field_new")
    field.index_copy_(0, targets, field_new)


__all__ = [
    "DISPERSIVE_MODELS",
    "DispersiveBucket",
    "register_plan_buffers",
    "register_state_buffers",
    "update_bucket",
]
