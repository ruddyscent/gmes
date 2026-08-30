"""Device-resident exact-width tensor buckets for linear dispersive media."""

from dataclasses import dataclass

import numpy as np
import torch

DISPERSIVE_MODELS = frozenset(("drude", "lorentz", "dcp-ade", "dcp-plrc", "dcp-rc"))
DISPERSIVE_GROUPING_SCOPES = ("combined", "two-level", "dcp-convolution")


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


@dataclass(frozen=True)
class DispersiveSpan:
    """One logical bucket's non-overlapping rows in an exact-schema arena."""

    descriptor: DispersiveBucket
    start: int
    stop: int

    def __post_init__(self):
        if self.start < 0 or self.stop - self.start != self.descriptor.target_count:
            raise ValueError("dispersive span does not match its logical bucket")


@dataclass(frozen=True)
class DispersiveExecutionGroup:
    """Two adjacent logical buckets sharing one physical recurrence."""

    component: str
    prefix: str
    recurrence: str
    pole_count: int
    point_count: int
    target_count: int
    spans: tuple

    def __post_init__(self):
        offset = 0
        for span in self.spans:
            descriptor = span.descriptor
            if (
                descriptor.component != self.component
                or descriptor.pole_count != self.pole_count
                or descriptor.point_count != self.point_count
                or span.start != offset
            ):
                raise ValueError("dispersive group spans changed exact schema or order")
            offset = span.stop
        if len(self.spans) != 2 or offset != self.target_count:
            raise ValueError("dispersive group must contain exactly two full spans")


def _group_recurrence(descriptor):
    if descriptor.model in {"drude", "lorentz"}:
        return "two-level"
    if descriptor.model in {"dcp-plrc", "dcp-rc"}:
        return "dcp-convolution"
    return None


def _matching_pair(first, second, scope):
    recurrence = _group_recurrence(first)
    if recurrence is None or recurrence != _group_recurrence(second):
        return None
    if scope != "combined" and recurrence != scope:
        return None
    if (
        first.component != second.component
        or first.pole_count != second.pole_count
        or first.point_count != second.point_count
    ):
        return None
    models = {first.model, second.model}
    if recurrence == "two-level" and models != {"drude", "lorentz"}:
        return None
    if recurrence == "dcp-convolution" and models != {"dcp-plrc", "dcp-rc"}:
        return None
    return recurrence


class DispersiveExecutionOverlay(torch.nn.Module):
    """CPU-only exact-schema arenas over stable logical bucket identities."""

    _COMMON_BUFFERS = (
        "field_now",
        "field_new",
        "curl",
        "gather_a",
        "gather_b",
        "response",
    )

    def __init__(
        self, plan, descriptors, *, paired_real, dtype, device, scope="combined"
    ):
        super().__init__()
        device = torch.device(device)
        if device.type != "cpu":
            raise ValueError("exact-schema dispersive grouping is CPU-only")
        if scope not in DISPERSIVE_GROUPING_SCOPES:
            raise ValueError(f"unsupported dispersive grouping scope: {scope!r}")

        entries = []
        groups = []
        spans_by_prefix = {}
        channels = 2 if paired_real else 1
        index = 0
        while index < len(descriptors):
            first = descriptors[index]
            second = descriptors[index + 1] if index + 1 < len(descriptors) else None
            recurrence = (
                _matching_pair(first, second, scope) if second is not None else None
            )
            if recurrence is None:
                entries.append(first)
                index += 1
                continue

            descriptors_in_group = (first, second)
            first_stop = first.target_count
            target_count = first_stop + second.target_count
            prefix = f"dispersive_group_{len(groups)}"
            spans = (
                DispersiveSpan(first, 0, first_stop),
                DispersiveSpan(second, first_stop, target_count),
            )
            group = DispersiveExecutionGroup(
                component=first.component,
                prefix=prefix,
                recurrence=recurrence,
                pole_count=first.pole_count,
                point_count=first.point_count,
                target_count=target_count,
                spans=spans,
            )

            def concatenate(suffix, dim):
                return torch.cat(
                    tuple(
                        getattr(plan, f"{descriptor.prefix}_{suffix}")
                        for descriptor in descriptors_in_group
                    ),
                    dim=dim,
                ).contiguous()

            targets = concatenate("targets", 0)
            if torch.unique(targets).numel() != target_count:
                raise ValueError(
                    "grouped dispersive buckets must own disjoint target rows"
                )
            self.register_buffer(f"{prefix}_targets", targets, persistent=False)
            component = plan.components[first.component]
            for term_index, _term in enumerate(component.stencil):
                for direction in ("positive", "negative"):
                    suffix = f"source_{term_index}_{direction}"
                    self.register_buffer(
                        f"{prefix}_{suffix}",
                        concatenate(suffix, 0),
                        persistent=False,
                    )
            for suffix, dim in (("a", 2), ("c", 1)):
                self.register_buffer(
                    f"{prefix}_{suffix}",
                    concatenate(suffix, dim),
                    persistent=False,
                )
            if recurrence == "dcp-convolution":
                self.register_buffer(
                    f"{prefix}_b",
                    concatenate("b", 2),
                    persistent=False,
                )

            common_shape = (target_count, channels)
            for suffix in self._COMMON_BUFFERS:
                self.register_buffer(
                    f"{prefix}_{suffix}",
                    torch.zeros(common_shape, dtype=dtype, device=device),
                    persistent=False,
                )
            pole_shape = (group.pole_count, target_count, channels)
            if recurrence == "two-level":
                for suffix in ("previous", "current", "pole_work", "pole_delta"):
                    self.register_buffer(
                        f"{prefix}_{suffix}",
                        torch.zeros(pole_shape, dtype=dtype, device=device),
                        persistent=False,
                    )
            else:
                point_shape = (group.point_count, target_count, channels, 2)
                for suffix, shape in (
                    ("pole_state", pole_shape),
                    ("point_state", point_shape),
                    ("pole_work", pole_shape),
                    ("point_work", point_shape),
                ):
                    self.register_buffer(
                        f"{prefix}_{suffix}",
                        torch.zeros(shape, dtype=dtype, device=device),
                        persistent=False,
                    )

            entries.append(group)
            groups.append(group)
            for span in spans:
                spans_by_prefix[span.descriptor.prefix] = (group, span)
            index += 2

        self.entries = tuple(entries)
        self.groups = tuple(groups)
        self.scope = scope
        self._spans_by_prefix = spans_by_prefix
        self.requires_grad_(False)

    def logical_state_views(self, descriptor):
        """Return canonical persistent state views for one grouped bucket."""
        item = self._spans_by_prefix.get(descriptor.prefix)
        if item is None:
            return None
        group, span = item
        suffixes = (
            ("previous", "current")
            if group.recurrence == "two-level"
            else ("pole_state", "point_state")
        )
        count = span.stop - span.start
        return tuple(
            (
                suffix,
                getattr(self, f"{group.prefix}_{suffix}").narrow(1, span.start, count),
            )
            for suffix in suffixes
        )


def _canonical_state_dict_post_hook(module, state_dict, prefix, _local_metadata):
    """Expose grouped physical state through canonical contiguous bucket tensors."""
    for name in module._grouped_dispersive_state_names:
        key = f"{prefix}{name}"
        state_dict[key] = (
            state_dict[key].detach().clone(memory_format=torch.contiguous_format)
        )


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


def register_state_buffers(
    module,
    descriptors,
    *,
    paired_real,
    dtype,
    device,
    execution_overlay=None,
):
    """Allocate exact-width mutable state and fixed scratch storage."""
    channels = 2 if paired_real else 1
    grouped_state_names = []

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
        logical_views = (
            execution_overlay.logical_state_views(descriptor)
            if execution_overlay is not None
            else None
        )
        if logical_views is not None:
            for suffix, value in logical_views:
                name = f"{prefix}_{suffix}"
                module.register_buffer(name, value)
                grouped_state_names.append(name)
            continue

        for suffix in DispersiveExecutionOverlay._COMMON_BUFFERS:
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

    if grouped_state_names:
        module._grouped_dispersive_state_names = tuple(grouped_state_names)
        module.register_state_dict_post_hook(_canonical_state_dict_post_hook)


def _prepare_indexed_curl(
    plan,
    state,
    component_name,
    buffers,
    prefix,
    field_now,
    curl,
    gather_a,
    gather_b,
):
    channels = 2 if state.paired_real else 1
    field = state.field(component_name).reshape(-1, channels)
    targets = getattr(buffers, f"{prefix}_targets")
    torch.index_select(field, 0, targets, out=field_now)

    component = plan.components[component_name]
    for term_index, term in enumerate(component.stencil):
        source = state.field(term.source).reshape(-1, channels)
        positive = getattr(buffers, f"{prefix}_source_{term_index}_positive")
        negative = getattr(buffers, f"{prefix}_source_{term_index}_negative")
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


def _prepare_curl(plan, state, descriptor):
    prefix = descriptor.prefix
    return _prepare_indexed_curl(
        plan,
        state,
        descriptor.component,
        plan,
        prefix,
        getattr(state, f"{prefix}_field_now"),
        getattr(state, f"{prefix}_curl"),
        getattr(state, f"{prefix}_gather_a"),
        getattr(state, f"{prefix}_gather_b"),
    )


def _update_two_level_tensors(
    a,
    c,
    previous,
    current,
    pole_work,
    pole_delta,
    field_now,
    field_new,
    curl,
    response,
):
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


def _update_two_level(
    plan,
    state,
    descriptor,
    field_now,
    field_new,
    curl,
    response,
):
    prefix = descriptor.prefix
    _update_two_level_tensors(
        getattr(plan, f"{prefix}_a"),
        getattr(plan, f"{prefix}_c"),
        getattr(state, f"{prefix}_previous"),
        getattr(state, f"{prefix}_current"),
        getattr(state, f"{prefix}_pole_work"),
        getattr(state, f"{prefix}_pole_delta"),
        field_now,
        field_new,
        curl,
        response,
    )


def _update_dcp_ade(
    plan,
    state,
    descriptor,
    field_now,
    field_new,
    curl,
    response,
    combo,
):
    prefix = descriptor.prefix
    a = getattr(plan, f"{prefix}_a")
    b = getattr(plan, f"{prefix}_b")
    c = getattr(plan, f"{prefix}_c")
    field_old = getattr(state, f"{prefix}_field_old")
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


def _update_dcp_convolution_tensors(
    a,
    b,
    c,
    pole_state,
    point_state,
    pole_work,
    point_work,
    field_now,
    field_new,
    curl,
    response,
    point_response,
):
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


def _update_dcp_convolution(
    plan,
    state,
    descriptor,
    field_now,
    field_new,
    curl,
    response,
    point_response,
):
    prefix = descriptor.prefix
    _update_dcp_convolution_tensors(
        getattr(plan, f"{prefix}_a"),
        getattr(plan, f"{prefix}_b"),
        getattr(plan, f"{prefix}_c"),
        getattr(state, f"{prefix}_pole_state"),
        getattr(state, f"{prefix}_point_state"),
        getattr(state, f"{prefix}_pole_work"),
        getattr(state, f"{prefix}_point_work"),
        field_now,
        field_new,
        curl,
        response,
        point_response,
    )


def _update_recurrence(
    plan,
    state,
    descriptor,
    field_now,
    field_new,
    curl,
    gather_a,
    response,
):
    if descriptor.model in {"drude", "lorentz"}:
        _update_two_level(
            plan,
            state,
            descriptor,
            field_now,
            field_new,
            curl,
            response,
        )
    elif descriptor.model == "dcp-ade":
        _update_dcp_ade(
            plan,
            state,
            descriptor,
            field_now,
            field_new,
            curl,
            response,
            gather_a,
        )
    else:
        _update_dcp_convolution(
            plan,
            state,
            descriptor,
            field_now,
            field_new,
            curl,
            response,
            gather_a,
        )


def update_bucket(plan, state, descriptor):
    """Apply one exact-width bucket with unique indexed destinations."""
    field, targets = _prepare_curl(plan, state, descriptor)
    prefix = descriptor.prefix
    field_new = getattr(state, f"{prefix}_field_new")
    _update_recurrence(
        plan,
        state,
        descriptor,
        getattr(state, f"{prefix}_field_now"),
        field_new,
        getattr(state, f"{prefix}_curl"),
        getattr(state, f"{prefix}_gather_a"),
        getattr(state, f"{prefix}_response"),
    )
    field.index_copy_(0, targets, field_new)


def update_group(plan, state, overlay, group):
    """Apply two exact-schema buckets through one physical recurrence."""
    prefix = group.prefix
    field_now = getattr(overlay, f"{prefix}_field_now")
    field_new = getattr(overlay, f"{prefix}_field_new")
    curl = getattr(overlay, f"{prefix}_curl")
    gather_a = getattr(overlay, f"{prefix}_gather_a")
    gather_b = getattr(overlay, f"{prefix}_gather_b")
    response = getattr(overlay, f"{prefix}_response")
    field, targets = _prepare_indexed_curl(
        plan,
        state,
        group.component,
        overlay,
        prefix,
        field_now,
        curl,
        gather_a,
        gather_b,
    )
    if group.recurrence == "two-level":
        _update_two_level_tensors(
            getattr(overlay, f"{prefix}_a"),
            getattr(overlay, f"{prefix}_c"),
            getattr(overlay, f"{prefix}_previous"),
            getattr(overlay, f"{prefix}_current"),
            getattr(overlay, f"{prefix}_pole_work"),
            getattr(overlay, f"{prefix}_pole_delta"),
            field_now,
            field_new,
            curl,
            response,
        )
    else:
        _update_dcp_convolution_tensors(
            getattr(overlay, f"{prefix}_a"),
            getattr(overlay, f"{prefix}_b"),
            getattr(overlay, f"{prefix}_c"),
            getattr(overlay, f"{prefix}_pole_state"),
            getattr(overlay, f"{prefix}_point_state"),
            getattr(overlay, f"{prefix}_pole_work"),
            getattr(overlay, f"{prefix}_point_work"),
            field_now,
            field_new,
            curl,
            response,
            gather_a,
        )
    field.index_copy_(0, targets, field_new)


def update_execution_entry(plan, state, overlay, entry):
    """Apply one canonical bucket or one exact-schema physical group."""
    if isinstance(entry, DispersiveExecutionGroup):
        update_group(plan, state, overlay, entry)
    else:
        update_bucket(plan, state, entry)


__all__ = [
    "DISPERSIVE_MODELS",
    "DispersiveBucket",
    "DispersiveExecutionGroup",
    "DispersiveExecutionOverlay",
    "DispersiveSpan",
    "register_plan_buffers",
    "register_state_buffers",
    "update_bucket",
    "update_execution_entry",
    "update_group",
]
