"""Preserve unittest subTest continuation for intentionally sequential pytest tests."""

from __future__ import annotations

from contextlib import contextmanager


class FailureCollector:
    """Collect ordinary assertion failures and report them after every case runs."""

    def __init__(self) -> None:
        self._failures: list[Exception] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.raise_if_any()
            return False
        if self._failures and isinstance(exc_value, Exception):
            raise ExceptionGroup(
                "retained sequential subcases failed", [*self._failures, exc_value]
            ) from None
        return False

    @contextmanager
    def case(self, label: str):
        try:
            yield
        except Exception as failure:
            failure.add_note(f"retained sequential subcase: {label}")
            self._failures.append(failure)

    def raise_if_any(self) -> None:
        if self._failures:
            raise ExceptionGroup("retained sequential subcases failed", self._failures)
