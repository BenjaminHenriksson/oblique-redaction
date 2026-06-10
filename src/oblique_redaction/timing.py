"""Timing / logging helpers used by every pipeline step.

Output format:
    [HH:MM:SS +123.45s] ▶ Loading polygon
    [HH:MM:SS +123.78s] ✓ Loading polygon (0.33s)
    [HH:MM:SS +180.12s] ▶ Raycasting (n=512,000)
    [HH:MM:SS +181.12s]   · Raycasting: 64,000/512,000 (12.5%) @ 64,000/s, eta 7.0s
    [HH:MM:SS +188.05s] ✓ Raycasting (7.93s)

Usage:
    from oblique_redaction.timing import init_logger, step, log

    init_logger()
    with step("Loading polygon"):
        ...

    with step("Raycasting", total=n_pixels) as s:
        for i in range(0, n_pixels, batch):
            ...
            s.tick(i + batch)
"""
from __future__ import annotations

import logging
import time

_T0: float = 0.0  # set in init_logger; "wall + elapsed" formatting reads this

log = logging.getLogger("oblique_redaction")


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        wall = time.strftime("%H:%M:%S", time.localtime())
        elapsed = time.monotonic() - _T0
        return f"[{wall} +{elapsed:7.2f}s] {record.getMessage()}"


def init_logger(level: int = logging.INFO) -> None:
    """Install our handler/formatter on the package logger. Idempotent."""
    global _T0
    _T0 = time.monotonic()
    handler = logging.StreamHandler()
    handler.setFormatter(_Formatter())
    log.handlers.clear()
    log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False


class _Step:
    """Context manager that logs entry/exit and exposes a throttled progress tick."""

    def __init__(self, name: str, total: int | None = None):
        self.name = name
        self.total = total
        self.start: float = 0.0
        self._last_tick: float = 0.0

    def __enter__(self) -> _Step:
        self.start = time.monotonic()
        self._last_tick = self.start
        suffix = f" (n={self.total:,})" if self.total else ""
        log.info(f"▶ {self.name}{suffix}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dt = time.monotonic() - self.start
        if exc_type is None:
            log.info(f"✓ {self.name} ({dt:.2f}s)")
        else:
            log.error(f"✗ {self.name} ({dt:.2f}s) — {exc_type.__name__}: {exc}")

    def tick(self, done: int, msg: str = "") -> None:
        """Throttled progress line. Calls more often than ~1 Hz are dropped."""
        now = time.monotonic()
        if now - self._last_tick < 1.0 and done != self.total:
            return
        self._last_tick = now
        elapsed = now - self.start
        suffix = f" — {msg}" if msg else ""
        if self.total and done > 0:
            rate = done / elapsed
            eta = (self.total - done) / rate if rate > 0 else 0.0
            pct = 100.0 * done / self.total
            log.info(
                f"  · {self.name}: {done:,}/{self.total:,} ({pct:.1f}%) "
                f"@ {rate:,.0f}/s, eta {eta:.1f}s{suffix}"
            )
        else:
            log.info(f"  · {self.name}: {done:,}{suffix}")


def step(name: str, total: int | None = None) -> _Step:
    return _Step(name, total=total)
