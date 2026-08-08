"""Shared per-chunk progress rendering for every trainer.

Colab launchers set ``OTHELLO_DISABLE_TQDM=1`` because tqdm's nested bars are
rendered by a piped notebook cell as thousands of separate log lines. This
module provides the replacement contract every trainer uses:

* interactive terminal -> real ``tqdm`` bars;
* piped/Colab launcher  -> :class:`LineProgress`, one ``\\r``-refreshed line.

:class:`LineProgress` erases its line when the chunk finishes, immediately
before the trainer prints that chunk's permanent metrics line. The result is
exactly one permanent row per chunk, with the live bar always sitting on the
row below the last finished chunk.

The notebook side of the contract matters just as much: the cell must stream
the child process byte-wise (``stdout.read1``) rather than with line-based
iteration. Text-mode line iteration applies universal-newline translation,
which turns every ``\\r`` refresh into its own output line and reproduces the
exact log flood this module exists to avoid.
"""

from __future__ import annotations

import os
import sys
import time

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is available on Colab by default
    def tqdm(x=None, **_):
        return x if x is not None else NoOpBar()


class NoOpBar:
    """Minimal stand-in for tqdm when it isn't installed."""

    def update(self, _n: int = 1) -> None: ...
    def set_postfix(self, **_: object) -> None: ...
    def close(self) -> None: ...
    def __enter__(self): return self
    def __exit__(self, *_): ...


def tqdm_disabled() -> bool:
    return os.environ.get("OTHELLO_DISABLE_TQDM", "0") == "1"


class LineProgress:
    """Single-line ``\\r``-refreshed progress bar for piped/non-tty output."""

    def __init__(self, iterable, desc: str = "", min_interval: float = 1.0):
        self._iterable = iterable
        self._desc = desc
        self._min_interval = float(min_interval)
        try:
            self._total = len(iterable)
        except TypeError:
            self._total = None
        self._count = 0
        self._postfix = ""
        self._started_at = time.perf_counter()
        self._last_render_at = 0.0
        self._last_render_len = 0
        self._closed = False

    def __iter__(self):
        try:
            for item in self._iterable:
                yield item
                self._count += 1
                self._maybe_render()
        finally:
            self.close()

    def update(self, n: int = 1) -> None:
        """Manual advance for call sites that do not iterate the bar."""
        self._count += n
        self._maybe_render()

    def set_postfix(self, **kwargs) -> None:
        self._postfix = " ".join(f"{key}={value}" for key, value in kwargs.items())

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def _maybe_render(self) -> None:
        now = time.perf_counter()
        if now - self._last_render_at < self._min_interval:
            return
        self._last_render_at = now
        elapsed = max(now - self._started_at, 1e-9)
        rate = self._count / elapsed
        if self._total:
            fraction = min(self._count / self._total, 1.0)
            filled = int(20 * fraction)
            bar = ("=" * filled + ">" + "." * (19 - filled)) if filled < 20 else "=" * 20
            remaining = (self._total - self._count) / rate if rate > 0 else 0.0
            line = (
                f"{self._desc} [{bar}] {fraction:4.0%} {self._count}/{self._total} "
                f"{rate:.2f}it/s eta {self._format_seconds(remaining)}"
            )
        else:
            line = f"{self._desc} {self._count}it {rate:.2f}it/s"
        if self._postfix:
            line += " | " + self._postfix
        pad = max(self._last_render_len - len(line), 0)
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_render_len = len(line)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._last_render_len:
            sys.stdout.write("\r" + " " * self._last_render_len + "\r")
            sys.stdout.flush()
            self._last_render_len = 0


def make_chunk_progress(iterable, desc: str):
    """Per-chunk progress: tqdm when interactive, ``LineProgress`` when piped."""
    if tqdm_disabled():
        return LineProgress(iterable, desc=desc)
    return tqdm(iterable, desc=desc, leave=False)


def make_manual_progress(desc: str):
    """Progress for loops that advance a bar by hand instead of iterating it.

    The batch count is unknown up front for the prefix-grouped objectives, so
    the piped bar renders a spinner-style ``Nit`` counter rather than a
    percentage. It still occupies exactly one refreshed line.
    """
    if tqdm_disabled():
        return LineProgress(None, desc=desc)
    return tqdm(desc=desc, leave=False)


def close_progress(pbar) -> None:
    """Close a bar produced by :func:`make_chunk_progress`, if it can be."""
    if hasattr(pbar, "close"):
        pbar.close()
