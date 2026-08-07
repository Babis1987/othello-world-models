"""Lossless text and table formatting shared by comparison reports."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pct(value: Any, digits: int = 2, *, signed: bool = False) -> str:
    """Format a level (a proportion) as a percentage."""
    number = _finite_float(value)
    if number is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{number * 100:{sign}.{digits}f}%"


def _pp(value: Any, digits: int = 2, *, signed: bool = True) -> str:
    """Format a difference of proportions in percentage points.

    Differences between two percentages are percentage points, not percent.  v1
    printed factorial effects with a ``%`` suffix while the surrounding prose
    called them percentage-point contrasts; this keeps the two consistent.
    """
    number = _finite_float(value)
    if number is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{number * 100:{sign}.{digits}f} pp"


def _number(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number:,.{digits}f}"


def _count(value: Any) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{int(number):,}"


def _millions(value: Any) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number / 1_000_000:.2f}M"


def _megabytes(value: Any) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number / (1024**2):.1f} MiB"


def _short_hash(value: Any) -> str:
    text = str(value or "")
    return f"`{text[:12]}…`" if len(text) > 12 else f"`{text or 'n/a'}`"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _md_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    numeric_columns: Iterable[int] = (),
) -> list[str]:
    numeric = set(numeric_columns)
    align = [
        "---:" if index in numeric else "---"
        for index in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(_escape_cell(item) for item in headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_cell(item) for item in row) + " |"
        for row in rows
    )
    return lines


def _latex_escape(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("★", r"$\star$"),
        ("−", "-"),
        ("–", "--"),
    ):
        text = text.replace(source, target)
    return text


def _latex_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> list[str]:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{'l' * len(headers)}}}",
        r"\toprule",
        " & ".join(_latex_escape(item).replace("<br>", " ") for item in headers)
        + r" \\",
        r"\midrule",
    ]
    lines += [
        " & ".join(_latex_escape(item) for item in row) + r" \\" for row in rows
    ]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    return lines
