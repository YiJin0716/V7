"""Tabular results as plain data. Values stay raw (numbers, tuples, names); text and HTML exist
only in the reprs. `pd.DataFrame(rows)` and `pa.Table.from_pylist(rows)` take Records as is."""

from __future__ import annotations

import html
from typing import Any


def _text(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (tuple, list)):
        inner = (f"({_text(x)})" if isinstance(x, (tuple, list)) else _text(x) for x in v)
        return ", ".join(inner) or "-"
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_text(x)}" for k, x in v.items())
    return str(v)


class Records(list):
    """Rows: dicts with the same keys."""

    def __getitem__(self, i):  # slices stay Records, so they keep the table repr
        got = super().__getitem__(i)
        return Records(got) if isinstance(i, slice) else got

    def _cells(self) -> tuple[list[str], list[list[str]]]:
        cols = list(self[0]) if self else []
        return cols, [[_text(r[c]) for c in cols] for r in self]

    def __repr__(self) -> str:
        cols, rows = self._cells()
        widths = [max([len(c), *(len(r[i]) for r in rows)]) for i, c in enumerate(cols)]
        if not cols:
            return "(none)"
        pad = lambda cells: "  ".join(x.ljust(w) for x, w in zip(cells, widths, strict=True))
        return "\n".join(pad(cells).rstrip() for cells in [cols, *rows])

    def _repr_html_(self) -> str:
        cols, rows = self._cells()
        cells = lambda tag, xs: "".join(f"<{tag}>{html.escape(x)}</{tag}>" for x in xs)
        body = "".join(f"<tr>{cells('td', r)}</tr>" for r in rows)
        return f"<table><tr>{cells('th', cols)}</tr>{body}</table>"


class Record(dict):
    """Named values (a Case's facts, a view's render choices)."""

    def __repr__(self) -> str:
        width = max(map(len, self), default=0)
        return "\n".join(f"{k.ljust(width)}  {_text(v)}" for k, v in self.items())

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><td style='color:#888;padding-right:12px'>{html.escape(k)}</td>"
            f"<td>{html.escape(_text(v))}</td></tr>"
            for k, v in self.items()
        )
        return f"<table style='font-size:12px'>{rows}</table>"
