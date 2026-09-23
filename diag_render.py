"""diag_render.py

Generic Developer Console renderer for arbitrary JSON-like diagnostics
sections.

Pure-python (no Streamlit dependency) so it can be unit-tested in
isolation. ``app.py`` imports it as ``_render_diag_items``.

Supported shapes:

  * dict       -> ``key: value`` lines; nested containers recurse
  * list/tuple -> bulleted items; nested containers recurse
  * set        -> bulleted items (iteration order is display-only)
  * scalar     -> inline value; ``None`` / empty string -> "—"

The long-standing appearance is preserved: ``key: value`` layout, ``—``
for empty values, ``•`` for sequence items, and the dedicated
extraction-update format for dicts carrying ``operation``/``field``/
``value``. Nested structures render recursively; whole objects are never
stringified.
"""


def _format_scalar(value):
    if value is None or value == "":
        return "—"
    return str(value)


def _is_extraction_update(item):
    return (
        isinstance(item, dict)
        and "operation" in item
        and "field" in item
        and "value" in item
    )


def _render_extraction_update(item, indent):
    """Dedicated multi-line rendering for extraction updates (operation /
    field / value, optional confidence + label verdict)."""
    pad = " " * indent
    conf = item.get("confidence")
    conf_suffix = ""
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        conf_suffix = f" [confidence {conf:.2f}]"
    verdict = ""
    if "label" in item:
        reason = item.get("reasons") or []
        reason_suffix = f" — {', '.join(reason)}" if reason else ""
        verdict = f"  →  {item['label']}{reason_suffix}"
    return [
        f"{pad}{item['operation']} {item['field']}{conf_suffix}{verdict}:",
        f"{' ' * (indent + 4)}{item['value']}",
    ]


def _render_sequence(items, indent):
    items = list(items)
    if not items:
        return []
    if all(isinstance(v, dict) for v in items):
        lines = []
        for sub in items:
            if _is_extraction_update(sub):
                lines.extend(_render_extraction_update(sub, indent))
            else:
                lines.extend(render_diag_items(sub, indent))
        return lines
    lines = []
    for item in items:
        if isinstance(item, (dict, list, tuple, set)):
            lines.extend(render_diag_items(item, indent))
        else:
            lines.append(f"{' ' * indent}• {_format_scalar(item)}")
    return lines


def render_diag_items(items, indent=2):
    """Render a diagnostics section value to display lines.

    Generic recursive renderer for arbitrary JSON-like structures: dict,
    list, tuple, set, and scalar values. Containers recurse; scalars render
    inline. Extraction-update dicts keep their dedicated appearance.
    """
    if isinstance(items, dict):
        lines = []
        for key, value in items.items():
            pad = " " * indent
            if isinstance(value, dict):
                lines.append(f"{pad}{key}")
                lines.extend(render_diag_items(value, indent + 4))
            elif isinstance(value, (list, tuple, set)):
                if len(value) == 0:
                    lines.append(f"{pad}{key}: —")
                elif all(isinstance(v, dict) for v in value):
                    lines.append(f"{pad}{key}")
                    lines.extend(_render_sequence(value, indent + 4))
                else:
                    lines.append(f"{pad}{key}:")
                    lines.extend(_render_sequence(value, indent + 4))
            else:
                lines.append(f"{pad}{key}: {_format_scalar(value)}")
        return lines
    if isinstance(items, (list, tuple, set)):
        return _render_sequence(items, indent)
    return [_format_scalar(items)]
