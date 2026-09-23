"""developer_console_audit.py — observation-only audit of the Developer
Console render plan.

This module measures the *Developer Console UX* without touching the UI. It
consumes the render plan that ``developer_console.build_render_plan`` already
produces (a flat list of blocks::

    {
        "kind": "performance" | "prompt" | "section",
        "key": str,                 # "Performance", "Prompt", "ProjectState", ...
        "display_name": str,
        "text": str,                # exact rendered display text
        "expanded": bool,           # collapsed/expanded state the plan reflects
    }

) and computes deterministic, observation-only measurements:

    Navigation      total turns, sections per turn, average sections,
                    maximum content depth, expandable count, collapsed count,
                    prompt location depth
    Discoverability prompt / project state / extraction / performance visible
                    without expanding
    Interaction     clicks required to reach Prompt / Project State / Memory /
                    Conversation Failure (cumulative-collapsed model, see
                    ``_clicks_to``)
    Density         average characters per section, largest/smallest section
    Layout          panel width estimate, estimated scrolling distance,
                    visible sections without scrolling

Observation-only contract (mirrors the other audit modules):

    * pure standard library — no Streamlit, no ``mentor``, no prompt logic,
      no conversation logic, no diagnostics producers;
    * never imports ``developer_console`` (it reads the plan *shape*, keeping
      this module a dependency-free leaf);
    * never mutates the plan, never changes any diagnostics, never affects
      UI behaviour — it only reads the plan lists the renderer already emits;
    * golden/rendered output stays byte-identical.

Every metric is a pure function of the plan entries (``key`` / ``kind`` /
``display_name`` / ``text`` / ``expanded``), so a plan re-rendered from the
same diagnostics always yields the same audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "TARGET_KEYS",
    "DeveloperConsoleAudit",
    "DeveloperConsoleAuditSummary",
    "analyze_developer_console",
    "developer_console_diagnostics_section",
]

#: Default assumptions about the on-screen panel (not observable from the
#: render plan; used only to turn line counts into layout estimates). Callers
#: can override via ``panel_config``.
DEFAULT_PANEL_HEIGHT_LINES = 20

#: The section keys the audit treats as interaction/discoverability targets.
#: These mirror the stable ``key`` values that ``developer_console`` emits in
#: every render-plan block.
TARGET_KEYS = {
    "prompt": "Prompt",
    "project_state": "ProjectState",
    "memory": "Memory",
    "conversation_failure": "ConversationFailure",
    "performance": "Performance",
    "extraction": "Extraction",
}


def _plan_entries(render_plan) -> List[Dict[str, Any]]:
    """Normalise a render plan to a list of dict entries (defensive)."""
    if not isinstance(render_plan, (list, tuple)):
        return []
    entries: List[Dict[str, Any]] = []
    for entry in render_plan or []:
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _entry_text(entry: Dict[str, Any]) -> str:
    text = entry.get("text")
    if text is None:
        return ""
    return str(text)


def _entry_lines(entry: Dict[str, Any]) -> int:
    return len(_entry_text(entry).splitlines())


def _max_content_depth(text: str) -> int:
    """Structural nesting depth of a rendered block, recovered from the text.

    The renderer indents every nested container by 4 spaces (see
    ``diag_render``): a flat dict renders at indent 2 (depth 1), its children
    at indent 6 (depth 2), grandchildren at indent 10 (depth 3), and so on.
    ``""`` / single-line / unindented text has depth 1.
    """
    depth = 0
    for line in (text or "").splitlines():
        stripped = line.lstrip(" ")
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        depth = max(depth, 1 + indent // 4)
    return depth if depth else 1 if (text or "").strip() else 0


def _max_line_width(entries: List[Dict[str, Any]]) -> int:
    """Longest rendered line across all blocks (0 when there is none)."""
    width = 0
    for entry in entries:
        for line in _entry_text(entry).splitlines():
            width = max(width, len(line))
    return width


def _block(entries: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    for entry in entries:
        if entry.get("key") == key:
            return entry
    return None


def _visible(entry: Optional[Dict[str, Any]]) -> bool:
    """A block is visible without interaction only when it exists and is
    expanded (a collapsed expander shows just its header)."""
    return bool(entry and entry.get("expanded"))


def _clicks_to(entries: List[Dict[str, Any]], key: str) -> Optional[int]:
    """Clicks required to reach ``key`` reading the panel top-down.

    Model: every collapsed block from the top of the panel through *and
    including* the target must be expanded with one click each before the
    target's content can be reached. Returns ``None`` when the block is not
    present in the plan.
    """
    clicks = 0
    for entry in entries:
        if not entry.get("expanded"):
            clicks += 1
        if entry.get("key") == key:
            return clicks
    return None


def _block_profile(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Per-block measurement used by the diagnostics projection."""
    text = _entry_text(entry)
    expanded = bool(entry.get("expanded"))
    return {
        "key": entry.get("key", ""),
        "kind": entry.get("kind", ""),
        "display_name": entry.get("display_name", entry.get("key", "")),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "expanded": expanded,
        "visible": expanded,
        "depth": _max_content_depth(text),
    }


def _extrema(
    entries: List[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(largest, smallest) block by character count, in plan order on ties."""
    if not entries:
        return None, None
    largest = entries[0]
    smallest = entries[0]
    for entry in entries[1:]:
        if len(_entry_text(entry)) > len(_entry_text(largest)):
            largest = entry
        if len(_entry_text(entry)) < len(_entry_text(smallest)):
            smallest = entry
    return largest, smallest


def _visible_without_scrolling(
    entries: List[Dict[str, Any]], panel_height_lines: int
) -> int:
    """Number of leading blocks that fit fully within ``panel_height_lines``."""
    used = 0
    count = 0
    for entry in entries:
        lines = _entry_lines(entry)
        if used + lines <= panel_height_lines:
            used += lines
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# DeveloperConsoleAudit (per-turn record)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeveloperConsoleAudit:
    """Observation-only per-turn measurements of one Developer Console render
    plan. All fields are derived purely from the plan entries."""

    turn_index: int = 0
    total_turns: int = 1
    sections_per_turn: int = 0
    maximum_depth: int = 0
    expandable_count: int = 0
    collapsed_count: int = 0
    prompt_location_depth: Optional[int] = None
    prompt_visible_without_expanding: bool = False
    project_state_visible: bool = False
    extraction_visible: bool = False
    performance_visible: bool = False
    clicks_to_prompt: Optional[int] = None
    clicks_to_project_state: Optional[int] = None
    clicks_to_memory: Optional[int] = None
    clicks_to_conversation_failure: Optional[int] = None
    average_characters_per_section: float = 0.0
    largest_section: Optional[Dict[str, Any]] = None
    smallest_section: Optional[Dict[str, Any]] = None
    panel_width_estimate: int = 0
    estimated_scrolling_distance: int = 0
    visible_sections_without_scrolling: int = 0
    panel_height_lines: int = DEFAULT_PANEL_HEIGHT_LINES
    section_profiles: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "total_turns": self.total_turns,
            "sections_per_turn": self.sections_per_turn,
            "maximum_depth": self.maximum_depth,
            "expandable_count": self.expandable_count,
            "collapsed_count": self.collapsed_count,
            "prompt_location_depth": self.prompt_location_depth,
            "prompt_visible_without_expanding": self.prompt_visible_without_expanding,
            "project_state_visible": self.project_state_visible,
            "extraction_visible": self.extraction_visible,
            "performance_visible": self.performance_visible,
            "clicks_to_prompt": self.clicks_to_prompt,
            "clicks_to_project_state": self.clicks_to_project_state,
            "clicks_to_memory": self.clicks_to_memory,
            "clicks_to_conversation_failure": self.clicks_to_conversation_failure,
            "average_characters_per_section": self.average_characters_per_section,
            "largest_section": self.largest_section,
            "smallest_section": self.smallest_section,
            "panel_width_estimate": self.panel_width_estimate,
            "estimated_scrolling_distance": self.estimated_scrolling_distance,
            "visible_sections_without_scrolling": self.visible_sections_without_scrolling,
            "panel_height_lines": self.panel_height_lines,
            "section_profiles": list(self.section_profiles),
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DeveloperConsoleAudit":
        d = d or {}
        return cls(
            turn_index=int(d.get("turn_index", 0)),
            total_turns=int(d.get("total_turns", 1)),
            sections_per_turn=int(d.get("sections_per_turn", 0)),
            maximum_depth=int(d.get("maximum_depth", 0)),
            expandable_count=int(d.get("expandable_count", 0)),
            collapsed_count=int(d.get("collapsed_count", 0)),
            prompt_location_depth=(
                int(d["prompt_location_depth"])
                if d.get("prompt_location_depth") is not None
                else None
            ),
            prompt_visible_without_expanding=bool(
                d.get("prompt_visible_without_expanding", False)
            ),
            project_state_visible=bool(d.get("project_state_visible", False)),
            extraction_visible=bool(d.get("extraction_visible", False)),
            performance_visible=bool(d.get("performance_visible", False)),
            clicks_to_prompt=(
                int(d["clicks_to_prompt"]) if d.get("clicks_to_prompt") is not None else None
            ),
            clicks_to_project_state=(
                int(d["clicks_to_project_state"])
                if d.get("clicks_to_project_state") is not None
                else None
            ),
            clicks_to_memory=(
                int(d["clicks_to_memory"]) if d.get("clicks_to_memory") is not None else None
            ),
            clicks_to_conversation_failure=(
                int(d["clicks_to_conversation_failure"])
                if d.get("clicks_to_conversation_failure") is not None
                else None
            ),
            average_characters_per_section=float(
                d.get("average_characters_per_section", 0.0)
            ),
            largest_section=dict(d["largest_section"]) if d.get("largest_section") else None,
            smallest_section=dict(d["smallest_section"]) if d.get("smallest_section") else None,
            panel_width_estimate=int(d.get("panel_width_estimate", 0)),
            estimated_scrolling_distance=int(d.get("estimated_scrolling_distance", 0)),
            visible_sections_without_scrolling=int(
                d.get("visible_sections_without_scrolling", 0)
            ),
            panel_height_lines=int(d.get("panel_height_lines", DEFAULT_PANEL_HEIGHT_LINES)),
            section_profiles=list(d.get("section_profiles", []) or []),
        )

    def to_display(self) -> Dict[str, Any]:
        """Developer Console projection of this turn's audit."""
        return developer_console_diagnostics_section(self)


def analyze_developer_console(
    *,
    render_plan: Any,
    turn_index: int = 0,
    total_turns: int = 1,
    panel_config: Optional[Dict[str, Any]] = None,
) -> DeveloperConsoleAudit:
    """Deterministic, observation-only audit of one render plan.

    ``render_plan`` is the list of blocks produced by
    ``developer_console.build_render_plan``. ``panel_config`` may provide
    ``panel_height_lines`` (default ``DEFAULT_PANEL_HEIGHT_LINES``) used for
    the scrolling/visibility layout estimates. Mutates nothing; the plan is
    only read.
    """
    entries = _plan_entries(render_plan)
    config = dict(panel_config or {})
    panel_height = int(config.get("panel_height_lines", DEFAULT_PANEL_HEIGHT_LINES))

    sections = len(entries)
    collapsed = sum(0 if e.get("expanded") else 1 for e in entries)
    max_depth = max((_max_content_depth(_entry_text(e)) for e in entries), default=0)

    prompt_block = _block(entries, TARGET_KEYS["prompt"])
    prompt_location = None
    for idx, entry in enumerate(entries, start=1):
        if entry.get("key") == TARGET_KEYS["prompt"]:
            prompt_location = idx
            break

    largest, smallest = _extrema(entries)
    profile = [_block_profile(e) for e in entries]

    def _section_ref(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if entry is None:
            return None
        return {
            "key": entry.get("key", ""),
            "kind": entry.get("kind", ""),
            "display_name": entry.get("display_name", entry.get("key", "")),
            "characters": len(_entry_text(entry)),
            "lines": _entry_lines(entry),
        }

    avg_chars = (
        round(sum(len(_entry_text(e)) for e in entries) / sections, 1)
        if sections
        else 0.0
    )

    return DeveloperConsoleAudit(
        turn_index=int(turn_index),
        total_turns=int(total_turns),
        sections_per_turn=sections,
        maximum_depth=max_depth,
        expandable_count=sections,
        collapsed_count=collapsed,
        prompt_location_depth=prompt_location,
        prompt_visible_without_expanding=_visible(prompt_block),
        project_state_visible=_visible(_block(entries, TARGET_KEYS["project_state"])),
        extraction_visible=_visible(_block(entries, TARGET_KEYS["extraction"])),
        performance_visible=_visible(_block(entries, TARGET_KEYS["performance"])),
        clicks_to_prompt=_clicks_to(entries, TARGET_KEYS["prompt"]),
        clicks_to_project_state=_clicks_to(entries, TARGET_KEYS["project_state"]),
        clicks_to_memory=_clicks_to(entries, TARGET_KEYS["memory"]),
        clicks_to_conversation_failure=_clicks_to(
            entries, TARGET_KEYS["conversation_failure"]
        ),
        average_characters_per_section=avg_chars,
        largest_section=_section_ref(largest),
        smallest_section=_section_ref(smallest),
        panel_width_estimate=_max_line_width(entries),
        estimated_scrolling_distance=sum(_entry_lines(e) for e in entries),
        visible_sections_without_scrolling=_visible_without_scrolling(
            entries, panel_height
        ),
        panel_height_lines=panel_height,
        section_profiles=profile,
    )


# ---------------------------------------------------------------------------
# Developer Console per-turn section
# ---------------------------------------------------------------------------


def developer_console_diagnostics_section(audit: Optional[DeveloperConsoleAudit]) -> dict:
    """Developer Console projection of one turn's Developer Console audit.

    Returns an empty dict when there is no audit this turn.
    """
    if not audit:
        return {}
    largest = audit.largest_section
    smallest = audit.smallest_section
    largest_text = (
        f"{largest['display_name']} ({largest['characters']} chars)"
        if largest
        else "—"
    )
    smallest_text = (
        f"{smallest['display_name']} ({smallest['characters']} chars)"
        if smallest
        else "—"
    )
    return {
        "Turn": audit.turn_index,
        "Navigation → Sections": audit.sections_per_turn,
        "Navigation → Max Content Depth": audit.maximum_depth,
        "Navigation → Expandable": audit.expandable_count,
        "Navigation → Collapsed": audit.collapsed_count,
        "Navigation → Prompt Location (1-based)": audit.prompt_location_depth or "—",
        "Discoverability → Prompt": "Yes" if audit.prompt_visible_without_expanding else "No",
        "Discoverability → Project State": "Yes" if audit.project_state_visible else "No",
        "Discoverability → Extraction": "Yes" if audit.extraction_visible else "No",
        "Discoverability → Performance": "Yes" if audit.performance_visible else "No",
        "Interaction → Clicks to Prompt": audit.clicks_to_prompt if audit.clicks_to_prompt is not None else "—",
        "Interaction → Clicks to Project State": audit.clicks_to_project_state if audit.clicks_to_project_state is not None else "—",
        "Interaction → Clicks to Memory": audit.clicks_to_memory if audit.clicks_to_memory is not None else "—",
        "Interaction → Clicks to Conversation Failure": audit.clicks_to_conversation_failure if audit.clicks_to_conversation_failure is not None else "—",
        "Density → Avg Chars per Section": round(audit.average_characters_per_section, 1),
        "Density → Largest Section": largest_text,
        "Density → Smallest Section": smallest_text,
        "Layout → Panel Width Estimate": f"{audit.panel_width_estimate} chars",
        "Layout → Estimated Scroll Lines": audit.estimated_scrolling_distance,
        "Layout → Visible Sections (no scroll)": audit.visible_sections_without_scrolling,
    }


# ---------------------------------------------------------------------------
# DeveloperConsoleAuditSummary (session-level aggregate)
# ---------------------------------------------------------------------------


@dataclass
class DeveloperConsoleAuditSummary:
    """Append-only aggregate of per-turn Developer Console audits.

    JSON-safe and never read by any decision path — Developer Console +
    offline analysis only.
    """

    turn_records: List[dict] = field(default_factory=list)

    # ---- aggregation -------------------------------------------------------

    def add_record(self, record: Any) -> None:
        """Append an audit (DeveloperConsoleAudit, dict, or None)."""
        if not record:
            return
        if isinstance(record, DeveloperConsoleAudit):
            self.turn_records.append(record.to_dict())
        elif isinstance(record, dict):
            self.turn_records.append(
                DeveloperConsoleAudit.from_dict(record).to_dict()
            )

    def merge(self, other: "DeveloperConsoleAuditSummary") -> None:
        self.turn_records.extend(other.turn_records)

    # ---- metrics -----------------------------------------------------------

    def total_turns(self) -> int:
        return len(self.turn_records)

    def _records(self) -> List[DeveloperConsoleAudit]:
        return [DeveloperConsoleAudit.from_dict(r) for r in self.turn_records]

    def average_sections_per_turn(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(sum(r.sections_per_turn for r in self._records()) / turns, 2)

    def maximum_depth(self) -> int:
        return max((r.maximum_depth for r in self._records()), default=0)

    def average_expandable_count(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(sum(r.expandable_count for r in self._records()) / turns, 2)

    def average_collapsed_count(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(sum(r.collapsed_count for r in self._records()) / turns, 2)

    def average_prompt_location_depth(self) -> Optional[float]:
        values = [
            r.prompt_location_depth
            for r in self._records()
            if r.prompt_location_depth is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 2)

    def prompt_discovery_rate(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(
            sum(1 for r in self._records() if r.prompt_visible_without_expanding)
            / turns,
            2,
        )

    def project_state_discovery_rate(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(
            sum(1 for r in self._records() if r.project_state_visible) / turns, 2
        )

    def extraction_discovery_rate(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(
            sum(1 for r in self._records() if r.extraction_visible) / turns, 2
        )

    def performance_discovery_rate(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(
            sum(1 for r in self._records() if r.performance_visible) / turns, 2
        )

    def _average_optional(self, values: List[Optional[int]]) -> Optional[float]:
        present = [v for v in values if v is not None]
        if not present:
            return None
        return round(sum(present) / len(present), 2)

    def average_clicks_to_prompt(self) -> Optional[float]:
        return self._average_optional(
            [r.clicks_to_prompt for r in self._records()]
        )

    def average_clicks_to_project_state(self) -> Optional[float]:
        return self._average_optional(
            [r.clicks_to_project_state for r in self._records()]
        )

    def average_clicks_to_memory(self) -> Optional[float]:
        return self._average_optional([r.clicks_to_memory for r in self._records()])

    def average_clicks_to_conversation_failure(self) -> Optional[float]:
        return self._average_optional(
            [r.clicks_to_conversation_failure for r in self._records()]
        )

    def average_characters_per_section(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(
            sum(r.average_characters_per_section for r in self._records()) / turns,
            1,
        )

    def largest_section(self) -> Optional[Dict[str, Any]]:
        best = None
        for record in self._records():
            candidate = record.largest_section
            if candidate is None:
                continue
            if best is None or candidate.get("characters", 0) > best.get(
                "characters", 0
            ):
                best = candidate
        return best

    def smallest_section(self) -> Optional[Dict[str, Any]]:
        best = None
        for record in self._records():
            candidate = record.smallest_section
            if candidate is None:
                continue
            if best is None or candidate.get("characters", 0) < best.get(
                "characters", 0
            ):
                best = candidate
        return best

    def average_panel_width_estimate(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(
            sum(r.panel_width_estimate for r in self._records()) / turns, 1
        )

    def total_scrolling_distance(self) -> int:
        return sum(r.estimated_scrolling_distance for r in self._records())

    def average_visible_sections_without_scrolling(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(
            sum(r.visible_sections_without_scrolling for r in self._records())
            / turns,
            2,
        )

    # ---- persistence / display ----------------------------------------------

    def to_dict(self) -> dict:
        return {"turn_records": list(self.turn_records)}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DeveloperConsoleAuditSummary":
        d = d or {}
        return cls(turn_records=list(d.get("turn_records", []) or []))

    def to_display(self) -> dict:
        """Developer Console projection of the running aggregate."""
        largest = self.largest_section()
        smallest = self.smallest_section()
        largest_text = (
            f"{largest.get('display_name') or largest.get('key')} "
            f"({largest.get('characters', 0)} chars)"
            if largest
            else "—"
        )
        smallest_text = (
            f"{smallest.get('display_name') or smallest.get('key')} "
            f"({smallest.get('characters', 0)} chars)"
            if smallest
            else "—"
        )
        display: Dict[str, Any] = {
            "Turns Analyzed": self.total_turns(),
            "Navigation → Avg Sections per Turn": self.average_sections_per_turn(),
            "Navigation → Max Content Depth": self.maximum_depth(),
            "Navigation → Avg Expandable": self.average_expandable_count(),
            "Navigation → Avg Collapsed": self.average_collapsed_count(),
            "Navigation → Avg Prompt Location": (
                self.average_prompt_location_depth() or "—"
            ),
            "Discoverability → Prompt Visible Rate": f"{self.prompt_discovery_rate():.0%}",
            "Discoverability → Project State Visible Rate": f"{self.project_state_discovery_rate():.0%}",
            "Discoverability → Extraction Visible Rate": f"{self.extraction_discovery_rate():.0%}",
            "Discoverability → Performance Visible Rate": f"{self.performance_discovery_rate():.0%}",
            "Interaction → Avg Clicks to Prompt": self.average_clicks_to_prompt() or "—",
            "Interaction → Avg Clicks to Project State": self.average_clicks_to_project_state() or "—",
            "Interaction → Avg Clicks to Memory": self.average_clicks_to_memory() or "—",
            "Interaction → Avg Clicks to Conversation Failure": self.average_clicks_to_conversation_failure() or "—",
            "Density → Avg Chars per Section": self.average_characters_per_section(),
            "Density → Largest Section": largest_text,
            "Density → Smallest Section": smallest_text,
            "Layout → Avg Panel Width Estimate": f"{self.average_panel_width_estimate()} chars",
            "Layout → Total Scroll Lines": self.total_scrolling_distance(),
            "Layout → Avg Visible Sections (no scroll)": self.average_visible_sections_without_scrolling(),
        }
        return display
