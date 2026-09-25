import streamlit as st
import requests
import os
import io
import json
from datetime import datetime
import urllib.parse
import re
import base64
from audio_recorder_streamlit import audio_recorder
from constants import MERMAID_KEYWORDS
from session_lifecycle import SessionLifecycle

# --- Backend Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- Developer Console (collapsible VS Code-style right-side inspector) ---
# All console logic lives in developer_console.py (pure, unit-tested). This
# file only executes the Streamlit primitives against the Conversation tree it
# produces. UI-only — no diagnostic producer, conversation, extraction, or
# objective logic is touched.
from developer_console import (  # noqa: E402
    SectionStatus,
    build_timeline,
    collect_turns,
    highlight_matches,
    highlight_prompt,
    prompt_viewer_stats,
    render_section_text,
    render_timing_lines,
    section_category,
    section_status,
    section_subtitle,
    select_turn,
    status_icon,
    status_label,
    turn_label,
)

# Deterministic conversation export utilities (pure, unit-tested). UI-only —
# no conversation, extraction, or objective logic is touched.
from conversation_export import (  # noqa: E402
    build_conversation_copy,
    build_conversation_diagnostics_copy,
    build_full_markdown,
    conversation_metrics_from_messages,
    serialize_session_json,
)

def _html_escape(text):
    """Escape a plain string for safe embedding inside ``st.html`` blocks."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _nav_key(section_key):
    """Sanitize a section key into a stable Streamlit widget-key suffix."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(section_key))


def _nav_items(turn):
    """Flatten one turn into the IDE navigation tree (VS Code-style "files").

    Every row is a dict of ``key / name / subtitle / status / reason /
    category / kind / text`` derived from the same pure helpers the rest of the
    console uses (``section_status`` + ``section_subtitle``); the synthetic
    ``Performance`` block is prepended when the turn carries timing. Read-only —
    diagnostics are consumed by reference and never mutated.
    """
    items = []
    timing = turn.timing or {}

    performance_text = render_timing_lines(timing)
    if performance_text:
        status, reason = section_status("Performance", timing=turn.timing)
        items.append(
            {
                "key": "Performance",
                "name": "Performance",
                "subtitle": section_subtitle("Performance", timing=turn.timing),
                "status": status,
                "reason": reason,
                "category": section_category("Performance"),
                "kind": "performance",
                "text": performance_text,
            }
        )

    for section in turn.sections:
        status, reason = section_status(section.key, section.items)
        items.append(
            {
                "key": section.key,
                "name": section.display_name,
                "subtitle": section_subtitle(section.key, section.items),
                "status": status,
                "reason": reason,
                "category": section_category(section.key),
                "kind": "prompt" if section.is_prompt else "section",
                "text": render_section_text(section),
            }
        )
    return items

# Timeline stage names → the primary diagnostic section that stage surfaces.
# Used ONLY as a UI search hint so typing a stage name ("Memory Extraction",
# "Prompt Builder") keeps the producing turn and opens the section it renders.
_TIMELINE_STAGE_SECTIONS = (
    ("User Message", None),
    ("Memory Extraction", "Extraction"),
    ("Objective Engine", "ObjectiveTrace"),
    ("Recovery", "Recovery"),
    ("Question Family", "QuestionFamilies"),
    ("Prompt Builder", "Prompt"),
    ("LLM", "Pipeline"),
    ("Response", "ConversationMove"),
    ("Diagnostics", None),
)


def _search_matches_item(query, item) -> bool:
    """True when the (lower-cased) query appears in a nav item's searchable
    fields: section title (name/key/category), diagnostics body (rendered
    content), status labels (Healthy/Warning/Problem), subtitle and reason."""
    if not query:
        return True
    q = query.strip().lower()
    haystack = " ".join(
        str(f)
        for f in (
            item["name"], item["key"], item["category"], item["subtitle"],
            item["reason"], item["text"], status_label(item["status"]),
        )
        if f is not None
    ).lower()
    return q in haystack


def _clip(text, limit=36):
    """Shorten a preview string for a tree row (trailing ellipsis)."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _turn_summary(turn):
    """Overall health + flattened nav items for one turn.

    The worst ``SectionStatus`` across all sections drives the folder icon;
    the items feed search and the explorer section rows. Pure UI read-only."""
    items = _nav_items(turn)
    rank = {
        SectionStatus.HEALTHY: 0,
        SectionStatus.WARNING: 1,
        SectionStatus.PROBLEM: 2,
    }
    worst = SectionStatus.HEALTHY
    for item in items:
        if rank[item["status"]] > rank[worst]:
            worst = item["status"]
    return worst, items


def _explorer_plan(turns, query):
    """Build the Explorer view-model: one entry per matching turn.

    Each entry carries the turn's overall status, its label, a short message
    preview, and the sections that survive the active search. Search keeps a
    turn when its label matches, when any of its sections match, or when the
    query names one of the fixed pipeline-timeline stages; within a kept turn
    only matching sections are shown. Pure read-only UI computation.
    """
    q = (query or "").strip().lower()
    plan = []
    for turn in turns:
        worst, items = _turn_summary(turn)
        kept = items
        turn_match = True
        if q:
            kept = [it for it in items if _search_matches_item(q, it)]
            timeline_hit = False
            if len(q) >= 2:
                for stage_name, primary_key in _TIMELINE_STAGE_SECTIONS:
                    if q in stage_name.lower():
                        timeline_hit = True
                        if primary_key is not None:
                            for it in items:
                                if it["key"] == primary_key and not any(
                                    k["key"] == it["key"] for k in kept
                                ):
                                    kept.append(it)
                        break
            turn_match = (
                turn_label(turn).lower().find(q) != -1
                or bool(kept)
                or timeline_hit
            )
        if not turn_match:
            continue
        plan.append(
            {
                "turn_index": turn.turn_index,
                "label": turn_label(turn),
                "worst": worst,
                "subtitle": _clip(turn.content_preview),
                "kept": kept,
            }
        )
    return plan


def _explorer_html(plan, query, expanded_index, active_key):
    """VSCode-style Explorer: a folder-row per matching turn, opened one at a
    time, with flat per-section rows underneath the open folder. Rows are real
    HTML buttons carrying ``data-expturn`` / ``data-nav`` keys (unique per
    turn) that a scoped script re-dispatches to the hidden Streamlit buttons.
    Search filters the tree: matching turns remain, everything else hides."""
    rank = {
        SectionStatus.HEALTHY: 0,
        SectionStatus.WARNING: 1,
        SectionStatus.PROBLEM: 2,
    }
    rows = []
    for entry in plan:
        idx = entry["turn_index"]
        is_expanded = idx == expanded_index
        chevron = "▾" if is_expanded else "▸"
        sub = (
            f'<span class="dc-exp-sub">{_html_escape(entry["subtitle"])}</span>'
            if entry["subtitle"]
            else ""
        )
        rows.append(
            f'<button type="button" class="dc-exp-row dc-tree-item{" active" if is_expanded else ""}" '
            f'data-expturn="{idx}" aria-expanded="{"true" if is_expanded else "false"}">'
            f'<span class="dc-exp-chev">{chevron}</span>'
            f'<span class="dc-exp-icon">{status_icon(entry["worst"])}</span>'
            f'<span class="dc-exp-body">'
            f'<span class="dc-exp-label">{_html_escape(entry["label"])}</span>{sub}'
            f'</span></button>'
        )
        if not is_expanded:
            continue
        if not entry["kept"]:
            rows.append('<div class="dc-nav-empty">No matching sections</div>')
            continue
        for item in entry["kept"]:
            active = " active" if item["key"] == active_key else ""
            icon = status_icon(item["status"])
            name = _html_escape(item["name"])
            sub_text = _html_escape(item["subtitle"]) if item["subtitle"] else ""
            reason = _html_escape(item["reason"]) if item["reason"] else ""
            detail = f'<span class="dc-nav-reason">{reason}</span>' if reason else ""
            sub_html = (
                f'<span class="dc-nav-sub">{sub_text}{detail}</span>'
                if (sub_text or reason)
                else ""
            )
            rows.append(
                f'<button type="button" class="dc-nav-row dc-exp-child dc-tree-item{active}" '
                f'data-nav="{idx}_{_nav_key(item["key"])}">'
                f'<span class="dc-nav-icon">{icon}</span>'
                f'<span class="dc-nav-body">'
                f'<span class="dc-nav-name">{name}</span>{sub_html}'
                f'</span></button>'
            )
    if not rows:
        return (
            '<div class="dc-nav-empty">No matches</div>'
            if query
            else '<div class="dc-nav-empty">No turns</div>'
        )
    return "\n".join(rows)


def _render_highlighted(text, query):
    """Search hit rendering: the exact section text with every match wrapped in
    ``<mark class="dc-search-hit">`` inside a scrollable monospace block."""
    st.markdown(
        f'<pre class="dc-ide-pre dc-highlight">{highlight_matches(text, query)}</pre>',
        unsafe_allow_html=True,
    )


def _render_prompt_editor(turn_index, text, query=""):
    """Full IDE-style **Prompt editor** for the final prompt sent to the LLM.

    Toolbar shows the ``prompt.txt`` file tab, a Copy button, and the
    character/word/line/byte counts; the body is a monospace, independently
    scrollable (both axes) editor with a sticky line-number gutter and
    lightweight syntax highlighting. When a search query is active the body
    renders the highlighted matches instead. UI-only — the prompt dict is
    never mutated."""
    if query:
        st.markdown(
            f'<div class="dc-ide-tab">📜 prompt.txt</div>'
            f'<pre class="dc-ide-pre dc-highlight">{highlight_matches(text, query)}</pre>',
            unsafe_allow_html=True,
        )
        return
    text = text or ""
    stats = prompt_viewer_stats(text)
    code_id = f"dc-pv-code-{turn_index}-prompt"
    gutter = "".join(f"<i>{n}</i>\n" for n in range(1, stats["lines"] + 1))
    if not gutter:
        gutter = "<i>1</i>"
    highlighted = highlight_prompt(text)
    html_block = f"""
    <div class="dc-prompt-editor">
      <div class="dc-pv-toolbar">
        <span class="dc-pv-tab">📜 prompt.txt</span>
        <span class="dc-pv-stats">{stats['characters']} chars · {stats['words']} words · {stats['lines']} lines · {stats['bytes']} B</span>
        <button class="dc-pv-copy" type="button" data-for="{code_id}">⧉ Copy</button>
      </div>
      <div class="dc-pv-body">
        <div class="dc-pv-gutter"><pre>{gutter}</pre></div>
        <pre class="dc-pv-code" id="{code_id}">{highlighted}</pre>
      </div>
    </div>
    <script>
    (function () {{
      var root = document.currentScript && document.currentScript.parentElement;
      if (!root) return;
      var btn = root.querySelector('.dc-pv-copy');
      var pre = root.querySelector('.dc-pv-code');
      if (!btn || !pre) return;
      btn.addEventListener('click', function () {{
        var text = pre.innerText;
        function fallback() {{
          var ta = document.createElement('textarea');
          ta.value = text;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          try {{ document.execCommand('copy'); }} catch (e) {{}}
          document.body.removeChild(ta);
        }}
        if (navigator.clipboard && window.isSecureContext) {{
          navigator.clipboard.writeText(text).then(markCopied).catch(fallback);
        }} else {{
          fallback(); markCopied();
        }}
        function markCopied() {{
          var old = btn.textContent;
          btn.textContent = '✓ Copied';
          setTimeout(function () {{ btn.textContent = old; }}, 1200);
        }}
      }});
    }})();
    </script>
    """
    st.html(html_block)


def _render_ide_pane(turn_index, item, query):
    """Right-hand editor pane: ONE section at a time (VS Code "opened file").
    The Prompt renders as the full editor; everything else as a large
    scrollable monospace panel with search highlighting when active."""
    if item["kind"] == "prompt":
        _render_prompt_editor(turn_index, item["text"], query)
        return
    if query:
        _render_highlighted(item["text"], query)
    else:
        st.code(item["text"], language="text")


def _render_timeline(turn, query=""):
    """Full-width horizontal pipeline timeline for one selected turn.

    UI-only facade over ``developer_console.build_timeline``: renders the fixed
    pipeline stages as a horizontal, independently scrollable flow of cards
    (icon + name + wall-time + status colour), each card showing its curated
    metadata as chips and a native ``<details>/<summary>`` block that discloses
    that stage's detailed diagnostics. Search filters the cards. Read-only —
    timing and diagnostics are never mutated."""
    stages = build_timeline(turn)
    if query:
        q = query.strip().lower()
        stages = [
            s for s in stages
            if q in " ".join(
                str(f).lower()
                for f in (s.label, s.key, s.status_reason,
                          *(f"{k} {v}" for k, v in s.metadata), s.detail)
            )
        ]
    if not stages:
        st.caption(
            "No timeline data for this turn." if not query
            else "No timeline matches this search."
        )
        return

    status_class = {
        SectionStatus.HEALTHY.value: "healthy",
        SectionStatus.WARNING.value: "warning",
        SectionStatus.PROBLEM.value: "problem",
    }
    cards = []
    for i, stage in enumerate(stages):
        status = SectionStatus(stage.status)
        duration = ""
        if stage.duration_ms is not None:
            ms = stage.duration_ms
            duration = f"{ms / 1000:.2f}s" if ms >= 1000 else f"{int(round(ms))}ms"
        chips = "".join(
            f'<span class="dc-tl-chip">{_html_escape(k)}: <b>{_html_escape(v)}</b></span>'
            for k, v in stage.metadata
        )
        cards.append(f"""
        <div class="dc-tl-card {status_class[stage.status]}">
          <div class="dc-tl-head">
            <span class="dc-tl-icon">{_html_escape(stage.icon)}</span>
            <span class="dc-tl-name">{_html_escape(stage.label)}</span>
            <span class="dc-tl-flag">{status_icon(status)}</span>
          </div>
          <div class="dc-tl-ms">{_html_escape(duration)}</div>
          {f'<div class="dc-tl-reason">{_html_escape(stage.status_reason)}</div>' if stage.status_reason else ''}
          {f'<div class="dc-tl-chips">{chips}</div>' if chips else ''}
          {f'<details class="dc-tl-detail"><summary>Diagnostics</summary><pre class="dc-tl-pre">{_html_escape(stage.detail)}</pre></details>' if stage.detail else ''}
        </div>
        {f'<div class="dc-tl-arrow">→</div>' if i < len(stages) - 1 else ''}
        """)
    st.html('<div class="dc-tl-flow">' + "\n".join(cards) + "</div>")


def _modal_close_button():
    """Small ✕ button in the modal header (also targeted by the ESC/click-out
    script so the overlay can be dismissed from keyboard or backdrop)."""
    return st.button("✕", key="dc_modal_close", help="Close (Esc)")


def render_developer_console_panel():
    """VS Code-style Developer Console modal (UI-only).

    Header (title + ✕, search, Diagnostics / Timeline view toggle) is rendered
    UNCONDITIONALLY so the modal can always be dismissed — the close button
    never disappears behind an early return. Body: a VSCode-style Explorer
    sidebar (~260px) listing every diagnostic turn as a folder (one expanded at
    a time, per-turn section memory) + a single-file right pane, or a friendly
    empty state otherwise. The Timeline view keeps a compact turn selector.
    Pure logic stays in ``developer_console.py``; this function only runs
    Streamlit primitives. Diagnostics and conversation are untouched.
    """
    turns = collect_turns(st.session_state.get("messages", []))
    st.session_state.setdefault("dev_console_view", "diagnostics")
    st.session_state.setdefault("dev_console_active", None)

    # --- Header row 1: title + close (ALWAYS rendered) ---
    title_col, close_col = st.columns([7, 1], gap="small", vertical_alignment="center")
    with title_col:
        st.markdown(
            '<div class="dc-header">🧰 <b>Developer Console</b>'
            '<span class="dc-legend">🟢 Healthy · 🟡 Warning · 🔴 Problem</span></div>',
            unsafe_allow_html=True,
        )
    with close_col:
        if _modal_close_button():
            st.session_state.dev_console_open = False
            st.rerun()

    # --- Header row 2: search + view toggle (ALWAYS rendered) ---
    query = st.text_input(
        "Search",
        key="dc_ide_search",
        placeholder="Search sections, prompt, timeline, diagnostics…",
        label_visibility="collapsed",
    )
    view = st.radio(
        "View",
        options=("📊 Diagnostics", "🕓 Timeline"),
        index=0 if st.session_state["dev_console_view"] == "diagnostics" else 1,
        horizontal=True,
        key="dc_ide_view",
        label_visibility="collapsed",
        help="Diagnostics tree or pipeline Timeline",
    )
    st.session_state["dev_console_view"] = "timeline" if view == "🕓 Timeline" else "diagnostics"

    # --- Body: empty state (no diagnostic turns) vs. explorer/content ---
    if not turns:
        st.markdown(
            '<div class="dc-empty">'
            '<div class="dc-empty-icon">🔍</div>'
            '<div class="dc-empty-title">No diagnostic turns yet</div>'
            '<div class="dc-empty-sub">Start a conversation to inspect reasoning.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # --- Timeline view: keep a compact turn selector for the pipeline flow ---
    if st.session_state["dev_console_view"] == "timeline":
        labels = [turn_label(t) for t in turns]
        current = st.session_state.get("dev_console_selected_turn")
        if current not in {t.turn_index for t in turns}:
            current = turns[-1].turn_index
        picked = st.selectbox(
            "Turn",
            options=labels,
            index=labels.index(turn_label(select_turn(turns, current))),
            key="dc_ide_turn",
            label_visibility="collapsed",
            help="Inspect a specific turn's timeline",
        )
        selected = next(t for t in turns if turn_label(t) == picked)
        st.session_state["dev_console_selected_turn"] = selected.turn_index
        _render_timeline(selected, query)
        return

    # --- Explorer: VS Code-style multi-turn sidebar + single-file pane ---
    plan = _explorer_plan(turns, query)

    # Snapshot the pre-search nav so clearing the query restores it.
    prev_searching = st.session_state.get("dev_console_searching", False)
    searching = bool(query.strip())
    if searching and not prev_searching:
        st.session_state["dev_console_presearch"] = {
            "turn": st.session_state.get("dev_console_selected_turn"),
            "section": st.session_state.get("dev_console_active"),
        }
    st.session_state["dev_console_searching"] = searching

    memory = st.session_state.setdefault("dev_console_turn_section", {})
    expanded = st.session_state.get("dev_console_selected_turn")
    active = st.session_state.get("dev_console_active")
    turn_indexes = {t.turn_index for t in turns}

    if query:
        # Keep a manual selection as long as it still matches the filter;
        # otherwise auto-select the first matched turn + its first section.
        plan_by_index = {e["turn_index"]: e for e in plan}
        entry = plan_by_index.get(expanded)
        kept_keys = {it["key"] for it in entry["kept"]} if entry else set()
        if entry is None or active not in kept_keys:
            matched = [e for e in plan if e["kept"]]
            if matched:
                expanded = matched[0]["turn_index"]
                active = matched[0]["kept"][0]["key"]
            elif plan:
                expanded = plan[0]["turn_index"]
                active = None
            else:
                expanded = None
                active = None
    else:
        # Restore the pre-search selection the first time the query clears.
        if prev_searching and "dev_console_presearch" in st.session_state:
            saved = st.session_state.pop("dev_console_presearch")
            expanded, active = saved.get("turn"), saved.get("section")
        # First open: default to the most recent turn.
        if "dev_console_nav_seeded" not in st.session_state:
            st.session_state["dev_console_nav_seeded"] = True
            expanded = turns[-1].turn_index
        if expanded not in turn_indexes:
            expanded = None
        # Each turn remembers its last-opened section (per-turn memory).
        if expanded is not None:
            item_keys = [it["key"] for it in _nav_items(select_turn(turns, expanded))]
            remembered = memory.get(expanded)
            if remembered in item_keys:
                active = remembered
            elif active not in item_keys:
                active = item_keys[0] if item_keys else None
        else:
            active = None

    st.session_state["dev_console_selected_turn"] = expanded
    st.session_state["dev_console_active"] = active

    nav_col, pane_col = st.columns([0.22, 0.78], gap="small")
    with nav_col:
        with st.container(key="dc_ide_nav"):
            st.markdown(_explorer_html(plan, query, expanded, active), unsafe_allow_html=True)
            for entry in plan:
                if st.button(" ", key=f"dc_turnbtn_{entry['turn_index']}", help=entry["label"]):
                    idx = entry["turn_index"]
                    if expanded == idx:
                        st.session_state["dev_console_selected_turn"] = None
                        st.session_state["dev_console_active"] = None
                    else:
                        st.session_state["dev_console_selected_turn"] = idx
                        item_keys = [it["key"] for it in _nav_items(select_turn(turns, idx))]
                        remembered = memory.get(idx)
                        st.session_state["dev_console_active"] = (
                            remembered if remembered in item_keys
                            else (item_keys[0] if item_keys else None)
                        )
                    st.rerun()
            if expanded is not None:
                entry = next((e for e in plan if e["turn_index"] == expanded), None)
                if entry is not None:
                    for item in entry["kept"]:
                        if st.button(
                            " ",
                            key=f"dc_navbtn_{expanded}_{_nav_key(item['key'])}",
                            help=item["name"],
                        ):
                            st.session_state["dev_console_selected_turn"] = expanded
                            st.session_state["dev_console_active"] = item["key"]
                            memory[expanded] = item["key"]
                            st.rerun()
            if query:
                n = sum(len(e["kept"]) for e in plan)
                m = sum(1 for e in plan if e["kept"])
                st.caption(
                    f"{n} match" + ("es" if n != 1 else "")
                    + f" · {m} turn" + ("s" if m != 1 else "")
                    + " — ↑/↓ navigates results"
                )

    with pane_col:
        with st.container(key="dc_ide_pane"):
            if expanded is None:
                st.markdown(
                    '<div class="dc-empty-sub">'
                    + ("No sections match this search." if (query and not plan) else "Select a turn to inspect its diagnostics.")
                    + '</div>',
                    unsafe_allow_html=True,
                )
            else:
                turn = select_turn(turns, expanded)
                items = {it["key"]: it for it in _nav_items(turn)}
                active_item = items.get(active)
                if active_item is None:
                    st.caption("No matching section in this turn.")
                else:
                    st.markdown(
                        f'<div class="dc-ide-tab">📄 {_html_escape(active_item["name"])}'
                        f'<span class="dc-ide-tab-status">{status_icon(active_item["status"])}</span></div>',
                        unsafe_allow_html=True,
                    )
                    _render_ide_pane(expanded, active_item, query)


def _render_console_overlay():
    """Backdrop + fixed overlay shell around the IDE inspector.

    The shell (dimmed backdrop + ~80vw x 85vh modal) is drawn with plain
    HTML/CSS so it always overlays the app. Closing is handled by the real ✕
    Streamlit button; ESC, backdrop clicks and nav-row clicks are delegated by a
    persistent once-registered global script; and background scroll-lock is
    reconciled on every rerun (locked while this overlay exists, unlocked after
    it is removed). Nothing here mutates diagnostics or the conversation.
    """
    st.html('<div class="dc-modal-backdrop"></div>')
    with st.container(key="dc_modal"):
        render_developer_console_panel()
    # The modal is now in the DOM: lock background scrolling (the persistent
    # script reconciles the state again on the next rerun / after closing).
    st.html("""<script>
      if (window.__dcReconcile) window.__dcReconcile();
    </script>""", unsafe_allow_javascript=True)

# --- Page Config ---
st.set_page_config(
    page_title="ReqGPT | Mission Control",
    page_icon="🚀",
    layout="centered"
)

# --- Inject Beautiful Cyberpunk / High-Tech CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* General Font override */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Main Background custom subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #151922 100%) !important;
    }
    
    /* Sleek Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0e14 !important;
        border-right: 1px solid #1f2937 !important;
    }
    
    /* Custom checklist items card styling */
    .checklist-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, border 0.2s, background 0.2s;
    }
    .checklist-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 123, 255, 0.4);
        background: rgba(255, 255, 255, 0.04);
    }
    
    /* Checklist complete state badge */
    .badge-complete {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 20px;
        font-weight: 600;
        float: right;
    }
    
    /* Checklist missing state badge */
    .badge-missing {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%) !important;
        color: white !important;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 20px;
        font-weight: 600;
        float: right;
    }
    
    /* Card headers */
    .card-header {
        font-weight: 600;
        color: #f3f4f6;
        margin-bottom: 4px;
        font-size: 0.95rem;
    }
    
    /* Card values */
    .card-value {
        color: #9ca3af;
        font-size: 0.85rem;
        font-style: italic;
    }

    /* ------------------------------------------------------------------ */
    /* Developer Console — VS Code-style IDE inspector (modal overlay).    */
    /* Scoped to the console modal container + header cluster; the chat    */
    /* column and sidebar are untouched.                                   */
    /* ------------------------------------------------------------------ */
    @keyframes dc-pop-in {
        from { transform: scale(0.98) translateY(8px); opacity: 0; }
        to   { transform: scale(1) translateY(0); opacity: 1; }
    }

    /* Backdrop: full-viewport dim behind the modal. Clicking it closes. */
    .dc-modal-backdrop {
        position: fixed;
        inset: 0;
        z-index: 999;
        background: rgba(2, 6, 12, 0.62);
        backdrop-filter: blur(3px);
    }

    /* The modal frame: large, centered, dark, rounded, soft shadow. */
    .st-key-dc_modal {
        position: fixed;
        top: 7.5vh;
        left: 10vw;
        width: 80vw;
        height: 85vh;
        z-index: 1000;
        background: #0d131b;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.65);
        animation: dc-pop-in 0.16s ease-out;
        overflow: hidden;
        padding: 0.6rem 0.8rem 0.7rem 0.8rem;
    }

    /* Modal header: title + close, then turn/search/view. */
    .st-key-dc_modal .dc-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: 0.02em;
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        padding: 0.1rem 0 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .st-key-dc_modal .dc-legend {
        font-size: 0.62rem;
        font-weight: 500;
        color: #64748b;
        letter-spacing: 0.04em;
    }
    .st-key-dc_modal_close button {
        font-size: 0.8rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        color: #cbd5e1;
    }
    .st-key-dc_modal_close button:hover {
        background: rgba(239,68,68,0.2);
        border-color: rgba(239,68,68,0.5);
        color: #fca5a5;
    }

    /* Turn selector + search row. */
    .st-key-dc_ide_turn { margin: 0.4rem 0 0.25rem 0; }
    .st-key-dc_ide_turn [data-baseweb="select"] > div { border-color: rgba(255,255,255,0.12); }
    .st-key-dc_ide_search { margin: 0.15rem 0 0.25rem 0; }
    .st-key-dc_ide_search input {
        font-size: 0.78rem;
        border-radius: 6px;
        border-color: rgba(255,255,255,0.12);
    }

    /* Diagnostics / Timeline segmented toggle. */
    .st-key-dc_ide_view { margin: 0.1rem 0 0.4rem 0; }
    .st-key-dc_ide_view > div { display: flex; }
    .st-key-dc_ide_view label {
        font-size: 0.74rem;
        padding: 0.16rem 0.8rem;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 6px;
        margin-right: 0.25rem;
        background: rgba(255,255,255,0.02);
        white-space: nowrap;
    }
    .st-key-dc_ide_view label:has(input:checked) {
        background: rgba(14,165,233,0.18);
        border-color: rgba(14,165,233,0.55);
        color: #ffffff;
    }

    /* Navigation tree: category labels + one file-row per section. */
    .dc-nav-empty {
        font-size: 0.72rem;
        color: #64748b;
        padding: 0.4rem 0.25rem;
        font-style: italic;
    }

    /* Friendly empty state shown when the conversation has no diagnostic
       turns. Rendered as the BODY only — the header (incl. close) stays up. */
    .dc-empty {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.35rem;
        min-height: calc(85vh - 12rem);
        text-align: center;
        padding: 2rem 1rem;
    }
    .dc-empty-icon {
        font-size: 2.4rem;
        opacity: 0.55;
    }
    .dc-empty-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .dc-empty-sub {
        font-size: 0.82rem;
        color: #64748b;
        max-width: 34rem;
        line-height: 1.5;
    }
    .dc-nav-row {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        width: 100%;
        text-align: left;
        background: transparent;
        border: none;
        border-left: 2px solid transparent;
        border-radius: 4px;
        padding: 0.22rem 0.3rem;
        margin: 1px 0;
        cursor: pointer;
        font-family: inherit;
        color: #cbd5e1;
    }
    .dc-nav-row:hover { background: rgba(255,255,255,0.05); }
    .dc-nav-row.active {
        background: rgba(14,165,233,0.12);
        border-left-color: #38bdf8;
        color: #f1f5f9;
    }
    .dc-nav-icon { font-size: 0.8rem; width: 1rem; text-align: center; flex: 0 0 auto; }
    .dc-nav-body { min-width: 0; flex: 1 1 auto; }
    .dc-nav-name {
        font-size: 0.76rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .dc-nav-sub {
        font-size: 0.64rem;
        color: #64748b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .dc-nav-reason {
        flex: 0 0 auto;
        font-size: 0.6rem;
        color: #94a3b8;
        max-width: 9rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Explorer: VS Code-style sidebar — a folder-row per turn (one open at a
       time) with flat per-section rows underneath. Independent scroll. */
    .dc-exp-row {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        width: 100%;
        text-align: left;
        background: transparent;
        border: none;
        border-left: 2px solid transparent;
        border-radius: 4px;
        padding: 0.24rem 0.3rem;
        margin: 2px 0 1px 0;
        cursor: pointer;
        font-family: inherit;
        color: #e2e8f0;
    }
    .dc-exp-row:hover { background: rgba(255,255,255,0.06); }
    .dc-exp-row.active {
        background: rgba(14,165,233,0.10);
        border-left-color: #38bdf8;
        color: #f1f5f9;
    }
    .dc-exp-chev {
        flex: 0 0 auto;
        width: 0.7rem;
        text-align: center;
        font-size: 0.6rem;
        color: #7dd3fc;
    }
    .dc-exp-icon { flex: 0 0 auto; width: 1rem; text-align: center; font-size: 0.8rem; }
    .dc-exp-body { min-width: 0; flex: 1 1 auto; }
    .dc-exp-label {
        display: block;
        font-size: 0.78rem;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .dc-exp-sub {
        display: block;
        font-size: 0.63rem;
        color: #64748b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    /* Section rows are indented under their turn folder. */
    .dc-nav-row.dc-exp-child { padding-left: 1.15rem; }
    .dc-tree-item { outline: none; }

    /* The Explorer sidebar scrolls independently of the content pane. */
    .st-key-dc_ide_nav { max-height: calc(85vh - 11rem); overflow-y: auto; }

    /* Real Streamlit buttons behind the nav rows stay hidden. */
    [class*="st-key-dc_navbtn_"] { display: none; }
    [class*="st-key-dc_turnbtn_"] { display: none; }

    /* Right pane: single selected section, editor-style. */
    .st-key-dc_ide_pane {
        height: 100%;
        overflow: hidden;
        padding: 0 0 0 0.75rem;
        border-left: 1px solid rgba(255,255,255,0.08);
    }
    .dc-ide-tab {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.74rem;
        font-weight: 700;
        color: #e2e8f0;
        padding: 0.35rem 0.5rem;
        margin-bottom: 0.4rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 6px 6px 0 0;
        border-bottom: 2px solid #38bdf8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .dc-ide-tab-status { margin-left: auto; font-size: 0.7rem; }
    .dc-ide-pane-scroll {
        max-height: calc(85vh - 11rem);
        overflow: auto;
        padding-right: 0.3rem;
    }
    .dc-ide-body {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.74rem;
        line-height: 1.5;
        color: #cbd5e1;
        white-space: pre-wrap;
        word-break: break-word;
        background: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 6px;
        padding: 0.45rem 0.5rem;
        margin: 0.2rem 0;
    }

    /* Prompt Viewer: full-file editor with line-number gutter + toolbar. */
    .dc-prompt-viewer,
    .dc-prompt-editor {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        background: rgba(255,255,255,0.015);
        margin: 0.2rem 0;
        overflow: hidden;
    }
    .dc-pv-toolbar {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.3rem 0.5rem;
        background: rgba(255,255,255,0.03);
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .dc-pv-title,
    .dc-pv-tab {
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7dd3fc;
    }
    .dc-pv-stats {
        flex: 1 1 auto;
        font-size: 0.62rem;
        color: #64748b;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .dc-pv-copy {
        font-size: 0.66rem;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        color: #cbd5e1;
        background: rgba(59,130,246,0.14);
        border: 1px solid rgba(59,130,246,0.4);
        border-radius: 4px;
        padding: 0.12rem 0.5rem;
        cursor: pointer;
        white-space: nowrap;
    }
    .dc-pv-copy:hover { background: rgba(59,130,246,0.28); }
    .dc-pv-body {
        display: grid;
        grid-template-columns: max-content auto;
        width: max-content;
        min-width: 100%;
        max-height: calc(85vh - 13.5rem);
        overflow: auto;
    }
    .dc-pv-gutter {
        position: sticky;
        left: 0;
        z-index: 1;
        background: rgba(15, 20, 30, 0.98);
        border-right: 1px solid rgba(255,255,255,0.06);
        text-align: right;
        user-select: none;
    }
    .dc-pv-gutter pre {
        margin: 0;
        padding: 0.4rem 0.5rem 0.4rem 0;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.72rem;
        line-height: 1.5;
        color: #475569;
        background: transparent;
    }
    .dc-pv-code {
        margin: 0;
        padding: 0.4rem 0.5rem;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.72rem;
        line-height: 1.5;
        color: #cbd5e1;
        white-space: pre;
        overflow: visible;
        background: transparent;
    }
    .dc-pv-code .pc-h { display: block; color: #93c5fd; font-weight: 700; }
    .dc-pv-code .pc-hline { display: block; color: #334155; font-weight: 700; }
    .dc-pv-code .pc-list { color: #a5b4fc; }
    .dc-pv-code .pc-tokline { color: #fde68a; }
    .dc-pv-code .pc-tok { color: #fbbf24; font-weight: 700; }
    .dc-pv-code .pc-list .pc-tok { color: #fbbf24; }

    /* Search hits: highlighted section body while a query is active. */
    .dc-highlight {
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 0.74rem;
        line-height: 1.5;
        color: #cbd5e1;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 4px;
        padding: 0.45rem 0.5rem;
        margin: 0.2rem 0;
    }
    mark.dc-search-hit {
        background: rgba(245, 158, 11, 0.25);
        color: #fbbf24;
        border-radius: 2px;
        padding: 0 1px;
    }
    .dc-search-caption {
        font-size: 0.64rem;
        color: #64748b;
        padding: 0.15rem 0.25rem;
    }

    /* Pipeline Timeline: horizontal IDE-style node flow. */
    .dc-tl-flow {
        display: flex;
        flex-wrap: nowrap;
        gap: 0.4rem;
        overflow-x: auto;
        overflow-y: hidden;
        padding: 0.3rem 0.1rem 0.6rem 0.1rem;
    }
    .dc-tl-card {
        flex: 0 0 230px;
        min-width: 230px;
        max-width: 260px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.08);
        border-top: 3px solid #64748b;
        border-radius: 8px;
        padding: 0.45rem 0.6rem;
    }
    .dc-tl-card.healthy { border-top-color: #22c55e; }
    .dc-tl-card.warning { border-top-color: #eab308; }
    .dc-tl-card.problem { border-top-color: #ef4444; }
    .dc-tl-arrow {
        flex: 0 0 auto;
        align-self: center;
        color: #475569;
        font-size: 1rem;
    }
    .dc-tl-head { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }
    .dc-tl-icon { font-size: 0.95rem; line-height: 1; }
    .dc-tl-name { font-weight: 600; color: #f3f4f6; font-size: 0.82rem; }
    .dc-tl-ms { color: #94a3b8; font-size: 0.7rem; margin-left: auto; font-variant-numeric: tabular-nums; }
    .dc-tl-flag { font-size: 0.8rem; }
    .dc-tl-reason { color: #facc15; font-size: 0.7rem; margin-top: 0.2rem; }
    .dc-tl-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.3rem; }
    .dc-tl-chip {
        font-size: 0.64rem;
        color: #cbd5e1;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 4px;
        padding: 0.04rem 0.3rem;
    }
    .dc-tl-chip b { color: #e2e8f0; font-weight: 600; }
    .dc-tl-detail { margin-top: 0.35rem; }
    .dc-tl-detail summary {
        cursor: pointer;
        font-size: 0.68rem;
        color: #38bdf8;
        user-select: none;
    }
    .dc-tl-detail summary:hover { color: #7dd3fc; }
    .dc-tl-pre {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
        font-size: 0.66rem;
        line-height: 1.4;
        white-space: pre-wrap;
        word-break: break-word;
        background: rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 6px;
        padding: 0.35rem;
        margin: 0.35rem 0 0 0;
        max-height: 180px;
        overflow-y: auto;
        color: #cbd5e1;
    }

    /* Header cluster: Console toggle + ⋮ menu, right-aligned with title. */
    .st-key-dev_console_toggle { margin-top: 1.35rem; }
    .st-key-dev_console_toggle button {
        font-size: 0.75rem;
        padding: 0.4rem 0.55rem;
        border-radius: 6px;
        white-space: nowrap;
        font-weight: 600;
    }
    .st-key-dc_header_menu { margin-top: 1.35rem; }
    .st-key-dc_header_menu button {
        font-size: 1rem;
        color: #cbd5e1;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.03);
        border-radius: 6px;
        padding: 0.4rem 0.5rem;
    }
    .dc-menu-title {
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }

    /* Responsive: keep the modal slightly larger on narrow windows but ALWAYS
       leave visible margins — it must never occupy the whole browser window. */
    @media (max-width: 1199px) {
        .st-key-dc_modal {
            top: 4vh;
            left: 4vw;
            width: 92vw;
            height: 92vh;
        }
        /* Keep the Explorer on narrow windows — it IS the turn selector. */
        .st-key-dc_ide_pane { border-left: none; padding-left: 0; }
    }
</style>
""", unsafe_allow_html=True)

st.html("""<script>
/* Developer Console modal glue. This runs on EVERY Streamlit rerun (it is
   injected unconditionally), but the document-level listeners are registered
   only once per page load. Closing the modal is driven by the real ✕ button;
   this script re-dispatches ESC, backdrop clicks and nav-row clicks to it, and
   keeps the background scroll-lock in sync with the modal's presence.
   UI-only: it never touches diagnostics or conversation state. */
(function () {
  var CLOSE_SEL = '[class*="st-key-dc_modal_close"] button';

  function scrollTargets() {
    return [
      document.documentElement,
      document.body,
      document.querySelector('[data-testid="stAppViewContainer"]'),
      document.querySelector('[data-testid="stMain"]'),
    ];
  }
  function lockBackground() {
    scrollTargets().forEach(function (el) { if (el) el.style.overflow = 'hidden'; });
  }
  function unlockBackground() {
    scrollTargets().forEach(function (el) { if (el) el.style.overflow = ''; });
  }
  function reconcile() {
    if (document.querySelector('.st-key-dc_modal')) {
      lockBackground();
    } else {
      unlockBackground();
    }
  }
  function closeModal() {
    var btn = document.querySelector(CLOSE_SEL);
    if (btn) btn.click();
  }

  function findRealBtn(prefix, suffix) {
    var target = 'st-key-' + prefix + suffix;
    var found = null;
    document.querySelectorAll('[class*="st-key-' + prefix + '"] button')
      .forEach(function (b) {
        var wrap = b.closest('[class*="st-key-' + prefix + '"]');
        if (wrap && wrap.classList && wrap.classList.contains(target)) found = b;
      });
    return found;
  }

  if (!window.__dcModalReady) {
    window.__dcModalReady = true;

    document.addEventListener('click', function (ev) {
      var t = ev.target;
      if (t && t.classList && t.classList.contains('dc-modal-backdrop')) {
        ev.preventDefault();
        ev.stopPropagation();
        closeModal();
        return;
      }
      var turnRow = t && t.closest ? t.closest('.dc-exp-row') : null;
      if (turnRow && turnRow.getAttribute('data-expturn') != null) {
        var idx = turnRow.getAttribute('data-expturn');
        var tb = findRealBtn('dc_turnbtn_', idx);
        if (tb) tb.click();
        return;
      }
      var row = t && t.closest ? t.closest('.dc-nav-row') : null;
      if (row && row.getAttribute('data-nav')) {
        var key = row.getAttribute('data-nav');
        var real = findRealBtn('dc_navbtn_', key);
        if (real) real.click();
      }
    }, true);

    document.addEventListener('keydown', function (ev) {
      if (!document.querySelector('.st-key-dc_modal')) return;
      if (ev.key === 'Escape') {
        ev.preventDefault();
        ev.stopPropagation();
        closeModal();
        return;
      }
      if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
        var rows = Array.prototype.slice.call(document.querySelectorAll('.dc-tree-item'));
        if (!rows.length) return;
        var cur = rows.indexOf(document.activeElement);
        var next = ev.key === 'ArrowDown' ? cur + 1 : cur - 1;
        if (next >= rows.length) next = 0;
        if (next < 0) next = rows.length - 1;
        var el = rows[next];
        if (el) { el.focus(); el.scrollIntoView({ block: 'nearest' }); }
        ev.preventDefault();
      }
    });
  }

  /* Reconcile the scroll-lock after the DOM settles. A debounced MutationObserver
     (registered once) reacts to the modal being added/removed during any rerun,
     so the background locks while the modal is visible and always unlocks the
     moment it disappears — no stale overlay or stuck scroll. */
  var reconcileTimer = null;
  function scheduleReconcile() {
    clearTimeout(reconcileTimer);
    reconcileTimer = setTimeout(reconcile, 60);
  }
  if (window.__dcObserverRunning !== true) {
    window.__dcObserverRunning = true;
    var observerTarget = document.body || document.documentElement;
    if (observerTarget) {
      var observer = new MutationObserver(scheduleReconcile);
      observer.observe(observerTarget, { childList: true, subtree: true });
    }
  }

  reconcile();
  scheduleReconcile();
  window.__dcReconcile = reconcile;
  window.__dcCloseModal = closeModal;
})();
</script>""", unsafe_allow_javascript=True)

# --- Persistence Helpers ---
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def save_session(name):
    data = {
        "messages": st.session_state.messages,
        "dt_phase": st.session_state.dt_phase,
        "timestamp": datetime.now().isoformat()
    }
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name) or "unnamed"
    filepath = os.path.join(SESSION_DIR, f"{safe_name}.json")
    try:
        with open(filepath, "w") as f:
            json.dump(data, f)
    except Exception as e:
        st.error(f"Failed to save session: {e}")

def load_session(name):
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name) or "unnamed"
    filepath = os.path.join(SESSION_DIR, f"{safe_name}.json")
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            st.session_state.messages = data["messages"]
            st.session_state.dt_phase = data["dt_phase"]
    except FileNotFoundError:
        st.error(f"Session '{name}' not found.")
    except Exception as e:
        st.error(f"Failed to load session: {e}")

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Design Thinking Mentor. I'll guide you through the Empathize stage by asking thoughtful questions — not giving answers. What project or idea would you like to explore today?"}
    ]
if "dt_phase" not in st.session_state:
    st.session_state.dt_phase = "Empathize"
if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = "Design Thinking Coach"
if "enable_voice" not in st.session_state:
    st.session_state.enable_voice = False
if "dev_console_open" not in st.session_state:
    st.session_state.dev_console_open = False

# --- Session Lifecycle Startup ---
# Opening the application creates a FRESH session unless an explicit session
# restore is requested (session_lifecycle.SessionLifecycle is the single,
# unit-tested decision authority). Runs once per page load, captures the
# bound session id, and never requires a manual browser refresh afterwards.
if "session_binding_initialized" not in st.session_state:
    st.session_state.session_binding_initialized = True
    decision = SessionLifecycle.decide_startup(
        explicit_restore=None,
        project_name=st.session_state.get("project_name", "MyProject"),
    )
    try:
        start_resp = requests.post(f"{BACKEND_URL}/session/start", json={
            "mode": decision.backend_mode,
            "username": "User",
            "project_name": st.session_state.get("project_name", "MyProject"),
        }, timeout=10)
        if start_resp.status_code == 200:
            start_data = start_resp.json()
            st.session_state.session_id = start_data.get("session_id", "")
            st.session_state.session_action = start_data.get("action", decision.action)
            st.session_state.manual_refresh_needed = start_data.get(
                "manual_refresh_needed", True
            )
            print(
                f"[app] Startup session {start_data.get('action')} "
                f"bound: {st.session_state.session_id}"
            )
        else:
            print(f"[app] Startup session start returned {start_resp.status_code}")
            st.session_state.manual_refresh_needed = True
    except Exception as e:
        print(f"[app] Startup session start failed (backend may not be ready yet): {e}")
        st.session_state.manual_refresh_needed = True

# --- Backend Communication ---

def call_backend_text(text, phase, doc_ctx="", username="User"):
    try:
        payload = {
            "text": text, 
            "pod": phase, 
            "username": username, 
            "project_name": st.session_state.get("project_name", "MyProject"),
            "context_doc": doc_ctx, 
            "interaction_mode": st.session_state.get("interaction_mode", "Design Thinking Coach"),
            "enable_voice": st.session_state.get("enable_voice", False)
        }
        response = requests.post(f"{BACKEND_URL}/text", json=payload)

        if response.status_code == 200:
            reply = urllib.parse.unquote(response.headers.get("X-Reply", ""))
            timing_raw = response.headers.get("X-Timing")
            timing = json.loads(urllib.parse.unquote(timing_raw)) if timing_raw else None
            diag_raw = response.headers.get("X-Diagnostics")
            diagnostics = json.loads(urllib.parse.unquote(diag_raw)) if diag_raw else None
            return reply, response.content, timing, diagnostics # reply text, audio bytes, timing, diagnostics
        else:
            try:
                err_msg = response.json().get("error", f"Error {response.status_code}")
            except Exception:
                err_msg = f"Error {response.status_code}"
            return f"⚠️ {err_msg}", None, None, None
    except Exception as e:
        return f"⚠️ Backend Unreachable: {e}", None, None, None

def call_backend_voice(audio_bytes, phase, doc_ctx="", username="User"):
    try:
        headers = {
            "X-Pod": phase,
            "X-Username": username,
            "X-Project-Name": urllib.parse.quote(st.session_state.get("project_name", "MyProject")),
            "X-Context-Doc": urllib.parse.quote(doc_ctx[:2000]), # Limit context size for headers
            "X-Interaction-Mode": st.session_state.get("interaction_mode", "Design Thinking Coach"),
            "X-Enable-Voice": str(st.session_state.get("enable_voice", True))
        }
        response = requests.post(f"{BACKEND_URL}/voice", data=audio_bytes, headers=headers)

        if response.status_code == 200:
            transcript = urllib.parse.unquote(response.headers.get("X-Transcript", ""))
            reply = urllib.parse.unquote(response.headers.get("X-Reply", ""))
            timing_raw = response.headers.get("X-Timing")
            timing = json.loads(urllib.parse.unquote(timing_raw)) if timing_raw else None
            diag_raw = response.headers.get("X-Diagnostics")
            diagnostics = json.loads(urllib.parse.unquote(diag_raw)) if diag_raw else None
            return transcript, reply, response.content, timing, diagnostics
        else:
            try:
                err_msg = response.json().get("error", f"Error {response.status_code}")
            except Exception:
                err_msg = f"Error {response.status_code}"
            return None, f"⚠️ {err_msg}", None, None, None
    except Exception as e:
        return None, f"⚠️ Backend Unreachable: {e}", None, None, None

def call_backend_reqgpt(prompt):
    try:
        payload = {"prompt": prompt}
        response = requests.post(f"{BACKEND_URL}/generate_requirements", json=payload)
        if response.status_code == 200:
            return response.json().get("requirement", "")
        else:
            return f"⚠️ Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"⚠️ Backend Unreachable: {e}"

def call_backend_visualize(username="User", doc_ctx=""):
    try:
        payload = {"username": username, "context_doc": doc_ctx}
        response = requests.post(f"{BACKEND_URL}/visualize", json=payload)
        if response.status_code == 200:
            return response.text
        else:
            return "graph TD\n  A[⚠️ Backend Error] --> B[Could not generate]"
    except Exception as e:
        return f"graph TD\n  A[⚠️ Unreachable] --> B[{str(e)[:50]}]"

# --- Brain Logic (Moved to Backend) ---
# generate_chat_response is now handled by call_backend_text and call_backend_voice

# --- Mermaid Rendering ---
def render_mermaid(code):
    # Strip markdown code blocks if present
    code = code.replace("```mermaid", "").replace("```", "").strip()
    
    # Supported keywords
    if not any(k in code for k in MERMAID_KEYWORDS):
        return # Not mermaid code

    html_code = f"""
    <div id="mermaid-container" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <pre class="mermaid" style="display: flex; justify-content: center;">
            {code}
        </pre>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: true, 
            theme: 'base',
            themeVariables: {{
                'primaryColor': '#007bff',
                'edgeColor': '#555555'
            }},
            securityLevel: 'loose'
        }});
        // Force a re-render in case startOnLoad misses the dynamic content
        setTimeout(() => {{
            mermaid.contentLoaded();
        }}, 500);
    </script>
    """
    # Dynamic height based on lines of code (rough estimate)
    lines = code.count("\n")
    calc_height = max(400, (lines + 2) * 25)
    st.components.v1.html(html_code, height=calc_height, scrolling=True)

def format_prd():
    doc = f"# Product Requirements Document (PRD)\n"
    doc += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    doc += f"## Project Log\n"
    for m in st.session_state.messages:
        role = "Consultant" if m["role"] == "assistant" else "User"
        doc += f"**{role}:** {m['content']}\n\n"
    return doc

# --- UI Layout ---
# Top-right header cluster: title left; Console button + ⋮ menu right.
title_col, console_col, menu_col = st.columns([0.76, 0.12, 0.12], gap="small", vertical_alignment="center")
with title_col:
    st.title("🚀 ThinkingPods: Mission Control")
with console_col:
    if st.button(
        "🧰 Console",
        key="dev_console_toggle",
        use_container_width=True,
        help="Open the Developer Console inspector (Esc closes)",
    ):
        st.session_state.dev_console_open = True
        st.rerun()
with menu_col:
    with st.popover("⋮", key="dc_header_menu", help="Developer Console menu"):
        st.markdown('<div class="dc-menu-title">Console actions</div>', unsafe_allow_html=True)
        if st.button("♻ Reset View", key="dc_menu_reset", use_container_width=True):
            st.session_state.pop("dc_ide_search", None)
            st.session_state.pop("dev_console_active", None)
            st.session_state.pop("dev_console_view", None)
            st.session_state.pop("dev_console_selected_turn", None)
            st.session_state.pop("dev_console_turn_section", None)
            st.session_state.pop("dev_console_searching", None)
            st.session_state.pop("dev_console_presearch", None)
            st.session_state.pop("dev_console_nav_seeded", None)
            st.rerun()
        if st.button("🧰 Open Console", key="dc_menu_open", use_container_width=True):
            st.session_state.dev_console_open = True
            st.rerun()
        st.divider()
        st.markdown('<div class="dc-menu-title">Export</div>', unsafe_allow_html=True)
        if st.button("📋 Copy Diagnostics", key="dc_menu_copy", use_container_width=True):
            st.code(
                build_conversation_diagnostics_copy(
                    st.session_state.get("messages", []),
                    project_name=st.session_state.get("project_name", "MyProject"),
                ),
                language="markdown",
            )
        _console_json = serialize_session_json(
            st.session_state.get("messages", []),
            project_name=st.session_state.get("project_name", "MyProject"),
            dt_phase=st.session_state.get("dt_phase", "Empathize"),
            session_id=st.session_state.get("session_id", ""),
            timestamp=datetime.now().isoformat(),
            saved_missions=None,
            conversation_metrics=conversation_metrics_from_messages(
                st.session_state.get("messages", [])
            ),
        )
        st.download_button(
            "⬇️ Export JSON",
            data=_console_json,
            file_name="console_session.json",
            mime="application/json",
            use_container_width=True,
        )

with st.sidebar:
    # New Session button at the very top of the sidebar — always visible without scrolling
    if st.button("🆕 New Session", use_container_width=True):
        try:
            resp = requests.post(f"{BACKEND_URL}/session/new", json={
                "username": "User",
                "project_name": st.session_state.get("project_name", "MyProject")
            })
            if resp.status_code == 200:
                result = resp.json()
                st.session_state.session_id = result.get("session_id", "")
                st.session_state.session_action = "fresh"
                st.session_state.manual_refresh_needed = False
                st.success(f"New session started: {result.get('session_id', 'unknown')}")
            else:
                st.warning("Could not create new session")
        except Exception as e:
            st.error(f"Failed to create new session: {e}")
        # Clear frontend state
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Design Thinking Mentor. I'll guide you through the Empathize stage by asking thoughtful questions — not giving answers. What project or idea would you like to explore today?"}]
        st.session_state.dt_phase = "Empathize"
        st.rerun()

    st.divider()
    st.header("💾 Mission Memory")

    # Save/Load UI
    project_name = st.text_input("Project Name", value="MyProject", key="project_name")
    if st.button("Save Mission"):
        save_session(project_name)
        st.success(f"Saved to {project_name}")

    try:
        existing_sessions = [f.replace(".json", "") for f in os.listdir(SESSION_DIR) if f.endswith(".json")]
    except FileNotFoundError:
        existing_sessions = []
    if existing_sessions:
        selected_session = st.selectbox("Load Mission", ["None"] + existing_sessions)
        if selected_session != "None" and st.button("Load Now"):
            load_session(selected_session)
            st.rerun()

    st.divider()

    # Interaction Mode Selector (ChatGPT-like feature)
    st.header("💬 Interaction Mode")
    interaction_mode = st.selectbox(
        "Choose AI behavior:",
        ["Design Thinking Coach"],
        index=0,
        help="Empathize-only mentor: the application tracks state and decides the next missing question."
    )
    st.session_state.interaction_mode = interaction_mode

    st.divider()
    st.header("🛤️ Journey Status")
    st.markdown(f"**Current Phase:** `{st.session_state.dt_phase}`")

    if st.session_state.get("interaction_mode") == "Design Thinking Coach" and st.session_state.get("dt_phase") in ("Empathize", "Discovery", "empathize", "general"):
        st.subheader("📋 Empathize Checklist")
        # Fetch status from backend
        try:
            status_res = requests.post(f"{BACKEND_URL}/session/status", json={
                "username": "User",
                "project_name": st.session_state.get("project_name", "MyProject")
            })
            if status_res.status_code == 200:
                status_data = status_res.json()
                
                # Render Checklist Cards
                checklist = status_data.get("checklist", {})
                for key, item in checklist.items():
                    complete = item["complete"]
                    desc = item["description"].title()
                    val = item["value"]
                    
                    if complete:
                        badge = '<span class="badge-complete">✓ Complete</span>'
                        val_html = f'<div class="card-value">{val}</div>'
                    else:
                        badge = '<span class="badge-missing">✗ Missing</span>'
                        val_html = '<div class="card-value" style="color: #6b7280; font-style: normal;">Not gathered yet</div>'
                        
                    st.markdown(f"""
                    <div class="checklist-card">
                        {badge}
                        <div class="card-header">{desc}</div>
                        {val_html}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show extracted facts
                known_facts = status_data.get("known_facts", [])
                if known_facts:
                    st.subheader("💡 Extracted Facts")
                    for fact in known_facts:
                        st.markdown(f"- {fact}")
                        
                # Show assumptions
                assumptions = status_data.get("assumptions", [])
                if assumptions:
                    st.subheader("🔍 Assumptions")
                    for asm in assumptions:
                        st.markdown(f"- *{asm}*")
                        
                # Show unknown facts
                unknown_facts = status_data.get("unknown_facts", [])
                if unknown_facts:
                    st.subheader("❓ Unknown Facts")
                    for unk in unknown_facts:
                        st.markdown(f"- {unk}")
                        
                # Show open questions
                open_questions = status_data.get("open_questions", [])
                if open_questions:
                    st.subheader("💬 Open Questions")
                    for q in open_questions:
                        st.markdown(f"- {q}")
            else:
                st.warning("Could not sync mentor session status.")
        except Exception as e:
            st.warning("Backend offline / status unavailable.")

    st.caption("This version is intentionally locked to the Empathize phase.")
    if st.button("Refresh Empathize Status"):
        st.session_state.dt_phase = "Empathize"
        st.rerun()

    st.divider()
    st.header("🔍 Analysis Tools")
    col_a, col_b = st.columns(2)
    trigger_critique = col_a.button("🛡️ Critique")
    trigger_visualize = col_b.button("🗺️ Visualize")
    
    st.divider()
    st.header("✨ AI Power Tools")
    trigger_reqgpt = st.button("💎 Generate ISO Requirements")
    if trigger_reqgpt:
        with st.spinner("🚀 Calling specialized ReqGPT model..."):
            # Use the last user message or a generic prompt
            last_user_msg = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "Requirement: The system shall")
            specialized_req = call_backend_reqgpt(last_user_msg)
            st.session_state.messages.append({"role": "assistant", "content": f"**Specialized Requirement:**\n\n{specialized_req}"})
            st.rerun()

    st.divider()
    st.header("📄 Export")
    _export_messages = st.session_state.get("messages", [])
    _export_missions = existing_sessions if existing_sessions else None

    if st.button("📋 Copy Conversation", use_container_width=True):
        st.code(
            build_conversation_copy(_export_messages),
            language="markdown",
        )

    if st.button("📋 Copy Conversation + Diagnostics", use_container_width=True):
        st.code(
            build_conversation_diagnostics_copy(
                _export_messages, project_name=project_name
            ),
            language="markdown",
        )

    _export_markdown = build_full_markdown(
        _export_messages,
        project_name=project_name,
        timestamp=datetime.now().isoformat(),
        saved_missions=_export_missions,
    )
    st.download_button(
        label="⬇️ Export Markdown",
        data=_export_markdown,
        file_name=f"{project_name}_export.md",
        mime="text/markdown",
        use_container_width=True,
    )

    _export_json = serialize_session_json(
        _export_messages,
        project_name=project_name,
        dt_phase=st.session_state.get("dt_phase", "Empathize"),
        session_id=st.session_state.get("session_id", ""),
        timestamp=datetime.now().isoformat(),
        saved_missions=_export_missions,
        conversation_metrics=conversation_metrics_from_messages(_export_messages),
    )
    st.download_button(
        label="⬇️ Export JSON",
        data=_export_json,
        file_name=f"{project_name}_session.json",
        mime="application/json",
        use_container_width=True,
    )

    if "prd_content" not in st.session_state:
        st.session_state.prd_content = None
    if st.button("Prepare PRD"):
        st.session_state.prd_content = format_prd()
    if st.session_state.prd_content:
        st.download_button(
            label="Download .md PRD",
            data=st.session_state.prd_content,
            file_name=f"{project_name}_PRD.md",
            mime="text/markdown"
        )

    st.divider()
    st.header("🗄️ Knowledge Vault")
    uploaded_file = st.file_uploader("Upload Project Context", type=["txt", "md", "csv"])
    doc_context = ""
    if uploaded_file:
        try:
            doc_context = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            doc_context = uploaded_file.read().decode("utf-8", errors="replace")
            st.warning("File contained non-UTF-8 characters; some content may be garbled.")
        st.success(f"Document Ingested (Limited to ~2000 chars for memory safety)")

    st.divider()
    if st.button("🗑️ Clear Mission / Start Over"):
        try:
            requests.post(f"{BACKEND_URL}/reset", json={
                "username": "User",
                "project_name": st.session_state.get("project_name", "MyProject")
            })
        except Exception as e:
            pass
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ThinkingPods. Let's start fresh in Empathize. What project or idea should we explore?"}]
        st.session_state.dt_phase = "Empathize"
        st.rerun()

    st.divider()
    st.header("🎤 Voice Interface")
    audio_bytes = audio_recorder(text="Click to Speak", icon_size="2x", key="recorder")
    enable_voice = st.checkbox("Enable AI Voice Replies", key="enable_voice")

# --- Chat Column + Developer Console Dock ---
# The chat always uses the full width; the Developer Console opens as a
# centered modal overlay (backdrop + ESC/click-out to close) rendered after
# the chat. UI-only change — diagnostics and conversation are untouched.
chat_col = st.container()

with chat_col:
    # Display Chat History
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Detect any mermaid keywords in the message to trigger rendering
            if any(k in message["content"] for k in MERMAID_KEYWORDS):
                render_mermaid(message["content"])
            
            # Play and show audio if present
            if message["role"] == "assistant" and message.get("audio"):
                if enable_voice:
                    try:
                        audio_data = base64.b64decode(message["audio"])
                        autoplay_flag = not message.get("audio_played", False)
                        st.audio(audio_data, format="audio/wav", autoplay=autoplay_flag, key=f"audio_{idx}")
                        if autoplay_flag:
                            message["audio_played"] = True
                    except Exception as e:
                        st.error(f"Error playing audio: {e}")

if st.session_state.get("dev_console_open", False):
    _render_console_overlay()

# --- Process Specialized Tasks ---
if trigger_critique:
    with st.spinner("🔍 Auditing requirements..."):
        ai_reply, audio_reply, _, _ = call_backend_text("Critique the requirements for weak words.", st.session_state.dt_phase, doc_ctx=doc_context)
        audio_b64 = base64.b64encode(audio_reply).decode('utf-8') if audio_reply else None
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_reply,
            "audio": audio_b64,
            "audio_played": False
        })
        st.rerun()

if trigger_visualize:
    with st.chat_message("assistant"):
        with st.spinner("🗺️ Mapping logic..."):
            diagram_code = call_backend_visualize(username="User", doc_ctx=doc_context)
            # Add to history as a diagram block
            full_reply = f"Here is the system visualization:\n\n```mermaid\n{diagram_code}\n```"
            st.markdown(full_reply)
            render_mermaid(diagram_code)
            st.session_state.messages.append({"role": "assistant", "content": full_reply})

# --- Input Handlers ---
if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
    with st.spinner("👂 Hearing and thinking..."):
        transcript, ai_reply, audio_reply, _timing, _diag = call_backend_voice(audio_bytes, st.session_state.get("dt_phase", "Discovery"), doc_ctx=doc_context)
        if ai_reply and ai_reply.startswith("⚠️"):
            st.session_state.last_audio_bytes = audio_bytes
            st.error(ai_reply)
        elif transcript or ai_reply:
            st.session_state.last_audio_bytes = audio_bytes
            user_msg = transcript if transcript else "🎤 *[Silence / Mic Check]*"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            audio_b64 = base64.b64encode(audio_reply).decode('utf-8') if audio_reply else None
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_reply,
                "audio": audio_b64,
                "audio_played": False,
                "timing": _timing,
                "diagnostics": _diag
            })
            st.rerun()
        else:
            st.session_state.last_audio_bytes = audio_bytes
            st.warning("I couldn't quite hear that. Could you try again? 🎤")

user_input = st.chat_input("Type your reply here...")

# --- Process Input ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            ai_reply, audio_reply, _timing, _diag = call_backend_text(user_input, st.session_state.get("dt_phase", "Discovery"), doc_ctx=doc_context)
            st.markdown(ai_reply)
            audio_b64 = base64.b64encode(audio_reply).decode('utf-8') if audio_reply else None
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_reply,
                "audio": audio_b64,
                "audio_played": False,
                "timing": _timing,
                "diagnostics": _diag
            })
            st.rerun()
