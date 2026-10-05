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
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def get_backend_url():
    """Retrieve active backend URL from session state or environment."""
    val = st.session_state.get("backend_url_input") if hasattr(st, "session_state") else None
    if val and val.strip():
        return val.strip().rstrip("/")
    return os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")

BACKEND_URL = DEFAULT_BACKEND_URL

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
    /* ============================================================
     * Design system — Catppuccin Mocha tokens + semantic aliases.
     * One palette, used everywhere.
     * ============================================================ */
    :root {
      --ctp-rosewater:#f5e0dc; --ctp-flamingo:#f2cdcd; --ctp-pink:#f5c2e7;
      --ctp-mauve:#cba6f7;    --ctp-red:#f38ba8;     --ctp-maroon:#eba0ac;
      --ctp-peach:#fab387;    --ctp-yellow:#f9e2af;  --ctp-green:#a6e3a1;
      --ctp-teal:#94e2d5;     --ctp-sky:#89dceb;     --ctp-sapphire:#74c7ec;
      --ctp-blue:#89b4fa;     --ctp-lavender:#b4befe;
      --ctp-text:#cdd6f4;     --ctp-subtext1:#bac2de;--ctp-subtext0:#a6adc8;
      --ctp-overlay2:#9399b2; --ctp-overlay1:#7f849c;--ctp-overlay0:#6c7086;
      --ctp-surface2:#585b70; --ctp-surface1:#45475a;--ctp-surface0:#313244;
      --ctp-base:#1e1e2e;     --ctp-mantle:#181825;  --ctp-crust:#11111b;

      --bg-app:       #1e1e2e;
      --bg-elevated:  #181825;
      --bg-surface:   #313244;
      --bg-surface-2: #45475a;
      --border:        rgba(186,194,222,0.08);
      --border-strong:rgba(186,194,222,0.18);
      --text:         #cdd6f4;
      --text-muted:   #bac2de;
      --text-dim:     #9399b2;
      --text-faint:   #6c7086;
      --accent:       #89b4fa;
      --accent-soft:  rgba(137,180,250,0.14);
      --accent-line:  rgba(137,180,250,0.35);
      --accent-2:     #cba6f7;
      --success:      #a6e3a1;
      --warning:      #f9e2af;
      --danger:       #f38ba8;
      --info:         #89dceb;
      --teal:         #94e2d5;

      --r-sm: 6px; --r: 10px; --r-lg: 14px; --r-xl: 18px;
      --sh-sm: 0 1px 2px rgba(0,0,0,0.20);
      --sh:    0 6px 18px rgba(0,0,0,0.28);
      --sh-lg: 0 22px 56px rgba(0,0,0,0.52);
    }

    /* ----------------------------------------------------------
     * Global typography, motion, and streamlit chrome.
     * ---------------------------------------------------------- */
    html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea {
      font-family: 'Outfit','Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif !important;
    }
    html, body {
      color: var(--text) !important;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    #MainMenu, [data-testid="stToolbar"], footer { visibility: hidden; height: 0; }
    [data-testid="stDecoration"] { display: none; }
    ::selection { background: var(--accent-soft); color: var(--text); }

    /* ----------------------------------------------------------
     * App background: solid base + a single soft accent glow.
     * ---------------------------------------------------------- */
    .stApp {
      background:
        radial-gradient(900px 600px at 100% -10%, rgba(137,180,250,0.06), transparent 60%),
        var(--bg-app) !important;
    }

    /* ----------------------------------------------------------
     * Main column: center the chat column for readability.
     * ---------------------------------------------------------- */
    .block-container,
    [data-testid="stMain"] > .block-container {
      max-width: 880px;
      padding-top: 1.6rem;
      padding-bottom: 8rem;
      padding-left: 1.6rem;
      padding-right: 1.6rem;
    }
    @media (min-width: 1400px) {
      .block-container { max-width: 940px; }
    }

    /* ----------------------------------------------------------
     * Sidebar.
     * ---------------------------------------------------------- */
    section[data-testid="stSidebar"] {
      background: var(--bg-elevated) !important;
      border-right: 1px solid var(--border) !important;
      padding: 0.9rem 0.85rem 2rem 0.85rem;
    }
    section[data-testid="stSidebar"] > div { gap: 0; }
    /* Section headers: tracked uppercase, muted. */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] [data-testid="stHeader"] {
      font-size: 0.7rem !important;
      font-weight: 700 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.10em !important;
      color: var(--text-dim) !important;
      margin: 1.2rem 0 0.55rem 0 !important;
      padding: 0 0 0.35rem 0 !important;
      border-bottom: 1px solid var(--border) !important;
      line-height: 1.2 !important;
    }
    section[data-testid="stSidebar"] > div > div:first-child h2,
    section[data-testid="stSidebar"] > div > div:first-child [data-testid="stHeader"] {
      margin-top: 0 !important;
    }
    section[data-testid="stSidebar"] hr {
      margin: 0.9rem 0 !important;
      border-color: var(--border) !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
      color: var(--text-faint) !important;
    }
    section[data-testid="stSidebar"] code {
      background: var(--bg-surface) !important;
      color: var(--accent) !important;
      border-radius: var(--r-sm) !important;
      padding: 0.05rem 0.35rem !important;
      border: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] iframe {
      border-radius: var(--r) !important;
      border: 1px solid var(--border) !important;
      background: var(--bg-elevated) !important;
    }

    /* ----------------------------------------------------------
     * Top header (title + Console + menu).
     * ---------------------------------------------------------- */
    h1 {
      font-weight: 700 !important;
      letter-spacing: -0.012em !important;
      font-size: 1.55rem !important;
      line-height: 1.2 !important;
      color: var(--text) !important;
      margin: 0 !important;
      padding: 0 !important;
    }
    [data-testid="stMain"] > .block-container > div:first-child {
      padding-bottom: 0.9rem !important;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.2rem !important;
    }
    /* Drop the old margin-top hacks on header buttons. */
    .st-key-dev_console_toggle,
    .st-key-dc_header_menu { margin-top: 0 !important; }
    .st-key-dev_console_toggle button {
      width: 100%;
      background: var(--accent-soft) !important;
      border: 1px solid var(--accent-line) !important;
      color: var(--text) !important;
      border-radius: var(--r) !important;
      font-weight: 500 !important;
      font-size: 0.82rem !important;
      padding: 0.45rem 0.55rem !important;
    }
    .st-key-dev_console_toggle button:hover {
      background: var(--accent-line) !important;
      transform: translateY(-1px);
    }
    .st-key-dc_header_menu button {
      width: 100%;
      background: var(--bg-elevated) !important;
      border: 1px solid var(--border) !important;
      color: var(--text-muted) !important;
      border-radius: var(--r) !important;
      font-size: 0.95rem !important;
      padding: 0.35rem 0.55rem !important;
      line-height: 1 !important;
    }
    .st-key-dc_header_menu button:hover {
      background: var(--bg-surface) !important;
      color: var(--text) !important;
      border-color: var(--border-strong) !important;
    }

    /* ----------------------------------------------------------
     * Buttons (sidebar + body).
     * ---------------------------------------------------------- */
    .stButton > button,
    [data-testid="baseButton"] {
      background: var(--bg-elevated) !important;
      color: var(--text) !important;
      border: 1px solid var(--border) !important;
      border-radius: var(--r) !important;
      font-weight: 500 !important;
      transition: background .15s ease, border-color .15s ease, transform .12s ease;
      box-shadow: none !important;
    }
    .stButton > button:hover,
    [data-testid="baseButton"]:hover {
      background: var(--bg-surface) !important;
      border-color: var(--border-strong) !important;
      transform: translateY(-1px);
    }
    .stButton > button:focus-visible {
      outline: 2px solid var(--accent) !important;
      outline-offset: 2px !important;
    }
    .stButton > button[kind="primary"],
    [data-testid="baseButton"][kind="primary"] {
      background: var(--accent-soft) !important;
      border-color: var(--accent-line) !important;
      color: var(--text) !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton"][kind="primary"]:hover {
      background: var(--accent-line) !important;
    }
    .stDownloadButton > button {
      width: 100%;
    }

    /* ----------------------------------------------------------
     * Inputs: text_input / textarea / chat_input / selectbox / file_uploader.
     * ---------------------------------------------------------- */
    .stTextInput input,
    .stTextArea textarea,
    .stChatInput textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
      background: var(--bg-elevated) !important;
      color: var(--text) !important;
      border: 1px solid var(--border) !important;
      border-radius: var(--r) !important;
      caret-color: var(--accent) !important;
      transition: border-color .15s ease, box-shadow .15s ease;
    }
    .stTextInput input:focus,
    .stChatInput textarea:focus,
    [data-baseweb="input"] input:focus,
    [data-baseweb="textarea"] textarea:focus {
      border-color: var(--accent) !important;
      box-shadow: 0 0 0 3px var(--accent-soft) !important;
      outline: none !important;
    }
    [data-baseweb="select"] > div {
      background: var(--bg-elevated) !important;
      border-color: var(--border) !important;
      border-radius: var(--r) !important;
      color: var(--text) !important;
    }
    [data-baseweb="select"] > div:hover {
      border-color: var(--border-strong) !important;
    }
    [data-baseweb="select"] > div:focus,
    [data-baseweb="select"] > div:focus-within {
      border-color: var(--accent) !important;
      box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }
    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
      background: var(--bg-elevated) !important;
      border: 1px dashed var(--border-strong) !important;
      border-radius: var(--r) !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
      border-color: var(--accent) !important;
    }
    /* Popover menu container */
    [data-testid="stPopover"] > div:first-child {
      background: var(--bg-elevated) !important;
      border: 1px solid var(--border) !important;
      border-radius: var(--r) !important;
      color: var(--text) !important;
    }
    [data-testid="stPopoverContent"],
    [data-baseweb="popover"] {
      background: var(--bg-elevated) !important;
      border: 1px solid var(--border) !important;
      border-radius: var(--r) !important;
      color: var(--text) !important;
      box-shadow: var(--sh) !important;
    }

    /* ----------------------------------------------------------
     * Chat column: bubbles, anchored footer, comfortable rhythm.
     * ---------------------------------------------------------- */
    [data-testid="stChatMessage"] {
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      padding: 0.85rem 1rem;
      margin: 0.65rem 0;
      box-shadow: var(--sh-sm);
      transition: border-color .15s ease, background .15s ease;
    }
    [data-testid="stChatMessage"]:hover {
      border-color: var(--border-strong);
    }
    [data-testid="stChatMessage"][data-testid="chat-message-user"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
      background: linear-gradient(180deg, rgba(137,180,250,0.10), rgba(137,180,250,0.04));
      border-color: var(--accent-line);
      border-left: 3px solid var(--accent);
    }
    [data-testid="stChatMessage"] [data-testid="chatAvatar"] {
      background: var(--bg-surface) !important;
      border: 1px solid var(--border);
    }
    [data-testid="stChatMessage"] p { line-height: 1.6; }
    [data-testid="stChatMessage"] p:first-child { margin-top: 0; }
    [data-testid="stChatMessage"] p:last-child  { margin-bottom: 0; }
    /* Chat input: anchored, glow on focus */
    [data-testid="stChatInput"] {
      background: transparent;
      padding-bottom: 0.5rem;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInputTextArea"] {
      background: var(--bg-elevated) !important;
      border: 1px solid var(--border) !important;
      border-radius: var(--r-xl) !important;
      box-shadow: var(--sh);
      transition: border-color .15s ease, box-shadow .15s ease;
    }
    [data-testid="stChatInput"]:focus-within > div,
    [data-testid="stChatInputTextArea"]:focus {
      border-color: var(--accent) !important;
      box-shadow: 0 0 0 4px var(--accent-soft), var(--sh) !important;
    }

    /* ----------------------------------------------------------
     * Checklist cards (replaces the float-right banner grid).
     * ---------------------------------------------------------- */
    .checklist-card {
      display: flex;
      align-items: flex-start;
      gap: 0.7rem;
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      border-radius: var(--r);
      padding: 0.7rem 0.8rem;
      margin: 0 0 0.5rem 0;
      transition: background .15s ease, border-color .15s ease, transform .12s ease;
    }
    .checklist-card:hover {
      border-color: var(--border-strong);
      transform: translateY(-1px);
    }
    .checklist-card.complete {
      border-left: 3px solid var(--success);
    }
    .checklist-card.missing {
      border-left: 3px solid var(--warning);
    }
    .checklist-status {
      flex: 0 0 auto;
      width: 9px; height: 9px;
      margin-top: 0.40rem;
      border-radius: 50%;
    }
    .checklist-card.complete .checklist-status {
      background: var(--success);
      box-shadow: 0 0 0 3px rgba(166,227,161,0.18);
    }
    .checklist-card.missing .checklist-status {
      background: var(--warning);
      box-shadow: 0 0 0 3px rgba(249,226,175,0.18);
    }
    .checklist-body { min-width: 0; flex: 1 1 auto; }
    .checklist-header {
      font-weight: 600;
      color: var(--text);
      font-size: 0.9rem;
      line-height: 1.3;
    }
    .checklist-value {
      color: var(--text-muted);
      font-size: 0.82rem;
      margin-top: 0.15rem;
      line-height: 1.4;
      word-break: break-word;
    }
    .checklist-card.missing .checklist-value {
      color: var(--text-faint);
      font-style: italic;
    }

    /* ----------------------------------------------------------
     * Code, dividers, success/warning, alerts.
     * ---------------------------------------------------------- */
    .stCode, code, [data-testid="stCode"] {
      background: var(--bg-elevated) !important;
      border: 1px solid var(--border) !important;
      border-radius: var(--r) !important;
      color: var(--text) !important;
    }
    .stAlert {
      border-radius: var(--r) !important;
      border: 1px solid var(--border) !important;
      background: var(--bg-elevated) !important;
    }
    hr { border-color: var(--border) !important; opacity: 1; }
    .stSpinner > div { border-top-color: var(--accent) !important; }
    .stMarkdown a { color: var(--accent) !important; text-decoration: none; border-bottom: 1px dashed var(--accent-line); }
    .stMarkdown a:hover { color: var(--text) !important; border-bottom-color: var(--accent); }

    /* ----------------------------------------------------------
     * Scrollbar.
     * ---------------------------------------------------------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
      background: var(--bg-surface-2);
      border-radius: 10px;
      border: 2px solid var(--bg-app);
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

    /* ===========================================================
     * Developer Console — VS Code-style IDE inspector (modal).
     * Preserved layout from the original feature; only the accent
     * palette is harmonized with the design system.
     * =========================================================== */
    @keyframes dc-pop-in {
      from { transform: scale(0.98) translateY(8px); opacity: 0; }
        to   { transform: scale(1) translateY(0); opacity: 1; }
    }
    .dc-modal-backdrop {
      position: fixed; inset: 0; z-index: 999;
      background: rgba(17,17,27,0.62);
      backdrop-filter: blur(3px);
    }
    .st-key-dc_modal {
      position: fixed; top: 7.5vh; left: 10vw;
      width: 80vw; height: 85vh; z-index: 1000;
      background: #1e1e2e;
      border: 1px solid var(--border-strong);
      border-radius: 14px;
      box-shadow: var(--sh-lg);
      animation: dc-pop-in .16s ease-out;
      overflow: hidden;
      padding: 0.6rem 0.8rem 0.7rem 0.8rem;
    }
    .st-key-dc_modal .dc-header {
      font-size: 0.95rem; font-weight: 700;
      color: var(--text); letter-spacing: 0.02em;
      display: flex; align-items: baseline; gap: 0.6rem;
      padding: 0.1rem 0 0.35rem 0;
      border-bottom: 1px solid var(--border);
    }
    .st-key-dc_modal .dc-legend {
      font-size: 0.62rem; font-weight: 500;
      color: var(--text-faint); letter-spacing: 0.04em;
    }
    .st-key-dc_modal_close button {
      font-size: 0.8rem; padding: 0.2rem 0.55rem;
      border-radius: 6px;
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border-strong);
      color: var(--text-muted);
    }
    .st-key-dc_modal_close button:hover {
      background: rgba(243,139,168,0.2);
      border-color: rgba(243,139,168,0.5);
      color: #eba0ac;
    }
    .st-key-dc_ide_turn { margin: 0.4rem 0 0.25rem 0; }
    .st-key-dc_ide_turn [data-baseweb="select"] > div { border-color: var(--border-strong); }
    .st-key-dc_ide_search { margin: 0.15rem 0 0.25rem 0; }
    .st-key-dc_ide_search input {
      font-size: 0.78rem; border-radius: 6px; border-color: var(--border-strong);
    }
    .st-key-dc_ide_view { margin: 0.1rem 0 0.4rem 0; }
    .st-key-dc_ide_view > div { display: flex; }
    .st-key-dc_ide_view label {
      font-size: 0.74rem; padding: 0.16rem 0.8rem;
      border: 1px solid var(--border-strong); border-radius: 6px;
      margin-right: 0.25rem; background: rgba(255,255,255,0.02); white-space: nowrap;
    }
    .st-key-dc_ide_view label:has(input:checked) {
      background: rgba(137,180,250,0.18);
      border-color: rgba(137,180,250,0.55);
      color: #ffffff;
    }
    .dc-nav-empty { font-size: 0.72rem; color: var(--text-faint); padding: 0.4rem 0.25rem; font-style: italic; }
    .dc-empty {
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 0.35rem; min-height: calc(85vh - 12rem); text-align: center; padding: 2rem 1rem;
    }
    .dc-empty-icon { font-size: 2.4rem; opacity: 0.55; }
    .dc-empty-title { font-size: 1.02rem; font-weight: 700; color: #e2e8f0; }
    .dc-empty-sub { font-size: 0.82rem; color: var(--text-faint); max-width: 34rem; line-height: 1.5; }
    .dc-nav-row {
      display: flex; align-items: center; gap: 0.4rem; width: 100%;
      text-align: left; background: transparent; border: none;
      border-left: 2px solid transparent; border-radius: 4px;
      padding: 0.22rem 0.3rem; margin: 1px 0;
      cursor: pointer; font-family: inherit; color: var(--text-muted);
    }
    .dc-nav-row:hover { background: rgba(255,255,255,0.05); }
    .dc-nav-row.active {
      background: rgba(137,180,250,0.12);
      border-left-color: var(--accent);
      color: var(--text);
    }
    .dc-nav-icon { font-size: 0.8rem; width: 1rem; text-align: center; flex: 0 0 auto; }
    .dc-nav-body { min-width: 0; flex: 1 1 auto; }
    .dc-nav-name {
      font-size: 0.76rem; font-weight: 600;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .dc-nav-sub { font-size: 0.64rem; color: var(--text-faint); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .dc-nav-reason {
      flex: 0 0 auto; font-size: 0.6rem; color: var(--text-dim);
      max-width: 9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .dc-exp-row {
      display: flex; align-items: center; gap: 0.35rem; width: 100%;
      text-align: left; background: transparent; border: none;
      border-left: 2px solid transparent; border-radius: 4px;
      padding: 0.24rem 0.3rem; margin: 2px 0 1px 0;
      cursor: pointer; font-family: inherit; color: var(--text);
    }
    .dc-exp-row:hover { background: rgba(255,255,255,0.06); }
    .dc-exp-row.active {
      background: rgba(137,180,250,0.10);
      border-left-color: var(--accent);
      color: var(--text);
    }
    .dc-exp-chev { flex: 0 0 auto; width: 0.7rem; text-align: center; font-size: 0.6rem; color: var(--accent-2); }
    .dc-exp-icon { flex: 0 0 auto; width: 1rem; text-align: center; font-size: 0.8rem; }
    .dc-exp-body { min-width: 0; flex: 1 1 auto; }
    .dc-exp-label {
      display: block; font-size: 0.78rem; font-weight: 700;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .dc-exp-sub {
      display: block; font-size: 0.63rem; color: var(--text-faint);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .dc-nav-row.dc-exp-child { padding-left: 1.15rem; }
    .dc-tree-item { outline: none; }
    .st-key-dc_ide_nav { max-height: calc(85vh - 11rem); overflow-y: auto; }
    [class*="st-key-dc_navbtn_"] { display: none; }
    [class*="st-key-dc_turnbtn_"] { display: none; }
    .st-key-dc_ide_pane {
      height: 100%; overflow: hidden;
      padding: 0 0 0 0.75rem;
      border-left: 1px solid var(--border);
    }
    .dc-ide-tab {
      display: flex; align-items: center; gap: 0.4rem;
      font-size: 0.74rem; font-weight: 700; color: var(--text);
      padding: 0.35rem 0.5rem; margin-bottom: 0.4rem;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 6px 6px 0 0; border-bottom: 2px solid var(--accent);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .dc-ide-tab-status { margin-left: auto; font-size: 0.7rem; }
    .dc-ide-pane-scroll {
      max-height: calc(85vh - 11rem); overflow: auto; padding-right: 0.3rem;
    }
    .dc-ide-body {
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.74rem; line-height: 1.5; color: var(--text-muted);
      white-space: pre-wrap; word-break: break-word;
      background: rgba(255,255,255,0.015);
      border: 1px solid var(--border);
      border-radius: 6px; padding: 0.45rem 0.5rem; margin: 0.2rem 0;
    }
    .dc-prompt-viewer, .dc-prompt-editor {
      border: 1px solid var(--border-strong); border-radius: 6px;
      background: rgba(255,255,255,0.015); margin: 0.2rem 0; overflow: hidden;
    }
    .dc-pv-toolbar {
      display: flex; align-items: center; gap: 0.6rem;
      padding: 0.3rem 0.5rem;
      background: rgba(255,255,255,0.03);
      border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .dc-pv-title, .dc-pv-tab {
      font-size: 0.66rem; font-weight: 700; letter-spacing: 0.1em;
      text-transform: uppercase; color: var(--accent-2);
    }
    .dc-pv-stats {
      flex: 1 1 auto; font-size: 0.62rem; color: var(--text-faint);
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .dc-pv-copy {
      font-size: 0.66rem; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      color: var(--text-muted); background: rgba(137,180,250,0.14);
      border: 1px solid rgba(137,180,250,0.4); border-radius: 4px;
      padding: 0.12rem 0.5rem; cursor: pointer; white-space: nowrap;
    }
    .dc-pv-copy:hover { background: rgba(137,180,250,0.28); }
    .dc-pv-body {
      display: grid; grid-template-columns: max-content auto;
      width: max-content; min-width: 100%;
      max-height: calc(85vh - 13.5rem); overflow: auto;
    }
    .dc-pv-gutter {
      position: sticky; left: 0; z-index: 1;
      background: rgba(15,20,30,0.98);
      border-right: 1px solid var(--border);
      text-align: right; user-select: none;
    }
    .dc-pv-gutter pre {
      margin: 0; padding: 0.4rem 0.5rem 0.4rem 0;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.72rem; line-height: 1.5; color: #475569; background: transparent;
    }
    .dc-pv-code {
      margin: 0; padding: 0.4rem 0.5rem;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.72rem; line-height: 1.5; color: var(--text-muted);
      white-space: pre; overflow: visible; background: transparent;
    }
    .dc-pv-code .pc-h { display: block; color: #93c5fd; font-weight: 700; }
    .dc-pv-code .pc-hline { display: block; color: #334155; font-weight: 700; }
    .dc-pv-code .pc-list { color: #a5b4fc; }
    .dc-pv-code .pc-tokline { color: #fde68a; }
    .dc-pv-code .pc-tok { color: #fbbf24; font-weight: 700; }
    .dc-pv-code .pc-list .pc-tok { color: #fbbf24; }
    .dc-highlight {
      white-space: pre-wrap; word-break: break-word;
      font-size: 0.74rem; line-height: 1.5; color: var(--text-muted);
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border);
      border-radius: 4px; padding: 0.45rem 0.5rem; margin: 0.2rem 0;
    }
    mark.dc-search-hit {
      background: rgba(249,226,175,0.25); color: var(--warning);
      border-radius: 2px; padding: 0 1px;
    }
    .dc-search-caption {
      font-size: 0.64rem; color: var(--text-faint); padding: 0.15rem 0.25rem;
    }
    .dc-tl-flow {
      display: flex; flex-wrap: nowrap; gap: 0.4rem;
      overflow-x: auto; overflow-y: hidden; padding: 0.3rem 0.1rem 0.6rem 0.1rem;
    }
    .dc-tl-card {
      flex: 0 0 230px; min-width: 230px; max-width: 260px;
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border-strong); border-top: 3px solid var(--text-faint);
      border-radius: 8px; padding: 0.45rem 0.6rem;
    }
    .dc-tl-card.healthy { border-top-color: var(--success); }
    .dc-tl-card.warning { border-top-color: var(--warning); }
    .dc-tl-card.problem { border-top-color: var(--danger); }
    .dc-tl-arrow { flex: 0 0 auto; align-self: center; color: var(--text-faint); font-size: 1rem; }
    .dc-tl-head { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }
    .dc-tl-icon { font-size: 0.95rem; line-height: 1; }
    .dc-tl-name { font-weight: 600; color: var(--text); font-size: 0.82rem; }
    .dc-tl-ms { color: var(--text-dim); font-size: 0.7rem; margin-left: auto; font-variant-numeric: tabular-nums; }
    .dc-tl-flag { font-size: 0.8rem; }
    .dc-tl-reason { color: var(--warning); font-size: 0.7rem; margin-top: 0.2rem; }
    .dc-tl-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.3rem; }
    .dc-tl-chip {
      font-size: 0.64rem; color: var(--text-muted);
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border-strong); border-radius: 4px; padding: 0.04rem 0.3rem;
    }
    .dc-tl-chip b { color: var(--text); font-weight: 600; }
    .dc-tl-detail { margin-top: 0.35rem; }
    .dc-tl-detail summary {
      cursor: pointer; font-size: 0.68rem;
      color: var(--accent); user-select: none;
    }
    .dc-tl-detail summary:hover { color: var(--text); }
    .dc-tl-pre {
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
      font-size: 0.66rem; line-height: 1.4; white-space: pre-wrap;
      word-break: break-word; background: rgba(0,0,0,0.35);
      border: 1px solid var(--border); border-radius: 6px;
      padding: 0.35rem; margin: 0.35rem 0 0 0;
      max-height: 180px; overflow-y: auto; color: var(--text-muted);
    }
    .dc-menu-title {
      font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.08em; color: var(--text-faint); margin-bottom: 0.25rem;
    }
    @media (max-width: 1199px) {
      .st-key-dc_modal {
        top: 4vh; left: 4vw; width: 92vw; height: 92vh;
      }
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
        start_resp = requests.post(f"{get_backend_url()}/session/start", json={
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
        response = requests.post(f"{get_backend_url()}/text", json=payload)

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
        response = requests.post(f"{get_backend_url()}/voice", data=audio_bytes, headers=headers)

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
        response = requests.post(f"{get_backend_url()}/generate_requirements", json=payload)
        if response.status_code == 200:
            return response.json().get("requirement", "")
        else:
            return f"⚠️ Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"⚠️ Backend Unreachable: {e}"

def call_backend_visualize(username="User", doc_ctx=""):
    try:
        payload = {"username": username, "context_doc": doc_ctx}
        response = requests.post(f"{get_backend_url()}/visualize", json=payload)
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

    backend_url = get_backend_url()
    html_code = f"""
    <div id="mermaid-container" style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <pre class="mermaid" style="display: flex; justify-content: center;">
            {code}
        </pre>
    </div>
    <script src="{backend_url}/static/mermaid.min.js"></script>
    <script type="module">
        let m = window.mermaid;
        if (!m) {{
            try {{
                const mod = await import('https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs');
                m = mod.default;
            }} catch(e) {{
                console.error("Mermaid load failed:", e);
            }}
        }}
        if (m) {{
            m.initialize({{ 
                startOnLoad: true, 
                theme: 'base',
                themeVariables: {{
                    'primaryColor': '#007bff',
                    'edgeColor': '#555555'
                }},
                securityLevel: 'loose'
            }});
            setTimeout(() => {{
                m.contentLoaded();
            }}, 300);
        }}
    </script>
    """
    # Dynamic height based on lines of code (rough estimate)
    lines = code.count("\n")
    calc_height = max(400, (lines + 2) * 25)
    st.components.v1.html(html_code, height=calc_height, scrolling=True)

def format_prd():
    project_name = st.session_state.get("project_name", "MyProject")
    phase = st.session_state.get("dt_phase", "Empathize")
    doc = f"# Product Requirements Document (PRD): {project_name}\n\n"
    doc += f"**Phase:** {phase}  \n"
    doc += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
    doc += f"**Tool:** ThinkingPods Offline AI Design Thinking Assistant\n\n"
    doc += "---\n\n"

    # Fetch structured state from backend
    checklist = {}
    known_facts = []
    assumptions = []
    open_questions = []
    try:
        res = requests.post(f"{get_backend_url()}/session/status", json={
            "username": "User",
            "project_name": project_name
        }, timeout=3)
        if res.status_code == 200:
            data = res.json()
            checklist = data.get("checklist", {})
            known_facts = data.get("known_facts", [])
            assumptions = data.get("assumptions", [])
            open_questions = data.get("open_questions", [])
    except Exception as e:
        doc += f"> *Note: Structured state fetch degraded ({e})*\n\n"

    # Executive Summary
    doc += "## 1. Executive Summary\n"
    doc += (
        f"This document defines the user discovery, problem formulation, and foundational requirements "
        f"for **{project_name}** gathered during the Design Thinking Empathize phase.\n\n"
    )

    # Core Discoveries Table / Sections
    doc += "## 2. Foundational Discoveries\n\n"
    def _val(key, default="*Not yet specified*"):
        item = checklist.get(key, {})
        v = item.get("value")
        return v if (item.get("complete") and v) else default

    doc += f"### 2.1 Target Audience (Personas)\n{_val('target_audience')}\n\n"
    doc += f"### 2.2 Core Problem Statement\n{_val('pain_point')}\n\n"
    doc += f"### 2.3 User Motivation & Underlying Driver\n{_val('motivation')}\n\n"
    doc += f"### 2.4 Current Workarounds & Existing Tools\n{_val('existing_solution')}\n\n"
    doc += f"### 2.5 Cadence & Urgency (Frequency)\n{_val('frequency')}\n\n"
    doc += f"### 2.6 Real-World Observations & Evidence\n{_val('evidence')}\n\n"

    if known_facts:
        doc += "## 3. Verified Project Insights\n"
        for fact in known_facts:
            doc += f"- {fact}\n"
        doc += "\n"

    if assumptions:
        doc += "## 4. Key Assumptions to Validate\n"
        for asm in assumptions:
            doc += f"- [ ] {asm}\n"
        doc += "\n"

    if open_questions:
        doc += "## 5. Open Design Questions\n"
        for q in open_questions:
            doc += f"- ❓ {q}\n"
        doc += "\n"

    # Full Transcript
    doc += "## 6. Complete Interview & Mentoring Transcript\n\n"
    for m in st.session_state.messages:
        role = "🤖 Coach" if m["role"] == "assistant" else "👤 User"
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
            resp = requests.post(f"{get_backend_url()}/session/new", json={
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

    # Cloud / Backend Connection setting for Streamlit Community Cloud
    with st.expander("🌐 Cloud / Backend Connection", expanded=False):
        current_backend = get_backend_url()
        user_backend = st.text_input(
            "Backend API URL:",
            value=st.session_state.get("backend_url_input", current_backend),
            help="For Streamlit Community Cloud, enter your public backend URL (e.g. from Cloudflare Tunnel or Colab)."
        )
        if user_backend != st.session_state.get("backend_url_input"):
            st.session_state.backend_url_input = user_backend.strip().rstrip("/")
            st.rerun()

        # Quick live health probe
        active_url = get_backend_url()
        try:
            h_probe = requests.get(f"{active_url}/health", timeout=3)
            if h_probe.status_code == 200:
                st.caption(f"🟢 **Backend Connected:** `{active_url}`")
            else:
                st.caption(f"🟡 **Backend Status {h_probe.status_code}:** `{active_url}`")
        except Exception:
            st.caption(f"🔴 **Backend Unreachable:** `{active_url}`")
            st.info("💡 **On Streamlit Cloud?** Run `run_public_tunnel.bat` or Google Colab to get a free public HTTPS URL, then paste it above.")

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
            status_res = requests.post(f"{get_backend_url()}/session/status", json={
                "username": "User",
                "project_name": st.session_state.get("project_name", "MyProject")
            })
            if status_res.status_code == 200:
                status_data = status_res.json()
                st.session_state.latest_checklist = status_data.get("checklist", {})
                
                # Render Checklist Cards
                checklist = status_data.get("checklist", {})
                for key, item in checklist.items():
                    complete = item["complete"]
                    desc = item["description"].title()
                    val = item["value"]
                    
                    if complete:
                        state_cls = "complete"
                        val_display = val
                    else:
                        state_cls = "missing"
                        val_display = "Not gathered yet"
                    
                    st.markdown(f"""
                    <div class="checklist-card {state_cls}">
                        <div class="checklist-status"></div>
                        <div class="checklist-body">
                            <div class="checklist-header">{desc}</div>
                            <div class="checklist-value">{val_display}</div>
                        </div>
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
            requests.post(f"{get_backend_url()}/reset", json={
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
    # Live Top Progress Bar & PRD Action (active during chat)
    if st.session_state.messages:
        chk = st.session_state.get("latest_checklist", {})
        if chk:
            completed_count = sum(1 for item in chk.values() if item.get("complete"))
            total_count = max(len(chk), 6)
            pct = min(1.0, completed_count / total_count)

            prog_col, prd_col = st.columns([0.76, 0.24], vertical_alignment="center")
            with prog_col:
                st.progress(pct, text=f"🎯 **Empathize Phase:** {completed_count}/{total_count} Discoveries Completed ({int(pct*100)}%)")
            with prd_col:
                if st.button("📄 Preview PRD", key="top_preview_prd", use_container_width=True):
                    st.session_state.prd_content = format_prd()
                    st.rerun()

            if completed_count >= total_count:
                st.success("🎉 **Empathize Phase Complete!** All foundational insights discovered. Click **Preview PRD** to inspect your requirements.")

    # Live In-App PRD Previewer Modal/Expander
    if st.session_state.get("prd_content"):
        with st.expander("📄 **Product Requirements Document (PRD) — Live Preview**", expanded=True):
            st.markdown(st.session_state.prd_content)
            dl_c1, dl_c2 = st.columns([0.75, 0.25])
            with dl_c1:
                st.download_button(
                    label="⬇️ Download PRD (.md)",
                    data=st.session_state.prd_content,
                    file_name=f"{st.session_state.get('project_name', 'MyProject')}_PRD.md",
                    mime="text/markdown",
                    key="inline_prd_dl",
                    use_container_width=True,
                )
            with dl_c2:
                if st.button("✕ Close Preview", key="close_inline_prd", use_container_width=True):
                    st.session_state.prd_content = None
                    st.rerun()
        st.divider()

    # Display Welcome Screen with Quick-Start Scenario Cards if no messages yet
    if not st.session_state.messages:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8)); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 22px; margin-bottom: 20px;">
            <h2 style="margin-top:0; font-size: 1.4rem; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                💡 Welcome to ThinkingPods
            </h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 8px;">
                Your local, privacy-first AI Design Thinking mentor. Start by choosing a problem scenario below, or type your own project idea:
            </p>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("👵 **Elderly Medication**\n\n*Help seniors remember prescriptions on time*", use_container_width=True, key="starter_elderly"):
                st.session_state.starter_input = "I want to help elderly people take medications on time"
                st.rerun()
            if st.button("🥦 **Grocery Waste**\n\n*Help families avoid wasting fresh food every week*", use_container_width=True, key="starter_grocery"):
                st.session_state.starter_input = "I want to help busy families reduce grocery waste"
                st.rerun()
        with sc2:
            if st.button("🎓 **Student Deadlines**\n\n*Help college students track assignment deadlines*", use_container_width=True, key="starter_students"):
                st.session_state.starter_input = "I want to help college students track assignment deadlines"
                st.rerun()
            if st.button("💼 **Freelance Invoices**\n\n*Help freelancers get paid without awkward follow-ups*", use_container_width=True, key="starter_freelance"):
                st.session_state.starter_input = "I want to help freelancers get invoices paid on time"
                st.rerun()

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
if not user_input and st.session_state.get("starter_input"):
    user_input = st.session_state.pop("starter_input")

# --- Process Input ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            ai_reply, audio_reply, _timing, _diag = call_backend_text(user_input, st.session_state.get("dt_phase", "Discovery"), doc_ctx=doc_context)
            def _stream_reply():
                for word in ai_reply.split(" "):
                    yield word + " "
                    time.sleep(0.01)
            st.write_stream(_stream_reply)
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
