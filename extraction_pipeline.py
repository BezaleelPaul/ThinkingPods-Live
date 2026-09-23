"""
Extraction pipeline — deterministic + optional LLM fact extraction.

Responsibilities:
  * `strip_model_output` — clean reasoning tags from LLM output
  * `clean_val` — sanitise a raw value
  * `RuleBasedExtractor` — regex-based deterministic extraction (no LLM)
  * `InputProcessor` — rule-based + optional LLM extraction orchestrator
  * `merge_extracted_to_state` — apply extracted dict to ProjectState + SessionData

Owns NO session persistence, NO lifecycle decisions, NO reply generation.
"""

__all__ = [
    "strip_model_output",
    "clean_val",
    "RuleBasedExtractor",
    "InputProcessor",
    "merge_extracted_to_state",
]

import json
import os
import re
import time

from timing import add_stage_ms
from memory_extractor import (
    LIST_FIELDS,
    MessageType,
    Operation,
    ProjectState,
    StateField,
    ExtractionResult,
    ExtractionUpdate,
)
from state_manager import StateManager
from session_manager import SessionData


def strip_model_output(text: str) -> str:
    if not text:
        return ""
    think_open = "<" + "think" + ">"
    think_close = "</" + "think" + ">"
    text = re.sub(
        re.escape(think_open) + r"[\s\S]*?" + re.escape(think_close),
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```(?:json|thinking)?\s*[\s\S]*?```", "", text)
    return text.strip()


def clean_val(v):
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("null", "none", ""):
        return None
    return s


# ---------------------------------------------------------------------------
# Deterministic rule-strength confidence (observation-only).
#
# Every fact the rule-based extractor captures carries a confidence derived
# from the strength of the rule that matched it:
#
#   * explicit / unambiguous strong rule match      -> 0.98  (high)
#   * strong frequency cadence phrase               -> 0.82  (high)
#   * explicit evidence observation marker          -> 0.74  (high)
#   * weak / generic wording                        -> 0.62  (medium)
#   * ambiguous / inferred wording                  -> 0.42  (low)
#
# Confidence NEVER feeds back into any decision path (Objective Engine,
# StateManager, lifecycle, reply generation). It only enriches the Developer
# Console view. LLM-produced updates carry the default 1.0 (no rule signal).
# ---------------------------------------------------------------------------

CONFIDENCE_HIGH = 0.98      # explicit, unambiguous rule match (e.g. problem)
CONFIDENCE_STRONG = 0.82    # strong frequency cadence phrase (e.g. frequency)
CONFIDENCE_EVIDENCE = 0.74  # explicit evidence observation marker (e.g. evidence)
CONFIDENCE_MEDIUM = 0.62    # weak wording / keyword fallback
CONFIDENCE_LOW = 0.42       # ambiguous wording / inferred
DEFAULT_CONFIDENCE = 1.0    # no rule signal (LLM-produced facts)


class RuleBasedExtractor:
    """Deterministic extraction - works without any LLM call."""

    # (pattern, handler, confidence) triples — confidence reflects the
    # strength of the matched rule, not a probability.
    AUDIENCE_PATTERNS = [
        (r"\b(elderly people|old people|older adults|seniors|elderly|senior citizens)\b", "elderly people", CONFIDENCE_HIGH),
        (r"\b(college students|students|university students)\b", "college students", CONFIDENCE_HIGH),
        (r"\b(children|kids|parents|caregivers)\b", lambda m: m.group(0), CONFIDENCE_HIGH),
        (r"\b(rural villagers|village community|villagers)\b", "rural villagers / village community", CONFIDENCE_HIGH),
        (r"\b(patients|doctors|nurses|healthcare workers)\b", lambda m: m.group(0), CONFIDENCE_HIGH),
    ]

    PAIN_PATTERNS = [
        (r"forget(s|ting)?\s+(her|his|their|my|the)?\s*medication", "forgetting medication", CONFIDENCE_HIGH),
        (r"miss(es|ing)?\s+(deadlines|assignments|medication|doses)", lambda m: f"missing {m.group(0).split()[1]}", CONFIDENCE_HIGH),
        (r"\b(struggle|struggling|problem|issue|challenge|pain point)\b.{0,40}", lambda m: m.group(0)[:80], CONFIDENCE_MEDIUM),
    ]

    MOTIVATION_PATTERNS = [
        (r"\bmy (grandmother|grandfather|grandma|grandpa|mom|dad|mother|father|parent|friend|family member)\b",
         lambda m: f"personal connection - {m.group(0)}", CONFIDENCE_HIGH),
        (r"\bbecause (my|our|the) .{5,60}", lambda m: m.group(0)[:80], CONFIDENCE_MEDIUM),
    ]

    FREQUENCY_PATTERNS = [
        # Precise cadence first (ordered: longer/more-specific phrases before
        # the generic tokens so "every single day" is captured verbatim, not
        # reduced to a bare "daily").
        (r"\b(every single day|every single night|every single week|every single month)\b", lambda m: m.group(0), CONFIDENCE_STRONG),
        (r"\b(every day|each day|every week|each week|every month|every morning|every evening|every night|every weekend|every weekday|every afternoon)\b", lambda m: m.group(0), CONFIDENCE_STRONG),
        (r"\b(once in a while|from time to time|every so often|now and then)\b", lambda m: m.group(0), CONFIDENCE_MEDIUM),
        (r"\b(once|twice|three times|a couple of times|several times|multiple times|many times)\s+(a|per|each)\s+(day|week|month|year|hour)\b", lambda m: m.group(0), CONFIDENCE_HIGH),
        (r"\b(on most days|most days|most of the time|most weeks|all the time)\b", lambda m: m.group(0), CONFIDENCE_STRONG),
        (r"\b(every day|daily|weekly|monthly|yearly|annually|hourly|nightly|often|always|constantly|regularly|frequently|occasionally|rarely|seldom|usually|normally|typically|sometimes|never)\b", lambda m: m.group(0), CONFIDENCE_MEDIUM),
    ]

    WORKFLOW_PATTERNS = [
        (r"\b(currently|right now|today|they use|we use|existing|currently use)\b.{0,60}", lambda m: m.group(0)[:80], CONFIDENCE_MEDIUM),
        (r"\b(pill box|alarm|reminder app|calendar|sticky notes|phone alarm)\b", lambda m: m.group(0), CONFIDENCE_HIGH),
    ]

    EVIDENCE_PATTERNS = [
        (r"\b(i (have )?seen|i noticed|i observed|research shows|statistics|data|survey|interview)\b.{0,60}",
         lambda m: m.group(0)[:80], CONFIDENCE_EVIDENCE),
    ]

    # Consequences / effects of the problem. Distinguished from problems by
    # their causal cues ("affects", "causes", "leads to", ...). The object of
    # the cue is rephrased so the stored value reads as an impact, not a bare
    # noun (e.g. "academic performance" -> "academic performance suffers").
    IMPACT_PATTERNS = [
        (r"\baffects?\s+(.+?)(?=[.!?,]|$)", lambda m: f"{m.group(1).strip()} suffers", CONFIDENCE_HIGH),
        (r"\bcauses?\s+(.+?)(?=[.!?,]|$)", lambda m: m.group(1).strip(), CONFIDENCE_HIGH),
        (r"\bresults?\s+in\s+(.+?)(?=[.!?,]|$)", lambda m: m.group(1).strip(), CONFIDENCE_HIGH),
        (r"\bleads?\s+to\s+(.+?)(?=[.!?,]|$)", lambda m: m.group(1).strip(), CONFIDENCE_HIGH),
        (r"\bcreates?\s+(.+?)(?=[.!?,]|$)", lambda m: m.group(1).strip(), CONFIDENCE_HIGH),
        (r"\breduces?\s+(.+?)(?=[.!?,]|$)", lambda m: f"{m.group(1).strip()} reduced", CONFIDENCE_HIGH),
        (r"\bimproves?\s+(.+?)(?=[.!?,]|$)", lambda m: f"{m.group(1).strip()} improved", CONFIDENCE_HIGH),
        (r"\bprevents?\s+(.+?)(?=[.!?,]|$)", lambda m: f"{m.group(1).strip()} prevented", CONFIDENCE_HIGH),
        (r"\bwastes?\s+(.+?)(?=[.!?,]|$)", lambda m: f"{m.group(1).strip()} wasted", CONFIDENCE_HIGH),
        (r"\bmakes?\s+(.+?)\s+(difficult|hard|easier|worse)(?=[.!?,]|$)",
         lambda m: f"{m.group(1).strip()} {m.group(2)}", CONFIDENCE_HIGH),
    ]

    @classmethod
    def _first_match(cls, text, patterns):
        lower = text.lower()
        for pattern, handler, confidence in patterns:
            m = re.search(pattern, lower, re.IGNORECASE)
            if m:
                value = handler(m) if callable(handler) else handler
                return value, confidence
        return None, None

    @classmethod
    def extract(cls, user_message, project_name="MyProject", _timing=None):
        _pre = time.perf_counter() if _timing is not None else None
        text = user_message.strip()
        if not text or len(text) < 3:
            if _pre is not None:
                add_stage_ms(_timing, "preprocess", _pre, time.perf_counter())
            return {}

        result = {}
        confidences = {}
        lower = text.lower()
        if _pre is not None:
            add_stage_ms(_timing, "preprocess", _pre, time.perf_counter())

        _t = time.perf_counter() if _timing is not None else None
        audience, conf = cls._first_match(text, cls.AUDIENCE_PATTERNS)
        if audience:
            result["target_audience"] = audience
            confidences["target_audience"] = conf

        pain, conf = cls._first_match(text, cls.PAIN_PATTERNS)
        if pain:
            result["pain_point"] = pain
            confidences["pain_point"] = conf
        elif any(w in lower for w in ("problem", "issue", "struggle", "forget", "miss")):
            result["pain_point"] = text[:120]
            confidences["pain_point"] = CONFIDENCE_MEDIUM

        motivation, conf = cls._first_match(text, cls.MOTIVATION_PATTERNS)
        if motivation:
            result["motivation"] = motivation
            confidences["motivation"] = conf
        elif "because" in lower:
            idx = lower.index("because")
            result["motivation"] = text[idx:idx + 100].strip()
            confidences["motivation"] = CONFIDENCE_LOW

        frequency, conf = cls._first_match(text, cls.FREQUENCY_PATTERNS)
        if frequency:
            result["frequency"] = frequency
            confidences["frequency"] = conf

        workflow, conf = cls._first_match(text, cls.WORKFLOW_PATTERNS)
        if workflow:
            result["existing_solution"] = workflow
            confidences["existing_solution"] = conf

        evidence, conf = cls._first_match(text, cls.EVIDENCE_PATTERNS)
        if evidence:
            result["evidence"] = evidence
            confidences["evidence"] = conf

        impact, conf = cls._first_match(text, cls.IMPACT_PATTERNS)
        if impact:
            result["impact"] = impact
            confidences["impact"] = conf

        build_match = re.search(
            r"(?:build|create|make|develop)\s+(?:an?\s+)?(\w+(?:\s+\w+){0,3})\s+(?:for|to help|that helps)",
            lower,
        )
        if build_match and (not project_name or project_name == "MyProject"):
            words = build_match.group(1).strip()
            if words not in ("system", "app", "application", "project"):
                result["project_name"] = words.title()
            elif "medication" in lower or "medicine" in lower or "pill" in lower:
                result["project_name"] = "Medication Reminder"
            elif "reminder" in lower:
                result["project_name"] = "Reminder System"

        if "medication" in lower or "medicine" in lower or "pill" in lower:
            if not result.get("project_name") and (not project_name or project_name == "MyProject"):
                result["project_name"] = "Medication Reminder"

        if "i don't know" in lower or "not sure" in lower or "depends on" in lower:
            result.setdefault("unknown_facts", []).append(text[:80])
        if "assume" in lower or "probably" in lower or "might" in lower:
            result.setdefault("assumptions", []).append(text[:80])

        if confidences:
            result["confidences"] = confidences

        if _t is not None:
            add_stage_ms(_timing, "rules", _t, time.perf_counter())
        return result


class InputProcessor:
    """Extract structured data: rule-based first, LLM refinement optional."""

    @staticmethod
    def extract_structured_data(model, user_message, project_name="MyProject", project_state=None, use_llm=None, _timing=None, _prof=None):
        if use_llm is None:
            use_llm = os.getenv("MENTOR_USE_LLM_EXTRACTION", "false").lower() == "true"

        extracted = RuleBasedExtractor.extract(user_message, project_name, _timing=_timing)

        if not use_llm:
            return extracted

        if project_state is None:
            return extracted

        llm_extracted = InputProcessor._llm_extract(model, user_message, project_state, project_name, _timing=_timing, _prof=_prof)
        return InputProcessor._merge_extractions(extracted, llm_extracted)

    @staticmethod
    def _merge_extractions(primary, secondary):
        merged = dict(primary)
        for key, val in secondary.items():
            if key in ("assumptions", "unknown_facts", "open_questions"):
                merged.setdefault(key, [])
                for item in val or []:
                    if item and item not in merged[key]:
                        merged[key].append(item)
            elif val and not merged.get(key):
                merged[key] = val
        return merged

    @staticmethod
    def _llm_extract(model, user_message, project_state, project_name, _timing=None, _prof=None):
        _t = time.perf_counter() if _timing is not None else None
        existing = {
            "project_name": project_name,
            "target_audience": project_state.personas[0] if project_state.personas else None,
            "pain_point": project_state.problems[0] if project_state.problems else None,
            "motivation": project_state.pain_points[0] if project_state.pain_points else None,
            "existing_solution": project_state.current_solutions[0] if project_state.current_solutions else None,
            "frequency": project_state.frequency,
            "evidence": project_state.evidence[0] if project_state.evidence else None,
        }
        prompt = f"""Extract design thinking facts from the user message. Reply with ONLY valid JSON, no other text.

User: "{user_message}"
Existing: {json.dumps(existing)}

JSON keys: project_name, target_audience, pain_point, motivation, existing_solution, frequency, evidence, assumptions, unknown_facts, open_questions
Use null for unknown scalar fields. Use [] for empty lists."""
        if _t is not None:
            add_stage_ms(_timing, "llm_prompt", _t, time.perf_counter())

        try:
            from timing import timed_ollama_chat  # local import: keeps module import-light

            _a = time.perf_counter() if _timing is not None else None
            response = timed_ollama_chat(
                _prof,
                purpose="legacy_extraction",
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 256},
                gated=True,  # runs only under MENTOR_USE_LLM_EXTRACTION=true
            )
            if _a is not None:
                add_stage_ms(_timing, "llm_api", _a, time.perf_counter())
            _p = time.perf_counter() if _timing is not None else None
            try:
                reply = strip_model_output(response["message"]["content"])
                start, end = reply.find("{"), reply.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(reply[start: end + 1])
            finally:
                if _p is not None:
                    add_stage_ms(_timing, "llm_parse", _p, time.perf_counter())
        except Exception as e:
            print(f"LLM extraction failed (rule-based fallback active): {e}")
        return {}


def merge_extracted_to_state(state: ProjectState, session_data: SessionData, extracted: dict, _timing=None) -> None:
    if not extracted:
        return

    _LEGACY_TO_STATE = {
        "target_audience": (StateField.PERSONAS, True),
        "pain_point": (StateField.PROBLEMS, True),
        "motivation": (StateField.PAIN_POINTS, True),
        "existing_solution": (StateField.CURRENT_SOLUTIONS, True),
        "evidence": (StateField.EVIDENCE, True),
        "impact": (StateField.IMPACTS, True),
        "frequency": (StateField.FREQUENCY, False),
    }

    mgr = StateManager(state)
    updates = []
    confidences = extracted.get("confidences", {}) or {}
    for legacy_field, (state_field, is_list) in _LEGACY_TO_STATE.items():
        val = clean_val(extracted.get(legacy_field))
        if val:
            op = Operation.ADD if is_list else Operation.SET
            confidence = confidences.get(legacy_field, DEFAULT_CONFIDENCE)
            updates.append(ExtractionUpdate(op, state_field, val, confidence=confidence))
    if updates:
        mgr.apply_extraction(ExtractionResult(message_type=MessageType.MEANINGFUL, updates=updates), _timing=_timing)

    _s = time.perf_counter() if _timing is not None else None
    proj_name = clean_val(extracted.get("project_name"))
    if proj_name and (not session_data.project_name or session_data.project_name == "MyProject"):
        session_data.project_name = proj_name

    for item in extracted.get("assumptions", []) or []:
        val = clean_val(item)
        if val and val not in session_data.assumptions:
            session_data.assumptions.append(val)

    for item in extracted.get("unknown_facts", []) or []:
        val = clean_val(item)
        if val and val not in session_data.unknown_facts:
            session_data.unknown_facts.append(val)

    for item in extracted.get("open_questions", []) or []:
        val = clean_val(item)
        if val and val not in session_data.open_questions:
            session_data.open_questions.append(val)
    if _s is not None:
        add_stage_ms(_timing, "merge_session", _s, time.perf_counter())
