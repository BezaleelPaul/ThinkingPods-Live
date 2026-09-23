"""
knowledge_base.py — Structured concept knowledge for the Empathize pipeline.

This module provides a deterministic, LLM-independent knowledge base of
design-thinking concepts. It contains zero inference logic — only structured
concept definitions that can be extended without touching the pipeline.

Architecture role
-----------------
Concept knowledge used during extraction and prompt building.
    │
    ├── Traffic Congestion
    ├── Medication Adherence
    ├── Food Waste
    ├── Assignment Procrastination
    ├── Teacher Attendance
    └── Household Water Wastage
    │
    ▼
Hypotheses (inferred personas, problems, solutions, pain_points, etc.)

Design principles
-----------------
- Frozen dataclasses: same inputs → same outputs
- No LLM calls, no randomness, no I/O
- Modular: new domains = new concept entries only
- Explainable: every inference traces to a concept entry
- Deterministic: lookup by exact match or alias normalization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Concept Category Enum
# ---------------------------------------------------------------------------


class ConceptCategory(str, Enum):
    """High-level category of a design-thinking concept."""

    MOBILITY = "mobility"
    """Transportation, traffic, commuting, urban mobility."""

    HEALTHCARE = "healthcare"
    """Medication, treatment adherence, patient care, medical routines."""

    FOOD_SYSTEMS = "food_systems"
    """Food waste, supply chain, restaurants, agriculture."""

    EDUCATION = "education"
    """Student/teacher workflows, assignments, attendance, learning."""

    HOME_UTILITIES = "home_utilities"
    """Water, energy, household waste, domestic routines."""

    WORKPLACE = "workplace"
    """Office workflows, productivity, remote work, team coordination."""

    GENERAL = "general"
    """Cross-cutting or uncategorized concepts."""


# ---------------------------------------------------------------------------
# Scored Inference Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredInference:
    """
    A single inferred field value with its confidence.

    The confidence represents how strongly the concept suggests this
    inference. It is NOT an LLM confidence — it is a deterministic weight
    assigned by the knowledge base designer.
    """

    value: str
    """The inferred value (e.g., "commuters", "pill organizers")."""

    confidence: float
    """Deterministic weight in [0.0, 1.0]. Higher = more strongly suggested."""

    rationale: str = ""
    """Human-readable explanation of why this inference is suggested."""


@dataclass(frozen=True)
class ConceptInferences:
    """
    All field-level inferences that a concept can suggest.

    Each list contains ScoredInference objects sorted by confidence descending.
    """

    personas: tuple[ScoredInference, ...] = field(default_factory=tuple)
    """Who is affected — target audiences, user groups."""

    problems: tuple[ScoredInference, ...] = field(default_factory=tuple)
    """Core problems/challenges being experienced."""

    current_solutions: tuple[ScoredInference, ...] = field(default_factory=tuple)
    """Existing workarounds, tools, or methods people use."""

    pain_points: tuple[ScoredInference, ...] = field(default_factory=tuple)
    """Frustrations, emotional impact, why it hurts."""

    evidence: tuple[ScoredInference, ...] = field(default_factory=tuple)
    """Observations, data, research validating the problem."""

    frequency: tuple[ScoredInference, ...] = field(default_factory=tuple)
    """How often the problem occurs (temporal qualifiers)."""

    def all_inferences(self) -> tuple[ScoredInference, ...]:
        """Flatten all inference lists into a single tuple."""
        return (
            self.personas
            + self.problems
            + self.current_solutions
            + self.pain_points
            + self.evidence
            + self.frequency
        )


# ---------------------------------------------------------------------------
# Concept Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Concept:
    """
    A reusable design-thinking concept with structured inferences.

    Immutable — the knowledge base stores concepts. No logic lives here.
    """

    id: str
    """Unique identifier (slug): "traffic_congestion", "medication_adherence"."""

    canonical_name: str
    """Human-readable canonical name: "Traffic Congestion"."""

    aliases: tuple[str, ...] = field(default_factory=tuple)
    """Alternative names/keywords for matching: ("traffic jam", "congestion", "rush hour traffic")."""

    category: ConceptCategory = ConceptCategory.GENERAL
    """High-level domain category."""

    inferences: ConceptInferences = field(default_factory=ConceptInferences)
    """All field-level inferences this concept suggests."""

    notes: str = ""
    """Optional designer notes, sources, or context."""

    def matches(self, query: str) -> bool:
        """
        Check if a user query matches this concept.

        Matches against canonical_name or any alias (case-insensitive,
        substring). Normalizes whitespace and punctuation.
        """
        normalized = query.lower().strip()
        if normalized in self.canonical_name.lower():
            return True
        return any(normalized in alias.lower() for alias in self.aliases)


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """
    Deterministic registry of design-thinking concepts.

    No LLM, no randomness, no I/O — pure lookup.

    Usage
    -----
    >>> kb = KnowledgeBase()
    >>> concept = kb.find("traffic congestion")
    >>> concept.inferences.problems[0].value
    'traffic congestion'
    """

    def __init__(self, concepts: tuple[Concept, ...] | None = None) -> None:
        """
        Parameters
        ----------
        concepts:
            Optional tuple of Concept objects. If None, uses the built-in
            seed concepts (see _SEED_CONCEPTS).
        """
        if concepts is None:
            concepts = _SEED_CONCEPTS
        self._concepts: tuple[Concept, ...] = concepts
        self._by_id: dict[str, Concept] = {c.id: c for c in concepts}
        # Build alias index for fast lookup
        self._alias_index: dict[str, Concept] = {}
        for c in concepts:
            self._alias_index[c.canonical_name.lower()] = c
            for alias in c.aliases:
                self._alias_index[alias.lower()] = c

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    def find(self, query: str) -> Concept | None:
        """
        Find a concept by query string.

        Tries exact match on canonical_name or alias first, then falls back
        to substring match. Returns None if no match.
        """
        if not query:
            return None
        normalized = query.lower().strip()

        # Exact match
        if normalized in self._alias_index:
            return self._alias_index[normalized]

        # Substring match (longest match wins)
        matches = [
            c for key, c in self._alias_index.items()
            if key in normalized or normalized in key
        ]
        if matches:
            # Prefer longest key (most specific)
            return max(matches, key=lambda c: len(c.canonical_name))
        return None

    def get(self, concept_id: str) -> Concept | None:
        """Retrieve a concept by its unique ID."""
        return self._by_id.get(concept_id)

    def all(self) -> tuple[Concept, ...]:
        """Return all concepts in registration order."""
        return self._concepts

    def by_category(self, category: ConceptCategory) -> tuple[Concept, ...]:
        """Filter concepts by category."""
        return tuple(c for c in self._concepts if c.category == category)

    # ------------------------------------------------------------------
    # Batch / explainability helpers
    # ------------------------------------------------------------------

    def get_all_inferences(self) -> tuple[ScoredInference, ...]:
        """Flatten all inferences from all concepts."""
        all_inf = []
        for c in self._concepts:
            all_inf.extend(c.inferences.all_inferences())
        # Deduplicate by value, keep highest confidence
        by_value: dict[str, ScoredInference] = {}
        for inf in all_inf:
            if inf.value not in by_value or inf.confidence > by_value[inf.value].confidence:
                by_value[inf.value] = inf
        return tuple(sorted(by_value.values(), key=lambda x: -x.confidence))

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._by_id

    def __iter__(self):
        return iter(self._concepts)


# ---------------------------------------------------------------------------
# Seed Concepts — Regression Test Domains
# ---------------------------------------------------------------------------
# Each concept captures reusable domain knowledge.

_SEED_CONCEPTS: tuple[Concept, ...] = (

    # 1. Traffic Congestion
    Concept(
        id="traffic_congestion",
        canonical_name="Traffic Congestion",
        aliases=(
            "traffic jam",
            "congestion",
            "rush hour traffic",
            "gridlock",
            "traffic delays",
            "road congestion",
        ),
        category=ConceptCategory.MOBILITY,
        inferences=ConceptInferences(
            personas=(
                ScoredInference("commuters", 0.95, "Daily travelers on congested routes"),
                ScoredInference("delivery drivers", 0.85, "Time-sensitive route-dependent workers"),
                ScoredInference("rideshare drivers", 0.8, "Income depends on travel time"),
                ScoredInference("public transit riders", 0.7, "Bus delays from same congestion"),
                ScoredInference("urban residents", 0.6, "Noise, pollution, quality of life"),
            ),
            problems=(
                ScoredInference("traffic congestion", 0.98, "Core concept"),
                ScoredInference("unpredictable travel time", 0.9, "Cannot plan arrival"),
                ScoredInference("excessive fuel consumption", 0.7, "Idling and stop-and-go"),
                ScoredInference("increased emissions", 0.65, "Environmental impact"),
            ),
            current_solutions=(
                ScoredInference("navigation apps (Waze, Google Maps)", 0.9, "Real-time rerouting"),
                ScoredInference("leaving earlier", 0.8, "Buffer time"),
                ScoredInference("public transit", 0.6, "Avoids driving"),
                ScoredInference("carpooling", 0.5, "Reduces cars on road"),
            ),
            pain_points=(
                ScoredInference("stress and frustration", 0.9, "Daily unpredictable delays"),
                ScoredInference("lost productive time", 0.85, "Hours wasted in traffic"),
                ScoredInference("late arrivals", 0.8, "Missed appointments, meetings"),
                ScoredInference("higher transportation costs", 0.7, "Fuel, vehicle wear"),
            ),
            evidence=(
                ScoredInference("traffic volume data", 0.9, "Sensor/camera counts"),
                ScoredInference("average commute time studies", 0.8, "Census/transportation dept"),
                ScoredInference("GPS probe data", 0.8, "Aggregate speed profiles"),
            ),
            frequency=(
                ScoredInference("daily (rush hour)", 0.95, "Peak periods"),
                ScoredInference("weekdays", 0.9, "Work commute pattern"),
                ScoredInference("holiday weekends", 0.6, "Special events"),
            ),
        ),
        notes=(
            "Core concept: traffic congestion during peak hours in metropolitan areas. "
            "Inferences cover commuters, delivery/logistics, and urban residents. "
            "Frequency is strongly daily/weekday."
        ),
    ),

    # 2. Medication Adherence
    Concept(
        id="medication_adherence",
        canonical_name="Medication Adherence",
        aliases=(
            "medication reminder",
            "forgetting medication",
            "pill reminder",
            "medication non-adherence",
            "missed doses",
            "medication compliance",
        ),
        category=ConceptCategory.HEALTHCARE,
        inferences=ConceptInferences(
            personas=(
                ScoredInference("elderly patients", 0.95, "Polypharmacy, memory decline"),
                ScoredInference("chronic disease patients", 0.9, "Long-term daily meds"),
                ScoredInference("caregivers", 0.85, "Managing others' medications"),
                ScoredInference("busy professionals", 0.7, "Hectic schedules"),
                ScoredInference("patients with cognitive impairment", 0.8, "Memory challenges"),
            ),
            problems=(
                ScoredInference("forgetting to take medication", 0.98, "Core concept"),
                ScoredInference("taking wrong dose", 0.8, "Confusion, complex regimens"),
                ScoredInference("stopping medication early", 0.7, "Feeling better, side effects"),
                ScoredInference("complex dosing schedules", 0.85, "Multiple meds, different times"),
            ),
            current_solutions=(
                ScoredInference("pill organizers", 0.9, "Weekly/daily compartments"),
                ScoredInference("phone alarms", 0.85, "Simple reminders"),
                ScoredInference("medication apps", 0.8, "Tracking + reminders"),
                ScoredInference("pharmacy blister packs", 0.7, "Pre-sorted doses"),
                ScoredInference("caregiver reminders", 0.75, "Human prompt"),
            ),
            pain_points=(
                ScoredInference("anxiety about health risks", 0.9, "Fear of complications"),
                ScoredInference("guilt over missed doses", 0.8, "Self-blame"),
                ScoredInference("frustration with complex regimens", 0.85, "Too many pills/times"),
                ScoredInference("financial waste", 0.6, "Unused expensive medication"),
            ),
            evidence=(
                ScoredInference("WHO adherence studies (50%)", 0.9, "Global meta-analyses"),
                ScoredInference("hospital readmission data", 0.8, "Non-adherence → complications"),
                ScoredInference("pharmacy refill records", 0.85, "MPR/PDC metrics"),
            ),
            frequency=(
                ScoredInference("daily", 0.95, "Most meds are daily"),
                ScoredInference("multiple times daily", 0.8, "BID/TID/QID dosing"),
                ScoredInference("weekly (some)", 0.4, "Weekly meds like methotrexate"),
            ),
        ),
        notes=(
            "Core concept: patients forgetting or incorrectly taking prescribed medication. "
            "Strongest for elderly, chronic conditions, complex regimens. "
            "Current solutions are low-tech (pill boxes) and digital (apps/alarms)."
        ),
    ),

    # 3. Food Waste
    Concept(
        id="food_waste",
        canonical_name="Food Waste",
        aliases=(
            "food waste",
            "wasted food",
            "restaurant food waste",
            "plate waste",
            "food spoilage",
            "uneaten food",
        ),
        category=ConceptCategory.FOOD_SYSTEMS,
        inferences=ConceptInferences(
            personas=(
                ScoredInference("restaurant owners/managers", 0.9, "Direct cost impact"),
                ScoredInference("household consumers", 0.85, "Home food waste"),
                ScoredInference("grocery stores", 0.8, "Spoilage, aesthetic standards"),
                ScoredInference("food service staff", 0.7, "Preparation waste"),
                ScoredInference("farmers/producers", 0.6, "Upstream cosmetic rejection"),
            ),
            problems=(
                ScoredInference("food waste", 0.98, "Core concept"),
                ScoredInference("oversized portions", 0.9, "Restaurants serve too much"),
                ScoredInference("cosmetic standards", 0.8, "Ugly produce rejected"),
                ScoredInference("poor inventory management", 0.75, "Spoilage before use"),
                ScoredInference("lack of donation infrastructure", 0.7, "Legal/logistical barriers"),
            ),
            current_solutions=(
                ScoredInference("smaller portion options", 0.8, "Half portions, sides"),
                ScoredInference("food donation programs", 0.75, "Too Good To Go, food banks"),
                ScoredInference("composting", 0.6, "Diverts from landfill"),
                ScoredInference("dynamic pricing", 0.6, "Discount near-expiry"),
                ScoredInference("staff training", 0.55, "Prep waste reduction"),
            ),
            pain_points=(
                ScoredInference("financial loss", 0.9, "Direct cost of wasted inventory"),
                ScoredInference("environmental guilt", 0.75, "Methane from landfills"),
                ScoredInference("regulatory pressure", 0.7, "Waste diversion mandates"),
                ScoredInference("customer perception", 0.6, "Sustainability expectations"),
            ),
            evidence=(
                ScoredInference("FAO 1/3 food wasted globally", 0.9, "UN FAO reports"),
                ScoredInference("restaurant waste audits", 0.8, "Pre/post-consumer split"),
                ScoredInference("household waste studies", 0.8, "Diary/weight methods"),
            ),
            frequency=(
                ScoredInference("daily (per meal service)", 0.9, "Continuous in food service"),
                ScoredInference("weekly (household)", 0.7, "Grocery cycles"),
            ),
        ),
        notes=(
            "Core concept: edible food discarded at retail, service, and household levels. "
            "Restaurant plate waste (oversized portions) is a major lever. "
            "Strong financial + environmental pain points."
        ),
    ),

    # 4. Assignment Procrastination
    Concept(
        id="assignment_procrastination",
        canonical_name="Assignment Procrastination",
        aliases=(
            "procrastination",
            "late assignments",
            "missed deadlines",
            "homework procrastination",
            "student procrastination",
            "putting off work",
        ),
        category=ConceptCategory.EDUCATION,
        inferences=ConceptInferences(
            personas=(
                ScoredInference("college students", 0.95, "Autonomy + workload"),
                ScoredInference("high school students", 0.85, "Multiple classes"),
                ScoredInference("graduate students", 0.8, "Thesis/projects"),
                ScoredInference("online learners", 0.8, "Self-paced, no structure"),
            ),
            problems=(
                ScoredInference("procrastination", 0.98, "Core concept"),
                ScoredInference("missed deadlines", 0.9, "Direct consequence"),
                ScoredInference("rushed low-quality work", 0.85, "Last-minute cramming"),
                ScoredInference("grade penalties", 0.8, "Late submission policies"),
                ScoredInference("all-nighters", 0.75, "Sleep deprivation"),
            ),
            current_solutions=(
                ScoredInference("calendar/planner apps", 0.85, "Google Calendar, Notion"),
                ScoredInference("breaking tasks into steps", 0.8, "Chunking"),
                ScoredInference("study groups/accountability", 0.7, "Social pressure"),
                ScoredInference("Pomodoro technique", 0.7, "Timed focus blocks"),
                ScoredInference("professor reminders", 0.6, "Canvas/Blackboard notifications"),
            ),
            pain_points=(
                ScoredInference("stress and anxiety", 0.95, "Looming deadlines"),
                ScoredInference("guilt and self-blame", 0.85, "Knowing better but not doing"),
                ScoredInference("lower grades", 0.8, "Direct academic impact"),
                ScoredInference("burnout", 0.75, "Cycle of cramming and recovery"),
            ),
            evidence=(
                ScoredInference("Piers Steel meta-analysis (80-95%)", 0.9, "Procrastination prevalence"),
                ScoredInference("Tice & Baumeister (1997)", 0.8, "Procrastination → stress/grades"),
                ScoredInference("student survey data", 0.85, "Self-reported rates"),
            ),
            frequency=(
                ScoredInference("per assignment cycle", 0.9, "Every deadline"),
                ScoredInference("weekly (recurring)", 0.7, "Weekly problem sets"),
                ScoredInference("end-of-term crunch", 0.7, "Multiple simultaneous deadlines"),
            ),
        ),
        notes=(
            "Core concept: delaying starting/completing academic work until deadline pressure. "
            "Near-universal among students. Solutions are mostly self-regulation tools. "
            "Pain points are heavily emotional (guilt, anxiety) + academic (grades)."
        ),
    ),

    # 5. Teacher Attendance
    Concept(
        id="teacher_attendance",
        canonical_name="Teacher Attendance",
        aliases=(
            "teacher absenteeism",
            "teacher absence",
            "substitute teacher",
            "teacher no-show",
            "faculty attendance",
        ),
        category=ConceptCategory.EDUCATION,
        inferences=ConceptInferences(
            personas=(
                ScoredInference("school administrators", 0.95, "Manage coverage"),
                ScoredInference("students", 0.9, "Learning disruption"),
                ScoredInference("substitute teachers", 0.8, "Last-minute coverage"),
                ScoredInference("parents", 0.7, "Childcare, learning concerns"),
                ScoredInference("other teachers", 0.75, "Cover classes during prep"),
            ),
            problems=(
                ScoredInference("teacher absenteeism", 0.98, "Core concept"),
                ScoredInference("instructional continuity loss", 0.9, "Substitutes ≠ regular teacher"),
                ScoredInference("last-minute scramble", 0.85, "Admin burden"),
                ScoredInference("student behavior issues", 0.8, "Less structure with subs"),
                ScoredInference("equity gaps", 0.7, "High-poverty schools hit harder"),
            ),
            current_solutions=(
                ScoredInference("substitute teacher pools", 0.8, "District-maintained lists"),
                ScoredInference("attendance incentives", 0.7, "Bonuses for low absence"),
                ScoredInference("automated absence reporting", 0.75, "Frontline/Aesop systems"),
                ScoredInference("teacher collaboration/coverage", 0.6, "Team covers for team"),
                ScoredInference("remote teaching option", 0.5, "Teach from home when mild"),
            ),
            pain_points=(
                ScoredInference("administrative burden", 0.9, "Daily coverage scramble"),
                ScoredInference("student learning loss", 0.95, "Substitutes less effective"),
                ScoredInference("teacher burnout (covering peers)", 0.8, "Lost prep periods"),
                ScoredInference("parent complaints", 0.7, "Quality concerns"),
            ),
            evidence=(
                ScoredInference("NCES teacher absence data (6%)", 0.9, "National averages"),
                ScoredInference("Gershenson et al. (2017) learning loss", 0.85, "10 days → measurable drop"),
                ScoredInference("substitute fill-rate reports", 0.8, "District tracking"),
            ),
            frequency=(
                ScoredInference("daily (average 6-10 days/year)", 0.9, "Chronic low-level"),
                ScoredInference("seasonal spikes (flu season)", 0.8, "Winter peaks"),
                ScoredInference("professional development days", 0.6, "Planned absences"),
            ),
        ),
        notes=(
            "Core concept: teachers absent from classroom, requiring substitutes. "
            "Chronic ~6-10 days/year average. Substitutes less effective → learning loss. "
            "Admin burden high; fill rates often <80%."
        ),
    ),

    # 6. Household Water Wastage
    Concept(
        id="household_water_wastage",
        canonical_name="Household Water Wastage",
        aliases=(
            "water waste",
            "water wastage",
            "leaky faucet",
            "running toilet",
            "long showers",
            "water conservation",
        ),
        category=ConceptCategory.HOME_UTILITIES,
        inferences=ConceptInferences(
            personas=(
                ScoredInference("homeowners", 0.95, "Pay water bills, maintain property"),
                ScoredInference("renters", 0.7, "May not control fixtures"),
                ScoredInference("property managers", 0.75, "Multi-unit costs"),
                ScoredInference("environmentally conscious", 0.8, "Values conservation"),
            ),
            problems=(
                ScoredInference("household water wastage", 0.98, "Core concept"),
                ScoredInference("leaky faucets/pipes", 0.9, "Continuous drip"),
                ScoredInference("running toilets", 0.85, "Silent leak, high volume"),
                ScoredInference("excessive irrigation", 0.8, "Overwatering lawns"),
                ScoredInference("long showers", 0.75, "Behavioral waste"),
            ),
            current_solutions=(
                ScoredInference("fix leaks (DIY/plumber)", 0.9, "High ROI repair"),
                ScoredInference("low-flow fixtures", 0.85, "Showerheads, aerators"),
                ScoredInference("smart irrigation controllers", 0.75, "Weather-based"),
                ScoredInference("shower timers", 0.6, "Behavioral nudge"),
                ScoredInference("water bill monitoring", 0.65, "Detect spikes"),
            ),
            pain_points=(
                ScoredInference("high water bills", 0.9, "Direct financial cost"),
                ScoredInference("environmental concern", 0.8, "Drought, aquifer depletion"),
                ScoredInference("property damage (leaks)", 0.75, "Mold, structural"),
                ScoredInference("guilt over waste", 0.6, "Personal values"),
            ),
            evidence=(
                ScoredInference("EPA: 10% homes leak 90 gal/day", 0.9, "WaterSense data"),
                ScoredInference("running toilet = 200 gal/day", 0.85, "Standard estimate"),
                ScoredInference("outdoor = 30% household use", 0.8, "EPA WaterSense"),
            ),
            frequency=(
                ScoredInference("continuous (leaks)", 0.9, "24/7 until fixed"),
                ScoredInference("daily (showers)", 0.85, "Behavioral"),
                ScoredInference("seasonal (irrigation)", 0.8, "Summer peaks"),
            ),
        ),
        notes=(
            "Core concept: residential water lost to leaks, inefficient fixtures, behavior. "
            "Leaks are silent and continuous — highest waste per incident. "
            "Fixing leaks has highest ROI; behavioral changes need nudges."
        ),
    ),

)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """Return the process-wide KnowledgeBase instance."""
    global _KNOWLEDGE_BASE
    if _KNOWLEDGE_BASE is None:
        _KNOWLEDGE_BASE = KnowledgeBase()
    return _KNOWLEDGE_BASE


# ---------------------------------------------------------------------------
# Example API Usage
# ---------------------------------------------------------------------------
# >>> kb = get_knowledge_base()
# >>> concept = kb.find("traffic congestion")
# >>> concept
# Concept(id='traffic_congestion', canonical_name='Traffic Congestion', ...)
# >>> concept.inferences.problems[0]
# ScoredInference(value='traffic congestion', confidence=0.98, rationale='Core concept')
# >>> kb.find("medication reminder")
# Concept(id='medication_adherence', canonical_name='Medication Adherence', ...)
# >>> kb.find("unknown concept")
# None
# >>> kb.by_category(ConceptCategory.HEALTHCARE)
# (Concept(id='medication_adherence', ...),)
# >>> kb.get_all_inferences()[:3]
# (ScoredInference(value='traffic congestion', confidence=0.98, ...), ...)