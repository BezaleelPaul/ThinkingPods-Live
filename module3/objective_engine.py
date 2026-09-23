"""
module3.objective_engine — orchestration layer of the Objective Engine.

Responsibility
--------------
Wire up the deterministic, read-only Empathize discovery pipeline:

    ProjectState
        |  CompletenessChecker.evaluate
        v
    CoverageReport
        |  information-gain scoring (module3.scoring_engine)
        |    fallback: PriorityEngine (WRAP_UP / legacy comparison)
        v
    CandidateScore list / PriorityDecision
        |  enrich_phase
        v
    ConversationObjective (structured output for Module 4)

Selection policy
----------------
Instead of pure checklist priority ("what item is missing and declared most
important?"), the engine scores every unmet candidate by information gain:
intrinsic importance, how many later objectives it unlocks, whether the user
has already discussed the topic, and whether a question family for it was
already asked. On canonical Empathize states with no conversation context the
scorer reproduces the legacy ordering exactly; with history it pivots
naturally toward what the user just opened up about. The full candidate /
score / selected / rejected trace is attached to the ConversationObjective
for the Developer Console.

Read-only contract
------------------
The Objective Engine inspects ``ProjectState`` (and optionally a read-only
conversation context) and returns its decision. It never mutates the state,
never calls the LLM, never touches the network.

Determinism
-----------
Because CompletenessChecker, the scorer, the rule registry, and sufficiency
are all pure and side-effect-free, the ObjectiveEngine is itself a pure
function of its inputs. The same ``ProjectState`` (and same context) ALWAYS
produces the same ``ConversationObjective`` (right down to ordering of
reasoning strings and the float value of ``confidence``).

Extensibility
-------------
This engine will eventually support future Design-Thinking stages (Define,
Ideate, Prototype, Validation, ...) — each represented by a different rule
registry and a different ConversationObjective shape. The ObjectiveEngine
accepts a custom rule registry via its ``rules`` constructor argument, so
supporting a new stage is "build a rule registry, register an Objective enum
value", not "engine code edits".
"""

from __future__ import annotations

from typing import List, Optional

from memory_extractor import ProjectState, REQUIRED_FIELDS, StateField

from .completeness_checker import CompletenessChecker
from .objective import ConversationObjective, CoverageReport, Objective
from .priority_engine import PriorityDecision, PriorityEngine
from .rules import DEFAULT_RULES, Rule, RuleRegistry, rule_for_field
from .scoring_engine import (
    CandidateScore,
    ObjectiveContext,
    build_candidate_scores,
)
from .sufficiency import SufficiencyChecker, SufficiencyLevel, SufficiencyReport

__all__ = ["ObjectiveEngine"]


class ObjectiveEngine:
    """
    Deterministic, read-only Objective Engine for the Empathize v2 pipeline.

    The single public entry point is :meth:`determine_next`. It takes a
    ``ProjectState``, returns a ``ConversationObjective``, and changes
    nothing else.

    Construction
    ------------
    The engine is parameterised by the rule registry it uses for
    prioritisation. The default is :data:`module3.rules.DEFAULT_RULES`
    (the canonical Empathize discovery order: personas -> problems ->
    frequency -> current_solutions -> pain_points -> evidence). Passing
    a custom registry lets later stages reuse this engine without
    rewriting the orchestration logic.

    Thread-safety / state
    ---------------------
    The engine stores only immutable configuration (its rule registry and
    the two sub-components). It has no per-turn state. Calling
    :meth:`determine_next` from multiple threads on different state objects
    is safe (Python GIL aside) — there is no shared mutable state to
    corrupt.
    """

    def __init__(self, rules: Optional[RuleRegistry] = None) -> None:
        """
        Parameters
        ----------
        rules:
            Optional custom rule registry. Defaults to
            :data:`module3.rules.DEFAULT_RULES` (the Empathize order).
        """
        # The engine is a thin composition wrapper. We hold the
        # sub-components so callers can introspect the configuration, and
        # so a future Objective Engine carrying extra steps (e.g. a
        # "stage-transition" rule evaluator) slots in here cleanly.
        self._priority_engine: PriorityEngine = PriorityEngine(rules=rules)
        # CompletenessChecker is stateless — hold it for interface
        # symmetry and so tests can swap a different checker in.
        self._completeness_checker: CompletenessChecker = CompletenessChecker()
        # SufficiencyChecker is stateless — the same requirement applies:
        # it classifies ProjectState values into Unknown/Partial/Sufficient/
        # Complete so diagnostics can explain why an objective advanced.
        self._sufficiency_checker: SufficiencyChecker = SufficiencyChecker()

    # ------------------------------------------------------------------
    # Read-only accessors (configuration surface; no runtime state)
    # ------------------------------------------------------------------

    @property
    def rules(self) -> RuleRegistry:
        """Return the rule registry the engine was configured with."""
        return self._priority_engine.rules

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def determine_next(
        self,
        state: ProjectState,
        context: Optional[ObjectiveContext] = None,
    ) -> ConversationObjective:
        """
        Compute the next conversational objective from ``state``.

        This is the ONLY public method of the Objective Engine. It runs
        the deterministic pipeline described in the module docstring:

            state -> CoverageReport -> candidate scoring
                 -> ConversationObjective (with reasoning + confidence
                    + selection trace)

        Parameters
        ----------
        state:
            A :class:`memory_extractor.ProjectState` instance. The engine
            reads it via :class:`CompletenessChecker`; never writes to it.
        context:
            Optional read-only conversation context (raw user messages +
            asked question families). ``None`` behaves exactly like an empty
            context, reproducing the legacy priority ordering — so existing
            callers and tests that only pass a state keep working unchanged.

        Returns
        -------
        ConversationObjective
            Frozen, structured output for Module 4 to consume.

        Raises
        ------
        TypeError
            If ``state`` is not a ProjectState or ``context`` is not an
            ObjectiveContext. Module 3 is the strict ProjectState-only
            boundary — no MentorSession fallback, no compatibility shim.
        RuntimeError
            If the report is not all-complete but no rule covered the
            missing field(s). This would indicate a rule-registry
            misconfiguration rather than a runtime fault.
        """
        if not isinstance(state, ProjectState):
            raise TypeError(
                f"ObjectiveEngine.determine_next requires a ProjectState, "
                f"got {type(state).__name__}"
            )
        if context is None:
            context = ObjectiveContext()
        if not isinstance(context, ObjectiveContext):
            raise TypeError(
                f"context must be an ObjectiveContext, got {type(context).__name__}"
            )

        # --- 1. read-only coverage evaluation ----------------------------
        coverage: CoverageReport = self._completeness_checker.evaluate(state)

        # --- 2. decision: information-gain scoring (legacy WRAP_UP path
        # kept verbatim for the all-complete condition).
        sufficiency: SufficiencyReport = self._sufficiency_checker.evaluate(state)
        if coverage.all_complete:
            decision: PriorityDecision = self._priority_engine.decide(coverage)
            return self._compose_objective(
                coverage=coverage,
                decision=decision,
                sufficiency=sufficiency,
                selection_trace=None,
            )

        scores: List[CandidateScore] = build_candidate_scores(
            self.rules, coverage, context
        )
        if not scores:
            raise RuntimeError(
                "No candidate rules matched the missing fields. The rule "
                "registry does not cover every required field — please add "
                "a rule for each StateField you want the engine to chase."
            )

        # --- 2b. pick the winner deterministically -----------------------
        winner: CandidateScore = scores[0]
        tied = [s for s in scores if s.score == winner.score]
        confidence = (
            PriorityEngine.CONFIDENCE_UNIQUE
            if len(tied) == 1
            else PriorityEngine.CONFIDENCE_TIEBREAK
        )
        winning_rule: Optional[Rule] = rule_for_field(self.rules, winner.field)
        tied_rules: List[Rule] = [
            rule_for_field(self.rules, s.field)
            for s in tied
            if s.field != winner.field
        ]

        # --- 2c. objective sufficiency (Unknown/Partial/Sufficient/Complete)
        # The sufficiency classification is a READ-ONLY layer over the same
        # ProjectState. It never gates WRAP_UP or reorders candidates — it
        # explains the decision (see _compose_objective) and feeds
        # diagnostics with the advancement reason.
        decision = PriorityDecision(
            objective=winner.objective,
            winning_rule=winning_rule,
            tied_rules=tied_rules,
            confidence=confidence,
            reasoning=self._scoring_reasoning(scores, winner, tied),
        )

        # --- 3. enrich the decision into the final ConversationObjective -
        return self._compose_objective(
            coverage=coverage,
            decision=decision,
            sufficiency=sufficiency,
            selection_trace=self._build_selection_trace(scores, winner, tied),
        )

    # ------------------------------------------------------------------
    # Internal: conversion from PriorityDecision + CoverageReport ->
    # final ConversationObjective (reasoning + confidence + field lists)
    # ------------------------------------------------------------------

    def _compose_objective(
        self,
        coverage: CoverageReport,
        decision: PriorityDecision,
        sufficiency: SufficiencyReport,
        selection_trace: Optional[dict] = None,
    ) -> ConversationObjective:
        """
        Compose the final ConversationObjective.

        Keeps the PriorityDecision focused on priority logic and the
        CoverageReport focused on facts; this method owns the
        "what do we tell Module 4" view. It computes:

          * ``missing_fields`` / ``completed_fields`` straight from
            CoverageReport (preserving declared enum order so output is
            deterministic, not list-iteration-order-dependent).
          * ``confidence`` straight from the PriorityDecision (no extra
            arithmetic — same input, same confidence).
          * ``sufficiency`` — serialisable per-field levels from the
            SufficiencyReport.
          * ``advancement`` / ``advancement_reason`` — the engine's verdict
            on whether this objective advanced, stayed active, or wrapped
            up (see :meth:`_advancement_for`).
          * ``reasoning`` — the PriorityDecision's reasoning strings live
            here, with one or two extra context lines prepended to record
            the coverage state and missing/completed field counts. The
            ordering is: (a) macro coverage summary, (b) priority-engine
            reasoning, (c) sufficiency summary, (d) advancement verdict, so
            a reader log-grepping the strings sees the cause before the
            effect.
        """
        missing: List[StateField] = coverage.missing_fields
        completed: List[StateField] = coverage.completed_fields

        advancement, advancement_reason = self._advancement_for(
            decision.objective, sufficiency
        )

        # The reasoning list is composed deterministically so it can be
        # asserted on in tests and hashed for analytics. We start with a
        # one-line macro view, then append the engine's own reasoning.
        reasoning: List[str] = []
        reasoning.append(
            f"Coverage: {coverage.num_complete}/{len(REQUIRED_FIELDS)} "
            f"fields populated."
        )
        if completed:
            reasoning.append(
                "Completed fields: "
                + ", ".join(sf.name for sf in completed)
            )
        if missing:
            reasoning.append(
                "Missing fields: "
                + ", ".join(sf.name for sf in missing)
            )
        # Append the priority engine's reasoning verbatim.
        reasoning.extend(decision.reasoning)
        # Sufficiency context: why each field is (or is not) usable.
        reasoning.append(
            "Field sufficiency: "
            + ", ".join(
                f"{sf.name}={sufficiency.level_of(sf).value}"
                for sf in REQUIRED_FIELDS
            )
        )
        reasoning.append(f"Advancement: {advancement} — {advancement_reason}")

        return ConversationObjective(
            objective=decision.objective,
            missing_fields=missing,
            completed_fields=completed,
            confidence=decision.confidence,
            reasoning=reasoning,
            sufficiency=sufficiency.as_dict(),
            advancement=advancement,
            advancement_reason=advancement_reason,
            selection_trace=selection_trace,
        )

    def _scoring_reasoning(
        self,
        scores: List[CandidateScore],
        winner: CandidateScore,
        tied: List[CandidateScore],
    ) -> List[str]:
        """
        Build the human-readable reasoning strings for an information-gain
        decision: the ranked candidates with their component scores, why the
        winner won, and why each loser was rejected.
        """
        lines: List[str] = [
            f"Information-gain scoring over {len(scores)} unmet candidate "
            "objectives:"
        ]
        for s in scores:
            lines.append(
                f"Objective.{s.objective.name}: score {s.score:.2f} "
                f"(info value {s.info_value:.2f}, unlock +{s.unlock_bonus:.2f}, "
                f"discussion {s.discussion_bonus:+.2f}, "
                f"repeat {s.repeat_penalty:+.2f})."
            )
        if len(tied) == 1:
            lines.append(
                f"Selected Objective.{winner.objective.name} (highest "
                f"information-gain score {winner.score:.2f})."
            )
        else:
            tied_names = ", ".join(
                f"Objective.{s.objective.name}" for s in tied
            )
            lines.append(
                f"Information-gain tie among {len(tied)} candidates: "
                f"{tied_names}."
            )
            lines.append(
                f"Tiebreak: chose Objective.{winner.objective.name} "
                f"(declared StateField enum order)."
            )
        for s in scores:
            if s.field != winner.field:
                lines.append(
                    f"Rejected Objective.{s.objective.name} "
                    f"(score {s.score:.2f}): lower information-gain than "
                    f"the selected objective."
                )
        return lines

    def _build_selection_trace(
        self,
        scores: List[CandidateScore],
        winner: CandidateScore,
        tied: List[CandidateScore],
    ) -> dict:
        """
        Build the Developer-Console trace: every candidate objective with its
        component scores, the selected objective, the reason it was selected,
        and the rejected candidates with their reasons. Pure plain data so it
        round-trips through JSON (the X-Diagnostics wire contract).
        """
        candidates = [
            {
                "field": s.field.value,
                "objective": s.objective.value,
                "score": round(s.score, 4),
                "components": {
                    "info_value": round(s.info_value, 4),
                    "unlock_bonus": round(s.unlock_bonus, 4),
                    "discussion_bonus": round(s.discussion_bonus, 4),
                    "repeat_penalty": round(s.repeat_penalty, 4),
                },
                "discussed": s.discussed,
                "repeated": s.repeated,
            }
            for s in scores
        ]
        rejected = [
            {
                "objective": s.objective.value,
                "reason": (
                    f"Lower information-gain score "
                    f"({s.score:.2f} vs {winner.score:.2f})."
                ),
            }
            for s in scores
            if s.field != winner.field
        ]
        return {
            "mode": "information_gain",
            "candidates": candidates,
            "selected": winner.objective.value,
            "reason_selected": (
                f"Highest information-gain score ({winner.score:.2f})."
            ),
            "rejected": rejected,
        }

    def _advancement_for(
        self,
        objective: Objective,
        sufficiency: SufficiencyReport,
    ) -> tuple[str, str]:
        """
        Determine this turn's advancement verdict from the current state.

        The engine has no cross-turn history, so the verdict is derived from
        the *current* sufficiency snapshot:

          * WRAP_UP        → all required fields are populated.
          * target is at SUFFICIENT/COMPLETE → the field already holds a
            usable answer; the objective can advance.
          * target is at PARTIAL → some signal exists but no usable answer;
            the mentor should clarify, so the objective stays active.
          * target is at UNKNOWN → nothing captured; objective stays active.

        Returns
        -------
        (advancement, advancement_reason)
            A ``(verdict, human-readable reason)`` pair.
        """
        if objective is Objective.WRAP_UP:
            return (
                "WRAP_UP",
                "All required Empathize fields are populated.",
            )

        field = StateField[objective.name]
        level = sufficiency.level_of(field)
        if level in (SufficiencyLevel.SUFFICIENT, SufficiencyLevel.COMPLETE):
            return (
                "ADVANCED",
                f"{field.name} already holds sufficient information "
                f"({level.value}); advancing to the next unmet objective.",
            )
        if level is SufficiencyLevel.PARTIAL:
            return (
                "STAYED_ACTIVE",
                f"{field.name} holds only partial information; a clarifying "
                f"question keeps the objective active.",
            )
        return (
            "STAYED_ACTIVE",
            f"No {field.name} information has been captured yet; the "
            f"objective stays active.",
        )
