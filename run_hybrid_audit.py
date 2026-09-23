"""
run_hybrid_audit.py — deterministic Hybrid Extraction Quality Audit (Old vs New).

Measurement ONLY. This script replays every golden conversation fixture in
BOTH pipeline modes and reports a side-by-side comparison:

  * OLD  (COMPLEXITY_GATE=false): the pre-gate behavior. On any turn where the
         objective-aware hybrid decision is satisfied by the deterministic
         rules, the LLM extractor is skipped and the fixture's canned LLM
         extraction runs in SHADOW mode (never applied) so we can quantify
         what skipping the LLM may have missed.
  * NEW  (COMPLEXITY_GATE=true): the Semantic Complexity Gate is active.
         Rules-satisfied turns classified LOW still skip the LLM (and are
         shadow-audited); MEDIUM/HIGH turns now run the LLM for real, so
         fewer turns land on the risky skip path.

The run is fully deterministic:

  * ``ollama`` is stubbed (deterministic fallback replies).
  * ``mentor.MemoryExtractor.extract`` is patched with the fixture's canned
    extraction — the same source used for the shadow run AND the LLM-invoked
    turns.
  * the "average extraction time estimate" uses a documented, configurable
    per-LLM-call constant (``LLM_EXTRACTION_EST_MS``, default 1500 ms); it is
    an estimate for the report only, never a measurement or a decision input.

No real LLM is called. Nothing is applied from shadow mode. Nothing about
extraction, the hybrid decision, objectives, lifecycle, prompts, or replies
is changed — this script only reports measurements.

Usage:
    python run_hybrid_audit.py
"""

import os
import sys
from unittest import mock

os.environ["HYBRID_AUDIT"] = "true"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.goldens.fixtures import load_all_fixtures  # noqa: E402
from tests.goldens.runner import (  # noqa: E402
    _RaisingOllama,
    _StubOllama,
    _build_extraction,
    _seed_session,
)

# Representative average LLM extraction latency used ONLY for the report's
# time estimate. Tune via LLM_EXTRACTION_EST_MS. Not a measurement.
DEFAULT_LLM_EXTRACTION_EST_MS = float(
    os.environ.get("LLM_EXTRACTION_EST_MS", "1500")
)


def _run_audit(gate_enabled: bool) -> dict:
    from hybrid_audit import RISK_HIGH, RISK_LOW, RISK_MEDIUM, RISK_NONE

    old_gate = os.environ.get("COMPLEXITY_GATE")
    os.environ["COMPLEXITY_GATE"] = "true" if gate_enabled else "false"
    try:
        total_turns = 0
        llm_invoked = 0
        hybrid_skips = 0
        shadow_executed = 0
        no_difference = 0
        with_additional = 0
        with_objective_change = 0
        risk_counts = {RISK_NONE: 0, RISK_LOW: 0, RISK_MEDIUM: 0, RISK_HIGH: 0}
        complexity_counts = {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "NONE": 0,
        }
        additional_field_counts: dict[str, int] = {}
        high_risk_examples: list[dict] = []

        for fixture in load_all_fixtures():
            import mentor

            _seed_session(fixture)
            for index, turn in enumerate(fixture.turns):
                total_turns += 1
                ollama_stub = (
                    _RaisingOllama()
                    if turn.ollama_reply is None
                    else _StubOllama([turn.ollama_reply])
                )
                extraction = _build_extraction(turn)
                with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
                    with mock.patch(
                        "mentor.MemoryExtractor.extract", return_value=extraction
                    ):
                        _reply, _session, _timing, diagnostics = (
                            mentor.process_mentor_turn(
                                turn.user,
                                username=fixture.username,
                                project_name=fixture.project_name,
                            )
                        )

                hybrid = diagnostics.get("HybridExtraction") or {}
                if hybrid.get("LLM Invoked") == "Yes":
                    llm_invoked += 1

                complexity = (diagnostics.get("SemanticComplexity") or {}).get(
                    "Complexity"
                )
                complexity_counts[complexity or "NONE"] = (
                    complexity_counts.get(complexity or "NONE", 0) + 1
                )

                audit = diagnostics.get("HybridAudit")
                if not audit:
                    continue
                hybrid_skips += 1
                shadow_executed += 1

                additional = audit.get("Additional Fields", [])
                would_change_objective = audit.get("Would Change Objective") == "Yes"
                risk = audit.get("Risk Level", RISK_NONE)

                risk_counts[risk] = risk_counts.get(risk, 0) + 1
                if risk == RISK_NONE or not additional:
                    no_difference += 1
                if additional:
                    with_additional += 1
                    for label in additional:
                        additional_field_counts[label] = (
                            additional_field_counts.get(label, 0) + 1
                        )
                if would_change_objective:
                    with_objective_change += 1
                if risk == RISK_HIGH:
                    high_risk_examples.append(
                        {
                            "fixture": fixture.name,
                            "turn": index,
                            "user": turn.user,
                            "shadow_fields": audit.get("Shadow LLM Fields", []),
                            "additional_fields": additional,
                            "would_change_objective": would_change_objective,
                        }
                    )

        return {
            "mode": "NEW" if gate_enabled else "OLD",
            "total_turns": total_turns,
            "llm_invoked": llm_invoked,
            "hybrid_skips": hybrid_skips,
            "shadow_executed": shadow_executed,
            "no_difference": no_difference,
            "with_additional": with_additional,
            "with_objective_change": with_objective_change,
            "risk_counts": risk_counts,
            "complexity_counts": complexity_counts,
            "additional_field_counts": additional_field_counts,
            "high_risk_examples": high_risk_examples,
        }
    finally:
        if old_gate is None:
            os.environ.pop("COMPLEXITY_GATE", None)
        else:
            os.environ["COMPLEXITY_GATE"] = old_gate


def _print_mode(result: dict) -> None:
    risk_counts = result["risk_counts"]
    print(f"  LLM invocations:                         {result['llm_invoked']}")
    print(f"  Hybrid skips (LLM skipped):              {result['hybrid_skips']}")
    print(f"  Shadow LLM executed:                     {result['shadow_executed']}")
    print(f"  Risk:  NONE={risk_counts['NONE']}  "
          f"LOW={risk_counts['LOW']}  "
          f"MEDIUM={risk_counts['MEDIUM']}  "
          f"HIGH={risk_counts['HIGH']}")


def main():
    print("=" * 68)
    print("Hybrid Extraction Quality Audit  —  OLD (gate off) vs NEW (gate on)")
    print("=" * 68)

    old = _run_audit(gate_enabled=False)
    new = _run_audit(gate_enabled=True)

    print(f"\n--- OLD (COMPLEXITY_GATE=false) ---")
    print(f"  Total turns:                              {old['total_turns']}")
    _print_mode(old)
    print(f"\n--- NEW (COMPLEXITY_GATE=true) ---")
    print(f"  Total turns:                              {new['total_turns']}")
    _print_mode(new)

    print("-" * 68)
    print("Complexity distribution (NEW mode):")
    cc = new["complexity_counts"]
    print(
        f"  HIGH={cc['HIGH']}  MEDIUM={cc['MEDIUM']}  "
        f"LOW={cc['LOW']}  NONE={cc['NONE']}"
    )

    print("-" * 68)
    skips_saved = old["hybrid_skips"] - new["hybrid_skips"]
    llm_delta = new["llm_invoked"] - old["llm_invoked"]
    risky_old = old["risk_counts"]["HIGH"] + old["risk_counts"]["MEDIUM"]
    risky_new = new["risk_counts"]["HIGH"] + new["risk_counts"]["MEDIUM"]
    est_delta_ms = llm_delta * DEFAULT_LLM_EXTRACTION_EST_MS
    print("Delta (NEW - OLD):")
    print(f"  LLM invocations:   {old['llm_invoked']} -> {new['llm_invoked']} "
          f"(+{llm_delta})")
    print(f"  Hybrid skips:      {old['hybrid_skips']} -> {new['hybrid_skips']} "
          f"(-{skips_saved})")
    print(f"  HIGH/MEDIUM-risk skips: {risky_old} -> {risky_new}")
    print(f"  HIGH-risk skips:   {old['risk_counts']['HIGH']} -> "
          f"{new['risk_counts']['HIGH']}")
    print(
        "  Est. LLM extraction time "
        f"(@{DEFAULT_LLM_EXTRACTION_EST_MS:g} ms/call): "
        f"OLD {old['llm_invoked'] * DEFAULT_LLM_EXTRACTION_EST_MS:.0f} ms -> "
        f"NEW {new['llm_invoked'] * DEFAULT_LLM_EXTRACTION_EST_MS:.0f} ms"
    )
    print(f"  Avg extraction time/turn (est.): "
          f"OLD {old['llm_invoked'] * DEFAULT_LLM_EXTRACTION_EST_MS / old['total_turns']:.0f} ms -> "
          f"NEW {new['llm_invoked'] * DEFAULT_LLM_EXTRACTION_EST_MS / new['total_turns']:.0f} ms")
    if llm_delta > 0:
        print(
            f"  Cost of info-loss protection: +{est_delta_ms:.0f} ms "
            f"across {llm_delta} turns that now run the LLM instead of a "
            f"risky skip"
        )

    print("-" * 68)
    print("Additional fields missed by frequency (OLD -> NEW):")
    all_labels = sorted(
        set(old["additional_field_counts"]) | set(new["additional_field_counts"])
    )
    if not all_labels:
        print("  (none)")
    for label in all_labels:
        print(
            f"  {label}: {old['additional_field_counts'].get(label, 0)} -> "
            f"{new['additional_field_counts'].get(label, 0)}"
        )

    print("-" * 68)
    print("Examples of HIGH-risk skips remaining in NEW mode:")
    if not new["high_risk_examples"]:
        print("  (none)")
    for ex in new["high_risk_examples"][:5]:
        print(f"  [{ex['fixture']} turn {ex['turn']}] {ex['user']!r}")
        print(f"      additional={ex['additional_fields']} "
              f"objective_change={ex['would_change_objective']}")
    print("=" * 68)

    return {"old": old, "new": new}


if __name__ == "__main__":
    main()
