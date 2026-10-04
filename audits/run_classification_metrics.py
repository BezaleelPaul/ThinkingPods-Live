"""
run_classification_metrics.py — Rules vs Transformer shadow-vote metrics.

Measurement ONLY. Replays every golden conversation fixture against the live
``mentor.process_mentor_turn`` pipeline with BOTH observation-only pretrained
classifiers switched on:

    SENTIMENT_ENABLED=true       (sentiment of the user message)
    CLASSIFIER_ENABLED=true      (zero-shot shadow vote on MessageType)

and reports the numbers that answer "is the transformer more effective than
the current rules?":

  * Shadow-vote availability (how many turns produced an ok vote)
  * Rules vs transformer AGREEMENT RATE on MessageType
  * Every DISAGREEMENT (message, rules' answer, shadow's answer) so a human
    can judge which side was right
  * Sentiment distribution across all replayed user messages
  * Average classifier latency per turn

Nothing about objectives, lifecycle, prompts, extraction, or replies is
changed — the classifiers are observation-only and never feed a decision
path. Deterministic: ``ollama`` is stubbed and ``MemoryExtractor.extract``
is patched with each fixture's canned extraction (same as the other audit
runners).

Requirements: ``pip install transformers`` and the models downloaded once
(see OFFLINE_USAGE.md). If the models are unavailable the report degrades
to Status=unavailable counts instead of failing.

Usage:
    python audits/run_classification_metrics.py
"""

import os
import sys
from collections import Counter
from unittest import mock

# Both classifiers ON for this measurement run (mirrors run_hybrid_audit).
os.environ["SENTIMENT_ENABLED"] = "true"
os.environ["CLASSIFIER_ENABLED"] = "true"
os.environ.setdefault("COMPLEXITY_GATE", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.goldens.fixtures import load_all_fixtures  # noqa: E402
from tests.goldens.runner import (  # noqa: E402
    _RaisingOllama,
    _StubOllama,
    _build_extraction,
    _seed_session,
)


def _warm_classifiers():
    """Load both HF pipelines BEFORE the first ``mock.patch.dict("sys.modules")``.

    The golden replay harness patches ``sys.modules`` per turn; on exit the
    patch restores the dict, EVICTING any module first imported during the
    block. If torch/transformers first import inside the block, later turns
    re-import a second torch state (``THPDtypeType.tp_dict == nullptr``
    assert, then a 0xC0000005 access violation). Warming up outside any
    patch keeps them resident in ``sys.modules`` for the whole run.
    """
    from message_classifier import classify_message
    from sentiment import analyze_sentiment

    analyze_sentiment("warm-up")
    classify_message("warm-up", rule_label="MEANINGFUL")


def _collect():
    """Replay all fixtures; return per-turn measurement rows."""
    rows = []
    for fixture in load_all_fixtures():
        import mentor  # noqa: F402

        _seed_session(fixture)
        for turn in fixture.turns:
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
                    _reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                        turn.user,
                        username=fixture.username,
                        project_name=fixture.project_name,
                    )
            rows.append(
                {
                    "fixture": fixture.name,
                    "message": turn.user,
                    "rules": (diagnostics.get("Extraction") or {}).get("message_type"),
                    "shadow": diagnostics.get("MessageClassification") or {},
                    "sentiment": diagnostics.get("SentimentAnalysis") or {},
                }
            )
    return rows


def _latency_ms(section):
    text = str(section.get("Latency") or "0")
    try:
        return float(text.replace(" ms", ""))
    except ValueError:
        return 0.0


def _print_report(rows):
    total = len(rows)
    votes = [r["shadow"] for r in rows]
    ok_votes = [v for v in votes if v.get("Status") == "ok"]
    unavailable = [v for v in votes if v.get("Status") == "unavailable"]

    decided = [v for v in ok_votes if v.get("Agrees With Rules") in ("Yes", "No")]
    agree = [v for v in decided if v.get("Agrees With Rules") == "Yes"]
    disagree = [r for r in rows if (r["shadow"] or {}).get("Agrees With Rules") == "No"]

    sentiments = [r["sentiment"] for r in rows]
    sent_ok = [s for s in sentiments if s.get("Status") == "ok"]
    sent_counts = Counter(s.get("Label") or "?" for s in sent_ok)

    print("=" * 68)
    print("Classification Metrics — Rules vs Transformer (shadow vote)")
    print("=" * 68)
    print(f"  Turns replayed:              {total}")
    print(f"  Shadow votes (ok):           {len(ok_votes)}")
    print(f"  Shadow votes (unavailable):  {len(unavailable)}")
    print()

    print("--- MessageType Agreement (rules vs zero-shot) ---")
    if decided:
        rate = len(agree) / len(decided) * 100
        print(f"  Decided votes:   {len(decided)}")
        print(f"  Agreement:       {len(agree)} ({rate:.1f}%)")
        print(f"  Disagreement:    {len(disagree)}")
    else:
        print("  (no ok votes — are transformers/models installed?)")
    print()

    if disagree:
        print("--- Disagreements (judge which side is right) ---")
        for r in disagree:
            shadow = r["shadow"]
            print(f"  [{r['fixture']}]")
            print(f"    message: {r['message']!r}")
            print(
                f"    rules:   {r['rules']}   "
                f"shadow: {shadow.get('Shadow Label')} "
                f"({shadow.get('Confidence')})"
            )
        print()

    print("--- Sentiment Distribution ---")
    if sent_ok:
        for label in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
            n = sent_counts.get(label, 0)
            rate = n / len(sent_ok) * 100
            print(f"  {label:<10} {n:>3}/{len(sent_ok)} ({rate:5.1f}%)")
    else:
        print("  (no ok sentiment results)")
    print()

    lat = [_latency_ms(v) for v in ok_votes if _latency_ms(v) > 0]
    if lat:
        print(f"  Avg shadow latency:   {sum(lat) / len(lat):.1f} ms (n={len(lat)})")
    sent_lat = [_latency_ms(s) for s in sent_ok if _latency_ms(s) > 0]
    if sent_lat:
        print(
            f"  Avg sentiment latency: {sum(sent_lat) / len(sent_lat):.1f} ms "
            f"(n={len(sent_lat)})"
        )
    if unavailable or any(s.get("Status") == "unavailable" for s in sentiments):
        print()
        print("  NOTE: some classifiers were unavailable — run:")
        print("    pip install transformers")
        print('    python -c "from transformers import pipeline;')
        print(
            "    pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english');"
        )
        print(
            "    pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-1')\""
        )
    print("=" * 68)


def main():
    _warm_classifiers()
    rows = _collect()
    _print_report(rows)
    return rows


if __name__ == "__main__":
    main()
