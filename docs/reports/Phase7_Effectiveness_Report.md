# Phase 7 — Conversational Layer Effectiveness Report

## Executive Summary

- Turns evaluated per arm: 108 across 40 novel scenarios + 1 control
- NET improvement (prevented − introduced): **13** (prevented 27 / introduced 14)
- Responses changed by layer: 42
- Extra LLM calls: 0 (one attempt/turn in every arm)

## Methodology

- Three arms over identical conversations:
  - **BASELINE**: layer disabled via mocked NORMAL-context classifier
  - **DETECTION_ONLY**: real classification recorded; pause/ack/suppression neutralised
  - **ACTIVE**: production behaviour untouched
- Deterministic fallback path exercised (raising ollama stub); prompts captured
- Decision-level rubric (10 criteria, 0/1/2) + failure taxonomy per turn

## Dataset

| Category | Scenarios | Turns |
|---|---|---|
| A-NORMAL | 5 | 10 |
| B-CORRECTION | 7 | 21 |
| C-TOPIC_SHIFT | 6 | 17 |
| D-DIRECT_QUESTION | 5 | 14 |
| E-CONFUSION | 4 | 11 |
| F-HYPOTHETICAL | 6 | 13 |
| G-AMBIGUOUS | 2 | 6 |
| H-MIXED | 5 | 11 |

## Arm: BASELINE

- turns=108 pauses=0 acks=0 suppressions=0
- total failures=30 {'MISSED_CONVERSATIONAL_MOVE': 30}
- median latency=33.14 ms (avg 31.24), llm_attempts=108

| Criterion | Avg (0-2) |
|---|---|
| conversational_appropriateness | 1.444 |
| understanding | 1.417 |
| dt_continuity | 2 |
| information_preservation | 2 |
| unnecessary_interruption | 2 |
| repetition_avoidance | 2 |
| topic_respect | 1.88 |
| direct_question_handling | 1.889 |
| ambiguity_handling | 2 |
| recovery | 2 |

## Arm: DETECTION_ONLY

- turns=108 pauses=0 acks=2 suppressions=0
- total failures=30 {'MISSED_CONVERSATIONAL_MOVE': 30}
- median latency=35.47 ms (avg 32.13), llm_attempts=108

| Criterion | Avg (0-2) |
|---|---|
| conversational_appropriateness | 1.435 |
| understanding | 1.926 |
| dt_continuity | 2 |
| information_preservation | 2 |
| unnecessary_interruption | 2 |
| repetition_avoidance | 2 |
| topic_respect | 1.88 |
| direct_question_handling | 1.889 |
| ambiguity_handling | 2 |
| recovery | 2 |

## Arm: ACTIVE

- turns=108 pauses=27 acks=2 suppressions=11
- total failures=17 {'MISSED_CONVERSATIONAL_MOVE': 3, 'FAILED_RESUMPTION': 13, 'FALSE_PAUSE': 1}
- median latency=33.19 ms (avg 30.92), llm_attempts=108

| Criterion | Avg (0-2) |
|---|---|
| conversational_appropriateness | 1.926 |
| understanding | 1.926 |
| dt_continuity | 2 |
| information_preservation | 2 |
| unnecessary_interruption | 1.981 |
| repetition_avoidance | 2 |
| topic_respect | 1.972 |
| direct_question_handling | 2 |
| ambiguity_handling | 2 |
| recovery | 2 |

## Baseline vs Layer (ACTIVE)

- Prevented: {'MISSED_CONVERSATIONAL_MOVE': 27}
- Introduced: {'FAILED_RESUMPTION': 13, 'FALSE_PAUSE': 1}
- false_pause_rate=0.0093 missed_move(layer)=0.0909 missed_move(baseline)=0.9091
- information_loss layer=0.0 baseline=0.0
- derailment(layer)=0.0 repetition layer/baseline=0/0

### Detection-only vs Active deltas
- turns whose failure-set changed when behaviour activated: 41
- ('corr_not_issue', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('corr_trying_to_say', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('corr_trying_to_say', 3): [] -> ['FAILED_RESUMPTION']
- ('corr_assuming', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('corr_assuming', 3): [] -> ['FAILED_RESUMPTION']
- ('corr_situation_different', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('corr_personally', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('corr_not_the_point', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('corr_not_the_point', 3): [] -> ['FAILED_RESUMPTION']
- ('top_moved_on', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('top_moved_on', 3): [] -> ['FAILED_RESUMPTION']
- ('top_leave_aside', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('top_leave_aside', 3): [] -> ['FAILED_RESUMPTION']
- ('top_discuss_different', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('top_forget_previous', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('dq_how_approach', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('dq_how_approach', 3): [] -> ['FAILED_RESUMPTION']
- ('dq_what_try', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []
- ('dq_what_try', 3): [] -> ['FAILED_RESUMPTION']
- ('dq_help_figure', 2): ['MISSED_CONVERSATIONAL_MOVE'] -> []

### Behavioural misses remaining (novel-wording gaps)
- corr_misunderstood T2: “I think you misunderstood what I'm saying.”
- top_no_longer T2: “That's no longer what I'm working on.”
- amb_maybe_problem T2: “Maybe I'm the problem.”
### Classification quality (ACTIVE arm, label-level)
{"LABEL_MISS": 3, "WRONG_INTERPRETATION": 1}

## State / Objective integrity

- derailment rate (layer): 0.0
- contamination findings: see failure tables per arm

## Resumption

- FAILED_RESUMPTION (layer): 13

## Latency / LLM calls

| Arm | median ms | avg ms | llm attempts |
|---|---|---|---|
| baseline | 33.14 | 31.24 | 108 |
| detection_only | 35.47 | 32.13 | 108 |
| active | 33.19 | 30.92 | 108 |

## Safety

- matrix accuracy 100%; no-trace verified: True

## Representative Before/After (changed replies)

- `corr_trying_to_say` T2 [correction] “That's not quite what I was trying to say.”
  - BASELINE: "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
  - LAYER(CORRECTION/HIGH): "Got it - thanks for clarifying. Tell me a bit more about what you have in mind whenever you're ready."
- `corr_trying_to_say` T3 [normal] “Seniors skip refills every month.”
  - BASELINE: "That's a useful angle. To understand why it weighs on people, why does this problem matter to the people exper"
  - LAYER(NORMAL_DT/HIGH): "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
- `corr_assuming` T2 [correction] “You're assuming something I didn't say.”
  - BASELINE: "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
  - LAYER(CORRECTION/HIGH): "Got it - thanks for clarifying. Tell me a bit more about what you have in mind whenever you're ready."
- `corr_assuming` T3 [normal] “It's the evening dose that gets skipped.”
  - BASELINE: "That's a useful angle. To understand why it weighs on people, why does this problem matter to the people exper"
  - LAYER(NORMAL_DT/HIGH): "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
- `corr_situation_different` T2 [correction] “Actually, my situation is different.”
  - BASELINE: "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
  - LAYER(CORRECTION/HIGH): "Got it - thanks for clarifying. Tell me a bit more about what you have in mind whenever you're ready."
- `corr_personally` T2 [correction] “I meant the problem I'm personally having.”
  - BASELINE: "That's a clear picture of the situation. To understand how urgently this needs solving, how often does the pro"
  - LAYER(CORRECTION/HIGH): "Got it - thanks for clarifying. Tell me a bit more about what you have in mind whenever you're ready."
- `corr_personally` T3 [normal] “I keep missing my own medication alarms.”
  - BASELINE: "That's a useful gauge of scale. To see how widely this reaches people, what percentage of the people you're de"
  - LAYER(NORMAL_DT/HIGH): "That's a clear picture of the situation. To understand how urgently this needs solving, how often does the pro"
- `corr_not_the_point` T2 [correction] “No, that's not the point.”
  - BASELINE: "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
  - LAYER(CORRECTION/HIGH): "Got it - thanks for clarifying. Tell me a bit more about what you have in mind whenever you're ready."
- `corr_not_the_point` T3 [normal] “Caregivers carry the burden here.”
  - BASELINE: "That's a useful angle. To understand why it weighs on people, why does this problem matter to the people exper"
  - LAYER(NORMAL_DT/HIGH): "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
- `top_moved_on` T2 [topic_shift] “Actually, I've moved on to a different project.”
  - BASELINE: "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
  - LAYER(TOPIC_SHIFT/HIGH): "Understood - let's switch to the new direction. Tell me about what you're working on now."
- `top_moved_on` T3 [normal] “It's about reducing food waste in canteens.”
  - BASELINE: "That's a useful angle. To understand why it weighs on people, why does this problem matter to the people exper"
  - LAYER(NORMAL_DT/HIGH): "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
- `top_leave_aside` T2 [topic_shift] “Let's leave that aside for now.”
  - BASELINE: "That's a real problem taking shape. To make it concrete, could you give me an example of the problem they run "
  - LAYER(TOPIC_SHIFT/HIGH): "Understood - let's switch to the new direction. Tell me about what you're working on now."

## Verdict

B — PROMISING BUT NEEDS WORK: net improvement exists, but important failure modes remain (see behavioural miss list on novel wording).