# Phase 4 — Conversational Replay Evaluation Report

## Category results

| Category | Detection | Pause correctness | Contamination | Family pollution |
|---|---|---|---|---|
| CORRECTION | 6/6 | 6/6 | 0 | 0 |
| TOPIC_SHIFT | 7/7 | 7/7 | 0 | 0 |
| DIRECT_QUESTION | 3/3 | 3/3 | 0 | 0 |
| CONFUSED | 9/9 | 8/8 | 0 | 0 |
| HYPOTHETICAL | 5/5 | 5/5 | 0 | 0 |
| AMBIGUOUS | 5/5 | 5/5 | 0 | 0 |
| NORMAL_DT CONTROL | 6/6 | 6/6 | 0 | 0 |

## Findings (decision-level mismatches)

- none

## Naturalness grouping

### GOOD (31)
- `correction_flow` T2 [CORRECTION] “That's not what I meant.” — simulated-LLM reply follows mode guidance (acknowledge-first)
- `correction_flow` T3 [NORMAL_DT] “My grandmother always forgets her pills” — normal DT exchange
- `correction_variants_bare` T2 [CORRECTION] “No, you're misunderstanding me.” — gate's conversational fallback addresses the user move
- `correction_variants_bare` T3 [CORRECTION] “I didn't say that.” — gate's conversational fallback addresses the user move
- `correction_variants_bare` T4 [CORRECTION] “Actually, that's not what I'm talking about.” — gate's conversational fallback addresses the user move
- `correction_variants_bare` T5 [CORRECTION] “No, I mean my own situation.” — gate's conversational fallback addresses the user move
- `topic_shift_flow` T2 [TOPIC_SHIFT] “Never mind the coding thing. I'm working on some” — gate's conversational fallback addresses the user move
- `topic_shift_flow` T3 [NORMAL_DT] “It happens every single day” — normal DT exchange
- `topic_shift_variants` T2 [TOPIC_SHIFT] “Forget that, let's talk about something else.” — gate's conversational fallback addresses the user move
- `topic_shift_variants` T3 [TOPIC_SHIFT] “Actually, I've changed the project completely.” — gate's conversational fallback addresses the user move
- `topic_shift_variants` T4 [TOPIC_SHIFT] “Let's switch topics.” — gate's conversational fallback addresses the user move
- `topic_shift_variants` T5 [TOPIC_SHIFT] “I want to talk about something else.” — gate's conversational fallback addresses the user move
- `direct_question_flow` T2 [DIRECT_QUESTION] “I'm stuck. What should I do?” — gate's conversational fallback addresses the user move
- `direct_question_flow` T3 [NORMAL_DT] “They currently use paper planners” — normal DT exchange
- `direct_question_embedded` T2 [DIRECT_QUESTION] “I'm struggling with coding logic. What should I ” — simulated-LLM reply follows mode guidance (acknowledge-first)
- `confusion_flow` T2 [CONFUSED] “I don't understand.” — simulated-LLM reply follows mode guidance (acknowledge-first)
- `confusion_flow` T3 [NORMAL_DT] “It happens every week” — normal DT exchange
- `confusion_variants` T2 [CONFUSED] “What does that mean?” — gate's conversational fallback addresses the user move
- `confusion_variants` T3 [CONFUSED] “I'm confused.” — gate's conversational fallback addresses the user move
- `confusion_variants` T7 [CONFUSED] “I don't get what you mean.” — gate's conversational fallback addresses the user move
- `confusion_variants` T8 [CONFUSED] “Can you explain that?” — gate's conversational fallback addresses the user move
- `hypothetical_flow` T2 [HYPOTHETICAL] “Suppose I were a dog.” — simulated-LLM reply follows mode guidance (acknowledge-first)
- `hypothetical_flow` T3 [NORMAL_DT] “They skip breakfast most days” — normal DT exchange
- `hypothetical_stakeholder_framing` T2 [HYPOTHETICAL] “What if this were designed for children?” — gate's conversational fallback addresses the user move
- `hypothetical_stakeholder_framing` T3 [HYPOTHETICAL] “Let's say the users were elderly people.” — simulated-LLM reply follows mode guidance (acknowledge-first)
- `hypothetical_stakeholder_framing` T4 [HYPOTHETICAL] “Imagine we're designing this for a school.” — gate's conversational fallback addresses the user move
- `normal_control` T1 [NORMAL_DT] “I want to help elderly people use smartphones.” — normal DT exchange
- `normal_control` T2 [NORMAL_DT] “Students struggle to understand the assignment.” — normal DT exchange
- `normal_control` T3 [NORMAL_DT] “The problem is that people don't know where to s” — normal DT exchange
- `normal_control` T4 [NORMAL_DT] “We're building an app for small businesses.” — normal DT exchange
- `normal_control` T5 [NORMAL_DT] “I've noticed this happens about three times a we” — normal DT exchange

### ACCEPTABLE (21)
- `correction_flow` T1 [NORMAL_DT] “I want to help students keep track of assignment” — canonical DT template (questionnaire tone, unchanged baseline)
- `correction_variants_bare` T1 [NORMAL_DT] “I want to help students organize their homework” — canonical DT template (questionnaire tone, unchanged baseline)
- `topic_shift_flow` T1 [NORMAL_DT] “I'm building a tool to help students organize ho” — canonical DT template (questionnaire tone, unchanged baseline)
- `topic_shift_variants` T1 [NORMAL_DT] “I'm building a study planner for college student” — canonical DT template (questionnaire tone, unchanged baseline)
- `topic_shift_variants` T6 [NORMAL_DT] “I'm starting a new project for local libraries.” — canonical DT template (questionnaire tone, unchanged baseline)
- `direct_question_flow` T1 [NORMAL_DT] “Students in my area struggle with exam preparati” — canonical DT template (questionnaire tone, unchanged baseline)
- `direct_question_embedded` T1 [NORMAL_DT] “College students juggle a lot of coursework” — canonical DT template (questionnaire tone, unchanged baseline)
- `confusion_flow` T1 [NORMAL_DT] “I want to reduce missed deadlines for university” — canonical DT template (questionnaire tone, unchanged baseline)
- `confusion_variants` T1 [NORMAL_DT] “I want to help students keep track of assignment” — canonical DT template (questionnaire tone, unchanged baseline)
- `confusion_variants` T4 [NORMAL_DT] “Not sure honestly.” — canonical DT template (questionnaire tone, unchanged baseline)
- `confusion_variants` T5 [NORMAL_DT] “Maybe weekly.” — canonical DT template (questionnaire tone, unchanged baseline)
- `confusion_variants` T6 [CONFUSED] “I don't know, probably students.” — observation-only flag; standard reply
- `hypothetical_flow` T1 [NORMAL_DT] “I want to improve campus dining for students” — canonical DT template (questionnaire tone, unchanged baseline)
- `hypothetical_stakeholder_framing` T1 [NORMAL_DT] “I want to help students keep track of assignment” — canonical DT template (questionnaire tone, unchanged baseline)
- `ambiguous_preservation` T1 [NORMAL_DT] “I want to help students keep track of assignment” — canonical DT template (questionnaire tone, unchanged baseline)
- `ambiguous_preservation` T2 [AMBIGUOUS] “I'm a dog.” — ambiguity preserved; normal DT template follows (robotic but safe)
- `ambiguous_preservation` T3 [AMBIGUOUS] “I'm an alien.” — ambiguity preserved; normal DT template follows (robotic but safe)
- `ambiguous_preservation` T4 [AMBIGUOUS] “I'm a potato.” — ambiguity preserved; normal DT template follows (robotic but safe)
- `ambiguous_preservation` T5 [NORMAL_DT] “I work with developers.” — canonical DT template (questionnaire tone, unchanged baseline)
- `ambiguous_preservation` T6 [NORMAL_DT] “Actually I'm a developer building an app.” — canonical DT template (questionnaire tone, unchanged baseline)
- `normal_control` T6 [NORMAL_DT] “Mostly they forget things and it stresses everyo” — canonical DT template (questionnaire tone, unchanged baseline)

### BAD (0)
- (none)

### REGRESSION (0)
- (none)

## False positives (hard: wrongly paused)
- ('Imagine badges as rewards in the classroom economy.', 'HYPOTHETICAL', 'HIGH')
- ("Suppose it rains - attendance drops, that's all.", 'HYPOTHETICAL', 'HIGH')

## Paused but content preserved (intentional, Phase 5)
- ('Teachers suppose that students review notes nightly.', 'HYPOTHETICAL', 'HIGH')

## Soft flags (labelled non-NORMAL without pausing)
- none

## Cross-turn resumption
- `resume_correction`: PASS (canonical-resume=True)
- `resume_topic_shift`: PASS (canonical-resume=True)
- `resume_direct_q`: PASS (canonical-resume=True)
- `resume_confused`: PASS (canonical-resume=True)
- `resume_hypothetical`: PASS (canonical-resume=True)

## Safety
- Matrix accuracy: 100%; no-trace verified: True
## Headline metrics
- NORMAL_DT control preservation: see category table (NORMAL_DT CONTROL row)
- ProjectState contamination (fixtures): 0
- Pause-turn pivot-language ingestion (resumption probe): 0
- family_asked pollution: 0
- Cross-turn resume: 5/5
- Extra LLM calls introduced: 0
- Messages classified in FP sweep: 48
