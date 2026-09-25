# Mistake & Failure Tracking Report

**Date:** System Audit
**Total Turns Audited:** 32
**Clean Turns:** 26
**Turns with Mistakes:** 6 (18.8%)

## 1. Executive Summary

| Category | Occurrences | Percentage | Severity |
|---|---|---|---|
| **Repeated information** | 4 | 12.5% | Warning |
| **Meta intent** | 2 | 6.2% | Warning |

## 2. Actionable Fix Recommendations

### Repeated information
- **Explanation:** User re-offered information because previous mentor reply did not confirm or store it. (Details: Repeated information detected.)
- **Recommended Fix:** Review extractor field mappings and ensure prompt reflects acknowledged facts.

### Meta intent
- **Explanation:** User turn was non-substantive (greeting, validation, trust check) rather than design content. (Details: Meta intent detected.)
- **Recommended Fix:** Ensure meta-intent handler acknowledges the user smoothly and bridges back to design thinking.

## 3. Sample Turn-by-Turn Mistake Log

#### Mistake #1: Repeated information (`golden:01_straightforward_full_conversation` T6)
- **User:** "I have seen her skip doses at least twice last week"
- **Root Cause:** User re-offered information because previous mentor reply did not confirm or store it. (Details: Repeated information detected.)
- **Action:** Review extractor field mappings and ensure prompt reflects acknowledged facts.

#### Mistake #2: Meta intent (`golden:01_straightforward_full_conversation` T7)
- **User:** "yes"
- **Root Cause:** User turn was non-substantive (greeting, validation, trust check) rather than design content. (Details: Meta intent detected.)
- **Action:** Ensure meta-intent handler acknowledges the user smoothly and bridges back to design thinking.

#### Mistake #3: Repeated information (`golden:04_late_evidence` T6)
- **User:** "I surveyed 20 students last month"
- **Root Cause:** User re-offered information because previous mentor reply did not confirm or store it. (Details: Repeated information detected.)
- **Action:** Review extractor field mappings and ensure prompt reflects acknowledged facts.

#### Mistake #4: Repeated information (`golden:07_existing_solution` T1)
- **User:** "They all use sticky notes and a paper planner"
- **Root Cause:** User re-offered information because previous mentor reply did not confirm or store it. (Details: Repeated information detected.)
- **Action:** Review extractor field mappings and ensure prompt reflects acknowledged facts.

#### Mistake #5: Meta intent (`golden:09_short_message` T1)
- **User:** "Who are the users?"
- **Root Cause:** User turn was non-substantive (greeting, validation, trust check) rather than design content. (Details: Meta intent detected.)
- **Action:** Ensure meta-intent handler acknowledges the user smoothly and bridges back to design thinking.

#### Mistake #6: Repeated information (`golden:10_long_message_rule_based` T2)
- **User:** "I care because my friend experiences this too"
- **Root Cause:** User re-offered information because previous mentor reply did not confirm or store it. (Details: Repeated information detected.)
- **Action:** Review extractor field mappings and ensure prompt reflects acknowledged facts.
