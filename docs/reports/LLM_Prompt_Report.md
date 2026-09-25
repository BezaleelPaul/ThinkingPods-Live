# LLM Prompt Report — Mentor Pipeline

**Generated:** 2026-07-29  
**Method:** Captured via `ollama.chat` monkey-patch during one real `process_mentor_turn()` call  
**Test message:** *"We need a system to help elderly people remember their medication schedules. My mother is 78 and often forgets to take her blood pressure pills. I'm worried she might have a stroke."*  
**Model (Call 1):** `qwen2.5:3b` (MemoryExtractor)  
**Model (Call 2):** `optimized-pods` (Mentor Response)  

Raw prompt dumps saved to `_prompt_dumps/`.

---

## Call 1: MemoryExtractor — Full Prompt Breakdown

**Role:** Deterministic semantic extraction → structured JSON  
**Total size:** 22,639 chars / 2,445 words / ~5,659 estimated tokens  
**Generation options:** `temperature=0.0, top_p=1.0, num_predict=300, repeat_penalty=1.0`

### Section Map

| # | Section | Lines | Chars | Words | Est. Tokens | % of Total | Type |
|---|---------|-------|-------|-------|-------------|------------|------|
| a | System Identity + Conservatism Warning | 1–7 | 430 | 45 | 107 | 1.9% | **static** |
| b | Strict Rules + Output Schema | 8–32 | 720 | 90 | 180 | 3.2% | **static** |
| c | Field Names Table | 33–41 | 690 | 82 | 172 | 3.0% | **static** |
| d | Field Definitions with Examples | 42–93 | 3,299 | 388 | 824 | 14.6% | **static** |
| e | Critical Differentiation Rules | 94–130 | 1,472 | 196 | 368 | 6.5% | **static** |
| f | Operation Rules | 131–138 | 492 | 58 | 123 | 2.2% | **static** |
| g | Message Type Definitions | 139–145 | 484 | 63 | 121 | 2.1% | **static** |
| **h** | **Comprehensive Examples** | **146–605** | **13,563** | **1,326** | **3,390** | **59.9%** | **static** |
| i | CURRENT PROJECT STATE | 606–618 | 564 | 57 | 141 | 2.5% | **dynamic** |
| j | PREVIOUS ASSISTANT MESSAGE | 619–620 | 696 | 103 | 174 | 3.1% | **dynamic** |
| k | USER MESSAGE | 621–623 | 198 | 34 | 49 | 0.9% | **dynamic** |
| l | OUTPUT marker | 624–625 | 20 | 3 | 5 | 0.1% | **static** |

### Static vs. Dynamic

| Balance | Chars | % | Est. Tokens |
|---------|-------|---|-------------|
| **Static** | 21,157 | **93.5%** | ~5,289 |
| **Dynamic** | 1,481 | **6.5%** | ~370 |

### Key Finding: Dominance of Examples

Section **h (Comprehensive Examples)** is the single largest component at **59.9%** of the prompt.
It contains **20 example conversations**, each with:
- A user message (realistic but synthetic)
- A full JSON output following the exact schema the model must produce

Every example follows the same JSON template — `message_type`, `updates[]`, `operation`, `field`, `value`, `context`, `domain`, `keywords`, `raw_text`, `extraction_method: "llm"`. This creates massive repetition: the field name `"evidence"` appears 21 times in the examples alone.

### Prompt Cache Opportunity

If the static portions (sections a–h) were cached, the per-turn cost drops from **~5,659 tokens** to **~370 tokens** — a **93% reduction** in input tokens for this call.

---

## Call 2: Mentor Response — Full Prompt Breakdown

**Role:** Natural-language mentor reply (Empathize summary)  
**Total size:** 1,877 chars / 263 words / ~469 estimated tokens  
**Generation options:** `temperature=0.4, top_p=0.85, num_predict=150`

### Section Map

| # | Section | Lines | Chars | Words | Est. Tokens | % of Total | Type |
|---|---------|-------|-------|-------|-------------|------------|------|
| a | Role header | 1–3 | 85 | 9 | 21 | 4.5% | **static** |
| b | Current Objective | 5–7 | 80 | 7 | 20 | 4.3% | **dynamic** |
| c | Empathize Summary | 9–24 | 413 | 53 | 103 | 22.0% | **dynamic** |
| d | Latest Conversation | 26–31 | 948 | 140 | 237 | 50.5% | **dynamic** |
| e | Instructions | 33–39 | 347 | 54 | 86 | 18.5% | **static** |

### Static vs. Dynamic

| Balance | Chars | % | Est. Tokens |
|---------|-------|---|-------------|
| **Static** | 433 | **23.1%** | ~108 |
| **Dynamic** | 1,443 | **76.9%** | ~360 |

### Key Finding: Conversation History Dominates

Section **d (Latest Conversation)** takes **50.5%** — a single previous assistant turn + the new user message.  
Section **c (Empathize Summary)** takes **22.0%** — the structured summary built by Module 5's SummaryBuilder.

This prompt is already compact (~469 tokens). The only growth vector is conversation history accumulating over multiple turns.

---

## Side-by-Side Comparison

| Metric | MemoryExtractor | Mentor Response | Ratio |
|--------|----------------|----------------|-------|
| **Input chars** | 22,639 | 1,877 | 12.1x |
| **Input words** | 2,445 | 263 | 9.3x |
| **Input est. tokens** | 5,659 | 469 | 12.1x |
| **Output chars** | 557 | 837 | 0.7x |
| **Output words** | 64 | 123 | 0.5x |
| **Output est. tokens** | 139 | 209 | 0.7x |
| **Static %** | 93.5% | 23.1% | — |
| **Dynamic %** | 6.5% | 76.9% | — |
| **Lines** | 625 | 39 | 16.0x |
| **Output/Input ratio** | 2.5% | 44.5% | — |

## Largest Sections (Ranked)

| Rank | Section | Call | Size (chars) | % of that call |
|------|---------|------|-------------|----------------|
| 1 | Comprehensive Examples | MemoryExtractor | 13,563 | 59.9% |
| 2 | Field Definitions | MemoryExtractor | 3,299 | 14.6% |
| 3 | Critical Differentiation Rules | MemoryExtractor | 1,472 | 6.5% |
| 4 | Latest Conversation | Mentor Response | 948 | 50.5% |
| 5 | Strict Rules + Schema | MemoryExtractor | 720 | 3.2% |
| 6 | Previous Assistant Message | MemoryExtractor | 696 | 3.1% |
| 7 | Field Names Table | MemoryExtractor | 690 | 3.0% |
| 8 | CURRENT PROJECT STATE | MemoryExtractor | 564 | 2.5% |
| 9 | Operation Rules | MemoryExtractor | 492 | 2.2% |
| 10 | Message Type Definitions | MemoryExtractor | 484 | 2.1% |

## Largest Repeated Sections

### 1. Example JSON Output Template (repeated 20×)
Within the Comprehensive Examples section, the JSON output template:
```json
{
  "message_type": "MEANINGFUL",
  "updates": [
    {
      "operation": "ADD",
      "field": "...",
      "value": "...",
      "context": "...",
      "domain": "...",
      "keywords": [...],
      "raw_text": "...",
      "extraction_method": "llm"
    }
  ]
}
```
This template (or its `NO_UPDATE`/`AMBIGUOUS` variant) is repeated **20 times** with different values. The structural boilerplate alone accounts for roughly **7,000–8,000 chars** (~30% of the prompt).

### 2. Field Name References
- `"evidence"` appears 19× in static rules, 21× in examples, 2× in dynamic section
- `"pain_points"` appears 3× in rules, 13× in examples
- `"frequency"` appears 7× in rules, 9× in examples

### 3. Warning Decorators
The `━━━` separator lines and warning emoji/formatting blocks appear in every section header and many rule lines. Roughly **300–400 chars** of purely decorative formatting.

## Static vs. Dynamic Summary

### MemoryExtractor
```
Static  ██████████████████████████████████████████████████ 93.5%
Dynamic ████                                               6.5%
```

The MemoryExtractor prompt is overwhelmingly static. The per-turn variable data (project state + previous message + user message) is only **1,481 chars**.

### Mentor Response
```
Static  ████████████                                     23.1%
Dynamic ████████████████████████████████████████          76.9%
```

The Mentor Response prompt is mostly dynamic, driven by the conversation history and empathize summary.

## Implications

1. **MemoryExtractor is the prime optimization target.** Its static portion (21K chars, 93.5%) is sent on every turn. A prompt cache could eliminate this entirely.

2. **The Comprehensive Examples section is the largest sub-target.** 13.5K chars / 20 examples. Reducing to 5–8 high-quality examples would cut ~2,000–4,000 estimated tokens per turn with minimal quality impact.

3. **The Mentor Response prompt is already efficient.** At ~470 tokens, further optimization yields diminishing returns. The only growth vector is conversation history accumulation.

4. **The two prompts have opposite character profiles.** MemoryExtractor is instruction-heavy (tell the model *how* to think). Mentor Response is data-heavy (tell the model *what* to talk about).
