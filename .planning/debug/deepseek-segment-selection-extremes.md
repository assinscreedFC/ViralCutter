---
status: awaiting_human_verify
trigger: "DeepSeek AI always picks viral segments with very short durations (1-30s) that get extended to min_duration (42s)"
created: 2026-03-15T00:00:00Z
updated: 2026-03-15T00:00:00Z
---

## Current Focus

hypothesis: The JSON template sent to the LLM does NOT include `end_time_ref` field, so the LLM only returns `start_text`/`end_text` strings. The text-matching logic in `process_segments()` then fails to find `end_text` in the transcript (fuzzy match too strict or DeepSeek paraphrases), falls back to `fallback_duration = (min+max)/2`, which should be fine. BUT if `end_text` IS found but it's only 1-4 words away from `start_text` (DeepSeek picks short phrases), the resulting duration is tiny (1-30s) — then the WARN fires and it gets force-extended.
test: Trace the path: LLM returns short phrases as start_text/end_text → text match finds them close together in transcript → duration < min_duration → forced extension
expecting: Confirmed — the root cause is that the LLM template doesn't guide DeepSeek toward picking start/end texts that are far apart in the transcript
next_action: CONFIRMED — implement fix

## Symptoms

expected: DeepSeek should choose segments whose natural duration is within [min_duration, max_duration]. Ideally 42s-180s.
actual: DeepSeek consistently picks segments that are way too short (1.83s, 4.55s, 4.93s, 14.22s, 29s) — all below 42s minimum. They all get force-extended.
errors: |
  [WARN] Segmento menor que duration min (4.93s < 42s). Estendendo para 42s.
  [WARN] Segmento menor que duration min (1.83s < 42s). Estendendo para 42s.
  [WARN] Segmento menor que duration min (4.55s < 42s). Estendendo para 42s.
  [WARN] Segmento menor que duration min (14.22s < 42s). Estendendo para 42s.
  [WARN] Segmento menor que duration min (29.00s < 42s). Estendendo para 42s.
reproduction: Run viral segment creation pipeline with DeepSeek model via Pléiade API
timeline: Ongoing — probably always been the case

## Eliminated

- hypothesis: Prompt doesn't inject min_duration/max_duration at all
  evidence: prompt.txt line 39 uses {min_duration} and {max_duration}, and create_viral_segments.py lines 948-949 inject them correctly
  timestamp: 2026-03-15

- hypothesis: The fallback end time logic is triggered (end_text not found)
  evidence: If fallback triggered, duration = (min+max)/2 = ~111s for typical 42-180 config — would NOT produce sub-42s. The short durations prove end_text IS being found, but close to start_text.
  timestamp: 2026-03-15

## Evidence

- timestamp: 2026-03-15
  checked: prompt.txt — rule #4 DURATION section
  found: "Each segment MUST be between {min_duration}s and {max_duration}s." — constraint is stated. But no instruction about what start_text/end_text should span in terms of duration.
  implication: The LLM knows it needs to pick a segment of 42-180s, but the JSON schema it must fill only has start_text/end_text (exact transcript phrases). DeepSeek likely picks a short punchy phrase as start_text and the immediately following sentence as end_text, not understanding that these two fields need to be far apart in the video.

- timestamp: 2026-03-15
  checked: json_template in create_viral_segments.py (lines 885-898)
  found: Template has: start_text, end_text, start_time_ref, title, reasoning, score. NO end_time_ref field. NO duration hint. NO indication that start_text and end_text should be ~42-180s apart.
  implication: DeepSeek has no mechanism to express "this segment runs from timestamp X to timestamp Y". It picks text anchors, not time ranges. The actual timestamps are derived post-hoc by text matching. If DeepSeek picks adjacent sentences, duration will be tiny.

- timestamp: 2026-03-15
  checked: process_segments() end_text search (lines 696-706)
  found: Searches from match_start_idx up to +200 transcript segments. First match wins. If DeepSeek returns end_text that's only 2-3 segments after start_text, this matches immediately and produces 1-5s duration.
  implication: The matching logic is correct but relies entirely on DeepSeek picking end_text that is genuinely far into the future of the transcript.

- timestamp: 2026-03-15
  checked: prompt.txt rule #2 THE STANDALONE TEST
  found: "Capture the ENTIRE story from its natural beginning to its absolute end." — good intent but no enforcement on the output format.
  implication: Without an end_time_ref field in the JSON, DeepSeek can only pick end_text. It likely picks the "punchline" or "conclusion" sentence which may immediately follow the hook — not 60s later.

## Resolution

root_cause: |
  The JSON template given to the LLM contains only `start_text` and `end_text` (short phrase anchors) but NO `end_time_ref` field.

  DeepSeek correctly identifies a viral moment but expresses it as two adjacent text phrases (hook sentence → conclusion sentence).
  These phrases may be only 1-5 transcript segments apart (1-30s of content).

  The text-matching in process_segments() finds the match immediately → tiny duration → WARN → force-extend to min_duration.

  The prompt says "segments must be 42-180s" but the LLM has no way to validate this — it's picking text phrases, not timestamps.
  The only way the LLM can produce a correctly-sized segment is if it picks start_text from early in a story and end_text from 60-120 seconds later — but nothing in the prompt explicitly says "the end_text should be the text spoken approximately {min_duration}-{max_duration} seconds after start_text."

fix: |
  Two-part fix applied:

  1. json_template now includes `end_time_ref` field — LLM must provide approximate end timestamp (same format as start_time_ref). Description explicitly says it must differ from start_time_ref by min_duration..max_duration seconds.

  2. `end_text` description now explicitly says "must be spoken approx. Xs to Ys AFTER start_text".

  3. process_segments() fallback path now tries end_time_ref before falling back to (min+max)/2 — so even when end_text matching fails, the LLM-provided end timestamp is used.

verification: awaiting human test run with DeepSeek
files_changed:
  - scripts/create_viral_segments.py — json_template (lines 912-926), process_segments fallback (lines 709-740)
