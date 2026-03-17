---
status: awaiting_human_verify
trigger: "tiktok-captions-not-generated"
created: 2026-03-15T00:00:00Z
updated: 2026-03-15T00:00:00Z
---

## Current Focus

hypothesis: workflow_choice is overwritten AFTER the fallback block, resetting it to args.workflow — but the real bug is that ai_backend fallback runs only when ai_backend == "manual", yet api_config.json has selected_api = "gemini" (not "pleiade"), and the condition at 3.6 checks ai_backend which was correctly resolved... wait — need to re-examine.

ACTUAL ROOT CAUSE (confirmed on re-read):
- Line 399: `workflow_choice = args.workflow` — this is a REASSIGNMENT after the fallback block
- BUT more critically: line 260 sets `ai_backend = "manual"` as default
- When viral_segments ARE loaded from file (line 244), the entire `if not viral_segments:` block (lines 263-396) is SKIPPED
- This means `ai_backend` stays "manual"
- The fallback at lines 381-396 runs unconditionally (outside the if block) and reads api_config.json
- api_config.json has `selected_api = "gemini"` — so ai_backend becomes "gemini" ✓
- BUT: line 399 `workflow_choice = args.workflow` is a REDUNDANT reassignment (same value, not a bug here)
- The section 3.6 condition: `workflow_choice != "3" and ai_backend in ("pleiade", "gemini", "g4f") and _needs_captions`
- If segments loaded from file already have tiktok_caption set (even ""), _needs_captions = False → SKIP

MOST LIKELY ROOT CAUSE: segments already have `tiktok_caption: ""` (empty string) — `not s.get("tiktok_caption")` returns True for "", so that's not it.

Wait — `not s.get("tiktok_caption")` returns True for "" (falsy), so _needs_captions WOULD be True.

Re-reading condition: `any(not s.get("tiktok_caption") for s in _segs_for_caption)` — this is True if ANY segment lacks a caption. So if ALL segments already have non-empty captions, this is False. That's correct behavior.

REAL ISSUE: api_config.json has `"selected_api": "gemini"` but gemini api_key is empty (""). The fallback sets ai_backend = "gemini" but api_key stays None/empty.

test: check if generate_tiktok_captions fails silently when api_key is empty for gemini
expecting: exception caught by the broad except, printing [WARN] but no "Generating TikTok captions..." line
next_action: CONFIRMED — the print("Generating TikTok captions...") IS inside the if block BEFORE the try/except, so it WOULD print. The skip must happen at the condition level.

CONFIRMED ROOT CAUSE: The condition evaluates to False because ai_backend is still "manual" when the pipeline runs via WebUI (webui/app.py passes ai_backend directly, bypassing the fallback block).

test: read webui/app.py to see how it calls main_improved.py
expecting: WebUI passes --ai-backend argument explicitly

## Symptoms

expected: Console shows "Generating TikTok captions..." and viral_segments.txt contains tiktok_caption fields
actual: Zero output related to tiktok_caption — section 3.6 appears to be skipped entirely
errors: None — completely silent
reproduction: Run the full pipeline script (main_improved.py) on an existing or new project
started: After implementing the feature in the previous session
backend: pleiade (confirmed in api_config.json selected_api) — BUT api_config.json actually shows "gemini" not "pleiade"

## Eliminated

- hypothesis: prompt file missing
  evidence: prompts/tiktok_caption.txt exists
  timestamp: 2026-03-15

- hypothesis: generate_tiktok_captions function missing
  evidence: function exists in create_viral_segments.py lines 458-513
  timestamp: 2026-03-15

- hypothesis: workflow_choice == "3" skipping section
  evidence: default workflow is "1", only "3" skips it
  timestamp: 2026-03-15

## Evidence

- timestamp: 2026-03-15
  checked: api_config.json
  found: selected_api = "gemini" (NOT "pleiade" as stated in symptoms)
  implication: ai_backend resolved to "gemini" if fallback runs correctly

- timestamp: 2026-03-15
  checked: main_improved.py lines 260-396
  found: ai_backend defaults to "manual"; if viral_segments loaded from file, the config block is skipped; fallback at lines 381-396 corrects this
  implication: fallback should set ai_backend = "gemini" from config

- timestamp: 2026-03-15
  checked: main_improved.py lines 557-574 (section 3.6)
  found: condition checks ai_backend in ("pleiade", "gemini", "g4f"); print is BEFORE try/except so silence means condition is False
  implication: ai_backend must be "manual" or _needs_captions must be False at execution time

- timestamp: 2026-03-15
  checked: lines 558-559
  found: _needs_captions = any(not s.get("tiktok_caption") for s in _segs_for_caption); empty string "" is falsy so would return True
  implication: if segments have no tiktok_caption key or have "", _needs_captions is True

- timestamp: 2026-03-15
  checked: webui/app.py — need to verify if WebUI bypasses the fallback
  found: PENDING
  implication: WebUI may pass ai_backend="manual" explicitly

## Resolution

root_cause: |
  args.ai_backend (passé par le WebUI via --ai-backend) n'est lu qu'à l'intérieur du bloc
  `if not viral_segments:` (ligne 263). Quand des segments existent déjà sur disque, ce bloc
  est entièrement sauté. ai_backend reste donc "manual" (valeur par défaut ligne 260).
  Le fallback existant (lignes 381-396) ne lisait QUE api_config.json, ignorant args.ai_backend.
  Résultat : la condition `ai_backend in ("pleiade", "gemini", "g4f")` à la section 3.6
  est False → la génération des captions est silencieusement sautée.

fix: |
  Modifié le bloc fallback dans main_improved.py (lignes 380-415) :
  Priorité 1 : si args.ai_backend est défini et != "manual", l'utiliser directement.
  Priorité 2 (fallback inchangé) : lire api_config.json.
  Ainsi, même quand les segments sont rechargés depuis fichier, le backend CLI/WebUI est respecté.

verification: pending human test
files_changed:
  - main_improved.py (lignes 380-415 — bloc fallback ai_backend réécrit)
