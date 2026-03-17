"""
=============================================================================
PROMPT SECTIONS — Sections adaptatives pour le prompt LLM enrichi
=============================================================================
Chaque section est injectée dans prompt_enhanced.txt UNIQUEMENT quand la
feature correspondante est activée. Quand tout est désactivé, le prompt
se comporte exactement comme prompt.txt original.

Usage dans create_viral_segments.py :
    from scripts.prompt_sections import build_enhanced_prompt
    prompt = build_enhanced_prompt(
        enable_zoom=True,
        enable_power_words=True,
        enable_music=True,
        enable_jumpcuts=True,
        music_files=["chill_beat.mp3", "energy_pop.mp3"]
    )
=============================================================================
"""

import os


# =============================================================================
# SECTION : ZOOM DYNAMIQUE
# =============================================================================
ZOOM_RULES = """
### DYNAMIC ZOOM CUES
You must identify **punchline moments** for dynamic zoom effects.

**BEST PRACTICES (DO THIS):**
- Zoom on the FINAL word of a punchline or shocking revelation
- Zoom on emotional peaks (surprise, laughter, anger)
- Maximum 3 zoom cues per 60-second clip
- Zoom intensity between 1.05x and 1.15x

**ANTI-PATTERNS (NEVER DO THIS):**
- Do NOT zoom on boring transitions or filler moments
- Do NOT zoom more than once every 3 seconds (cooldown)
- Do NOT zoom during camera movement or face tracking transitions
- Do NOT zoom on every sentence — it loses all impact

For each segment, provide a `zoom_cues` array:
```json
"zoom_cues": [
    {"timestamp": 138.5, "intensity": 1.1, "duration": 1.5, "reason": "punchline"}
]
```
"""

# =============================================================================
# SECTION : POWER WORDS (Sous-titres colorés)
# =============================================================================
POWER_WORDS_RULES = """
### POWER WORDS (Colored Subtitles)
You must tag important words with semantic categories for colored highlighting.

**CATEGORIES:**
- `"importance"` -> Yellow highlight — Numbers, key facts, attention-grabbing words
- `"success"` -> Green highlight — Money, wins, achievements, positive outcomes
- `"danger"` -> Red highlight — Risks, failures, warnings, negative outcomes

**BEST PRACTICES (DO THIS):**
- Tag 5-8 power words MAX per clip (less is more)
- Focus on: numbers/amounts, action verbs, strong adjectives
- Tag words that make viewers stop scrolling

**ANTI-PATTERNS (NEVER DO THIS):**
- Do NOT tag filler words (the, and, but, so...)
- Do NOT tag more than 2 words per sentence
- Do NOT use the same color for consecutive words (alternate for visual variety)
- Do NOT tag every noun — only words with EMOTIONAL weight

For each segment, provide a `power_words` array:
```json
"power_words": [
    {"word": "million", "category": "success"},
    {"word": "crashed", "category": "danger"},
    {"word": "secret", "category": "importance"}
]
```
"""

# =============================================================================
# SECTION : MUSIQUE DE FOND
# =============================================================================
MUSIC_RULES_TEMPLATE = """
### BACKGROUND MUSIC SELECTION
You must suggest a music mood AND select the best track from available files.

**AVAILABLE MUSIC FILES:**
{music_list}

**MOOD OPTIONS:** energetic, chill, dramatic, funny, suspenseful, inspirational

**BEST PRACTICES (DO THIS):**
- Match energy: funny clips -> upbeat/energetic, emotional clips -> dramatic/chill
- Instrumental only (no lyrics that compete with speech)
- TikTok-style beats work best for clip virality

**ANTI-PATTERNS (NEVER DO THIS):**
- Do NOT choose sad music for comedy content
- Do NOT choose aggressive music for wholesome content
- Do NOT choose music with prominent vocals

For each segment, provide:
```json
"music_mood": "energetic",
"music_file": "chill_beat.mp3"
```
If no file matches the mood well, set `"music_file": null` and only suggest the mood.
"""

# =============================================================================
# SECTION : JUMPCUTS
# =============================================================================
JUMPCUT_RULES = """
### JUMPCUT HINTS
Identify pauses that should be PRESERVED (not cut) for dramatic effect.

**BEST PRACTICES (DO THIS):**
- Flag dramatic pauses before reveals: "And the winner is... [PAUSE] ... him!"
- Flag reaction pauses where facial expression tells the story
- Flag comedic timing pauses (the beat before a punchline)

**ANTI-PATTERNS (NEVER DO THIS):**
- Do NOT flag boring dead air or awkward silences
- Do NOT flag transitions between topics
- Do NOT flag more than 2 preserved pauses per clip

For each segment, provide optional `jumpcut_hints`:
```json
"jumpcut_hints": ["keep_pause_at_142.5s", "keep_pause_at_155.0s"]
```
If no important pauses exist, set `"jumpcut_hints": []`.
"""


def build_enhanced_prompt(
    enable_zoom: bool = False,
    enable_power_words: bool = False,
    enable_music: bool = False,
    enable_jumpcuts: bool = False,
    music_files: list[str] | None = None
) -> str:
    """
    Construit le prompt adaptatif en injectant uniquement les sections
    correspondant aux features activées.
    """
    # --- Charger le template ---
    prompt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(prompt_dir, "prompt_enhanced.txt")

    # Fallback au prompt classique si le template n'existe pas
    if not os.path.exists(template_path):
        template_path = os.path.join(prompt_dir, "prompt.txt")
        print("[PromptBuilder] WARNING: prompt_enhanced.txt not found, using prompt.txt")
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # --- Injecter les sections conditionnelles ---
    zoom_section = ZOOM_RULES if enable_zoom else ""
    power_words_section = POWER_WORDS_RULES if enable_power_words else ""
    jumpcut_section = JUMPCUT_RULES if enable_jumpcuts else ""

    # Section musique avec la liste des fichiers
    if enable_music and music_files:
        music_list = "\n".join(f"- {f}" for f in music_files)
        music_section = MUSIC_RULES_TEMPLATE.replace("{music_list}", music_list)
    elif enable_music:
        music_section = MUSIC_RULES_TEMPLATE.replace(
            "{music_list}",
            "No music files available. Suggest mood only."
        )
    else:
        music_section = ""

    # --- Construire les instructions additionnelles ---
    enhanced_parts = []
    if enable_zoom:
        enhanced_parts.append("Include `zoom_cues` for each segment.")
    if enable_power_words:
        enhanced_parts.append("Include `power_words` for each segment.")
    if enable_music:
        enhanced_parts.append("Include `music_mood` and `music_file` for each segment.")
    if enable_jumpcuts:
        enhanced_parts.append("Include `jumpcut_hints` for each segment.")

    enhanced_instructions = ""
    if enhanced_parts:
        enhanced_instructions = (
            "\n**ADDITIONAL PRODUCTION METADATA:**\n"
            + "\n".join(f"- {p}" for p in enhanced_parts)
        )

    # --- Remplacer les placeholders ---
    prompt = template.replace("{zoom_rules}", zoom_section)
    prompt = prompt.replace("{power_words_rules}", power_words_section)
    prompt = prompt.replace("{music_rules}", music_section)
    prompt = prompt.replace("{jumpcut_rules}", jumpcut_section)
    prompt = prompt.replace("{enhanced_instructions}", enhanced_instructions)

    # Nettoyer les lignes vides multiples
    while "\n\n\n" in prompt:
        prompt = prompt.replace("\n\n\n", "\n\n")

    return prompt
