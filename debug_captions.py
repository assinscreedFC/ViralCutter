"""
Debug captions : vérifie les excerpts ET appelle Pléiade pour tester les captions.
Usage: python debug_captions.py
"""
import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from create_viral_segments import (
    load_transcript, preprocess_transcript_for_ai, _extract_excerpt,
    call_pleiade, clean_json_response_simple
)

PROJECT = r"VIRALS\Le pire duo va en prison #5 (ft Djilsi, Maxime Biaggi, Gotaga)"

# Segments du dernier run — mettre à jour après chaque run
# Charge depuis viral_segments.txt si présent, sinon fallback hardcodé
_seg_file = os.path.join(PROJECT, "viral_segments.txt")
if os.path.exists(_seg_file):
    with open(_seg_file, encoding="utf-8") as _f:
        _data = json.load(_f)
    SEGMENTS = [
        {
            "title": s["title"],
            "start_time": s["start_time"],
            "end_time": s["end_time"],
            "reasoning": s.get("reasoning", ""),
        }
        for s in _data.get("segments", [])
    ]
    print(f"[INFO] Chargé {len(SEGMENTS)} segments depuis viral_segments.txt")
else:
    SEGMENTS = [
        {"title": "Scène Choc : La Vérité sur le Lavabo",    "start_time": 115.567, "end_time": 196.122, "reasoning": ""},
        {"title": "Le rêve de Gotaga en politique",           "start_time": 2815.326, "end_time": 2857.326, "reasoning": ""},
    ]

# ── 1. Excerpts ────────────────────────────────────────────────────────────────
transcript_segments = load_transcript(PROJECT)
transcript_text = preprocess_transcript_for_ai(transcript_segments)

print(f"Transcript total: {len(transcript_text)} caractères\n")
print("=" * 60)
print("EXCERPTS")
print("=" * 60)

for seg in SEGMENTS:
    excerpt = _extract_excerpt(transcript_text, seg["start_time"], seg["end_time"])
    duration = seg["end_time"] - seg["start_time"]
    print(f"\n[{seg['title']}]")
    print(f"  {seg['start_time']}s → {seg['end_time']}s  (durée: {duration:.1f}s)")
    print(f"  excerpt ({len(excerpt)} chars): {excerpt[:300]}...")
    print()

# ── 2. Build caption prompt (tous les segments) ────────────────────────────────
with open(os.path.join("prompts", "tiktok_caption.txt"), encoding="utf-8") as f:
    caption_template = f.read()

segments_json = json.dumps([
    {
        "index": i,
        "title": seg["title"],
        "reasoning": seg.get("reasoning", ""),
        "duration": int(seg["end_time"] - seg["start_time"]),
        "transcript_excerpt": _extract_excerpt(transcript_text, seg["start_time"], seg["end_time"])
    }
    for i, seg in enumerate(SEGMENTS)
], ensure_ascii=False, indent=2)

prompt = caption_template.replace("{segments_json}", segments_json)

print("=" * 60)
print("PROMPT CAPTION (résumé):")
print(f"  Longueur: {len(prompt)} chars")
print(f"  Segments: {len(SEGMENTS)}")
print()

# ── 3. Appel Pléiade ───────────────────────────────────────────────────────────
print("=" * 60)
print("APPEL LLM PLÉIADE...")
print("=" * 60)

try:
    response = call_pleiade(prompt)
    # Extrait le think block séparément pour lisibilité
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    response_no_think = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    print(f"\nThink block ({len(think_text)} chars):\n{think_text[:600]}{'...' if len(think_text)>600 else ''}\n")
    print(f"Réponse (sans think, {len(response_no_think)} chars):\n{response_no_think[:600]}\n")

    data = clean_json_response_simple(response)
    captions = data.get("captions", [])

    print("=" * 60)
    print("CAPTIONS PARSÉES")
    print("=" * 60)

    if len(captions) < len(SEGMENTS):
        print(f"\n[WARN] LLM a généré {len(captions)} caption(s) pour {len(SEGMENTS)} segments — manquants: {[i for i in range(len(SEGMENTS)) if i not in {c.get('index') for c in captions}]}\n")

    for c in captions:
        idx = c.get("index", "?")
        caption = c.get("caption", "")
        seg_title = SEGMENTS[idx]["title"] if isinstance(idx, int) and idx < len(SEGMENTS) else "?"
        excerpt_full = _extract_excerpt(transcript_text, SEGMENTS[idx]["start_time"], SEGMENTS[idx]["end_time"]) if isinstance(idx, int) and idx < len(SEGMENTS) else ""
        excerpt_preview = excerpt_full[:150]

        print(f"\n[Seg {idx}] {seg_title}")
        print(f"  Titre (hint): {seg_title}")
        print(f"  Transcript:   {excerpt_preview}...")
        print(f"  Caption:      {caption}")
        print(f"  Longueur:     {len(caption)} chars")

        # Validation basique
        hashtags = re.findall(r'#\w+', caption)
        issues = []
        if len(caption) < 80:
            issues.append(f"trop courte ({len(caption)} < 80 chars)")
        if len(caption) > 200:
            issues.append(f"trop longue ({len(caption)} > 200 chars)")
        if len(hashtags) < 3:
            issues.append(f"pas assez de hashtags ({len(hashtags)} < 3)")
        if len(hashtags) > 5:
            issues.append(f"trop de hashtags ({len(hashtags)} > 5)")
        # Vérifie que caption n'utilise pas le titre mot-à-mot
        title_words = set(seg_title.lower().split()) - {"la", "le", "les", "de", "du", "en", "et", ":", "#"}
        caption_lower = caption.lower()
        title_reuse = [w for w in title_words if len(w) > 4 and w in caption_lower]

        # Grounding : vérifie combien de mots clés de la caption sont dans le transcript
        STOP_WORDS = {"le", "la", "les", "de", "du", "en", "et", "un", "une", "des", "il", "elle",
                      "ils", "elles", "je", "tu", "nous", "vous", "que", "qui", "mais", "pas", "ne",
                      "se", "son", "sa", "ses", "ce", "cette", "ces", "au", "aux", "sur", "dans", "par"}
        caption_words = set(re.sub(r'[^a-zàâéèêëîïôùûüç]', ' ', caption_lower).split()) - STOP_WORDS - {""}
        caption_words = {w for w in caption_words if len(w) > 3}
        excerpt_lower = excerpt_full.lower()
        grounded = [w for w in caption_words if w in excerpt_lower]
        ungrounded = [w for w in caption_words if w not in excerpt_lower and not w.startswith('#')]
        grounding_ratio = len(grounded) / max(len(caption_words), 1)

        if issues:
            print(f"  [WARN] {', '.join(issues)}")
        else:
            print(f"  [OK]   Format valide — {len(hashtags)} hashtags: {' '.join(hashtags)}")

        if grounding_ratio < 0.25:
            print(f"  [HALLUCINATION RISK] {grounding_ratio:.0%} mots ancrés dans transcript — mots suspects: {ungrounded[:6]}")
        else:
            print(f"  [GROUNDING] {grounding_ratio:.0%} ancrés ({grounded[:4]})")

        if title_reuse:
            print(f"  [WARN] Réutilise mots du titre: {title_reuse} — vérifier si justifié par transcript")

except Exception as e:
    print(f"[ERREUR] {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("FIN DEBUG")
