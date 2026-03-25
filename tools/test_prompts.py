"""
Test script : relance la detection virale + captions sur un dossier VIRALS existant.
Compare ancien vs nouveau prompt sans re-telecharger ni re-transcrire.

Usage : python test_prompts.py
"""
import json
import os
import sys
import time

# Setup path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

from scripts.analysis import create_viral_segments


def find_latest_project() -> str | None:
    """Trouve le dernier dossier dans VIRALS/ qui contient un input.tsv ou input.srt."""
    virals_dir = os.path.join(BASE_DIR, "VIRALS")
    if not os.path.isdir(virals_dir):
        print("[ERREUR] Dossier VIRALS/ introuvable.")
        return None

    projects = []
    for name in os.listdir(virals_dir):
        folder = os.path.join(virals_dir, name)
        if os.path.isdir(folder):
            has_transcript = os.path.exists(os.path.join(folder, "input.tsv")) or \
                             os.path.exists(os.path.join(folder, "input.srt"))
            if has_transcript:
                mtime = os.path.getmtime(folder)
                projects.append((mtime, name, folder))

    if not projects:
        print("[ERREUR] Aucun projet avec transcription trouve dans VIRALS/.")
        return None

    projects.sort(reverse=True)
    return projects[0][2]


def run_test(project_folder: str) -> None:
    project_name = os.path.basename(project_folder)
    print(f"\n{'='*70}")
    print(f"TEST : {project_name}")
    print(f"{'='*70}")

    # Sauvegarder les anciens resultats
    old_segments_path = os.path.join(project_folder, "viral_segments.txt")
    old_segments = None
    if os.path.exists(old_segments_path):
        with open(old_segments_path, "r", encoding="utf-8") as f:
            old_segments = json.load(f)
        print(f"\n[ANCIEN] {len(old_segments.get('segments', []))} segments trouves")
        for i, seg in enumerate(old_segments.get("segments", [])):
            t0 = seg.get("start_time", 0)
            t1 = seg.get("end_time", 0)
            score = seg.get("viral_score", seg.get("score", "?"))
            caption = seg.get("tiktok_caption", "")
            print(f"  [{i}] {t0:.0f}s-{t1:.0f}s (score={score}) {seg.get('title', '')[:60]}")
            if caption:
                print(f"       Caption: {caption[:100]}")

    # Lancer la detection virale avec les nouveaux prompts
    print(f"\n{'─'*70}")
    print("[NOUVEAU] Lancement de la detection virale...")
    start = time.time()

    result = create_viral_segments.create(
        num_segments=6,
        viral_mode=True,
        themes="",
        tempo_minimo=25,
        tempo_maximo=90,
        ai_mode="pleiade",
        project_folder=project_folder,
        enable_scoring=True,
        min_score=60
    )

    elapsed_detect = time.time() - start
    new_segments = result.get("segments", [])
    content_type = result.get("content_type", [])
    print(f"\n[RESULTAT] {len(new_segments)} segments en {elapsed_detect:.1f}s")
    print(f"[CONTENT TYPE] {content_type}")

    # Generer les captions
    if new_segments:
        print(f"\n[CAPTIONS] Generation des captions TikTok...")
        start = time.time()
        transcript = create_viral_segments.load_transcript(project_folder)
        transcript_text = create_viral_segments.preprocess_transcript_for_ai(transcript)
        new_segments = create_viral_segments.generate_tiktok_captions(
            new_segments, transcript_text,
            ai_mode="pleiade",
            content_type=content_type
        )
        elapsed_caption = time.time() - start
        print(f"[CAPTIONS] Done en {elapsed_caption:.1f}s")

    # Afficher les nouveaux resultats
    print(f"\n{'─'*70}")
    print("[NOUVEAUX SEGMENTS]")
    for i, seg in enumerate(new_segments):
        t0 = seg.get("start_time", 0)
        t1 = seg.get("end_time", 0)
        dur = t1 - t0
        score = seg.get("viral_score", seg.get("score", "?"))
        title = seg.get("title", "Sans titre")
        caption = seg.get("tiktok_caption", "")
        reasoning = seg.get("reasoning", "")

        print(f"\n  [{i}] {title}")
        print(f"      Temps: {t0:.0f}s - {t1:.0f}s ({dur:.0f}s)")
        print(f"      Score: {score}")
        print(f"      Raison: {reasoning[:120]}")
        print(f"      Caption: {caption}")

    # Analyse comparative
    print(f"\n{'='*70}")
    print("ANALYSE COMPARATIVE")
    print(f"{'='*70}")

    if old_segments:
        old_segs = old_segments.get("segments", [])
        print(f"\n  Ancien: {len(old_segs)} segments | Nouveau: {len(new_segments)} segments")

        # Distribution temporelle
        if old_segs:
            old_times = [(s.get("start_time", 0) + s.get("end_time", 0)) / 2 for s in old_segs]
            old_min, old_max = min(old_times), max(old_times)
            print(f"  Ancien — plage couverte: {old_min:.0f}s a {old_max:.0f}s (spread: {old_max - old_min:.0f}s)")

        if new_segments:
            new_times = [(s.get("start_time", 0) + s.get("end_time", 0)) / 2 for s in new_segments]
            new_min, new_max = min(new_times), max(new_times)
            print(f"  Nouveau — plage couverte: {new_min:.0f}s a {new_max:.0f}s (spread: {new_max - new_min:.0f}s)")

        # Qualite captions
        print(f"\n  --- Captions ---")
        ai_words = ["palpable", "authentique", "touchant", "entre rires", "face aux critiques",
                     "un moment", "le créateur", "le protagoniste", "est palpable"]
        for label, segs in [("ANCIEN", old_segs), ("NOUVEAU", new_segments)]:
            captions = [s.get("tiktok_caption", "") for s in segs]
            has_fyp = sum(1 for c in captions if "#fyp" in c.lower())
            has_pourtoi = sum(1 for c in captions if "#pourtoi" in c.lower())
            ai_count = sum(1 for c in captions for w in ai_words if w.lower() in c.lower())
            print(f"  {label}: #fyp={has_fyp}/{len(captions)}, #pourtoi={has_pourtoi}/{len(captions)}, mots-IA detectes={ai_count}")
    else:
        print("  Pas d'anciens segments pour comparer.")

    # Sauvegarder les nouveaux resultats dans un fichier separe (ne pas ecraser)
    test_output = os.path.join(project_folder, "test_new_segments.json")
    with open(test_output, "w", encoding="utf-8") as f:
        json.dump({"segments": new_segments, "content_type": content_type}, f, ensure_ascii=False, indent=2)
    print(f"\n[SAUVEGARDE] Resultats sauves dans: {test_output}")
    print("(Les anciens segments ne sont PAS ecrases)")


if __name__ == "__main__":
    project = find_latest_project()
    if project:
        run_test(project)
