"""
Benchmark des modèles Pléiade pour ViralCutter.
Utilise un vrai transcript depuis VIRALS/ ou le transcript de test embarqué.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --project "VIRALS/nom_du_projet"
    python scripts/benchmark_models.py --models athene-v2:latest qwen3:30b
    python scripts/benchmark_models.py --apply
"""
import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

# Charger .env depuis la racine du projet
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Ajouter scripts/ au path pour importer create_viral_segments
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))
from create_viral_segments import (  # noqa: E402
    call_pleiade,
    clean_json_response,
    load_transcript,
    preprocess_transcript_for_ai,
)

import requests  # noqa: E402

# ─────────────────────────── Transcript fallback ──────────────────────────────
# Utilisé uniquement si aucun vrai projet n'est disponible
FALLBACK_TRANSCRIPT = """(0s) So I quit my job today. (4s) After 8 years. (6s) Just walked in and said I'm done. (11s) \
My boss looked at me like I had two heads. (15s) But here's the thing, (18s) \
last quarter I generated 4 million in revenue for the company. (23s) \
And my bonus? (26s) Zero. (28s) They said the targets weren't met at the company level. (33s) \
Company level. (35s) Meanwhile the CEO just bought a third yacht. (40s) \
So yeah, I walked out. (43s) And you know what happened next? (46s) \
Three clients followed me. (49s) They called me personally within 48 hours. (53s) \
That's 2.1 million in annual recurring revenue. (58s) That I now keep. (61s) \
People always say don't burn bridges. (65s) But what if the bridge was already on fire (69s) \
and you were the only one keeping it standing? (73s) \
The moment I stopped being afraid of losing my job (78s) \
was the moment I realized I was the job. (83s)"""

# Prompt minimal de test (ne dépend pas de prompt.txt)
TEST_PROMPT_TEMPLATE = """You are a viral video editor. Analyze this transcript and find 1-2 viral segments.

TRANSCRIPT:
{transcript}

OUTPUT FORMAT (JSON only, no explanation):
{{
  "segments": [
    {{
      "start_text": "exact first words at segment start",
      "end_text": "exact last words at segment end",
      "start_time_ref": 0,
      "end_time_ref": 0,
      "title": "viral title",
      "reasoning": "why this is viral",
      "score": 85
    }}
  ]
}}

Rules:
- start_time_ref and end_time_ref must be integers from the (Xs) tags
- end_time_ref - start_time_ref must be between 15 and 90
- start_text and end_text must be different phrases from different moments
- score: 90+ exceptional, 75-89 strong, 60-74 decent
"""

REQUIRED_FIELDS = {"start_text", "end_text", "start_time_ref", "end_time_ref", "title", "reasoning", "score"}

# ─────────────────────────── Chargement transcript réel ───────────────────────

def find_first_virals_project() -> str | None:
    """Retourne le premier dossier VIRALS/ contenant un input.tsv ou input.srt."""
    virals_dir = os.path.join(ROOT_DIR, "VIRALS")
    if not os.path.isdir(virals_dir):
        return None
    for entry in sorted(os.listdir(virals_dir)):
        full = os.path.join(virals_dir, entry)
        if os.path.isdir(full):
            if os.path.exists(os.path.join(full, "input.tsv")) or os.path.exists(os.path.join(full, "input.srt")):
                return full
    return None


def load_real_transcript(project_folder: str, max_chars: int = 10000) -> tuple[str, str]:
    """Charge et formate un transcript réel. Retourne (transcript_str, source_label)."""
    segments = load_transcript(project_folder)
    if not segments:
        return "", ""
    text = preprocess_transcript_for_ai(segments)
    if len(text) > max_chars:
        # Tronquer proprement au dernier tag de temps dans la limite
        cut = text[:max_chars]
        last_tag = cut.rfind("(")
        if last_tag > 0:
            cut = cut[:last_tag].rstrip()
        text = cut
    label = os.path.basename(project_folder)[:50]
    return text, label

# ─────────────────────────── Découverte des modèles ───────────────────────────

def fetch_available_models() -> list[str]:
    """Récupère la liste des modèles disponibles via /api/models (endpoint OpenWebUI)."""
    api_url = os.getenv("PLEIADE_API_URL", "").rstrip("/")
    api_key = os.getenv("PLEIADE_API_KEY", "")
    if not api_url or not api_key:
        return []
    try:
        resp = requests.get(
            f"{api_url}/api/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        return [m["id"] for m in models if "id" in m]
    except Exception as e:
        print(f"[WARN] Impossible de récupérer la liste des modèles: {e}")
        return []

# ─────────────────────────── Évaluation qualité ───────────────────────────────

def evaluate_response(raw: str) -> dict:
    """Évalue la qualité d'une réponse brute.

    Score qualité = JSON valide (30) + schéma complet (30) + durée cohérente (20) + score distribution (20)
    """
    result = {
        "json_valid": False,
        "segment_count": 0,
        "schema_score": 0,
        "duration_ok": False,
        "score_dist": 0,
        "quality_score": 0,
        "error": None,
    }

    parsed = clean_json_response(raw)
    if not parsed or not isinstance(parsed, dict):
        result["error"] = "Réponse vide ou non-JSON"
        return result

    segments = parsed.get("segments", [])
    if not segments:
        result["error"] = "Aucun segment dans la réponse"
        return result

    result["json_valid"] = True
    result["segment_count"] = len(segments)

    # Schéma (30 pts) : champs présents dans au moins un segment
    fields_found: set = set()
    for seg in segments:
        if isinstance(seg, dict):
            fields_found |= set(seg.keys())
    result["schema_score"] = int(30 * len(REQUIRED_FIELDS & fields_found) / len(REQUIRED_FIELDS))

    # Durée cohérente (20 pts) : au moins un segment dans [15, 120]
    for seg in segments:
        if isinstance(seg, dict):
            try:
                gap = int(seg.get("end_time_ref", 0)) - int(seg.get("start_time_ref", 0))
                if 15 <= gap <= 120:
                    result["duration_ok"] = True
                    break
            except (ValueError, TypeError):
                pass

    # Distribution scores LLM (20 pts)
    llm_scores = [seg.get("score", 0) for seg in segments if isinstance(seg, dict) and "score" in seg]
    if llm_scores:
        avg = sum(llm_scores) / len(llm_scores)
        varied = len(set(llm_scores)) > 1 or len(llm_scores) == 1
        if avg >= 60 and varied:
            result["score_dist"] = 20
        elif avg >= 60 or varied:
            result["score_dist"] = 10

    json_pts = 30 if result["json_valid"] else 0
    duration_pts = 20 if result["duration_ok"] else 0
    result["quality_score"] = json_pts + result["schema_score"] + duration_pts + result["score_dist"]
    return result


# ─────────────────────────── Benchmark principal ──────────────────────────────

def run_benchmark(models: list[str], transcript: str) -> list[dict]:
    prompt = TEST_PROMPT_TEMPLATE.format(transcript=transcript)
    results = []

    for model in models:
        print(f"  → {model} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            raw = call_pleiade(prompt, model_name=model)
            elapsed = round(time.time() - t0, 1)
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            results.append({"model": model, "elapsed": elapsed, "quality_score": 0, "error": str(e)})
            print(f"ERREUR ({e})")
            continue

        metrics = evaluate_response(raw)
        metrics["model"] = model
        metrics["elapsed"] = elapsed
        results.append(metrics)

        status = "✓" if metrics["json_valid"] else "✗"
        print(f"{status}  qualité={metrics['quality_score']}/100  temps={elapsed}s")

    return results


def print_table(results: list[dict]) -> None:
    print("\n" + "─" * 72)
    print(f"  {'Modèle':<27} {'Qualité':>8} {'JSON':>6} {'Schéma':>8} {'Durée':>7} {'Temps':>8}")
    print("─" * 72)
    for r in sorted(results, key=lambda x: x.get("quality_score", 0), reverse=True):
        valid = "✅" if r.get("json_valid") else "❌"
        dur = "✅" if r.get("duration_ok") else "❌"
        schema = f"{r.get('schema_score', 0)}/30"
        print(
            f"  {r['model']:<27} {r.get('quality_score', 0):>7}/100"
            f"  {valid}  {schema:>8}  {dur}   {r.get('elapsed', 0):>6.1f}s"
        )
    print("─" * 72)


def pick_winner(results: list[dict]) -> dict | None:
    valid = [r for r in results if r.get("json_valid")]
    if not valid:
        return None
    return max(valid, key=lambda r: r["quality_score"])


def apply_config(model: str) -> None:
    """Met à jour .env et api_config.json avec le modèle gagnant."""
    env_path = os.path.join(ROOT_DIR, ".env")
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()
    lines = [
        f"PLEIADE_CHAT_MODEL={model}" if l.startswith("PLEIADE_CHAT_MODEL=") else l
        for l in lines
    ]
    if not any(l.startswith("PLEIADE_CHAT_MODEL=") for l in content.splitlines()):
        lines.append(f"PLEIADE_CHAT_MODEL={model}")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    cfg_path = os.path.join(ROOT_DIR, "api_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["selected_api"] = "pleiade"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Configuration appliquée : {model}")
    print(f"   .env              → PLEIADE_CHAT_MODEL={model}")
    print(f"   api_config.json   → selected_api=pleiade")


# ─────────────────────────── Point d'entrée ───────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark modèles Pléiade pour ViralCutter")
    parser.add_argument("--models", nargs="+", help="Modèles à tester (défaut: auto-détecté)")
    parser.add_argument("--project", help="Dossier projet VIRALS à utiliser pour le transcript")
    parser.add_argument("--apply", action="store_true", help="Appliquer le gagnant sans confirmation")
    args = parser.parse_args()

    print("=" * 72)
    print("         BENCHMARK MODÈLES PLÉIADE — ViralCutter")
    print("=" * 72)

    # ── Chargement du transcript ────────────────────────────────────────────
    transcript = ""
    source_label = ""

    if args.project:
        project_path = args.project if os.path.isabs(args.project) else os.path.join(ROOT_DIR, args.project)
        transcript, source_label = load_real_transcript(project_path)

    if not transcript:
        project_path = find_first_virals_project()
        if project_path:
            transcript, source_label = load_real_transcript(project_path)

    if not transcript:
        print("[WARN] Aucun projet VIRALS trouvé, utilisation du transcript de test.")
        transcript = FALLBACK_TRANSCRIPT
        source_label = "transcript de test (embarqué)"

    print(f"\nTranscript : {source_label}")
    print(f"Taille     : {len(transcript)} chars")

    # ── Sélection des modèles ───────────────────────────────────────────────
    if args.models:
        models = args.models
    else:
        print("\nRécupération des modèles disponibles sur Pléiade...")
        models = fetch_available_models()
        if models:
            print(f"Modèles trouvés ({len(models)}) : {', '.join(models)}")
        else:
            print("Impossible de récupérer la liste. Utilisation des modèles connus.")
            models = ["athene-v2:latest", "Qwen3:30b", "deepseek-R1:70b"]

    # ── Benchmark ───────────────────────────────────────────────────────────
    print(f"\nTests en cours ({len(models)} modèles) :")
    results = run_benchmark(models, transcript)
    print_table(results)

    winner = pick_winner(results)
    if not winner:
        print("\n❌ Aucun modèle n'a retourné du JSON valide.")
        return

    print(f"\n🏆 Gagnant (qualité prioritaire) : {winner['model']}")
    print(f"   Score qualité : {winner['quality_score']}/100  |  Temps : {winner['elapsed']}s")

    if args.apply:
        apply_config(winner["model"])
    else:
        answer = input("\nAppliquer cette configuration ? [o/N] ").strip().lower()
        if answer in ("o", "oui", "y", "yes"):
            apply_config(winner["model"])
        else:
            print("Configuration inchangée.")


if __name__ == "__main__":
    main()
