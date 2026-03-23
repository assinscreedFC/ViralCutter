"""
=============================================================================
ADD MUSIC — Ajout de musique de fond aux clips viraux
=============================================================================
Sélection intelligente basée sur le mood LLM du segment :
  1. Cache metadata (BPM + énergie) scanné par librosa → music/metadata.json
  2. Matching mood → profil BPM/énergie → meilleure piste par clip
  3. Auto-fetch via fetch_music si le dossier est vide/périmé
  4. Mix ffmpeg : volume 12%, fade in 1s, fade out 2s

Moods reconnus (générés par le LLM via MUSIC_RULES_TEMPLATE) :
    energetic, upbeat, dramatic, chill, sad, motivational, neutral
=============================================================================
"""

import logging
import os
from scripts.run_cmd import run as run_cmd
import json
import random
import time

logger = logging.getLogger(__name__)

try:
    from scripts.fetch_music import fetch_music, DEFAULT_MUSIC_DIR, needs_refresh
except ImportError:
    try:
        from fetch_music import fetch_music, DEFAULT_MUSIC_DIR, needs_refresh
    except ImportError:
        fetch_music = None
        DEFAULT_MUSIC_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "music"
        )
        needs_refresh = None

try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# Volume de la musique de fond (12% du mix)
MUSIC_VOLUME = 0.12
# Durée de fade out en secondes
FADE_OUT_DURATION = 2.0
# Nom du fichier de cache metadata
METADATA_CACHE_FILE = "metadata.json"
# Revalider le cache si plus vieux que N jours
METADATA_CACHE_MAX_AGE_DAYS = 7

# Profils mood → (bpm_target, bpm_tolerance, energy_level)
# energy_level: "high" > 0.06, "medium" 0.03-0.06, "low" < 0.03
MOOD_PROFILES: dict[str, dict] = {
    "energetic":    {"bpm": 130, "bpm_tol": 20, "energy": "high"},
    "upbeat":       {"bpm": 115, "bpm_tol": 20, "energy": "medium"},
    "dramatic":     {"bpm": 100, "bpm_tol": 25, "energy": "high"},
    "motivational": {"bpm": 120, "bpm_tol": 20, "energy": "high"},
    "neutral":      {"bpm": 110, "bpm_tol": 30, "energy": "medium"},
    "chill":        {"bpm": 80,  "bpm_tol": 20, "energy": "low"},
    "sad":          {"bpm": 70,  "bpm_tol": 20, "energy": "low"},
}

ENERGY_THRESHOLDS = {"high": 0.06, "medium": 0.03, "low": 0.0}


# ---------------------------------------------------------------------------
# Metadata cache
# ---------------------------------------------------------------------------

def _cache_path(music_dir: str) -> str:
    return os.path.join(music_dir, METADATA_CACHE_FILE)


def _audio_files_in(music_dir: str) -> list[str]:
    exts = (".mp3", ".wav", ".ogg", ".m4a", ".flac")
    return [
        f for f in os.listdir(music_dir)
        if f.lower().endswith(exts) and f != METADATA_CACHE_FILE
    ] if os.path.exists(music_dir) else []


def load_music_metadata_cache(music_dir: str) -> dict:
    """Charge le cache metadata depuis music/metadata.json."""
    path = _cache_path(music_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_music_metadata_cache(music_dir: str, force: bool = False) -> dict:
    """
    Scanne tous les fichiers audio avec librosa et sauvegarde BPM + énergie
    dans music/metadata.json. Évite de rescanner les fichiers déjà en cache.

    Returns: dict {filename: {"bpm": float, "energy": float, "scanned_at": int}}
    """
    cache = load_music_metadata_cache(music_dir)
    audio_files = _audio_files_in(music_dir)

    if not audio_files:
        return cache

    updated = False
    now = int(time.time())

    for filename in audio_files:
        # Rescanner si : absent du cache, ou cache trop vieux, ou force
        entry = cache.get(filename)
        age_days = (now - entry.get("scanned_at", 0)) / 86400 if entry else float("inf")

        if force or entry is None or age_days > METADATA_CACHE_MAX_AGE_DAYS:
            full_path = os.path.join(music_dir, filename)
            bpm, energy = _compute_audio_metadata(full_path)
            cache[filename] = {"bpm": bpm, "energy": energy, "scanned_at": now}
            logger.info(f"[MUSIC] Scanned: {filename} BPM={bpm:.0f} energy={energy:.4f}")
            updated = True

    if updated:
        try:
            with open(_cache_path(music_dir), "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"[WARN] Impossible de sauvegarder le cache metadata: {e}")

    return cache


def _compute_audio_metadata(audio_path: str) -> tuple[float, float]:
    """Compute BPM and energy in a single librosa.load call."""
    if not HAS_LIBROSA:
        return 0.0, 0.0
    try:
        y, sr = librosa.load(audio_path, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))
        return bpm, energy
    except Exception as e:
        logger.warning(f"Audio metadata detection failed for {os.path.basename(audio_path)}: {e}")
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Sélection par mood
# ---------------------------------------------------------------------------

def _mood_score(bpm: float, energy: float, profile: dict) -> float:
    """
    Score de correspondance entre une piste et un profil mood.
    Retourne une valeur entre 0 et 1 (plus élevé = meilleur match).
    """
    # Score BPM : pénalité proportionnelle à l'écart par rapport à la cible
    if bpm > 0:
        bpm_score = max(0.0, 1.0 - abs(bpm - profile["bpm"]) / profile["bpm_tol"])
    else:
        bpm_score = 0.3  # valeur neutre si BPM inconnu

    # Score énergie : vérifie si le niveau correspond à la cible
    energy_target = profile["energy"]
    if energy_target == "high":
        energy_score = min(1.0, energy / ENERGY_THRESHOLDS["high"])
    elif energy_target == "low":
        # Pour "low", pénaliser les pistes trop énergiques
        energy_score = max(0.0, 1.0 - energy / ENERGY_THRESHOLDS["high"])
    else:  # medium
        # Centré autour du seuil medium
        mid = (ENERGY_THRESHOLDS["high"] + ENERGY_THRESHOLDS["medium"]) / 2
        energy_score = max(0.0, 1.0 - abs(energy - mid) / mid)

    return bpm_score * 0.6 + energy_score * 0.4


def select_music_by_mood(
    music_dir: str,
    mood: str | None = None,
    exclude: set[str] | None = None,
) -> str | None:
    """
    Sélectionne la piste la plus adaptée au mood depuis music_dir.

    Args:
        music_dir:  Dossier contenant les fichiers audio
        mood:       Mood LLM (energetic, chill, etc.) ou None pour neutre
        exclude:    Ensemble de noms de fichiers à éviter (anti-répétition)

    Returns:
        Chemin absolu de la piste sélectionnée, ou None si aucune piste.
    """
    if not os.path.exists(music_dir):
        return None

    cache = build_music_metadata_cache(music_dir)
    audio_files = _audio_files_in(music_dir)

    if not audio_files:
        return None

    if not HAS_LIBROSA:
        logger.info("[INFO] librosa non installé — sélection aléatoire")
        candidates = [f for f in audio_files if f not in (exclude or set())]
        return os.path.join(music_dir, random.choice(candidates or audio_files))

    profile = MOOD_PROFILES.get(mood or "neutral", MOOD_PROFILES["neutral"])
    exclude = exclude or set()

    scored = []
    for filename in audio_files:
        if filename in exclude:
            continue
        meta = cache.get(filename, {})
        bpm = meta.get("bpm", 0.0)
        energy = meta.get("energy", 0.0)
        score = _mood_score(bpm, energy, profile)
        scored.append((filename, score, bpm, energy))

    if not scored:
        # Tous exclus → on lève l'exclusion
        scored = [
            (f, _mood_score(
                cache.get(f, {}).get("bpm", 0.0),
                cache.get(f, {}).get("energy", 0.0),
                profile,
            ), 0, 0)
            for f in audio_files
        ]

    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0]
    logger.info(
        f"[MUSIC] mood={mood or 'neutral'} → {best[0]} "
        f"(score={best[1]:.2f}, BPM={best[2]:.0f}, energy={best[3]:.4f})"
    )
    return os.path.join(music_dir, best[0])


def select_best_music(music_dir: str, clip_duration: float, mood: str | None = None) -> str | None:
    """Compatibilité : appelle select_music_by_mood avec mood optionnel."""
    return select_music_by_mood(music_dir, mood=mood)


# ---------------------------------------------------------------------------
# Beat utilities
# ---------------------------------------------------------------------------

def get_beat_times(audio_path: str) -> list[float]:
    """Retourne les timestamps des beats dans un fichier audio."""
    if not HAS_LIBROSA:
        return []
    try:
        y, sr = librosa.load(audio_path, duration=60)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        return librosa.frames_to_time(beat_frames, sr=sr).tolist()
    except Exception:
        return []


def snap_to_beat(duration: float, beat_times: list[float]) -> float:
    """Trouve le beat le plus proche de la durée cible pour un cut propre."""
    if not beat_times:
        return duration
    closest = min(beat_times, key=lambda b: abs(b - duration))
    if abs(closest - duration) <= 2.0:
        return closest
    return duration


# ---------------------------------------------------------------------------
# FFmpeg mix
# ---------------------------------------------------------------------------

def mix_music_to_clip(
    clip_path: str,
    music_path: str,
    output_path: str,
    music_volume: float = MUSIC_VOLUME,
) -> bool:
    """Mixe la musique de fond avec le clip vidéo via ffmpeg."""
    try:
        probe = run_cmd(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", clip_path],
            check=False, text=True
        )
        clip_duration = float(probe.stdout.strip())
        fade_start = max(0, clip_duration - FADE_OUT_DURATION)

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-hide_banner",
            "-i", clip_path,
            "-i", music_path,
            "-filter_complex",
            f"[1:a]volume={music_volume},afade=t=in:st=0:d=1,"
            f"afade=t=out:st={fade_start}:d={FADE_OUT_DURATION},"
            f"atrim=0:{clip_duration}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

        run_cmd(cmd, text=True)
        return True

    except Exception as e:
        logger.error(f"[ERROR] Échec mix musique pour {os.path.basename(clip_path)}: {e}")
        if e.stderr:
            logger.info(f"  ffmpeg: {e.stderr[:200]}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] {e}")
        return False


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def _segment_index_from_filename(filename: str) -> int | None:
    """Extrait l'index de segment depuis un nom de fichier type '002_Title...'."""
    try:
        return int(filename.split("_")[0])
    except (ValueError, IndexError):
        return None


def add_music_to_project(
    project_folder: str,
    music_dir: str | None = None,
    music_file: str | None = None,
    music_volume: float = MUSIC_VOLUME,
    segments: list[dict] | None = None,
) -> None:
    """
    Ajoute de la musique de fond à tous les clips finaux du projet.

    Si `segments` est fourni (liste depuis viral_segments.json), la piste est
    choisie par clip selon le champ `music_mood` de chaque segment.
    """
    burned_folder = os.path.join(project_folder, "burned_sub")
    final_folder = os.path.join(project_folder, "final")
    source_folder = burned_folder if os.path.exists(burned_folder) else final_folder

    if not os.path.exists(source_folder):
        logger.error(f"[ERROR] Aucun dossier de clips trouvé: {source_folder}")
        return

    output_folder = os.path.join(project_folder, "with_music")
    os.makedirs(output_folder, exist_ok=True)

    # Résoudre le dossier musique
    if music_file and os.path.exists(music_file):
        resolved_music_dir = None  # fichier fixe
    else:
        resolved_music_dir = music_dir or DEFAULT_MUSIC_DIR
        # Auto-fetch si vide ou périmé
        if fetch_music and needs_refresh and needs_refresh(resolved_music_dir):
            fetch_music(music_dir=resolved_music_dir)
        # Pré-construire le cache metadata une seule fois
        if HAS_LIBROSA:
            build_music_metadata_cache(resolved_music_dir)

    clips = sorted([
        f for f in os.listdir(source_folder)
        if f.endswith((".mp4", ".mkv")) and "temp_" not in f
    ])

    success_count = 0
    used_tracks: set[str] = set()  # anti-répétition entre clips

    for clip_name in clips:
        clip_path = os.path.join(source_folder, clip_name)
        output_name = clip_name.replace(".mp4", "_music.mp4").replace(".mkv", "_music.mkv")
        output_path = os.path.join(output_folder, output_name)

        # Déterminer le mood du segment correspondant
        mood = None
        if segments:
            idx = _segment_index_from_filename(clip_name)
            if idx is not None and idx < len(segments):
                mood = segments[idx].get("music_mood") or segments[idx].get("mood")

        # Sélectionner la musique
        if music_file and os.path.exists(music_file):
            selected_music = music_file
        else:
            selected_music = select_music_by_mood(
                resolved_music_dir, mood=mood, exclude=used_tracks
            )

        if not selected_music:
            logger.warning(f"[WARN] Aucune musique disponible pour {clip_name}")
            continue

        used_tracks.add(os.path.basename(selected_music))
        logger.info(f"[MUSIC] {clip_name} — mood={mood or 'auto'} → {os.path.basename(selected_music)}")

        if mix_music_to_clip(clip_path, selected_music, output_path, music_volume):
            success_count += 1

    logger.info(f"[MUSIC] Terminé: {success_count}/{len(clips)} clips dans {output_folder}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add background music to ViralCutter clips")
    parser.add_argument("--project-folder", default="tmp")
    parser.add_argument("--music-dir")
    parser.add_argument("--music-file")
    parser.add_argument("--volume", type=float, default=MUSIC_VOLUME)
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rescan de tous les fichiers audio")

    args = parser.parse_args()

    if args.rebuild_cache:
        d = args.music_dir or DEFAULT_MUSIC_DIR
        logger.info(f"[MUSIC] Reconstruction du cache metadata dans {d}")
        build_music_metadata_cache(d, force=True)
    else:
        add_music_to_project(args.project_folder, args.music_dir, args.music_file, args.volume)
