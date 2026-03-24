"""
=============================================================================
ADD DISTRACTION VIDEO — Format split-screen TikTok
=============================================================================
Empile la vidéo principale (haut, 1080×960) + une vidéo de distraction
satisfaisante (bas, 1080×960) pour produire un format 9:16 (1080×1920).

Format TikTok viral : podcast/commentary en haut + gameplay/satisfying en bas.

Workflow :
  1. Auto-fetch si dossier distraction/ vide (via fetch_distraction)
  2. Pour chaque clip dans with_music/ (ou burned_sub/ si pas de musique) :
     - Sélection aléatoire d'une vidéo distraction (ou fichier forcé)
     - FFmpeg : scale+crop les deux moitiés en 1080×960, vstack → 1080×1920
     - Loop automatique si la distraction est plus courte que le clip principal
     - Audio distraction muté (seul l'audio principal est conservé)
     - Output dans split_screen/ du projet

Usage CLI :
    python -m scripts.add_distraction_video --project VIRALS/mon_projet
    python -m scripts.add_distraction_video --project VIRALS/mon_projet --distraction-file chemin/video.mp4
=============================================================================
"""

import logging
import os
import subprocess
import random
import argparse

from scripts.run_cmd import run as run_cmd
from scripts.ffmpeg_utils import get_best_encoder, build_quality_params, _build_preset_flags, get_video_duration

logger = logging.getLogger(__name__)

try:
    from scripts.fetch_distraction import fetch_distraction, DEFAULT_DISTRACTION_DIR, needs_refresh
except ImportError:
    try:
        from fetch_distraction import fetch_distraction, DEFAULT_DISTRACTION_DIR, needs_refresh
    except ImportError:
        fetch_distraction = None
        DEFAULT_DISTRACTION_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "distraction"
        )
        needs_refresh = None


# Résolution de sortie finale (9:16 TikTok)
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
HALF_WIDTH = 1080

# Ratio par défaut de la distraction (35% de la hauteur totale)
DISTRACTION_RATIO_DEFAULT = 0.35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _video_files_in(directory: str) -> list[str]:
    """Liste les fichiers vidéo dans un dossier."""
    if not os.path.exists(directory):
        return []
    exts = (".mp4", ".mkv", ".webm", ".avi", ".mov")
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(exts)
    ]



def _pick_distraction_video(distraction_dir: str, distraction_file: str | None) -> str | None:
    """
    Retourne le chemin de la vidéo de distraction à utiliser.
    Si distraction_file est fourni et existe, l'utilise directement.
    Sinon, pioche au hasard dans distraction_dir.
    """
    if distraction_file and os.path.isfile(distraction_file):
        return distraction_file

    videos = _video_files_in(distraction_dir)
    if not videos:
        return None
    return random.choice(videos)


# ---------------------------------------------------------------------------
# FFmpeg : stack deux vidéos verticalement
# ---------------------------------------------------------------------------

def _build_ffmpeg_filter(
    main_duration: float,
    available_distraction: float,
    main_crop_y: int | None = None,
    distraction_ratio: float = DISTRACTION_RATIO_DEFAULT,
) -> str:
    """
    Construit le filtre FFmpeg pour empiler les deux vidéos.

    - Vidéo principale  : HALF_WIDTH × main_height  (haut)
    - Vidéo distraction : HALF_WIDTH × dist_height  (bas)
    - distraction_ratio : part de la hauteur totale pour la distraction (ex: 0.35 = 35%)
    - Si la distraction disponible est plus courte, elle est loopée
    - L'audio de la distraction est désactivé
    - main_crop_y : offset Y du crop pour la vidéo principale (None = crop depuis le haut, y=0).
    """
    main_height = int(OUTPUT_HEIGHT * (1 - distraction_ratio))
    dist_height = OUTPUT_HEIGHT - main_height

    # Affiche la vidéo de distraction en entier (letterbox si besoin) pour éviter
    # qu'un contenu paysage 16:9 ou portrait 9:16 soit trop agressivement cropé.
    distraction_scale_crop = (
        f"scale={HALF_WIDTH}:{dist_height}:force_original_aspect_ratio=decrease,"
        f"pad={HALF_WIDTH}:{dist_height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    crop_y = main_crop_y if main_crop_y is not None else 0
    main_scale_crop = (
        f"scale={HALF_WIDTH}:{main_height}:force_original_aspect_ratio=increase,"
        f"crop={HALF_WIDTH}:{main_height}:0:{crop_y}"
    )

    top_filter = f"[0:v]{main_scale_crop}[top]"

    if available_distraction > 0 and available_distraction < main_duration:
        loop_count = int(main_duration / available_distraction) + 1
        bot_filter = f"[1:v]loop={loop_count}:32767:0,{distraction_scale_crop},trim=duration={main_duration:.3f}[bot]"
    else:
        bot_filter = f"[1:v]{distraction_scale_crop}[bot]"

    stack_filter = "[top][bot]vstack=inputs=2[v]"

    return f"{top_filter};{bot_filter};{stack_filter}"


def stack_videos(
    main_video: str,
    distraction_video: str,
    output_path: str,
    main_crop_y: int | None = None,
    distraction_ratio: float = DISTRACTION_RATIO_DEFAULT,
) -> bool:
    """
    Empile main_video (haut) + distraction_video (bas) dans output_path.
    Démarre la distraction à un offset aléatoire (évite de toujours partir de 0s).
    Audio : uniquement celui de main_video.
    Retourne True si succès.
    """
    main_duration = get_video_duration(main_video)
    distraction_duration = get_video_duration(distraction_video)

    if main_duration <= 0:
        logger.info(f"[SPLIT] Durée invalide pour {os.path.basename(main_video)}, ignoré")
        return False

    # Offset aléatoire dans la vidéo de distraction
    max_start = max(0.0, distraction_duration - main_duration)
    distraction_start = random.uniform(0, max_start) if max_start > 1.0 else 0.0
    available_distraction = distraction_duration - distraction_start

    if distraction_start > 1.0:
        logger.info(f"[SPLIT] Distraction offset : {distraction_start:.0f}s / {distraction_duration:.0f}s")

    video_filter = _build_ffmpeg_filter(main_duration, available_distraction, main_crop_y=main_crop_y, distraction_ratio=distraction_ratio)

    encoder_name, encoder_preset = get_best_encoder()
    cmd = [
        "ffmpeg",
        "-y",
        "-i", main_video,
        "-ss", f"{distraction_start:.3f}", "-i", distraction_video,
        "-filter_complex", video_filter,
        "-map", "[v]",
        "-map", "0:a?",       # audio du clip principal uniquement (optionnel)
        "-t", str(main_duration),
        "-c:v", encoder_name,
        *_build_preset_flags(encoder_name, encoder_preset),
        *build_quality_params(encoder_name),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        result = run_cmd(cmd, text=True, check=False)
        if result.returncode != 0:
            logger.error(f"[SPLIT] Erreur FFmpeg : {result.stderr[-500:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"[SPLIT] Timeout pour {os.path.basename(main_video)}")
        return False
    except FileNotFoundError:
        logger.error("[ERROR] ffmpeg introuvable.")
        return False


# ---------------------------------------------------------------------------
# Traitement d'un projet complet
# ---------------------------------------------------------------------------

def add_distraction_to_project(
    project_folder: str,
    distraction_dir: str = DEFAULT_DISTRACTION_DIR,
    distraction_file: str | None = None,
    no_fetch: bool = False,
    main_crop_y: int | None = None,
    distraction_ratio: float = DISTRACTION_RATIO_DEFAULT,
) -> list[str]:
    """
    Ajoute la vidéo de distraction à tous les clips d'un projet.

    Cherche les clips dans (par ordre de priorité) :
      1. with_music/   (clips avec musique de fond)
      2. burned_sub/   (clips avec sous-titres uniquement)

    Output dans : {project_folder}/split_screen/

    Returns:
        Liste des chemins des vidéos split-screen générées.
    """
    # Auto-fetch si nécessaire (sauf si désactivé)
    if not no_fetch and fetch_distraction is not None and needs_refresh is not None:
        if needs_refresh(distraction_dir):
            logger.info("[SPLIT] Téléchargement des vidéos de distraction...")
            fetch_distraction(distraction_dir)
    elif not _video_files_in(distraction_dir):
        logger.warning(f"[SPLIT][WARN] Aucune vidéo dans {distraction_dir} et fetch_distraction indisponible")
        return []

    # Chercher les clips sources
    source_dirs = [
        os.path.join(project_folder, "with_music"),
        os.path.join(project_folder, "burned_sub"),
        os.path.join(project_folder, "final"),  # fallback quand burn pas encore appliqué
    ]
    source_clips: list[str] = []
    for d in source_dirs:
        clips = _video_files_in(d)
        if clips:
            source_clips = sorted(clips)
            logger.info(f"[SPLIT] Source clips : {d} ({len(source_clips)} fichiers)")
            break

    if not source_clips:
        logger.warning(f"[SPLIT][WARN] Aucun clip trouvé dans {project_folder}")
        return []

    # Dossier output
    output_dir = os.path.join(project_folder, "split_screen")
    os.makedirs(output_dir, exist_ok=True)

    generated: list[str] = []

    for clip_path in source_clips:
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        output_path = os.path.join(output_dir, f"{clip_name}_split.mp4")

        if os.path.exists(output_path):
            logger.info(f"[SPLIT] Déjà existant, ignoré : {os.path.basename(output_path)}")
            generated.append(output_path)
            continue

        # Sélection vidéo distraction
        distraction = _pick_distraction_video(distraction_dir, distraction_file)
        if not distraction:
            logger.warning(f"[SPLIT][WARN] Aucune vidéo distraction disponible pour {clip_name}")
            continue

        logger.info(f"[SPLIT] {os.path.basename(clip_path)} + {os.path.basename(distraction)} → {os.path.basename(output_path)}")
        success = stack_videos(clip_path, distraction, output_path, main_crop_y=main_crop_y, distraction_ratio=distraction_ratio)
        if success:
            generated.append(output_path)
            logger.info(f"[SPLIT] OK : {os.path.basename(output_path)}")
        else:
            logger.error(f"[SPLIT] ÉCHEC : {os.path.basename(clip_path)}")

    logger.info(f"[SPLIT] {len(generated)}/{len(source_clips)} clips générés dans {output_dir}")
    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format split-screen TikTok (vidéo principale + distraction)")
    parser.add_argument("--project", required=True, help="Dossier du projet (ex: VIRALS/mon_projet)")
    parser.add_argument("--distraction-dir", default=DEFAULT_DISTRACTION_DIR, help="Dossier des vidéos de distraction")
    parser.add_argument("--distraction-file", default=None, help="Vidéo de distraction spécifique (override aléatoire)")
    parser.add_argument("--no-fetch", action="store_true", help="Désactiver l'auto-fetch (utiliser le cache uniquement)")
    parser.add_argument("--distraction-ratio", type=float, default=DISTRACTION_RATIO_DEFAULT,
                        help=f"Part de la hauteur pour la distraction (0.20-0.50, défaut {DISTRACTION_RATIO_DEFAULT})")
    args = parser.parse_args()

    results = add_distraction_to_project(
        project_folder=args.project,
        distraction_dir=args.distraction_dir,
        distraction_file=args.distraction_file,
        no_fetch=args.no_fetch,
        distraction_ratio=args.distraction_ratio,
    )
    logger.info(f"\n[SPLIT] Terminé — {len(results)} vidéos split-screen créées")
