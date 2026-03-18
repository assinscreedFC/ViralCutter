"""
=============================================================================
FETCH MUSIC — Cache incrémental de musiques royalty-free
=============================================================================
Télécharge automatiquement des pistes via yt-dlp depuis des chaînes YouTube
royalty-free. Le cache s'étend toutes les 24h sans jamais supprimer l'existant.

Comportement :
  - Premier lancement : télécharge DEFAULT_COUNT pistes par source
  - Toutes les 24h : ajoute EXPAND_COUNT nouvelles pistes par source
  - --no-overwrites : les pistes déjà présentes ne sont jamais re-téléchargées
  - Timestamp du dernier fetch dans music/.last_fetch

Usage CLI :
    python -m scripts.fetch_music                  # Expansion auto si >24h
    python -m scripts.fetch_music --force          # Force maintenant
    python -m scripts.fetch_music --count 20       # Nombre de pistes par source
    python -m scripts.fetch_music --status         # Affiche l'état du cache
    python -m scripts.fetch_music --list-sources   # Liste les sources configurées
=============================================================================
"""

import logging
import os
import subprocess
import json
import time
import argparse

logger = logging.getLogger(__name__)

# Dossier de destination par défaut
DEFAULT_MUSIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "music"
)

# Fichier timestamp du dernier fetch
LAST_FETCH_FILE = ".last_fetch"

# Intervalle entre deux expansions automatiques (en heures)
FETCH_INTERVAL_HOURS = 24

# Nombre de pistes à télécharger au premier lancement (dossier vide)
DEFAULT_COUNT = 15

# Nombre de pistes ajoutées à chaque expansion toutes les 24h
EXPAND_COUNT = 5

# Nombre minimum de pistes avant de déclencher un fetch immédiat
MIN_TRACKS = 5

# Sources royalty-free — chaînes YouTube publiques
# Format : {"name": str, "url": str, "count": int}
MUSIC_SOURCES = [
    {
        "name": "NCS — NoCopyrightSounds",
        "url": "https://www.youtube.com/@NoCopyrightSounds/videos",
        "count": DEFAULT_COUNT,
    },
    {
        "name": "Lofi Girl — Chill Beats",
        "url": "https://www.youtube.com/@LofiGirl/videos",
        "count": DEFAULT_COUNT,
    },
    {
        "name": "Chillhop Music",
        "url": "https://www.youtube.com/@Chillhop/videos",
        "count": DEFAULT_COUNT,
    },
]


# ---------------------------------------------------------------------------
# Helpers fichiers
# ---------------------------------------------------------------------------

def _music_files_in(music_dir: str) -> list[str]:
    """Liste les fichiers audio dans le dossier."""
    if not os.path.exists(music_dir):
        return []
    exts = (".mp3", ".wav", ".ogg", ".m4a", ".flac")
    return [
        os.path.join(music_dir, f)
        for f in os.listdir(music_dir)
        if f.lower().endswith(exts)
    ]


def _last_fetch_path(music_dir: str) -> str:
    return os.path.join(music_dir, LAST_FETCH_FILE)


def _read_last_fetch(music_dir: str) -> float:
    """Retourne le timestamp UNIX du dernier fetch, ou 0 si jamais fait."""
    path = _last_fetch_path(music_dir)
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return 0.0


def _write_last_fetch(music_dir: str) -> None:
    """Enregistre l'heure actuelle comme dernier fetch."""
    os.makedirs(music_dir, exist_ok=True)
    with open(_last_fetch_path(music_dir), "w") as f:
        f.write(str(time.time()))


def _hours_since_last_fetch(music_dir: str) -> float:
    last = _read_last_fetch(music_dir)
    if last == 0.0:
        return float("inf")
    return (time.time() - last) / 3600


# ---------------------------------------------------------------------------
# Logique de déclenchement
# ---------------------------------------------------------------------------

def needs_refresh(music_dir: str) -> bool:
    """
    Retourne True si un fetch doit être déclenché :
      - Dossier vide ou < MIN_TRACKS pistes
      - Dernier fetch il y a plus de FETCH_INTERVAL_HOURS heures
    """
    files = _music_files_in(music_dir)
    if len(files) < MIN_TRACKS:
        return True
    return _hours_since_last_fetch(music_dir) >= FETCH_INTERVAL_HOURS


# ---------------------------------------------------------------------------
# Téléchargement
# ---------------------------------------------------------------------------

def download_from_source(source: dict, music_dir: str, count: int) -> int:
    """
    Télécharge des pistes audio depuis une source YouTube via yt-dlp.
    Ne re-télécharge jamais les fichiers déjà présents (--no-overwrites).
    Retourne le nombre de nouvelles pistes ajoutées.
    """
    os.makedirs(music_dir, exist_ok=True)
    logger.info(f"[MUSIC] Téléchargement depuis : {source['name']} ({count} pistes max)")

    output_template = os.path.join(music_dir, "%(title).60s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist-reverse",
        "--playlist-end", str(count),
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",       # VBR ~130 kbps, suffisant pour fond
        "--no-overwrites",             # ne jamais re-télécharger l'existant
        "--no-post-overwrites",
        "--ignore-errors",
        "--quiet",
        "--no-warnings",
        "--embed-metadata",
        "--output", output_template,
        source["url"],
    ]

    before = set(_music_files_in(music_dir))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode not in (0, 1):  # 1 = erreurs partielles tolérées
            logger.warning(f"[WARN] yt-dlp exit {result.returncode} : {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        logger.warning(f"[WARN] Timeout téléchargement {source['name']}")
    except FileNotFoundError:
        logger.error("[ERROR] yt-dlp introuvable. Installe avec : pip install yt-dlp")
        return 0

    after = set(_music_files_in(music_dir))
    new_files = after - before
    if new_files:
        for f in sorted(new_files):
            logger.info(f"[MUSIC]   + {os.path.basename(f)}")
    logger.info(f"[MUSIC] {len(new_files)} nouvelles pistes depuis {source['name']}")
    return len(new_files)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def fetch_music(
    music_dir: str = DEFAULT_MUSIC_DIR,
    sources: list[dict] | None = None,
    force: bool = False,
    count: int | None = None,
) -> int:
    """
    Expansion incrémentale du cache musical.

    - Si force=False et cache récent (<24h) et assez de pistes : ne fait rien.
    - Sinon : télécharge EXPAND_COUNT pistes par source (ou DEFAULT_COUNT si vide).
    - --no-overwrites garantit qu'aucune piste existante n'est retéléchargée.

    Returns:
        Nombre total de nouvelles pistes ajoutées lors de cet appel.
    """
    if sources is None:
        sources = MUSIC_SOURCES

    existing = _music_files_in(music_dir)
    hours_ago = _hours_since_last_fetch(music_dir)

    if not force and not needs_refresh(music_dir):
        next_in = FETCH_INTERVAL_HOURS - hours_ago
        logger.info(
            f"[MUSIC] Cache OK — {len(existing)} pistes, "
            f"prochain refresh dans {next_in:.1f}h"
        )
        return 0

    # Premier remplissage ou expansion ?
    is_first = len(existing) < MIN_TRACKS
    per_source = count if count is not None else (DEFAULT_COUNT if is_first else EXPAND_COUNT)

    reason = "premier remplissage" if is_first else f"expansion +24h ({hours_ago:.1f}h écoulées)"
    logger.info(f"[MUSIC] {reason.capitalize()} — {per_source} pistes/source")

    total = 0
    for source in sources:
        total += download_from_source(source, music_dir, per_source)

    _write_last_fetch(music_dir)

    final_count = len(_music_files_in(music_dir))
    logger.info(f"[MUSIC] Cache total : {final_count} pistes dans {music_dir}")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache incrémental de musiques royalty-free")
    parser.add_argument("--music-dir", default=DEFAULT_MUSIC_DIR, help="Dossier de destination")
    parser.add_argument("--force", action="store_true", help="Forcer le fetch maintenant")
    parser.add_argument("--count", type=int, default=None, help="Pistes par source (override)")
    parser.add_argument("--list-sources", action="store_true", help="Lister les sources et quitter")
    parser.add_argument("--status", action="store_true", help="Afficher l'état du cache et quitter")
    args = parser.parse_args()

    if args.list_sources:
        for s in MUSIC_SOURCES:
            logger.info(f"  - {s['name']}  ({s['count']} pistes/fetch)  {s['url']}")

    elif args.status:
        d = args.music_dir
        files = _music_files_in(d)
        hours = _hours_since_last_fetch(d)
        last_str = f"{hours:.1f}h ago" if hours < float("inf") else "jamais"
        next_str = f"dans {FETCH_INTERVAL_HOURS - hours:.1f}h" if hours < FETCH_INTERVAL_HOURS else "maintenant"
        logger.info(f"[MUSIC] Dossier   : {d}")
        logger.info(f"[MUSIC] Pistes    : {len(files)}")
        logger.info(f"[MUSIC] Dernier fetch : {last_str}")
        logger.info(f"[MUSIC] Prochain  : {next_str}")

    else:
        fetch_music(args.music_dir, force=args.force, count=args.count)
