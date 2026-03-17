"""
=============================================================================
FETCH DISTRACTION — Cache incrémental de vidéos de distraction
=============================================================================
Télécharge automatiquement des vidéos "satisfaisantes" via yt-dlp pour le
format split-screen TikTok (vidéo principale en haut, distraction en bas).

Deux modes :
  - search  (défaut) : requêtes ytsearch tournantes → contenu varié et récent
  - channels         : chaînes YouTube fixes → plus fiable mais moins varié

Comportement :
  - Premier lancement : télécharge DEFAULT_COUNT vidéos
  - Toutes les 48h : ajoute EXPAND_COUNT nouvelles vidéos
  - --no-overwrites : les vidéos déjà présentes ne sont jamais re-téléchargées
  - Timestamp du dernier fetch dans distraction/.last_fetch

Usage CLI :
    python -m scripts.fetch_distraction                  # Expansion auto si >48h
    python -m scripts.fetch_distraction --force          # Force maintenant
    python -m scripts.fetch_distraction --count 5        # Nombre de vidéos
    python -m scripts.fetch_distraction --mode channels  # Mode chaînes fixes
    python -m scripts.fetch_distraction --status         # Affiche l'état du cache
=============================================================================
"""

import os
import subprocess
import random
import time
import argparse

# Dossier de destination par défaut
DEFAULT_DISTRACTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "distraction"
)

# Fichier timestamp du dernier fetch
LAST_FETCH_FILE = ".last_fetch"

# Intervalle entre deux expansions automatiques (en heures)
FETCH_INTERVAL_HOURS = 48

# Nombre de vidéos à télécharger au premier lancement (dossier vide)
DEFAULT_COUNT = 10

# Nombre de vidéos ajoutées à chaque expansion
EXPAND_COUNT = 3

# Nombre minimum de vidéos avant de déclencher un fetch immédiat
MIN_VIDEOS = 3

# Mode par défaut : "search" (requêtes ytsearch) ou "channels" (chaînes fixes)
DEFAULT_MODE = "search"

# ---------------------------------------------------------------------------
# Sources — Mode "search" : requêtes yt-dlp ytsearch tournantes
# À chaque fetch, on pioche aléatoirement dans cette liste
# → contenu varié et récent sans toujours les mêmes chaînes
# ---------------------------------------------------------------------------
DISTRACTION_SEARCH_QUERIES = [
    "ytsearch5:minecraft parkour satisfying no commentary",
    "ytsearch5:subway surfers gameplay no commentary",
    "ytsearch5:satisfying pressure washing compilation",
    "ytsearch5:satisfying slime compilation no talking",
    "ytsearch5:candy cutting asmr no commentary",
    "ytsearch5:satisfying cooking compilation no talking",
    "ytsearch5:satisfying sand cutting compilation",
    "ytsearch5:satisfying kinetic sand asmr",
    "ytsearch5:temple run gameplay no commentary",
    "ytsearch5:satisfying oddly satisfying compilation",
]

# ---------------------------------------------------------------------------
# Sources — Mode "channels" : chaînes YouTube fixes (fallback fiable)
# ---------------------------------------------------------------------------
DISTRACTION_SOURCES_CHANNELS = [
    {
        "name": "Subway Surfers Official",
        "url": "https://www.youtube.com/@SubwaySurfers/videos",
    },
    {
        "name": "Satisfying Videos",
        "url": "https://www.youtube.com/@SatisfyingVideos/videos",
    },
    {
        "name": "Power Wash Show",
        "url": "https://www.youtube.com/@PowerWashShow/videos",
    },
]


# ---------------------------------------------------------------------------
# Helpers fichiers
# ---------------------------------------------------------------------------

def _video_files_in(distraction_dir: str) -> list[str]:
    """Liste les fichiers vidéo dans le dossier."""
    if not os.path.exists(distraction_dir):
        return []
    exts = (".mp4", ".mkv", ".webm", ".avi", ".mov")
    return [
        os.path.join(distraction_dir, f)
        for f in os.listdir(distraction_dir)
        if f.lower().endswith(exts)
    ]


def _last_fetch_path(distraction_dir: str) -> str:
    return os.path.join(distraction_dir, LAST_FETCH_FILE)


def _read_last_fetch(distraction_dir: str) -> float:
    """Retourne le timestamp UNIX du dernier fetch, ou 0 si jamais fait."""
    path = _last_fetch_path(distraction_dir)
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return 0.0


def _write_last_fetch(distraction_dir: str) -> None:
    os.makedirs(distraction_dir, exist_ok=True)
    with open(_last_fetch_path(distraction_dir), "w") as f:
        f.write(str(time.time()))


def _hours_since_last_fetch(distraction_dir: str) -> float:
    last = _read_last_fetch(distraction_dir)
    if last == 0.0:
        return float("inf")
    return (time.time() - last) / 3600


# ---------------------------------------------------------------------------
# Logique de déclenchement
# ---------------------------------------------------------------------------

def needs_refresh(distraction_dir: str) -> bool:
    """
    Retourne True si un fetch doit être déclenché :
      - Dossier vide ou < MIN_VIDEOS vidéos
      - Dernier fetch il y a plus de FETCH_INTERVAL_HOURS heures
    """
    files = _video_files_in(distraction_dir)
    if len(files) < MIN_VIDEOS:
        return True
    return _hours_since_last_fetch(distraction_dir) >= FETCH_INTERVAL_HOURS


# ---------------------------------------------------------------------------
# Téléchargement — mode search
# ---------------------------------------------------------------------------

def _download_search(query: str, distraction_dir: str) -> int:
    """
    Télécharge des vidéos via une requête ytsearch.
    Retourne le nombre de nouvelles vidéos ajoutées.
    """
    os.makedirs(distraction_dir, exist_ok=True)
    print(f"[DISTRACTION] Recherche : {query}")

    output_template = os.path.join(distraction_dir, "%(title).80s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "--no-overwrites",
        "--no-post-overwrites",
        "--ignore-errors",
        "--quiet",
        "--no-warnings",
        "--output", output_template,
        query,
    ]

    before = set(_video_files_in(distraction_dir))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode not in (0, 1):
            print(f"[WARN] yt-dlp exit {result.returncode} : {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        print(f"[WARN] Timeout pour : {query}")
    except FileNotFoundError:
        print("[ERROR] yt-dlp introuvable. Installe avec : pip install yt-dlp")
        return 0

    after = set(_video_files_in(distraction_dir))
    new_files = after - before
    for f in sorted(new_files):
        print(f"[DISTRACTION]   + {os.path.basename(f)}")
    print(f"[DISTRACTION] {len(new_files)} nouvelles vidéos via recherche")
    return len(new_files)


# ---------------------------------------------------------------------------
# Téléchargement — mode channels
# ---------------------------------------------------------------------------

def _download_channel(source: dict, distraction_dir: str, count: int) -> int:
    """
    Télécharge des vidéos depuis une chaîne YouTube.
    Retourne le nombre de nouvelles vidéos ajoutées.
    """
    os.makedirs(distraction_dir, exist_ok=True)
    print(f"[DISTRACTION] Chaîne : {source['name']} ({count} vidéos max)")

    output_template = os.path.join(distraction_dir, "%(title).80s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist-reverse",
        "--playlist-end", str(count),
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "--no-overwrites",
        "--no-post-overwrites",
        "--ignore-errors",
        "--quiet",
        "--no-warnings",
        "--output", output_template,
        source["url"],
    ]

    before = set(_video_files_in(distraction_dir))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode not in (0, 1):
            print(f"[WARN] yt-dlp exit {result.returncode} : {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        print(f"[WARN] Timeout pour : {source['name']}")
    except FileNotFoundError:
        print("[ERROR] yt-dlp introuvable. Installe avec : pip install yt-dlp")
        return 0

    after = set(_video_files_in(distraction_dir))
    new_files = after - before
    for f in sorted(new_files):
        print(f"[DISTRACTION]   + {os.path.basename(f)}")
    print(f"[DISTRACTION] {len(new_files)} nouvelles vidéos depuis {source['name']}")
    return len(new_files)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def fetch_distraction(
    distraction_dir: str = DEFAULT_DISTRACTION_DIR,
    force: bool = False,
    count: int | None = None,
    mode: str = DEFAULT_MODE,
) -> int:
    """
    Expansion incrémentale du cache de vidéos de distraction.

    - Si force=False et cache récent (<48h) et assez de vidéos : ne fait rien.
    - mode="search" : utilise des requêtes ytsearch tournantes (défaut)
    - mode="channels" : utilise les chaînes YouTube fixes

    Returns:
        Nombre total de nouvelles vidéos ajoutées lors de cet appel.
    """
    existing = _video_files_in(distraction_dir)
    hours_ago = _hours_since_last_fetch(distraction_dir)

    if not force and not needs_refresh(distraction_dir):
        next_in = FETCH_INTERVAL_HOURS - hours_ago
        print(
            f"[DISTRACTION] Cache OK — {len(existing)} vidéos, "
            f"prochain refresh dans {next_in:.1f}h"
        )
        return 0

    is_first = len(existing) < MIN_VIDEOS
    per_source = count if count is not None else (DEFAULT_COUNT if is_first else EXPAND_COUNT)

    reason = "premier remplissage" if is_first else f"expansion +48h ({hours_ago:.1f}h écoulées)"
    print(f"[DISTRACTION] {reason.capitalize()} — mode={mode}, {per_source} vidéos")

    total = 0

    if mode == "search":
        # Pioche des requêtes aléatoires pour varier le contenu
        queries_to_use = random.sample(
            DISTRACTION_SEARCH_QUERIES,
            min(per_source, len(DISTRACTION_SEARCH_QUERIES))
        )
        for query in queries_to_use:
            total += _download_search(query, distraction_dir)
    else:
        # Mode channels : download depuis les chaînes fixes
        for source in DISTRACTION_SOURCES_CHANNELS:
            total += _download_channel(source, distraction_dir, per_source)

    _write_last_fetch(distraction_dir)

    final_count = len(_video_files_in(distraction_dir))
    print(f"[DISTRACTION] Cache total : {final_count} vidéos dans {distraction_dir}")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache incrémental de vidéos de distraction")
    parser.add_argument("--distraction-dir", default=DEFAULT_DISTRACTION_DIR, help="Dossier de destination")
    parser.add_argument("--force", action="store_true", help="Forcer le fetch maintenant")
    parser.add_argument("--count", type=int, default=None, help="Nombre de vidéos à télécharger (override)")
    parser.add_argument("--mode", choices=["search", "channels"], default=DEFAULT_MODE, help="Mode de recherche (défaut: search)")
    parser.add_argument("--status", action="store_true", help="Afficher l'état du cache et quitter")
    args = parser.parse_args()

    if args.status:
        d = args.distraction_dir
        files = _video_files_in(d)
        hours = _hours_since_last_fetch(d)
        last_str = f"{hours:.1f}h ago" if hours < float("inf") else "jamais"
        next_str = f"dans {FETCH_INTERVAL_HOURS - hours:.1f}h" if hours < FETCH_INTERVAL_HOURS else "maintenant"
        print(f"[DISTRACTION] Dossier     : {d}")
        print(f"[DISTRACTION] Vidéos      : {len(files)}")
        print(f"[DISTRACTION] Dernier fetch : {last_str}")
        print(f"[DISTRACTION] Prochain    : {next_str}")
        for f in sorted(files):
            print(f"              - {os.path.basename(f)}")
    else:
        fetch_distraction(args.distraction_dir, force=args.force, count=args.count, mode=args.mode)
