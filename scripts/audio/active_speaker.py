"""
=============================================================================
ACTIVE SPEAKER DETECTION — Wrapper optionnel pour LR-ASD
=============================================================================
LR-ASD (Light-weight and Robust Active Speaker Detection, IJCV 2025)
Détecte qui parle dans une vidéo multi-locuteurs avec 94.45% mAP.

Installation :
    git clone https://github.com/Junhua-Liao/LR-ASD.git
    pip install -r LR-ASD/requirements.txt

Usage dans edit_video.py :
    from scripts.audio.active_speaker import detect_active_speaker, is_asd_available
    if is_asd_available():
        speaker_bboxes = detect_active_speaker(video_path)
=============================================================================
"""

import os
import sys

# Tenter d'importer LR-ASD
LR_ASD_AVAILABLE = False
_asd_model = None

try:
    # LR-ASD doit être cloné dans le dossier du projet ou installé
    lr_asd_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "LR-ASD")
    if os.path.exists(lr_asd_path):
        sys.path.insert(0, lr_asd_path)
        LR_ASD_AVAILABLE = True
except Exception:
    pass


def is_asd_available() -> bool:
    """Vérifie si LR-ASD est installé et disponible."""
    return LR_ASD_AVAILABLE


def detect_active_speaker(video_path: str, output_dir: str | None = None) -> list[dict]:
    """
    Détecte le locuteur actif frame par frame.

    Retourne une liste de dicts :
    [
        {
            "frame": 0,
            "timestamp": 0.0,
            "active_speaker_bbox": [x1, y1, x2, y2],
            "all_faces": [[x1, y1, x2, y2], ...]
        },
        ...
    ]

    Si LR-ASD n'est pas installé, retourne une liste vide.
    """
    if not LR_ASD_AVAILABLE:
        print("[WARN] LR-ASD non installé. Utilisez le face tracking standard.")
        print("[INFO] Pour installer : git clone https://github.com/Junhua-Liao/LR-ASD.git")
        return []

    try:
        # Import dynamique pour éviter les erreurs si pas installé
        from LR_ASD_inference import run_inference

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(video_path), "asd_cache")
        os.makedirs(output_dir, exist_ok=True)

        results = run_inference(video_path, output_dir)
        return results

    except ImportError:
        print("[WARN] LR-ASD importé mais module d'inférence introuvable.")
        print("[INFO] Vérifiez l'installation : https://github.com/Junhua-Liao/LR-ASD")
        return []
    except Exception as e:
        print(f"[ERROR] LR-ASD inference failed: {e}")
        return []


def get_speaker_bbox_at_time(asd_results: list[dict], timestamp: float) -> list[int] | None:
    """Retourne la bbox du locuteur actif au timestamp donné."""
    if not asd_results:
        return None

    # Trouver le résultat le plus proche du timestamp
    closest = min(asd_results, key=lambda r: abs(r.get("timestamp", 0) - timestamp))
    if abs(closest.get("timestamp", 0) - timestamp) < 0.5:  # tolérance 500ms
        return closest.get("active_speaker_bbox")
    return None
