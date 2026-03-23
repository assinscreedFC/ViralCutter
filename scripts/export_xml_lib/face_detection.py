import json
import os
try:
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Dynamic cuts may fail if coords missing.")

from scripts.frame_utils import downscale_for_analysis

# Process every Nth frame — faces don't move drastically frame-to-frame
_SKIP_EVERY_N = 5


def _cache_path(video_path: str) -> str:
    """Return the JSON cache path for face detection results."""
    return os.path.splitext(video_path)[0] + "_faces.json"


def _load_cache(video_path: str) -> list | None:
    """Load cached face data if available."""
    path = _cache_path(video_path)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Face detection cache loaded: {path} ({len(data)} entries)")
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: corrupted face cache, re-detecting. ({e})")
    return None


def _save_cache(video_path: str, face_data: list) -> None:
    """Persist face detection results to JSON."""
    path = _cache_path(video_path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(face_data, f)
        print(f"Face detection cache saved: {path}")
    except OSError as e:
        print(f"Warning: could not save face cache ({e})")


def detect_faces_jit(video_path):
    """
    Runs face detection on the fly if pre-computed coords are missing.
    Returns: list of {'frame': int, 'faces': [[x1,y1,x2,y2]]}

    Optimizations:
    - Checks JSON cache first; skips detection if cached.
    - Downscales frames to 360p before detection.
    - Processes every Nth frame (frame skipping).
    """
    if not INSIGHTFACE_AVAILABLE:
        print("ERROR: InsightFace not loaded.")
        return []

    # Normalize path for Windows OpenCV
    video_path = os.path.abspath(video_path)

    # --- Cache check ---
    cached = _load_cache(video_path)
    if cached is not None:
        return cached

    print(f"Running JIT Face Detection on: {video_path}")

    # Initialize InsightFace
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"InsightFace Init Error: {e}. Trying CPU only.")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"CRITICAL ERROR: Could not open video file for JIT detection: {video_path}")
        return []

    face_data = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Adaptive skip interval based on FPS (roughly every 5th frame at 30fps)
    skip_n = max(1, int(fps / 6))
    print(f"Video opened. Total frames: {total_frames}, FPS: {fps:.1f}, skip_n: {skip_n}")

    faces_found_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Frame skipping ---
        if frame_idx % skip_n != 0:
            frame_idx += 1
            continue

        # --- Downscale for faster detection ---
        small, scale = downscale_for_analysis(frame, max_width=360)

        faces = app.get(small)
        current_faces = []
        for face in faces:
            # Scale bounding box back to original resolution
            bbox = (face.bbox * scale).astype(int).tolist()
            current_faces.append(bbox)

        if current_faces:
            face_data.append({
                "frame": frame_idx,
                "faces": current_faces
            })
            faces_found_count += 1
            if faces_found_count <= 5:  # Debug first few detections
                print(f"  [DEBUG] Frame {frame_idx}: Found {len(faces)} faces: {current_faces}")

        if frame_idx % 200 == 0:
            print(f"  Scanning faces: {frame_idx}/{total_frames}...")

        frame_idx += 1

    cap.release()
    print(f"JIT Detection Complete. Found faces in {len(face_data)} frames.")

    # --- Save cache ---
    _save_cache(video_path, face_data)

    return face_data
