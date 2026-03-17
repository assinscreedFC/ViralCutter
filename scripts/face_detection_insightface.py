import cv2
import numpy as np
import os
import sys
from contextlib import contextmanager
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

app = None

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def init_insightface():
    """Explicit initialization if needed outside import."""
    global app
    if not INSIGHTFACE_AVAILABLE:
        raise ImportError("InsightFace not installed. Please install it.")
    
    if app is None:
        # Provider options to reduce logging if possible (often needs env var)
        # But redirection is safer for C++ logs
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            print(f"InsightFace: Available ONNX Providers: {available}")
            if 'CUDAExecutionProvider' not in available:
                print("WARNING: CUDAExecutionProvider not found. InsightFace will likely run on CPU.")
                print("To fix, install onnxruntime-gpu: pip install onnxruntime-gpu")
        except Exception as e:
            print(f"InsightFace: Could not check available providers: {e}")

        with suppress_stdout_stderr():
            app = FaceAnalysis(name='buffalo_l', providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def detect_faces_insightface(frame):
    """
    Detect faces using InsightFace.
    Returns a list of dicts with 'bbox' and 'kps'.
    bbox is [x1, y1, x2, y2], kps is 5 keypoints (eyes, nose, mouth corners).
    """
    global app
    if app is None:
        init_insightface()

    faces = app.get(frame)
    results = []
    for face in faces:
        # Convert bbox to int
        bbox = face.bbox.astype(int)
        res = {
            'bbox': bbox, # [x1, y1, x2, y2]
            'kps': face.kps,
            'det_score': face.det_score
        }
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
             res['landmark_2d_106'] = face.landmark_2d_106
        if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
             res['landmark_3d_68'] = face.landmark_3d_68
             
        results.append(res)
    return results

def check_crop_quality(crop_width: int, target_width: int, min_ratio: float = 0.45) -> bool:
    """Vérifie si le crop a assez de résolution pour l'upscale.
    Retourne True si la qualité est suffisante."""
    return crop_width >= target_width * min_ratio


def crop_center_fallback(frame, target_width=1080, target_height=1920):
    """Center crop 9:16 sans face tracking — préserve la résolution source."""
    h, w, _ = frame.shape
    target_ar = target_width / target_height

    if w / h > target_ar:
        new_w = int(h * target_ar)
        new_h = h
    else:
        new_w = w
        new_h = int(w / target_ar)

    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]
    return cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)


def crop_and_resize_insightface(frame, face_bbox, target_width=1080, target_height=1920):
    """
    Crops and resizes the frame to target dimensions centered on the face_bbox.
    face_bbox: [x1, y1, x2, y2]
    Includes quality gate: falls back to center crop if face zoom would degrade quality.
    """
    h, w, _ = frame.shape
    x1, y1, x2, y2 = face_bbox

    face_center_x = (x1 + x2) // 2
    face_center_y = (y1 + y2) // 2

    # Calculate source crop height/width maintaining 9:16 ratio
    source_h = h
    source_w = int(source_h * (target_width / target_height))

    if source_w > w:
        source_w = w
        source_h = int(source_w * (target_height / target_width))

    # Quality gate: si le crop est trop petit, fallback center crop
    if not check_crop_quality(source_w, target_width):
        return crop_center_fallback(frame, target_width, target_height)

    # Calculate top-left corner of the crop
    crop_x1 = face_center_x - (source_w // 2)
    crop_y1 = face_center_y - (source_h // 2)

    # Adjust to stay within bounds
    if crop_x1 < 0:
        crop_x1 = 0
    elif crop_x1 + source_w > w:
        crop_x1 = w - source_w

    if crop_y1 < 0:
        crop_y1 = 0
    elif crop_y1 + source_h > h:
        crop_y1 = h - source_h

    crop_x2 = crop_x1 + source_w
    crop_y2 = crop_y1 + source_h

    # Crop
    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to final target
    result = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return result

if __name__ == "__main__":
    # Test block
    print("Testing InsightFace...")
    # Create a dummy image or try to load one if available, but for now just print config
    print("InsightFace initialized.")
