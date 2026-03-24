from __future__ import annotations

import json
import logging
import cv2
import numpy as np
import os
import mediapipe as mp

from scripts.run_cmd import run as run_cmd
from scripts.frame_utils import downscale_for_analysis
from scripts.ffmpeg_utils import get_best_encoder, create_ffmpeg_pipe
from typing import Callable

logger = logging.getLogger(__name__)
from scripts.one_face import crop_and_resize_single_face, resize_with_padding, detect_face_or_body, crop_center_zoom, crop_to_smart_region
from scripts.two_face import crop_and_resize_two_faces, detect_face_or_body_two_faces
try:
    from scripts.face_detection_insightface import init_insightface, detect_faces_insightface, crop_and_resize_insightface, get_face_embedding, cosine_similarity
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not found or error importing. Install with: pip install insightface onnxruntime-gpu")


def apply_zoom_effect(frame: np.ndarray, current_time: float, zoom_cues: list,
                      frame_width: int = 1080, frame_height: int = 1920) -> np.ndarray:
    """Apply smooth zoom effect based on LLM-generated zoom_cues."""
    for cue in zoom_cues:
        ts = cue.get("timestamp", 0)
        duration = cue.get("duration", 1.0)
        intensity = cue.get("intensity", 1.0)
        if duration <= 0 or intensity <= 1.0:
            continue
        if not (ts <= current_time <= ts + duration):
            continue

        # Normalized progress [0, 1]
        t = (current_time - ts) / duration

        # Ease-in-out: ramp up 0-30%, hold 30-70%, ramp down 70-100%
        if t < 0.3:
            easing = t / 0.3  # 0 -> 1
            easing = max(0.0, min(1.0, easing * easing * (3 - 2 * easing)))  # smoothstep
        elif t < 0.7:
            easing = 1.0
        else:
            easing = (1.0 - t) / 0.3  # 1 -> 0
            easing = max(0.0, min(1.0, easing * easing * (3 - 2 * easing)))

        zoom_factor = 1.0 + (intensity - 1.0) * easing
        if zoom_factor <= 1.0:
            return frame

        # Center crop by zoom_factor then resize back
        h, w = frame.shape[:2]
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return frame


def get_center_bbox(bbox: list) -> tuple[float, float]:
    # bbox: [x1, y1, x2, y2]
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def get_center_rect(rect: tuple) -> tuple[float, float]:
    # rect: (x, y, w, h)
    return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)

def sort_by_proximity(new_faces: list, old_faces: list, center_func: Callable) -> list:
    """
    Sorts new_faces to match the order of old_faces based on distance.
    new_faces: list of face objects (bbox or tuple)
    old_faces: list of face objects (bbox or tuple)
    center_func: function that takes a face object and returns (cx, cy)
    """
    if not old_faces or len(old_faces) != 2 or len(new_faces) != 2:
        return new_faces
    
    old_c1 = center_func(old_faces[0])
    old_c2 = center_func(old_faces[1])
    
    new_c1 = center_func(new_faces[0])
    new_c2 = center_func(new_faces[1])
    
    # Cost if we keep order: [new1, new2]
    # dist(old1, new1) + dist(old2, new2)
    dist_keep = ((old_c1[0]-new_c1[0])**2 + (old_c1[1]-new_c1[1])**2) + \
                ((old_c2[0]-new_c2[0])**2 + (old_c2[1]-new_c2[1])**2)
                
    # Cost if we swap: [new2, new1]
    # dist(old1, new2) + dist(old2, new1)
    dist_swap = ((old_c1[0]-new_c2[0])**2 + (old_c1[1]-new_c2[1])**2) + \
                ((old_c2[0]-new_c1[0])**2 + (old_c2[1]-new_c1[1])**2)
                
    # If swapping reduces total movement distance, do it
    if dist_swap < dist_keep:
        return [new_faces[1], new_faces[0]]
    
    return new_faces

def generate_short_fallback(input_file: str, output_file: str, index: int, project_folder: str, final_folder: str, no_face_mode: str = "padding") -> None:
    """Fallback function: Center Crop (Zoom), Padding, Saliency, or Motion tracking if detection fails."""
    logger.info(f"Processing (Fallback): {input_file} | Mode: {no_face_mode}")
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        logger.error(f"Error opening video: {input_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Target dimensions (9:16)
    target_width = 1080
    target_height = 1920

    process = create_ffmpeg_pipe(output_file, fps, target_width, target_height)

    # Smart region state (saliency / motion)
    smart_cx = width // 2
    smart_cy = height // 2
    prev_gray = None
    saliency_detector = None
    if no_face_mode == "saliency":
        saliency_detector = cv2.saliency.StaticSaliencySpectralResidual.create()

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if no_face_mode == "zoom":
                result = crop_center_zoom(frame)
            elif no_face_mode == "saliency":
                _, sal_map = saliency_detector.computeSaliency(frame)
                sal_map = (sal_map * 255).astype(np.uint8)
                m = cv2.moments(sal_map)
                if m['m00'] > 0:
                    cx = int(m['m10'] / m['m00'])
                    cy = int(m['m01'] / m['m00'])
                    smart_cx = int(smart_cx * 0.90 + cx * 0.10)
                    smart_cy = int(smart_cy * 0.90 + cy * 0.10)
                result = crop_to_smart_region(frame, smart_cx, smart_cy)
            elif no_face_mode == "motion":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    m = cv2.moments(diff)
                    if m['m00'] > 500:
                        cx = int(m['m10'] / m['m00'])
                        cy = int(m['m01'] / m['m00'])
                        smart_cx = int(smart_cx * 0.85 + cx * 0.15)
                        smart_cy = int(smart_cy * 0.85 + cy * 0.15)
                prev_gray = gray
                result = crop_to_smart_region(frame, smart_cx, smart_cy)
            else:
                result = resize_with_padding(frame)

            try:
                process.stdin.write(result.tobytes())
            except (BrokenPipeError, OSError) as e:
                logger.error("ffmpeg pipe broken at frame %d: %s", frame_idx, e)
                break
            frame_idx += 1

        cap.release()
        if not process.stdin.closed:
            process.stdin.close()
        process.wait()

        finalize_video(input_file, output_file, index, fps, project_folder, final_folder)
    finally:
        if cap.isOpened():
            cap.release()
        if not process.stdin.closed:
            process.stdin.close()
        process.wait()

def finalize_video(input_file: str, output_file: str, index: int, fps: float, project_folder: str, final_folder: str) -> None:
    """Mux audio and video with 3-level fallback: copy -> re-encode -> video-only."""
    audio_file = os.path.join(project_folder, "cuts", f"output-audio-{index}.aac")
    final_output = os.path.join(final_folder, f"final-output{str(index).zfill(3)}_processed.mp4")

    # Attempt 1: copy audio stream (fast, zero loss)
    run_cmd(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", input_file, "-vn", "-acodec", "copy", audio_file], check=False)

    # Attempt 2: re-encode to AAC if copy failed
    if not (os.path.exists(audio_file) and os.path.getsize(audio_file) > 0):
        logger.warning("Audio copy failed for %s, trying re-encode...", os.path.basename(input_file))
        run_cmd(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", input_file, "-vn", "-c:a", "aac", "-b:a", "192k", audio_file], check=False)

    if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
            "-i", output_file,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            final_output
        ]
        try:
            run_cmd(command)
            logger.info("Final file generated: %s", final_output)
            try:
                os.remove(audio_file)
                os.remove(output_file)
            except OSError:
                pass  # temp files may already be removed
        except Exception as e:
            logger.error("Error muxing: %s — falling back to video-only", e)
            import shutil
            shutil.copy2(output_file, final_output)
            logger.info("Final file generated (video-only fallback): %s", final_output)
    else:
        # Attempt 3: video-only output (never drop the clip)
        logger.warning("No audio in source %s — outputting video-only", os.path.basename(input_file))
        import shutil
        shutil.copy2(output_file, final_output)
        logger.info("Final file generated (no audio): %s", final_output)
        try:
            os.remove(output_file)
        except OSError:
            pass


def calculate_mouth_ratio(landmarks: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) using 68-point landmarks (inner lips).
    Indices: 
    Inner Lips: 60-67 (0-indexed 60 to 67)
    Left Corner: 60
    Right Corner: 64
    Top Center: 62
    Bottom Center: 66
    """
    if landmarks is None:
        return 0
    
    # 3D points (x,y,z) or 2D (x,y). We use first 2 cols.
    pts = landmarks.astype(float)
    
    # Simple vertical vs horizontal
    # Vertical
    p62 = pts[62]
    p66 = pts[66]
    h = np.linalg.norm(p62[:2] - p66[:2])
    
    # Horizontal
    p60 = pts[60]
    p64 = pts[64]
    w = np.linalg.norm(p60[:2] - p64[:2])
    
    if w < 1e-6: return 0
    
    return h / w

def generate_short_mediapipe(input_file: str, output_file: str, index: int, face_mode: str, project_folder: str, final_folder: str, face_detection: object, face_mesh: object, pose: object, detection_period: float | None = None, no_face_mode: str = "padding", zoom_out_factor: float = 2.2) -> None:
    try:
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            logger.error(f"Error opening video: {input_file}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pipe_proc = create_ffmpeg_pipe(output_file, fps)

        next_detection_frame = 0
        current_interval = int(5 * fps) # Initial guess

        # Initial Interval Logic if predefined

        if detection_period is not None:
             current_interval = max(1, int(detection_period * fps))
        elif face_mode == "2":
             current_interval = int(1.0 * fps)
        
        coordinate_log: list[dict] = []
        last_detected_faces = None
        last_frame_face_positions = None
        last_success_frame = -1000
        max_frames_without_detection = int(3.0 * fps) # 3 seconds timeout

        transition_duration = int(fps)
        transition_frames = []

        # Load zoom_cues from viral_segments.txt
        segments_path = os.path.join(project_folder, "viral_segments.txt")
        zoom_cues = []
        if os.path.exists(segments_path):
            with open(segments_path, "r") as f:
                vs = json.load(f)
            if "segments" in vs and index < len(vs["segments"]):
                zoom_cues = vs["segments"][index].get("zoom_cues", [])

        for frame_index in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_index >= next_detection_frame:
                # Detect ALL faces (up to 2 in our implementation)
                small_mp, scale_mp = downscale_for_analysis(frame, max_width=480)
                detections = detect_face_or_body_two_faces(small_mp, face_detection, face_mesh, pose)
                # Scale coordinates back to original resolution
                if detections and scale_mp > 1.0:
                    detections = [(int(x * scale_mp), int(y * scale_mp), int(w * scale_mp), int(h * scale_mp)) for x, y, w, h in detections]
                
                # Dynamic Logic
                target_faces = 1
                if face_mode == "2":
                    target_faces = 2
                elif face_mode == "auto":
                    if detections and len(detections) >= 2:
                        target_faces = 2
                    else:
                        target_faces = 1
                
                # Filter detections based on target
                current_detections = []
                if detections:
                    # Sort detections by approximate Area (w*h) descending to pick main faces first
                    detections.sort(key=lambda s: s[2] * s[3], reverse=True)
                    
                    if len(detections) >= target_faces:
                        current_detections = detections[:target_faces]
                    elif len(detections) > 0:
                        # Fallback
                        current_detections = detections[:1] 
                        target_faces = 1 
                    
                    # Apply Consistency Check (Proximity)
                    if target_faces == 2 and len(current_detections) == 2:
                         if last_detected_faces is not None and len(last_detected_faces) == 2:
                             current_detections = sort_by_proximity(current_detections, last_detected_faces, get_center_rect)
                
                # Check for stability/lookahead could go here but skipping for brevity unless requested.
                
                if current_detections and len(current_detections) == target_faces:
                    if last_frame_face_positions is not None:
                        start_faces = np.array(last_frame_face_positions)
                        end_faces = np.array(current_detections)
                        try:
                            transition_frames = np.linspace(start_faces, end_faces, transition_duration, dtype=int)
                        except Exception as e:
                            # Fallback if shapes mismatch unexpectedly
                            transition_frames = []
                    else:
                        transition_frames = []
                    last_detected_faces = current_detections
                    last_success_frame = frame_index
                else:
                    pass
                
                # Update next detection frame
                step = 5
                
                if detection_period is not None:
                    if isinstance(detection_period, dict):
                         # If we are targeting 2 faces, we use '2' interval, else '1'
                         key = str(target_faces)
                         val = detection_period.get(key, detection_period.get('1', 0.2))
                         step = max(1, int(val * fps))
                    else:
                         step = max(1, int(detection_period * fps))
                elif target_faces == 2:
                    step = int(1.0 * fps)
                else:
                    step = int(5) # 5 frames for 1 face
                
                next_detection_frame = frame_index + step

            if len(transition_frames) > 0:
                current_faces = transition_frames[0]
                transition_frames = transition_frames[1:]
            elif last_detected_faces is not None and (frame_index - last_success_frame) <= max_frames_without_detection:
                current_faces = last_detected_faces
            else:
                if no_face_mode == "zoom":
                    result = crop_center_zoom(frame)
                else:
                    result = resize_with_padding(frame)
                coordinate_log.append({"frame": frame_index, "faces": []})
                try:
                    pipe_proc.stdin.write(result.tobytes())
                except (BrokenPipeError, OSError) as e:
                    logger.error("ffmpeg pipe broken: %s", e)
                    break
                continue

            last_frame_face_positions = current_faces

            if hasattr(current_faces, '__len__') and len(current_faces) == 2:
                 result = crop_and_resize_two_faces(frame, current_faces, zoom_out_factor=zoom_out_factor)
            else:
                 # Ensure it's list of tuples or single tuple? current_faces is list of tuples from detection
                 # If 1 face: [ (x,y,w,h) ]
                 if hasattr(current_faces, '__len__') and len(current_faces) > 0:
                     f = current_faces[0]
                     result = crop_and_resize_single_face(frame, f)
                 else:
                     if no_face_mode == "zoom":
                         result = crop_center_zoom(frame)
                     else:
                         result = resize_with_padding(frame)

            if zoom_cues:
                current_time = frame_index / fps
                result = apply_zoom_effect(result, current_time, zoom_cues)

            try:
                pipe_proc.stdin.write(result.tobytes())
            except (BrokenPipeError, OSError) as e:
                logger.error("ffmpeg pipe broken: %s", e)
                break

        cap.release()
        pipe_proc.stdin.close()
        pipe_proc.wait()

        finalize_video(input_file, output_file, index, fps, project_folder, final_folder)

    except Exception as e:
        logger.error(f"Error in MediaPipe processing: {e}")
        raise e # Rethrow to trigger fallback
    finally:
        if 'cap' in dir() and cap.isOpened():
            cap.release()
        if 'pipe_proc' in dir():
            if not pipe_proc.stdin.closed:
                pipe_proc.stdin.close()
            pipe_proc.wait()

def generate_short_haar(input_file: str, output_file: str, index: int, project_folder: str, final_folder: str, detection_period: float | None = None, no_face_mode: str = "padding") -> None:
    """Face detection using OpenCV Haar Cascades."""
    logger.info(f"Processing (Haar Cascade): {input_file}")
    
    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        logger.error("Error: Could not load Haar Cascade XML. Falling back to center crop.")
        generate_short_fallback(input_file, output_file, index, project_folder, final_folder)
        return

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        logger.error(f"Error opening video: {input_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pipe_proc = create_ffmpeg_pipe(output_file, fps)

    # Logic copied from generate_short_mediapipe
    detection_interval = int(2 * fps) # Default check every 2 seconds
    if detection_period is not None:
        detection_interval = max(1, int(detection_period * fps))
    last_detected_faces = None
    last_frame_face_positions = None
    last_success_frame = -1000
    max_frames_without_detection = int(3.0 * fps)

    transition_duration = int(fps) # 1 second smooth transition
    transition_frames = []

    try:
        for frame_index in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_index % detection_interval == 0:
                small_haar, scale_haar = downscale_for_analysis(frame, max_width=480)
                gray = cv2.cvtColor(small_haar, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                detections = []
                if len(faces) > 0:
                    # Pick largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    # Scale coordinates back to original resolution
                    if scale_haar > 1.0:
                        largest_face = tuple(int(c * scale_haar) for c in largest_face)
                    # Ensure int type
                    detections = [tuple(map(int, largest_face))]

                if detections:
                    if last_frame_face_positions is not None:
                        # Simple linear interpolation for smoothing
                        start_faces = np.array(last_frame_face_positions)
                        end_faces = np.array(detections)

                        # Generate transition frames
                        steps = transition_duration
                        transition_frames = []
                        for s in range(steps):
                            t = (s + 1) / steps
                            interp = (1 - t) * start_faces + t * end_faces
                            transition_frames.append(interp.astype(int).tolist()) # Convert back to list of lists/tuples
                    else:
                        transition_frames = []
                    last_detected_faces = detections
                    last_success_frame = frame_index
                else:
                    pass

            if len(transition_frames) > 0:
                current_faces = transition_frames[0]
                transition_frames = transition_frames[1:]
            elif last_detected_faces is not None and (frame_index - last_success_frame) <= max_frames_without_detection:
                current_faces = last_detected_faces
            else:
                # No face detected for a while -> Center/Padding fallback
                if no_face_mode == "zoom":
                    result = crop_center_zoom(frame)
                else:
                    result = resize_with_padding(frame)
                try:
                    pipe_proc.stdin.write(result.tobytes())
                except (BrokenPipeError, OSError) as e:
                    logger.error("ffmpeg pipe broken: %s", e)
                    break
                continue

            last_frame_face_positions = current_faces
            # haar detections are list containing one tuple (x,y,w,h)
            # current_faces is list of one tuple
            if isinstance(current_faces, list):
                 face_bbox = current_faces[0]
            else:
                 face_bbox = current_faces # Should be handled

            result = crop_and_resize_single_face(frame, face_bbox)
            try:
                pipe_proc.stdin.write(result.tobytes())
            except (BrokenPipeError, OSError) as e:
                logger.error("ffmpeg pipe broken: %s", e)
                break

        cap.release()
        if not pipe_proc.stdin.closed:
            pipe_proc.stdin.close()
        pipe_proc.wait()

        finalize_video(input_file, output_file, index, fps, project_folder, final_folder)
    finally:
        if cap.isOpened():
            cap.release()
        if not pipe_proc.stdin.closed:
            pipe_proc.stdin.close()
        pipe_proc.wait()

def generate_short_insightface(input_file: str, output_file: str, index: int, project_folder: str, final_folder: str, face_mode: str = "auto", detection_period: float | dict | None = None, filter_threshold: float = 0.35, two_face_threshold: float = 0.60, confidence_threshold: float = 0.30, dead_zone: int = 40, focus_active_speaker: bool = False, active_speaker_mar: float = 0.03, active_speaker_score_diff: float = 1.5, include_motion: bool = False, active_speaker_motion_deadzone: float = 3.0, active_speaker_motion_sensitivity: float = 0.05, active_speaker_decay: float = 2.0, no_face_mode: str = "padding", zoom_out_factor: float = 2.2, ema_alpha: float = 0.18, detection_resolution: int = 480, vertical_offset: float = 0.0, single_face_zoom: float = 1.0) -> str:
    """Face detection using InsightFace (SOTA)."""
    logger.info(f"Processing (InsightFace): {input_file} | Mode: {face_mode}")
    
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        logger.error(f"Error opening video: {input_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    pipe_proc = create_ffmpeg_pipe(output_file, fps)

    # Dynamic Interval Logic
    next_detection_frame = 0

    last_detected_faces = None
    last_frame_face_positions = None
    last_success_frame = -1000
    max_frames_without_detection = int(3.0 * fps) # 3 seconds timeout

    smooth_bbox_centers = None   # FIX 0B: list of (cx, cy, w_half, h_half) — EMA state per face (center + dimensions)
    EMA_ALPHA = ema_alpha        # FIX: use parameter instead of hardcoded value
    NOISE_THRESHOLD = dead_zone if dead_zone > 0 else 30  # FIX 0A: use dead_zone parameter instead of hardcoded 30
    INTERP_ALPHA = 0.3           # FIX 0C: frame-by-frame interpolation factor (0.2=smooth, 0.5=responsive)
    render_faces = None          # FIX 0C: current render position (interpolated towards last_detected_faces)

    # Current state of face mode (1 or 2)
    # If auto, we decide per detection interval
    current_num_faces_state = 1
    if face_mode == "2":
        current_num_faces_state = 2

    # Smart region tracking state (saliency / motion fallback)
    smart_cx = frame_width // 2
    smart_cy = frame_height // 2
    smart_prev_gray = None
    smart_saliency = None
    if no_face_mode == "saliency":
        smart_saliency = cv2.saliency.StaticSaliencySpectralResidual.create()

    frame_1_face_count = 0
    frame_2_face_count = 0

    buffered_frame = None
    
    # Timeline tracking: list of (frame_index, mode_str)
    # We will compress this later.
    timeline_frames = [] # Store mode for *every written frame* or at least detection points
    coordinate_log = [] # Store raw face coordinates frame-by-frame
    
    # For Active Speaker Logic
    # Map of "Face ID" to activity score?
    # Since we don't have ID tracker, we blindly assign score to faces based on proximity to previous frame
    # A list of dictionaries: [{'center': (x,y), 'activity': score}, ...]
    faces_activity_state = []

    # FEAT: Re-ID — lock onto the first detected person via embedding
    target_embedding = None

    # Load zoom_cues from viral_segments.txt
    segments_path = os.path.join(project_folder, "viral_segments.txt")
    zoom_cues = []
    if os.path.exists(segments_path):
        with open(segments_path, "r") as f:
            vs = json.load(f)
        if "segments" in vs and index < len(vs["segments"]):
            zoom_cues = vs["segments"][index].get("zoom_cues", [])

    for frame_index in range(total_frames):
        if buffered_frame is not None:
             frame = buffered_frame
             ret = True
             buffered_frame = None
        else:
             ret, frame = cap.read()

        if not ret or frame is None:
            break

        if frame_index >= next_detection_frame:
            # Detect faces on downscaled frame for performance
            small_if, scale_if = downscale_for_analysis(frame, max_width=detection_resolution)  # FIX: use detection_resolution parameter
            faces = detect_faces_insightface(small_if)
            # Scale bounding boxes and landmarks back to original resolution
            if faces:
                for f in faces:
                    if scale_if > 1.0:
                        f['bbox'] = (f['bbox'] * scale_if).astype(int)
                        if 'landmark_3d_68' in f and f['landmark_3d_68'] is not None:
                            f['landmark_3d_68'][:, :2] *= scale_if
                        if 'landmark_2d_106' in f and f['landmark_2d_106'] is not None:
                            f['landmark_2d_106'][:, :2] *= scale_if
                    else:
                        # InsightFace returns float32 bbox — ensure int for array slicing
                        f['bbox'] = f['bbox'].astype(int)
            if faces:
                scores = [f"{f.get('det_score',0):.2f}" for f in faces]
                logger.debug(f"DEBUG: Frame {frame_index} | Raw Faces: {len(faces)} | Scores: {scores}")
            else:
                pass # logger.debug(f"DEBUG: Frame {frame_index} | No Raw Faces")

            # --- ACTIVITY / SPEAKER DETECTION ---
            # (Feature currently disabled for stability - relying on simple size checks)
            last_raw_faces = faces 
            # ------------------------------------

            # --- INTELLIGENT FILTERING ---
            valid_faces = []
            if faces:
                # 1. Filter by confidence (Using user threshold)
                faces = [f for f in faces if f.get('det_score', 0) > confidence_threshold]
                
                if faces:
                    # Pre-calculate areas and SPEAKER SCORE
                    for f in faces:
                        w = f['bbox'][2] - f['bbox'][0]
                        h = f['bbox'][3] - f['bbox'][1]
                        f['area'] = w * h
                        f['center'] = ((f['bbox'][0] + f['bbox'][2]) / 2, (f['bbox'][1] + f['bbox'][3]) / 2)
                        
                        act = f.get('activity', 0)
                        f['effective_area'] = f['area'] * (1.0 + (act * 0.05))

                    # Find largest face
                    max_area = max(f['area'] for f in faces)
                    
                    # 2. Relative Size Filter
                    valid_faces = [f for f in faces if f['area'] > (filter_threshold * max_area)]
                    
                    if len(valid_faces) < len(faces):
                        logger.debug(f"DEBUG: Filtered {len(faces)-len(valid_faces)} small faces. Max Area: {max_area}. Filter Thresh: {filter_threshold}")
                    
                    faces = valid_faces
            
            # --- ACTIVE SPEAKER UPDATE ---
            if faces:
                # 1. Update activity scores for current faces
                # Simple matching to previous state
                current_state_map = []
                
                for f in faces:
                    # Calculate instantaneous openness
                    mar = 0
                    if 'landmark_3d_68' in f:
                        mar = calculate_mouth_ratio(f['landmark_3d_68'])
                    elif 'landmark_2d_106' in f:
                        # Fallback or Todo: map 106 to 68 approximate
                        # 106 indices: 52-71 are lips.
                        # Inner roughly 64-71?
                        # Let's rely on 3d_68 which is standard in buffalo_l
                        pass
                    
                    f['mouth_ratio'] = mar
                    # Heuristic: Ratio > 0.05 implies openish, > 0.1 talk.
                    # Adjust thresholds: 0.03 is common for closed mouth, 0.05 is starting to open.
                    
                    # Log raw MAR for debugging
                    # print(f"DEBUG: Frame {frame_index} Face {i} MAR: {mar:.4f}")
                    
                    is_talking = 1.0 if mar > active_speaker_mar else 0.0 
                    

            # --- CROWD MODE LOGIC ---
            # If too many faces, don't even try to track. Fallback to No-Face logic (Zoom/Padding)
            CROWD_THRESHOLD = 7 
            # FIX: Use last_raw_faces (before size filtering) so we count background people too!
            is_crowd = len(last_raw_faces) >= CROWD_THRESHOLD
            if is_crowd:
                logger.debug(f"DEBUG: Crowd Mode Active! {len(faces)} faces >= {CROWD_THRESHOLD}. Triggering Fallback (No Face Mode).")
                faces = [] 
                valid_faces = [] # CAUTION: Must clear strict backup too!
                # FORCE RESET HISTORY so it doesn't "stick" to the last face found
                last_detected_faces = None
                smooth_bbox_centers = None  # Reset EMA state too
                render_faces = None         # FIX 0C: Reset interpolation state too
                faces_activity_state = []
                zoom_ema_bbox = None # Reset smoothing too
            # ---------------------------

            # Update Activity State - Two Pass for Global Motion Compensation
            if focus_active_speaker and faces:
                # Pass 1: Global Motion (Camera Shake) Calculation
                # We calculate motion for ALL confident faces (before size filtering) to get best global estimate
                raw_motions = []
                
                # First, ensure we have a temporary mapping of current faces to history
                # We do this non-destructively just to get motion values
                for f in faces:
                    my_c = f['center']
                    best_dist = 9999
                    if faces_activity_state:
                         for old_s in faces_activity_state:
                             old_c = old_s['center']
                             dist = np.sqrt((my_c[0]-old_c[0])**2 + (my_c[1]-old_c[1])**2)
                             if dist < best_dist:
                                 best_dist = dist
                    
                    if best_dist < 200:
                        f['_raw_motion'] = best_dist
                    else:
                        f['_raw_motion'] = 0.0
                    
                    if include_motion:
                        raw_motions.append(f['_raw_motion'])

                global_motion = 0.0
                if include_motion and len(raw_motions) >= 2:
                    global_motion = min(raw_motions)

                # Pass 2: Update Scores for ALL faces
                current_state_map = []
                for f in faces:
                     # Helper: Is talking?
                     is_talking = f.get('mouth_ratio', 0) > active_speaker_mar
                     
                     # Calculate Compensated Motion
                     motion_bonus = 0.0
                     if include_motion and faces_activity_state:
                         comp_motion = max(0.0, f.get('_raw_motion', 0.0) - global_motion)
                         f['motion_val'] = comp_motion # Store for debug
                         
                         if comp_motion > active_speaker_motion_deadzone:
                              motion_bonus = min(2.5, (comp_motion - active_speaker_motion_deadzone) * active_speaker_motion_sensitivity)
                     else:
                        f['motion_val'] = 0.0
                     
                     # Accumulate Score
                     matched_score = 0.0
                     
                     # Re-find match to update history
                     my_c = f['center']
                     best_dist = 9999
                     best_idx = -1
                     if faces_activity_state:
                         for i, old_s in enumerate(faces_activity_state):
                             old_c = old_s['center']
                             dist = np.sqrt((my_c[0]-old_c[0])**2 + (my_c[1]-old_c[1])**2)
                             if dist < best_dist:
                                 best_dist = dist
                                 best_idx = i
                     
                     if best_idx != -1 and best_dist < 200:
                         old_val = faces_activity_state[best_idx]['activity']
                         change = -abs(active_speaker_decay)
                         if is_talking:
                             change = 1.5
                         
                         new_val = old_val + change + motion_bonus
                         # Increased cap to 20.0 to allow motion differences to separate two 'talking' faces
                         matched_score = max(0.0, min(20.0, new_val))
                     else:
                         matched_score = 1.0 if is_talking else 0.0
                     
                     f['activity_score'] = matched_score
                     current_state_map.append({'center': f['center'], 'activity': matched_score})
                 
                faces_activity_state = current_state_map
            else:
                faces_activity_state = []

            faces = valid_faces
            
            # Decide 1 or 2 faces
            target_faces = 1
            if face_mode == "2":
                target_faces = 2
            elif face_mode == "auto":
                if len(faces) >= 2:
                    # Default decision variable
                    decided = False
                    
                    if focus_active_speaker:
                         # EXPERIMENTAL: Decide based on activity
                         f1 = faces[0]
                         f2 = faces[1]
                         score1 = f1.get('activity_score', 0)
                         score2 = f2.get('activity_score', 0)
                         
                         y1 = f1['center'][1]
                         y2 = f2['center'][1]
                         pos1 = "Top" if y1 < y2 else "Bottom"
                         pos2 = "Top" if y2 < y1 else "Bottom"
                         
                         # Debug Active Speaker
                         logger.debug(f"DEBUG: Frame {frame_index} | {pos1} (MAR: {f1.get('mouth_ratio',0):.3f}, Mov: {f1.get('motion_val',0):.1f}, Score: {score1:.1f}) | {pos2} (MAR: {f2.get('mouth_ratio',0):.3f}, Mov: {f2.get('motion_val',0):.1f}, Score: {score2:.1f})")


                         # If one is clearly dominant active speaker
                         # Lower threshold to make it more sensitive?
                         # Score difference > 2.0 (approx 2-3 frames of talking difference vs silence)
                         diff = abs(score1 - score2)
                         # Check strict dominance first
                         if diff > active_speaker_score_diff:
                             # Pick the winner
                             target_faces = 1
                             decided = True
                             # Ensure the list is sorted by activity so [0] is the winner
                             if score2 > score1:
                                 # Swap ensures [0] is the active one for later 1-face crop logic which takes [0]
                                 faces = [f2, f1]
                             logger.debug(f"DEBUG: Active Speaker Focus Triggered! Diff ({diff:.2f}) > Thresh ({active_speaker_score_diff}). Focusing on Face {'2' if score2 > score1 else '1'}.")
                             
                         elif score1 > 4.0 and score2 > 4.0:
                             # Both talking -> 2 faces
                             # Raised threshold to 4.0 to avoid noise triggering split
                             target_faces = 2
                             decided = True
                             logger.debug(f"DEBUG: Dual Active Speakers! Both scores > 4.0. Forcing Split Mode.")
                         
                         # If scores are low (both silent), fallback to size ratio (decided=False) or force 1 if very silent?
                         # Let's fallback to size.

                    if not decided:
                        # Standard Logic: Check relative sizes (effective area)
                        faces_sorted_temp = sorted(faces, key=lambda f: f.get('effective_area', 0), reverse=True)
                        largest = faces_sorted_temp[0]['effective_area']
                        second = faces_sorted_temp[1]['effective_area']
    
                        # Two-Face Constraint
                        if second > (two_face_threshold * largest):
                            target_faces = 2
                        else:
                            target_faces = 1
                else:
                    target_faces = 1
            
            # If no faces found effectively after filter
            if not faces and not valid_faces:
                 # Logic ensures faces = valid_faces already
                 pass
            
            # -----------------------------
            
            # Fallback Lookahead: If detection fails or partial
            # But DO NOT look ahead if we are in Crowd Mode (we explicitly wanted 0 faces)
            if len(faces) < target_faces and not is_crowd:
                # Try 1 frame ahead
                ret2, frame2 = cap.read()
                if ret2 and frame2 is not None:
                     small_if2, scale_if2 = downscale_for_analysis(frame2, max_width=480)
                     faces2 = detect_faces_insightface(small_if2)
                     # Scale bounding boxes back to original resolution
                     if faces2 and scale_if2 > 1.0:
                         for f in faces2:
                             f['bbox'] = (f['bbox'] * scale_if2).astype(int)  # FIX: ensure int for array slicing
                             if 'landmark_3d_68' in f and f['landmark_3d_68'] is not None:
                                 f['landmark_3d_68'][:, :2] *= scale_if2
                             if 'landmark_2d_106' in f and f['landmark_2d_106'] is not None:
                                 f['landmark_2d_106'][:, :2] *= scale_if2
                     elif faces2:
                         # FIX: InsightFace returns float32 bbox — ensure int even without scaling
                         for f in faces2:
                             f['bbox'] = f['bbox'].astype(int)
                     
                     # --- Apply same filtering to lookahead ---
                     valid_faces2 = []
                     if faces2:
                         faces2 = [f for f in faces2 if f.get('det_score', 0) > 0.50]
                         if faces2:
                             for f in faces2:
                                 w = f['bbox'][2] - f['bbox'][0]
                                 h = f['bbox'][3] - f['bbox'][1]
                                 f['area'] = w * h
                                 f['center'] = ((f['bbox'][0] + f['bbox'][2]) / 2, (f['bbox'][1] + f['bbox'][3]) / 2)
                                 f['effective_area'] = f['area'] # Default for lookahead
                             max_area2 = max(f['area'] for f in faces2)
                             # STRICTER FILTER: threshold of max area
                             valid_faces2 = [f for f in faces2 if f['area'] > (filter_threshold * max_area2)]
                     faces2 = valid_faces2
                     # ----------------------------------------


                     # If lookahead found what we wanted OR found something better than nothing
                     if len(faces2) >= target_faces:
                         faces = faces2 # Use lookahead faces for current frame
                     elif len(faces) == 0 and len(faces2) > 0:
                         faces = faces2 # Better than nothing
                         
                     buffered_frame = frame2 # Store for next iteration

            detections = []

            # FEAT: Re-ID scoring — compute cosine similarity to locked target
            if target_embedding is not None and faces:
                for f in faces:
                    emb = get_face_embedding(f)
                    if emb is not None:
                        f['reid_score'] = cosine_similarity(target_embedding, emb)
                    else:
                        f['reid_score'] = 0.0

            if len(faces) >= target_faces:
                # FEAT: Re-ID — capture target embedding from first valid detection
                if target_embedding is None and target_faces == 1:
                    # Lock onto the first detected face (largest by area)
                    best_face = max(faces, key=lambda f: f.get('effective_area', 0))
                    emb = get_face_embedding(best_face)
                    if emb is not None:
                        target_embedding = emb
                        logger.info("FEAT: Re-ID target locked (512-D embedding captured)")

                # --- FACE TRACKING / SORTING ---
                # Instead of just Area, we prioritize faces closer to the LAST detected face
                # This prevents switching to a background person if sizes are similar

                # FEAT: Re-ID preferred face selection (1-face mode only)
                # INVARIANT: Re-ID multi-face lock ONLY applies when target_faces == 1.
                # When target_faces == 2, detections are handled by the proximity/area sort below.
                if target_embedding is not None and target_faces == 1:
                    if len(faces) >= 2:
                        # Multi-face: strict Re-ID lock to prevent oscillation
                        best_reid = max(faces, key=lambda f: f.get('reid_score', 0))
                        if best_reid.get('reid_score', 0) > 0.45:  # was 0.35 — stricter to prevent oscillation
                            detections = [best_reid['bbox']]
                            faces_sorted_by_reid = True
                        else:
                            # Fallback to proximity sort (below)
                            faces_sorted_by_reid = False
                    else:
                        # Single face: keep existing logic
                        faces_with_reid = [f for f in faces if f.get('reid_score', 0) > 0.5]
                        if faces_with_reid:
                            faces_sorted = sorted(faces_with_reid, key=lambda f: f['reid_score'], reverse=True)
                            faces_sorted_by_reid = True
                        else:
                            faces_sorted_by_reid = False
                else:
                    faces_sorted_by_reid = False

                if not faces_sorted_by_reid and last_detected_faces is not None and len(last_detected_faces) == target_faces:
                   # Define score function: High Area is good, Low Distance to old is good.
                   # But simpler: calculate Intersection over Union (IOU) or Distance to old bbox center

                   # We want to match existing slots.
                   # For 1 face:
                   if target_faces == 1:
                       old_center = get_center_bbox(last_detected_faces[0])

                       def sort_score(f):
                           # Distance score (lower is better)
                           dist = np.sqrt((f['center'][0] - old_center[0])**2 + (f['center'][1] - old_center[1])**2)
                           # Continuity bonus: strongly prefer the face closest to last tracked position
                           continuity = -150 if dist < 150 else 0  # was -80/80 — stronger bias toward current face
                           # EFFECTIVE Area score (higher is better)
                           return dist + continuity - (f['effective_area'] * 0.0001)

                       faces_sorted = sorted(faces, key=sort_score)
                   else:
                       # For 2 faces, just sort by effective area for now as proximity sort happens later
                       faces_sorted = sorted(faces, key=lambda f: f['effective_area'], reverse=True)
                elif not faces_sorted_by_reid:
                   # No history, sort by effective area
                   if focus_active_speaker and target_faces == 1:
                        # Pick the one with highest activity score
                        faces_sorted = sorted(faces, key=lambda f: f.get('activity_score', 0), reverse=True)
                   else:
                        faces_sorted = sorted(faces, key=lambda f: f.get('effective_area', 0), reverse=True)
                
                if not detections:
                    # detections not yet set by Re-ID lock — use sorted faces
                    if target_faces == 2:
                        # Convert [x1, y1, x2, y2] to (x, y, w, h) logic is later
                        # Ensure we have 2 faces
                        f1 = faces_sorted[0]['bbox']
                        f2 = faces_sorted[1]['bbox']

                        if last_detected_faces is not None and len(last_detected_faces) == 2:
                            detections = sort_by_proximity([f1, f2], last_detected_faces, get_center_bbox)
                        else:
                            detections = [f1, f2]

                        current_num_faces_state = 2
                    else:
                        # 1 face
                        detections = [faces_sorted[0]['bbox']]
                        current_num_faces_state = 1
                else:
                    # Re-ID already locked detections
                    current_num_faces_state = 1
            else:
                 # If we wanted 2 but found 1, or wanted 1 found 0
                 if len(faces) > 0:
                     # Fallback to 1 face if found at least 1
                     faces_sorted = sorted(faces, key=lambda f: f['effective_area'], reverse=True)
                     detections = [faces_sorted[0]['bbox']]
                     current_num_faces_state = 1
                 else:
                     detections = []

            # FEAT: Save previous detections for adaptive interval drift check
            prev_detected_faces = last_detected_faces

            if detections:
                # --- EMA SMOOTHING (FIX 0B: smooth center + dimensions) ---
                new_states = []  # FIX 0B: (cx, cy, w_half, h_half) per detection
                for d in detections:
                    if isinstance(d, np.ndarray):
                        x1, y1, x2, y2 = d[:4]
                    else:
                        x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    wh = (x2 - x1) / 2.0
                    hh = (y2 - y1) / 2.0
                    new_states.append((cx, cy, wh, hh))

                if smooth_bbox_centers is not None and len(smooth_bbox_centers) == len(new_states):
                    smoothed = []
                    for (old_cx, old_cy, old_wh, old_hh), (new_cx, new_cy, new_wh, new_hh) in zip(smooth_bbox_centers, new_states):
                        dist = ((old_cx - new_cx)**2 + (old_cy - new_cy)**2) ** 0.5
                        if dist < NOISE_THRESHOLD:
                            # FIX 0B: still smooth dimensions even if center is in dead zone
                            s_wh = EMA_ALPHA * new_wh + (1 - EMA_ALPHA) * old_wh
                            s_hh = EMA_ALPHA * new_hh + (1 - EMA_ALPHA) * old_hh
                            smoothed.append((old_cx, old_cy, s_wh, s_hh))
                        else:
                            s_cx = EMA_ALPHA * new_cx + (1 - EMA_ALPHA) * old_cx
                            s_cy = EMA_ALPHA * new_cy + (1 - EMA_ALPHA) * old_cy
                            s_wh = EMA_ALPHA * new_wh + (1 - EMA_ALPHA) * old_wh
                            s_hh = EMA_ALPHA * new_hh + (1 - EMA_ALPHA) * old_hh
                            smoothed.append((s_cx, s_cy, s_wh, s_hh))
                    smooth_bbox_centers = smoothed

                    # FIX 0B: Rebuild bboxes with smoothed centers AND smoothed dimensions
                    smoothed_detections = []
                    for (s_cx, s_cy, s_wh, s_hh) in smooth_bbox_centers:
                        smoothed_detections.append(np.array([
                            int(s_cx - s_wh), int(s_cy - s_hh),
                            int(s_cx + s_wh), int(s_cy + s_hh)
                        ]))
                    detections = smoothed_detections
                else:
                    # First detection or face count changed: init EMA state
                    # FIX 0B: store (cx, cy, w_half, h_half) instead of just (cx, cy)
                    smooth_bbox_centers = list(new_states)

                last_detected_faces = detections
                last_success_frame = frame_index
            else:
                pass


            # Update next detection frame based on NEW state
            step = 5 # Default fallback (very fast)
            
            if detection_period is not None:
                if isinstance(detection_period, dict):
                    # Period depends on state
                    key = str(current_num_faces_state) 
                    # fallback to '1' if key not found (should be there)
                    val = detection_period.get(key, detection_period.get('1', 0.2)) 
                    step = max(1, int(val * fps))
                else:
                    # Legacy float support (should not happen with new main.py but good safety)
                    step = max(1, int(detection_period * fps))
            elif current_num_faces_state == 2:
                step = int(0.5 * fps)  # 0.5s for 2 faces (was 1.0s)
            else:
                step = 5 # 5 frames for 1 face (~0.16s at 30fps)

            # Minimum step when multiple faces detected to prevent oscillation
            if len(faces) >= 2 and current_num_faces_state == 1:
                step = max(step, int(0.5 * fps))  # was 0.4 — more stable transitions

            # FEAT: Adaptive detection interval — override step based on tracking stability
            if not detections:
                # Lost face — detect more aggressively to recover fast
                step = max(1, step // 2)
            elif prev_detected_faces is not None and len(prev_detected_faces) > 0 and len(detections) > 0:
                # Check positional stability between consecutive detections
                new_center = get_center_bbox(detections[0])
                old_center = get_center_bbox(prev_detected_faces[0])
                drift = ((new_center[0] - old_center[0])**2 + (new_center[1] - old_center[1])**2) ** 0.5
                if drift < 50:
                    step = min(15, step + 1)  # Stable — slow down detection (save GPU)
                else:
                    step = 5  # Moving — standard rate

            next_detection_frame = frame_index + step

        if last_detected_faces is not None and (frame_index - last_success_frame) <= max_frames_without_detection:
            # FIX 0C: Interpolate render_faces towards last_detected_faces every frame
            if render_faces is None or len(render_faces) != len(last_detected_faces):
                # First frame or face count changed: snap to target
                render_faces = [np.array(f, dtype=float) for f in last_detected_faces]
            else:
                interpolated = []
                for rf, tf in zip(render_faces, last_detected_faces):
                    rf_f = np.array(rf, dtype=float)
                    tf_f = np.array(tf, dtype=float)
                    rf_f = rf_f + INTERP_ALPHA * (tf_f - rf_f)  # FIX 0C: linear interpolation per frame
                    interpolated.append(rf_f)
                render_faces = interpolated
            # FIX 0C: use interpolated render_faces for crop instead of raw detection
            current_faces = [np.array(rf, dtype=int) for rf in render_faces]
        else:
            # Fallback for this frame
            if no_face_mode == "zoom":
                result = crop_center_zoom(frame)
            elif no_face_mode == "saliency":
                _, sal_map = smart_saliency.computeSaliency(frame)
                sal_map = (sal_map * 255).astype(np.uint8)
                m = cv2.moments(sal_map)
                if m['m00'] > 0:
                    cx = int(m['m10'] / m['m00'])
                    cy = int(m['m01'] / m['m00'])
                    smart_cx = int(smart_cx * 0.90 + cx * 0.10)
                    smart_cy = int(smart_cy * 0.90 + cy * 0.10)
                result = crop_to_smart_region(frame, smart_cx, smart_cy)
            elif no_face_mode == "motion":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if smart_prev_gray is not None:
                    diff = cv2.absdiff(smart_prev_gray, gray)
                    m = cv2.moments(diff)
                    if m['m00'] > 500:
                        cx = int(m['m10'] / m['m00'])
                        cy = int(m['m01'] / m['m00'])
                        smart_cx = int(smart_cx * 0.85 + cx * 0.15)
                        smart_cy = int(smart_cy * 0.85 + cy * 0.15)
                smart_prev_gray = gray
                result = crop_to_smart_region(frame, smart_cx, smart_cy)
            else:
                result = resize_with_padding(frame)
            try:
                pipe_proc.stdin.write(result.tobytes())
            except (BrokenPipeError, OSError) as e:
                logger.error("ffmpeg pipe broken: %s", e)
                break
            timeline_frames.append((frame_index, "1")) # Fix: Ensure fallback is treated as single face for subs

            # Fix XML Log sync (Empty faces for fallback)
            coords_entry = {"frame": frame_index, "src_size": [frame_width, frame_height], "faces": []}
            coordinate_log.append(coords_entry)
            
            continue

        last_frame_face_positions = current_faces
        
        target_len = len(current_faces)
        
        if target_len == 2:
             frame_2_face_count += 1
             # Convert [x1, y1, x2, y2] to (x, y, w, h)
             f1 = current_faces[0]
             f2 = current_faces[1]
             rect1 = (f1[0], f1[1], f1[2]-f1[0], f1[3]-f1[1])
             rect2 = (f2[0], f2[1], f2[2]-f2[0], f2[3]-f2[1])
             result = crop_and_resize_two_faces(frame, [rect1, rect2], zoom_out_factor=zoom_out_factor)
             timeline_frames.append((frame_index, "2"))
        else:
             frame_1_face_count += 1
             # 1 face
             # current_faces[0] is [x1, y1, x2, y2]
             result = crop_and_resize_insightface(frame, current_faces[0], vertical_offset=vertical_offset, single_face_zoom=single_face_zoom)  # FIX: pass visual params
             timeline_frames.append((frame_index, "1"))
             
        # Capture Coordinates (Frame-by-Frame)
        coords_entry = {"frame": frame_index, "src_size": [frame_width, frame_height], "faces": []}
        try:
            # We want to store [x1, y1, x2, y2, rh] for each face
            if isinstance(current_faces, (list, tuple)):
                processed_faces_log = []
                for f in current_faces:
                    f_list = list(map(int, f[:4])) # Standard bbox
                    # Calculate rh (relative height)
                    face_h = f_list[3] - f_list[1]
                    rh = face_h / float(frame_height)
                    f_list.append(float(f"{rh:.4f}")) # Append as 5th element
                    processed_faces_log.append(f_list)
                coords_entry["faces"] = processed_faces_log
                
            elif isinstance(current_faces, np.ndarray):
                # Similar logic for numpy
                processed_faces_log = []
                for f in current_faces:
                    f_list = f[:4].astype(int).tolist()
                    face_h = f_list[3] - f_list[1]
                    rh = face_h / float(frame_height)
                    f_list.append(float(f"{rh:.4f}"))
                    processed_faces_log.append(f_list)
                coords_entry["faces"] = processed_faces_log
        except (ValueError, IndexError, AttributeError):
            pass  # coords logging is best-effort
        coordinate_log.append(coords_entry)

        if zoom_cues:
            current_time = frame_index / fps
            result = apply_zoom_effect(result, current_time, zoom_cues)

        try:
            pipe_proc.stdin.write(result.tobytes())
        except (BrokenPipeError, OSError) as e:
            logger.error("ffmpeg pipe broken: %s", e)
            break

    cap.release()
    if not pipe_proc.stdin.closed:
        pipe_proc.stdin.close()
    pipe_proc.wait()

    # Compress timeline into segments
    # [(start_time, end_time, mode), ...]
    compressed_timeline = []
    if timeline_frames:
        curr_mode = timeline_frames[0][1]
        start_f = timeline_frames[0][0]
        
        for i in range(1, len(timeline_frames)):
            frame_idx, mode = timeline_frames[i]
            if mode != curr_mode:
                # End current segment
                # Convert frame to seconds
                end_f = timeline_frames[i-1][0]
                compressed_timeline.append({
                    "start": float(start_f) / fps,
                    "end": float(end_f) / fps, # or frame_idx / fps for continuity
                    "mode": curr_mode
                })
                # Start new
                curr_mode = mode
                start_f = frame_idx
        
        # Add last
        end_f = timeline_frames[-1][0]
        compressed_timeline.append({
             "start": float(start_f) / fps,
             "end": (float(end_f) + 1) / fps,
             "mode": curr_mode
        })
    
    # Save timeline JSON
    timeline_file = output_file.replace(".mp4", "_timeline.json")
    try:
        with open(timeline_file, "w") as f:
            json.dump(compressed_timeline, f)
        logger.info(f"Timeline saved: {timeline_file}")
    except Exception as e:
        logger.error(f"Error saving timeline: {e}")

    # Save Coords JSON
    coords_file = output_file.replace(".mp4", "_coords.json")
    try:
        with open(coords_file, "w") as f:
            json.dump(coordinate_log, f)
        logger.info(f"Face Coordinates saved: {coords_file}")
    except Exception as e:
        logger.error(f"Error saving coords: {e}")

    finalize_video(input_file, output_file, index, fps, project_folder, final_folder)
    
    # Return dominant mode logic (or keep 15% rule as overall fallback)
    if frame_2_face_count > (total_frames * 0.15):
        return "2"
    return "1"


def edit(project_folder: str = "tmp", face_model: str = "insightface", face_mode: str = "auto", detection_period: float | dict | None = None, filter_threshold: float = 0.35, two_face_threshold: float = 0.60, confidence_threshold: float = 0.30, dead_zone: int = 40, focus_active_speaker: bool = False, active_speaker_mar: float = 0.03, active_speaker_score_diff: float = 1.5, include_motion: bool = False, active_speaker_motion_deadzone: float = 3.0, active_speaker_motion_sensitivity: float = 0.05, active_speaker_decay: float = 2.0, segments_data: dict | None = None, no_face_mode: str = "padding", zoom_out_factor: float = 2.2, ema_alpha: float = 0.18, detection_resolution: int = 480, vertical_offset: float = 0.0, single_face_zoom: float = 1.0) -> None:
    # Lazy init solutions only when needed to avoid AttributeError if import failed partially
    mp_face_detection = None
    mp_face_mesh = None
    mp_pose = None
    
    index = 0
    cuts_folder = os.path.join(project_folder, "cuts")
    final_folder = os.path.join(project_folder, "final")
    os.makedirs(final_folder, exist_ok=True)
    
    face_modes_log = {}
    
    # Priority: User Choice -> Fallbacks
    
    insightface_working = False
    
    # Only init InsightFace if selected or default
    if INSIGHTFACE_AVAILABLE and (face_model == "insightface"):
        try:
            logger.info("Initializing InsightFace...")
            init_insightface()
            insightface_working = True
            logger.info("InsightFace Initialized Successfully.")
        except Exception as e:
            logger.warning(f"InsightFace Initialization Failed ({e}). Will try MediaPipe.")
            insightface_working = False

    mediapipe_working = False
    use_haar = False
    
    # If insightface failed OR user chose mediapipe, init mediapipe
    should_use_mediapipe = (face_model == "mediapipe") or (face_model == "insightface" and not insightface_working)
    
    if should_use_mediapipe:
        try:
            # Check if solutions is available (it might not be if import failed silently or partial)
            if not hasattr(mp, 'solutions'):
                raise ImportError("mediapipe.solutions not found")
                
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
            mp_pose = mp.solutions.pose
            
            # Try to init with model_selection=0 (Short Range) as a smoketest
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
                pass
            mediapipe_working = True
            logger.info("MediaPipe Initialized Successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe Initialization Failed ({e}). Switching to OpenCV Haar Cascade.")
            mediapipe_working = False
            use_haar = True
    
    # Logic for MediaPipe replaced by dynamic pass
    # mp_num_faces = 2 if face_mode == "2" else 1  

    import glob
    found_files = sorted(glob.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))

    if not found_files:
        logger.warning(f"No files found in {cuts_folder}.")
        # Try finding lookahead in case listdir failed? No, glob is fine.
        return

    # Pre-create MediaPipe sessions once (reused across all files)
    _mp_face_det = None
    _mp_face_msh = None
    _mp_pose_sess = None
    if mediapipe_working:
        try:
            _mp_face_det = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)
            _mp_face_msh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.2, min_tracking_confidence=0.2)
            _mp_pose_sess = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            _mp_face_det.__enter__()
            _mp_face_msh.__enter__()
            _mp_pose_sess.__enter__()
        except Exception as e:
            logger.warning(f"Failed to pre-create MediaPipe sessions: {e}")
            _mp_face_det = _mp_face_msh = _mp_pose_sess = None

    for input_file in found_files:
        input_filename = os.path.basename(input_file)
        
        # Extract Index
        index = 0
        try:
             parts = input_filename.split('_')
             if parts[0].isdigit(): index = int(parts[0])
             elif input_filename.startswith("output"): # output000
                 idx_str = input_filename[6:9]
                 if idx_str.isdigit(): index = int(idx_str)
        except (ValueError, IndexError): pass
        
        output_file = os.path.join(final_folder, f"temp_video_no_audio_{index}.mp4")

        # Determine Final Name (Title)
        base_name_final = input_filename.replace("_original_scale.mp4", "")
        # If legacy name, try to improve it
        if input_filename.startswith("output") and segments_data and index < len(segments_data):
             title = segments_data[index].get("title", f"Segment_{index}")
             safe_title = "".join([c for c in title if c.isalnum() or c in " _-"]).strip().replace(" ", "_")[:60]
             base_name_final = f"{index:03d}_{safe_title}"

        if os.path.exists(input_file):
            success = False
            detected_mode = "1" # Default if detection fails or fallback

            # 1. Try InsightFace
            if insightface_working:
                try:
                    # Capture returned mode
                    res = generate_short_insightface(input_file, output_file, index, project_folder, final_folder, face_mode=face_mode, detection_period=detection_period,
                                                     filter_threshold=filter_threshold, two_face_threshold=two_face_threshold, confidence_threshold=confidence_threshold, dead_zone=dead_zone, focus_active_speaker=focus_active_speaker,
                                                     active_speaker_mar=active_speaker_mar, active_speaker_score_diff=active_speaker_score_diff, include_motion=include_motion,
                                                     active_speaker_motion_deadzone=active_speaker_motion_deadzone,
                                                     active_speaker_motion_sensitivity=active_speaker_motion_sensitivity,
                                                     active_speaker_decay=active_speaker_decay,
                                                     no_face_mode=no_face_mode,
                                                     zoom_out_factor=zoom_out_factor,
                                                     ema_alpha=ema_alpha,                    # FIX: pass new params
                                                     detection_resolution=detection_resolution,
                                                     vertical_offset=vertical_offset,          # FIX: pass visual params
                                                     single_face_zoom=single_face_zoom)
                    if res: detected_mode = res
                    success = True
                except Exception as e:
                    logger.exception("Critical error during video processing")
                    logger.error(f"InsightFace processing failed for {input_filename}: {e}")
                    logger.warning("Falling back to MediaPipe/Haar...")
            
            # 2. Try MediaPipe if InsightFace failed or not available
            if not success and mediapipe_working and _mp_face_det is not None:
                try:
                    generate_short_mediapipe(input_file, output_file, index, face_mode, project_folder, final_folder, _mp_face_det, _mp_face_msh, _mp_pose_sess, detection_period=detection_period, no_face_mode=no_face_mode, zoom_out_factor=zoom_out_factor)
                    # We don't easily know detected mode here without return, assuming '1' or '2' based on last frame?
                    # Ideally function should return as well.
                    detected_mode = "1" # Placeholder, user didn't complain about stats.
                    # detected_mode = str(mp_num_faces) # Error fix: mp_num_faces not defined
                    if face_mode == "2":
                        detected_mode = "2"
                    success = True
                except Exception as e:
                     logger.error(f"MediaPipe processing failed: {e}")
            
            # 3. Try Haar if others failed
            if not success and (use_haar or (not mediapipe_working and not insightface_working)):
                 try:
                    logger.info("Attempts with Haar Cascade...")
                    generate_short_haar(input_file, output_file, index, project_folder, final_folder, detection_period=detection_period, no_face_mode=no_face_mode)
                    success = True
                 except Exception as e2:
                    logger.error(f"Haar fallback also failed: {e2}")

            # 4. Last Resort: Center Crop
            if not success:
                generate_short_fallback(input_file, output_file, index, project_folder, final_folder, no_face_mode=no_face_mode)
                detected_mode = "1"
                success = True
            
            # Save mode
            face_modes_log[f"output{str(index).zfill(3)}"] = detected_mode

        if success:
             try:
                 new_mp4_name = f"{base_name_final}.mp4"
                 new_mp4_path = os.path.join(final_folder, new_mp4_name)
                 
                 # Source is what finalize_video created
                 # finalize_video creates `final-output{index}_processed.mp4`
                 generated_mp4_name = f"final-output{str(index).zfill(3)}_processed.mp4"
                 generated_mp4_path = os.path.join(final_folder, generated_mp4_name)
                 
                 # 1. Rename MP4
                 if os.path.exists(generated_mp4_path):
                     if os.path.exists(new_mp4_path): os.remove(new_mp4_path)
                     os.rename(generated_mp4_path, new_mp4_path)
                     logger.info(f"Renamed Output to Title: {new_mp4_name}")
                     
                     # 2. Rename JSON Subtitle (if exists and hasn't been renamed by cut_segments)
                     subs_folder = os.path.join(project_folder, "subs")
                     
                     # Check if legacy name exists
                     old_json_name = f"final-output{str(index).zfill(3)}_processed.json"
                     old_json_path = os.path.join(subs_folder, old_json_name)
                     
                     new_json_name = f"{base_name_final}_processed.json"
                     new_json_path = os.path.join(subs_folder, new_json_name)
                     
                     if os.path.exists(old_json_path):
                         if os.path.exists(new_json_path): os.remove(new_json_path)
                         os.rename(old_json_path, new_json_path)
                         logger.info(f"Renamed Subtitles to Title: {new_json_name}")
                         
                     # 3. Rename Timeline JSON
                     # Timeline is temp_video_no_audio_{index}_timeline.json (created by generate_short...)
                     old_timeline_name = f"temp_video_no_audio_{index}_timeline.json"
                     old_timeline_path = os.path.join(final_folder, old_timeline_name)
                     
                     new_timeline_name = f"{base_name_final}_timeline.json"
                     new_timeline_path = os.path.join(final_folder, new_timeline_name)
                     
                     if os.path.exists(old_timeline_path):
                         if os.path.exists(new_timeline_path): os.remove(new_timeline_path)
                         os.rename(old_timeline_path, new_timeline_path)
                         logger.info(f"Renamed Timeline to Title: {new_timeline_name}")
                         
                     # 4. Rename Coords JSON
                     old_coords_name = f"temp_video_no_audio_{index}_coords.json"
                     old_coords_path = os.path.join(final_folder, old_coords_name)
                     
                     new_coords_name = f"{base_name_final}_coords.json"
                     new_coords_path = os.path.join(final_folder, new_coords_name)
                     
                     if os.path.exists(old_coords_path):
                         if os.path.exists(new_coords_path): os.remove(new_coords_path)
                         os.rename(old_coords_path, new_coords_path)
                         logger.info(f"Renamed Coords to Title: {new_coords_name}")
                         
             except Exception as e:
                 logger.warning(f"Warning: Could not rename file with title: {e}") 
        
    # Cleanup MediaPipe sessions
    for _sess in (_mp_face_det, _mp_face_msh, _mp_pose_sess):
        if _sess is not None:
            try:
                _sess.__exit__(None, None, None)
            except Exception:
                pass

    # Save Face Modes to JSON for subtitle usage
    modes_file = os.path.join(project_folder, "face_modes.json")
    try:
        with open(modes_file, "w") as f:
            json.dump(face_modes_log, f)
        logger.info(f"Detect Stats saved: {modes_file}")
    except Exception as e:
        logger.error(f"Error saving face modes: {e}")

if __name__ == "__main__":
    edit()