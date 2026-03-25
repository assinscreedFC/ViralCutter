import os

from scripts.core.run_cmd import run as run_cmd
from .utils import get_video_dims

def render_segmented_overlays(ass_path, segments, video_path, output_dir):
    """
    Renders segments using a physical transparent PNG canvas to ensure alpha correctness.
    """
    width, height, _, fps = get_video_dims(video_path)
    ass_path_sanitized = ass_path.replace("\\", "/").replace(":", "\\:")
    
    # Generate Base Canvas
    canvas_png = os.path.join(output_dir, "base_canvas.png")
    run_cmd([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black@0.0:s={width}x{height}",
        "-frames:v", "1", "-c:v", "png", canvas_png
    ])
    
    overlay_data = []
    print(f"Rendering {len(segments)} subtitle segments (Mode: Canvas + QTRLE)...")
    
    for i, seg in enumerate(segments):
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        duration = end - start
        if duration <= 0: continue
        
        filename = f"caption_{i}.mov"
        out_path = os.path.join(output_dir, filename)
        
        # QTRLE (QuickTime Animation) - The absolute reference for Alpha.
        # Slightly larger files than PNG, but 100% compatible.
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", canvas_png,
            "-vf", f"format=rgba,setpts=PTS+{start}/TB,ass='{ass_path_sanitized}',setpts=PTS-{start}/TB,format=rgba",
            "-t", str(duration),
            "-c:v", "qtrle",
            "-pix_fmt", "argb", # qtrle uses argb pixel format usually
            "-an",
            out_path
        ]
        
        try:
            run_cmd(cmd)
            rel_path = os.path.join("captions", filename).replace("\\", "/")
            overlay_data.append({ "path": rel_path, "start": start, "end": end, "index": i })
            print(f"  [Seg {i}] Rendered {duration:.2f}s")
        except Exception as e:
            print(f"  [Seg {i}] Failed: {e}")
            
    if os.path.exists(canvas_png): os.remove(canvas_png)

    return overlay_data
