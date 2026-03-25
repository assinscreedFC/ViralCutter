from datetime import timedelta
import json
import subprocess

def timestamp_to_srt(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    micros = td.microseconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{micros//1000:03}"

def json_to_srt(json_data):
    """
    Converts internal JSON subtitle format to SRT.
    If 'words' key is present, generates word-level timestamps (Karaoke/Editing style).
    Otherwise, uses segment-level timestamps.
    """
    srt_content = ""
    counter = 1
    
    for block in json_data:
        # Check if words detail is available for Word-Level SRT
        if isinstance(block, dict) and "words" in block and block["words"]:
            for word_obj in block["words"]:
                start = word_obj.get('start', 0)
                end = word_obj.get('end', 0)
                text = word_obj.get('word', "")
                
                srt_content += f"{counter}\n"
                srt_content += f"{timestamp_to_srt(start)} --> {timestamp_to_srt(end)}\n"
                srt_content += f"{text}\n\n"
                counter += 1
        else:
            # Fallback to segment level
            start = 0
            end = 0
            text = ""
            if isinstance(block, dict):
                start = block.get('start', 0)
                end = block.get('end', 0)
                text = block.get('text', "")
            elif isinstance(block, (list, tuple)) and len(block) >= 3:
                start, end, text = block[0], block[1], block[2]
                
            srt_content += f"{counter}\n"
            srt_content += f"{timestamp_to_srt(start)} --> {timestamp_to_srt(end)}\n"
            srt_content += f"{text}\n\n"
            counter += 1
            
    return srt_content

def get_video_dims(vid_path):
    """Returns (width, height, duration_frames, fps)"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json",
            vid_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        fps_str = stream["r_frame_rate"]  # e.g. "30000/1001"
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den > 0 else 30.0
        duration = float(data["format"]["duration"])

        return width, height, int(duration * fps), fps
    except Exception as e:
        print(f"Error probing video: {e}")
        return 1920, 1080, 300, 30.0
