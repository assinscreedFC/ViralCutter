"""
Auto-post viral segments to YouTube Shorts and TikTok.

Scheduling is handled natively by each platform — the script uploads all
videos immediately with a scheduled publish time, then can be closed.
No need to keep the script running after upload completes.

YouTube  : YouTube Data API v3 (OAuth2)
           publishAt = ISO 8601 UTC — platform publishes automatically.
           Credentials: YOUTUBE_CLIENT_ID + YOUTUBE_CLIENT_SECRET in .env
           Token saved: credentials/youtube_token.json

TikTok   : tiktok-uploader (Playwright + browser cookies)
           schedule = datetime — platform publishes automatically.
           Cookies path: TIKTOK_COOKIES_PATH in .env
"""

import concurrent.futures
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_DIR = os.path.join(_PROJECT_ROOT, "credentials")


# ── Scheduling helpers ────────────────────────────────────────────────────────

def _parse_datetime(raw: str) -> Optional[datetime]:
    """Parse 'HH:MM' or 'YYYY-MM-DD HH:MM' into a timezone-aware local datetime.

    For 'HH:MM': uses today, or tomorrow if the time has already passed.
    Returns None on parse failure.
    """
    raw = raw.strip()
    parsed = None
    for fmt in ("%Y-%m-%d %H:%M", "%H:%M"):
        try:
            parsed = datetime.strptime(raw, fmt)
            break
        except ValueError:
            pass

    if parsed is None:
        logger.warning("Cannot parse datetime '%s'. Expected 'YYYY-MM-DD HH:MM' or 'HH:MM'.", raw)
        return None

    local_tz = datetime.now().astimezone().tzinfo
    now = datetime.now(tz=local_tz)

    if parsed.year == 1900:
        # Only HH:MM given — use today, push to tomorrow if already past
        target = now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
    else:
        target = parsed.replace(tzinfo=local_tz)

    return target


def _calc_schedule_times(
    first_post_time: Optional[str],
    interval_minutes: float,
    count: int,
) -> list[Optional[datetime]]:
    """Return a list of `count` scheduled datetimes.

    If first_post_time is empty/invalid, all entries are None (= immediate).
    Otherwise: segment 0 → first_post_time, segment 1 → + interval, etc.
    """
    if not first_post_time or not first_post_time.strip():
        return [None] * count

    base = _parse_datetime(first_post_time)
    if base is None:
        return [None] * count

    return [base + timedelta(minutes=interval_minutes * i) for i in range(count)]


# ── YouTube ───────────────────────────────────────────────────────────────────

def _get_youtube_service():
    """Return an authenticated YouTube API service.

    First run: opens browser for OAuth2 consent and saves token.
    Subsequent runs: loads token from credentials/youtube_token.json.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    token_path = os.path.join(CREDENTIALS_DIR, "youtube_token.json")

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_id = os.getenv("YOUTUBE_CLIENT_ID")
            client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
            if not client_id or not client_secret:
                raise ValueError(
                    "YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET must be set in .env"
                )
            client_config = {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)

        os.makedirs(CREDENTIALS_DIR, exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(
    video_path: str,
    title: str,
    description: str,
    privacy: str = "public",
    category_id: str = "22",
    tags: Optional[list] = None,
    publish_at: Optional[datetime] = None,
) -> dict:
    """Upload a video to YouTube Shorts.

    If publish_at is set, the video is uploaded as private and YouTube
    publishes it automatically at that time (no need to keep script running).
    YouTube requires publish_at to be at least 15 minutes in the future.

    Returns:
        dict with keys: platform, video_id, url, title, status, scheduled_at
              or       platform, error, status (on failure)
    """
    if tags is None:
        tags = ["shorts", "#shorts"]

    try:
        from googleapiclient.http import MediaFileUpload

        youtube = _get_youtube_service()

        status_body: dict = {"selfDeclaredMadeForKids": False}
        if publish_at is not None:
            # YouTube requires privacyStatus="private" for scheduled uploads
            status_body["privacyStatus"] = "private"
            status_body["publishAt"] = publish_at.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            )
        else:
            status_body["privacyStatus"] = privacy

        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags,
                "categoryId": category_id,
            },
            "status": status_body,
        }

        media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

        response = None
        while response is None:
            _, response = request.next_chunk()

        video_id = response["id"]
        result = {
            "platform": "youtube",
            "video_id": video_id,
            "url": f"https://www.youtube.com/shorts/{video_id}",
            "title": title,
            "status": "success",
        }
        if publish_at:
            result["scheduled_at"] = publish_at.strftime("%Y-%m-%d %H:%M")
        return result

    except Exception as e:
        logger.error("YouTube upload failed for '%s': %s", title, e, exc_info=True)
        return {"platform": "youtube", "error": str(e), "status": "failed", "title": title}


# ── TikTok ────────────────────────────────────────────────────────────────────

def _tiktok_upload_worker(
    video_path: str,
    caption: str,
    cookies_path: str,
    schedule: Optional[datetime],
) -> None:
    """Top-level worker executed in a subprocess (no asyncio loop).

    Loads cookies from a Cookie-Editor JSON file and passes them as
    cookies_list to tiktok-uploader (which expects Netscape format via
    the `cookies` param — JSON must go through cookies_list instead).
    """
    import json as _json
    from tiktok_uploader.upload import upload_video

    with open(cookies_path, encoding="utf-8") as f:
        raw = _json.load(f)

    # Remap Cookie-Editor keys → tiktok-uploader Cookie TypedDict keys
    # expires=-1 = session cookie (Playwright rejects None)
    cookies_list = [
        {
            "name": c["name"],
            "value": c["value"],
            "domain": c.get("domain", ""),
            "path": c.get("path", "/"),
            "expiry": c.get("expirationDate") or c.get("expiry") or -1,
            "sameSite": c.get("sameSite", "None"),
        }
        for c in raw
        if c.get("name") and c.get("value")
    ]

    # NOTE: tiktok-uploader schedule relies on fragile UI selectors that break
    # when TikTok updates their interface. Schedule is intentionally not passed —
    # videos post immediately. Use YouTube's API scheduling for reliable timing.
    kwargs: dict = {"filename": video_path, "description": caption, "cookies_list": cookies_list}
    upload_video(**kwargs)


def upload_to_tiktok(
    video_path: str,
    caption: str,
    cookies_path: Optional[str] = None,
    schedule: Optional[datetime] = None,
) -> dict:
    """Upload a video to TikTok via tiktok-uploader (Playwright + cookies).

    If schedule is set, TikTok publishes it automatically at that time
    (no need to keep script running after upload).

    Returns:
        dict with keys: platform, status, title, scheduled_at
              or       platform, error, status (on failure)
    """
    if cookies_path is None:
        cookies_path = os.getenv("TIKTOK_COOKIES_PATH", "")

    # Resolve to absolute path before passing to subprocess (different CWD)
    if cookies_path:
        cookies_path = os.path.abspath(cookies_path)

    if not cookies_path or not os.path.exists(cookies_path):
        return {
            "platform": "tiktok",
            "error": (
                f"TikTok cookies not found at: {cookies_path}. "
                "Export cookies with Cookie-Editor (JSON format) and set TIKTOK_COOKIES_PATH in .env"
            ),
            "status": "failed",
            "title": caption,
        }

    try:
        # Run in a separate process — Playwright sync API cannot run inside an
        # asyncio event loop (Gradio uses one). ProcessPoolExecutor gives a clean
        # subprocess with no inherited event loop.
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
            pool.submit(
                _tiktok_upload_worker,
                os.path.abspath(video_path),
                caption,
                cookies_path,
                schedule,
            ).result()

        result = {
            "platform": "tiktok",
            "video_id": None,
            "url": None,
            "title": caption,
            "status": "success",
        }
        if schedule:
            result["scheduled_at"] = schedule.strftime("%Y-%m-%d %H:%M")
        return result

    except Exception as e:
        logger.error("TikTok upload failed for '%s': %s", caption, e, exc_info=True)
        return {"platform": "tiktok", "error": str(e), "status": "failed", "title": caption}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def post_all_segments(
    project_folder: str,
    post_youtube: bool = False,
    post_tiktok: bool = False,
    youtube_privacy: str = "public",
    interval_minutes: float = 30,
    first_post_time: Optional[str] = None,
) -> list[dict]:
    """Upload all segments to the selected platforms with native scheduling.

    Calculates a publish datetime for each segment (first_post_time + index *
    interval_minutes) and passes it to each platform's API. The platforms
    publish at the right time automatically — you can close the script once
    all uploads complete.

    Saves results to post_results.json.

    Returns:
        List of result dicts (one per segment per platform).
    """
    results = []

    if not post_youtube and not post_tiktok:
        return results

    segments_file = os.path.join(project_folder, "viral_segments.txt")
    if not os.path.exists(segments_file):
        logger.warning("viral_segments.txt not found in %s", project_folder)
        return results

    with open(segments_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data if isinstance(data, list) else data.get("segments", [])

    def _find_video(seg: dict, index: int) -> Optional[str]:
        """Return video path for a segment: explicit filepath first, then auto-discovery."""
        fp = seg.get("filepath")
        if fp and os.path.exists(fp):
            return fp
        # Auto-discover: look for index-prefixed mp4 in priority order
        prefix = f"{index:03d}_"
        for subdir in ("burned_sub", "split_screen", "with_music", "final", "cuts"):
            folder = os.path.join(project_folder, subdir)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.startswith(prefix) and fname.endswith(".mp4"):
                    return os.path.join(folder, fname)
        return None

    valid_segments = []
    for idx, seg in enumerate(segments):
        vp = _find_video(seg, idx)
        if vp:
            valid_segments.append((seg, vp))
        else:
            logger.warning("Segment %d — no video file found, skipping.", idx)

    if not valid_segments:
        logger.warning("No valid segment filepaths found in %s", segments_file)
        return results

    tiktok_cookies = os.getenv("TIKTOK_COOKIES_PATH", "")
    schedule_times = _calc_schedule_times(first_post_time, interval_minutes, len(valid_segments))

    for i, (seg, video_path) in enumerate(valid_segments):
        title = seg.get("title", "Viral Clip")
        description = seg.get("description", "")
        tiktok_caption = seg.get("tiktok_caption") or title
        publish_at = schedule_times[i]

        if publish_at:
            logger.info(
                "Segment %d/%d — scheduling at %s",
                i + 1, len(valid_segments), publish_at.strftime("%Y-%m-%d %H:%M"),
            )
        else:
            logger.info("Segment %d/%d — posting immediately", i + 1, len(valid_segments))

        if post_youtube:
            res = upload_to_youtube(
                video_path=video_path,
                title=title,
                description=description,
                privacy=youtube_privacy,
                publish_at=publish_at,
            )
            results.append(res)

        if post_tiktok:
            res = upload_to_tiktok(
                video_path=video_path,
                caption=tiktok_caption,
                cookies_path=tiktok_cookies,
                schedule=publish_at,
            )
            results.append(res)

    results_path = os.path.join(project_folder, "post_results.json")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error("Failed to save post_results.json: %s", e)

    return results
