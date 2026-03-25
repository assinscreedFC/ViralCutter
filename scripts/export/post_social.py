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
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv

from scripts.core.run_cmd import run as run_cmd

load_dotenv()

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CREDENTIALS_DIR = os.path.join(_PROJECT_ROOT, "credentials")


# ── Pre-upload validation ─────────────────────────────────────────────────────

def validate_clip_for_upload(clip_path: str) -> list[str]:
    """Vérifie qu'un clip est valide pour upload (durée, dimensions, audio).

    Returns:
        Liste de warnings (vide = OK). Ne bloque pas l'upload.
    """
    warnings: list[str] = []
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
               "-show_format", "-show_streams", clip_path]
        result = run_cmd(cmd, text=True, check=False, timeout=30)
        data = json.loads(result.stdout)

        duration = float(data.get("format", {}).get("duration", 0))
        if duration < 5:
            warnings.append(f"Durée trop courte : {duration:.1f}s (min recommandé : 15s)")
        elif duration > 600:
            warnings.append(f"Durée trop longue : {duration:.1f}s (max TikTok : 600s)")

        streams = data.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        if not audio_streams:
            warnings.append("Aucune piste audio détectée.")

        if video_streams:
            w = int(video_streams[0].get("width", 0))
            h = int(video_streams[0].get("height", 0))
            if w > 1080 or h > 1920:
                warnings.append(f"Dimensions {w}×{h} dépassent 1080×1920.")
    except Exception as e:
        warnings.append(f"Impossible de valider le clip : {e}")

    return warnings


# ── Retry helper ──────────────────────────────────────────────────────────────

def _upload_with_retry(upload_fn, max_retries: int = 3, base_delay: float = 30.0) -> dict:
    """Exécute upload_fn avec retry exponentiel sur exception.

    Retourne le résultat de upload_fn (dict) dès que status != 'failed',
    ou le dernier résultat d'erreur après épuisement des tentatives.
    """
    last_result: dict = {}
    for attempt in range(max_retries):
        last_result = upload_fn()
        if last_result.get("status") != "failed":
            return last_result
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.warning("[POST] Retry %d/%d dans %.0fs — %s",
                           attempt + 1, max_retries, delay,
                           last_result.get("error", "erreur inconnue"))
            time.sleep(delay)
    return last_result


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

    If first_post_time is provided: seg 0 → first_post_time, seg 1 → + interval, etc.
    If first_post_time is empty and interval > 0: seg 0 → None (immediate),
        seg 1 → now + interval, seg 2 → now + 2*interval, etc.
    If first_post_time is empty and interval == 0: all None (all immediate).
    """
    if not first_post_time or not first_post_time.strip():
        if interval_minutes <= 0 or count <= 1:
            return [None] * count
        # First post immediate, rest scheduled from now + interval (UTC)
        now_utc = datetime.now(tz=timezone.utc)
        return [None] + [
            now_utc + timedelta(minutes=interval_minutes * i)
            for i in range(1, count)
        ]

    base = _parse_datetime(first_post_time)
    if base is None:
        return [None] * count

    # Convert to UTC — tiktok-uploader requires naive or UTC-aware datetimes
    return [base.astimezone(timezone.utc) + timedelta(minutes=interval_minutes * i) for i in range(count)]


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

def _wait_for_upload_complete(page, timeout: int = 300) -> None:
    """Attend que TikTok confirme la fin du traitement serveur.

    Apres le clic "Post", TikTok peut afficher un spinner puis rediriger
    ou afficher un message de succes. On poll jusqu'a ce que la page quitte
    l'ecran d'upload OU timeout (defaut 5min).
    """
    import time

    upload_url = "tiktok.com/upload"
    start = time.time()
    poll_interval = 3
    _logger = logging.getLogger(__name__)

    while time.time() - start < timeout:
        current_url = page.url
        if upload_url not in current_url:
            _logger.info("[TikTok] Traitement termine (redirection detectee)")
            return

        try:
            success = page.locator("text=Your video has been uploaded")
            if success.is_visible(timeout=1000):
                _logger.info("[TikTok] Traitement termine (message de succes)")
                return
        except Exception:
            pass

        time.sleep(poll_interval)

    _logger.warning("[TikTok] Timeout apres %ds — fermeture du browser", timeout)


def _tiktok_upload_worker(
    video_path: str,
    caption: str,
    cookies_path: str,
    schedule: Optional[datetime],
) -> None:
    """Top-level worker executed in a subprocess (no asyncio loop).

    Loads cookies from a Cookie-Editor JSON file and uses TikTokUploader
    directly to keep control of the browser after upload — waits for
    TikTok server-side processing before closing.
    """
    import json as _json
    from tiktok_uploader.upload import TikTokUploader
    from tiktok_uploader import config as tiktok_config

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

    video_dict: dict = {"path": video_path, "description": caption}
    if schedule is not None:
        video_dict["schedule"] = schedule

    # Augmenter le timeout d'upload pour les videos longues (defaut lib: 180s)
    original_uploading_wait = tiktok_config.uploading_wait
    tiktok_config.uploading_wait = 600  # 10 min pour uploader le fichier

    # Empecher la lib de fermer le browser — on gere l'attente nous-memes
    original_quit = tiktok_config.quit_on_end
    tiktok_config.quit_on_end = False

    uploader = TikTokUploader(cookies_list=cookies_list, headless=False)

    try:
        try:
            uploader.upload_videos([video_dict])
        except Exception as upload_err:
            # La lib tiktok-uploader peut crash sur des selecteurs obsoletes
            # (ex: .TUXButton--primary) meme si TikTok a accepte la video.
            # On laisse _wait_for_upload_complete verifier le resultat reel.
            logging.getLogger(__name__).warning(
                "[TikTok] Erreur lib upload (selecteurs?): %s — verification en cours...", upload_err
            )
        _wait_for_upload_complete(uploader.page, timeout=300)
    finally:
        tiktok_config.uploading_wait = original_uploading_wait
        tiktok_config.quit_on_end = original_quit
        uploader.close()


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
            ).result(timeout=900)  # 15 min max (upload + traitement serveur)

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

        # Validation pré-upload (warnings non bloquants)
        clip_warnings = validate_clip_for_upload(video_path)
        for w in clip_warnings:
            logger.warning("[VALIDATE] Segment %d — %s", i + 1, w)

        if post_youtube:
            res = _upload_with_retry(lambda: upload_to_youtube(
                video_path=video_path,
                title=title,
                description=description,
                privacy=youtube_privacy,
                publish_at=publish_at,
            ))
            results.append(res)

        if post_tiktok:
            res = _upload_with_retry(lambda: upload_to_tiktok(
                video_path=video_path,
                caption=tiktok_caption,
                cookies_path=tiktok_cookies,
                schedule=publish_at,
            ))
            results.append(res)

    results_path = os.path.join(project_folder, "post_results.json")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error("Failed to save post_results.json: %s", e)

    return results
