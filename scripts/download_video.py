from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from urllib.parse import urlparse

import yt_dlp
import sys
from i18n.i18n import I18nAuto
i18n = I18nAuto()

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = {"http", "https"}

BLOCKED_DOMAINS = {"localhost", "localhost.localdomain"}


def _is_private_hostname(hostname: str) -> bool:
    """Return True if *hostname* resolves to a private/reserved IP address."""
    try:
        addr = ipaddress.ip_address(hostname)
        return addr.is_private or addr.is_reserved or addr.is_loopback
    except ValueError:
        pass
    try:
        resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canonname, sockaddr in resolved:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_reserved or ip.is_loopback:
                return True
    except (socket.gaierror, OSError):
        pass
    return False


def validate_url(url: str) -> None:
    """Validate *url* before passing it to yt-dlp.

    Raises ``ValueError`` if the URL is unsafe (wrong scheme, private IP,
    file path, etc.).
    """
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. Only http/https URLs are accepted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname.")

    if hostname in BLOCKED_DOMAINS:
        raise ValueError(f"URL hostname '{hostname}' is not allowed.")

    if _is_private_hostname(hostname):
        raise ValueError(
            f"URL resolves to a private/reserved IP address and is not allowed."
        )

def sanitize_filename(name: str) -> str:
    """Remove caracteres inválidos e emojis para evitar erro de encoding no Windows."""
    # Remove caracteres reservados do sistema de arquivos
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name)
    
    # Remove emojis e caracteres não suportados pelo console Windows (CP1252)
    # Isso mantém acentos (á, ç, é) mas remove 😱, etc.
    try:
        cleaned = cleaned.encode('cp1252', 'ignore').decode('cp1252')
    except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
        # Fallback se não tiver CP1252: remove tudo não-ascii (remove acentos)
        cleaned = cleaned.encode('ascii', 'ignore').decode('ascii')
        
    cleaned = cleaned.strip()
    return cleaned

def progress_hook(d: dict) -> None:
    if d['status'] == 'downloading':
        try:
            p = d.get('_percent_str', '').replace('%','')
            logger.info(f"[download] {p}% - {d.get('_eta_str', 'N/A')} remaining")
        except (ValueError, KeyError, TypeError):
            pass  # progress info may be missing or malformed
    elif d['status'] == 'finished':
        logger.info(f"[download] Download concluído: {d['filename']}")

def download(url: str, base_root: str = "VIRALS", download_subs: bool = True, quality: str = "best") -> tuple[str, str]:
    validate_url(url)

    # 1. Extrair informações do vídeo para pegar o título
    logger.info(i18n("Extracting video information..."))
    title = None
    
    # ... (Keep existing title extraction logic) ...
    # Instead of repeating it effectively, I will rely on the diff to keep it or re-write it if I have to replace the whole block.
    # Since replace_file_content works on line ranges, I should be careful.
    # Let's assume I'm replacing the whole function body or significant parts.
    
    # Tentativa 1: Com cookies
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True, 'cookiesfrombrowser': ('chrome',)}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title')
    except Exception as e:
        try:
            logger.warning(i18n("Warning: Failed to extract info with cookies: {}").format(e))
        except UnicodeEncodeError:
            logger.warning(i18n("Warning: Failed to extract info with cookies: [Encoding Error in Message]"))

    # Tentativa 2: Sem cookies
    if not title:
        try:
             with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title')
        except Exception as e:
            try:
                logger.error(i18n("Error getting video info (without cookies): {}").format(e))
            except UnicodeEncodeError:
                logger.error(i18n("Error getting video info (without cookies): [Encoding Error in Message]"))

    # Fallback final
    if title:
        safe_title = sanitize_filename(title)
        try:
            logger.info(i18n("Detected title: {}").format(title))
        except UnicodeEncodeError:
            # Fallback for Windows consoles that choke on Emojis
            clean_title = title.encode('ascii', 'replace').decode('ascii')
            logger.info(i18n("Detected title: {}").format(clean_title))
    else:
        logger.warning(i18n("WARNING: Title could not be obtained. Using 'Unknown_Video'."))
        safe_title = i18n("Unknown_Video")

    # 2. Criar estrutura de pastas
    project_folder = os.path.join(base_root, safe_title)
    os.makedirs(project_folder, exist_ok=True)
    
    # Caminho final do vídeo
    output_filename = 'input' 
    output_path_base = os.path.join(project_folder, output_filename)
    final_video_path = f"{output_path_base}.mp4"

    # Verificação inteligente
    if os.path.exists(final_video_path):
        if os.path.getsize(final_video_path) > 1024: 
            try:
                logger.info(i18n("Video already exists at: {}").format(final_video_path))
            except UnicodeEncodeError:
                logger.info(i18n("Video already exists at: {}").format(final_video_path.encode('ascii', 'replace').decode('ascii')))
            logger.info(i18n("Skipping download and reusing local file."))
            return final_video_path, project_folder
        else:
            logger.info(i18n("Existing file found but seems corrupted/empty. Downloading again..."))
            try:
                os.remove(final_video_path)
            except OSError:
                pass

    # Limpeza de temp
    temp_path = f"{output_path_base}.temp.mp4"
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    # Mapeamento de Qualidade
    quality_map = {
        "best": 'bestvideo+bestaudio/best',
        "1080p": 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        "720p": 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        "480p": 'bestvideo[height<=480]+bestaudio/best[height<=480]'
    }
    selected_format = quality_map.get(quality, 'bestvideo+bestaudio/best')
    logger.info(i18n("Configuring download quality: {} -> {}").format(quality, selected_format))

    ydl_opts = {
        'format': selected_format,
        'overwrites': True,
        'outtmpl': output_path_base, 
        'postprocessor_args': [
            '-movflags', 'faststart'
        ],
        'merge_output_format': 'mp4',
        'progress_hooks': [progress_hook],
        # Opções de Legenda
        'writesubtitles': download_subs,
        'writeautomaticsub': download_subs,
        'subtitleslangs': ['pt.*', 'en.*', 'sp.*'], # Prioritize generic PT, EN, SP
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        },
        'skip_download': False,
        'quiet': False,
        'no_warnings': False,
        'force_ipv4': True,
    }
    

    
    if download_subs:
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegSubtitlesConvertor',
            'format': 'srt',
        }]

    try:
        logger.info(i18n("Downloading video to: {}...").format(project_folder))
    except UnicodeEncodeError:
        logger.info(i18n("Downloading video to: {}...").format(project_folder.encode('ascii', 'replace').decode('ascii')))
    
    # Tentativa 1: Com configuração original
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        error_str = str(e)
        if "No address associated with hostname" in error_str or "Failed to resolve" in error_str:
            logger.error(i18n("\n[CRITICAL ERROR] Connection Failure: Could not access YouTube."))
            logger.info(i18n("Check your internet connection or if there is any DNS block."))
            logger.info(i18n("Details: {}").format(e))
            sys.exit(1)
        
        elif download_subs and ("Unable to download video subtitles" in error_str or "429" in error_str):
            logger.warning(i18n("Error downloading subtitles ({}).").format(e))
            logger.info(i18n("Retrying ONLY the video (without subtitles)..."))
            
            ydl_opts['writesubtitles'] = False
            ydl_opts['writeautomaticsub'] = False
            ydl_opts['postprocessors'] = [p for p in ydl_opts.get('postprocessors', []) if 'Subtitle' not in p.get('key', '')]
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as e2:
                logger.error(i18n("Fatal error on second attempt: {}").format(e2))
                raise
        elif "is not a valid URL" in error_str:
             logger.error(i18n("Error: the entered link is not valid."))
             raise 
        else:
            logger.error(i18n("Download error: {}").format(e))
            raise
    except Exception as e:
        logger.error(i18n("Unexpected error: {}").format(e))
        raise

    # RENOMEAR LEGENDA PARA PADRÃO (input.vtt ou input.srt)
    # Se for VTT, converte para SRT para garantir compatibilidade.
    try:
        import glob
        # Pega a primeira que encontrar
        potential_subs = glob.glob(os.path.join(project_folder, "input.*.vtt")) + glob.glob(os.path.join(project_folder, "input.*.srt"))
        
        if potential_subs:
            best_sub = potential_subs[0]
            ext = os.path.splitext(best_sub)[1]
            new_name = os.path.join(project_folder, "input.srt") # Vamos padronizar tudo para .srt
            
            if ext.lower() == '.vtt':
                try:
                    logger.info(i18n("Formatting complex VTT subtitle ({}) to clean SRT...").format(os.path.basename(best_sub)))
                except UnicodeEncodeError:
                    logger.info(i18n("Formatting complex VTT subtitle ({}) to clean SRT...").format(os.path.basename(best_sub).encode('ascii', 'replace').decode('ascii')))
                try:
                    with open(best_sub, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    srt_content = []
                    counter = 1
                    
                    seen_texts = set()
                    last_text = ""
                    
                    for line in lines:
                        clean_line = line.strip()
                        # Ignora Headers e Metadados do VTT/Youtube
                        if clean_line.startswith("WEBVTT") or \
                           clean_line.startswith("X-TIMESTAMP") or \
                           clean_line.startswith("NOTE") or \
                           clean_line.startswith("Kind:") or \
                           clean_line.startswith("Language:"):
                            continue
                        
                        if "-->" in clean_line:
                            # Parse Timestamp
                            parts = clean_line.split("-->")
                            start = parts[0].strip()
                            # Remove tags de posicionamento "align:start position:0%"
                            end = parts[1].strip().split(' ')[0] 
                            
                            def fix_time(t):
                                t = t.replace('.', ',')
                                if t.count(':') == 1: 
                                    t = "00:" + t
                                return t
                            
                            current_start = fix_time(start)
                            current_end = fix_time(end)
                            
                        elif clean_line:
                             # Texto: remover tags complexas <00:00:00.560><c> etc
                             # O YouTube usa formato karaoke. Ex: "Quanto<...> custa<...>"
                             # Precisamos do texto limpo.
                             text = re.sub(r'<[^>]+>', '', clean_line).strip()
                             
                             if not text: continue
                             
                             # Lógica para remover duplicatas do estilo "Roll-up" ou "Karaoke"
                             # O YouTube repete a linha anterior às vezes.
                             # Ex:
                             # 1: "Quanto custa"
                             # 2: "Quanto custa\nQuantos quilos"
                             
                             # Vamos pegar apenas a ULTIMA linha se tiver quebras
                             lines_in_text = text.split('\n')
                             final_line = lines_in_text[-1].strip()
                             
                             if not final_line: continue

                             # Filtro de duplicidade consecutivo
                             if final_line == last_text:
                                 continue
                             
                             # Evita blocos ultra curtos (glitch de 10ms) que repetem texto
                             # Mas aqui estamos processando texto.
                             
                             srt_content.append(f"{counter}\n")
                             srt_content.append(f"{current_start} --> {current_end}\n")
                             srt_content.append(f"{final_line}\n\n")
                             
                             last_text = final_line
                             counter += 1
                    
                    with open(new_name, 'w', encoding='utf-8') as f_out:
                        f_out.writelines(srt_content)
                    
                    try:
                        logger.info(i18n("Subtitle converted and cleaned: {}").format(new_name))
                    except UnicodeEncodeError:
                        logger.info(i18n("Subtitle converted and cleaned: {}").format(new_name.encode('ascii', 'replace').decode('ascii')))
                    try: os.remove(best_sub)
                    except OSError: pass
                    
                except Exception as e_conv:
                    logger.error(i18n("Failed to convert VTT: {}. Keeping original.").format(e_conv))
                    # Fallback: rename apenas
                    new_name_fallback = os.path.join(project_folder, "input.vtt")
                    if os.path.exists(new_name_fallback) and new_name_fallback != best_sub:
                        try: os.remove(new_name_fallback)
                        except OSError: pass
                    os.rename(best_sub, new_name_fallback)

            else:
                # Já é SRT, só renomeia
                if os.path.exists(new_name) and new_name != best_sub:
                    try: os.remove(new_name)
                    except OSError: pass
                os.rename(best_sub, new_name)
                try:
                    logger.info(i18n("SRT subtitle renamed to: {}").format(new_name))
                except UnicodeEncodeError:
                    logger.info(i18n("SRT subtitle renamed to: {}").format(new_name.encode('ascii', 'replace').decode('ascii')))
            
            # Limpa sobras
            for extra in potential_subs[1:]:
                try: os.remove(extra)
                except OSError: pass

    except Exception as e_ren:
        logger.error(i18n("Error processing subtitles: {}").format(e_ren))

    return final_video_path, project_folder