from __future__ import annotations

import logging
import os
import shutil
from scripts.run_cmd import run as run_cmd
from scripts.ffmpeg_utils import get_best_encoder, build_quality_params, _build_preset_flags
import sys
import tempfile

logger = logging.getLogger(__name__)

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def burn_video_file(video_path: str, subtitle_path: str, output_path: str) -> tuple[bool, str]:
    """
    Burns subtitles into a single video file.
    """
    # Copy .ass to a temp path with no special chars (apostrophe, comma, spaces
    # all break FFmpeg's filtergraph parser). Keep on D: drive, not C:.
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = tempfile.mkdtemp(dir=_project_root)
    tmp_ass = os.path.join(tmp_dir, "sub.ass")
    shutil.copy2(subtitle_path, tmp_ass)

    subtitle_file_ffmpeg = tmp_ass.replace('\\', '/').replace(':', '\\:').replace("'", "'\\\\''")

    encoder_name, encoder_preset = get_best_encoder()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-hide_banner",
        '-i', video_path,
        '-vf', f"subtitles='{subtitle_file_ffmpeg}'",
        '-c:v', encoder_name,
        *_build_preset_flags(encoder_name, encoder_preset),
        *build_quality_params(encoder_name),
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        output_path,
    ]
    try:
        run_cmd(cmd)
        return True, f"{encoder_name} Success"
    except Exception as e:
        err_msg = f"Error burning subtitles on {os.path.basename(video_path)}: {e}"
        logger.error(err_msg)
        return False, err_msg
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def burn(project_folder: str = "tmp", source_folder: str | None = None, name_suffix_strip: str = "") -> None:
    # Converter para absoluto para não ter erro no filtro do ffmpeg
    if project_folder and not os.path.isabs(project_folder):
        project_folder_abs = os.path.abspath(project_folder)
    else:
        project_folder_abs = project_folder

    # Caminhos das pastas
    subs_folder = os.path.join(project_folder_abs, 'subs_ass')
    videos_folder = source_folder if source_folder else os.path.join(project_folder_abs, 'final')
    output_folder = os.path.join(project_folder_abs, 'burned_sub')  # Pasta para salvar os vídeos com legendas

    # Cria a pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(videos_folder):
        logger.error(f"Pasta de vídeos finais não encontrada: {videos_folder}")
        return

    # Itera sobre os arquivos de vídeo na pasta final
    files = os.listdir(videos_folder)
    if not files:
        logger.warning("Nenhum arquivo encontrado em 'final' para queimar legendas.")
        return

    for video_file in files:
        if video_file.endswith(('.mp4', '.mkv', '.avi')):  # Formatos suportados
            # Se for temp file (ex: temp_video_no_audio), ignora se existir a versão final
            if "temp_video_no_audio" in video_file:
                continue

            # Extrai o nome base do vídeo (sem extensão)
            video_name = os.path.splitext(video_file)[0]

            # Strip suffix when looking up .ass (e.g. "_split" for split_screen videos)
            video_name_for_ass = video_name
            if name_suffix_strip and video_name.endswith(name_suffix_strip):
                video_name_for_ass = video_name[:-len(name_suffix_strip)]

            # Define o caminho para a legenda correspondente
            subtitle_file = os.path.join(subs_folder, f"{video_name_for_ass}.ass")

            # Tentar também com sufixo _processed caso a convenção seja diferente
            if not os.path.exists(subtitle_file):
                subtitle_file_processed = os.path.join(subs_folder, f"{video_name_for_ass}_processed.ass")
                if os.path.exists(subtitle_file_processed):
                    subtitle_file = subtitle_file_processed
            
            # Verifica se a legenda existe
            if os.path.exists(subtitle_file):
                # Define o caminho de saída para o vídeo com legendas
                output_file = os.path.join(output_folder, f"{video_name}_subtitled.mp4")

                logger.info(f"Burning: {video_name}...")
                success, msg = burn_video_file(os.path.join(videos_folder, video_file), subtitle_file, output_file)
                if success:
                    logger.info(f"Done: {output_file}")
                else:
                    logger.error(f"Fail: {msg}")
            else:
                logger.warning(f"Legenda não encontrada para: {video_name} em {subtitle_file}")
