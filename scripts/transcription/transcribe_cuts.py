import logging
import os
import sys

from scripts.core.run_cmd import run as run_cmd

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def transcribe(project_folder="tmp"):
    def generate_whisperx(input_file, output_folder, model='large-v3'):
        output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}.srt")
        json_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}.json")  # Define the JSON output file

        # Skip processing if the JSON file already exists
        if os.path.exists(json_file):
            logger.info(f"Arquivo já existe, pulando: {json_file}")
            return

        command = [
            "whisperx",
            input_file,
            "--model", model,
            "--task", "transcribe",
            "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
            "--chunk_size", "10",
            "--vad_onset", "0.4",
            "--vad_offset", "0.3",
            "--compute_type", "float32",
            "--batch_size", "10",
            "--output_dir", output_folder,
            "--output_format", "srt",
            "--output_format", "json",
        ]

        logger.info(f"Transcrevendo: {input_file}...")
        result = run_cmd(command, text=True, check=False)
        logger.info(f"Comando executado: {command}")
        
        if result.returncode != 0:
            logger.error("Erro durante a transcrição:")
            logger.error(result.stderr)
        else:
            logger.info(f"Transcrição concluída. Arquivo salvo em: {output_file} e {json_file}")
            # print(result.stdout) 

    # Define o diretório de entrada e o diretório de saída
    input_folder = os.path.join(project_folder, 'final')
    output_folder = os.path.join(project_folder, 'subs')
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        logger.error(f"Pasta de entrada não encontrada: {input_folder}")
        return

    # Itera sobre todos os arquivos na pasta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):  # Filtra apenas arquivos .mp4
            input_file = os.path.join(input_folder, filename)
            generate_whisperx(input_file, output_folder)

