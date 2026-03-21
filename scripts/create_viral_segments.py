from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import ast
import io

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Configura stdout para evitar erros de encoding no Windows (substitui caracteres inválidos por ?)
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    try:
        # Mantém encoding original mas ignora erros (substitui por ?)
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding or 'utf-8', errors='replace', line_buffering=True)
    except (AttributeError, ValueError, OSError):
        pass  # stdout reconfiguration non critique

# Tenta importar bibliotecas de IA opcionalmente
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import g4f
    HAS_G4F = True
except ImportError:
    HAS_G4F = False

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

def clean_json_response(response_text: str) -> dict:
    """
    Limpa a resposta focando em encontrar o objeto JSON que contém a chave "segments".
    Estratégia: 
    1. Busca a palavra "segments", encontra o '{' anterior e usa raw_decode.
    2. Fallback: Parsear lista de segmentos item a item (recuperação de JSON truncado).
    """
    if not isinstance(response_text, str):
        response_text = str(response_text)
    
    if not response_text:
        return {"segments": []}

    # 1. Limpeza preliminar
    # Remove tags de pensamento (DeepSeek R1)
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    
    # Normaliza escapes excessivos (\n virando \\n) e aspas se parecer necessário
    try:
        if "\\n" in response_text or "\\\"" in response_text:
             # Tenta um decode básico de escapes
             response_text = response_text.replace("\\n", "\n").replace("\\\"", "\"").replace("\\'", "'")
    except (ValueError, TypeError):
        pass

    # 2. Busca pela palavra-chave "segments"
    # Procura índices de todas as ocorrências de 'segments'
    matches = [m.start() for m in re.finditer(r'segments', response_text)]
    
    if not matches:
        # Se não achou segments, retorna vazio
        return {"segments": []}

    # Tenta extrair JSON válido a partir de cada ocorrência
    for match_idx in matches:
        # Procura o '{' mais próximo ANTES de "segments"
        # Limita busca a 5000 chars para trás para performance
        start_search = max(0, match_idx - 5000)
        snippet_before = response_text[start_search:match_idx]
        
        # Encontra o ÚLTIMO '{' no snippet
        last_open_rel = snippet_before.rfind('{')
        
        if last_open_rel != -1:
            real_start = start_search + last_open_rel
            candidate_text = response_text[real_start:]
            
            # Tentativa A: json.raw_decode
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(candidate_text)
                if 'segments' in obj and isinstance(obj['segments'], list):
                    return obj
            except json.JSONDecodeError:
                pass

            # Tentativa B: ast.literal_eval
            try:
                balance = 0
                in_string = False
                escape = False
                found_end = -1
                
                for i, char in enumerate(candidate_text):
                    if escape:
                        escape = False
                        continue
                    if char == '\\':
                        escape = True
                        continue
                    if char == "'" or char == '"':
                        in_string = not in_string
                        continue
                        
                    if not in_string:
                        if char == '{':
                            balance += 1
                        elif char == '}':
                            balance -= 1
                            if balance == 0:
                                found_end = i
                                break
                
                if found_end != -1:
                    clean_cand = candidate_text[:found_end+1]
                    obj = ast.literal_eval(clean_cand)
                    if 'segments' in obj and isinstance(obj['segments'], list):
                        return obj
            except (json.JSONDecodeError, ValueError, SyntaxError):
                pass

    # 3. Fallback: Extração bruta de markdown
    try:
        match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except json.JSONDecodeError:
        pass
        
    # 4. LAST RESORT: Fragment Parser (Para JSON truncado/incompleto)
    # Procura por "segments": [ e tenta parsear item por item
    try:
        match_list = re.search(r'"segments"\s*:\s*\[', response_text)
        if match_list:
            start_pos = match_list.end()
            current_pos = start_pos
            found_segments = []
            decoder = json.JSONDecoder()
            
            while True:
                while current_pos < len(response_text) and response_text[current_pos] in ' \t\n\r,':
                    current_pos += 1
                
                if current_pos >= len(response_text):
                    break
                    
                if response_text[current_pos] == ']':
                    break
                
                try:
                    obj, end_pos = decoder.raw_decode(response_text[current_pos:])
                    if isinstance(obj, dict):
                        found_segments.append(obj)
                    current_pos += end_pos
                except json.JSONDecodeError:
                    break
                    
            if found_segments:
                logger.info(f"[INFO] Recuperado {len(found_segments)} segmentos de JSON truncado.")
                return {"segments": found_segments}
    except (json.JSONDecodeError, ValueError):
        pass

    return {"segments": []}


def preprocess_transcript_for_ai(segments: list[dict]) -> str:
    """
    Concatenates transcript segments into a single string with embedded time tags.
    """
    if not segments:
        return ""

    full_text = ""
    last_tag_time = -100  # Force first tag
    
    # Try to start with (0s) based on first segment
    first_start = segments[0].get('start', 0)
    full_text += f"({int(first_start)}s) "
    last_tag_time = first_start

    for seg in segments:
        text = seg.get('text', '').strip()
        end_time = seg.get('end', 0)
        
        full_text += text + " "
        
        if end_time - last_tag_time >= 4:
            full_text += f"({int(end_time)}s) "
            last_tag_time = end_time

    return full_text.strip()

def call_gemini(prompt: str, api_key: str, model_name: str = 'gemini-2.5-flash-lite-preview-09-2025') -> str:
    if not HAS_GEMINI:
        raise ImportError("A biblioteca 'google-generativeai' não está instalada. Instale com: pip install google-generativeai")
    
    genai.configure(api_key=api_key)
    # Usando modelo definido na config ou o padrão
    model = genai.GenerativeModel(model_name) 
    
    max_retries = 5
    base_wait = 30

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str:
                wait_time = base_wait * (attempt + 1)
                
                match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                if match:
                    wait_time = float(match.group(1)) + 5.0
                
                logger.info(f"[429] Quota Exceeded. Waiting {wait_time:.2f}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Erro na API do Gemini: {e}")
                return "{}"
    
    logger.error("Falha após max retries no Gemini.")
    return "{}"

def call_g4f(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    if not HAS_G4F:
        raise ImportError("A biblioteca 'g4f' não está instalada. Instale com: pip install g4f")
    
    max_retries = 3
    base_wait = 5
    
    for attempt in range(max_retries):
        try:
            response = g4f.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            
            if isinstance(response, dict):
                if 'error' in response:
                    raise Exception(f"API Error: {response['error']}")
                if 'choices' in response and isinstance(response['choices'], list):
                    if len(response['choices']) > 0:
                         content = response['choices'][0].get('message', {}).get('content', '')
                         if content:
                             return content
                if not response:
                     raise ValueError("Empty Dict response")

                return json.dumps(response)

            if not response:
                logger.warning(f"[WARN] G4F retornou resposta vazia. Tentativa {attempt+1}/{max_retries}")
                time.sleep(base_wait)
                continue
            
            if isinstance(response, str):
                return response

            try:
                return json.dumps(response, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(response)
            
        except Exception as e:
            logger.error(f"[WARN] Erro na API do G4F (Tentativa {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = base_wait * (2 ** attempt)
                time.sleep(wait_time)
            
    logger.error(f"Falha crítica após {max_retries} tentativas no G4F.")
    return "{}"

CONTENT_TYPES = [
    "anime", "comedy", "commentary", "cooking", "education", "gaming",
    "manga", "motivation", "music", "news", "podcast", "sport", "talkshow", "vlog"
]


def classify_content(transcript_excerpt: str, ai_mode: str, api_key: str | None = None, model_name: str | None = None) -> list[str]:
    """Classifie le type de contenu (multi-label) à partir des 2 premières minutes de transcript."""
    prompt = f"""Classify this video transcript into one or more categories (a video can belong to multiple categories, e.g. gaming + comedy).

Available categories: {', '.join(CONTENT_TYPES)}

Return ONLY JSON: {{"content_types": ["gaming", "comedy"], "confidence": 0.9}}

Rules:
- List ALL categories that clearly apply (usually 1-3)
- Only include categories with confidence >= 0.5
- Order by relevance (most relevant first)

TRANSCRIPT EXCERPT (first 2 minutes):
{transcript_excerpt[:3000]}"""

    if ai_mode == "pleiade":
        response = call_pleiade(prompt, model_name=model_name)
    elif ai_mode == "gemini":
        response = call_gemini(prompt, api_key, model_name=model_name or "gemini-2.5-flash-lite-preview-09-2025")
    elif ai_mode == "g4f":
        response = call_g4f(prompt, model_name=model_name or "gpt-4o-mini")
    else:
        return []

    try:
        data = clean_json_response_simple(response)
        content_types = data.get("content_types", [])
        confidence = data.get("confidence", 0)
        if isinstance(content_types, str):
            content_types = [content_types]
        valid = [ct for ct in content_types if ct in CONTENT_TYPES and confidence >= 0.5]
        if valid:
            logger.info(f"[INFO] Types de contenu détectés: {', '.join(valid)} (confiance: {confidence})")
            return valid
        # Fallback: try legacy single content_type field
        ct = data.get("content_type", "")
        if ct in CONTENT_TYPES and confidence >= 0.5:
            logger.info(f"[INFO] Type de contenu détecté: {ct} (confiance: {confidence})")
            return [ct]
    except Exception as e:
        logger.warning(f"[WARN] Échec classification contenu: {e}")

    return []


def clean_json_response_simple(text: str) -> dict:
    """Parse a JSON object from an LLM response (handles nested structures, think tags, markdown fences)."""
    if not text:
        return {}
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Try raw_decode from first '{' to handle nested objects
    start = text.find('{')
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Fallback: markdown code block
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def load_content_signals(content_type: str | list[str]) -> str:
    """Charge les signaux viraux spécifiques aux types de contenu (supporte multi-label)."""
    if not content_type:
        return ""
    types = [content_type] if isinstance(content_type, str) else content_type
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sections = []
    for ct in types:
        signals_path = os.path.join(base_dir, "prompts", f"signals_{ct}.txt")
        if os.path.exists(signals_path):
            with open(signals_path, 'r', encoding='utf-8') as f:
                sections.append(f"## Signals: {ct}\n{f.read()}")
    return ("\n\n" + "\n\n".join(sections)) if sections else ""


def score_segments(segments: list[dict], ai_mode: str, api_key: str | None = None, model_name: str | None = None, min_score: int = 70) -> list[dict]:
    """Passe de scoring : note chaque segment sur 5 dimensions (0-100). Filtre sous min_score."""
    if not segments:
        return segments

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scoring_path = os.path.join(base_dir, "prompts", "scoring.txt")

    if not os.path.exists(scoring_path):
        logger.warning("[WARN] prompts/scoring.txt non trouvé, scoring ignoré.")
        return segments

    with open(scoring_path, 'r', encoding='utf-8') as f:
        scoring_template = f.read()

    segments_json = json.dumps([
        {"index": i, "title": s.get("title", ""), "reasoning": s.get("reasoning", ""), "hook": s.get("hook", "")}
        for i, s in enumerate(segments)
    ], ensure_ascii=False)

    prompt = scoring_template.replace("{segments_json}", segments_json)

    logger.info(f"[INFO] Scoring de {len(segments)} segments...")

    if ai_mode == "pleiade":
        response = call_pleiade(prompt, model_name=model_name)
    elif ai_mode == "gemini":
        response = call_gemini(prompt, api_key, model_name=model_name or "gemini-2.5-flash-lite-preview-09-2025")
    elif ai_mode == "g4f":
        response = call_g4f(prompt, model_name=model_name or "gpt-4o-mini")
    else:
        return segments

    try:
        data = clean_json_response_simple(response)
        scores = data.get("scores", [])

        # Mapper les scores aux segments
        score_map = {s["index"]: s for s in scores}
        scored_segments = []
        for i, seg in enumerate(segments):
            if i in score_map:
                seg["viral_score"] = score_map[i].get("total", 0)
                seg["score_details"] = score_map[i]
                if seg["viral_score"] >= min_score:
                    scored_segments.append(seg)
                else:
                    logger.info(f"[SCORING] Segment '{seg.get('title', '')}' filtré (score={seg['viral_score']} < {min_score})")
            else:
                scored_segments.append(seg)  # Garder si pas de score

        scored_segments.sort(key=lambda x: x.get("viral_score", x.get("score", 0)), reverse=True)
        logger.info(f"[INFO] Scoring terminé: {len(scored_segments)}/{len(segments)} segments retenus (seuil={min_score})")
        return scored_segments

    except Exception as e:
        logger.warning(f"[WARN] Échec du scoring: {e}. Tous les segments conservés.")
        return segments


def _call_ai(prompt: str, ai_mode: str, api_key: str | None = None, model_name: str | None = None) -> str | None:
    """Helper centralisé pour appeler le LLM selon le mode."""
    if ai_mode == "pleiade":
        return call_pleiade(prompt, model_name=model_name)
    elif ai_mode == "gemini":
        return call_gemini(prompt, api_key, model_name=model_name or "gemini-2.5-flash-lite-preview-09-2025")
    elif ai_mode == "g4f":
        return call_g4f(prompt, model_name=model_name or "gpt-4o-mini")
    return None


def _validate_one_segment(seg: dict, transcript_text: str, validation_template: str, ai_mode: str, api_key: str | None, model_name: str | None) -> dict:
    """Valide UN segment via le LLM. Retourne {"decision": "keep"/"reject", "reason": "..."}."""
    excerpt = _extract_excerpt(transcript_text, seg.get("start_time", 0), seg.get("end_time"), max_chars=1200)
    segment_data = {
        "title": seg.get("title", ""),
        "hook": seg.get("hook", ""),
        "reasoning": seg.get("reasoning", ""),
        "transcript_excerpt": excerpt
    }
    prompt = validation_template.replace("{segment_json}", json.dumps(segment_data, ensure_ascii=False))

    response = _call_ai(prompt, ai_mode, api_key, model_name)
    if not response:
        return {"decision": "keep", "reason": "AI non disponible"}

    try:
        data = clean_json_response_simple(response)
        return {"decision": data.get("decision", "keep"), "reason": data.get("reason", "")}
    except Exception:
        return {"decision": "keep", "reason": "Erreur parsing, segment conservé"}


def _find_replacement_segment(
    rejected_seg: dict, reject_reason: str, validated_segments: list[dict],
    transcript_text: str, transcript_segments: list[dict],
    min_duration: int, max_duration: int,
    replacement_template: str, json_template: str,
    ai_mode: str, api_key: str | None, model_name: str | None
) -> dict | None:
    """Demande au LLM de trouver un segment de remplacement. Retourne le segment aligné ou None."""
    existing_info = json.dumps([
        {"title": s.get("title", ""), "start": s.get("start_time", 0), "end": s.get("end_time", 0)}
        for s in validated_segments
    ], ensure_ascii=False)

    prompt = replacement_template\
        .replace("{rejected_title}", rejected_seg.get("title", ""))\
        .replace("{reject_reason}", reject_reason)\
        .replace("{rejected_start}", str(int(rejected_seg.get("start_time", 0))))\
        .replace("{rejected_end}", str(int(rejected_seg.get("end_time", 0))))\
        .replace("{existing_segments}", existing_info)\
        .replace("{min_duration}", str(min_duration))\
        .replace("{max_duration}", str(max_duration))\
        .replace("{transcript_chunk}", transcript_text)\
        .replace("{json_template}", json_template)

    response = _call_ai(prompt, ai_mode, api_key, model_name)
    if not response:
        return None

    try:
        data = clean_json_response(response)
        raw_segments = data.get("segments", [])
        if not raw_segments:
            return None

        result = process_segments(raw_segments[:1], transcript_segments, min_duration, max_duration, output_count=1)
        new_segments = result.get("segments", [])
        return new_segments[0] if new_segments else None
    except Exception as e:
        logger.warning(f"[VALIDATION] Erreur remplacement: {e}")
        return None


MAX_VALIDATION_RETRIES = 5  # Max tentatives de remplacement par segment rejeté


def validate_segments(
    segments: list[dict], transcript_text: str, transcript_segments: list[dict],
    min_duration: int, max_duration: int, json_template: str,
    ai_mode: str, api_key: str | None = None, model_name: str | None = None
) -> list[dict]:
    """Valide chaque segment un par un. Si rejeté, cherche un remplacement (max MAX_VALIDATION_RETRIES fois)."""
    if not segments:
        return segments

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    validation_path = os.path.join(base_dir, "prompts", "validation.txt")
    replacement_path = os.path.join(base_dir, "prompts", "replacement.txt")

    if not os.path.exists(validation_path):
        logger.warning("[WARN] prompts/validation.txt non trouvé, validation ignorée.")
        return segments

    with open(validation_path, 'r', encoding='utf-8') as f:
        validation_template = f.read()

    replacement_template = ""
    if os.path.exists(replacement_path):
        with open(replacement_path, 'r', encoding='utf-8') as f:
            replacement_template = f.read()

    validated = []
    total_rejected = 0
    total_replaced = 0

    for i, seg in enumerate(segments):
        candidate = seg
        retries = 0
        accepted = False

        while retries <= MAX_VALIDATION_RETRIES:
            label = f"[VALIDATION {i+1}/{len(segments)}]"
            if retries > 0:
                label += f" (remplacement #{retries})"

            logger.info(f"{label} Test: '{candidate.get('title', '')}'")
            result = _validate_one_segment(candidate, transcript_text, validation_template, ai_mode, api_key, model_name)

            if result["decision"] == "keep":
                logger.info(f"{label} KEEP — {result['reason']}")
                validated.append(candidate)
                accepted = True
                break

            # Rejeté
            logger.info(f"{label} REJECT — {result['reason']}")
            total_rejected += 1
            retries += 1

            if retries > MAX_VALIDATION_RETRIES:
                logger.info(f"{label} Max tentatives atteint, segment abandonné.")
                break

            # Chercher un remplacement
            if not replacement_template:
                logger.info(f"{label} Pas de prompt de remplacement, segment abandonné.")
                break

            logger.info(f"{label} Recherche d'un remplacement...")
            new_seg = _find_replacement_segment(
                candidate, result["reason"], validated,
                transcript_text, transcript_segments,
                min_duration, max_duration,
                replacement_template, json_template,
                ai_mode, api_key, model_name
            )

            if not new_seg:
                logger.info(f"{label} Aucun remplacement trouvé, segment abandonné.")
                break

            # Vérifier que le remplacement n'est pas le même segment
            old_start = int(candidate.get("start_time", 0))
            new_start = int(new_seg.get("start_time", -1))
            if abs(old_start - new_start) < 10:
                logger.info(f"{label} Remplacement trop similaire (même timestamps), segment abandonné.")
                break

            total_replaced += 1
            candidate = new_seg

    logger.info(f"[INFO] Validation terminée: {len(validated)}/{len(segments)} segments retenus "
                f"({total_rejected} rejetés, {total_replaced} remplacements tentés)")
    return validated


def _extract_excerpt(transcript_text: str, start_time: float, end_time: float | None = None, max_chars: int = 800) -> str:
    """Extrait le texte de la transcription entre start_time et end_time (ou max_chars si pas d'end_time)."""
    def find_pos(t: float) -> int:
        p = transcript_text.find(f"({int(t)}s)")
        if p != -1:
            return p
        for delta in range(1, 15):
            for candidate in [int(t) - delta, int(t) + delta]:
                p = transcript_text.find(f"({candidate}s)")
                if p != -1:
                    return p
        return -1

    start_pos = find_pos(start_time)
    if start_pos == -1:
        return transcript_text[:max_chars]

    if end_time is not None:
        end_pos = find_pos(end_time)
        if end_pos != -1 and end_pos > start_pos:
            return transcript_text[start_pos:end_pos].strip()

    return transcript_text[start_pos:start_pos + max_chars].strip()


def generate_tiktok_captions(
    segments: list[dict],
    transcript_text: str,
    ai_mode: str,
    api_key: str | None = None,
    model_name: str | None = None,
    content_type: list[str] | None = None
) -> list[dict]:
    """Génère une caption TikTok engageante + hashtags pour chaque segment (1 seule passe LLM)."""
    if not segments:
        return segments

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(base_dir, "prompts", "tiktok_caption.txt")

    if not os.path.exists(prompt_path):
        logger.warning("[WARN] prompts/tiktok_caption.txt non trouvé, captions ignorées.")
        return segments

    with open(prompt_path, 'r', encoding='utf-8') as f:
        caption_template = f.read()

    segments_json = json.dumps([
        {
            "index": i,
            "title": s.get("title", ""),
            "reasoning": s.get("reasoning", ""),
            "duration": int(s.get("duration", 0)),
            "transcript_excerpt": _extract_excerpt(transcript_text, s.get("start_time", 0), s.get("end_time"))
        }
        for i, s in enumerate(segments)
    ], ensure_ascii=False)

    types_str = ", ".join(content_type) if content_type else "general"
    prompt = caption_template.replace("{segments_json}", segments_json)
    prompt = prompt.replace("{content_type}", types_str)

    # Debug: save caption prompt
    try:
        debug_caption_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VIRALS", "debug_caption_prompt.txt")
        with open(debug_caption_path, "w", encoding="utf-8") as _f:
            _f.write(prompt)
    except Exception:
        pass

    logger.info(f"[INFO] Génération des captions TikTok pour {len(segments)} segments...")

    if ai_mode == "pleiade":
        response = call_pleiade(prompt, model_name=model_name)
    elif ai_mode == "gemini":
        response = call_gemini(prompt, api_key, model_name=model_name or "gemini-2.5-flash-lite-preview-09-2025")
    elif ai_mode == "g4f":
        response = call_g4f(prompt, model_name=model_name or "gpt-4o-mini")
    else:
        return segments

    try:
        # Debug: save caption raw response
        try:
            debug_resp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VIRALS", "debug_caption_response.txt")
            with open(debug_resp_path, "w", encoding="utf-8") as _f:
                _f.write(response if isinstance(response, str) else str(response))
        except Exception:
            pass

        data = clean_json_response_simple(response)
        captions = data.get("captions", [])

        caption_map = {c["index"]: c["caption"] for c in captions if "index" in c and "caption" in c}
        for i, seg in enumerate(segments):
            caption = caption_map.get(i, "")
            # Garantir #fyp et #pourtoi (filet de securite)
            if caption and "#fyp" not in caption.lower():
                caption = caption.rstrip() + " #fyp"
            if caption and "#pourtoi" not in caption.lower():
                caption = caption.rstrip() + " #pourtoi"
            seg["tiktok_caption"] = caption

        logger.info(f"[INFO] Captions TikTok générées pour {len(caption_map)}/{len(segments)} segments.")
        return segments

    except Exception as e:
        logger.warning(f"[WARN] Échec de la génération des captions TikTok: {e}. Segments conservés sans caption.")
        return segments


MAX_CAPTION_RETRIES = 5  # Max tentatives de correction par caption rejetée


def validate_captions(
    segments: list[dict], transcript_text: str,
    ai_mode: str, api_key: str | None = None, model_name: str | None = None
) -> list[dict]:
    """Valide chaque caption TikTok une par une. Si rejetée, utilise la version corrigée par le LLM (max MAX_CAPTION_RETRIES fois)."""
    if not segments:
        return segments

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    validation_path = os.path.join(base_dir, "prompts", "caption_validation.txt")

    if not os.path.exists(validation_path):
        logger.warning("[WARN] prompts/caption_validation.txt non trouvé, validation captions ignorée.")
        return segments

    with open(validation_path, 'r', encoding='utf-8') as f:
        validation_template = f.read()

    total_fixed = 0

    for i, seg in enumerate(segments):
        caption = seg.get("tiktok_caption", "")
        if not caption:
            continue

        retries = 0
        while retries <= MAX_CAPTION_RETRIES:
            label = f"[CAPTION VALIDATION {i+1}/{len(segments)}]"
            if retries > 0:
                label += f" (correction #{retries})"

            excerpt = _extract_excerpt(transcript_text, seg.get("start_time", 0), seg.get("end_time"), max_chars=800)
            caption_data = {"caption": caption, "transcript_excerpt": excerpt}
            prompt = validation_template.replace("{caption_json}", json.dumps(caption_data, ensure_ascii=False))

            response = _call_ai(prompt, ai_mode, api_key, model_name)
            if not response:
                break

            try:
                data = clean_json_response_simple(response)
            except Exception:
                break

            decision = data.get("decision", "keep")
            reason = data.get("reason", "")

            if decision == "keep":
                logger.info(f"{label} KEEP — {reason}")
                # Utiliser fixed_caption si fourni (peut être nettoyé)
                fixed = data.get("fixed_caption", "")
                if fixed and fixed != caption:
                    seg["tiktok_caption"] = fixed
                break

            # Rejetée
            logger.info(f"{label} REJECT — {reason}")
            fixed = data.get("fixed_caption", "")
            retries += 1

            if fixed and fixed != caption:
                total_fixed += 1
                caption = fixed
                seg["tiktok_caption"] = fixed
                # Re-valider la version corrigée au prochain tour de boucle
            else:
                logger.info(f"{label} Pas de correction proposée, caption conservée telle quelle.")
                break

            if retries > MAX_CAPTION_RETRIES:
                logger.info(f"{label} Max corrections atteint, caption conservée.")
                break

        # Filet de sécurité hashtags
        final_caption = seg.get("tiktok_caption", "")
        if final_caption and "#fyp" not in final_caption.lower():
            final_caption = final_caption.rstrip() + " #fyp"
        if final_caption and "#pourtoi" not in final_caption.lower():
            final_caption = final_caption.rstrip() + " #pourtoi"
        seg["tiktok_caption"] = final_caption

    logger.info(f"[INFO] Validation captions terminée: {total_fixed} captions corrigées sur {len(segments)}")
    return segments


def call_pleiade(prompt: str, model_name: str | None = None) -> str:
    """Appelle l'API Pléiade via HTTP (requests) — aucune dépendance openai."""
    if not HAS_REQUESTS:
        raise ImportError("La bibliothèque 'requests' n'est pas installée.")

    api_url = os.getenv("PLEIADE_API_URL")
    api_key = os.getenv("PLEIADE_API_KEY")

    if not api_url or not api_key:
        raise ValueError("PLEIADE_API_URL et PLEIADE_API_KEY doivent être définis dans .env")

    model = model_name or os.getenv("PLEIADE_CHAT_MODEL", "deepseek-R1:70b")

    endpoint = api_url.rstrip("/") + "/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    max_retries = 3
    base_wait = 10

    for attempt in range(max_retries):
        try:
            resp = _requests.post(endpoint, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait_time = base_wait * (attempt + 1)
                logger.info(f"[429] Pléiade rate limit. Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except _requests.exceptions.RequestException as e:
            logger.error(f"Erreur API Pléiade (tentative {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_wait)

    logger.error("Échec après max retries sur Pléiade.")
    return "{}"


def load_transcript(project_folder: str) -> list[dict]:
    """Parses input.tsv or input.srt from the project folder."""
    input_tsv = os.path.join(project_folder, 'input.tsv')
    input_srt = os.path.join(project_folder, 'input.srt')

    transcript_segments = []
    
    # Try to load TSV first (more reliable time)
    if os.path.exists(input_tsv):
        try:
            with open(input_tsv, 'r', encoding='utf-8') as f:
                # Skip header
                lines = f.readlines()[1:] 
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        start_ms = float(parts[0])
                        end_ms = float(parts[1])
                        text = parts[2]
                        transcript_segments.append({
                            'start': start_ms / 1000.0, 
                            'end': end_ms / 1000.0, 
                            'text': text
                        })
        except Exception as e:
            logger.error(f"Error parsing TSV: {e}")

    # Fallback to SRT parser if TSV empty/failed
    if not transcript_segments and os.path.exists(input_srt):
         with open(input_srt, 'r', encoding='utf-8') as f:
             srt_content = f.read()
         pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\n\n).)*)', re.DOTALL)
         matches = pattern.findall(srt_content)
         
         def srt_time_to_seconds(t_str):
             h, m, s = t_str.replace(',', '.').split(':')
             return int(h) * 3600 + int(m) * 60 + float(s)

         for m in matches:
             start_sec = srt_time_to_seconds(m[1])
             end_sec = srt_time_to_seconds(m[2])
             text = m[3].replace('\n', ' ')
             transcript_segments.append({'start': start_sec, 'end': end_sec, 'text': text})

    if not transcript_segments:
        raise ValueError("Could not parse transcript from TSV or SRT.")
    
    return transcript_segments

def process_segments(raw_segments: list[dict], transcript_segments: list[dict], min_duration: int, max_duration: int, output_count: int | None = None) -> dict:
    """
    Aligns raw AI segments (with reference tags) to actual transcript timestamps.
    Applies constraints, validation, and deduplication.
    """
    
    all_segments = raw_segments
    tempo_minimo = min_duration
    tempo_maximo = max_duration
    
    # Sort segments by score (descending)
    try:
        all_segments.sort(key=lambda x: int(x.get('score', 0)), reverse=True)
    except (ValueError, TypeError):
        pass

    # --- POST-PROCESSING: Match Text to Timestamps ---
    processed_segments = []
    
    logger.debug(f"[DEBUG] Matching {len(all_segments)} raw segments to timestamps...")
    
    for seg in all_segments:
        try:
            # 1. Parse Reference Time
            ref_time_str = seg.get('start_time_ref', '(0s)')
            ref_time_val = 0
            try:
                if isinstance(ref_time_str, str):
                    match = re.search(r'\d+', ref_time_str)
                    if match:
                         ref_time_val = int(match.group())
                else:
                    ref_time_val = int(ref_time_str)
            except (ValueError, TypeError):
                ref_time_val = 0
                
            # Find segment index closest to ref_time
            start_idx = 0
            min_diff = 999999
            for i, s in enumerate(transcript_segments):
                diff = abs(s['start'] - ref_time_val)
                if diff < min_diff:
                    min_diff = diff
                    start_idx = i
                if s['start'] > ref_time_val + 10: 
                    break
            
            # Backtrack
            start_idx = max(0, start_idx - 5)
            
            # 2. Find Exact Start Text
            start_text_target = seg.get('start_text', '').lower().strip()
            start_text_target = re.sub(r'[^\w\s]', '', start_text_target)

            final_start_time = -1
            match_start_idx = -1

            search_limit = min(len(transcript_segments), start_idx + 50)
            window = 4  # number of consecutive segments to concatenate

            for i in range(start_idx, search_limit):
                # Build a sliding window of concatenated text
                end_w = min(len(transcript_segments), i + window)
                window_text = ' '.join(transcript_segments[j]['text'] for j in range(i, end_w)).lower()
                window_text = re.sub(r'[^\w\s]', '', window_text)

                if start_text_target and (start_text_target in window_text or window_text in start_text_target):
                    final_start_time = transcript_segments[i]['start']
                    match_start_idx = i
                    break

            # Fallback
            if final_start_time == -1:
                final_start_time = transcript_segments[start_idx]['start'] if start_idx < len(transcript_segments) else ref_time_val
                match_start_idx = start_idx

            # 3. Find End Text
            end_text_target = seg.get('end_text', '').lower().strip()
            end_text_target = re.sub(r'[^\w\s]', '', end_text_target)

            final_end_time = -1

            if match_start_idx != -1:
                search_end_limit = min(len(transcript_segments), match_start_idx + 200)

                for i in range(match_start_idx, search_end_limit):
                    end_w = min(len(transcript_segments), i + window)
                    window_text = ' '.join(transcript_segments[j]['text'] for j in range(i, end_w)).lower()
                    window_text = re.sub(r'[^\w\s]', '', window_text)

                    if end_text_target and (end_text_target in window_text or window_text in end_text_target):
                        final_end_time = transcript_segments[min(end_w - 1, len(transcript_segments) - 1)]['end']
                        break
            
            # Fallback End Time — try end_time_ref before using duration average
            if final_end_time == -1:
                end_ref_str = seg.get('end_time_ref', '')
                end_ref_val = -1
                try:
                    if isinstance(end_ref_str, str):
                        m = re.search(r'\d+', end_ref_str)
                        if m:
                            end_ref_val = int(m.group())
                    elif end_ref_str:
                        end_ref_val = int(end_ref_str)
                except Exception:
                    end_ref_val = -1

                if end_ref_val > final_start_time + tempo_minimo * 0.5:
                    # Find transcript segment closest to end_ref_val
                    best_end_idx = len(transcript_segments) - 1
                    min_end_diff = 999999
                    for i, s in enumerate(transcript_segments):
                        diff = abs(s['start'] - end_ref_val)
                        if diff < min_end_diff:
                            min_end_diff = diff
                            best_end_idx = i
                        if s['start'] > end_ref_val + 10:
                            break
                    final_end_time = transcript_segments[best_end_idx]['end']
                    logger.debug(f"[DEBUG] end_text not matched — using end_time_ref={end_ref_val}s → end_time={final_end_time:.2f}s")
                else:
                    fallback_duration = (tempo_minimo + tempo_maximo) / 2
                    final_end_time = final_start_time + fallback_duration
                    logger.debug(f"[DEBUG] end_text and end_time_ref both unavailable — fallback duration {fallback_duration:.0f}s")
            
            # Calculate Duration
            duration = final_end_time - final_start_time
            
            # Log duration warnings but keep LLM choice as-is
            if duration < tempo_minimo:
                logger.warning(f"[WARN] Segmento menor que duration min ({duration:.2f}s < {tempo_minimo}s). Mantendo timestamps do LLM.")
            if duration > tempo_maximo:
                logger.warning(f"[WARN] Segmento excede max duration ({duration:.2f}s > {tempo_maximo}s). Mantendo timestamps do LLM.")

            # Construct Final Segment
            processed_segments.append({
                "title": seg.get('title', 'Viral Segment'),
                "start_time": final_start_time,
                "end_time": final_end_time,
                "hook": seg.get('title', ''), 
                "reasoning": seg.get('reasoning', ''),
                "score": seg.get('score', 0),
                "duration": duration
            })

        except Exception as e:
            logger.error(f"[WARN] Error processing segment {seg}: {e}")
            continue

    # Deduplication
    unique_segments = []
    processed_segments.sort(key=lambda x: int(x.get('score', 0)), reverse=True)
    
    for candidate in processed_segments:
        is_dup = False
        for existing in unique_segments:
            s1, e1 = candidate['start_time'], candidate['end_time']
            # Simple float equality isn't safe, but max/min handles it
            s2, e2 = existing['start_time'], existing['end_time']
            
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            
            if overlap_end > overlap_start:
                intersection = overlap_end - overlap_start
                if intersection > 5: # more than 5 seconds overlap
                    is_dup = True
                    logger.debug(f"[DEBUG] Dropping overlap: '{candidate.get('title')}' ({s1:.1f}-{e1:.1f}) overlaps with '{existing.get('title')}' ({s2:.1f}-{e2:.1f}) by {intersection:.1f}s")
                    break
        if not is_dup:
            unique_segments.append(candidate)

    all_segments = unique_segments
    logger.debug(f"[DEBUG] Finished processing. {len(all_segments)} segments valid.")

    if output_count and len(all_segments) > output_count:
        logger.info(f"[INFO] LLM retornou {len(all_segments)} segmentos (pedido: {output_count}). Mantendo todos.")

    final_result = {"segments": all_segments}
    
    # Validação básica de que temos start_time
    validated_segments = []
    for seg in final_result['segments']:
        if 'start_time' in seg:
             validated_segments.append(seg)
    
    final_result['segments'] = validated_segments
    
    return final_result


def create(num_segments: int | None, viral_mode: bool, themes: str | None, tempo_minimo: int, tempo_maximo: int, ai_mode: str = "manual", api_key: str | None = None, project_folder: str = "tmp", chunk_size_arg: int | str | None = None, model_name_arg: str | None = None, content_type: list[str] | None = None, enable_scoring: bool = True, min_score: int = 70, enable_validation: bool = True) -> dict:
    quantidade_de_virals = num_segments if num_segments is not None else 3

    # 1. Load Transcript
    transcript_segments = load_transcript(project_folder)

    # 2. Pre-process Content
    formatted_content = preprocess_transcript_for_ai(transcript_segments)
    content = formatted_content

    # Load Config and Prompt
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'api_config.json')
    prompt_path = os.path.join(base_dir, 'prompt.txt')

    config = {
        "selected_api": "gemini",
        "gemini": {
            "api_key": "",
            "model": "gemini-2.5-flash-lite-preview-09-2025",
            "chunk_size": 15000
        },
        "g4f": {
            "model": "gpt-4o-mini",
            "chunk_size": 2000
        }
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                if "gemini" in loaded_config: config["gemini"].update(loaded_config["gemini"])
                if "g4f" in loaded_config: config["g4f"].update(loaded_config["g4f"])
                if "pleiade" in loaded_config: config["pleiade"] = loaded_config["pleiade"]
                if "selected_api" in loaded_config: config["selected_api"] = loaded_config["selected_api"]
        except Exception as e:
            logger.error(f"Erro ao ler api_config.json: {e}")

    # Config Vars
    current_chunk_size = 15000
    model_name = ""
    
    if ai_mode == "gemini":
        cfg_chunk = config["gemini"].get("chunk_size", 15000)
        current_chunk_size = chunk_size_arg if chunk_size_arg and int(chunk_size_arg) > 0 else cfg_chunk
        cfg_model = config["gemini"].get("model", "gemini-2.5-flash-lite-preview-09-2025")
        model_name = model_name_arg if model_name_arg else cfg_model
        if not api_key: api_key = config["gemini"].get("api_key", "")
            
    elif ai_mode == "g4f":
        cfg_chunk = config["g4f"].get("chunk_size", 2000)
        current_chunk_size = chunk_size_arg if chunk_size_arg and int(chunk_size_arg) > 0 else cfg_chunk
        cfg_model = config["g4f"].get("model", "gpt-4o-mini")
        model_name = model_name_arg if model_name_arg else cfg_model

    elif ai_mode == "pleiade":
        cfg_chunk = config.get("pleiade", {}).get("chunk_size", 15000)
        current_chunk_size = chunk_size_arg if chunk_size_arg and int(chunk_size_arg) > 0 else cfg_chunk
        model_name = model_name_arg if model_name_arg else os.getenv("PLEIADE_CHAT_MODEL", "deepseek-R1:70b")

    elif ai_mode == "local":
        current_chunk_size = chunk_size_arg if chunk_size_arg and int(chunk_size_arg) > 0 else 3000
        model_name = model_name_arg if model_name_arg else ""

    # --- Classification automatique du contenu (multi-label) ---
    if not content_type and ai_mode in ("pleiade", "gemini", "g4f"):
        content_type = classify_content(content[:3000], ai_mode, api_key, model_name)
    # Normaliser en liste pour usage uniforme
    if isinstance(content_type, str) and content_type:
        content_type = [content_type]
    elif not content_type:
        content_type = []

    content_signals = load_content_signals(content_type)

    system_prompt_template = ""
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt_template = f.read()
        # Injecter les signaux spécifiques au type de contenu
        if content_signals:
            system_prompt_template = system_prompt_template.replace(
                "### YOUR TASK",
                content_signals + "\n\n### YOUR TASK"
            )
    else:
        logger.warning("Aviso: prompt.txt não encontrado. Usando prompt interno.")
        system_prompt_template = """You are a World-Class Viral Video Editor.
{context_instruction}
Analyze the transcript below with time tags (XXs). Find {amount} viral segments.
Constraints: Each segment MUST be between {min_duration} seconds and {max_duration} seconds.
IMPORTANT: Output "Title", "Hook", and "Reasoning" in the SAME LANGUAGE as the transcript (e.g., if transcript is Portuguese, output Portuguese).
TRANSCRIPT:
{transcript_chunk}
OUTPUT JSON ONLY:
{json_template}"""


    json_template = '''{"segments": [{"start_text": "exact 5-10 first words at segment start", "end_text": "exact 5-10 last words at segment end (DIFFERENT from start_text)", "start_time_ref": 0, "end_time_ref": 0, "title": "viral title in transcript language", "reasoning": "why this is viral (1 sentence)", "score": 75}]}

Rules: integers only for times. Duration (end - start) must be ''' + str(tempo_minimo) + '''-''' + str(tempo_maximo) + '''s. Target 60-80s. Score: 90+ = exceptional (max 1), 75-89 = strong, 60-74 = decent, <60 = skip.'''

    # Chunking
    chunk_size = int(current_chunk_size)
    overlap_size = max(1000, int(chunk_size * 0.1))
    
    chunks = []
    start = 0
    content_len = len(content)

    logger.debug(f"[DEBUG] Chunking content (Size: {content_len}) with Chunk Size: {chunk_size} and Overlap: {overlap_size}")

    while start < content_len:
        end = min(start + chunk_size, content_len)
        if end < content_len:
            last_space = content.rfind(' ', start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        chunk_text = content[start:end]
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end >= content_len:
            break
        next_start = max(start + 1, end - overlap_size)
        safe_space = content.rfind(' ', start, next_start)
        if safe_space != -1:
            start = safe_space + 1
        else:
            start = next_start

    if viral_mode:
        virality_instruction = f"""analyze the segment for potential virality and identify {{amount}} most viral segments from the transcript"""
    else:
        virality_instruction = f"""analyze the segment for potential virality and identify {{amount}} the best parts based on the list of themes {themes}."""

    num_chunks = len(chunks)
    amount_per_chunk = max(2, -(-quantidade_de_virals // num_chunks))  # at least 2 candidates per chunk

    output_texts = []
    for i, chunk in enumerate(chunks):
        context_instruction = ""
        if num_chunks > 1:
            chunk_times = [int(m) for m in re.findall(r'\((\d+)s\)', chunk)]
            if chunk_times:
                context_instruction = (
                    f"Part {i+1} of {num_chunks} (timestamps {chunk_times[0]}s to {chunk_times[-1]}s). "
                    f"Find segments ONLY within this time range. "
                )
            else:
                context_instruction = f"Part {i+1} of {num_chunks}. "

        chunk_amount = amount_per_chunk

        try:
            prompt = system_prompt_template.format(
                context_instruction=context_instruction,
                virality_instruction=virality_instruction,
                min_duration=tempo_minimo,
                max_duration=tempo_maximo,
                transcript_chunk=chunk,
                json_template=json_template,
                amount=chunk_amount
            )
        except KeyError as e:
            prompt = system_prompt_template
            prompt = prompt.replace("{context_instruction}", context_instruction)
            prompt = prompt.replace("{virality_instruction}", virality_instruction)
            prompt = prompt.replace("{min_duration}", str(tempo_minimo))
            prompt = prompt.replace("{max_duration}", str(tempo_maximo))
            prompt = prompt.replace("{transcript_chunk}", chunk)
            prompt = prompt.replace("{json_template}", json_template)
            prompt = prompt.replace("{amount}", str(chunk_amount))

        output_texts.append(prompt)

    try:
        full_prompt_path = os.path.join(project_folder, "prompt_full.txt")
        full_prompt = system_prompt_template
        full_prompt = full_prompt.replace("{context_instruction}", "Full Video Transcript Analysis")
        full_prompt = full_prompt.replace("{virality_instruction}", virality_instruction)
        full_prompt = full_prompt.replace("{min_duration}", str(tempo_minimo))
        full_prompt = full_prompt.replace("{max_duration}", str(tempo_maximo))
        full_prompt = full_prompt.replace("{transcript_chunk}", content) 
        full_prompt = full_prompt.replace("{json_template}", json_template)
        full_prompt = full_prompt.replace("{amount}", str(quantidade_de_virals))
        
        with open(full_prompt_path, "w", encoding="utf-8") as f:
            f.write(full_prompt)
    except Exception as e:
        logger.warning(f"[WARN] Could not save prompt_full.txt: {e}")

    all_raw_segments = []

    logger.info(f"Processando {len(output_texts)} chunks usando modo: {ai_mode.upper()}")

    local_llm_instance = None
    if ai_mode == "local":
        if not HAS_LLAMA_CPP:
            logger.error("Error: llama-cpp-python not installed. Please install it to use Local mode.")
            return {"segments": []}
            
        models_dir = os.path.join(base_dir, 'models')
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
             if os.path.exists(model_name):
                 model_path = model_name
             else:
                 logger.error(f"Error: Model not found at {model_path}")
                 return {"segments": []}
        
        logger.info(f"[INFO] Loading Local Model: {os.path.basename(model_path)} (This may take a while)...")
        try:
            local_llm_instance = Llama(
                model_path=model_path,
                n_gpu_layers=-1, 
                n_ctx=8192,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"segments": []}

    for i, prompt in enumerate(output_texts):
        response_text = ""
        manual_prompt_path = os.path.join(project_folder, f"prompt_part_{i+1}.txt")
        try:
            with open(manual_prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception as e:
            logger.error(f"[ERRO] Falha ao salvar prompt.txt: {e}")
        
        if ai_mode == "manual":
            logger.info(f"\n[INFO] O prompt foi salvo em: {manual_prompt_path}")
            logger.info("\n" + "="*60)
            logger.info(f"CHUNK {i+1}/{len(output_texts)}")
            logger.info("="*60)
            logger.info("COPIE O PROMPT ABAIXO (OU DO ARQUIVO GERADO) E COLE NA SUA IA PREFERIDA:")
            logger.info("-" * 20)
            logger.info(prompt)
            logger.info("-" * 20)
            logger.info("="*60)
            logger.info("Cole o JSON de resposta abaixo e pressione ENTER.")
            logger.info("Dica: Se o JSON tiver múltiplas linhas, tente colar tudo de uma vez ou minificado.")
            logger.info("Se preferir, digite 'file' para ler de um arquivo 'tmp/response.json'.")
            
            user_input = input("JSON ou 'file': ")
            
            if user_input.lower() == 'file':
                try:
                    response_json_path = os.path.join(project_folder, 'response.json')
                    with open(response_json_path, 'r', encoding='utf-8') as rf:
                        response_text = rf.read()
                except FileNotFoundError:
                    logger.info(f"Arquivo {response_json_path} não encontrado.")
            else:
                response_text = user_input
                if response_text.strip().startswith("{") and not response_text.strip().endswith("}"):
                    logger.info("Parece incompleto. Cole o resto e dê Enter (ou Ctrl+C para cancelar):")
                    try:
                        rest = sys.stdin.read()
                        response_text += rest
                    except (KeyboardInterrupt, EOFError):
                        pass

        elif ai_mode == "gemini":
            logger.info(f"Enviando chunk {i+1} para o Gemini (Model: {model_name})...")
            response_text = call_gemini(prompt, api_key, model_name=model_name)
        elif ai_mode == "g4f":
            logger.info(f"Enviando chunk {i+1} para o G4F (Model: {model_name})...")
            response_text = call_g4f(prompt, model_name=model_name)
        elif ai_mode == "pleiade":
            logger.info(f"Enviando chunk {i+1} para Pléiade (Model: {model_name})...")
            response_text = call_pleiade(prompt, model_name=model_name)
        elif ai_mode == "local" and local_llm_instance:
            logger.info(f"Processing chunk {i+1} with Local LLM...")
            try:
                output = local_llm_instance.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096,
                    temperature=0.7
                )
                response_text = output['choices'][0]['message']['content']
            except Exception as e:
                logger.error(f"Error evaluating local model: {e}")
                response_text = "{}"

        # --- Save RAW Response for Debugging ---
        try:
            raw_response_path = os.path.join(project_folder, f"response_raw_part_{i+1}.txt")
            with open(raw_response_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            logger.debug(f"[DEBUG] Raw response saved to: {raw_response_path}")
        except Exception as e:
            logger.warning(f"Failed to save raw response: {e}")

        # Processar resposta
        try:
            data = clean_json_response(response_text)
            chunk_segments = data.get("segments", [])
            logger.info(f"Encontrados {len(chunk_segments)} segmentos neste chunk.")
            all_raw_segments.extend(chunk_segments)
        except json.JSONDecodeError:
            logger.error(f"Erro: Resposta inválida.")
        except Exception as e:
            logger.error(f"Erro desconhecido ao processar chunk: {e}")

    # Call the alignment / processing logic
    result = process_segments(
        all_raw_segments,
        transcript_segments,
        tempo_minimo,
        tempo_maximo,
        output_count=quantidade_de_virals
    )

    # --- Passe de validation (optionnelle) — valide segment par segment, remplace les rejetés ---
    if enable_validation and ai_mode in ("pleiade", "gemini", "g4f") and result.get("segments"):
        result["segments"] = validate_segments(
            result["segments"], content, transcript_segments,
            tempo_minimo, tempo_maximo, json_template,
            ai_mode, api_key, model_name
        )

    # --- Passe de scoring (optionnelle) ---
    if enable_scoring and ai_mode in ("pleiade", "gemini", "g4f") and result.get("segments"):
        result["segments"] = score_segments(
            result["segments"], ai_mode, api_key, model_name, min_score=min_score
        )

    # Stocker le content_type pour usage ulterieur (captions, etc.)
    if content_type:
        result["content_type"] = content_type

    return result