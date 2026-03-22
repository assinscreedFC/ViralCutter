# Technology Stack

**Project:** Manga/Manhwa to Video Pipeline (ViralCutter extension)
**Researched:** 2026-03-22

## Recommended Stack

### Panel Detection & Manga Analysis

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| MAGI v3 | latest | Full manga analysis (panels, text, characters, OCR, speaker assignment) | SOTA, CVPR 2024, 430 stars, single model does everything |
| Manga-Panel-Extractor (adenzu) | v1.1.4 | Fallback panel extraction for manga | 63 stars, lightweight, no ML needed |
| OpenCV | 4.x | Manhwa vertical strip splitting + bubble detection | Already in ViralCutter stack, simple gap detection |

### OCR & Text Extraction

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PaddleOCR | v2.x | FR/EN text extraction from speech bubbles | Best accuracy for stylized fonts, DBNet detection |
| ComiQ | latest | Comic-specific hybrid OCR wrapper | Combines PaddleOCR + EasyOCR + AI grouping into bubbles |
| manga-ocr | v0.1.14 | Japanese manga text (if needed) | 2.6k stars, specialized for JP, not useful for FR/EN |

### Speech Bubble Detection

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| YOLOv8m (ogkalu model) | v8 | Speech bubble detection + classification | Trained on 8k manga/webtoon/comic images, on HuggingFace |
| MAGI v3 | latest | Integrated bubble detection with speaker assignment | Handles full pipeline if manga is Japanese-style |

### TTS - Text to Speech

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Qwen3-TTS | 0.6B-CustomVoice | Multi-speaker narration with emotion control | Apache 2.0, FR support, voice design, 4-6GB VRAM, 97ms latency |
| Qwen3-TTS | 1.7B-CustomVoice | Higher quality alternative (if VRAM allows) | Better emotion control, 6-8GB VRAM |

### Video Generation

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| ffmpeg | 7.x | Ken Burns effects, transitions, audio sync, final assembly | Already in ViralCutter, zoompan filter built-in |
| kburns-slideshow | v1.10 | Reference patterns for Ken Burns + 30 transitions | MIT license, 71 stars, Python+ffmpeg |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pillow | 10.x | Image manipulation, panel cropping | Panel extraction post-processing |
| numpy | 1.x | Image array operations | Panel processing, mask operations |
| aubio | latest | Music onset detection for audio sync | Sync panel transitions to music beats |
| transformers | 4.x | MAGI model loading | Required by MAGI |
| ultralytics | 8.x | YOLOv8 inference for bubble detection | Bubble detection pipeline |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Manga analysis | MAGI v3 | Kumiko (181 stars) | Kumiko is OpenCV-only, no ML, last updated 2018, no OCR/speaker |
| Panel detection | MAGI + adenzu extractor | ComicPanelSegmentation (20 stars) | Too basic, only works on clean digital panels |
| Panel detection | MAGI | DeepPanel | Android-focused, TensorFlow Lite, not for Python pipeline |
| OCR (FR/EN) | PaddleOCR | Tesseract | Much lower accuracy on stylized comic fonts |
| OCR (FR/EN) | PaddleOCR | EasyOCR | Slower, lower accuracy than PaddleOCR on structured text |
| TTS | Qwen3-TTS | CosyVoice 2 | Higher VRAM, less language coverage, streaming focus |
| TTS | Qwen3-TTS | ChatTTS | No French support, limited to CN/EN |
| TTS | Qwen3-TTS | Fish Speech v1.5 | 102s latency (unusable), despite good quality |
| TTS | Qwen3-TTS | XTTS-v2 | Decent but lower quality than Qwen3, Coqui discontinued |
| TTS | Qwen3-TTS | Bark | Low quality, no voice cloning, slow |
| TTS | Qwen3-TTS | Piper | Fast but robotic, no emotion control, no cloning |
| Video | ffmpeg | MoviePy | Adds dependency for no gain, ffmpeg zoompan does everything |
| Video assembly | ffmpeg | OpenCV VideoWriter | No audio support, no transitions, lower quality encoding |

## Installation

```bash
# Core pipeline
pip install qwen-tts paddleocr ultralytics pillow numpy

# MAGI (from source)
git clone https://github.com/ragavsachdeva/magi.git
pip install -e ./magi

# Ken Burns reference (optional, for transition code patterns)
pip install aubio

# Qwen3-TTS optimization
pip install flash-attn --no-build-isolation

# Already in ViralCutter
# ffmpeg, opencv-python, torch, transformers
```

## VRAM Budget (RTX 3060 6GB)

| Model | VRAM | Notes |
|-------|------|-------|
| MAGI v3 | ~2-3 GB | Inference only, load/unload per batch |
| YOLOv8m bubble detector | ~0.5 GB | Lightweight |
| PaddleOCR | ~0.5 GB | CPU mode available as fallback |
| Qwen3-TTS 0.6B | ~4-5 GB | Load after unloading vision models |

**Strategy:** Sequential model loading. Never run MAGI and Qwen3-TTS simultaneously. Load vision models first (panels, bubbles, OCR), unload, then load TTS for narration.

## Sources

- [MAGI GitHub](https://github.com/ragavsachdeva/magi) - 430 stars
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS) - 9.8k stars
- [manga-ocr GitHub](https://github.com/kha-white/manga-ocr) - 2.6k stars
- [Manga-Panel-Extractor GitHub](https://github.com/adenzu/Manga-Panel-Extractor) - 63 stars
- [ComiQ GitHub](https://github.com/StoneSteel27/ComiQ) - 29 stars
- [kburns-slideshow GitHub](https://github.com/Trekky12/kburns-slideshow) - 71 stars
- [ogkalu bubble detector HuggingFace](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m)
- [PaddleOCR-VL-For-Manga HuggingFace](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga)
- [Qwen3-TTS technical report](https://arxiv.org/html/2601.15621v1)
- [BentoML TTS comparison](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
