# Architecture Patterns

**Domain:** Manga/Manhwa to Video Generation Pipeline
**Researched:** 2026-03-22

## Recommended Architecture

### High-Level Pipeline

```
Input (PDF/Images)
    |
    v
[Format Detector] -- aspect ratio check --> manga track OR manhwa track
    |
    v
[Panel Extractor] -- MAGI v3 (manga) or vertical-split (manhwa)
    |
    v
[Bubble Detector + OCR] -- YOLOv8m + PaddleOCR (FR/EN) or MAGI built-in (JP)
    |
    v
[Dialogue Parser] -- ordered text with speaker IDs (MAGI) or sequential (manhwa)
    |
    v
[TTS Engine] -- Qwen3-TTS: one voice per character, emotion from context
    |
    v
[Video Assembler] -- ffmpeg: Ken Burns on panels + transitions + audio sync
    |
    v
[Post-Processing] -- background music, 9:16 crop, subtitles (existing ViralCutter)
    |
    v
Output (MP4 short video)
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| FormatDetector | Detect manga vs manhwa from image dimensions | PanelExtractor |
| PanelExtractor | Extract individual panels from pages | BubbleDetector, VideoAssembler |
| BubbleDetector | Find speech bubbles, classify type (dialogue/SFX/caption) | OCREngine |
| OCREngine | Extract text from detected bubbles | DialogueParser |
| DialogueParser | Order text, assign speakers, determine emotion hints | TTSEngine |
| TTSEngine | Generate audio per dialogue line with appropriate voice | VideoAssembler |
| VideoAssembler | Combine panels + audio + effects into video | PostProcessor |
| PostProcessor | Add music, crop 9:16, burn subtitles | Output |

### Data Flow

```python
# Core data structures

@dataclass
class Panel:
    image: np.ndarray        # Cropped panel image
    bbox: tuple[int,int,int,int]  # x1,y1,x2,y2 on source page
    order: int               # Reading order index
    page_index: int          # Source page number

@dataclass
class Dialogue:
    text: str                # Extracted text content
    speaker_id: int          # Character cluster ID (-1 for narrator)
    bubble_type: str         # "dialogue" | "thought" | "narration" | "sfx"
    panel_index: int         # Which panel this belongs to
    emotion_hint: str        # "neutral" | "angry" | "sad" | "excited" etc.

@dataclass
class AudioSegment:
    audio_path: str          # Generated TTS audio file
    duration: float          # Duration in seconds
    dialogue: Dialogue       # Source dialogue

@dataclass
class VideoScene:
    panel: Panel             # Panel image to display
    audio_segments: list[AudioSegment]  # Dialogue audio for this panel
    display_duration: float  # Total display time (max of audio + padding)
    effect: str              # "zoom_in" | "zoom_out" | "pan_left" | "pan_right"
    transition: str          # "fade" | "slide_left" | "cut"
```

## Patterns to Follow

### Pattern 1: Sequential Model Loading (VRAM Management)

**What:** Load and unload ML models sequentially, never simultaneously.
**When:** RTX 3060 6GB cannot hold MAGI + Qwen3-TTS together.
**Example:**
```python
# Phase 1: Vision models
magi_model = load_magi()
panels, dialogues = magi_model.analyze(pages)
del magi_model
torch.cuda.empty_cache()

# Phase 2: TTS model
tts_model = load_qwen3_tts()
audio_segments = tts_model.generate_all(dialogues)
del tts_model
torch.cuda.empty_cache()

# Phase 3: ffmpeg (no GPU model needed)
assemble_video(panels, audio_segments)
```

### Pattern 2: Two-Track Format Detection

**What:** Separate processing paths for manga (page-based) vs manhwa (vertical scroll).
**When:** Always -- they require fundamentally different panel detection.
**Example:**
```python
def detect_format(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    ratio = h / w
    if ratio > 3.0:
        return "manhwa"  # Vertical scroll (typical: 800x20000+)
    return "manga"       # Page-based (typical: ~1:1.4 ratio)

def extract_panels(image, format_type):
    if format_type == "manhwa":
        return split_vertical_strip(image)  # Horizontal gap detection
    else:
        return magi_extract_panels(image)   # MAGI v3 full analysis
```

### Pattern 3: Voice Bank per Chapter

**What:** Create a voice profile for each detected character at chapter start, reuse throughout.
**When:** Multi-speaker narration of manga chapters.
**Example:**
```python
def build_voice_bank(character_clusters: list[int]) -> dict[int, str]:
    """Assign a voice description to each character cluster ID."""
    voice_templates = [
        "A young energetic male voice, slightly high-pitched",
        "A deep mature female voice, calm and authoritative",
        "A cheerful teenage girl voice, fast-paced",
        # ... more templates
    ]
    bank = {}
    for i, cluster_id in enumerate(character_clusters):
        bank[cluster_id] = voice_templates[i % len(voice_templates)]
    bank[-1] = "A neutral narrator voice, clear and measured"  # Narrator
    return bank
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Loading All Models at Once
**What:** Loading MAGI, YOLO, PaddleOCR, and Qwen3-TTS simultaneously.
**Why bad:** Exceeds 6GB VRAM, causes OOM crash.
**Instead:** Sequential loading with `torch.cuda.empty_cache()` between phases.

### Anti-Pattern 2: Processing Pages Independently
**What:** Analyzing each manga page in isolation without chapter context.
**Why bad:** Loses character identity across pages. Character A on page 1 is unknown on page 5.
**Instead:** Use MAGI v3 chapter-wide processing with character bank.

### Anti-Pattern 3: Fixed Panel Display Duration
**What:** Each panel shown for exactly 3 seconds regardless of content.
**Why bad:** Panels with lots of dialogue feel rushed; panels with no text feel slow.
**Instead:** Duration = max(TTS_audio_duration + 0.5s padding, 2.0s minimum).

### Anti-Pattern 4: One-Size-Fits-All Panel Detection
**What:** Using the same algorithm for manga pages and manhwa strips.
**Why bad:** MAGI expects page-sized grid layouts; manhwa is one continuous vertical image.
**Instead:** Format detection first, then route to appropriate extractor.

## Scalability Considerations

| Concern | Single Chapter | Full Volume (10+ chapters) | Batch Processing |
|---------|---------------|---------------------------|-----------------|
| VRAM | Sequential loading | Same -- one chapter at a time | Queue system |
| Storage | ~500MB temp | ~5GB temp per volume | Clean between volumes |
| Processing time | ~5-10 min per chapter | ~1-2 hours per volume | Parallelizable across CPU steps |
| TTS generation | 30-60s per chapter | Main bottleneck | Can batch dialogue lines |

## Integration with ViralCutter

The pipeline integrates as a new "content type" in ViralCutter's existing architecture:

```
main_improved.py
  --content-type manga    # NEW: triggers manga pipeline
  --content-type manhwa   # NEW: triggers manhwa pipeline

webui/app.py
  # New tab: "Manga/Manhwa to Video"
  # Input: PDF upload or image folder
  # Output: Short video clips (same VIRALS/ output as existing)
```

Reuse existing ViralCutter modules:
- `scripts/add_music.py` -- background music
- `scripts/burn_subtitles.py` -- subtitle overlay
- `utils/gpu.py` -- GPU detection
- `webui/` -- Gradio interface patterns

## Sources

- [MAGI readme](https://github.com/ragavsachdeva/magi/blob/main/readme.md) - Chapter-wide processing details
- [Qwen3-TTS usage](https://github.com/QwenLM/Qwen3-TTS) - Voice design API
- [ffmpeg zoompan](https://www.bannerbear.com/blog/how-to-do-a-ken-burns-style-effect-with-ffmpeg/) - Ken Burns implementation
- [kburns-slideshow](https://github.com/Trekky12/kburns-slideshow) - Transition patterns
