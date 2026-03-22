# Domain Pitfalls

**Domain:** Manga/Manhwa to Video Generation Pipeline
**Researched:** 2026-03-22

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: VRAM OOM from Simultaneous Model Loading
**What goes wrong:** Loading MAGI (~2-3GB) + Qwen3-TTS 0.6B (~4-5GB) simultaneously on RTX 3060 6GB.
**Why it happens:** Temptation to keep models loaded for speed.
**Consequences:** CUDA OOM crash, corrupted state, restart required.
**Prevention:** Strict sequential pipeline: load vision models -> process all pages -> unload -> load TTS -> generate all audio -> unload -> ffmpeg (CPU).
**Detection:** Monitor VRAM with `torch.cuda.memory_allocated()` before each model load.

### Pitfall 2: MAGI OCR is Japanese-Only
**What goes wrong:** Using MAGI's built-in OCR for French/English manga text, getting garbage output.
**Why it happens:** MAGI was trained on Manga109 (Japanese dataset). Its OCR module is Japanese-optimized.
**Consequences:** All extracted text is incorrect, TTS narrates nonsense.
**Prevention:** Use MAGI only for panel detection + character clustering + speaker assignment. Replace OCR step with PaddleOCR for FR/EN text. Use MAGI's bubble bounding boxes as PaddleOCR input regions.
**Detection:** Validate OCR output against known text patterns (e.g., contains Latin characters).

### Pitfall 3: Wrong Reading Order for Right-to-Left Manga
**What goes wrong:** Panels read left-to-right (Western order) on Japanese-style manga that reads right-to-left.
**Why it happens:** Default assumption of left-to-right reading direction.
**Consequences:** Story makes no sense, dialogue is out of order.
**Prevention:** MAGI v3 handles reading order natively. For non-MAGI pipeline: detect if manga is RTL (Japanese publishers) vs LTR (French/Western). Add a user toggle in WebUI.
**Detection:** First panel should contain establishing shot or chapter title -- validate with user preview.

### Pitfall 4: Manhwa Treated as Single Giant Image
**What goes wrong:** Passing a 800x20000px manhwa strip to MAGI or standard panel detectors.
**Why it happens:** No format detection step, same code path for all inputs.
**Consequences:** Model OOM (image too large), or wrong panel segmentation (treating entire strip as one panel).
**Prevention:** Format detection first (aspect ratio > 3:1 = manhwa). Split manhwa into segments at horizontal white gaps before panel analysis.
**Detection:** Check input image dimensions before processing.

## Moderate Pitfalls

### Pitfall 5: Speech Bubble vs Sound Effect Confusion
**What goes wrong:** Sound effects ("BOOM!", "CRACK!") get narrated as dialogue.
**Why it happens:** OCR extracts all text equally, no bubble type classification.
**Prevention:** Use YOLOv8 bubble detector that classifies bubble types, or MAGI v3 which distinguishes text vs onomatopoeia. Sound effects should either be skipped or handled differently (SFX audio overlay instead of TTS).

### Pitfall 6: Character Voice Inconsistency Across Pages
**What goes wrong:** Same character gets different voices on different pages.
**Why it happens:** Per-page processing without chapter-wide character tracking.
**Prevention:** Use MAGI v3's chapter-wide character clustering to maintain consistent IDs. Build voice bank at chapter level, not page level.

### Pitfall 7: Ken Burns Motion Sickness on Small Panels
**What goes wrong:** Aggressive zoom/pan on small panels causes uncomfortable viewing.
**Why it happens:** Same zoom parameters applied regardless of panel size or content.
**Prevention:** Scale Ken Burns intensity to panel size: large panels get subtle zoom (1.0-1.1x), small panels get no zoom (static display). Limit zoom speed to < 5% per second.

### Pitfall 8: TTS Silence Between Dialogue Lines
**What goes wrong:** Awkward pauses between dialogue lines in the same panel, or no pause between different panels.
**Why it happens:** Default TTS behavior generates audio per line without timing context.
**Prevention:** Add configurable padding: 0.2-0.3s between lines in same panel, 0.5-1.0s between panels. Let panel transition time fill inter-panel gaps.

### Pitfall 9: Qwen3-TTS VRAM With FlashAttention on Windows
**What goes wrong:** `flash-attn` installation fails on Windows or doesn't compile with CUDA.
**Why it happens:** FlashAttention 2 has limited Windows support, requires specific CUDA toolkit version.
**Prevention:** FlashAttention is optional optimization. Without it: 20-25% more VRAM, 30-40% slower. On 6GB GPU, may need to use 0.6B model without FlashAttention. Test both configurations.
**Detection:** Catch import error for flash_attn, fall back gracefully.

## Minor Pitfalls

### Pitfall 10: Double Pages (Spread Panels)
**What goes wrong:** Two-page spreads (single illustration across two pages) get split incorrectly.
**Why it happens:** Each page processed independently.
**Prevention:** Detect spread panels by checking for content continuity at page edges. Merge adjacent pages if needed. Low priority -- spreads are rare in chapter-by-chapter processing.

### Pitfall 11: Overlapping Panels in Artistic Layouts
**What goes wrong:** Non-rectangular or overlapping panels confuse grid-based extractors.
**Why it happens:** Artistic manga layouts break standard grid assumptions.
**Prevention:** MAGI v3 handles non-standard layouts better than OpenCV-based tools. Accept some loss on highly artistic layouts -- they're a minority of pages.

### Pitfall 12: PDF Rendering Quality
**What goes wrong:** Low-resolution panel images from PDF extraction.
**Why it happens:** PDF rendered at low DPI (72 or 96).
**Prevention:** Render PDFs at 300 DPI minimum. Use PyMuPDF (fitz) for quality extraction.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Panel extraction | Pitfall 4 (manhwa as giant image) | Format detection as first step |
| Panel extraction | Pitfall 3 (reading order) | User toggle RTL/LTR + MAGI auto-detect |
| OCR & text | Pitfall 2 (MAGI JP-only OCR) | PaddleOCR for FR/EN, MAGI for structure only |
| OCR & text | Pitfall 5 (SFX vs dialogue) | Bubble type classification |
| TTS generation | Pitfall 1 (VRAM OOM) | Sequential model loading |
| TTS generation | Pitfall 6 (voice inconsistency) | Chapter-wide voice bank |
| TTS generation | Pitfall 9 (FlashAttention Windows) | Graceful fallback |
| Video assembly | Pitfall 7 (Ken Burns intensity) | Scale to panel size |
| Video assembly | Pitfall 8 (silence gaps) | Configurable padding |

## Sources

- [MAGI limitations](https://github.com/ragavsachdeva/magi) - Japanese-focused training data
- [Qwen3-TTS hardware guide](https://deepwiki.com/mu-zi-lee/qwen3-tts-skill/8.2-memory-and-hardware-requirements) - VRAM requirements
- [manga-ocr limitations](https://github.com/kha-white/manga-ocr) - Japanese text only
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) - Windows compatibility notes
- Community discussions on manga panel detection challenges
