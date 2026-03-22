# Feature Landscape

**Domain:** Manga/Manhwa to Video Generation Pipeline
**Researched:** 2026-03-22

## Table Stakes

Features users expect. Missing = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Manga panel detection (page-based) | Core function -- extract panels from manga pages | Medium | MAGI v3 or Kumiko/adenzu extractor |
| Manhwa panel detection (vertical scroll) | Manhwa is 50%+ of webtoon market | Low | Horizontal gap detection on vertical strips |
| Speech bubble text extraction (FR/EN) | Need text to generate narration | Medium | PaddleOCR + YOLOv8 bubble detector |
| Reading order detection | Panels must be read in correct sequence | Medium | MAGI v3 handles this; manhwa is top-to-bottom |
| TTS narration generation | Voice is what makes it a video, not a slideshow | Medium | Qwen3-TTS with emotion control |
| Ken Burns zoom/pan on panels | Static images need motion to feel like video | Low | ffmpeg zoompan filter |
| Panel-to-panel transitions | Smooth flow between panels | Low | Fade, slide, or cut transitions via ffmpeg |
| Audio/video synchronization | Narration timing must match panel display | Medium | Panel duration = TTS audio duration |
| 9:16 vertical format output | Target is TikTok/Reels/Shorts | Low | Already in ViralCutter |
| Background music | Sets mood, fills silence | Low | Already in ViralCutter (add_music.py) |

## Differentiators

Features that set product apart. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Multi-speaker voices per character | Each character gets a distinct voice | High | Qwen3-TTS voice design + MAGI speaker clustering |
| Emotion-aware TTS | Voice changes with dialogue emotion (angry, sad, happy) | Medium | Qwen3-TTS natural language emotion prompts |
| Character face extraction for thumbnails | Auto-generate engaging thumbnails | Low | MAGI v3 detects character faces |
| Automatic content type detection | Manga vs manhwa auto-detect from image dimensions | Low | Aspect ratio check: height >> width = manhwa |
| Narrator voice for non-dialogue text | Sound effects, action descriptions get narrator treatment | Medium | MAGI distinguishes text types (dialogue vs SFX vs caption) |
| Smart panel timing | Panels with more text get longer display time | Low | Duration proportional to TTS audio length |
| Voice cloning from reference | Clone a specific voice for a character | Medium | Qwen3-TTS zero-shot cloning from 3s audio |
| Batch processing (full chapter/volume) | Process entire volumes at once | Low | Loop over pages |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Image-to-video AI animation (panels to anime) | Requires massive GPU (24GB+), inconsistent quality, slow | Ken Burns + transitions are sufficient and fast |
| Custom ML model training | Maintenance burden, MAGI/YOLO pretrained models are good enough | Use pretrained models, fine-tune only if proven need |
| Real-time streaming generation | Unnecessary complexity for batch video creation | Generate offline, upload finished video |
| Translation of manga text | Separate concern, already tools for this | Accept pre-translated or original language input |
| Full page rendering (no panel split) | Loses the "reveal" effect that makes manga videos engaging | Always extract and animate individual panels |
| Lip sync animation | Way beyond scope, requires video generation models | Static panels with voice-over is the standard format |

## Feature Dependencies

```
Manga/Manhwa Detection -> Panel Extraction -> Speech Bubble Detection -> OCR Text Extraction
                                                                            |
                                                                            v
                                                                   Speaker Assignment (MAGI)
                                                                            |
                                                                            v
                                                                   TTS Voice Generation (Qwen3-TTS)
                                                                            |
                                                                            v
Panel Extraction --------> Ken Burns Effect -----> Video Assembly <---- Audio Files
                                                        |
                                                        v
                                                   Background Music Mix
                                                        |
                                                        v
                                                   Final Output (9:16)
```

## MVP Recommendation

Prioritize:
1. **Panel extraction** (manga page-based) -- core prerequisite
2. **Speech bubble OCR** (PaddleOCR) -- extract readable text
3. **Single-voice TTS** (Qwen3-TTS 0.6B, one narrator voice) -- minimum viable narration
4. **Ken Burns + ffmpeg assembly** -- create watchable video output

Defer:
- Multi-speaker voices: Requires MAGI speaker clustering + voice bank -- Phase 2 feature
- Manhwa support: Different detection approach -- Phase 2 feature
- Emotion-aware TTS: Needs dialogue emotion analysis -- Phase 3 feature
- Voice cloning: Nice-to-have, not MVP -- Phase 3 feature

## Sources

- [manga-reader by pashpashpash](https://github.com/pashpashpash/manga-reader) - Reference implementation (GPT-4 Vision + ElevenLabs approach)
- [MAGI](https://github.com/ragavsachdeva/magi) - Full manga analysis capabilities
- [Qwen3-TTS blog](https://qwen.ai/blog?id=qwen3tts-0115) - Feature overview
- [kburns-slideshow](https://github.com/Trekky12/kburns-slideshow) - Ken Burns implementation reference
