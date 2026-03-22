# Research Summary: Manga/Manhwa to Video Pipeline

**Domain:** Comic-to-video generation (manga pages + manhwa webtoons -> narrated short videos)
**Researched:** 2026-03-22
**Overall confidence:** MEDIUM-HIGH

## Executive Summary

Building a manga/manhwa-to-video pipeline is feasible with existing open-source tools, but no single project covers the full pipeline end-to-end. The closest is `pashpashpash/manga-reader` (61 stars), which uses GPT-4 Vision + ElevenLabs -- both paid APIs. Our pipeline can replicate this fully open-source and free.

The most critical component is **MAGI** (430 stars, CVPR 2024) by Ragav Sachdeva, which is the state-of-the-art for manga understanding: it detects panels, text blocks, characters, speech bubble tails, performs OCR, orders panels, clusters characters, and assigns dialogue to speakers -- all in one model. This is the backbone of the pipeline.

For TTS, **Qwen3-TTS** (9.8k stars, Jan 2026, Apache 2.0) is the clear winner: it supports French natively, offers emotion control via natural language, voice cloning from 3s audio, voice design from text description, and runs on 6GB VRAM (0.6B model) or 8GB (1.7B). It is the best free ElevenLabs alternative available today.

For video generation, the pipeline should use **ffmpeg** directly (already in ViralCutter's stack) with Ken Burns zoom/pan effects on extracted panels, combined with `kburns-slideshow` patterns for transitions. MoviePy adds unnecessary complexity when ffmpeg filters handle everything.

Panel detection for **manhwa/webtoon** (vertical scroll) requires a different approach than manga (page-based). Manga uses MAGI or Kumiko for grid-based panels. Manhwa uses simple horizontal gap detection on long vertical strips -- a custom OpenCV solution of ~50 lines.

## Key Findings

**Stack:** MAGI (panel+OCR+speaker) + PaddleOCR (FR/EN text) + Qwen3-TTS (narration) + ffmpeg (video assembly)
**Architecture:** Two-track pipeline: manga track (MAGI full analysis) and manhwa track (vertical strip split + bubble detection)
**Critical pitfall:** MAGI is Japanese-focused. For FR/EN manga text, use PaddleOCR or ComiQ instead of MAGI's built-in OCR. MAGI's panel/character detection works regardless of language.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Phase 1: Panel Extraction Engine** - Build the panel detection for both manga (page-based) and manhwa (vertical scroll)
   - Addresses: Panel segmentation, image extraction
   - Avoids: Trying to build one-size-fits-all detector (manga vs manhwa are different problems)

2. **Phase 2: Text & Dialogue Extraction** - Speech bubble detection + OCR + speaker assignment
   - Addresses: Text extraction, speaker identification, reading order
   - Avoids: Building custom models when MAGI v3 handles speaker assignment

3. **Phase 3: TTS Narration** - Multi-speaker voice synthesis with emotion
   - Addresses: Voice generation, character voice differentiation
   - Avoids: Paid API dependency (ElevenLabs) by using Qwen3-TTS

4. **Phase 4: Video Assembly** - Ken Burns effects + transitions + audio sync
   - Addresses: Final video output, panel animation, audio/video sync
   - Avoids: Over-engineering with MoviePy when ffmpeg zoompan filter suffices

5. **Phase 5: WebUI Integration** - Integrate into ViralCutter's Gradio interface
   - Addresses: User-facing workflow
   - Avoids: Separate app -- reuse existing ViralCutter infrastructure

**Phase ordering rationale:**
- Phase 1 before 2: Need panels before extracting text from them
- Phase 2 before 3: Need text and speaker info before generating voices
- Phase 3 before 4: Need audio before assembling video with sync
- Phase 5 last: Integration after core pipeline works

**Research flags for phases:**
- Phase 1: MAGI v3 needs testing on non-Japanese manga -- may need fallback to Kumiko
- Phase 2: PaddleOCR for FR/EN needs validation on stylized comic fonts
- Phase 3: Qwen3-TTS 0.6B on RTX 3060 6GB needs benchmarking (tight fit with other models)
- Phase 4: Standard patterns, unlikely to need research

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Well-established tools, verified on GitHub |
| Features | HIGH | Clear feature landscape from existing projects |
| Architecture | MEDIUM | MAGI on non-Japanese manga untested by us |
| Pitfalls | HIGH | Well-documented issues in community |
| TTS (Qwen3) | HIGH | 9.8k stars, extensive benchmarks, Apache 2.0 |

## Gaps to Address

- MAGI v3 performance on French/English manga (designed for Japanese)
- Qwen3-TTS 0.6B actual VRAM usage alongside WhisperX on RTX 3060 6GB
- Manhwa vertical scroll splitting accuracy with varying art styles
- Reading order detection quality for non-standard panel layouts
