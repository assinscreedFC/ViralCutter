# ViralCutter

**100% Free, Local, and Unlimited Open-Source Alternative to Opus Clip**

Turn long YouTube videos into viral shorts optimized for TikTok, Instagram Reels, and YouTube Shorts – with state-of-the-art AI, dynamic captions, precise face tracking, and automatic translation. All running on your machine.

[![Stars](https://img.shields.io/github/stars/assinscreedFC/ViralCutter?style=social)](https://github.com/assinscreedFC/ViralCutter/stargazers)
[![Forks](https://img.shields.io/github/forks/assinscreedFC/ViralCutter/network/members)](https://github.com/assinscreedFC/ViralCutter/network/members)

[English](README_en.md) • [Português](README.md)

## Why ViralCutter is a Game Changer

Forget expensive subscriptions and minute limits. ViralCutter offers unlimited power on your own hardware.

| Feature | ViralCutter (Open-Source) | Opus Clip / Klap / Munch (SaaS) |
| :--- | :--- | :--- |
| **Price** | **Free & Unlimited** | $20–$100/mo + minute limits |
| **Privacy** | **100% Local** (Your data never leaves your PC) | Upload to third-party cloud |
| **AI & LLM** | **Flexible**: Gemini (Free), GPT-4, **Local GGUF (Offline)** | Only what they offer |
| **Face Tracking** | **Split Screen (2 faces)**, Active Speaker, Face Snap, Auto | Basic or extra cost |
| **Translation** | **Yes** (Translate captions to 10+ languages) | Limited features |
| **Editing** | **Export XML to Premiere Pro** (Beta), A/B variants | Limited web editor |
| **Watermark** | **ZERO** | Yes (on free plans) |

**Professional results, total privacy, and zero cost.**

## Key Features

- **AI Viral Cut**: Automatically identifies hooks and engaging moments using Gemini, GPT-4, or Local LLMs (Llama 3, DeepSeek, etc)
- **Ultra-Precise Transcription**: Powered by WhisperX with GPU acceleration for perfect subtitles
- **Dynamic Captions**: "Hormozi" style with word-by-word highlights, vibrant colors, emojis, and full customization
- **Advanced Face Tracking**:
  - **Auto-Crop 9:16**: Transforms horizontal to vertical while keeping the focus
  - **Smart Split Screen**: Detects 2 people talking and automatically splits the screen
  - **Face Snap**: Intelligently positions camera to key speakers
  - **Active Speaker (Experimental)**: The camera cuts to whoever is speaking
- **A/B Variants**: Generate multiple cut variations automatically
- **Video Translation**: Automatically generate translated subtitles (e.g., English Video -> Portuguese Subtitles)
- **Quality & Control**: Choose resolution (up to 4K/Best), format output, and save processing configurations
- **Performance**: Transcription with "slicing" (process 1x, cut N times), single-pass post-production, and ultra-fast installation via `uv`
- **Modern Interface**: Gradio 6 WebUI with Dark Mode, Project Gallery, and integrated Subtitle Editor
- **Advanced Audio**: Scene detection, pacing analysis, silence removal, background music integration
- **Output Flexibility**: Export XML for Premiere Pro (Beta), organize outputs, split parts for social media

## Web Interface (Inspired by Opus Clip)

![WebUI Home](https://github.com/user-attachments/assets/ba147149-fc5f-48fc-a03c-fc86b5dc0568)
*Intuitive control panel with AI backend selection and rendering controls*

![WebUI Library](https://github.com/user-attachments/assets/b0204e4b-0e5d-4ee4-b7b4-cac044b76c24)
*Library: OpusClip-style gallery with project management*

## Architecture

ViralCutter uses a modular, domain-driven pipeline:

```
scripts/
├── core/              # Configuration, models, FFmpeg utilities
├── download/          # YouTube video download, model management
├── transcription/     # WhisperX transcription, subtitle translation
├── analysis/          # AI scene detection, engagement scoring, segment creation
├── vision/            # Face detection (InsightFace, MediaPipe), layout templates
├── audio/             # Music, silence removal, active speaker detection
├── editing/           # Video composition, subtitle burning, color grading
├── postprod/          # A/B variants, speed ramps, effects, distraction videos
├── export/            # Premiere Pro XML export, social media optimization
└── quality/           # Validation, smart trimming, filler removal

pipeline/             # Main processing orchestration
webui/               # Gradio 6 web interface
```

Each module is independently testable with comprehensive test suite (359+ tests).

## Local Installation (Super Fast)

### Prerequisites (From Scratch Setup)

To run ViralCutter on a fresh computer, install these core tools:

1. **Visual Studio C++ Build Tools**
   Required to compile `insightface` and avoid compilation errors.
   - Download [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Run installer and check **"Desktop development with C++"**
   - Ensure Windows 10/11 SDK and MSVC v143 are selected, then install. Restart PC if needed.

2. **Python (3.10.x or 3.11.x recommended)**
   - Download from [python.org/downloads](https://www.python.org/downloads/)
   - **IMPORTANT:** Mark **"Add Python to PATH"** on the first screen before installing

3. **FFmpeg** (Audio/Video Processing)
   - On Windows, open PowerShell as Administrator and run:
     ```
     winget install ffmpeg
     ```
   - Restart terminal and verify: `ffmpeg -version`

4. **NVIDIA GPU Drivers** (Recommended)
   - Keep drivers updated (GeForce Experience or Nvidia website) to support CUDA 12.4+
   - **NVIDIA GPU is strongly recommended** for speed and local AI operations

### Step-by-Step Installation

1. **Install Dependencies**
   Open the ViralCutter folder and double-click one of these installers:
   - `install_dependencies.bat`: Standard installation (Recommended). Uses cloud AI like Gemini (Free) and GPT-4
   - `install_dependencies_advanced_LocalLLM.bat`: Advanced installation for full offline AI on hardware (Llama 3, etc). Requires good GPU and C++ Build Tools

   Both use the `uv` package manager for automatic setup.

2. **Configure AI (Optional)**
   - **Gemini (Recommended/Free)**: Add your API key in `config/api_config.json`
   - **Local (GGUF)**: Download `.gguf` models and place in `models/` folder. ViralCutter detects them automatically

3. **Run**
   - Double-click `run_webui.bat` to open the web interface
   - Or use CLI: `python main_improved.py --help`

### CLI Usage Examples

```bash
# Process YouTube video
python main_improved.py "https://www.youtube.com/watch?v=..."

# With custom configuration
python main_improved.py "https://www.youtube.com/watch?v=..." \
  --ai-backend gemini \
  --chunk-size 15 \
  --min-duration 8 \
  --max-duration 45

# List available options
python main_improved.py --help
```

## Tech Stack

- **Video Processing**: FFmpeg, OpenCV, MediaPipe, InsightFace
- **Transcription**: WhisperX (with GPU acceleration)
- **AI Models**:
  - Cloud: Gemini, GPT-4, g4f
  - Local: Llama 3, DeepSeek, other GGUF models
- **Web UI**: Gradio 6 (dark theme, responsive design)
- **Backend**: FastAPI, Uvicorn
- **Audio**: librosa, music libraries
- **CLI**: Click, Rich formatting
- **Testing**: pytest, pytest-asyncio (359+ tests)
- **Quality**: xgboost for engagement prediction, scene detection

## Output Examples

**Viral Clip with Highlight Captions**
<video src="https://github.com/user-attachments/assets/7a32edce-fa29-4693-985f-2b12313362f3" controls></video>

**Direct Comparison: Opus Clip vs ViralCutter** (same input video)
<video src="https://github.com/user-attachments/assets/12916792-dc0e-4f63-a76b-5698946f50f4" controls></video>

**2-Face Split Screen Mode**
<video src="https://github.com/user-attachments/assets/f5ce5168-04a2-4c9b-9408-949a5400d020" controls></video>

## Configuration

### api_config.json

Located in `config/api_config.json`:

```json
{
  "AI_MODEL_BACKEND": "gemini",
  "GEMINI_API_KEY": "your-key-here",
  "GPT4_API_KEY": "your-key-here",
  "PLEIADE_API_KEY": "your-key-here",
  "PLEIADE_API_URL": "https://pleiade.example.com",
  "YOUTUBE_CLIENT_ID": "your-client-id",
  "YOUTUBE_CLIENT_SECRET": "your-client-secret"
}
```

### WebUI Settings

All processing parameters can be configured via the web interface:
- AI model selection and chunk size
- Video quality selection (best, 1080p, 720p, 480p)
- Face detection interval and confidence
- Caption style and animation
- Output format and resolution

## Roadmap

- [x] Release code
- [x] Two-face split screen mode
- [x] Custom captions with word-by-word highlights
- [x] 100% Local AI models (Llama, DeepSeek, GGUF)
- [x] Automatic caption translation
- [x] Dynamic face tracking
- [x] XML Export to Premiere Pro (Beta)
- [x] A/B variants generation
- [x] Face snap (intelligent camera positioning)
- [x] Subtitle animations
- [ ] Permanent Demo on Hugging Face Spaces
- [ ] Automatic background music (Auto-Duck)
- [ ] Direct upload to TikTok/YouTube/Instagram
- [ ] More framing formats (beyond 9:16)
- [ ] Optional watermark

## Contributing

ViralCutter is community-maintained. Join us to democratize AI content creation!

- **Star the project** if it helps you
- **Report bugs** via GitHub Issues
- **Submit PRs** for improvements
- **Share feedback** on our discussions

## Development

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=scripts --cov-report=html

# Run specific test
pytest tests/test_core.py -v
```

### Project Structure

See [Architecture](#architecture) section above for module organization. Each module:
- Has its own `__init__.py`
- Is independently testable
- Follows single responsibility principle
- Has comprehensive type hints

### Current Version

Version 1.0.0 (updated from 0.8v Alpha) - Production-ready with major refactoring (March 2026)

Recent improvements:
- Modular domain-driven architecture
- Typed ProcessingConfig dataclass
- Gradio 6 compatibility
- Single-pass post-production
- Face snap intelligence
- A/B variants generation
- Subtitle animations
- FFmpeg progress bar compatibility

---

**ViralCutter: Because viral clips shouldn't cost a fortune.**
