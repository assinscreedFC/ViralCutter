# ViralCutter

**Alternativa open-source 100% gratuita, local e ilimitada ao Opus Clip**

Transforme vídeos longos do YouTube em shorts virais otimizados para TikTok, Instagram Reels e YouTube Shorts – com IA de ponta, legendas dinâmicas, rastreamento facial preciso e tradução automática. Tudo rodando na sua máquina.

[![Stars](https://img.shields.io/github/stars/assinscreedFC/ViralCutter?style=social)](https://github.com/assinscreedFC/ViralCutter/stargazers)
[![Forks](https://img.shields.io/github/forks/assinscreedFC/ViralCutter/network/members)](https://github.com/assinscreedFC/ViralCutter/network/members)

[English](README_en.md) • [Português](README.md)

## Por que ViralCutter é um "Game Changer"?

Esqueça assinaturas caras e limites de minutos. O ViralCutter oferece poder ilimitado no seu hardware.

| Feature | ViralCutter (Open-Source) | Opus Clip / Klap / Munch (SaaS) |
| :--- | :--- | :--- |
| **Preço** | **Gratuito e Ilimitado** | $20–$100/mês + limites de min. |
| **Privacidade** | **100% Local** (Seus dados não saem do PC) | Upload para nuvem de terceiros |
| **IA & LLM** | **Flexível**: Gemini (Free), GPT-4, **Local GGUF (Offline)** | Apenas o que eles oferecem |
| **Rastreamento Facial** | **Split Screen (2 faces)**, Active Speaker, Face Snap, Auto | Básico ou pago extra |
| **Tradução** | **Sim** (Traduza legendas p/ 10+ línguas) | Recursos limitados |
| **Edição** | **Exporta XML para Premiere Pro** (Beta), Variantes A/B | Editor web limitado |
| **Watermark** | **ZERO** | Sim (nos planos free) |

**Resultados profissionais, privacidade total e custo zero.**

## Funcionalidades Principais

- **Corte Viral com IA**: Identifica automaticamente ganchos e momentos engajadores usando Gemini, GPT-4 ou LLMs Locais (Llama 3, DeepSeek, etc)
- **Transcrição Ultra-Precisa**: Baseado em WhisperX com aceleração via GPU para legendas perfeitas
- **Legendas Dinâmicas**: Estilo "Hormozi" com highlight palavra por palavra, cores vibrantes, emojis e total customização
- **Rastreamento Facial Avançado**:
  - **Auto-Crop 9:16**: Transforma horizontal em vertical mantendo o foco
  - **Split Screen Inteligente**: Detecta 2 pessoas conversando e divide a tela automaticamente
  - **Face Snap**: Posiciona inteligentemente a câmera para falantes-chave
  - **Active Speaker (Experimental)**: A câmera corta para quem está falando
- **Variantes A/B**: Gera múltiplas variações de corte automaticamente
- **Tradução de Vídeo**: Gera legendas traduzidas automaticamente (ex: Vídeo em Inglês -> Legenda em Português)
- **Qualidade & Controle**: Escolha resolução (até 4K/Best), formate saída e salve configurações de processamento
- **Performance**: Transcrição com "slicing" (processa 1x, corta N vezes), pós-produção single-pass e instalação ultra-rápida via `uv`
- **Interface Moderna**: WebUI Gradio 6 com Modo Escuro, Galeria de Projetos e Editor de Legendas integrado
- **Áudio Avançado**: Detecção de cenas, análise de ritmo, remoção de silêncios, integração de música de fundo
- **Flexibilidade de Saída**: Exporta XML para Premiere Pro (Beta), organiza outputs, divide partes para redes sociais

## Interface Web (Inspirada no Opus Clip)

![WebUI Home](https://github.com/user-attachments/assets/ba147149-fc5f-48fc-a03c-fc86b5dc0568)
*Painel de controle intuitivo com seleção de backend de IA e controles de renderização*

![WebUI Library](https://github.com/user-attachments/assets/b0204e4b-0e5d-4ee4-b7b4-cac044b76c24)
*Biblioteca: Galeria estilo OpusClip com gerenciamento de projetos*

## Arquitetura

ViralCutter usa um pipeline modular, orientado a domínios:

```
scripts/
├── core/              # Configuração, modelos, utilitários FFmpeg
├── download/          # Download YouTube, gerenciamento de modelos
├── transcription/     # Transcrição WhisperX, tradução de legendas
├── analysis/          # Detecção IA de cenas, pontuação, criação de segmentos
├── vision/            # Detecção facial (InsightFace, MediaPipe), templates de layout
├── audio/             # Música, remoção de silêncios, detecção de falante ativo
├── editing/           # Composição de vídeo, queimadura de legendas, color grading
├── postprod/          # Variantes A/B, speed ramps, efeitos, vídeos de distração
├── export/            # Exporta XML para Premiere Pro, otimização redes sociais
└── quality/           # Validação, trimming inteligente, remoção de preenchimento

pipeline/             # Orquestração principal
webui/               # Interface web Gradio 6
```

Cada módulo é independentemente testável com suite de testes abrangente (359+ testes).

## Instalação Local (Super Rápida)

### Pré-requisitos (Instalação "do Zero")

Para rodar ViralCutter em um PC novo, instale estas ferramentas essenciais:

1. **Ferramentas de Build do Visual Studio (C++)**
   Necessário para compilar `insightface` e evitar erros de compilação.
   - Baixe [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Execute o instalador e marque **"Desenvolvimento para Desktop com C++"**
   - Certifique-se que Windows 10/11 SDK e MSVC v143 estão selecionados, depois instale. Reinicie o PC se necessário.

2. **Python (3.10.x ou 3.11.x recomendados)**
   - Baixe em [python.org/downloads](https://www.python.org/downloads/)
   - **IMPORTANTE:** Marque **"Add Python to PATH"** na primeira tela antes de instalar

3. **FFmpeg** (Processamento de Áudio/Vídeo)
   - No Windows, abra PowerShell como Administrador e execute:
     ```
     winget install ffmpeg
     ```
   - Reinicie o terminal e verifique: `ffmpeg -version`

4. **Drivers NVIDIA GPU** (Recomendado)
   - Mantenha drivers atualizados (GeForce Experience ou site Nvidia) para suportar CUDA 12.4+
   - **GPU NVIDIA é fortemente recomendada** para velocidade e operações IA locais

### Passo a Passo da Instalação

1. **Instale as Dependências**
   Abra a pasta ViralCutter e clique duas vezes em um destes instaladores:
   - `install_dependencies.bat`: Instalação padrão (Recomendada). Usa IAs em nuvem como Gemini (Free) e GPT-4
   - `install_dependencies_advanced_LocalLLM.bat`: Instalação avançada para IA 100% offline (Llama 3, etc). Requer GPU boa e C++ Build Tools

   Ambos usam o gerenciador `uv` para setup automático.

2. **Configure IA (Opcional)**
   - **Gemini (Recomendado/Free)**: Adicione sua chave em `config/api_config.json`
   - **Local (GGUF)**: Baixe modelos `.gguf` e coloque em `models/`. ViralCutter detecta automaticamente

3. **Execute**
   - Clique duas vezes em `run_webui.bat` para abrir a interface web
   - Ou use CLI: `python main_improved.py --help`

### Exemplos de Uso via CLI

```bash
# Processa vídeo YouTube
python main_improved.py "https://www.youtube.com/watch?v=..."

# Com configuração customizada
python main_improved.py "https://www.youtube.com/watch?v=..." \
  --ai-backend gemini \
  --chunk-size 15 \
  --min-duration 8 \
  --max-duration 45

# Listar opções disponíveis
python main_improved.py --help
```

## Tech Stack

- **Processamento de Vídeo**: FFmpeg, OpenCV, MediaPipe, InsightFace
- **Transcrição**: WhisperX (com aceleração GPU)
- **Modelos IA**:
  - Nuvem: Gemini, GPT-4, g4f
  - Local: Llama 3, DeepSeek, outros modelos GGUF
- **WebUI**: Gradio 6 (tema escuro, design responsivo)
- **Backend**: FastAPI, Uvicorn
- **Áudio**: librosa, bibliotecas de música
- **CLI**: Click, formatação Rich
- **Testes**: pytest, pytest-asyncio (359+ testes)
- **Qualidade**: xgboost para predição de engajamento, detecção de cenas

## Exemplos de Saída

**Clip viral com legendas highlight**
<video src="https://github.com/user-attachments/assets/7a32edce-fa29-4693-985f-2b12313362f3" controls></video>

**Comparação direta: Opus Clip vs ViralCutter** (mesmo vídeo de entrada)
<video src="https://github.com/user-attachments/assets/12916792-dc0e-4f63-a76b-5698946f50f4" controls></video>

**Modo Split Screen (2 faces)**
<video src="https://github.com/user-attachments/assets/f5ce5168-04a2-4c9b-9408-949a5400d020" controls></video>

## Configuração

### api_config.json

Localizado em `config/api_config.json`:

```json
{
  "AI_MODEL_BACKEND": "gemini",
  "GEMINI_API_KEY": "sua-chave-aqui",
  "GPT4_API_KEY": "sua-chave-aqui",
  "PLEIADE_API_KEY": "sua-chave-aqui",
  "PLEIADE_API_URL": "https://pleiade.example.com",
  "YOUTUBE_CLIENT_ID": "seu-client-id",
  "YOUTUBE_CLIENT_SECRET": "seu-client-secret"
}
```

### Configurações WebUI

Todos os parâmetros de processamento podem ser configurados via interface web:
- Seleção de modelo IA e tamanho de chunk
- Seleção de qualidade de vídeo (best, 1080p, 720p, 480p)
- Intervalo e confiança de detecção facial
- Estilo de legenda e animação
- Formato e resolução de saída

## Roadmap

- [x] Lançamento do código
- [x] Modo split screen com 2 faces
- [x] Legendas customizadas com highlight palavra por palavra
- [x] Modelos IA 100% locais (Llama, DeepSeek, GGUF)
- [x] Tradução automática de legendas
- [x] Rastreamento facial dinâmico
- [x] Exportação XML para Premiere Pro (Beta)
- [x] Geração de variantes A/B
- [x] Face snap (posicionamento inteligente de câmera)
- [x] Animações de legenda
- [ ] Demo permanente no Hugging Face Spaces
- [ ] Música de fundo automática (Auto-Duck)
- [ ] Upload direto para TikTok/YouTube/Instagram
- [ ] Mais formatos de enquadramento (além de 9:16)
- [ ] Watermark opcional

## Contribuir

ViralCutter é mantido pela comunidade. Junte-se a nós para democratizar criação de conteúdo com IA!

- **Dê uma estrela** ao projeto se ajudar você
- **Reporte bugs** via GitHub Issues
- **Envie PRs** para melhorias
- **Compartilhe feedback** em discussões

## Desenvolvimento

### Executar Testes

```bash
# Rodar todos os testes
pytest

# Com cobertura
pytest --cov=scripts --cov-report=html

# Rodar teste específico
pytest tests/test_core.py -v
```

### Estrutura do Projeto

Veja seção [Arquitetura](#arquitetura) acima para organização de módulos. Cada módulo:
- Tem seu próprio `__init__.py`
- É independentemente testável
- Segue princípio de responsabilidade única
- Tem type hints abrangentes

### Versão Atual

Versão 1.0.0 (atualizado de 0.8v Alpha) - Pronto para produção com refatoração maior (Março 2026)

Melhorias recentes:
- Arquitetura modular orientada a domínios
- Dataclass ProcessingConfig tipado
- Compatibilidade Gradio 6
- Pós-produção single-pass
- Inteligência Face snap
- Geração de variantes A/B
- Animações de legenda
- Compatibilidade com progress bar FFmpeg

---

**ViralCutter: Porque clips virais não precisam custar uma fortuna.**
