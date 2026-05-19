# UGC Video Ad Generator

Automated, lip-synced, cinematic UGC ad videos from a face photo + product image + script.

## Features
- 🎙️ ElevenLabs or Microsoft Edge TTS voice
- 👄 LivePortrait lip-sync + head movement
- 🎨 6 color grade presets (warm, cool, cinematic, matte, vibrant, neutral)
- 📽️ Film grain + vignette (phone-shot realism)
- 📳 Handheld camera shake simulation
- 💬 Auto-synced burned-in captions (TikTok / subtitle / minimal)
- 🎵 Background music mixer
- 📐 9:16 / 1:1 / 16:9 / 4:5 aspect ratios
- ⬆️ Real-ESRGAN upscaling (optional)

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/ugc-generator.git
cd ugc-generator
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/KwaiVGI/LivePortrait.git
cd LivePortrait && pip install -r requirements.txt && cd ..
cp .env.example .env   # edit .env with your paths
python main.py
```

Open http://localhost:7860

## Google Colab

```python
!git clone https://github.com/YOUR_USERNAME/ugc-generator.git
%cd ugc-generator
!pip install -r requirements.txt
!pip install torch torchvision torchaudio
!git clone https://github.com/KwaiVGI/LivePortrait.git
%cd LivePortrait && !pip install -r requirements.txt
%cd /content/ugc-generator

import os
os.environ["LIVE_PORTRAIT_DIR"] = "/content/ugc-generator/LivePortrait"

# Edit main.py last line: share=True
!python main.py
```
