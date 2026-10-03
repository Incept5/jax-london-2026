"""Chatterbox-Turbo text-to-speech with voice cloning, on Windows, Linux or Mac (PyTorch).

chatterbox_tts.py is the same demo for Apple Silicon (MLX).
Needs: pip install chatterbox-tts
The weights are 3.8 GB: copy 6-voice-windows-linux from the course USB drive first
(tar xf everything.tar -C ~), otherwise they download from Hugging Face on the first run.
"""
from pathlib import Path

import soundfile
import torch
from huggingface_hub import snapshot_download
from chatterbox.tts_turbo import ChatterboxTurboTTS

HERE = Path(__file__).parent
REPO = "ResembleAI/chatterbox-turbo"
VOICE = HERE / "data" / "morgan-freeman-voice-sample.wav"   # the voice to clone: try your own 10 s recording
TEXT = ("In the autumn of 2026, a quiet gathering took place in London. They called it JAX London. "
        "And on that Tuesday morning, as the light came through the windows, everyone in the room "
        "understood that something remarkable was about to unfold.")

device = "cuda" if torch.cuda.is_available() else "cpu"   # not "mps": Chatterbox sends float64 audio, which MPS rejects

try:   # use the copy from the USB drive without touching the (slow) internet
    weights = snapshot_download(REPO, local_files_only=True)
except Exception:
    print("Weights not found locally, downloading 3.8 GB from Hugging Face...")
    weights = snapshot_download(REPO, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"])

model = ChatterboxTurboTTS.from_local(weights, device)
# norm_loudness=False works around a chatterbox-tts 0.1.7 bug: with NumPy 2 it turns the voice sample into float64
wav = model.generate(TEXT, audio_prompt_path=str(VOICE), norm_loudness=False)

out = HERE / "output" / "turbo_demo.wav"
out.parent.mkdir(exist_ok=True)
soundfile.write(out, wav.squeeze(0).cpu().numpy(), model.sr)   # torchaudio.save now needs torchcodec
print(f"Saved {out} (ran on {device})")
