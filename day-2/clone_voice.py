"""Clone a voice: Chatterbox on the class Spark speaks any text in the voice of a short sample.

Works the same on Windows, Mac and Linux: the model runs on the Spark, so nothing to download.
Plays the result out loud and saves it to day-2/output/.
Needs: pip install openai sounddevice numpy
"""
import base64
import io
import os
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from openai import OpenAI

HERE = Path(__file__).parent
CHATTERBOX_URL = os.getenv("CHATTERBOX_URL", "http://192.168.8.246:8031/v1")   # no API key on this port

VOICE = HERE / "data" / "roosevelt-voice-sample.wav"   # 5-15 s of one speaker; try morgan-freeman-voice-sample.wav, or your own
TEXT = "Good morning, London. Today we build applications with open models that run in this very room."
LANGUAGE = "en"          # the language of TEXT: en, de, fr, es, ja ... (23 languages); the sample can be in another
EXAGGERATION = 0.5       # emotion: 0.25 flat, 0.5 natural, above 0.7 theatrical
CFG_WEIGHT = 0.5         # how tightly it follows the text: lower = more natural pacing
TEMPERATURE = 0.8        # randomness: every run gives a slightly different reading

client = OpenAI(base_url=CHATTERBOX_URL, api_key="none")


def clone_voice(text, sample):
    """Send the text and the voice sample, get back WAV bytes."""
    response = client.audio.speech.create(
        model="chatterbox",
        voice=LANGUAGE,                     # a language code, not a speaker name
        input=text,
        response_format="wav",
        extra_body={
            "audio_prompt": base64.b64encode(sample.read_bytes()).decode(),   # the voice to imitate
            "audio_prompt_format": sample.suffix[1:],                        # wav, mp3, m4a or ogg
            "exaggeration": EXAGGERATION,
            "cfg_weight": CFG_WEIGHT,
            "temperature": TEMPERATURE,
        },
    )
    return response.content


def play(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
        sd.play(audio, w.getframerate())
        sd.wait()


if __name__ == "__main__":
    print(f"Cloning the voice in {VOICE.name} ...")
    wav_bytes = clone_voice(TEXT, VOICE)

    out = HERE / "output" / f"cloned_{VOICE.stem}.wav"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(wav_bytes)
    print(f"Saved {out}")

    print("Playing ...")
    play(wav_bytes)
