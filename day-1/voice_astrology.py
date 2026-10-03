import base64
import io
import os
import sys
import wave
from datetime import date
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
HERE = Path(__file__).parent

# Everything runs on the class Spark: Whisper (speech-to-text), Qwen (Maude) and the voices.
SPARK = os.getenv("SPARK_BASE_URL", "http://192.168.8.246:8000/v1")
client = OpenAI(base_url=SPARK, api_key=(os.getenv("SPARK_API_KEY") or "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"))
chatterbox = OpenAI(base_url=SPARK.replace(":8000", ":8031"), api_key="none")   # no key on this port

# ---- things to change -----------------------------------------------------------
VOICE_CLIP = None        # e.g. HERE.parent / "day-2/data/morgan-freeman-voice-sample.wav" to clone a voice
KOKORO_VOICE = "bf_emma" # used when VOICE_CLIP is None
SYSTEM_PROMPT = ("You are Maude, an AI astrologer. Give a short, positive and optimistic horoscope for "
                 "tomorrow in three spoken sentences: no lists, no markdown, no emojis.")
RATE, SILENCE_RMS, SILENCE_SECS, MAX_SECS = 16000, 0.01, 1.2, 15
# ---------------------------------------------------------------------------------


def speak(text):
    if VOICE_CLIP:
        clip = Path(VOICE_CLIP)
        wav = chatterbox.audio.speech.create(
            model="chatterbox", voice="en", input=text, response_format="wav",
            extra_body={"audio_prompt": base64.b64encode(clip.read_bytes()).decode(),
                        "audio_prompt_format": clip.suffix[1:]}).content
    else:
        wav = client.audio.speech.create(model="kokoro", voice=KOKORO_VOICE, input=text,
                                         response_format="wav").content
    with wave.open(io.BytesIO(wav)) as w:
        sd.play(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate(), blocking=True)


def listen():
    """Record until a pause and return it as a WAV file object."""
    chunks, heard, quiet = [], False, 0
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32") as mic:
        for _ in range(MAX_SECS * 10):
            chunk, _ = mic.read(RATE // 10)
            chunks.append(chunk)
            loud = np.sqrt(np.mean(chunk ** 2)) > SILENCE_RMS
            heard |= loud
            quiet = 0 if loud else quiet + 1
            if heard and quiet >= SILENCE_SECS * 10:
                break
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.concatenate(chunks) * 32767).astype(np.int16).tobytes())
    buf.name = "you.wav"
    buf.seek(0)
    return buf


if __name__ == "__main__":
    speak("Hello, I'm Maude. Tell me your name and your star sign.")
    # python voice_astrology.py some.wav  uses a recording instead of the microphone
    audio = open(sys.argv[1], "rb") if len(sys.argv) > 1 else listen()
    heard = client.audio.transcriptions.create(model="whisper-large-v3", file=audio, language="en").text
    print("You:", heard)

    horoscope = client.chat.completions.create(
        model="qwen3.6-35b",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"{heard} Today's date is {date.today():%A, %d %B %Y}."}],
        temperature=0.8, max_tokens=200,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ).choices[0].message.content
    print("Maude:", horoscope)
    speak(horoscope)
