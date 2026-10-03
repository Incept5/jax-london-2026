# Spark 5 — the class inference server

A single NVIDIA DGX Spark (GB10, 128 GB unified memory) travels with the course and serves
**Qwen3.6-35B-A3B** (a 35B mixture-of-experts model with ~3B active parameters, NVFP4 weights)
through an **OpenAI-compatible API**. Anything that talks to OpenAI talks to this box: the
`openai` Python SDK, `curl`, LangChain, Open WebUI, Cursor, Claude Code via a proxy, and so on.
It also serves speech-to-text (Whisper) and text-to-speech (Kokoro) on the same API surface.

Set up 2026-09-27; it rejoins the five-box Incept5 cluster after the course.

## Connection details

| | |
|---|---|
| Base URL (classroom Wi-Fi `i5Net-MLO`) | `http://192.168.8.246:8000/v1` |
| Base URL (Incept5 tailnet) | `http://100.103.86.79:8000/v1` |
| Chat / vision model id | `qwen3.6-35b` |
| Speech-to-text model ids | `whisper-turbo` (fast, ~20x realtime), `whisper-large-v3` (accurate) |
| Text-to-speech, engine 1 | `kokoro` on the base URL above (15 fixed voices, e.g. `af_heart`, `am_michael`, `bf_emma`, `bm_george`) |
| Text-to-speech, engine 2 | **Chatterbox** at `http://192.168.8.246:8031/v1` (23 languages, expressive, **voice cloning**; no key needed) |
| Shared class API key | `sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4` |

The key is one shared credential for the whole class, rate-limited to 1,000 requests/min in
total. **It is withdrawn after the course**, at which point every call returns `401`. Nothing
you send is stored on the box beyond the request in flight, and the model runs entirely locally:
no data leaves the room.

The LAN address is a DHCP lease; if it ever changes, `spark-e049.local` resolves to the box on
the same network, and the instructor will announce the new address.

Quick check from a terminal:

```bash
curl http://192.168.8.246:8000/v1/models -H "Authorization: Bearer sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"
```

## How it is configured (for the curious)

| setting | value | why |
|---|---|---|
| engine | vLLM 0.30.0, stock `vllm/vllm-openai` image, CUDA on the GB10 | the throughput engine; scales with concurrent users instead of collapsing |
| weights | `unsloth/Qwen3.6-35B-A3B-NVFP4` (4-bit floating point) | ~25 GB of weights, so most of the 128 GB is free for KV cache |
| context window | 65,536 tokens per request | admission limit; KV pool holds 3.36M tokens = 51 full-length streams |
| concurrency | up to 64 simultaneous sequences | measured stable at 32 with zero failed requests |
| GPU memory reservation | 70 % | leaves room for Whisper and Kokoro on the same GPU |
| speculative decoding | MTP, 3 draft tokens per step (acceptance ≈2.6) | ~2x single-stream decode for free |
| KV cache dtype | FP8 | halves KV memory at no measured quality cost |
| prefix caching | on | a repeated system prompt or document is prefilled once |
| thinking | **off by default** in this course's examples | switchable per request, see below |
| front door | the Incept5 cluster gateway (per-key auth, rate limits, GPU clock governor) | vLLM itself is bound to localhost on the box |
| audio | Whisper large-v3 + large-v3-turbo (whisper.cpp), Kokoro-82M and Chatterbox-Multilingual, all on the same GPU | speech in, speech out, without leaving the box |

## What you will see: speed vs number of users

Measured on this box on 2026-09-27 with 2,048-token prompts and 256-token answers. The
columns are what **each individual user** experiences, not the box total.

| concurrent users | decode speed each user sees | prefill speed each user sees | time to first token (median / worst 1 %) | box total output |
|---:|---:|---:|---|---:|
| 1 | 89 tok/s | ~7,300 tok/s | 0.28 s / 0.29 s | 78 tok/s |
| 2 | 77 tok/s | ~5,900 tok/s | 0.35 s / 0.54 s | 117 tok/s |
| 4 | 46 tok/s | ~5,400 tok/s | 0.38 s / 0.96 s | 155 tok/s |
| 8 | 34 tok/s | ~4,600 tok/s | 0.45 s / 2.0 s | 193 tok/s |
| 16 | 19 tok/s | ~3,000 tok/s | 0.68 s / 3.9 s | 251 tok/s |
| 32 | 12 tok/s | — | 3.5 s mean / 12.5 s | 294 tok/s |

Reading it:

- **Decode** (tokens per second as the answer streams) is what you notice. Up to 4 users it is
  faster than you can read; at 16 it is still a comfortable reading pace; at 32 it feels slow
  but nothing queues or fails.
- **Prefill** is quoted for a 2,048-token prompt. A short question prefills in well under a
  tenth of a second at any of these loads, so the first token appears almost at once. The
  worst-case column is a user who arrives while others are mid-prefill.
- The box total keeps rising with users. That is the point of a mixture-of-experts model
  plus batching: adding users costs each of them a little and the box nothing.
- **Thinking mode** produces hundreds of reasoning tokens before the answer, so a "brief"
  answer can take ten times longer. Use it deliberately.

Vision was verified the same day: a 1024×659 photo plus a question cost 693 prompt tokens and
answered in 1.8 s end to end. Kokoro renders a sentence in ~0.2 s and Chatterbox in ~2–3 s; speech-to-text
transcribes a 3-second spoken question in ~0.3 s (`whisper-turbo`) or ~0.4 s (`whisper-large-v3`),
and a one-sentence answer with thinking off takes ~0.2 s, so a full spoken round trip is well
under a second of compute.

## Python setup

```bash
pip install openai
pip install sounddevice numpy     # only for the live voice chat (microphone + speaker)
```

Every example below uses the same four lines of setup. The `openai` SDK is used with a custom
`base_url`, the same pattern as the course's other OpenAI-compatible providers.

```python
from openai import OpenAI

BASE_URL = "http://192.168.8.246:8000/v1"
API_KEY = "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"
MODEL = "qwen3.6-35b"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
```

## 1. Text chat, with every knob exposed

The variables at the top are the ones worth playing with. The defaults are Qwen's own
recommendation for non-thinking mode (`temperature 0.7, top_p 0.8, top_k 20`); for thinking
mode Qwen recommends `temperature 1.0, top_p 0.95`. `top_k` and `min_p` are vLLM extensions,
so they travel in `extra_body` rather than as named SDK arguments.

```python
from openai import OpenAI

BASE_URL = "http://192.168.8.246:8000/v1"
API_KEY = "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"
MODEL = "qwen3.6-35b"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ---- things to experiment with -------------------------------------------------
SYSTEM_PROMPT = "You are a concise, helpful assistant."
TEMPERATURE = 0.7          # 0 = near-deterministic, 1+ = creative. Qwen non-thinking: 0.7
TOP_P = 0.8                # nucleus sampling. Qwen non-thinking: 0.8 (thinking: 0.95)
TOP_K = 20                 # only the 20 most likely tokens are ever considered
MIN_P = 0.0                # drop tokens below this fraction of the top token's probability
PRESENCE_PENALTY = 1.5     # discourages repeating itself (Qwen recommends 1.5)
MAX_TOKENS = 512           # hard cap on the answer (INCLUDING thinking tokens if on)
THINKING = False           # True = model reasons first (slower, better on hard problems)
STREAM = True              # print tokens as they arrive instead of waiting for the end
SEED = None                # set an int for repeatable sampling (best effort, not exact)
# ---------------------------------------------------------------------------------


def generate_response(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        presence_penalty=PRESENCE_PENALTY,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        stream=STREAM,
        extra_body={
            "top_k": TOP_K,
            "min_p": MIN_P,
            # This is how thinking is switched for Qwen3.6 on vLLM. When it is on, the
            # reasoning comes back in a separate `reasoning` field, never mixed into content.
            "chat_template_kwargs": {"enable_thinking": THINKING},
        },
    )

    if not STREAM:
        message = resp.choices[0].message
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            print("[thinking]", reasoning, "\n")
        print("usage:", resp.usage.prompt_tokens, "prompt +", resp.usage.completion_tokens, "completion tokens")
        return message.content

    # Streaming: reasoning deltas arrive first (if THINKING), then content deltas.
    answer, in_thinking = "", False
    for chunk in resp:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning", None)
        if reasoning:
            if not in_thinking:
                print("[thinking] ", end="")
                in_thinking = True
            print(reasoning, end="", flush=True)
        if delta.content:
            if in_thinking:
                print("\n[answer] ", end="")
                in_thinking = False
            print(delta.content, end="", flush=True)
            answer += delta.content
    print()
    return answer


if __name__ == "__main__":
    generate_response("In one sentence, what is speculative decoding?")

    # Try the same question with reasoning on. Note the token count and the time.
    THINKING, STREAM, TEMPERATURE, TOP_P = True, False, 1.0, 0.95
    print(generate_response("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball?"))
```

Things to try: set `TEMPERATURE = 0` and run the same prompt twice; set `MAX_TOKENS = 50` with
thinking on and watch the answer get cut off inside the reasoning (an empty `content` with
`finish_reason = "length"` is the tell); raise `PRESENCE_PENALTY` on a long story.

## 2. Vision: ask a question about an image

The same model reads images. An image is just another content part in the user message, sent
as a data URL (or a public `http(s)` URL the box can fetch). A 1024-pixel photo costs about
700 prompt tokens; larger images cost more, up to a few thousand for a full page.

```python
import base64
from pathlib import Path
from openai import OpenAI

BASE_URL = "http://192.168.8.246:8000/v1"
API_KEY = "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"
MODEL = "qwen3.6-35b"
HERE = Path(__file__).parent          # so the script runs from any working directory

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

IMAGE_PATH = HERE / "photo.jpg"       # any jpg/png; try a receipt, a chart, a screenshot
QUESTION = "Describe this photo in two sentences."
TEMPERATURE = 0.2                     # low temperature for factual image reading
MAX_TOKENS = 300
THINKING = False


def image_to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def ask_about_image(question: str, path: Path) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_to_data_url(path)}},
            ],
        }],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": THINKING}},
    )
    print("prompt tokens (text + image):", resp.usage.prompt_tokens)
    return resp.choices[0].message.content


if __name__ == "__main__":
    print(ask_about_image(QUESTION, IMAGE_PATH))
    print(ask_about_image("Transcribe any text you can see, exactly.", IMAGE_PATH))
```

Try a photo of a menu or a whiteboard and ask for a markdown table; try "How many people are
in this picture?" and check the count yourself.

## 3. Text-to-speech: two engines

Spark 5 runs two TTS models. **Kokoro** is small and fast with fixed, very clean voices.
**Chatterbox** is larger and slower but expressive, speaks 23 languages, and can **clone a voice**
from a few seconds of audio. Use Kokoro for a voice assistant, Chatterbox for demos of
what modern TTS can do.

### 3a. Kokoro (fast, fixed voices)

Kokoro is a small, fast TTS model. The voice name encodes language and gender: first letter
is the language, second is `f`/`m` for female/male. The 15 voices on the box:

| language | voices |
|---|---|
| American English | `af_heart` (default), `af_bella`, `am_michael`, `am_puck` |
| British English | `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis` |
| Spanish / French / Italian / Portuguese | `ef_dora`, `ff_siwis`, `if_sara`, `pf_dora` |
| Hindi / Japanese / Mandarin | `hf_alpha`, `jf_alpha`, `zf_xiaobei` |

Output is 24 kHz mono, `wav` or `mp3`, and `speed` runs from 0.5 to 2.0.

```python
from pathlib import Path
from openai import OpenAI

BASE_URL = "http://192.168.8.246:8000/v1"
API_KEY = "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"
HERE = Path(__file__).parent

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

TEXT = "Welcome to the JAX conference in London."
VOICE = "af_heart"           # try am_michael, bf_emma, bm_george, ff_siwis
SPEED = 1.0                  # 0.5 - 2.0
FORMAT = "wav"               # wav or mp3
OUT_FILE = HERE / f"speech.{FORMAT}"


def speak(text: str, out_file: Path) -> Path:
    resp = client.audio.speech.create(
        model="kokoro",
        voice=VOICE,
        input=text,
        speed=SPEED,
        response_format=FORMAT,
    )
    resp.write_to_file(out_file)
    return out_file


if __name__ == "__main__":
    path = speak(TEXT, OUT_FILE)
    print("wrote", path)      # open it, or: afplay speech.wav (macOS) / aplay speech.wav (Linux)
```

### 3b. Chatterbox (multilingual, expressive)

Chatterbox is served on its **own port, 8031**, with no API key (any string works). The
`voice` field carries the **language code** rather than a speaker: `en de es fr it pt nl
sv da no fi pl ru el tr ar he hi ja ko zh ms sw`. Without a reference clip it uses its
built-in default speaker. Three knobs travel in `extra_body` because they are Chatterbox's
own, not OpenAI's:

| knob | range | default | effect |
|---|---|---|---|
| `exaggeration` | 0.25 – 2.0 | 0.5 | emotional intensity; above ~0.7 gets theatrical |
| `cfg_weight` | 0.0 – 1.0 | 0.5 | how tightly it follows the text; lower = more natural pacing |
| `temperature` | 0.1 – 1.5 | 0.8 | sampling randomness, as for an LLM |

```python
from pathlib import Path
from openai import OpenAI

CHATTERBOX_URL = "http://192.168.8.246:8031/v1"
HERE = Path(__file__).parent

cb = OpenAI(base_url=CHATTERBOX_URL, api_key="none")   # no key on this port

TEXT = "Willkommen zum Bootcamp in London. Heute bauen wir mit offenen Modellen."
LANGUAGE = "de"              # try en, fr, es, ja, sv ... (23 languages)
EXAGGERATION = 0.5
CFG_WEIGHT = 0.5
TEMPERATURE = 0.8
FORMAT = "wav"               # wav or mp3
OUT_FILE = HERE / f"chatterbox_{LANGUAGE}.{FORMAT}"


def speak_chatterbox(text: str, out_file: Path) -> Path:
    resp = cb.audio.speech.create(
        model="chatterbox",
        voice=LANGUAGE,                       # language code, not a speaker name
        input=text,
        response_format=FORMAT,
        extra_body={
            "exaggeration": EXAGGERATION,
            "cfg_weight": CFG_WEIGHT,
            "temperature": TEMPERATURE,
        },
    )
    resp.write_to_file(out_file)
    return out_file


if __name__ == "__main__":
    print("wrote", speak_chatterbox(TEXT, OUT_FILE))
```

Generation is stochastic: the same text gives a slightly different reading each time, and
very occasionally a short sentence trails off into a few nonsense syllables. Regenerate, or
lower `temperature`.

### 3c. Voice cloning with Chatterbox

Record **5 to 15 seconds** of a voice (your own phone voice memo is ideal: one speaker, no
music, no long pauses) and pass the file as a base64 `audio_prompt`. Chatterbox then says any
text, in any of its languages, in that voice. The clip can be wav, mp3, m4a or ogg. It was
verified on this box with a 7-second clip: the output reproduced the requested sentence
word for word in about 2.7 s.

```python
import base64
from pathlib import Path
from openai import OpenAI

CHATTERBOX_URL = "http://192.168.8.246:8031/v1"
HERE = Path(__file__).parent

cb = OpenAI(base_url=CHATTERBOX_URL, api_key="none")

REFERENCE_CLIP = HERE / "my_voice.m4a"    # 5-15 s of the voice to clone
TEXT = "This sentence is spoken in a cloned voice, generated locally on the Spark."
LANGUAGE = "en"                           # the language of TEXT; the clip can be in another
EXAGGERATION = 0.5                        # keep near 0.5 for a faithful clone
CFG_WEIGHT = 0.5
TEMPERATURE = 0.8
OUT_FILE = HERE / "cloned.wav"


def clone_voice(text: str, reference: Path, out_file: Path) -> Path:
    clip_b64 = base64.b64encode(reference.read_bytes()).decode()
    resp = cb.audio.speech.create(
        model="chatterbox",
        voice=LANGUAGE,
        input=text,
        response_format="wav",
        extra_body={
            "audio_prompt": clip_b64,                       # the voice to imitate
            "audio_prompt_format": reference.suffix[1:],    # "m4a", "wav", "mp3", "ogg"
            "exaggeration": EXAGGERATION,
            "cfg_weight": CFG_WEIGHT,
            "temperature": TEMPERATURE,
        },
    )
    resp.write_to_file(out_file)
    return out_file


if __name__ == "__main__":
    print("wrote", clone_voice(TEXT, REFERENCE_CLIP, OUT_FILE))
    # Cross-lingual: same voice, different language.
    LANGUAGE = "fr"
    print("wrote", clone_voice("Bonjour à tous, et bienvenue à Londres.", REFERENCE_CLIP, HERE / "cloned_fr.wav"))
```

No reference clip to hand? Make one with Kokoro (example 3a, voice `bm_george`) and clone
that: it is a clean demonstration that the clone really follows the reference. Only clone
voices you have permission to use.

## 4. Speech-to-text (Whisper)

Two Whisper models run on the box: `whisper-turbo` (large-v3-turbo, fast) and
`whisper-large-v3` (the full model, more accurate on accents and noise). Both accept wav, mp3,
m4a, ogg and most other formats. `response_format="verbose_json"` adds per-segment timestamps
and the detected language; `"json"` returns only the text.

**Set the language when you know it.** With auto-detect, a short clip that starts with a
moment of quiet can be assigned the wrong language: in testing, `whisper-large-v3` heard an
English "What's the second…" as Welsh ("Oes y second…"). `language="en"` fixed it on the same
recording. Words that sound alike are a separate problem that no setting fixes: both models
heard one speaker's "tallest" as "tourist", even with a `prompt` hinting at tall buildings.

```python
from pathlib import Path
from openai import OpenAI

BASE_URL = "http://192.168.8.246:8000/v1"
API_KEY = "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"
HERE = Path(__file__).parent

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

AUDIO_FILE = HERE / "speech.wav"     # e.g. the file produced by the TTS example
STT_MODEL = "whisper-turbo"          # or "whisper-large-v3" for accuracy
LANGUAGE = "en"                      # ISO code, or None to auto-detect
TEMPERATURE = 0.0                    # 0 = most literal transcription
RESPONSE_FORMAT = "verbose_json"     # "json" for text only


def transcribe(path: Path):
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            language=LANGUAGE,
            temperature=TEMPERATURE,
            response_format=RESPONSE_FORMAT,
        )
    print("text:", result.text)
    for seg in getattr(result, "segments", None) or []:
        print(f"  {seg.start:5.1f}s - {seg.end:5.1f}s  {seg.text.strip()}")
    return result


if __name__ == "__main__":
    transcribe(AUDIO_FILE)
```

Run example 3 and then example 4 and you have a round trip: text → speech → text. Then record
yourself on your phone, drop the file next to the script, and compare the two Whisper models on
it.

## Putting it together: a voice assistant in twenty lines

Chain the three: transcribe a question, answer it, speak the answer. You need a recorded
question next to the script: record one on your phone, or on a Mac make one with
`say -o question.wav --data-format=LEI16@16000 "What is the tallest building in London?"`.

```python
from pathlib import Path
from openai import OpenAI

BASE_URL = "http://192.168.8.246:8000/v1"
API_KEY = "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"
HERE = Path(__file__).parent
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

with open(HERE / "question.wav", "rb") as f:
    question = client.audio.transcriptions.create(model="whisper-turbo", file=f, language="en").text
print("Q:", question)

answer = client.chat.completions.create(
    model="qwen3.6-35b",
    messages=[{"role": "system", "content": "Answer in one short spoken sentence."},
              {"role": "user", "content": question}],
    temperature=0.7, top_p=0.8, max_tokens=120,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
).choices[0].message.content
print("A:", answer)

client.audio.speech.create(model="kokoro", voice="af_heart", input=answer,
                           response_format="wav").write_to_file(HERE / "answer.wav")
```

## A live voice chat from your microphone

The same three calls in a loop, with your microphone instead of a file. It greets you, listens
until you pause, answers out loud, and listens again; follow-up questions ("and what's its
address?") work because recent turns are sent back each time. Say "stop", "that's enough" or
"goodbye" to end it, or press Ctrl-C.

Needs `pip install sounddevice numpy`. The first run asks for microphone permission for your
terminal or IDE (macOS: System Settings → Privacy & Security → Microphone). Headphones help:
on laptop speakers the microphone can hear the end of the reply.

```python
import io
import re
import time
import wave

import numpy as np
import sounddevice as sd
from openai import OpenAI

BASE_URL = "http://192.168.8.246:8000/v1"
API_KEY = "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"
MODEL = "qwen3.6-35b"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ---- things to experiment with -------------------------------------------------
SYSTEM_PROMPT = ("You are a voice assistant. The user's words come from speech recognition "
                 "and may contain misheard words; if something sounds odd, answer what they "
                 "most likely said. Answer in one short spoken sentence.")
STT_MODEL = "whisper-large-v3"   # or "whisper-turbo": ~0.1 s faster, a little less accurate
LANGUAGE = "en"                  # None = auto-detect, which can guess wrong on short clips
VOICE = "af_heart"               # any Kokoro voice from example 3a
HISTORY = 8                      # earlier messages kept, so follow-up questions work
SILENCE_RMS = 0.01               # mic level that counts as speech; raise it in a noisy room
SILENCE_SECS = 1.0               # stop recording after this much quiet
MAX_SECS = 15                    # longest question
STOP_WORDS = {"stop", "quit", "exit", "bye", "goodbye", "enough", "done", "finished"}
# ---------------------------------------------------------------------------------

RATE = 16000                     # Whisper's native sample rate


def speak(text: str) -> None:
    wav = client.audio.speech.create(model="kokoro", voice=VOICE, input=text,
                                     response_format="wav").content
    with wave.open(io.BytesIO(wav)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        sd.play(audio, w.getframerate(), blocking=True)


def listen():
    """Record until a pause. Returns a WAV file object, or None if nobody spoke."""
    chunks, heard, quiet = [], False, 0
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32") as mic:
        for _ in range(int(MAX_SECS * 10)):
            chunk, _ = mic.read(RATE // 10)                 # 0.1 s at a time
            chunks.append(chunk)
            loud = np.sqrt(np.mean(chunk ** 2)) > SILENCE_RMS
            heard |= loud
            quiet = 0 if loud else quiet + 1
            if heard and quiet >= SILENCE_SECS * 10:
                break
    if not heard:
        return None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.concatenate(chunks) * 32767).astype(np.int16).tobytes())
    buf.name = "question.wav"      # the API uses the name to detect the format
    buf.seek(0)
    return buf


def wants_to_stop(text: str) -> bool:
    # Stop if any short phrase contains a stop word: "Stop.", "OK, that's enough.",
    # "That's enough. Stop. Thank you." -- but not "How do I exit the subway?"
    for phrase in re.split(r"[.,!?;]", text.lower()):
        words = re.sub(r"[^a-z' ]", "", phrase).split()
        if 0 < len(words) <= 3 and STOP_WORDS & set(words):
            return True
    return False


def main() -> None:
    history = []
    speak("Hi, how can I help?")
    while True:
        print("Listening...")
        recording = listen()
        if recording is None:
            continue               # silence: Whisper would invent words, so don't send it
        question = client.audio.transcriptions.create(
            model=STT_MODEL, file=recording, language=LANGUAGE).text.strip()
        print("Q:", question)
        if not re.search(r"\w", question):
            continue               # background noise often comes back as just "."
        if wants_to_stop(question):
            speak("Goodbye!")
            break

        history.append({"role": "user", "content": question})
        answer = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history[-(HISTORY + 1):],
            temperature=0.7, top_p=0.8, max_tokens=120,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
        ).choices[0].message.content
        history.append({"role": "assistant", "content": answer})
        print("A:", answer)

        speak(answer)
        time.sleep(0.3)            # let the speaker fall silent so the mic doesn't hear it


if __name__ == "__main__":
    main()
```

What we learned building it (all on this box, 2026-09-27):

- **Most wrong answers start as wrong transcripts.** The LLM never hears you, only Whisper's
  text. "What's the second tallest…" transcribed as "It's the second tallest building in New
  York." got three different wrong buildings in three runs; the correctly transcribed question
  got Central Park Tower two times out of three. That is why this uses `whisper-large-v3` and
  pins `language="en"`: the extra ~0.1 s is worth it.
- **Tell the model its input is a transcript.** With the plain "answer in one sentence"
  prompt, "What's the second tourist building in London?" got a lecture on tourist
  buildings every time; with the prompt above the model sometimes answers "I think you mean
  the second tallest building…". Better, not solved.
- **Thinking stays off.** It would fix some answers but adds seconds of reasoning to every
  turn, which kills a conversation. The price is confident mistakes on facts: in one session it
  gave Central Park Tower's address as 505 West 57th Street (it is 225), and in another called One
  Vanderbilt the second tallest. A 3B-active model knows less than it sounds like it does;
  grounding it in real data is Day 2.
- **Don't send silence to Whisper.** Both models transcribe 3 s of pure silence as
  "Thank you.", which the assistant would then answer. So `listen()` returns `None` when
  nothing crossed `SILENCE_RMS`, and a transcript with no words in it (room noise comes back
  as just ".") is skipped.
- **Stop words need care both ways.** People say "OK, that's enough. Stop. Thank you very
  much.", not just "stop"; a 4-word limit missed that. Checking each short phrase between
  punctuation catches it without ending the chat on "How do I exit the subway?".

Things to try: switch `STT_MODEL` to `whisper-turbo` and ask the same tricky question; set
`LANGUAGE = None` and speak another language (Kokoro only speaks the language of its voice,
so change `VOICE` too); set `HISTORY = 0` and ask a follow-up; swap the system prompt back to
"Answer in one short spoken sentence." and see how often a misheard word derails it.

## Using it from other tools

Anything with an "OpenAI-compatible" or "custom base URL" setting works with the base URL,
the key and the model id above. Examples: Open WebUI (Connections → OpenAI API), LangChain's
`ChatOpenAI(base_url=..., api_key=..., model="qwen3.6-35b")`, Cursor and Continue (custom
OpenAI provider), and `curl`:

```bash
curl http://192.168.8.246:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b","messages":[{"role":"user","content":"Hello!"}],
       "max_tokens":100,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Notes and limits

- **Context:** 65,536 tokens per request, prompt plus answer. A request larger than that is
  rejected with a clear error rather than truncated.
- **Thinking:** off unless you pass `chat_template_kwargs: {"enable_thinking": true}`. With it
  on, size `max_tokens` for the reasoning, not the reply: a model that runs out of budget while
  thinking returns an empty answer with `finish_reason: "length"`.
- **Rate limit:** the shared key allows 1,000 requests per minute across the whole class.
  If you write a loop, keep it to a few requests per second and you will never notice it.
- **Determinism:** `temperature=0` gives near-identical but not bit-identical answers under
  concurrent load, because batching changes the arithmetic slightly. `seed` is best effort.
- **The model id `qwen3.8-flash-next` appears in `/v1/models`** but has no box behind it
  during the course; requests for it return 503. Use `qwen3.6-35b`.
- **Chatterbox's port 8031 has no key.** It is reachable by anyone on the classroom network;
  that is fine for the course and is why it is not part of the keyed API.
- **Privacy:** the box keeps no logs of prompt or answer content. It does count requests and
  tokens per key for the rate limit.
