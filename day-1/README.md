# Day 1 — AI Fundamentals, Local & Cloud Models, Embeddings

The Day‑1 code from the **JAX London 2026 "Hands‑On GenAI Development Bootcamp"**.
These scripts back the Day‑1 slide deck (`JAX London 2026 - Hands-on GenAI
Development Bootcamp Day 1`) and follow its narrative: from classic neural
networks, through what an LLM actually *is* (tokens → embeddings → attention →
decoding), to running models locally, remotely and in the cloud.

By the end of the day you should be able to explain a few things that trip most
people up:

- Why can't an LLM spell, and why is it bad at maths?
- Do LLMs remember the conversation? Why do they "forget" what you told them?
- Why do some models occasionally reply in Chinese?

Every file is a **standalone program** — there is no build step, no shared
`requirements.txt` and no package to install. Run each one directly with
`python <script>.py`; it pulls in only the libraries that one demo needs.

## Prerequisites

Most demos talk to a model. Depending on the script you need one (or more) of:

| Backend | How to get it | Used by |
|---|---|---|
| **Ollama** (`127.0.0.1:11434`) | Install from <https://ollama.com>, then `ollama pull qwen3.5:4b` and `ollama pull all-minilm` | most local demos |
| **LM Studio** (`127.0.0.1:1234`, OpenAI‑compatible) | Install, download `qwen3.5-4b`, start the local server | `getting_started_lm_studio.py`, `three_local_backends.py` |
| **MLX** | Apple‑Silicon only; weights load straight from disk | `three_local_backends.py` |
| **llama.cpp** | `pip install llama-cpp-python`, plus a local `.gguf` | `logit_probabilities.py` |
| **Cloud APIs** | An API key per provider (see below) | `basic_*.py`, `*_groq.py`, `together_chat.py` |

**Recommended models** (newer is almost always better — anything older than ~3
months is "too old"):

- LLMs: `qwen3.5:4b` (the workhorse), `gemma-4:e2b`; if you have the RAM,
  `qwen3.5:9b`, `gemma-4:e4b`, or the big `qwen3.6:27b` / `gemma-4:31b`.
- Embeddings: `all-minilm`, `embeddinggemma`.

**Cloud keys** — the cloud scripts call `load_dotenv()`, so put the matching key
in a `.env` file in the **repo root** (copy `../.env.example` to `../.env`). Run
the scripts from the repo root — e.g. `python day-1/basic_chatGPT.py` — so the
`.env` is picked up:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROK_API_KEY=...
GROQ_API_KEY=...
MISTRAL_API_KEY=...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
```

The class Spark's key (`SPARK_API_KEY=sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4`) is already in `.env.example` and built into
the Spark scripts, so they work without a `.env`. It is shared by the class and disabled after the course;
`SPARK5-STUDENT-ACCESS.md` has the URLs and examples.

---

## 1. Classic AI — a neural network you can train

| Script | What it shows |
|---|---|
| `icr_demo.py` | The "hello world" of neural nets: train a small Keras/TensorFlow model on the **MNIST** handwritten digits, report test accuracy (~97%), then display one prediction. Grounds the slides on weights, biases, activation functions and back‑propagation before we ever touch an LLM. Needs `pip install tensorflow matplotlib`. |

## 2. What an LLM is — tokens, embeddings, attention, decoding

| Script | What it shows |
|---|---|
| `simple_token_test.py` | **Tokenisation.** Loads the `Qwen/Qwen3.5-0.8B` tokenizer, encodes *"Attention Is All You Need"*, prints each token → ID, then decodes back. Shows why an LLM sees numbers, not letters (the spelling/maths question). Needs `pip install transformers`. |
| `simple_token_test2.py` | The same tokeniser on a **bilingual** phrase (*"Welcome to London Bridge, Bienvenidos al Puente de Londres"*) — see how English vs. Spanish words split into different numbers of subword tokens, and what happens to *Nueva York*. |
| `embedding_demo.py` | **Embeddings + cosine similarity.** Embeds `tea / coffee / mud / dirt` with `all-minilm` (via Ollama) and prints the full similarity matrix as a grid — "tea"≈"coffee", "mud"≈"dirt". Needs `pip install numpy tabulate`. |
| `embedding_example.py` | An interactive twin of the above: type a question and it ranks a fixed list of sentences by cosine similarity, reporting the mean, standard deviation and each result's **z‑score** so outliers stand out. Talks to Ollama's REST embeddings endpoint directly. |
| `word_embeddings.py` | **Visualising meaning.** Embeds the system dictionary with `sentence-transformers`, finds the nearest neighbours of a set of target words, reduces to 3‑D with **PCA**, and renders an interactive **Plotly** scatter. Caches vectors in `embeddings.pkl`. Needs `pip install sentence-transformers scikit-learn plotly`. |
| `image_embeddings.py` | **Lab: it's not only text you can embed.** CLIP puts photos and sentences in the same embedding space, so a sentence can find a photo in `day-2/data/`. `python image_embeddings.py --faces` embeds faces of public figures from the LFW dataset (~200 MB download) and checks whether each face's nearest neighbour is the same person. Needs `pip install sentence-transformers pillow scikit-learn`. |
| `face_recognition_demo.py` | **Lab: face recognition with your webcam.** Put a selfie in `day-1/faces/` named after you (the folder is git-ignored), run it, and the webcam labels you live using ArcFace face embeddings and cosine similarity (same person ≈ 0.5–0.8, others ≈ 0.1–0.3). `python face_recognition_demo.py photo.jpg` labels a photo instead of the webcam. Needs `pip install insightface onnxruntime opencv-python` (the model, ~280 MB, downloads on first run). Only use photos of people who have agreed. |
| `3d_plot.html` | A saved example of the Plotly 3‑D output from `word_embeddings.py` — open it in a browser without running anything. |
| `logit_probabilities.py` | **Decoding, made visible.** A Gradio app (on `llama-cpp` + a local GGUF) that shows the next‑token probability distribution as a table and pie chart, lets you inject any candidate token by rank to walk "the path not taken", and surfaces the stop‑token probability — the canonical "*why doesn't it stop?*" demo. Forces Qwen3 no‑think mode. Edit `DEFAULT_MODEL_PATH` to point at your own `.gguf`. Needs `pip install gradio llama-cpp-python plotly numpy`. |

## 3. Running models locally

| Script | What it shows |
|---|---|
| `getting_started_ollama.py` | The smallest possible Ollama call — one `POST` to `/api/generate`. |
| `getting_started_ollama_params.py` | The same call with the **parameters** that matter spelled out: `temperature`, `top_p`, `top_k`, `num_predict`, `seed`, `num_ctx`. |
| `getting_started_lm_studio.py` | The smallest possible **LM Studio** call (OpenAI‑compatible `/v1/chat/completions` on port 1234). |
| `three_local_backends.py` | The same prompt through **three local backends** — Ollama, LM Studio and MLX — side by side. Note how thinking mode is toggled differently per backend (`think=False`, `/no_think`, `enable_thinking=False`). |
| `local_performance_demo.py` | A timing harness: runs a batch of multilingual "difficult" questions through `qwen3.5:4b` (thinking off, `temperature=0`) and prints the latency of each — get a feel for size vs. speed. |
| `quantisation_taste_test.py` | **Lab: quantisation taste test.** The same four prompts through `qwen3.5:4b` at Q4_K_M (the default), Q8_0 and BF16, with the size and tokens/sec of each. Pull the extra two first (`ollama pull qwen3.5:4b-q8_0`, `ollama pull qwen3.5:4b-bf16`); any model you pass on the command line replaces the list. |
| `openai_everywhere.py` | **Lab: one script, many endpoints.** One OpenAI‑SDK call sent to Ollama, LM Studio, the class Spark and Together — only `base_url`, key and model change. Prints time and tokens/sec, so it doubles as a laptop‑vs‑Spark speed race. `python openai_everywhere.py spark` runs a single endpoint. Keys come from `.env`. |
| `model_test.py` | **Lab: test models on your own questions.** Five tricky questions, each with an automatic check, run against two models with thinking off and on. Prints the score, how many answers **ran out of tokens while still thinking** (a small reasoning model can think forever), and the time. Scores vary between runs: run it twice. |

## 4. Calling the cloud providers

Each of these follows the same shape: `load_dotenv()` → build a client →
`generate_response(prompt)` → print. They're deliberately near‑identical so you
can see how little changes between vendors. Note that OpenAI‑compatible
providers (Grok, Groq, Together…) reuse the `openai`/vendor SDK with just a
different model name or `base_url`.

| Script | Provider / model | Notes |
|---|---|---|
| `basic_chatGPT.py` | OpenAI · `gpt-5` | `pip install openai` |
| `basic_claude.py` | Anthropic · `claude-sonnet-4-6` | `pip install anthropic` |
| `basic_grok.py` | xAI Grok · `grok-4-latest` | uses the `openai` SDK with `base_url=https://api.x.ai/v1` |
| `basic_groq.py` | Groq · `qwen/qwen3-32b` | Groq = fast hardware host (US‑based — mind GDPR) |
| `basic_mistral.py` | Mistral · `mistral-medium-latest` | the EU vendor |
| `basic_fireworks.py` | Fireworks · `deepseek-v3p2` | raw `requests`, no SDK |
| `basic_together.py` | Together · `Qwen3-235B-A22B-Instruct` | `pip install together` |
| `getting_started_groq.py` | Groq, stripped to the bare minimum | the "first contact" version of `basic_groq.py` |
| `together_chat.py` | Together via raw `requests` | reads `TOGETHER_API_KEY` from `.env` |

## 5. Beyond text — vision, and a worked example

| Script | What it shows |
|---|---|
| `vision_demo_ollama.py` | **Multimodal.** Sends a local `image.jpg` plus a prompt to a vision model (`qwen3.6:27b`) and asks it to describe the image and guess where it was taken. Drop in any image. |
| `ai_astrology.py` | A small end‑to‑end app: prompts for a name and star sign, then has `qwen3.5:4b` (via Ollama) write a horoscope. Thinking is deliberately left **on** and the model's reasoning is rendered separately with `rich`. Needs `pip install ollama rich`. |
| `ai_astrology_groq.py` | The cloud twin of the above, swapping Ollama for **Groq** (`qwen/qwen3-32b`, `reasoning_effort="none"`). Keep the two in sync if you change one. |

---

## Suggested running order

1. `icr_demo.py` — see a classic neural network learn.
2. `simple_token_test.py` — see text become tokens.
3. `embedding_demo.py` → `embedding_example.py` → `word_embeddings.py` — see meaning become geometry.
4. `logit_probabilities.py` — watch the model choose the next token.
5. `getting_started_ollama.py` → `getting_started_ollama_params.py` → `three_local_backends.py` — run models locally and tune them.
6. The `basic_*.py` family — the same task across every major cloud provider.
7. `vision_demo_ollama.py` and `ai_astrology.py` — put it together.

## Voice

| Script | What it shows |
|---|---|
| `voice_astrology.py` | **Lab: Voice AI Astrology.** Say your name and star sign; Whisper transcribes it, Maude (Qwen3.6 on the class Spark) writes a horoscope, and Kokoro, or a **cloned voice** via Chatterbox, reads it out. Set `VOICE_CLIP` to a 5–15 s recording to clone it: `day-2/data/roosevelt-voice-sample.wav` is Theodore Roosevelt in 1912 (public domain). Needs `pip install sounddevice numpy openai python-dotenv` and `SPARK_API_KEY` in `.env`; `python voice_astrology.py some.wav` uses a recording instead of the microphone. |
