# Day 2 — Applications: RAG, Extraction, Tools, Vibe Coding & Agents

The Day‑2 code from the **JAX London 2026 "Hands‑On GenAI Development Bootcamp"**.
These scripts back the Day‑2 slide deck (`JAX London 2026 - Hands-on GenAI
Development Bootcamp Day 2`) and follow its narrative: now that you know what an
LLM *is* (Day 1), what can you *build* with one? The day moves from **RAG**
(retrieval‑augmented generation) through **summarisation, data extraction and
sentiment**, **structured (JSON) output**, **code generation → vibe coding**,
**tool / function calling**, the **Model Context Protocol (MCP)**, and finishes on
modern **agents**.

The recurring theme: almost everything here can be done **locally, privately and
cheaply** with small open‑weight models — you don't need a frontier API.

Every file is a **standalone program** — there is no build step and no package to
install. Run each one directly with `python <script>.py`; it pulls in only the
libraries that one demo needs. [`requirements.txt`](requirements.txt) in this
folder is a convenience superset that installs everything Day 2 needs at once
(`pip install -r day-2/requirements.txt`); you don't need all of it for any single
demo.

## Prerequisites

Most demos talk to a model. Depending on the script you need one (or more) of:

| Backend | How to get it | Used by |
|---|---|---|
| **Ollama** (`127.0.0.1:11434`) | Install from <https://ollama.com>, then `ollama pull qwen3.5:4b`, plus an embedder: `ollama pull embeddinggemma` (and/or `all-minilm`) | most local demos |
| **LM Studio** (`127.0.0.1:1234`, OpenAI‑compatible) | Install, download a vision model (`qwen3.5-4b`), start the local server | `find_beer.py`, `visual_ml_studio.py`, `test_lmstudio_vision.py` |
| **MLX** | Apple‑Silicon only; weights load straight from disk | `chatterbox_tts.py` (TTS) |
| **Cloud APIs** | An API key per provider (see below) | the `*_groq.py` twins |
| **Kaggle** | `pip install kagglehub` — datasets download on first run | `payroll.py`, `*_kaggle.py`, `kaggle_summary_complete.py` |

**Recommended models** — `qwen3.5:4b` is the workhorse LLM; for embeddings use
`embeddinggemma` or `all-minilm` (some RAG demos pull a `Qwen/Qwen3-Embedding`
model from Hugging Face instead). Newer is almost always better; anything older
than ~3 months is "too old".

**Cloud keys** — the `*_groq.py` scripts call `load_dotenv()`, so put your key in
a `.env` file in the **repo root** (copy `../.env.example` to `../.env`). Run the
scripts from the repo root — e.g. `python day-2/data_extraction_groq.py` — so the
`.env` is picked up:

```
GROQ_API_KEY=...
```

Sample books, images, audio and datasets live in **`data/`**.

---

## 1. Recap — multimodal: image → text

We open by revisiting Day 1's vision work and pushing it further: a vision model
can read charts, diagrams, sheet music, menus and photos.

| Script | What it shows |
|---|---|
| `visual_ollama.py` | The core **image → text** demo: base64‑encode an image and POST it to Ollama's vision model, asking it to describe the chart (the IEEE "Responsible AI risks" bar chart from the slides). |
| `visual_ml_studio.py` | The same idea against **LM Studio**'s OpenAI‑compatible endpoint. |
| `test_lmstudio_vision.py` | The smallest possible LM Studio vision call — one image, one sentence back. |
| `find_beer.py` | **Object detection / grounding.** Asks several LM Studio vision models to return bounding boxes for objects in a photo, then draws them with Pillow — and compares models. |
| `read_diagram.py` | Reads a diagram and returns a **Mermaid** description of its structure. |
| `read_music.py` | Reads a music score and names the **key signature**. |
| `read_ocr_menu.py` | **OCR** a menu/photo to text (the "document is a PDF/image" path into RAG). |
| `ice_cream_flavours.py` | A quick **menu‑reading** demo: base64‑encode an ice‑cream menu photo (`data/IMG_3319.jpeg`, with `IMG_3320`/`IMG_3321` as alternatives) and ask Ollama's `qwen3.5:4b` vision model to list the flavours. |
| `read_pdf.py` | **Lab: read a PDF page, then ask about its chart.** Renders ONE page of *Housing New York: A Five-Borough, Ten-Year Plan* (City of New York, 2014; `data/housing-new-york.pdf`, from nyc.gov/housing) and has the `qwen3.5:4b` vision model transcribe it to Markdown (tables become Markdown tables), saved as `housing-page-<n>.md`. Then ask questions about the chart or picture on that page. Page 23 (the default) is a bar chart of housing permits and completions; page 20 has a chart and a table: `python read_pdf.py 20`. About 10–15 s a page. Needs `pip install pymupdf`. |

## 2. RAG — Retrieval‑Augmented Generation

The heart of Day 2. Every `rag_*.py` shares one pipeline: **chunk** the text →
**embed** the chunks → embed the **query** → **cosine‑similarity** top‑k → pass the
query plus *only the retrieved context* to the LLM. They differ only by storage /
embedding backend, so you can compare approaches.

| Script | What it shows |
|---|---|
| `rag_alice_simple.py` | The minimal RAG loop on *Alice in Wonderland*, embeddings via `embeddinggemma` (Ollama). Start here. |
| `rag_alice_in_wonderland.py` | The same, refactored into a reusable in‑memory `Document`/RAG structure. |
| `rag_alice_in_wonderland_chromadb.py` | Stores the chunk embeddings in a **Chroma** vector database instead of memory — so you don't re‑embed every run. |
| `rag_alice_in_wonderland_transformers.py` | Runs the embedding model **locally via `transformers`** (no Ollama embed endpoint) — usually faster. |
| `rag_grimm_fairy_tales.py` | The Alice pipeline pointed at *Grimm's Fairy Tales* (works in multiple languages — there's a German copy in `data/`). |
| `rag_grimm_fairy_tales_groq.py` | The cloud twin: `sentence-transformers` embeddings + **Groq** (`llama-3.3-70b-versatile`) for generation. Keep it in sync with the local version. |
| `grimm_fairy_tales_rag_demo.py` | A richer demo using a **`Qwen3-Embedding`** model (batched, 512‑dim, with a reranking‑style top‑k) and `rich` output. |
| `alice_in_one_go.py` | The **CAG** counterpoint (cache‑augmented generation): the corpus is small enough to fit in a 64k context, so skip retrieval and feed the *whole book* to the model. |
| `kv_cache_vs_rag.py` | **Lab: KV-cache vs RAG.** Four questions about *Alice* on the class Spark, timing the **first token**: the whole book with a per-person tag at the start (so your first question really is uncached: ~6.5 s alone), then the same book again (read from the cache: ~0.9 s); a timestamp added at the **start** invalidates the cache, the same timestamp at the **end** doesn't; RAG with 4 chunks (~0.25 s, but it can miss things). With the whole room running it at once the Spark is shared, so expect several times longer. Needs `SPARK_API_KEY` in `.env` and `ollama pull all-minilm`. |

## 3. Summarisation & Data Extraction

Summarisation taken to the extreme *is* data extraction. These show both, plus
the common "scrape a page, then extract" pattern.

| Script | What it shows |
|---|---|
| `data_extraction_ollama.py` | `summarise()` and `extract()` helpers over a sample model card — the basic summarise/extract pattern. |
| `data_extraction_groq.py` | The cloud twin (Groq `llama-3.3-70b-versatile`). |
| `scrape.py` | A generic **BeautifulSoup** web scraper — the raw‑text source for extraction. |
| `scrape_gdpr_article.py` | Scrapes a GDPR article from gdpr-info.eu — the basis for the "summarise each article" exercise (see also `vibe/`). |
| `kaggle_summary_complete.py` | End‑to‑end: download a Kaggle dataset, run sentiment over it with `TextBlob`, and plot the results with matplotlib. |

## 4. Sentiment

Sentiment is just extraction where the thing you extract is a *mood*.

| Script | What it shows |
|---|---|
| `analyse_sentiment_01.py` | One‑word sentiment from a local model, with light "old‑fashioned" string cleanup to make a small model reliable. |
| `analyse_sentiment_02.py` | The same with slightly different post‑processing — compare robustness. |
| `analyse_sentiment_kaggle.py` | Run sentiment over a real dataset (Trump tweets from Kaggle). Use a `time.sleep(1)` if you point this at anything remote, to avoid rate‑limits. |

## 5. Structured (JSON) output

You get far more reliable output asking for **JSON** than free‑form text.

| Script | What it shows |
|---|---|
| `formatted_response_example.py` | Ollama's `format="json"` mode — ask for numbers 1‑10 in nine languages and parse the result. |
| `formatted_response_streaming.py` | The same, **streamed** token‑by‑token. |

## 6. Code generation → Vibe coding

LLMs are good at code (SQL, regex, whole programs). The deck demos Claude Code;
the local scripts here show generating, running and **testing** code, and the
high‑value special case of **SQL generation** from a schema.

Want the agentic "vibe coding" experience entirely on a local model? No API key
required:

- **[CODEX_OLLAMA_QUICKSTART.md](CODEX_OLLAMA_QUICKSTART.md)** — OpenAI's Codex
  CLI on an Ollama model. The easy path (no proxy). Start here.
- **[CLAUDE_CODE_OLLAMA_QUICKSTART.md](CLAUDE_CODE_OLLAMA_QUICKSTART.md)** — Claude
  Code on an Ollama model (one extra step: a tiny local translator).

| Script | What it shows |
|---|---|
| `payroll.py` | Downloads a payroll dataset from Kaggle and loads it into a **SQLite** database — the setup step. |
| `payroll2.py` | Feeds the **table schema plus a few sample rows** to a local LLM, extracts the generated SQL from the reply, then executes it. Giving the model sample data dramatically improves the SQL. |
| `payroll2_groq.py` | The cloud twin (Groq). |
| `vibe/` | The **vibe‑coding exercise** — `instruct.txt` is the natural‑language brief ("scrape the GDPR articles, build a site to browse them…") you hand to an agentic coding tool. |

## 7. Tool / function calling

Letting the model call your functions — the web, a database, the time, anything.

| Script | What it shows |
|---|---|
| `simple_tool_call.py` | The minimal pattern: a `tools` JSON schema + an `available_functions` dict — detect `tool_calls`, dispatch with the parsed arguments, append a `role:"tool"` message, re‑call. The currency‑conversion example from the slides. |
| `ollama_function_support.py` | Scales that into a **benchmark harness** that iterates over installed Ollama models and scores their function‑calling (success rate, latency, σ) — because tool‑calling is non‑deterministic and not every model can do it. |

## 8. MCP — Model Context Protocol

The standardised way to publish and consume tools. See **[MCP/README.md](MCP/README.md)**
for full detail.

| Script | What it shows |
|---|---|
| `MCP/mcp_server.py` | An MCP **server** exposing two tools (`get_country_info`, `get_weather`) over stdio JSON‑RPC. |
| `MCP/test_mcp_client_ollama.py` | An MCP **client** running an agentic loop against the server with a local Ollama model — e.g. "what's the weather in the capital of Germany?" chains two tool calls. |

## 9. Beyond — speech, and a note on agents

| Script | What it shows |
|---|---|
| `chatterbox_tts.py` | **Text → speech** with voice cloning via MLX (`mlx-audio`) — the "goodbye from Ricky" TTS demo. Apple‑Silicon only. Sample outputs live in `data/` (`ricky-london-ai-full.mp3`, `klopp-de*.mp3`). |
| `clone_voice.py` | **Clone a voice** from a 5–15 s sample (Roosevelt by default) with Chatterbox on the class Spark: plays it out loud and saves it to `output/`. Windows, Mac or Linux; nothing to download (`pip install openai sounddevice numpy`). |
| `chatterbox_turbo_tts.py` | The same voice-cloning demo with PyTorch (`pip install chatterbox-tts`), for **Windows and Linux** (also runs on a Mac). Uses an NVIDIA GPU if there is one, otherwise the CPU. Copy the weights from the USB drive's `6-voice-windows-linux` folder first; otherwise it downloads 3.8 GB. |

The closing slides (Embabel, Open‑Claw, Hermes Agent, running an agent 24/7 on a
Raspberry Pi, email/home access) are **conceptual** — no scripts in this folder.
The takeaway: *the future is small, local, open‑weight models you own and control.*

---

## Suggested running order

1. `visual_ollama.py` — recap: a model reads an image.
2. `rag_alice_simple.py` → `rag_alice_in_wonderland.py` → `_chromadb.py` — build RAG, then add a vector DB.
3. `alice_in_one_go.py` — compare RAG with the "fits in context" (CAG) approach.
4. `data_extraction_ollama.py` → `scrape_gdpr_article.py` — summarise and extract.
5. `analyse_sentiment_01.py` — extraction as sentiment.
6. `formatted_response_example.py` — make output reliable with JSON.
7. `payroll.py` → `payroll2.py` — generate and run SQL from a schema.
8. `simple_tool_call.py` → `ollama_function_support.py` — tool calling, then benchmark it.
9. `MCP/test_mcp_client_ollama.py` — the same idea standardised over MCP.

## 8. Agents working together

| Script | What it shows |
|---|---|
| `persona_debate.py` | **Lab: persona debate.** Three to five characters, each with a back-story, a personality and its own corpus as memory, debate a question over two rounds, then each votes for the most convincing *other* speaker. Pick a `SET`: **film**, the default (Darth Vader, Mary Poppins, Captain Kirk and the Terminator on breaking the rules; scripts are copyrighted, so their memories are a few lines written for the lab), **philosophers** (Socrates, Machiavelli, Nietzsche on *"Is it better to be feared or loved?"*, each reading their own book), **famous** (Oscar Wilde, Benjamin Franklin, Mark Twain on money and happiness) or the original **literary** set (Alice, the Hatter, Little Red-Cap). The philosophers' and famous people's texts are public-domain Project Gutenberg books in `data/`. Runs on the class Spark (`SPARK_API_KEY` in `.env`); switch `base_url`/`MODEL` to use Ollama. Change the question, swap a persona, or add your own. |
