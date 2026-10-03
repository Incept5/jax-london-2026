# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Teaching material for the **JAX London 2026 "Hands-On GenAI Development Bootcamp"** by John Davies (Incept5) — two live sessions, `day-1/` (AI fundamentals) and `day-2/` (applications), plus per-folder READMEs.

There is **no build system, no test suite, no linter, and no dependency manifest per script.** Every `.py` is a standalone program with an `if __name__ == "__main__":` block; run it directly and it imports only the libraries that one demo needs. Treat each script as an independent program, not part of a package.

## Running a demo

```bash
# from the repo root, with the venv active and the relevant model server running
python day-1/<script>.py
python day-2/<script>.py
```

Setup (once): `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. `requirements.txt` is a convenience *superset* — no single demo needs all of it. See `PYTHON_SETUP.md` for per-demo installs and toolchain notes (`tensorflow`/`torch` are large; `llama-cpp-python` compiles native code).

Scripts fall into two families:

- **Cloud-provider demos** (`basic_*.py`, the `*_groq.py` variants) call `load_dotenv()` and read a key from a `.env` in the **repo root** (copy `.env.example`). Run them *from the repo root* so the `.env` is found. Env var names match the provider: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROK_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`.
- **Local-inference demos** need a server already running on localhost, no key:
  - **Ollama** on `:11434` — `ollama pull qwen3.5:4b`, plus `embeddinggemma` / `all-minilm` for RAG.
  - **LM Studio** on `:1234` (OpenAI-compatible) — used by the vision demos (`find_beer.py`, `visual_ml_studio.py`, `test_lmstudio_vision.py`).
  - **MLX** / **llama.cpp** load weights directly from disk (Apple Silicon for MLX).

## Architecture / patterns that span files

These conventions are shared across many scripts — match them when adding or editing demos.

**Cloud demo shape.** `load_dotenv()` → construct client → a `generate_response(prompt)` function → `__main__` calling it. The `basic_*.py` files are deliberately near-identical across vendors. **OpenAI-compatible providers reuse the `openai` SDK with a custom `base_url`** (e.g. Grok → `https://api.x.ai/v1`) rather than a bespoke client; follow this for any new OpenAI-compatible endpoint.

**Local ↔ cloud twins.** Several day-2 demos ship as a pair: a local-Ollama version and a `*_groq.py` cloud version that swaps the model to `llama-3.3-70b-versatile` (`data_extraction`, `payroll2`, `rag_grimm_fairy_tales`). **When changing one, keep its twin in sync.**

**Local models = Qwen3.5 / Qwen3.6 / Gemma-4 family.** Workhorse is `qwen3.5:4b`. Tags differ per backend: Ollama `qwen3.5:4b`, LM Studio `qwen3.5-4b`. **Thinking mode is toggled deliberately**, in backend-specific ways: `think=False` (Ollama), `/no_think` suffix (LM Studio), `enable_thinking=False` in the chat template (MLX), `reasoning_effort="none"` (Groq). Preserve each script's on/off intent — `day-1/ai_astrology.py` deliberately keeps thinking *on* and renders the reasoning separately. For Qwen3 in non-thinking mode the demos use `temperature=0.7, top_p=0.8, top_k=20`; reuse these rather than inventing values.

**RAG pipeline (day-2)** — every `rag_*.py` shares one shape: chunk the text → embed chunks → embed the query → cosine-similarity top-k → pass the query plus *only the retrieved context* to the LLM. Implementations differ by storage/embedding backend, not by pipeline: `rag_alice_simple.py` (minimal), `rag_alice_in_wonderland.py` (in-memory), `_chromadb.py` (Chroma vector DB), `_transformers.py` (local sentence-transformers embeddings), the Grimm counterparts, and `grimm_fairy_tales_rag_demo.py` (Qwen3-Embedding). `alice_in_one_go.py` is the CAG counterpoint — the whole book fits in a 64k context, so retrieval is skipped. Sample texts/images/audio live in `day-2/data/`.

**Tool / function calling (day-2).** `simple_tool_call.py` is the minimal pattern: a `tools` JSON schema + an `available_functions` dict — detect `tool_calls` in the response, dispatch with the parsed arguments, append a `role:"tool"` message, re-call. `ollama_function_support.py` scales this into a harness that scores installed Ollama models. `MCP/` exposes the same idea over the Model Context Protocol: `mcp_server.py` publishes tools over stdio JSON-RPC, `test_mcp_client_ollama.py` runs the agentic loop against them.

**SQL generation (day-2).** `payroll.py` downloads a Kaggle dataset into a SQLite DB; `payroll2.py` feeds the table schema *plus a few sample rows* to the LLM, extracts the generated SQL from the reply, then executes it.

**`HERE = Path(__file__).parent`** is the standard idiom for locating sibling `data/` assets — keep it so scripts run from any working directory.

## Machine-specific paths

A few day-1 scripts (`three_local_backends.py`, `logit_probabilities.py`) historically hard-code absolute model paths under `/Users/jdavies/.lmstudio/models/...` or a `DEFAULT_MODEL_PATH` constant. Other machines must edit these to point at locally downloaded weights.

## Git

`origin` is `git@github.com:Incept5/jax-london-2026.git`, a **public** repo. The working tree tracks `origin/main`, so normal `git push origin main` works. macOS `.DS_Store` files and the `vibe/` experiment outputs are noise — don't commit them (`.gitignore` covers `.DS_Store`, `.env`, `.venv/`, `__pycache__/`, `TEACHER_NOTES.md`).
