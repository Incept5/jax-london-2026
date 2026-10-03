# Python Setup

How to get a working Python environment for the Bootcamp scripts. Do this once,
before the demos start.

Every script in [`day-1/`](day-1/) is a **standalone program** — there's no build
step and each file imports only the libraries that one demo needs. You can either
install everything up front from [`requirements.txt`](requirements.txt) (simplest),
or `pip install` per-demo as you go (lighter). Both paths are below.

## TL;DR

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then copy `.env.example` to `.env` and add your API keys (see
[Cloud API keys](#cloud-api-keys)).

---

## 1. Check your Python version (need 3.10+)

```bash
python3 --version    # macOS / Linux
python --version     # Windows
```

If Python is missing or older than 3.10:

- **macOS** — `brew install python`, or download from <https://python.org>.
- **Windows** — install from <https://python.org> (tick **"Add Python to PATH"**),
  or `winget install Python.Python.3.12`. The Microsoft Store build also works and
  needs no admin rights.
- **Linux** — `sudo apt install python3 python3-venv python3-pip` (Debian/Ubuntu).

## 2. Create and activate a virtual environment

A virtual environment (venv) keeps these packages isolated from your system Python.
Run this from the **repo root** so the `.venv` sits next to the scripts.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Your prompt should now start with `(.venv)`. The repo's `.gitignore` already
ignores `.venv/`, so it won't be committed.

> **PowerShell gotcha:** if activation is blocked with a script-execution error,
> run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once (no admin needed),
> then activate again.

Always upgrade pip inside a fresh venv — it avoids a lot of wheel/build errors:

```bash
python -m pip install --upgrade pip
```

## 3. Install the dependencies

### Option A — everything at once (recommended)

```bash
pip install -r requirements.txt
```

This installs the superset that lets **every** Day-1 script run. A few packages are
large or slow to build:

- `tensorflow`, `torch` — big downloads (hundreds of MB each).
- `llama-cpp-python` — compiles native code, so it needs a C/C++ toolchain
  (Xcode Command Line Tools on macOS, Build Tools for Visual Studio on Windows,
  `build-essential` on Linux). If you're not running `logit_probabilities.py`, you
  can comment it out of `requirements.txt`.

### Option B — per-demo, lighter

Install just the base, then add extras when you reach a demo. Each script's needs
are listed in [`day-1/README.md`](day-1/README.md).

```bash
# base: covers the local/cloud calling demos + embeddings
pip install requests python-dotenv ollama openai numpy tabulate rich
```

| Demo | Extra install |
|---|---|
| `simple_token_test.py` | `pip install transformers` |
| `icr_demo.py` | `pip install tensorflow tf-keras matplotlib` |
| `word_embeddings.py` | `pip install sentence-transformers scikit-learn plotly` |
| `logit_probabilities.py` | `pip install gradio llama-cpp-python plotly numpy` |
| `getting_started_lm_studio.py` | `pip install lmstudio` |
| `basic_claude.py` | `pip install anthropic` |
| `basic_groq.py`, `getting_started_groq.py`, `ai_astrology_groq.py` | `pip install groq` |
| `basic_mistral.py` | `pip install mistralai` |
| `basic_together.py` | `pip install together` |
| `three_local_backends.py` (MLX path, **Apple Silicon only**) | `pip install mlx-vlm` |

## 4. Local model servers

Most demos call a local model. Install whichever the script uses (see the
prerequisites table in [`day-1/README.md`](day-1/README.md)):

- **Ollama** (`127.0.0.1:11434`) — install from <https://ollama.com>, then
  `ollama pull qwen3.5:4b` and `ollama pull all-minilm`.
- **LM Studio** (`127.0.0.1:1234`) — install, download a model, start the local
  server.

## Cloud API keys

The cloud `basic_*.py` scripts call `load_dotenv()`. Create a `.env` file in the
repo root with whichever keys you have:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROK_API_KEY=...
GROQ_API_KEY=...
MISTRAL_API_KEY=...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
```


## 5. Smoke test

With the venv active and Ollama running:

```bash
python day-1/getting_started_ollama.py
```

If it prints a model response, you're set.

## Running scripts later

Each new terminal needs the venv activated first:

```bash
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python day-1/<script>.py
```

## Troubleshooting

- **`ModuleNotFoundError` after installing** — the venv probably isn't active.
  Check for `(.venv)` in your prompt and confirm `which python` / `where python`
  points inside `.venv`.
- **`pip` not recognised (Windows)** — use `python -m pip ...` instead of bare `pip`.
- **`llama-cpp-python` fails to build** — install a C/C++ toolchain (see Option A),
  or comment that line out if you don't need `logit_probabilities.py`.
- **PyPI blocked by a corporate proxy** — `pip install --proxy http://user:pass@host:port -r requirements.txt`.
- **`403 Forbidden` from Ollama/LM Studio (often while on a VPN)** — the server
  rejects the request because the VPN reroutes `localhost`. The scripts use the
  literal loopback IP `127.0.0.1` to avoid this. If you still hit a 403, either
  set `OLLAMA_ORIGINS="*"` and restart Ollama (`ollama serve`), or exclude
  `127.0.0.1` from the VPN tunnel / briefly disable the VPN — loopback traffic
  never leaves your machine anyway.

## Using PyCharm instead?

PyCharm auto-detects `.venv` when you open the project folder. If it doesn't:
**Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter
→ Virtualenv → Existing**, then point it at `.venv/bin/python` (macOS/Linux) or
`.venv\Scripts\python.exe` (Windows). Open `requirements.txt` and click
**Install requirements** when prompted.
