# Quickstart: Claude Code on a Local Model (Ollama)

Run Claude Code on your own laptop — **no API key, no cloud, no cost.**

You'll point **Claude Code** at a model running in **Ollama**. There's one extra
step compared to Codex: Claude Code speaks a different "language" than Ollama, so
you run a tiny **translator** (a proxy) in between. Don't worry — it's two
commands.

⏱️ ~15 minutes (plus model download time).

> Prefer the simplest possible setup? Codex needs no translator — see
> **[CODEX_OLLAMA_QUICKSTART.md](CODEX_OLLAMA_QUICKSTART.md)**.

---

## 1. Install the tools

```bash
# Ollama — the local model server (https://ollama.com)
ollama --version

# Claude Code
npm install -g @anthropic-ai/claude-code

# The translator (LiteLLM)
pip install 'litellm[proxy]'
```

## 2. Download a coding model

```bash
ollama pull qwen2.5-coder:14b
```

> Use a **coding** model. Claude Code needs a model that can call tools; a random
> chat model won't work. On a smaller machine try `qwen2.5-coder:7b`.

## 3. Start the translator

Create a file called `litellm_config.yaml`:

```yaml
model_list:
  - model_name: ollama-local
    litellm_params:
      model: ollama_chat/qwen2.5-coder:14b
      api_base: http://localhost:11434
```

Start it (leave this terminal open — it must keep running):

```bash
litellm --config litellm_config.yaml --port 4000
```

## 4. Launch Claude Code (in a *new* terminal)

```bash
export ANTHROPIC_BASE_URL="http://localhost:4000"
export ANTHROPIC_AUTH_TOKEN="dummy"
export ANTHROPIC_MODEL="ollama-local"
claude
```

✅ Claude Code starts, talking to your local model. Try:

```
write a Python script that prints the first 20 Fibonacci numbers, then run it
```

🎉 You're vibe coding locally.

---

## Make it one short command

Add this to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
alias oc='ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_AUTH_TOKEN=dummy ANTHROPIC_MODEL=ollama-local claude'
```

Start the translator (step 3), then in a new terminal just run `oc`.

> ⚠️ Type the quotes by hand. Copy‑pasted "curly quotes" will break the command.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Errors / "404" in Claude Code | `ANTHROPIC_BASE_URL` must be the translator (`http://localhost:4000`), **not** Ollama's `11434`. |
| `connection refused` | The translator or Ollama isn't running. Start `litellm …` and the Ollama app. |
| `model not found` | `ANTHROPIC_MODEL` must match `model_name` in the YAML, and the model must be pulled (`ollama pull …`). |
| It loops or ignores your request | Your model can't use tools — switch to `qwen2.5-coder` or `gpt-oss`. |
| Answers get cut off | Restart Ollama with more context: `OLLAMA_CONTEXT_LENGTH=32768 ollama serve` |
| Times out / too slow | Use a smaller model (`qwen2.5-coder:7b`). |

---

## Try it on today's exercise

Open `vibe/instruct.txt`, copy the brief, and paste it into Claude Code as your
first message. Watch it build the whole thing.
