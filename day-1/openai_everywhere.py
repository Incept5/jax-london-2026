"""One OpenAI-SDK call, many endpoints — the same request sent to each backend,
only base_url / api_key / model change. Prints time and tokens/sec, so it doubles
as a laptop-vs-Spark (and vs Apple) speed race.

    python openai_everywhere.py            # every endpoint that's up; others are skipped
    python openai_everywhere.py spark      # just one
    python openai_everywhere.py apple ollama

Before an endpoint will answer, its server has to be running:
  ollama    ollama serve  (and `ollama pull qwen3.5:4b`)
  lmstudio  LM Studio's local server on :1234, model qwen3.5-4b loaded
  apple     Apple's on-device model (Apple Intelligence, macOS 26+). No download,
            no key — the weights live in a system daemon. Expose it OpenAI-style:
                sudo fm license      # once per machine, accept the terms
                fm serve             # leave running; defaults to 127.0.0.1:1976
            (`fm` ships with macOS; needs Apple Intelligence turned on.)
  spark     the class Spark — reachable on the room network, key built in
  together  cloud; needs TOGETHER_API_KEY in the repo-root .env

Keys come from .env (see .env.example).
"""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# One script, many endpoints: everything below speaks the OpenAI API, only base_url,
# api_key and the model name change. Keys come from .env (see .env.example).
ENDPOINTS = {
    "ollama": {"base_url": "http://127.0.0.1:11434/v1", "api_key": "ollama", "model": "qwen3.5:4b",
               "extra_body": {"reasoning_effort": "none"}},
    "lmstudio": {"base_url": "http://127.0.0.1:1234/v1", "api_key": "lm-studio", "model": "qwen3.5-4b",
                 "extra_body": {"reasoning_effort": "none"}},
    # Apple's on-device model, via `fm serve` (see the module docstring for setup).
    "apple": {"base_url": "http://127.0.0.1:1976/v1", "api_key": "apple", "model": "system"},
    "spark": {"base_url": os.getenv("SPARK_BASE_URL", "http://192.168.8.246:8000/v1"),
              "api_key": (os.getenv("SPARK_API_KEY") or "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"), "model": "qwen3.6-35b",
              "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "together": {"base_url": "https://api.together.xyz/v1", "api_key": os.getenv("TOGETHER_API_KEY", ""),
                 "model": "Qwen/Qwen3.5-9B", "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
}

PROMPT = "In one sentence, what is the best thing to do in Victoria on a Sunday?"


def ask(name, cfg, prompt):
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"] or "missing", timeout=60, max_retries=0)
    client.chat.completions.create(model=cfg["model"], messages=[{"role": "user", "content": "Hi"}], max_tokens=1,
                                   stream=False, extra_body=cfg.get("extra_body", {}))  # warm-up, so model loading isn't timed
    start = time.time()
    r = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        stream=False,  # Apple's `fm serve` streams unless told not to; the others don't mind
        extra_body=cfg.get("extra_body", {}),
    )
    elapsed = time.time() - start
    tokens = r.usage.completion_tokens if r.usage else 0
    print(f"{name:9} {cfg['model']:18} {elapsed:5.1f} s  {tokens / elapsed:5.0f} tok/s  {r.choices[0].message.content.strip()}")


if __name__ == "__main__":
    for name in sys.argv[1:] or ENDPOINTS:
        try:
            ask(name, ENDPOINTS[name], PROMPT)
        except Exception as e:  # an endpoint that isn't running shouldn't stop the others
            print(f"{name:9} skipped: {type(e).__name__}: {str(e)[:80]}")
