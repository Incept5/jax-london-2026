import sys
import time

import ollama

# The same model at different quantisations. Pick sizes that fit your memory (and your Wi-Fi):
#   python day-1/quantisation_taste_test.py 0.8b 2b        (the default)
#   python day-1/quantisation_taste_test.py 4b             (or 9b with 32 GB+), or give full model names
# Download sizes (Q4 / Q8 / BF16): 0.8b - / 1.0 / 1.8 GB · 2b 1.9 / 2.7 / 4.6 GB · 4b 3.4 / 5.3 / 9.3 GB
#                                  9b 6.6 / 11 / 19 GB.  0.8b has no Q4 on Ollama: its default is already Q8.
QUANTS = {"0.8b": ["q8_0", "bf16"]}
SIZES = sys.argv[1:] or ["0.8b", "2b"]
MODELS = [m for s in SIZES for m in ([s] if ":" in s else [f"qwen3.5:{s}-{q}" for q in QUANTS.get(s, ["q4_K_M", "q8_0", "bf16"])])]

PROMPTS = [
    "What is 17 x 24? Reply with just the number.",
    "Name five London boroughs, comma separated.",
    "Translate into French: The early bird catches the worm.",
    "Write a haiku about the Victoria Embankment.",
]

client = ollama.Client()
installed = {m.model for m in client.list().models}

for model in MODELS:
    if model not in installed:
        print(f"\n=== {model}: not installed, run: ollama pull {model}")
        continue
    size_gb = next(m.size for m in client.list().models if m.model == model) / 1e9
    print(f"\n=== {model} ({size_gb:.1f} GB)")
    client.generate(model=model, prompt="Hi", think=False)  # load it first so timings are fair
    for prompt in PROMPTS:
        start = time.time()
        r = client.generate(model=model, prompt=prompt, think=False, options={"temperature": 0})
        tok_per_s = r.eval_count / (r.eval_duration / 1e9)
        print(f"  {prompt}\n    -> {r.response.strip()}  ({time.time() - start:.1f} s, {tok_per_s:.0f} tok/s)")
