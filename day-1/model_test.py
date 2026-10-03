import re
import sys
import time

from openai import OpenAI

# Test models on YOUR questions: every question has a check, so the score is automatic.
# Works with anything OpenAI-compatible; here it's Ollama. Add or change models freely.
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama", timeout=600)
MODELS = sys.argv[1:] or ["qwen3.5:0.8b", "qwen3.5:4b"]

QUESTIONS = [
    ("What is 37*27*1001? Reply with just the number.", lambda a: "999999" in a.replace(",", "")),
    ("Is 99,409 a prime number? Answer yes or no.", lambda a: re.search(r"\byes\b", a, re.I) is not None),
    ("How many letter r's are in the word 'strawberry'? Reply with just the number.", lambda a: re.search(r"\b3\b|three", a, re.I) is not None),
    ("Which is heavier: a kilo of feathers or a kilo of bricks?", lambda a: re.search(r"same|neither|equal|both", a, re.I) is not None),
    ("I need to wash my car. The car wash is 50 meters away. Should I walk or drive? Answer in one word.",
     lambda a: re.search(r"\bdrive\b", a, re.I) is not None),
]

# Thinking off vs on: "none" switches reasoning off, "medium" lets the model think first
EFFORTS = ["none", "medium"]

print(f"{'model':16} {'thinking':9} {'score':6} {'ran out':8} {'time':>6}")
for model in MODELS:
    for effort in EFFORTS:
        score, ran_out, start = 0, 0, time.time()
        for question, check in QUESTIONS:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": question}],
                max_tokens=4096,                # room for the reasoning, not just the answer
                extra_body={"reasoning_effort": effort},
            )
            score += check(r.choices[0].message.content or "")
            ran_out += r.choices[0].finish_reason == "length"   # still thinking when the budget ran out
        print(f"{model:16} {effort:9} {score}/{len(QUESTIONS):<4} {ran_out:<8} {time.time() - start:5.0f}s")
