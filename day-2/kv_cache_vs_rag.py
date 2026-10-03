import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import ollama
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
HERE = Path(__file__).parent
TEXT = (HERE / "data" / "alice_in_wonderland.txt").read_text()   # ~40k tokens: fits in the Spark's 64k context
WHOLE_BOOK = "--whole-book" in sys.argv   # the default is chapters I-III (~8k tokens): quick even when the whole room runs it

spark = OpenAI(base_url=os.getenv("SPARK_BASE_URL", "http://192.168.8.246:8000/v1"), api_key=(os.getenv("SPARK_API_KEY") or "sk-GMHXy89Nrhku0BfnA6FrYR6F_sOYJeJ4"))
MODEL = "qwen3.6-35b"
if WHOLE_BOOK:
    BOOK = TEXT
    QUESTIONS = [
        "What is written on the little bottle Alice finds?",
        "Who is at the mad tea-party?",
        "What does the Queen use as croquet mallets and balls?",
        "How does the story end?",
    ]
else:
    start = TEXT.index("CHAPTER I.", TEXT.index("CHAPTER I.") + 1)      # skip the table of contents
    BOOK = TEXT[start:TEXT.index("CHAPTER IV.", start)]
    QUESTIONS = [
        "What is written on the little bottle Alice finds?",
        "What is the pool Alice swims in made of?",
        "Who wins the Caucus-race, and what are the prizes?",
        "How many times does Alice change size, and what causes each change?",   # needs the whole text
    ]

# The whole room shares one Spark and one cache. Everyone sends the same text, so without this the first
# person to run the script would warm the cache for everybody else. A unique tag at the very START makes
# your copy of the text different from everyone else's, so your first question really is uncached.
MY_TAG = f"[reader {uuid.uuid4().hex[:8]}]\n"


def ask(context, question):
    """Stream the answer and time the FIRST token: that's the prefill, where the cache makes the difference."""
    start, first, answer = time.time(), None, ""
    stream = spark.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=120, stream=True,
        messages=[{"role": "system", "content": "Answer in one or two sentences, using only the text provided."},
                  {"role": "user", "content": f"{context}\n\nQuestion: {question}"}],   # stable text first, question last
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.time()
            answer += chunk.choices[0].delta.content
    return answer.strip(), first - start


def show(label, context, questions):
    print(f"\n=== {label}")
    for q in questions:
        answer, ttft = ask(context, q)
        print(f"first token {ttft:6.2f}s  {q}\n                    -> {answer}")


# 1. KV-cache ("CAG"): the whole text every time. The first question pays to read it (prefill, seconds);
#    after that the server reuses the cached keys and values for the identical beginning (a fraction of a second).
show("Whole text: first question uncached, then from the KV-cache", MY_TAG + BOOK, QUESTIONS)

# 2. The cache is a PREFIX cache: change anything at the start and everything after it must be recomputed.
now = f"Today is {datetime.now():%A %d %B %Y, %H:%M:%S}.\n"
show("Change the START (a timestamp before the text): cache miss", now + MY_TAG + BOOK, QUESTIONS[:1])
show("Change the END (the same timestamp after the text): cache hit", MY_TAG + BOOK + "\n" + now, QUESTIONS[:1])

# 3. RAG: chunk, embed, and send only the 4 chunks closest to the question (small prompt, but it can miss things)
chunks = [BOOK[i:i + 1500] for i in range(0, len(BOOK), 1200)]            # 1,500 characters, 300 overlap
vectors = np.array(ollama.embed(model="all-minilm", input=chunks).embeddings)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
print("\n=== RAG: only the 4 best-matching chunks")
for q in QUESTIONS:
    qv = np.array(ollama.embed(model="all-minilm", input=q).embeddings[0])
    best = np.argsort(vectors @ (qv / np.linalg.norm(qv)))[-4:]
    answer, ttft = ask("\n...\n".join(chunks[i] for i in sorted(best)), q)
    print(f"first token {ttft:6.2f}s  {q}\n                    -> {answer}")
