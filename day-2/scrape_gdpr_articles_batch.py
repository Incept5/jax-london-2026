#!/usr/bin/env python3
"""Scrape GDPR articles 20-25 and summarize each using Ollama (qwen3.5:4b)."""

import requests
from bs4 import BeautifulSoup
import re
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"


def fetch_page(url, headers):
    """The room's internet is slow, so read the saved copy in data/gdpr-info/ unless --live is given."""
    saved = HERE / "data" / "gdpr-info" / (url.rstrip("/").rsplit("/", 1)[-1] + ".html")
    if saved.exists() and "--live" not in sys.argv:
        print(f"Reading the saved copy of {url} (add --live to fetch it)")
        return saved.read_text(encoding="utf-8")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def scrape_article_text(article_num: int) -> tuple[str, str] | None:
    """Scrape a single GDPR article page. Returns (title, full_text) or None."""
    url = f"https://gdpr-info.eu/art-{article_num}-gdpr/"
    print(f"  Scraping {url} …", end=" ", flush=True)

    try:
        html = fetch_page(url, HEADERS)
    except requests.RequestException as e:
        print(f"FAILED ({e})")
        return None

    soup = BeautifulSoup(html, "html.parser")

    # --- title ---
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else f"Art. {article_num} GDPR"

    # --- full article body ---
    # The main content lives inside <div class="entry-content"> on this site.
    entry = soup.find("div", class_="entry-content")
    if not entry:
        # fallback: grab everything inside <article>
        entry = soup.find("article")
    if not entry:
        print("FAILED (no content container)")
        return None

    # Gather all paragraph / list-item text, skipping nav / footer pollution.
    paragraphs: list[str] = []
    for tag in entry.find_all(["p", "li", "h2", "h3"]):
        # skip children of nav / footer inside the entry (rare but safe)
        if tag.find_parent(["nav", "footer"]):
            continue
        text = tag.get_text(strip=True)
        if text and len(text) > 10:
            paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)
    if not full_text:
        print("FAILED (empty body)")
        return None

    print(f"OK  ({len(full_text):,} chars)")
    return title, full_text


def summarise_with_ollama(title: str, full_text: str) -> str:
    """Ask qwen3.5:4b for a concise 1-2 sentence summary of the article."""
    prompt = (
        "You are a legal assistant. Below is the full text of a GDPR article.\n\n"
        f"Title: {title}\n\n"
        f"{full_text}\n\n"
        "Write a concise, plain-English summary in ONE or TWO sentences that captures "
        "what this article is about. Return ONLY the summary — no preamble, no markdown, no quotes."
    )

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        summary = data["message"]["content"].strip()
        # Clean up common LLM verbosity
        summary = re.sub(r'^["\']|["\']$', '', summary)
        summary = re.sub(r'^(Here( is|’s)|Summary:|The summary)', '', summary, flags=re.IGNORECASE).strip()
        return summary
    except Exception as e:
        print(f"    Ollama error: {e}")
        return "(summary unavailable)"


def main():
    articles = range(20, 26)  # 20–25 inclusive
    results: list[dict] = []

    for num in articles:
        print(f"\nArticle {num}:")
        scraped = scrape_article_text(num)
        if scraped is None:
            print("  Skipping — could not scrape.")
            continue
        title, full_text = scraped

        print(f"  Summarising with {MODEL} …", end=" ", flush=True)
        summary = summarise_with_ollama(title, full_text)
        print(f"OK  ({len(summary)} chars)")

        results.append({
            "article": f"Art. {num}",
            "summary": summary,
            "full_article": full_text,
        })

    out_path = HERE / "data" / "gdpr_articles_20_25.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"GDPR": results}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} articles to {out_path}")


if __name__ == "__main__":
    main()
