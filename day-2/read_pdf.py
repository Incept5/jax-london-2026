import re
import sys
from pathlib import Path

import fitz          # PyMuPDF: turns a PDF page into an image
import ollama

HERE = Path(__file__).parent
PDF = HERE / "data" / "housing-new-york.pdf"   # "Housing New York: A Five-Borough, Ten-Year Plan" (City of New York, 2014)
PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 23   # page 23: a bar chart of housing permits and completions
MODEL = "qwen3.5:4b"   # vision-capable; try qwen3.6 or gemma4:26b for sharper reading
TRANSCRIBE = ("Transcribe this page as Markdown. Keep headings, paragraphs and lists. Turn any table into a "
              "Markdown table, and describe any chart or picture in one short paragraph of text (never an image link). "
              "Output only the Markdown.")

# 1. Render just one page: reading all 117 pages would take minutes
page = fitz.open(PDF)[PAGE - 1]
png = page.get_pixmap(dpi=150).tobytes("png")


def ask(prompt):
    reply = ollama.chat(model=MODEL, think=False, options={"temperature": 0, "num_ctx": 8192},
                        messages=[{"role": "user", "content": prompt, "images": [png]}])
    return reply.message.content.strip().removeprefix("```markdown").removesuffix("```").strip()


# 2. The page as Markdown, saved next to this script
markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)\n*", "", ask(TRANSCRIBE))   # small models like to invent image links
out = HERE / f"housing-page-{PAGE}.md"
out.write_text(markdown + "\n")
print(markdown, f"\n\n(saved to {out.name})\n")

# 3. Ask questions about the chart or pictures on the page (Enter on its own to finish)
while question := input("Ask about this page (Enter to finish): ").strip():
    print(ask(f"Look at the chart or picture on this page. {question} Answer briefly."), "\n")
