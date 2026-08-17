#!/usr/bin/env python3
"""Build docs/METHODOLOGY.{html,pdf} with the mermaid diagrams RENDERED.

The other build scripts pipe markdown straight through python-markdown, which
turns a ```mermaid fence into an escaped <pre><code> block -- the diagram source
would print as literal text. So here the mermaid blocks are pulled out before
conversion, kept verbatim, and re-inserted as <pre class="mermaid"> afterwards.
Headless Chrome then runs mermaid.js to draw them, with a virtual-time budget so
printing waits for the SVGs to exist rather than racing them.
"""
import re
import subprocess
import sys
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent.parent / "docs"
MD = DOCS / "METHODOLOGY.md"
HTML = DOCS / "METHODOLOGY.html"
PDF = DOCS / "METHODOLOGY.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"

CSS = """
@page { size: A4; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body { font-family:"Georgia","Times New Roman",serif; color:#1a1a1a; line-height:1.5;
  font-size:10.5pt; margin:0; }
h1 { font-size:19pt; color:#0b3d63; margin:0 0 6px; border-bottom:3px solid #0b3d63;
     padding-bottom:8px; }
h2 { font-size:13.5pt; color:#0b3d63; border-bottom:2px solid #0b3d63; padding-bottom:3px;
     margin:22px 0 10px; page-break-after:avoid; }
h3 { font-size:11.5pt; color:#12507f; margin:14px 0 6px; page-break-after:avoid; }
p { margin:6px 0; text-align:justify; }
code { font-family:"SF Mono",Menlo,Consolas,monospace; font-size:9pt;
  background:#f2f5f8; padding:1px 4px; border-radius:3px; }
table { border-collapse:collapse; width:100%; margin:10px 0; font-size:9.2pt;
  page-break-inside:avoid; }
th,td { border:1px solid #cdd6df; padding:5px 8px; text-align:center; }
th { background:#0b3d63; color:#fff; }
td:first-child, th:first-child { text-align:left; }
tr:nth-child(even) td { background:#f5f8fb; }
ul,ol { margin:6px 0 6px 6px; padding-left:18px; }
li { margin:3px 0; }
hr { border:none; border-top:1px solid #cdd6df; margin:18px 0; }
blockquote { border-left:4px solid #0b3d63; background:#eef4f9; margin:12px 0;
  padding:8px 12px; }
/* Each diagram gets its own page block so it is never split mid-figure. */
.mermaid { text-align:center; margin:14px 0; page-break-inside:avoid; }
.mermaid svg { max-width:100% !important; height:auto !important; }
"""

JS = """
mermaid.initialize({
  startOnLoad: true,
  theme: 'base',
  themeVariables: {
    fontFamily: 'Georgia, serif', fontSize: '13px',
    primaryColor: '#eef4f9', primaryTextColor: '#1a1a1a',
    primaryBorderColor: '#0b3d63', lineColor: '#4a5b6a',
    secondaryColor: '#e9f6ee', tertiaryColor: '#fdf6e3'
  },
  flowchart: { htmlLabels: true, curve: 'basis', useMaxWidth: true }
});
"""


def main():
    src = MD.read_text(encoding="utf-8")

    # 1. Lift the mermaid fences out so python-markdown cannot escape them.
    blocks = []

    def stash(m):
        blocks.append(m.group(1))
        return f"\n@@MERMAID{len(blocks) - 1}@@\n"

    src = re.sub(r"```mermaid\n(.*?)```", stash, src, flags=re.S)

    body = markdown.markdown(
        src, extensions=["tables", "fenced_code", "attr_list", "md_in_html"]
    )

    # 2. Put them back as elements mermaid.js will pick up. The placeholder may
    #    have been wrapped in <p> by the converter, so match either form.
    for i, code in enumerate(blocks):
        div = f'<pre class="mermaid">\n{code}</pre>'
        body = body.replace(f"<p>@@MERMAID{i}@@</p>", div).replace(f"@@MERMAID{i}@@", div)

    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>DCFI-UPA Methodology</title>"
        f"<style>{CSS}</style></head><body>{body}"
        f'<script src="{MERMAID}"></script><script>{JS}</script>'
        "</body></html>"
    )
    HTML.write_text(html, encoding="utf-8")
    print(f"wrote {HTML} ({len(html)} bytes, {len(blocks)} diagrams)")

    # 3. Print. --virtual-time-budget lets mermaid finish drawing before capture;
    #    without it Chrome prints the page before any SVG exists.
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=20000", "--run-all-compositor-stages-before-draw",
         f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True,
    )
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
