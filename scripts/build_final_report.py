#!/usr/bin/env python3
"""Build docs/DCFI_FINAL_RESULTS.{html,pdf} with figures and mermaid rendered.

Figures are inlined as base64 data URIs so the PDF is a single self-contained
file that can be emailed without its asset folder. Mermaid fences are lifted out
before markdown conversion (python-markdown would escape them into literal text)
and Chrome is given a virtual-time budget so printing waits for the draw.
"""
import base64
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS, FIGS = ROOT / "docs", ROOT / "figures" / "dcfi"
MD, HTML, PDF = (DOCS / "DCFI_FINAL_RESULTS.md", DOCS / "DCFI_FINAL_RESULTS.html",
                 DOCS / "DCFI_FINAL_RESULTS.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"

FIGMAP = {"FIG1": "fig1_baselines.png", "FIG2": "fig2_alpha.png",
          "FIG3": "fig3_capture.png", "FIG4": "fig4_c1.png",
          "FIG5": "fig5_gate.png"}

CSS = """
@page { size: A4; margin: 15mm 13mm; }
* { box-sizing: border-box; }
body { font-family:"Georgia","Times New Roman",serif; color:#1a1a1a; line-height:1.45;
  font-size:10pt; margin:0; }
h1 { font-size:19pt; line-height:1.2; margin:0 0 4px; color:#0b3d63; }
h2 { font-size:13pt; color:#0b3d63; border-bottom:2px solid #0b3d63; padding-bottom:3px;
     margin:22px 0 9px; page-break-after:avoid; }
h3 { font-size:11pt; color:#12507f; margin:14px 0 5px; page-break-after:avoid; }
p { margin:5px 0; text-align:justify; }
.subtitle { font-size:10.5pt; color:#444; font-style:italic; margin:6px 0 2px; }
.meta { font-size:8.6pt; color:#666; margin:2px 0; }
.titleblock { border-bottom:3px solid #0b3d63; padding-bottom:10px; margin-bottom:8px; }
code { font-family:"SF Mono",Menlo,Consolas,monospace; font-size:8.6pt;
  background:#f2f5f8; padding:1px 4px; border-radius:3px; }
pre { background:#f7f9fb; border:1px solid #dde3ea; border-radius:5px; padding:9px 11px;
  font-size:8.4pt; overflow-x:auto; page-break-inside:avoid; }
pre code { background:none; padding:0; }
table { border-collapse:collapse; width:100%; margin:9px 0; font-size:8.7pt;
  page-break-inside:avoid; }
th,td { border:1px solid #cdd6df; padding:4px 7px; text-align:center; }
th { background:#0b3d63; color:#fff; font-weight:bold; }
td:first-child, th:first-child { text-align:left; }
tr:nth-child(even) td { background:#f5f8fb; }
ul,ol { margin:5px 0 5px 6px; padding-left:17px; }
li { margin:2px 0; }
.abstract { background:#f7f9fb; border:1px solid #dde3ea; border-radius:6px;
  padding:11px 14px; margin:12px 0; font-size:9.4pt; }
.abstract b { color:#0b3d63; }
.callout { background:#eef4f9; border-left:4px solid #0b3d63; padding:8px 11px;
  margin:11px 0; font-size:9.2pt; page-break-inside:avoid; }
.obs { background:#f4f8f5; border-left:4px solid #0a7d3f; padding:8px 11px; margin:11px 0;
  font-size:9.2pt; page-break-inside:avoid; }
.novelty { background:#e9f6ee; border-left:4px solid #0a7d3f; padding:9px 12px; margin:12px 0;
  font-size:9.3pt; page-break-inside:avoid; }
.warn { background:#fdf0ee; border-left:4px solid #b3261e; padding:8px 11px; margin:11px 0;
  font-size:9.2pt; page-break-inside:avoid; }
.eq { text-align:center; font-family:"SF Mono",Menlo,monospace; font-size:9.5pt;
  background:#f2f5f8; padding:7px; border-radius:4px; margin:8px 0; }
.fig { text-align:center; margin:12px 0; page-break-inside:avoid; }
.fig img { max-width:100%; height:auto; border:1px solid #e3e2de; border-radius:4px; }
.mermaid { text-align:center; margin:12px 0; page-break-inside:avoid; }
.mermaid svg { max-width:100% !important; height:auto !important; }
"""

JS = """
mermaid.initialize({ startOnLoad:true, theme:'base', themeVariables:{
  fontFamily:'Georgia, serif', fontSize:'12px', primaryColor:'#eef4f9',
  primaryTextColor:'#1a1a1a', primaryBorderColor:'#0b3d63', lineColor:'#4a5b6a',
  secondaryColor:'#e9f6ee', tertiaryColor:'#fdf6e3' },
  flowchart:{ htmlLabels:true, curve:'basis', useMaxWidth:true } });
"""


def main():
    src = MD.read_text(encoding="utf-8")

    blocks = []

    def stash(m):
        blocks.append(m.group(1))
        return f"\n@@MERMAID{len(blocks) - 1}@@\n"

    src = re.sub(r"```mermaid\n(.*?)```", stash, src, flags=re.S)
    body = markdown.markdown(
        src, extensions=["tables", "fenced_code", "attr_list", "md_in_html"])

    for i, code in enumerate(blocks):
        div = f'<pre class="mermaid">\n{code}</pre>'
        body = body.replace(f"<p>@@MERMAID{i}@@</p>", div).replace(f"@@MERMAID{i}@@", div)

    # Inline each figure so the PDF travels as one file.
    for key, name in FIGMAP.items():
        p = FIGS / name
        if not p.exists():
            raise SystemExit(f"missing figure: {p} (run scripts/make_dcfi_figures.py)")
        uri = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
        body = body.replace(f'src="{key}"', f'src="{uri}"')

    html = ('<!doctype html><html><head><meta charset="utf-8">'
            "<title>DCFI-UPA Final Results</title>"
            f"<style>{CSS}</style></head><body>{body}"
            f'<script src="{MERMAID}"></script><script>{JS}</script></body></html>')
    HTML.write_text(html, encoding="utf-8")
    print(f"wrote {HTML} ({len(html)/1024:.0f} KB, {len(blocks)} diagrams, "
          f"{len(FIGMAP)} figures)")

    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=25000", "--run-all-compositor-stages-before-draw",
         f"--print-to-pdf={PDF}", HTML.as_uri()], check=True)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
