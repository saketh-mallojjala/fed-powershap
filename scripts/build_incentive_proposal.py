#!/usr/bin/env python3
"""Build docs/INCENTIVE_PROPOSAL.{html,pdf} from docs/INCENTIVE_PROPOSAL.md.

Reuses the print CSS from docs/FedPowerShap_Report.html so the proposal matches
the existing report styling. Markdown -> styled HTML via the `markdown` package,
then HTML -> PDF via headless Google Chrome (macOS path).
"""
import subprocess
from pathlib import Path

import markdown  # pip install markdown

DOCS = Path(__file__).resolve().parent.parent / "docs"
MD = DOCS / "INCENTIVE_PROPOSAL.md"
HTML = DOCS / "INCENTIVE_PROPOSAL.html"
PDF = DOCS / "INCENTIVE_PROPOSAL.pdf"

# Print CSS lifted verbatim from docs/FedPowerShap_Report.html
CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; }
body { font-family: "Georgia","Times New Roman",serif; color:#1a1a1a; line-height:1.5;
  font-size:10.5pt; max-width:none; margin:0; }
h1 { font-size:20pt; line-height:1.2; margin:0 0 4px; color:#0b3d63; }
h2 { font-size:14pt; color:#0b3d63; border-bottom:2px solid #0b3d63; padding-bottom:3px;
     margin:26px 0 10px; page-break-after:avoid; }
h3 { font-size:11.5pt; color:#12507f; margin:16px 0 6px; page-break-after:avoid; }
h4 { font-size:10.5pt; color:#333; margin:12px 0 4px; font-style:italic; }
p { margin:6px 0; text-align:justify; }
.subtitle { font-size:11pt; color:#444; font-style:italic; margin:6px 0 2px; }
.meta { font-size:9pt; color:#666; margin:2px 0; }
.titleblock { border-bottom:3px solid #0b3d63; padding-bottom:12px; margin-bottom:8px; }
code, .mono { font-family:"SF Mono",Menlo,Consolas,monospace; font-size:9pt;
  background:#f2f5f8; padding:1px 4px; border-radius:3px; }
table { border-collapse:collapse; width:100%; margin:10px 0; font-size:9.3pt;
  page-break-inside:avoid; }
th,td { border:1px solid #cdd6df; padding:5px 8px; text-align:center; }
th { background:#0b3d63; color:#fff; font-weight:bold; }
td:first-child, th:first-child { text-align:left; }
tr:nth-child(even) td { background:#f5f8fb; }
.win { color:#0a7d3f; font-weight:bold; }
.lose { color:#b3261e; font-weight:bold; }
ul,ol { margin:6px 0 6px 6px; padding-left:18px; }
li { margin:3px 0; }
.abstract { background:#f7f9fb; border:1px solid #dde3ea; border-radius:6px;
  padding:12px 16px; margin:14px 0; font-size:10pt; }
.abstract b { color:#0b3d63; }
.callout { background:#eef4f9; border-left:4px solid #0b3d63; padding:8px 12px;
  margin:12px 0; font-size:9.8pt; }
.novelty { background:#e9f6ee; border-left:4px solid #0a7d3f; padding:8px 12px; margin:12px 0; }
.warn { background:#fdf0ee; border-left:4px solid #b3261e; padding:8px 12px; margin:12px 0; }
.eq { text-align:center; font-family:"SF Mono",Menlo,monospace; font-size:10pt;
  background:#f2f5f8; padding:8px; border-radius:4px; margin:8px 0; }
"""

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main():
    body = markdown.markdown(
        MD.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "attr_list", "md_in_html"],
    )
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>FedShapIn Proposal</title><style>" + CSS + "</style></head><body>"
        + body + "</body></html>"
    )
    HTML.write_text(html, encoding="utf-8")
    print(f"wrote {HTML} ({len(html)} bytes)")

    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True,
    )
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
