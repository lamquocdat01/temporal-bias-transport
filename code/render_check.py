"""Render checks on the built PDFs (checklist C1/C3/C5/C6/C8, D4.4). Usage: python code/render_check.py
  C1/C8  page count of main.pdf / supp.pdf (pymupdf), TMM limits: initial <= 10, final target 8, supplement <= 4
  C5     every rendered reference has a DOI or URL; first-author family name on the page == registry metadata
  D4.4   forbidden ranking words in main.tex (better|outperform|wins|superior|dominates|should use) -> must be 0
  C3     tone words (first/novel/guarantee/superior/state-of-the-art) in main.tex
  C6     digits typed in main.tex outside macros/labels/cites, listed for audit (design constants allowed)
Output: results/render_check.json
"""
from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pymupdf

import variant as V

REPO = Path(__file__).resolve().parents[1]
PAP = REPO / "paper"
FORBIDDEN = r"better|outperform|wins|superior|dominat|should use"
TONE = r"\bfirst\b|\bnovel\b|guarantee|superior|state-of-the-art|unprecedented"


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def refs_from_pdf(pdf):
    d = pymupdf.open(pdf)
    txt = "\n".join(p.get_text() for p in d)
    tail = txt[txt.rfind("REFERENCES"):] if "REFERENCES" in txt else txt[txt.rfind("References"):]
    parts = re.split(r"\n\[(\d+)\]\s*\n?", "\n" + tail)
    refs = {int(parts[k]): " ".join(parts[k + 1].split()) for k in range(1, len(parts) - 1, 2)}
    return d.page_count, refs


def main():
    out = {}
    bbl = (PAP / "main.bbl").read_text(encoding="utf-8", errors="ignore")
    keys = re.findall(r"\\bibitem\{([^}]+)\}", bbl)
    rc = pd.read_csv(V.RES / "ref_check.csv").set_index("key")
    for name in ("main", "supp"):
        pages, refs = refs_from_pdf(PAP / f"{name}.pdf")
        out[f"{name}_pages"] = pages
        if name == "main":
            missing = [n for n, r in refs.items() if not re.search(r"doi:|https?://|Available:", r)]
            mism = []
            for n, key in enumerate(keys, start=1):
                if key in rc.index and isinstance(rc.loc[key, "first_author"], str) and rc.loc[key, "first_author"]:
                    fam = strip_acc(rc.loc[key, "first_author"].split(",")[0].split()[-1])
                    if fam not in strip_acc(refs.get(n, "")):
                        mism.append(dict(n=n, key=key, expected=fam, rendered=refs.get(n, "")[:80]))
            out.update(n_refs_rendered=len(refs), n_refs_bbl=len(keys), refs_without_identifier=missing,
                       first_author_mismatch=mism,
                       n_arxiv=sum("arXiv" in r for r in refs.values()))
    src = (PAP / "main.tex").read_text(encoding="utf-8")
    body = src[src.index(r"\begin{document}"):]
    out["forbidden_D44"] = [m.group(0) for m in re.finditer(FORBIDDEN, body, re.I)]
    out["tone_C3"] = [m.group(0) for m in re.finditer(TONE, body, re.I)]
    out["em_dashes"] = body.count("---")
    clean = re.sub(r"\\(cite|ref|label|input|includegraphics|bibliography\w*)\{[^}]*\}", "", body)
    clean = re.sub(r"\\[A-Za-z]+", " ", clean)
    out["typed_digits_C6"] = sorted(set(re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", clean)), key=lambda s: float(s.replace(",", "")))
    out["limits"] = dict(tmm_initial_max=10, tmm_final_target=8, tmm_supp_max=4)
    (V.RES / "render_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "typed_digits_C6"}, indent=1, ensure_ascii=False))
    print("typed digits:", out["typed_digits_C6"])


if __name__ == "__main__":
    main()
