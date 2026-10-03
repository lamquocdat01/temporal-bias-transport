"""paper/references.bib from registry metadata (C4/C5): every DOI is resolved by content negotiation at doi.org
(Crossref / DataCite for arXiv 10.48550/arXiv.<id>); fetched title + first author are logged to results/ref_check.csv
for verification. Entries without a DOI carry exactly one URL (official proceedings / software page).
Usage: python code/make_bib.py
"""
from __future__ import annotations
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

DOIS = {
    # dataset bias / shift
    "torralba2011": "10.1109/CVPR.2011.5995347", "beery2018": "10.1007/978-3-030-01270-0_28",
    "liu2025decade": "10.48550/arXiv.2403.08632", "recht2019": "10.48550/arXiv.1902.10811",
    "koh2021wilds": "10.48550/arXiv.2012.07421",
    "pierard2025": "10.1109/ICMEW68306.2025.11152102", "garg2022": "10.48550/arXiv.2201.04234",
    "miller2021": "10.48550/arXiv.2107.04649",
    "godau2023": "10.1007/978-3-031-43898-1_38", "godau2025": "10.1016/j.media.2025.103504",
    "shimodaira2000": "10.1016/S0378-3758(00)00115-4", "saerens2002": "10.1162/089976602753284446",
    "lipton2018": "10.48550/arXiv.1802.03916",
    # video benchmark audits
    "song2026": "10.48550/arXiv.2608.21854", "zakharova2026": "10.48550/arXiv.2607.04372",
    "liu2025vad": "10.48550/arXiv.2505.19022", "alwassel2018": "10.1007/978-3-030-01219-9_16",
    "sigurdsson2017": "10.1109/ICCV.2017.235",
    # TMM precedents
    "liang2025": "10.1109/TMM.2025.3543031", "luo2026": "10.1109/TMM.2026.3705139",
    "yang2024": "10.1109/TMM.2023.3276523", "huang2026": "10.1109/TMM.2026.3660108",
    "gao2024": "10.1109/TMM.2024.3385678",
    # adaptive video analytics
    "reducto2020": "10.1145/3387514.3405874", "chameleon2018": "10.1145/3230543.3230574",
    "noscope2017": "10.14778/3137628.3137664", "glimpse2015": "10.1145/2809695.2809711",
    "awstream2018": "10.1145/3230543.3230554", "xiao2021": "10.48550/arXiv.2105.08694",
    "dehghani2022": "10.48550/arXiv.2110.12894",
    # statistics
    "pearl2014": "10.1214/14-STS486", "manski2003": "10.1007/b97478", "kaplan1958": "10.1080/01621459.1958.10501452",
    "kitagawa1955": "10.1080/01621459.1955.10501299", "oaxaca1973": "10.2307/2525981",
    "holt1979": "10.2307/2344652", "charnes1962": "10.1002/nav.3800090303", "efron1979": "10.1214/aos/1176344552",
    "field2007": "10.1111/j.1467-9868.2007.00593.x", "cameron2011multiway": "10.1198/jbes.2010.07136",
    "owen2007pigeonhole": "10.1214/07-AOAS122", "bickel1981bootstrap": "10.1214/aos/1176345637",
    "vandervaart1998": "10.1017/CBO9780511802256", "imbens2004": "10.1111/j.1468-0262.2004.00555.x",
    "vardi1982": "10.1214/aos/1176345802", "nosek2018": "10.1073/pnas.1708274114",
    "menzel2021": "10.3982/ECTA15383",
    # datasets
    "cdnet2014": "10.1109/CVPRW.2014.126", "lasiesta2016": "10.1016/j.cviu.2016.08.005",
    "bmc2012": "10.1007/978-3-642-37410-4_25", "virat2011": "10.1109/CVPR.2011.5995586",
    "meva2021": "10.1109/WACV48630.2021.00110",
}
OVERRIDE = {"manski2003": {"title": "Partial Identification of Probability Distributions",
                            "author": "Manski, Charles F."}}
URL_ENTRIES = r"""
@inproceedings{mandoline2021,
  author = {Chen, Mayee and Goel, Karan and Sohoni, Nimit S. and Poms, Fait and Fatahalian, Kayvon and R{\'e}, Christopher},
  title = {Mandoline: Model Evaluation under Distribution Shift},
  booktitle = {Proc. 38th Int. Conf. Machine Learning (ICML)}, series = {PMLR}, volume = {139}, pages = {1617--1629},
  year = {2021}, url = {https://proceedings.mlr.press/v139/chen21i.html}
}
@inproceedings{focus2018,
  author = {Hsieh, Kevin and Ananthanarayanan, Ganesh and Bodik, Peter and Venkataraman, Shivaram and Bahl, Paramvir and Philipose, Matthai and Gibbons, Phillip B. and Mutlu, Onur},
  title = {Focus: Querying Large Video Datasets with Low Latency and Low Cost},
  booktitle = {Proc. 13th USENIX Symp. Operating Systems Design and Implementation (OSDI)}, pages = {269--286}, year = {2018},
  url = {https://www.usenix.org/conference/osdi18/presentation/hsieh}
}
@inproceedings{videostorm2017,
  author = {Zhang, Haoyu and Ananthanarayanan, Ganesh and Bodik, Peter and Philipose, Matthai and Bahl, Paramvir and Freedman, Michael J.},
  title = {Live Video Analytics at Scale with Approximation and Delay-Tolerance},
  booktitle = {Proc. 14th USENIX Symp. Networked Systems Design and Implementation (NSDI)}, pages = {377--392}, year = {2017},
  url = {https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/zhang}
}
@misc{ultralytics,
  author = {Jocher, Glenn and Qiu, Jing},
  title = {Ultralytics {YOLO} (version 8.4.7)}, year = {2026}, howpublished = {Software},
  url = {https://github.com/ultralytics/ultralytics}
}
@misc{mevadata,
  author = {{Kitware Inc. and IARPA}},
  title = {{MEVA}: Multiview Extended Video with Activities -- data and annotations ({CC BY} 4.0)}, year = {2021},
  howpublished = {Dataset}, url = {https://mevadata.org/}
}
"""


def fetch(doi: str) -> str:
    req = urllib.request.Request(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex; charset=utf-8",
                                                                     "User-Agent": "topic09-bib/1.0 (mailto:noreply@example.org)"})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 + 3 * i)
    raise RuntimeError(f"{doi}: {err}")


def field(b, name):
    m = re.search(rf"\b{name}\s*=\s*[{{\"](.+?)[}}\"]\s*(,|\n|\}})", b, re.I | re.S)
    return m.group(1).strip() if m else ""


def protect(title: str) -> str:
    """Brace words BibTeX styles would lowercase wrongly: acronyms and mixed-case names (CDnet, LASIESTA, YOLO)."""
    return re.sub(r"(?<![{\\])\b([A-Za-z]*[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9-]*|[A-Z]{2,})\b",
                  lambda m: "{" + m.group(1) + "}", title)


def clean(b: str, key: str, doi: str) -> str:
    b = b.strip()
    b = re.sub(r"^@(\w+)\{[^,]*,", lambda m: f"@{m.group(1).lower()}{{{key},", b, count=1)
    b = re.sub(r"\burl\s*=\s*\{[^}]*\},?\s*", "", b, flags=re.I)                 # DOI only, printed once (C5)
    b = re.sub(r"\b(copyright|keywords|ISSN|ISBN|publisher)\s*=\s*\{(?:[^{}]|\{[^{}]*\})*\},?\s*", "", b, flags=re.I) \
        if "arxiv" in doi.lower() else b
    b = re.sub(r"\bmonth\s*=\s*(\w+)", lambda m: f"month = {m.group(1).lower()[:3]}", b)
    b = re.sub(r"\bDOI\s*=\s*\{[^}]*\}", f"doi = {{{doi}}}", b, flags=re.I)
    if "arxiv" in doi.lower() and not re.search(r"\bnote\s*=", b):
        b = b.rstrip("}").rstrip().rstrip(",") + f",\n  note = {{arXiv preprint arXiv:{doi.split('arXiv.')[-1]}}}\n}}"
    b = re.sub(r"<[^>]+>", "", b)                                   # JATS/HTML markup in titles
    if "arxiv" not in doi.lower():
        b = re.sub(r"\s+", " ", b)
    for a, c in (("&amp;", "&"), ("‐", "-"), ("‑", "-"), ("’", "'"), ("\xa0", " "),
                 ("–", "--"), ("—", "---")):
        b = b.replace(a, c)
    b = re.sub(r"(?<!\\)&", lambda m: "\\&", b)
    b = re.sub(r"(title=\{[^}]*?)\*\}", lambda m: m.group(1) + "}", b)   # 'Rates*' footnote marker
    b = re.sub(r"(\btitle\s*=\s*\{)([^}]*)(\})", lambda m: m.group(1) + protect(m.group(2)) + m.group(3), b,
               count=1, flags=re.I)
    for k, v in OVERRIDE.get(key, {}).items():
        if not re.search(rf"\b{k}\s*=", b):
            b = b.rstrip().rstrip("}").rstrip().rstrip(",") + f", {k}={{{v}}} }}"
    return b


def main():
    out, log = [], []
    for key, doi in DOIS.items():
        b = fetch(doi)
        title, author = field(b, "title"), field(b, "author")
        log.append(dict(key=key, doi=doi, fetched_title=title, first_author=author.split(" and ")[0],
                        year=field(b, "year")))
        out.append(clean(b, key, doi))
        time.sleep(0.4)
    (REPO / "paper" / "references.bib").write_text(
        "% generated by code/make_bib.py from doi.org content negotiation -- do not edit by hand\n\n"
        + "\n\n".join(out) + "\n" + URL_ENTRIES, encoding="utf-8", newline="\n")
    pd.DataFrame(log).to_csv(REPO / "results" / "ref_check.csv", index=False)
    print(pd.DataFrame(log)[["key", "first_author", "year", "fetched_title"]].to_string())


if __name__ == "__main__":
    main()
