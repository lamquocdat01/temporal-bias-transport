"""Reproduce the paper's numbers from the released profiles (no video, no detector, no benchmark corpus needed).

    python make_numbers.py

For each run (orig = blind pre-registration, corrected = Amendment 01) the verification code/e2_e5.py is run on
profiles/<run>/ + results/<run>/ copied to a temporary directory, with the registrations in spec/ (their SHA-256 is
checked against spec/*.sha256 and git history by code/meva_outcomes.py::check_lock). For the corrected run
code/numbers_tex.py then regenerates the paper macros. The script compares, key by key, the regenerated numbers
with results/orig/numbers.json, results/corrected/numbers_corrected.json and paper/numbers.tex, and exits with
status 1 on any difference.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUM = {"orig": "numbers.json", "corrected": "numbers_corrected.json"}


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[p + k] = v
    return out


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tbt_"))
    for v in NUM:
        (tmp / v).mkdir()
        for src in (ROOT / "profiles" / v, ROOT / "results" / v):
            for f in src.iterdir():
                shutil.copyfile(f, tmp / v / f.name)
    (tmp / "paper").mkdir()
    env = dict(os.environ, NINE_PREREG_DIR="spec", NINE_RES_ORIG=str(tmp / "orig"),
               NINE_MEVA_WINDOWS=str(ROOT / "profiles" / "meva_target_windows_S8.parquet"),
               NINE_BENCH_PROC=str(tmp / "no_benchmark_corpus"), NINE_PAPER_OUT=str(tmp / "paper"),
               PYTHONIOENCODING="utf-8")
    for v in NUM:
        e = dict(env, NINE_VARIANT=v, NINE_RES=str(tmp / v))
        subprocess.run([sys.executable, str(ROOT / "code" / "e2_e5.py")], env=e, check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    e = dict(env, NINE_VARIANT="corrected", NINE_RES=str(tmp / "corrected"))
    subprocess.run([sys.executable, str(ROOT / "code" / "numbers_tex.py")], env=e, check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    n_diff = 0
    for v, f in NUM.items():
        a = flat(json.loads((ROOT / "results" / v / f).read_text()))
        b = flat(json.loads((tmp / v / f).read_text()))
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        n_diff += len(diff)
        print(f"{v:9s} {f:24s} keys {len(a):4d}  differing {len(diff)}" + (f"  e.g. {diff[:5]}" if diff else ""))
    same = (ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8") == (tmp / "paper" / "numbers.tex").read_text(encoding="utf-8")
    n_diff += 0 if same else 1
    print(f"paper/numbers.tex identical: {same}")
    print("RESULT:", "0 differences" if n_diff == 0 else f"{n_diff} differences")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if n_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
