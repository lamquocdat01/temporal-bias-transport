"""Join MEVA inventory (drops-123-r13) with Kitware activity annotation files.

Annotation repo: https://gitlab.kitware.com/meva/meva-data-repo (shallow clone at
<DATA_ROOT>/meva-data-repo). Layout:
    annotation/DIVA-phase-2/MEVA/kitware/<date>/<hh>/<clip>.activities.yml               (evaluation-level)
    annotation/DIVA-phase-2/MEVA/kitware-meva-training/<date>/<hh>/<clip>.activities.yml  (training-level)
    (+ .geom.yml / .types.yml KPF companions; nist-kf1-json/ and contrib/ are derived/3rd-party)
<clip> = date.start.end.site.camera (no ".r13").

has_activity_labels = clip has a Kitware *.activities.yml file (kitware OR kitware-meva-training).
Only the EXISTENCE of the file is used; activity contents/counts are never read.

Outputs:
    results/meva_inventory_labeled.csv   per clip: inventory + clip_id, has_activity_labels, label_source, has_nist_json
    results/meva_label_coverage.csv      per camera: n_clips, n_labeled, frac_labeled, hours, labeled_hours, date range
"""
import os
import re

import pandas as pd
from paths import MEVA_REPO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
ANN = str(MEVA_REPO / "annotation" / "DIVA-phase-2" / "MEVA")


def scan(sub, suffix):
    out = {}
    root = os.path.join(ANN, sub)
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(suffix):
                out[f[: -len(suffix)]] = os.path.relpath(os.path.join(dp, f), ANN).replace("\\", "/")
    return out


def main():
    inv = pd.read_csv(os.path.join(RES, "meva_inventory.csv"))
    inv = inv[inv.is_video].copy()
    inv["clip_id"] = inv.key.str.rsplit("/", n=1).str[-1].str.replace(r"\.r13\.avi$", "", regex=True)
    d = (pd.to_timedelta(inv.end) - pd.to_timedelta(inv.start)).dt.total_seconds().to_numpy().copy()
    d[d < 0] += 86400
    inv["dur_s"] = d

    kw = scan("kitware", ".activities.yml")
    kt = scan("kitware-meva-training", ".activities.yml")
    nj = scan("nist-kf1-json", ".activities.json")
    print(f"annotation files: kitware={len(kw)} kitware-meva-training={len(kt)} nist-kf1-json={len(nj)} "
          f"overlap kw&kt={len(set(kw) & set(kt))}")

    def src(c):
        a, b = c in kw, c in kt
        return "kitware+training" if a and b else "kitware" if a else "kitware-meva-training" if b else ""

    inv["label_source"] = inv.clip_id.map(src)
    inv["has_activity_labels"] = inv.label_source != ""
    inv["has_nist_json"] = inv.clip_id.isin(set(nj))
    all_ann = set(kw) | set(kt)
    unmatched = sorted(all_ann - set(inv.clip_id))
    print(f"annotated clip ids: {len(all_ann)}  matched to inventory: {len(all_ann) - len(unmatched)}  "
          f"unmatched: {len(unmatched)}  e.g. {unmatched[:5]}")
    inv.to_csv(os.path.join(RES, "meva_inventory_labeled.csv"), index=False)

    g = inv.groupby(["site", "camera"])
    cov = g.agg(n_clips=("key", "size"), n_labeled=("has_activity_labels", "sum"),
                hours=("dur_s", lambda s: s.sum() / 3600),
                gb=("size_bytes", lambda s: s.sum() / 1e9),
                date_first=("date", "min"), date_last=("date", "max"), n_dates=("date", "nunique")).reset_index()
    lab = inv[inv.has_activity_labels].groupby(["site", "camera"]).agg(
        labeled_hours=("dur_s", lambda s: s.sum() / 3600),
        labeled_date_first=("date", "min"), labeled_date_last=("date", "max")).reset_index()
    cov = cov.merge(lab, on=["site", "camera"], how="left")
    cov["labeled_hours"] = cov.labeled_hours.fillna(0.0)
    cov["frac_labeled"] = cov.n_labeled / cov.n_clips
    cov = cov[["site", "camera", "n_clips", "n_labeled", "frac_labeled", "hours", "labeled_hours", "gb",
               "date_first", "date_last", "n_dates", "labeled_date_first", "labeled_date_last"]]
    cov.to_csv(os.path.join(RES, "meva_label_coverage.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(cov.round(3).to_string(index=False))
    print(f"TOTAL clips={len(inv)} labeled={inv.has_activity_labels.sum()} "
          f"({inv.has_activity_labels.mean():.1%}) labeled_hours={inv[inv.has_activity_labels].dur_s.sum()/3600:.1f}")


if __name__ == "__main__":
    main()
