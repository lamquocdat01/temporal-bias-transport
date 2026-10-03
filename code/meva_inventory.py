"""MEVA inventory: list all keys under s3://mevadata-public-01/drops-123-r13/ (anonymous)
and parse clip filenames.

Filename format: 2018-03-07.16-50-00.16-55-00.bus.G331.r13.avi
    -> date, start, end, site, camera

Outputs:
    results/meva_inventory.csv            key,date,start,end,site,camera,size_bytes,is_video,ext
    results/meva_inventory_summary.csv    per site/camera totals (clips, GB, hours)
"""
import os
import re
import sys

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "mevadata-public-01"
PREFIX = "drops-123-r13/"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")

FNAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\.(?P<start>\d{2}-\d{2}-\d{2})\.(?P<end>\d{2}-\d{2}-\d{2})"
    r"\.(?P<site>[A-Za-z0-9\-]+)\.(?P<camera>G\d+)\.(?P<rel>r\d+)\.(?P<ext>[A-Za-z0-9]+)$"
)
VIDEO_EXT = {"avi", "mp4", "mkv", "mov"}


def list_keys():
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    pag = s3.get_paginator("list_objects_v2")
    rows = []
    for page in pag.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for o in page.get("Contents", []):
            rows.append((o["Key"], int(o["Size"])))
    return rows


def parse(key):
    base = key.rsplit("/", 1)[-1]
    m = FNAME_RE.match(base)
    if not m:
        ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
        return dict(date=None, start=None, end=None, site=None, camera=None, ext=ext, parsed=False)
    d = m.groupdict()
    return dict(date=d["date"], start=d["start"].replace("-", ":"), end=d["end"].replace("-", ":"),
                site=d["site"], camera=d["camera"], ext=d["ext"].lower(), parsed=True)


def main():
    os.makedirs(RES, exist_ok=True)
    rows = list_keys()
    recs = []
    for k, sz in rows:
        p = parse(k)
        recs.append(dict(key=k, date=p["date"], start=p["start"], end=p["end"], site=p["site"],
                         camera=p["camera"], size_bytes=sz, ext=p["ext"],
                         is_video=bool(p["parsed"] and p["ext"] in VIDEO_EXT)))
    df = pd.DataFrame(recs).sort_values("key").reset_index(drop=True)
    df.to_csv(os.path.join(RES, "meva_inventory.csv"), index=False)

    v = df[df.is_video].copy()
    t0 = pd.to_timedelta(v.start)
    t1 = pd.to_timedelta(v.end)
    dur = (t1 - t0).dt.total_seconds().to_numpy().copy()
    dur[dur < 0] += 86400  # clip crossing midnight
    v["dur_s"] = dur
    summ = (v.groupby(["site", "camera"])
             .agg(n_clips=("key", "size"), gb=("size_bytes", lambda s: s.sum() / 1e9),
                  hours=("dur_s", lambda s: s.sum() / 3600), n_dates=("date", "nunique"),
                  first_date=("date", "min"), last_date=("date", "max"))
             .reset_index())
    summ.to_csv(os.path.join(RES, "meva_inventory_summary.csv"), index=False)

    print(f"keys total: {len(df)}  video clips: {len(v)}  non-video/unparsed: {(~df.is_video).sum()}")
    print(f"video GB: {v.size_bytes.sum()/1e9:.1f}  hours: {v.dur_s.sum()/3600:.1f}")
    print("non-video ext counts:", df[~df.is_video].ext.value_counts().to_dict())
    print(v.groupby("site").agg(n=("key", "size"), gb=("size_bytes", lambda s: s.sum() / 1e9),
                                hours=("dur_s", lambda s: s.sum() / 3600)).round(1).to_string())
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(summ.round(2).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
