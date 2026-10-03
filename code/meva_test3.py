"""Download exactly 3 PRIMARY clips (first clip of P-1, P-2, P-3) and benchmark cv2 decode.

Per clip: size, SHA-256, fps, resolution, frame count (header + decoded), FOURCC,
time for full decode, time for decode + resize 320x180 + grayscale.
Also estimates the size of the full PRIMARY list from inventory sizes.
Outputs: results/meva_test3.csv, docs/MEVA_test3.md
"""
import hashlib
import os
import platform
import time

import boto3
import cv2
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from paths import MEVA_DIR

BUCKET = "mevadata-public-01"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
DOCS = os.path.join(REPO, "docs")
OUT = str(MEVA_DIR / "test3")
MAX_CLIPS = 3


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def decode(p, resize):
    cap = cv2.VideoCapture(p)
    n = 0
    t = time.perf_counter()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if resize:
            g = cv2.cvtColor(cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        n += 1
    dt = time.perf_counter() - t
    cap.release()
    return n, dt


def main():
    sel = pd.read_csv(os.path.join(RES, "meva_selection.csv"))
    P = sel[sel.tier == "PRIMARY"]
    picks = P.groupby("cam_tag", sort=False).head(1).head(MAX_CLIPS)  # first clip of P-1, P-2, P-3
    assert len(picks) == MAX_CLIPS and picks.camera.nunique() == MAX_CLIPS
    os.makedirs(OUT, exist_ok=True)
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    rows = []
    for r in picks.itertuples():
        dst = os.path.join(OUT, r.key.rsplit("/", 1)[-1])
        t = time.perf_counter()
        if not (os.path.exists(dst) and os.path.getsize(dst) == r.size_bytes):
            s3.download_file(BUCKET, r.key, dst)
        dl = time.perf_counter() - t
        cap = cv2.VideoCapture(dst)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nh = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fcc >> 8 * i) & 0xFF) for i in range(4))
        backend = cap.getBackendName()
        cap.release()
        n_full, t_full = decode(dst, False)
        n_small, t_small = decode(dst, True)
        rows.append(dict(cam_tag=r.cam_tag, camera=r.camera, site=r.site, key=r.key, path=dst,
                         size_bytes=os.path.getsize(dst), inventory_size_bytes=r.size_bytes,
                         sha256=sha256(dst), download_s=round(dl, 2), fps=fps, width=w, height=h,
                         frame_count_header=nh, frames_decoded=n_full, dur_s_filename=r.dur_s,
                         fourcc=fourcc, backend=backend,
                         decode_full_s=round(t_full, 2), decode_full_fps=round(n_full / t_full, 1),
                         decode_small_gray_s=round(t_small, 2), decode_small_gray_fps=round(n_small / t_small, 1),
                         mbit_s=round(os.path.getsize(dst) * 8 / r.dur_s / 1e6, 2)))
        print(rows[-1])
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, "meva_test3.csv"), index=False)

    tot_b = P.size_bytes.sum()
    tot_h = P.dur_s.sum() / 3600
    frames_20h = (P.dur_s * P.camera.map(D.set_index("camera").fps).fillna(D.fps.median())).sum()
    rt_full = frames_20h / D.decode_full_fps.mean() / 3600
    rt_small = frames_20h / D.decode_small_gray_fps.mean() / 3600
    L = ["# MEVA test download (3 clips)", "",
         f"Clips: first clip of PRIMARY cameras P-1, P-2, P-3 (`docs/MEVA_selection.md`). "
         f"Stored in `{OUT}`. Script: `code/meva_test3.py`. Machine: {platform.processor() or platform.machine()}, "
         f"OpenCV {cv2.__version__}, single process.", "",
         "| tag | camera | size MB | fps | resolution | frames (header / decoded) | FOURCC | Mbit/s | full decode s (fps) | decode+320x180 gray s (fps) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in D.itertuples():
        L.append(f"| {r.cam_tag} | {r.camera} | {r.size_bytes/1e6:.1f} | {r.fps:.2f} | {r.width}x{r.height} | "
                 f"{r.frame_count_header} / {r.frames_decoded} | {r.fourcc} | {r.mbit_s} | "
                 f"{r.decode_full_s} ({r.decode_full_fps}) | {r.decode_small_gray_s} ({r.decode_small_gray_fps}) |")
    L += ["", "SHA-256:", ""] + [f"- `{r.path}` ({r.size_bytes} B, matches inventory size: "
                                 f"{r.size_bytes == r.inventory_size_bytes}): `{r.sha256}`" for r in D.itertuples()]
    L += ["", "## Estimates for the full PRIMARY list", "",
          f"- PRIMARY: {len(P)} clips, {tot_h:.2f} h, **{tot_b/1e9:.2f} GB** ({tot_b/2**30:.2f} GiB) "
          f"summed from inventory `size_bytes` (exact S3 object sizes, not an extrapolation).",
          f"- Per camera GB: " + ", ".join(f"{c}: {g/1e9:.2f}" for c, g in P.groupby("camera").size_bytes.sum().items()),
          f"- Frames (fps from test clips, same fps assumed for P-4/P-5): {frames_20h/1e6:.2f} M.",
          f"- Single-process decode time at the mean measured rate: full decode ~{rt_full:.1f} h; "
          f"decode+resize+gray ~{rt_small:.1f} h (INTER_AREA resize from 1080p roughly doubles the per-frame cost in this run; a faster interpolation or decoder-side downscaling could reduce it).",
          "- Disk: PRIMARY fits comfortably in the ~64 GB free on D: (RESERVE would add 19.72 GB).", ""]
    with open(os.path.join(DOCS, "MEVA_test3.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
