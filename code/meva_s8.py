"""Detector subset S8 (docs/MEVA_detector_plan.md), metadata only.

Rule (fixed in the plan before any detector run on MEVA): for each PRIMARY camera, walk its
blocks in R5 order (block_rank in results/meva_selection_blocks.csv) and add blocks until the
camera total is >= 1.6 h; the R5 quiet block is always included (it is R5 rank 1 by step a).
Total must be >= 8 h. Writes results/meva_S8_blocks.csv.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MIN_CAM_H, MIN_TOTAL_H = 1.6, 8.0

b = pd.read_csv(REPO / "results" / "meva_selection_blocks.csv")
b = b[b.tier == "PRIMARY"].sort_values(["cam_tag", "block_rank"])
out = []
for cam, g in b.groupby("cam_tag", sort=True):
    tot, quiet = 0.0, False
    for r in g.itertuples(index=False):
        if tot >= MIN_CAM_H and quiet:
            break
        out.append(r._asdict())
        tot += r.hours
        quiet |= bool(r.quiet)
    assert quiet and tot >= MIN_CAM_H, cam
s8 = pd.DataFrame(out)
assert s8.hours.sum() >= MIN_TOTAL_H
s8.to_csv(REPO / "results" / "meva_S8_blocks.csv", index=False)
print(s8[["cam_tag", "camera", "block_id", "block_rank", "hours", "quiet", "n_clips"]].to_string(index=False))
print(f"S8: {len(s8)} blocks, {s8.n_clips.sum()} clips, {s8.hours.sum():.2f} h")
