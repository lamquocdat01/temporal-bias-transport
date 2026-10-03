"""Single place for raw-data and external locations (decision 28-09-2026: datasets live outside the Google Drive
sync area). Every location can be overridden by an environment variable; a machine-specific `paths_local.py`
(not part of the release) may define LOCAL = {name: path}.

    DATA_ROOT   $THS_DATASETS (default <DATA_ROOT>/ on the author's machine)
    MEVA_DIR    DATA_ROOT/MEVA_20h        (clips/, detections/, cue/, labels/, virat_gt/)
    MEVA_REPO   DATA_ROOT/meva-data-repo  (Kitware annotation/metadata clone)
    BMC_DIR     DATA_ROOT/BMC
    VIRAT_DIR   DATA_ROOT/VIRAT/videos    (VIRAT mp4)
    BENCH_PROC  $NINE_BENCH_PROC  benchmark corpus tables (frame_scores*.parquet, events_durations*.csv)
    SCM_DIR     $NINE_SCM_DIR     scm-scene-complexity checkout (frame iterator, configs/paths.yaml, data/detections_cpu)
    WEIGHTS     $NINE_WEIGHTS     yolo26s-seg.pt (SHA-256 3da1d83e...837a4)
    OV_DIR      $NINE_OV_DIR      exported OpenVINO IR
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from paths_local import LOCAL as _LOCAL
except ImportError:
    _LOCAL = {}


def _p(env: str, key: str, default: Path) -> Path:
    return Path(os.environ.get(env) or _LOCAL.get(key) or default)


DATA_ROOT = _p("THS_DATASETS", "DATA_ROOT", Path("<DATA_ROOT>/"))
MEVA_DIR = DATA_ROOT / "MEVA_20h"
MEVA_REPO = DATA_ROOT / "meva-data-repo"
BMC_DIR = DATA_ROOT / "BMC"
VIRAT_DIR = DATA_ROOT / "VIRAT" / "videos"
BENCH_PROC = _p("NINE_BENCH_PROC", "BENCH_PROC", DATA_ROOT / "_derived" / "09" / "benchmark_corpus")
SCM_DIR = _p("NINE_SCM_DIR", "SCM_DIR", DATA_ROOT / "_derived" / "09" / "scm-scene-complexity")
WEIGHTS = _p("NINE_WEIGHTS", "WEIGHTS", DATA_ROOT / "_models" / "yolo26s-seg.pt")
OV_DIR = _p("NINE_OV_DIR", "OV_DIR", DATA_ROOT / "_models" / "09_openvino")
