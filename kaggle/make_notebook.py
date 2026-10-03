"""Generate kaggle/s8_detect.ipynb (S8 detector run on Kaggle, CUDA f32, gated by B0.1).

The code cells can also be smoke-tested locally on CPU (no Kaggle): set S8_PKG=<pkg dir>,
S8_WORK=<scratch dir>, S8_DEVICE=cpu, S8_GATE_LIMIT=<frames per video>, S8_SKIP_PIP=1, S8_NO_S8=1.
Usage: python kaggle/make_notebook.py
"""
from pathlib import Path

import nbformat as nbf

MD_INTRO = """# Đề tài 09 — S8 detector trên Kaggle (route `cuda_fp32_kaggle`)

Theo `MEVA_detector_plan.md` (SHA-256 `1101322c…bd4be`, kiểm ở cell 3): 118 clip S8, **mọi khung, không stride**,
YOLO26s-seg (weights SHA `3da1d83e…837a4`), conf 0,25, imgsz 640, **CUDA FP32 (`half=False`)**, không lọc lớp.
Output: một JSON/clip đúng định dạng `code/meva_detect.py` (`{"meta":…, "frames":[{"frame":i,"boxes":[[x1,y1,x2,y2,conf,cls],…]}]}`)
+ `SHA256SUMS` + log phiên bản.

**Gate B0.1 hai bậc chạy TRƯỚC S8** (cell 5), pre-register ở `docs/GATE_B01_KAGGLE_PREREG.md` **v2** (SHA-256 `PREREG_SHA`, cell 1):
600 khung đã giải mã (CDnet2014 thermal/park 350 + LASIESTA I_OC_01 250) so với `detections_cpu`.
- **Bậc 1 bit-equivalent:** 600/600 khung khớp đủ hộp, Δbbox = 0 px, Δconf = 0, sự kiện (b) 41/41.
- **Bậc 2 event-equivalent:** box không ghép cặp chỉ được phép nếu conf < 0,6 (≥ 0,6 → FAIL), g(b) trùng 600/600 khung,
  41/41 sự kiện (b) trùng biên, Δbbox ≤ 1 px, |Δconf| ≤ 5e-3, không cặp box person nào có conf hai bên nằm hai phía 0,7.
  Notebook in số box không ghép mỗi bên.

Notebook in kết quả cả hai bậc. S8 chạy nếu đạt **bậc 2** (bậc 1 kéo theo bậc 2); `meta.gate_tier` = `"bit"` hoặc `"event"`.
Cả hai FAIL → notebook dừng, không chạy S8 (quay nhánh L).

Clip MEVA tải vào thư mục tạm **ngoài** `/kaggle/working` và bị xoá ngay sau khi xử lý (kể cả khi lỗi);
notebook dừng sạch nếu `/kaggle/working` vượt 18 GB (giới hạn Kaggle 20 GB).

Thiết lập Kaggle: Accelerator **GPU T4**, **Internet ON**, thêm dataset gói `kaggle_s8_pkg` (Add Input).
Chạy lại được (resume): clip đã có JSON trong `/kaggle/working/detections` (hoặc trong một output cũ gắn làm input) được bỏ qua.
"""

CELLS = [
    # 1 config
    '''# 1. Cấu hình
import os
SHARD, NSHARD = int(os.environ.get("S8_SHARD", 0)), int(os.environ.get("S8_NSHARD", 1))  # 2 phiên song song: (0,2) và (1,2)
TIME_BUDGET_H = 11.0          # dừng sạch trước giới hạn 12 h của phiên Kaggle
ROUTE = "cuda_fp32_kaggle"
DEVICE = os.environ.get("S8_DEVICE", "0")          # "0" = GPU; "cpu" chỉ để smoke test cục bộ
DEVICE = int(DEVICE) if DEVICE.isdigit() else DEVICE
GATE_LIMIT = int(os.environ["S8_GATE_LIMIT"]) if os.environ.get("S8_GATE_LIMIT") else None  # None = đủ 600 khung
CONF, IMGSZ, PERSON_CONF = 0.25, 640, 0.7
WEIGHTS_SHA = "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4"
PLAN_SHA = "1101322c0d35c8b653c902045b0f94b49a1816d26295c139be42e7850a3bd4be"
PREREG_SHA = "de51479610e3272be22f272339fd9deebb4cf5eba3f54230761fbe549eaebcf1"  # docs/GATE_B01_KAGGLE_PREREG.md v2
TIER2_MAX_DPX, TIER2_MAX_DCONF = 1, 5e-3   # bậc 2 (pre-register)
TIER2_UNMATCHED_CONF = 0.6    # bậc 2 v2: box không ghép phải có conf < 0,6
WORK_STOP_GB = 18.0           # /kaggle/working giới hạn 20 GB
ULTRALYTICS_VERSION = "8.4.7"  # docs/ENV.md
BUCKET = "mevadata-public-01"''',
    # 2 install + env
    '''# 2. Cài đặt + log phiên bản
import subprocess, sys
if not os.environ.get("S8_SKIP_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRALYTICS_VERSION}", "boto3"], check=True)
import json, time, hashlib, platform, shutil, glob
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, ultralytics
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert ultralytics.__version__ == ULTRALYTICS_VERSION, ultralytics.__version__
if DEVICE != "cpu":
    assert torch.cuda.is_available(), "cần GPU (Accelerator: GPU T4)"
ENV = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
           cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0) if DEVICE != "cpu" else "cpu",
           ultralytics=ultralytics.__version__, opencv=cv2.__version__, numpy=np.__version__,
           tf32=False, cudnn_deterministic=True, half=False, route=ROUTE)
W = Path(os.environ.get("S8_WORK", "/kaggle/working"))
DET = W / "detections"
DET.mkdir(parents=True, exist_ok=True)
(W / "env.json").write_text(json.dumps(ENV, indent=2))
print(json.dumps(ENV, indent=2))''',
    # 3 package check
    '''# 3. Gói input: kiểm SHA256SUMS, hash plan, hash weights
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()

PKG = Path(os.environ["S8_PKG"]) if os.environ.get("S8_PKG") else \\
    Path(sorted(glob.glob("/kaggle/input/**/s8_manifest.csv", recursive=True))[0]).parent
for line in (PKG / "SHA256SUMS").read_text().splitlines():
    h, name = line.split("  ", 1)
    assert sha256(PKG / name) == h, f"SHA256 lệch: {name}"
assert sha256(PKG / "MEVA_detector_plan.md") == PLAN_SHA, "plan hash lệch"
assert sha256(PKG / "yolo26s-seg.pt") == WEIGHTS_SHA, "weights hash lệch"
S8 = pd.read_csv(PKG / "s8_manifest.csv")
assert len(S8) == 118
print("package OK:", PKG, "| plan", PLAN_SHA[:8], "| weights", WEIGHTS_SHA[:8], "| S8 clips", len(S8))''',
    # 4 model
    '''# 4. Model (giống code/weights_equiv.predict: xyxy int, conf làm tròn 4 chữ số, cls int)
from ultralytics import YOLO
model = YOLO(str(PKG / "yolo26s-seg.pt"), task="segment")

def predict(frame):
    r = model.predict(frame, conf=CONF, imgsz=IMGSZ, device=DEVICE, half=False, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    d = r.boxes.data.cpu().numpy()
    return [[*(int(v) for v in row[:4]), round(float(row[4]), 4), int(row[5])] for row in d]''',
    # 5 gate
    '''# 5. Gate B0.1 hai bậc (docs/GATE_B01_KAGGLE_PREREG.md v2) — bậc 2 PHẢI PASS trước S8
def match(ref, new):
    used, pairs = set(), []
    for b in ref:
        best, bi = None, None
        for j, c in enumerate(new):
            if j in used or c[5] != b[5]:
                continue
            dist = max(abs(c[k] - b[k]) for k in range(4))
            if best is None or dist < best:
                best, bi = dist, j
        if bi is None:
            continue
        used.add(bi)
        pairs.append((b, new[bi], best))
    dpx = max((p[2] for p in pairs), default=0.0)
    dcf = max((abs(p[1][4] - p[0][4]) for p in pairs), default=0.0)
    straddle = sum(p[0][5] == 0 and (p[0][4] >= PERSON_CONF) != (p[1][4] >= PERSON_CONF) for p in pairs)
    pr, pn = {id(p[0]) for p in pairs}, {id(p[1]) for p in pairs}
    unm_ref = [b[4] for b in ref if id(b) not in pr]           # conf của box chỉ có ở tham chiếu
    unm_new = [c[4] for c in new if id(c) not in pn]           # conf của box chỉ có ở bản mới
    return len(pairs), dpx, dcf, straddle, len(pairs) == len(ref) == len(new), unm_ref, unm_new

def events(g):
    d = np.diff(np.r_[0, np.asarray(g, np.int8), 0])
    return list(zip(np.where(d == 1)[0].tolist(), (np.where(d == -1)[0] - 1).tolist()))

rows = []
for npz in sorted((PKG / "gate").glob("*.npz")):
    vid = npz.stem.replace("__", "/")
    z = np.load(npz)
    frames, idx = z["frames"], z["idx"].tolist()
    ref = {f["frame"]: f["boxes"] for f in json.load(open(npz.with_suffix(".ref.json")))["frames"]}
    assert set(ref) == set(idx), "frame index mismatch"
    for fr in frames[:3]:
        predict(fr)                                           # warm-up (như weights_equiv)
    n = len(idx) if GATE_LIMIT is None else GATE_LIMIT
    for fidx, fr in zip(idx[:n], frames[:n]):
        new, rb = predict(fr), ref.get(fidx, [])
        nm, dpx, dcf, strad, allm, ur, un = match(rb, new)
        rows.append(dict(video=vid, frame=fidx, n_ref=len(rb), n_new=len(new), all_matched=allm, max_dpx=dpx,
                         max_dconf=dcf, person_straddle_07=strad,
                         n_unm_ref=len(ur), n_unm_new=len(un),
                         max_unm_conf_ref=max(ur, default=0.0), max_unm_conf_new=max(un, default=0.0),
                         n_unm_ge06=sum(c >= TIER2_UNMATCHED_CONF for c in ur + un),
                         gb_ref=int(any(b[5] == 0 and b[4] >= PERSON_CONF for b in rb)),
                         gb_new=int(any(b[5] == 0 and b[4] >= PERSON_CONF for b in new))))
df = pd.DataFrame(rows)
per_video, ev_ref, ev_new, ev_same = {}, 0, 0, 0
for vid, x in df.groupby("video"):
    x = x.sort_values("frame")
    er = [(int(x.frame.iloc[a]), int(x.frame.iloc[b])) for a, b in events(x.gb_ref)]
    en = [(int(x.frame.iloc[a]), int(x.frame.iloc[b])) for a, b in events(x.gb_new)]
    same = len(set(er) & set(en))
    ev_ref += len(er); ev_new += len(en); ev_same += same
    per_video[vid] = dict(n_frames=len(x), frac_all_matched=float(x.all_matched.mean()), max_dpx=float(x.max_dpx.max()),
                          max_dconf=float(x.max_dconf.max()), g_b_equal=int((x.gb_ref == x.gb_new).sum()),
                          person_straddle_07=int(x.person_straddle_07.sum()),
                          n_unmatched_ref=int(x.n_unm_ref.sum()), n_unmatched_new=int(x.n_unm_new.sum()),
                          max_unmatched_conf_ref=float(x.max_unm_conf_ref.max()),
                          max_unmatched_conf_new=float(x.max_unm_conf_new.max()),
                          n_unmatched_conf_ge06=int(x.n_unm_ge06.sum()),
                          events_b_ref=len(er), events_b_new=len(en), events_b_same_bounds=same,
                          events_b_only_ref=sorted(set(er) - set(en)), events_b_only_new=sorted(set(en) - set(er)))
M = dict(n_frames=len(df), frames_all_matched=int(df.all_matched.sum()), max_dpx=float(df.max_dpx.max()),
         max_dconf=float(df.max_dconf.max()), g_b_equal_frames=int((df.gb_ref == df.gb_new).sum()),
         person_straddle_07=int(df.person_straddle_07.sum()),
         n_unmatched_ref=int(df.n_unm_ref.sum()), n_unmatched_new=int(df.n_unm_new.sum()),
         max_unmatched_conf_ref=float(df.max_unm_conf_ref.max()), max_unmatched_conf_new=float(df.max_unm_conf_new.max()),
         n_unmatched_conf_ge06=int(df.n_unm_ge06.sum()),
         events_b_ref=ev_ref, events_b_new=ev_new, events_b_same_bounds=ev_same)
T1 = {"n_frames == 600": M["n_frames"] == 600, "600/600 khung khớp đủ hộp": M["frames_all_matched"] == 600,
      "Δbbox == 0 px": M["max_dpx"] == 0, "Δconf == 0": M["max_dconf"] == 0,
      "sự kiện (b) 41/41": ev_ref == ev_new == 41}
T2 = {"n_frames == 600": M["n_frames"] == 600,
      f"0 box không ghép có conf ≥ {TIER2_UNMATCHED_CONF}": M["n_unmatched_conf_ge06"] == 0,
      "g(b) trùng 600/600 khung": M["g_b_equal_frames"] == 600,
      "41/41 sự kiện (b) trùng biên": ev_ref == ev_new == ev_same == 41,
      f"Δbbox ≤ {TIER2_MAX_DPX} px": M["max_dpx"] <= TIER2_MAX_DPX,
      f"|Δconf| ≤ {TIER2_MAX_DCONF}": M["max_dconf"] <= TIER2_MAX_DCONF,
      "0 cặp box person hai phía 0,7": M["person_straddle_07"] == 0}
T1 = {k: bool(v) for k, v in T1.items()}
T2 = {k: bool(v) for k, v in T2.items()}
PASS1, PASS2 = all(T1.values()), all(T2.values())
GATE_TIER = "bit" if PASS1 else ("event" if PASS2 else None)
gate = dict(gate="B0.1", route=ROUTE, prereg_sha256=PREREG_SHA, tier1_bit_PASS=PASS1, tier2_event_PASS=PASS2,
            gate_tier=GATE_TIER, PASS=PASS2, tier1_criteria=T1, tier2_criteria=T2, measured=M,
            per_video=per_video, env=ENV)
(W / "gate_b01.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
df.to_csv(W / "gate_b01_frames.csv", index=False)
print("=== ĐO ===")
for k, v in M.items():
    print(f"  {k}: {v}")
print(f"=== BOX KHÔNG GHÉP === chỉ-tham-chiếu: {M['n_unmatched_ref']} (conf max {M['max_unmatched_conf_ref']}) | "
      f"chỉ-mới: {M['n_unmatched_new']} (conf max {M['max_unmatched_conf_new']}) | conf ≥ {TIER2_UNMATCHED_CONF}: "
      f"{M['n_unmatched_conf_ge06']}")
for vid, v in per_video.items():
    print(f"  {vid}: chỉ-tham-chiếu {v['n_unmatched_ref']}, chỉ-mới {v['n_unmatched_new']}, conf ≥ {TIER2_UNMATCHED_CONF}: "
          f"{v['n_unmatched_conf_ge06']}")
for name, ok, T in (("BẬC 1 bit-equivalent", PASS1, T1), ("BẬC 2 event-equivalent (v2)", PASS2, T2)):
    print(f"=== {name}: {'PASS' if ok else 'FAIL'} ===")
    for k, v in T.items():
        print(f"  [{'x' if v else ' '}] {k}")
print(json.dumps(per_video, indent=2, ensure_ascii=False))
if not PASS2:
    raise RuntimeError("B0.1 FAIL cả hai bậc — DỪNG, không chạy S8 (quay nhánh L). "
                       "Gửi gate_b01.json + gate_b01_frames.csv cho Claude/anh Đạt.")
print(f"→ S8 được chạy, gate_tier = {GATE_TIER!r}")''',
    # 6 S8
    '''# 6. S8: tải từng clip từ S3 → kiểm size + SHA-256 → detect mọi khung → JSON → xoá clip
G = json.load(open(W / "gate_b01.json", encoding="utf-8"))
assert G["tier2_event_PASS"] is True and G["prereg_sha256"] == PREREG_SHA, "B0.1 chưa đạt bậc 2"
GATE_TIER = G["gate_tier"]

def du_gb(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9
if not os.environ.get("S8_NO_S8"):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 10}))
    for p in glob.glob("/kaggle/input/**/detections/*.json", recursive=True):   # resume từ output cũ
        if not (DET / Path(p).name).exists():
            shutil.copy2(p, DET / Path(p).name)
    # clip tạm NGOÀI /kaggle/working (không tính vào 20 GB output, không lưu vào version)
    TMP = Path(os.environ.get("S8_TMP", "/kaggle/temp" if Path("/kaggle/temp").is_dir() else "/tmp")) / "s8_clips"
    assert W.resolve() not in TMP.resolve().parents, "TMP không được nằm trong /kaggle/working"
    shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
    t_start = time.time()
    for i, r in enumerate(S8.itertuples(index=False)):
        if i % NSHARD != SHARD:
            continue
        stem = Path(r.key).stem
        out = DET / f"{stem}.json"
        if out.exists():
            continue
        if (time.time() - t_start) / 3600 > TIME_BUDGET_H:
            print("TIME BUDGET reached — lưu version, chạy lại để resume"); break
        if du_gb(W) > WORK_STOP_GB:
            print(f"/kaggle/working = {du_gb(W):.2f} GB > {WORK_STOP_GB} GB — dừng sạch, báo Claude"); break
        clip = TMP / Path(r.key).name
        try:
            s3.download_file(BUCKET, r.key, str(clip))
            assert clip.stat().st_size == r.size_bytes and sha256(clip) == r.sha256, f"clip lệch: {r.key}"
            t0 = time.time()
            cap = cv2.VideoCapture(str(clip))
            frames, probes, k = [], [], 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if k % 1000 == 0:
                    probes.append(hashlib.sha256(f.tobytes()).hexdigest())   # để đối chiếu giải mã với laptop
                frames.append({"frame": k, "boxes": predict(f)})
                k += 1
            cap.release()
        finally:
            clip.unlink(missing_ok=True)                      # xoá clip NGAY sau khi xử lý, kể cả khi lỗi
        assert not clip.exists()
        meta = dict(key=r.key, route=ROUTE, weights_sha256=WEIGHTS_SHA, conf=CONF, imgsz=IMGSZ, n_frames=k,
                    sec=round(time.time() - t0, 1), half=False, gpu=ENV["gpu"], torch=ENV["torch"], cuda=ENV["cuda"],
                    ultralytics=ENV["ultralytics"], opencv=ENV["opencv"], clip_sha256=r.sha256,
                    gate_tier=GATE_TIER, gate_prereg_sha256=PREREG_SHA,
                    decode_probe_every=1000, decode_probe_sha256=probes)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": meta, "frames": frames}, separators=(",", ":")))
        tmp.replace(out)
        print(f"[{i + 1}/118] {stem}: {k} fr, {k / (time.time() - t0):.1f} fps, elapsed {(time.time() - t_start) / 3600:.2f} h, "
              f"working {du_gb(W):.2f} GB, clip tạm còn {len(list(TMP.iterdir()))}", flush=True)''',
    # 7 finalize
    '''# 7. SHA-256 từng file + gói kết quả
done = sorted(DET.glob("*.json"))
(DET / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\\n" for p in done))
mine = [Path(k).stem for i, k in enumerate(S8.key) if i % NSHARD == SHARD]
print(f"shard {SHARD}/{NSHARD}: {sum((DET / f'{s}.json').exists() for s in mine)}/{len(mine)} clip; tổng có {len(done)}/118")
if not os.environ.get("S8_NO_S8"):
    shutil.make_archive(str(W / f"s8_detections_shard{SHARD}of{NSHARD}"), "zip", DET)
print(f"gate_tier = {json.load(open(W / 'gate_b01.json', encoding='utf-8'))['gate_tier']!r}; /kaggle/working = "
      f"{sum(f.stat().st_size for f in W.rglob('*') if f.is_file()) / 1e9:.2f} GB")
print("Tải về: detections/*.json, detections/SHA256SUMS, gate_b01.json, env.json (Output → Download).")''',
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(MD_INTRO)] + [nbf.v4.new_code_cell(c) for c in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = Path(__file__).with_name("s8_detect.ipynb")
    nbf.write(nb, out)
    print(out)


if __name__ == "__main__":
    main()
