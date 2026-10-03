# DETECTOR_PROVENANCE — detector nào sinh g_t của corpus 121 video (CP0, 2026-09-25)

## Kết luận
g_t trong `frame_scores.parquet` được sinh bởi **YOLO26s-seg (Ultralytics), CPU FP32, conf = 0,25, imgsz = 640, không lọc lớp** (80 lớp COCO; g_t = 1 khi khung có ≥1 box bất kỳ lớp).
Dự kiến trong PLAN §1 được **xác nhận**. Các tài liệu ghi "YOLOv3" là **mâu thuẫn** (xem bên dưới).

## Chuỗi bằng chứng (chỉ đọc, không sửa nguồn)
| # | Bằng chứng | Vị trí | Nội dung |
|---|---|---|---|
| 1 | Cấu hình | `<SCM_DIR>\configs\paths.yaml` | `yolo_model_path: .../ASMAG_Project_Code/models/yolo/yolo26s-seg.pt`, `p1_conf_thr: 0.25`, `p1_imgsz: 640` |
| 2 | Script sinh detections | `…\scm\dump_p1_detections.py` | `ultralytics.YOLO(model_path).predict(frame, conf, imgsz)`; ghi `{"frames":[{"frame","boxes":[[x1,y1,x2,y2,score,cls]]}]}` |
| 3 | Log chạy thật | `…\logs\dump_cdnet_cpu.log`, `dump_cdnet_cpu_fix.log`, `dump_lasiesta_bmc_cpu.log` | `[CONFIG] conf=0.25 imgsz=640 device=cpu` · `[DETECTOR] CPU FP32: …/yolo26s-seg.pt` |
| 4 | Dữ liệu detections | `…\data\detections_cpu\{CDnet2014,LASIESTA,BMC}` (53/48/20 JSON) | 490.411 box, conf nhỏ nhất = 0,25 đúng ngưỡng; 59 lớp COCO xuất hiện (nhiều nhất: car 272k, person 107k) |
| 5 | Cách tạo g_t | `03.Final Submission\…\certifiable-frame-skipping\scripts\dump_gate_scores.py` (dòng 42, 59) | `has_detection = len(boxes) > 0` → `g_t = 1 if has_detection`; `configs\default.yaml` trỏ `detections_root` tới `…/scm-scene-complexity/data/detections_cpu` (đường dẫn cũ trước khi dời vào `zz\`) |
| 6 | Kiểm độc lập trong CP0 | `code\e1_profile.py` | Dựng lại has-box từ detections_cpu cho **toàn bộ 166.589 khung** → **0 khung lệch** với g_t; các run g_t=1 tái tạo **đúng** 3.713 thời lượng trong `events_durations.csv` |
| 7 | VIRAT-P | `scripts\viratp_gatepass.py` + `data\detections_cpu_virat_full` | g_t = person (cls 0) ∧ conf ≥ 0,7; dựng lại từ JSON → **0 khung lệch** trên 41.823 khung, 1.119 sự kiện |

## Trọng số (weights)
- Đường dẫn trong log `ASMAG_Project_Code\models\yolo\yolo26s-seg.pt` **hiện không còn file .pt** (chỉ còn `yolo26s-seg.onnx`, SHA-256 `73af278f…9529480`).
- Bản .pt tìm thấy: `<WEIGHTS_DIR>/yolo26s-seg.pt` (23.467.933 byte, 25-04-2026, SHA-256 `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4`). Backup này có trước lần dump (log tháng 6) nên **nhiều khả năng** là cùng file, nhưng log không ghi hash → **không chứng minh được byte-identical**. Đề xuất: dùng bản backup này cho MEVA và kiểm tương đương bằng cách chạy lại trên 1–2 video benchmark ngắn (so box/g_t với detections_cpu) ở Đợt 2 trước khi chạy MEVA.

## Mâu thuẫn "YOLOv3"
| Tài liệu | Câu | Đánh giá |
|---|---|---|
| `content-aware-frameskip-evaluation\analysis\protocol_dependence\n1_crossdata.py` dòng 27 | "This is the SAME pipeline (YOLOv3/COCO always-on detector + ASMAGPlusGate motion score)" | **Sai** với bằng chứng 1–6 |
| `03.Final Submission\09.05 …\Revision_IoTJ\manuscript\Deep3_manuscript.tex` dòng 327 (và các bản tex khác của Deep3) | "Events come from always-on detections of a single-stage detector~\cite{yolov3} trained on COCO" | Trích dẫn YOLOv3 cho detector sinh sự kiện **không khớp** với detector thực tế (YOLO26s-seg). Dòng 329 nói YOLOv3-class 416² chỉ dùng cho đo năng lượng Orin — đó là chuyện khác. |
| `Revision_IoTJ\C1_orin\*` | YOLOv3-class dùng cho phép đo phần cứng | Không mâu thuẫn (vai trò đo năng lượng, không sinh g_t) |

**Hệ quả cho #09:** bài #09 phải ghi đúng "YOLO26s-seg, conf 0,25, imgsz 640" và (khai B9) nêu rằng đây là cùng detections dùng trong Deep3. Mâu thuẫn trong bản thảo Deep3 là việc của Deep3 — **báo anh Đạt, #09 không sửa** file nào ngoài phạm vi.

## MEVA S8 — route (cập nhật 28-09-2026, nhánh K của anh Đạt)
- **Route S8 = Kaggle CUDA FP32** (`meta.route = "cuda_fp32_kaggle"`, GPU T4, `half=False`, TF32 tắt, cuDNN deterministic), cùng weights backup SHA `3da1d83e…837a4`, conf 0,25, imgsz 640, mọi khung, không stride, không lọc lớp, ultralytics 8.4.7 (như `docs/ENV.md`). Notebook `kaggle/s8_detect.ipynb` (sinh bởi `kaggle/make_notebook.py`), gói input `kaggle/build_gate_pkg.py` → `<DATA_ROOT>/_derived\09\kaggle_s8_pkg` (116,3 MB, SHA256SUMS kèm). Notebook kiểm hash plan `1101322c…` và weights trước khi chạy.
- **Trạng thái: CHỜ gate B0.1 hai bậc** trên Kaggle, pre-register `docs/GATE_B01_KAGGLE_PREREG.md` **v2** (SHA-256 `de514796…eaebcf1`; v1 `7556b064…3f0911` commit `82c3637`; cả hai commit 29-09-2026 trước lần chạy đầu, chưa có kết quả GPU nào). 600 khung (CDnet2014 thermal/park 350 + LASIESTA I_OC_01 250) so với `detections_cpu`. **Bậc 1 bit-equivalent:** 600/600 khớp đủ hộp, Δbbox 0 px, Δconf 0, sự kiện (b) 41/41 (tiêu chí B0.1 torch_cpu cũ). **Bậc 2 event-equivalent (v2):** box không ghép cặp được phép nếu conf < 0,6 (≥ 0,6 → FAIL; in số box không ghép mỗi bên), g(b) trùng 600/600 khung, 41/41 sự kiện (b) trùng biên, Δbbox ≤ 1 px, |Δconf| ≤ 5e-3, 0 cặp box person có conf hai bên nằm hai phía 0,7. Notebook in cả hai bậc; S8 chạy nếu đạt bậc 2, `meta.gate_tier` = `bit`/`event` ghi trong mọi JSON. Cả hai FAIL → không chạy S8, quay nhánh L. Khung gate đóng gói ở dạng đã giải mã (npz, không mất dữ liệu) để loại khác biệt bộ giải mã ảnh.
- Đây là thay đổi route so với `MEVA_detector_plan.md` §2–3 (route OpenVINO CPU/Iris Xe trên laptop); tập S8, weights, tham số và định dạng không đổi. Mỗi JSON ghi thêm phiên bản torch/CUDA/GPU/ultralytics/OpenCV, SHA-256 clip, và `decode_probe_sha256` (SHA-256 khung 0, 1000, 2000…) để đối chiếu giải mã với laptop; số khung mỗi clip phải bằng `n_decoded` của cue laptop.
- **6 clip đã chạy trên laptop** (route `ov_gpu_fp32x`, P-1 G639 2018-03-13 17:05–17:35) sẽ được **thay bằng bản Kaggle** để toàn bộ S8 một route. Khi bản Kaggle về, 6 JSON laptop được chuyển sang `MEVA_20h\detections_laptop_ov_gpu_fp32x\` (giữ để so chéo route, không xoá).

## MEVA S8 — route CHỐT: `cuda_fp32_kaggle` (30-09-2026)
- **Gate B0.1 (Kaggle, prereg v2 `de514796…eaebcf1`), 1 phiên/1 shard:** **bậc 2 "event" PASS**, bậc 1 "bit" FAIL chỉ vì max |Δconf| = 1e-4 (một bước làm tròn 4 chữ số). Đo: 600/600 khung khớp đủ hộp, max Δbbox 0 px, max |Δconf| 1e-4, g(b) trùng 600/600, sự kiện (b) 41/41 trùng biên, 0 cặp person hai phía 0,7, **box không ghép: 0 (chỉ-tham-chiếu) / 0 (chỉ-mới)**, 0 box không ghép ≥ 0,6. Theo bảng quyết định: chạy S8, `meta.gate_tier = "event"`. Route S8 **không** bit-identical với `detections_cpu`; tương đương ở mức sự kiện (b). Bản sao: `results/kaggle_gate_b01.json` (SHA-256 `a2547f63…a06ba256`), `results/kaggle_gate_b01_frames.csv`, `results/kaggle_env.json`.
- **Môi trường (env.json, khớp prereg):** route `cuda_fp32_kaggle`, Tesla T4, torch 2.9.1+cu128, CUDA 12.8, cuDNN 91002 deterministic, `half=false`, `tf32=false`, ultralytics 8.4.7, OpenCV 4.13.0, numpy 2.0.2, Python 3.12.13.
- **Nhận kết quả:** `results.zip` (SHA-256 `d02479dc…6e717d3c`) → 118 JSON; **118/118** khớp `SHA256SUMS`; tập stem = S8 dựng lại từ `results/meva_S8_blocks.csv` × `meva_manifest.csv` (plan `1101322c…`), **không thiếu, không thừa**. Mọi JSON: route/half/weights `3da1d83e…`/conf 0,25/imgsz 640/ultralytics 8.4.7/prereg hash đúng, `clip_sha256` = `meva_manifest.sha256`, số khung = `n_decoded` của cue laptop (tổng 1.062.229 khung). JSON ở `<DATA_ROOT>/MEVA_20h\detections\` (+`SHA256SUMS`); `meva_manifest.csv` thêm cột `route`, `gate_tier`, `det_sha256`.
- **Bằng chứng bổ sung — đối chứng chéo route trên MEVA** (`code/meva_route_crosscheck.py` → `results/meva_route_crosscheck.{csv,json}`): 6 clip P-1 G639 2018-03-13 17:05–17:35 (54.005 khung), laptop `ov_gpu_fp32x` vs Kaggle, ghép box như `weights_equiv.match`.
  | | laptop | Kaggle | trùng |
  |---|---|---|---|
  | box | 286.148 | 286.074 | 286.068 cặp |
  | sự kiện (a) | 6 | 6 | 6/6 biên |
  | sự kiện (b) | 328 | 328 | 328/328 biên |
  | sự kiện (b′) (k = 6) | 93 | 93 | 93/93 biên |
  | khung g(a) / g(b) / g(b′) trùng | | | 54.005 / 54.005 / 54.005 |
  - 5/6 clip: ≤ 1 box không ghép mỗi bên (conf 0,25), max Δbbox 1 px, max |Δconf| 2e-4.
  - Clip `17-15-00`: 77 khung lệch; laptop có thêm box person nhỏ (vùng x≈790–850, y≈590–660) mà Kaggle không có. Ghép tham lam cho số méo (max Δbbox 229 px, |Δconf| 0,34, 2 cặp "hai phía 0,7"), vì box một phía bị ghép với box khác. Đếm chặt (cặp > 2 px tính là 2 box không ghép): 140 chỉ-laptop / 66 chỉ-Kaggle (cận trên, còn lẫn cặp tham lam), 10 box một phía ≥ 0,6. Box cao nhất 0,7217 (laptop) ở khung 2802–2804, nơi đã có person khác ≥ 0,7 → g(b) không đổi. Với cặp ≤ 2 px: max Δbbox 1 px, |Δconf| ≤ 2e-4, 0 cặp hai phía 0,7.
  - Kết luận: hai route tương đương ở mức sự kiện (a)/(b)/(b′) trên 6 clip MEVA thật. Lệch mức box có, tập trung ở một clip. Đây không phải gate (gate là B0.1 prereg), chỉ là bằng chứng bổ sung.
- 6 JSON laptop đã chuyển sang `<DATA_ROOT>/MEVA_20h\detections_laptop_ov_gpu_fp32x\` (+`SHA256SUMS`, không xoá). **Toàn bộ S8 = một route `cuda_fp32_kaggle`, tier `event`.**
