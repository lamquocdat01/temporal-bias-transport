# WEIGHTS_EQUIV — B0: tương đương detector (Đợt 2, 2026-09-27)

Script: `code/weights_equiv.py` (so từng khung với `detections_cpu`), `code/bench_routes.py` (tốc độ trên 3 clip MEVA thử).
Chi tiết: `results/weights_equiv_<route>.json`, `results/weights_equiv_frames_<route>.csv`, `results/bench_routes.csv`.

Video kiểm: **CDnet2014/thermal/park** (temporalROI 250–600, 350 khung đọc được) và **LASIESTA/I_OC_01** (250 khung). Hai video ngắn, cùng có sự kiện (a) và (b) không tầm thường. Tổng 600 khung, 41 sự kiện (b).
Nguồn khung và temporalROI lấy bằng cách **import nguyên** `scm/dump_p1_detections.py` (chỉ đọc). Box được tuần tự hóa giống bản gốc (xyxy int, conf làm tròn 4 chữ số).
Weights: `06.6 ASMAG Project - Backup 260426\AI model\yolo26s-seg.pt`, SHA-256 `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` (assert trong code); bản sao làm việc: `<OV_DIR>/`.

## B0.1 — PyTorch CPU FP32, backup weights
| Video | Khung | Số box khớp | max \|Δbbox\| | max \|Δconf\| | g(a) | g(b) | Sự kiện (b) ref / mới |
|---|---|---|---|---|---|---|---|
| thermal/park | 350 | 100 % | 0 px | 0 | 100 % | 100 % | 27 / 27 |
| I_OC_01 | 250 | 100 % | 0 px | 0 | 100 % | 100 % | 14 / 14 |

**Kết luận: PASS — trùng khớp tuyệt đối** (mọi box, lớp, bbox, conf). Bản backup tái tạo đúng detections đã dùng cho 121 video → **dùng backup cho MEVA**; không cần dump lại corpus.

## B0.2 — Tăng tốc
Tiêu chí: g(b) khớp ≥ 99,5 % khung **và** số sự kiện (b) lệch ≤ 1 %. Tốc độ đo trên 3 clip MEVA thử, 150 khung/clip sau 5 khung warm-up.

| Route | g(a) | g(b) | Sự kiện (b) | max \|Δconf\| | Tương đương | fps suy luận (TB / max) | fps đầu-cuối gồm decode (TB / max) |
|---|---|---|---|---|---|---|---|
| torch_cpu (tham chiếu) | 100 % | 100 % | 41 / 41 | 0 | ✔ | 2,56 / 2,80 | 2,50 / 2,73 |
| **OpenVINO CPU FP32** (IR dynamic) | 100 % | 100 % | 41 / 41 | 1e-4 | ✔ | 2,54 / 3,01 | 2,48 / 2,92 |
| **OpenVINO Iris Xe FP16** | 99,83 % | 99,50 % | 42 / 41 (+2,4 %) | 0,42 | ✘ (sự kiện lệch > 1 %) | 4,56 / 6,11 | 4,18 / 5,56 |
| OpenVINO Iris Xe, IR FP32, mặc định | 99,83 % | 99,50 % | 42 / 41 | 0,42 | ✘ | 4,18 / 4,56 | 3,82 / 4,18 |
| OpenVINO Iris Xe, IR FP32 + `INFERENCE_PRECISION_HINT=f32` *(route bổ sung)* | 100 % | 100 % | 41 / 41 | 2e-4 | ✔ | 5,69 / 6,56 | 5,13 / 5,84 |

Ghi chú kỹ thuật
- Lần export đầu (IR **tĩnh 640×640**) làm lệch g(b) tới 7 %: letterbox vuông khác letterbox chữ nhật mà PyTorch dùng. Mọi IR dùng cho số liệu trên đều export **dynamic**, nên có cùng letterbox chữ nhật với PyTorch (1080p → 384×640).
- GPU plugin của OpenVINO mặc định suy luận ở **f16** bất kể IR. Vì vậy "IR FP32 mặc định" cho kết quả giống hệt FP16. Chỉ khi ép `INFERENCE_PRECISION_HINT=f32` thì mới đạt tương đương. Route này **không có trong danh sách Đợt 2** mà em thêm vào; nó dùng cùng tiêu chí và cùng thước đo.
- Trên GPU, phần suy luận thuần của route f32 là ≈ 152 ms/khung. Vì thế route này **không thể đạt 7 fps**, kể cả khi máy rảnh.
- ⚠ Mọi phép đo tốc độ diễn ra khi máy **đang tải nặng bởi các phiên khác** (Đề tài 28: mô phỏng p2_sim_run, notebook; `a3_iw_ablation.py`; tải S3). Số tuyệt đối vì thế thấp hơn khi máy rảnh: CP0 đo torch_cpu 2,98 fps. Kết luận "< 7 fps" vẫn đứng vững nhờ ghi chú ở trên.

## Kết luận B0
- B0.1 **PASS**, B0.2: có 2 route tương đương (OV CPU FP32, OV GPU f32), nhưng **không route nào ≥ 7 fps** → theo quy tắc cố định: **tập con S8** (`docs/MEVA_detector_plan.md`).
