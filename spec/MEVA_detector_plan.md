# MEVA_detector_plan — kế hoạch phủ detector trên MEVA (B0.3, cố định TRƯỚC khi chạy detector trên MEVA)

Ngày: 2026-09-27. File này được băm SHA-256 (`MEVA_detector_plan.md.sha256`) và commit **trước** lần chạy detector đầu tiên trên clip MEVA. Hash được ghi trong commit message.

## 1. Quy tắc (từ PROMPT Đợt 2, B0.3; chỉ dựa trên tốc độ + tương đương, không dựa trên nội dung MEVA)
- Nếu có route đạt tương đương **và ≥ 7 fps** → FULL 20,83 h.
- Nếu không → **tập con S8**: mỗi camera lấy block theo thứ tự R5 cho tới khi ≥ 1,6 h, luôn gồm block quiet; tổng ≥ 8 h.

## 2. Áp dụng (số liệu: `docs/WEIGHTS_EQUIV.md`)
- Route tương đương: OpenVINO CPU FP32 (2,48 fps đầu-cuối) và OpenVINO Iris Xe FP32 ép f32 (5,13 fps). Route Iris Xe FP16 **không** tương đương.
- Không route nào ≥ 7 fps. Riêng suy luận GPU f32 đã mất ≈ 152 ms/khung → **S8**.
- **S8** (`code/meva_s8.py` → `results/meva_S8_blocks.csv`): 10 block, 118 clip, **9,83 h** (≈ 1,06 triệu khung).

| Camera | Block (R5 rank) | Giờ | Quiet |
|---|---|---|---|
| P-1 G639 school | 2018-03-13 17:05 (1) · 2018-03-12 10:00 (2) | 0,67 + 1,17 = 1,83 | ✔ block 1 |
| P-2 G326 admin | 2018-03-07 16:50 (1) · 2018-03-12 10:00 (2) | 0,83 + 1,17 = 2,00 | ✔ block 1 |
| P-3 G301 hospital | 2018-03-07 16:50 (1) · 2018-03-12 10:00 (2) | 2,00 | ✔ block 1 |
| P-4 G331 bus | 2018-03-07 16:50 (1) · 2018-03-12 10:00 (2) | 2,00 | ✔ block 1 |
| P-5 G638 school | 2018-03-07 16:50 (1) · 2018-03-12 10:00 (2) | 2,00 | ✔ block 1 |

⚠ S8 chỉ nằm trong **3 phiên ghi** (03-07 16:50 với 4 camera, 03-13 17:05 với 1 camera, 03-12 10:00 với 5 camera). Vì vậy bootstrap hai chiều camera × phiên trên S8 có rất ít cụm phiên. Điều này được ghi nhận ngay từ bây giờ, trước khi thấy bất kỳ kết quả nào.

## 3. Cách chạy
- Script `code/meva_detect.py`: mọi khung (**không stride**), conf 0,25, imgsz 640, không lọc lớp. Weights backup SHA `3da1d83e…837a4` (IR OpenVINO export từ đúng file này, dạng dynamic). Output cùng định dạng với `detections_cpu`, lưu ở `D:\THS Programing\data_MEVA_20h\detections\` kèm `meta.route`.
- Hai worker chạy song song, mỗi worker một route tương đương (`ov_gpu_fp32x`, `ov_cpu_fp32`). Mỗi clip do một worker xử lý; route của từng clip được ghi lại. Có checkpoint/resume theo clip.
- **Tập phân tích chính cho H2–H4 = S8, cố định.** Hàng đợi của worker có thể chạy tiếp (`--extend`) sang các block PRIMARY còn lại theo thứ tự R5, xoay vòng camera (rank 3 của P-1..P-5, rồi rank 4, …). Phần mở rộng này chỉ để **phân tích độ nhạy** (E6). Nó không thay đổi tập chính và không được dùng để quyết định điều gì dựa trên kết quả.
- Định nghĩa (c) (nhãn hoạt động MEVA) và cue d_t không cần detector, nên được tính trên toàn bộ 20,83 h.
- Kiểm dung lượng trống trước mỗi clip (dừng nếu < 20 GB).
