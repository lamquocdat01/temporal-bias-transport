# MEVA_detector_plan — Phụ lục A1 (2026-09-27, theo quyết định Q-D của anh Đạt)

Phụ lục này bổ sung cho `MEVA_detector_plan.md` (SHA-256 `1101322c…bd4be`, không sửa file gốc). File được băm (`MEVA_detector_plan_addendum.md.sha256`) và commit **trước** khi bất kỳ clip nào ngoài S8 chạy detector (tại thời điểm commit: 2/118 clip S8 đã xong, 0 clip mở rộng).

## Thay đổi duy nhất: thứ tự hàng đợi `--extend`
- **Cũ (§3):** các block PRIMARY còn lại theo R5, xoay vòng camera (rank 3 của P-1..P-5, rồi rank 4, …).
- **Mới:** ưu tiên **phiên ghi chưa có trong S8** (phiên = các block cùng ngày có khoảng thời gian chồng nhau, `meva_profile.sessions`). Thứ tự khóa sắp xếp:
  1. phiên mới trước phiên đã có trong S8;
  2. phiên có nhiều camera trước;
  3. tên phiên (ngày + giờ bắt đầu) tăng dần;
  4. camera tăng dần.
- **Lý do:** S8 chỉ có 3 phiên. Mở rộng theo phiên mới làm tăng số cụm phiên cho bootstrap hai chiều camera × phiên (t với df = số phiên − 1).
- Hàng đợi kết quả (code/meta, không nhìn nội dung): S8 (10 block) → 2018-03-11 11:15 (5 camera) → 2018-03-13 15:50 (5 camera) → 2018-03-09 10:10 (P-1) → 2018-03-11 13:50 (P-1) → 2018-03-11 16:10 (P-4) → các block còn lại của phiên 2018-03-13 17:05 (P-2, P-3, P-5).

## Không đổi
- Tập phân tích chính cho H2–H4 = **S8, cố định**.
- Phần mở rộng chỉ dùng cho độ nhạy.
- Không có điều kiện dừng hay mở rộng nào phụ thuộc kết quả; không stride; conf 0,25, imgsz 640; route và weights như plan gốc.
- Detector chỉ chạy khi máy trống (Q-E).

Ghi chú: selection v1 thực có **26** block PRIMARY (P-1: 6, P-2..P-5: 5 mỗi camera), không phải 25 như ghi ở mục Quyết định CP0; tổng 250 clip, 20,83 h không đổi.
