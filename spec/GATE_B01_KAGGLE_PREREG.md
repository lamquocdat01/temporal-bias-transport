# GATE B0.1 (Kaggle) — pre-register hai bậc, **v2** (29-09-2026, TRƯỚC lần chạy Kaggle đầu tiên)

Áp dụng cho route S8 `cuda_fp32_kaggle` (`docs/DETECTOR_PROVENANCE.md`, `kaggle/s8_detect.ipynb`).
File này được băm (`GATE_B01_KAGGLE_PREREG.md.sha256`) và commit **trước** khi notebook chạy trên Kaggle.
Notebook ghi SHA-256 của file này vào `gate_b01.json` và vào `meta` của mọi JSON S8. Không sửa ngưỡng sau khi thấy kết quả.

## Lịch sử phiên bản
- **v1** — commit `82c3637` (SHA-256 `7556b064b33be80a6a8f21cdd3675c8750695fc99ff93458b26a2510ab3f0911`). Bậc 2 đòi 600/600 khung khớp đủ hộp.
- **v2** — sửa **duy nhất** tiêu chí 1 của bậc 2: cho phép box không ghép cặp nếu conf < 0,6; box không ghép có conf ≥ 0,6 → FAIL; in số box không ghép mỗi bên. Bậc 1 và các tiêu chí 2–6 của bậc 2 giữ nguyên.
  - **Lý do:** v1 làm bậc 2 FAIL chỉ vì một box sát ngưỡng lọc conf 0,25 xuất hiện ở một bên — khác biệt số học nhỏ của GPU quanh ngưỡng lọc, không ảnh hưởng g(b) (ngưỡng 0,7). Biên 0,6 cách ngưỡng g(b) 0,1, nên box không ghép được phép không thể làm đổi g(b); box không ghép ≥ 0,6 là dấu hiệu khác biệt thật gần vùng quyết định → FAIL.
  - **Thời điểm:** sửa trước khi có **bất kỳ** kết quả GPU/Kaggle nào (chưa upload, chưa chạy). Kết quả duy nhất đã có là smoke CPU của code notebook (bậc 1 PASS, 0 box không ghép), không phải route Kaggle.

## Dữ liệu và cách so
- 600 khung đã giải mã (npz, không mất dữ liệu): CDnet2014 thermal/park khung 250–599 (350) + LASIESTA I_OC_01 khung 0–249 (250); cả hai liên tục.
- Tham chiếu: `detections_cpu` (cùng nguồn với B0.1 torch_cpu, `docs/WEIGHTS_EQUIV.md`).
- Box: `[x1, y1, x2, y2]` int, conf làm tròn 4 chữ số, cls int (như `code/weights_equiv.predict`).
- Ghép box trong mỗi khung: tham lam, cùng lớp, khoảng cách = max |Δ tọa độ| (không đổi so với code gate hiện có).
  Khung "khớp đủ" ⇔ số box hai bên bằng nhau và mọi box ghép được 1-1.
  **Box không ghép** = box của một bên không được ghép với box nào của bên kia (đếm riêng: chỉ-tham-chiếu, chỉ-mới).
- g(b) theo khung = có box person (cls 0) conf ≥ 0,7. Sự kiện (b) = run cực đại g = 1 trong mỗi video; biên = (khung đầu, khung cuối).
  Tham chiếu có **41** sự kiện (b).

## Bậc 1 — bit-equivalent (không đổi)
PASS ⇔ đủ 600 khung **và** 600/600 khung khớp đủ **và** max |Δbbox| = 0 px **và** max |Δconf| = 0 **và** sự kiện (b) 41/41.

## Bậc 2 — event-equivalent
PASS ⇔ tất cả:
1. đủ 600 khung; **mọi box không ghép (mọi lớp, cả hai bên) có conf < 0,6** — một box không ghép có conf ≥ 0,6 là FAIL (v2);
2. g(b) trùng **600/600** khung;
3. sự kiện (b): cả hai bên đúng **41**, và **41/41 trùng biên** (cùng video, cùng khung đầu, cùng khung cuối);
4. max |Δbbox| ≤ **1 px** trên mọi cặp box đã ghép (mọi lớp);
5. max |Δconf| ≤ **5e-3** (bao gồm biên) trên mọi cặp box đã ghép (mọi lớp);
6. **0** cặp box person đã ghép có conf hai bên nằm hai phía 0,7, tức (conf_ref ≥ 0,7) ≠ (conf_new ≥ 0,7).

Notebook in: số box không ghép mỗi bên (chỉ-tham-chiếu, chỉ-mới; tổng và theo video), conf lớn nhất của box không ghép mỗi bên,
số box không ghép có conf ≥ 0,6. Bậc 1 PASS kéo theo bậc 2 PASS.

## Quyết định (cố định)
| Kết quả | Hành động | Ghi |
|---|---|---|
| Bậc 1 PASS | Chạy S8 | `meta.gate_tier = "bit"` |
| Bậc 1 FAIL, bậc 2 PASS | Chạy S8 | `meta.gate_tier = "event"`; báo cáo ghi rõ route S8 **không** bit-identical với `detections_cpu`, tương đương ở mức sự kiện (b); công bố Δbbox/Δconf và số box không ghép thực đo |
| Cả hai FAIL | **Dừng**, không chạy S8; quay **nhánh L** (route laptop theo `MEVA_detector_plan.md` §2–3) | gửi `gate_b01.json` + `gate_b01_frames.csv` |

- Notebook luôn in kết quả **cả hai bậc** (từng tiêu chí và giá trị đo), kể cả khi bậc 1 đã PASS.
- Mỗi version/phiên Kaggle (kể cả resume, kể cả shard) chạy lại gate. Phiên nào cả hai bậc FAIL thì phiên đó không sinh JSON S8.
  Nếu các phiên đạt bậc khác nhau ("bit" / "event"), tier ghi theo từng JSON và báo cáo nêu rõ.
- Không có bậc 3, không nới ngưỡng thêm, không chọn lại khung gate sau khi thấy kết quả.
