# PREREG_DECISIONS — quyết định của anh Đạt (2026-09-27) ràng buộc `prereg/PREREG_MEVA.json` ở Đợt 3

Nguồn: phản hồi Q-A..Q-E sau báo cáo Đợt 2 tạm. File này được commit cùng STRATA_SPEC v2, **trước** mọi hồ sơ MEVA.

| # | Quyết định | Áp dụng |
|---|---|---|
| Q-A | STRATA_SPEC v2: pooled R → D → C (chính); v1 làm E6, báo **cả hai ω**; R-cens/D-cens vận chuyển như tầng thường, cận ĐL1(b) làm độ nhạy; false-open **theo khung** là chính, /giờ = × fps đích × 3.600; D mịn chỉ ở E6 | `docs/STRATA_SPEC.md` v2 |
| Q-B | Định nghĩa transport chính **(b′)** = (b) nối khe idle ≤ round(0,2 s × fps) (25 fps → 5, 30 fps → 6, lowFramerate → 0); (b) thô ở E6 | STRATA_SPEC v2 §1 (kèm căn cứ VIRAT-P) |
| Q-C | P1: pre-register **4 ô K ∈ {2, 5, 10, 30}** cho H2/H3; dải K = 2..30 ở phụ lục | STRATA_SPEC v2 §6 |
| Q-D | **H3, tiêu chí ràng buộc:** transport giảm **sai số tuyệt đối ≥ 50 %** so với ước lượng ngây thơ (trung bình benchmark). **Coverage chỉ báo cáo**, không ràng buộc. Báo **prediction interval** cho giá trị MEVA quan sát được, giá trị tới hạn **t với df = số phiên − 1**. CI phụ: bootstrap theo **block**. Detector `--extend`: ưu tiên phiên ghi mới | Đợt 3 + `docs/MEVA_detector_plan_addendum.md` |
| Q-E | Thứ tự chạy: (1) synthetic_check có checkpoint (đã chạy xong 80×300); (2) tải MEVA nền; (3) detector 2 worker **chỉ khi máy trống**; (4) cue MEVA sau. `meva_profile.py` chỉ chạy sau khi v2 băm + commit (có guard trong code) | vận hành |
| A4 | H4 dùng r\* từ LP `code/flip_radius.py` (công thức đóng chỉ là cận dưới) | PLAN §4.3 đã đính chính |

Căn cứ số cho Q-D (`results/synthetic_check.json`, `docs/theory_notes.md` §7): CI-t phủ estimand transport M\* ≥ 0,95 ở mọi ô (bỏ ô tất định P1_proc), nhưng lệch A1 do độ thô tầng làm coverage đối với giá trị đích thật và PI thấp (ví dụ P2 frame recall) → ràng buộc bằng giảm sai số, không bằng coverage.

## Bổ sung 2026-09-27 (lượt quyết định thứ hai)
| # | Quyết định | Áp dụng |
|---|---|---|
| 1 | Giữ STRATA_SPEC v2, **không ra v3**. Đợt 3 pre-register: các ô **P2 × frame recall và P3 × frame recall dự kiến KHÔNG đạt H3** (căn cứ: bộ sinh tổng hợp không thỏa A1 theo xây dựng; lệch A1 ở P2 frame recall tới +0,08 theo tầng, −0,065 tổng; `theory_notes.md` §7). **Trục E6 thêm: D mịn và mức nền cue** (trung vị chuẩn hóa 1.800 khung) | PREREG_MEVA.json (Đợt 3) |
| 2 | Cổng máy trống: CPU của tiến trình khác ≤ 60 %, RAM trống ≥ 6 GB; **ban ngày 1 worker (Iris Xe f32), ban đêm 2 worker** | `code/meva_detect.py` |
| 3 | Tải MEVA: 118 clip S8 đứng đầu manifest, phần còn lại theo sau; selection không đổi | `code/meva_download.py` |
| 4 | Căn cứ (b′) = **sửa phân mảnh track**, không phải lọc ảo giác; VIRAT_S_000200_05 bị loại vì **không có nhãn** | STRATA_SPEC v2 §1 (lời văn đã đúng, không đổi file) + bài |
| 5 | **Dự đoán H2–H4 cho (b′) không được dùng số liệu (c) MEVA đã xem** | PREREG_MEVA.json (Đợt 3) |

**Khai báo những gì đã xem trên MEVA trước pre-registration (cho mục 5):**
- *(tới 27-09)* chỉ **(c)** trên 7 block S8 của P-1..P-3 (5,50 h): duty 0,065; run idle dài nhất 1.119 s; ω/TV (c) so khác định nghĩa 0,57/0,65; median thời lượng (c) theo block 59–203 khung. *(Ghi chú 30-09: thực là **6** block; số liệu dựng lại với nhãn theo `n_decoded` **không đổi**: 6 block, 5,50 h, duty 0,0652, idle dài nhất 1.119 s, median 61–203 khung.)*
- *(tới 27-09)* (b′), (b), (a) trên MEVA: chưa xem gì;
- **Đã xem thêm tới 30-09** (sau khi nhận detections Kaggle S8 và dựng lại nhãn (c) theo `n_decoded`):
  - **(b′), (b), (a) trên S8** (10 block, 9,835 h): toàn bộ `E1_profile_meva.csv`, `meva_omega.csv`, `meva_wT_strata.csv` cho ba định nghĩa. Tổng ω (b′) = 0,514 (v2) / 0,615 (v1); TV 0,608 / 0,595. Bảng ω camera × phiên và π_pool (`results/omega_bprime_S8.csv`); tầng của ô P-4 × 03-07 (`results/omega_pool_diag_G331_2018-03-07_165000.json`). ω (b) v2 0,603; ω (a) v2 0,147;
  - **(c)** dựng lại: S8 10 block (9,17 h) duty 0,159, idle dài nhất 2.102 s; mọi (c) sẵn có 21 block (16,67 h) duty 0,135, idle dài nhất 2.500 s; ω (c) v2 S8 0,489, sẵn có 0,540;
  - đối chứng route 6 clip P-1 (sự kiện (a)/(b)/(b′) laptop vs Kaggle);
- **không** chính sách nào đã chạy trên MEVA (vẫn đúng tới 30-09).

**Đầu vào được phép cho dự đoán (b′), Đợt 3:** `benchmark_units.parquet`, `m_B_strata.csv`, `w_B_strata.csv`, `operating_points.json`, và hồ sơ MEVA **(b′)** (`meva_wT_strata.csv` với `definition == "bprime"`).
**Cấm:** mọi hàng `definition == "c"` trong `E1_profile_meva.csv` / `meva_omega.csv` / `meva_wT_strata.csv`. Script dựng prereg ở Đợt 3 sẽ assert điều này.

## Bổ sung 2026-09-30 — luật π_pool và tách H3
**Thời điểm:** đặt **SAU** khi đã xem bảng ω (b′) trên S8 (`results/omega_bprime_S8.csv`, `docs/DOT2_REPORT.md` mục 30-09) và **TRƯỚC** mọi kết cục chính sách trên MEVA. Tại thời điểm commit, chưa có chính sách nào chạy trên MEVA; chưa tính m_T, sai số transport hay PI nào trên MEVA.

| # | Luật | Áp dụng |
|---|---|---|
| P-1 | **π_pool(ô) = ω_v1(ô) − ω_v2(ô)**, tính cho từng ô camera × phiên (= một block S8), theo camera, theo phiên và tổng. ω = khối khung đích rơi vào U, định nghĩa (b′), arm main. **Ô có π_pool > 0,5 = "pooling-dependent"**: báo **riêng** (ω_v2, ω_v1, π_pool, số khung, tầng chiếm khối chính, hỗ trợ benchmark của tầng đó, ô v2 đích) và **loại khỏi mẫu số H3**. Khung của ô này bị bỏ khỏi cả giá trị transport lẫn giá trị MEVA quan sát dùng để tính sai số H3. Ngưỡng 0,5 cố định, không chỉnh theo kết cục. | `code/omega_pooling.py`, PREREG_MEVA.json (Đợt 3) |
| P-2 | **Tách H3 làm hai phần.** (i) **Phần transportable** = các ô không pooling-dependent, chỉ các khung có h ≠ U theo v2: ước lượng điểm transport **+ PI** (t, df = số phiên − 1, như Q-D). Tiêu chí ràng buộc H3 (giảm sai số tuyệt đối ≥ 50 %) **chỉ** đánh giá trên phần này. (ii) **Toàn MEVA**: **chỉ cận Manski**, không có ước lượng điểm. Với metric trong [0, 1]: [(1 − ω)·M̂_supp + ω·0, (1 − ω)·M̂_supp + ω·1], **độ rộng = ω** (metric theo giờ: nhân fps đích × 3.600). Không báo cáo "giá trị MEVA toàn phần" nào khác cận này. | PREREG_MEVA.json (Đợt 3), bài |

**Hệ quả xác định ngay khi áp luật lên bảng ω đã xem** (không phụ thuộc kết cục chính sách): đúng 1 ô pooling-dependent là **P-4 G331 × 2018-03-07 16:50** (π_pool = 0,993; 90.068 khung = 8,5 % khung S8). Mọi ô khác có π_pool ≤ 0,053. Ở mức camera, P-4 có π_pool = 0,414, và luật chỉ áp theo ô.

## Bổ sung 2026-10-01 — Quyết định Đợt 3 D3.1–D3.7 (anh Đạt chốt qua Cowork 27–30-09; ghi nguyên văn từ `PROMPT_09_DOT3_PREREG_20260930.md`)
**Thời điểm:** đặt **sau khi xem bảng ω (b′) theo ô camera × phiên (v1/v2, π_pool) và hồ sơ (c) sau dựng lại nhãn**, **trước mọi kết cục chính sách MEVA**. Lúc ghi mục này (2026-10-01 15:08 +0700) chưa có chính sách nào chạy trên MEVA; `code/meva_outcomes.py` vẫn khóa; chưa tính m_T, sai số transport, PI hay cận nào trên MEVA.

- **D3.1** STRATA_SPEC **v2 là chính** (băm trước MEVA), v1 là độ nhạy. Không ra v3.
- **D3.2** Ô có **π_pool = ω_v1 − ω_v2 > 0,5** = "pooling-dependent": báo riêng, **loại khỏi mẫu số H3**.
- **D3.3** H3 tách hai đối tượng: **(i) phần transportable** (khối 1 − ω_v2) → ước lượng điểm + PI, đây là nơi duy nhất đánh giá H3; **(ii) toàn MEVA** → chỉ cận Manski (độ rộng ≈ ω). Tiêu chí **ràng buộc** của H3 = giảm sai số tuyệt đối ≥ 50 % so với ngây thơ; coverage chỉ báo cáo.
- **D3.4** P1 vào H2/H3 bằng 4 ô K ∈ {2, 5, 10, 30}; dải K = 2..30 ở phụ lục.
- **D3.5** Pre-register dự đoán **thất bại** cho ô P2/P3 × frame recall (A1 không đủ với metric phụ thuộc trạng thái cue — theory §7).
- **D3.6** PI dùng t với df = số phiên − 1 (= 2 cho S8); CI phụ = bootstrap theo block. Không mở rộng MEVA theo kết quả.
- **D3.7** Dự đoán H2–H4 chỉ được dựng từ benchmark + hồ sơ MEVA (w_T, ω); **cấm** dùng bất kỳ số (c) hay số (b′) đã xem để hiệu chỉnh.

D3.2 và D3.3 trùng nội dung với luật P-1/P-2 (mục 30-09 ở trên); D3.4 = Q-C; D3.5 = mục 1 của lượt 27-09; D3.6 = Q-D. Không quyết định nào ở trên bị thay đổi.

**Dòng "đã xem" — cập nhật 2026-10-01 (trước pre-registration CP1):**
- ω (b′) theo ô camera × phiên, v2 và v1, và π_pool: `results/omega_bprime_S8.csv` (tổng ω_v2 0,514, ω_v1 0,615, π_pool 0,101; 1 ô pooling-dependent: P-4 G331 × 03-07 16:50, π_pool 0,993);
- tầng chiếm khối chính của ô P-4 × 03-07: `full|gt30|lt300|C1` (98,7 % khung block; 5 cửa sổ / 5 video benchmark; v2 gộp vào `P3:full|gt30|lt1800|C0`, 27 cửa sổ / 13 video, w_B 0,0063);
- (c) sau dựng lại nhãn theo `n_decoded`: S8 10 block 9,17 h duty 0,159, run idle dài nhất 2.102 s; mọi (c) sẵn có 21 block 16,67 h duty 0,135, run idle dài nhất 2.500 s; TV(w_B, w_T) (c) trên ô (b′) benchmark, v2: S8 0,607, sẵn có 0,604; ω (c) v2: 0,489 / 0,540;
- TV(w_B, w_T), v2, arm main, S8: (a) 0,522; (b) 0,572; (b′) 0,608 (thêm cột `TV_v2`, `omega_v2` và dòng `all_S8` vào `results/E1_profile_meva.csv`, 2026-10-01; các dòng cũ không đổi);
- **không** chính sách nào đã chạy trên MEVA; không kết cục nào (m_T, sai số, PI, cận) đã được tính.

## Bổ sung 2026-10-03 — AMENDMENT 01 (lỗi thứ tự khung LASIESTA; anh Đạt chọn P0 = A)

- Chi tiết, cam kết và dự đoán định tính: `prereg/AMENDMENT_01.md` (+ `.sha256`), commit trước khi chạy lại.
- **Quy tắc báo cáo (ràng buộc):** bộ corrected là kết luận chính; bộ gốc (pre-registered mù, `PREREG_MEVA.json` 209f2e22…) báo song song trong supplement (mục "Amendment 01") và CSV; H nào khác kết luận giữa hai bộ thì bài ghi cả hai, không chọn. Bảng "pre-registered vs thực hiện" thêm cột "orig / corrected".
- Dự đoán bộ corrected: `prereg/PREREG_MEVA_corrected.json` (hash riêng). Kết quả: `results/orig/` (bản gốc, không ghi đè) và `results/corrected/`.
