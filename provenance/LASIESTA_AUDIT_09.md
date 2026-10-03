# LASIESTA_AUDIT_09 — Đợt 5 P0: lỗi thứ tự khung LASIESTA có nhiễm vào #09 không? (03-10-2026)

Phát hiện gốc: đề tài #17, `17_Sim2Real_Certification\notes\B0_lasiesta_impact.md` (27-09). Code audit: `code/lasiesta_audit.py`, `code/lasiesta_audit_transport.py`. Kết quả: `results/lasiesta_audit/` (layer_check.json, videos_before_after.csv, shadow_benchmark.json, omega_old_new.csv, transport_old_new.csv, transport_summary.json). Đây là **tính bóng (shadow)**: không file kết quả, prereg hay bài nào bị ghi đè.

## Kết luận một dòng

**#09 BỊ NHIỄM, theo kiểu "nhất quán nhưng xáo trộn".** Lỗi có ở cả ba lớp: detections, cue và frame_scores/events. Cả ba đi qua cùng một `sorted(glob)` sắp theo chữ. Hệ quả: trên 48 video LASIESTA, sự kiện bị vỡ vụn giả, và cue được tính giữa những khung không kề nhau.
- **Không cần dump lại detector**, vì đánh lại chỉ số là đủ. Đã kiểm: ở 6/6 khung, detector chạy lại trên file theo thứ tự chữ cho đúng box đã lưu; trên file theo thứ tự số thì 0/6 khớp.
- **Phải tính lại cue** LASIESTA từ ảnh theo đúng thứ tự. Việc này mất vài phút và đã làm xong trong audit.
- Phía MEVA, CDnet (53/53), BMC (20/20) và VIRAT-P không bị ảnh hưởng.
- **Đề xuất phương án A.**

## 0.1–0.2 Ma trận lớp (video mẫu I_BS_01, 275 khung)

| Lớp | Code | 10 file đầu theo thứ tự lớp dùng | Đúng thứ tự? | Bằng chứng |
|---|---|---|---|---|
| Thứ tự số đúng (natural) | — | -1, -2, -3, -4, -5, -6, -7, -8, -9, -10 | — | — |
| (a) detections_cpu | `scm/dump_p1_detections.py` `iter_frames_dir`: `files = sorted(set(glob))`, chỉ số = vị trí | -1, -10, -100, -101, -102, -103, -104, -105, -106, -107 | **SAI** | Chạy lại yolo26s-seg (CPU, conf 0,25, imgsz 640) tại chỉ số 1, 2, 3, 10, 50, 100: khớp box đã lưu **6/6** trên file theo chữ, **0/6** trên file theo số |
| (b) cue d̃ (`cache/bench_cue.parquet`) | `code/cue.py` `bench_main` → `dp.make_frame_iter` → cùng `iter_frames_dir` | giống (a) | **SAI** | Cue cache so với cue tính lại theo chữ: max\|Δ\| = **0**. So với cue theo số: max\|Δ\| = 8,29 |
| (c) frame_scores.parquet / events_durations.csv | Deep3, dựng từ chỉ số khung của detections_cpu | frame 0 = -1, frame 1 = -10, frame 2 = -100, … | **SAI** | g_t(a) của frame_scores trùng has-box của detections_cpu tại cùng chỉ số (đúng 100 %) |

Phạm vi:
- Thứ tự chữ trùng thứ tự số ở **0/48** video LASIESTA, nhưng ở 53/53 video CDnet và 20/20 video BMC (tên file đệm số 0).
- Đánh số LASIESTA liên tục từ 1 đến n ở mọi video, nên phép hoán vị ngược luôn xác định duy nhất.

**Chẩn đoán:** (a), (b), (c) **sai cùng một kiểu**, tức "nhất quán nhưng xáo trộn". g_t và cue không lệch chỉ số với nhau, nhưng cả hai đều nằm trên một trục thời gian giả. Cụ thể:
- sự kiện bị cắt thành các khối khoảng 10 khung (file -10…-19, -100…-109 kề nhau theo thứ tự chữ);
- tại mỗi điểm nhảy khối, cue là hiệu của hai khung cách xa nhau trong thời gian thật.

Phần này tái lập đúng số của #17: (a) LASIESTA 936 → **393** sự kiện.

## 0.3 Trước / sau (chỉ đánh lại chỉ số; tính lại cue từ ảnh)

### 5 video mẫu (seed 42)

| video | (b) sự kiện cũ → mới | (b) trung vị thời lượng (khung) | (b′) sự kiện | (b′) idle dài nhất | d̃ q90 cũ → mới | d̃ q99 cũ → mới |
|---|---|---|---|---|---|---|
| I_CA_02 | 34 → 3 | 10 → 136 | 1 → 1 | 91 → 180 | 15,8 → 5,8 | 22,8 → 9,3 |
| I_SM_07 | 20 → 5 | 10 → 32 | 4 → 3 | 54 → 75 | 1,55 → 1,67 | 1,94 → 2,29 |
| O_RA_01 | 12 → 3 | 10 → 22 | 1 → 1 | 816 → 1.137 | 3,68 → 1,60 | 5,71 → 2,56 |
| O_SM_03 | 35 → 4 | 10 → 43,5 | 2 → 1 | 83 → 74 | 4,48 → 6,58 | 6,51 → 22,1 |
| O_SU_02 | 34 → 10 | 10 → 9,5 | 2 → 1 | 104 → 69 | 2,33 → 2,04 | 4,18 → 2,45 |

Duty cycle của (a) và (b) **không đổi** vì phép hoán vị giữ nguyên tổng. Duty của (b′) đổi nhẹ vì phép nối khe phụ thuộc vào thứ tự.

### Toàn bộ 48 video (tính đủ, không ngoại suy)

| Đại lượng | cũ | mới |
|---|---|---|
| Sự kiện (a) / (b) / (b′) | 936 / 1.083 / 121 | **393 / 226 / 73** |
| Trung vị (theo video) của trung vị thời lượng (b) | 10 | 34,5 |
| Trung vị idle dài nhất (b), khung | 91 | 98,5 |
| Idle dài nhất (b), toàn LASIESTA | 850 | 1.137 (vẫn < 1.800) |
| Trung vị d̃ q50 / q90 / q99 | 0,98 / 2,38 / 4,54 | 1,06 / 2,00 / 2,84 |
| Trung bình MAD thô | 8,08 | 6,26 |

## Ảnh hưởng lên benchmark và transport (tính bóng trên 121 video, cùng code, chỉ đổi input LASIESTA)

Phần "cũ" của bản tính bóng tái lập đúng ba nguồn:
- `operating_points.json`, chênh < 1e-9;
- `w_B_strata.csv`, TV < 1e-5;
- `E3_transport_cells.csv`, chênh ≤ 5e-7 (do CSV lưu %.6g).

Phần "mới" vì thế là ảnh hưởng thuần của việc sửa dữ liệu.

| Đại lượng | cũ | mới | Nhận xét |
|---|---|---|---|
| Sự kiện (b′) gộp 121 video | 565 | 517 | LASIESTA 121 → 73 |
| τ₂ / τ₃ / τ₄ (recall sự kiện gộp (b′) = 0,90) | 0,9984 / 0,9978 / 0,2266 | **0,9767 / 0,9683 / 0,2249** | τ₂ −2,2 %, τ₃ −3,0 %: P2/P3 xử lý **nhiều khung hơn** |
| P1 recall (b′), K = 2 / 5 / 10 / 30 | 0,893 / 0,773 / 0,681 / 0,533 | 0,900 / 0,783 / 0,688 / 0,539 | +0,6 đến +1,0 điểm |
| Số ô tầng (v2) / tập ô | 14 | 14, **cùng tập ô** | strata map gần như không đổi |
| TV(w_B cũ, w_B mới) | — | **0,011** | LASIESTA = 11,1 % số khung; TV trong LASIESTA 0,10 |
| Khối U benchmark | 0,0524 | 0,0491 | |
| **ω (S8, mọi đơn vị camera/phiên/all)** | 0,5141 (all) | **0,5141, không đổi ở mọi đơn vị** | |
| TV(w_B, w_T) (b′), all | 0,6084 | 0,6099 | camera: lệch tối đa 0,005 |
| Naive P2 processed / P3 processed | 0,528 / 0,624 | **0,616 / 0,713** | **+0,09**, do τ thấp hơn và cue LASIESTA khác |
| Naive P4 false-open/khung | 0,041 | 0,071 | |
| Transport 'all' P2 / P3 / P4 processed | 0,501 / 0,576 / 0,606 | 0,580 / 0,676 / 0,574 | |

### Ngoại suy lên H1–H4

Giá trị quan sát trên MEVA của P1 không phụ thuộc benchmark, nên với P1 các con số dưới đây là **chính xác**. Với P2–P4, giá trị quan sát trên MEVA đổi theo τ đông lạnh và chỉ biết được khi chạy lại.

- **H1** (đã quan sát, không mù):
  - Thành phần TV vẫn ≥ 0,3: (b′) 0,6084 → 0,6099 (đã tính). (a) chưa tính lại; (a) chỉ đổi trong LASIESTA (11 % khung), nên TV khó đổi hơn khoảng 0,01–0,02 so với 0,522.
  - Điều khoản duty-trong-CI: duty (a)/(b) của benchmark không đổi; (b′) đổi nhẹ.
  - → **Dự kiến không đổi (FAIL).**
- **H3:**
  - Ô P1 (92 ô có định nghĩa): PASS từ 69 còn **68**. Có 1 ô đổi PASS → FAIL, 0 ô đổi ngược lại. Trung vị tỉ số sai số P1 từ 0,212 xuống **0,161**, tức P1 transport tốt hơn.
  - Ô P2–P4 (100 ô): hiện đạt 12. Để H3 PASS (≥ 154/192) thì P2–P4 phải đạt ≥ 86/100. Không chứng minh được là bất khả, nhưng dịch chuyển phía benchmark có trung vị chỉ 0,013, so với sai số transport trung vị 0,082.
  - → **H3 FAIL gần như chắc chắn giữ nguyên.** Riêng mức P2–P4 (0,12 hiện tại) sẽ đổi.
- **H2:** hiện 0 ô đạt (A ∧ B) ở cả 3 metric. Naive và transport của P2–P4 dịch tới 0,09 nên H2 **có thể đổi**; với P1, (A) ∧ (B) ở event recall vẫn dự kiến FAIL.
- **H4:** AUC 0,739 so với ngưỡng 0,75, **sát ngưỡng**. Δ_B và r\* (đổi theo m_B) cùng các lật quan sát của P2–P4 (đổi theo τ) đều thay đổi. → **H4 có thể đổi kết luận theo cả hai chiều.** Đây là rủi ro chính.
- **Câu headline** "P1 vs chính sách có trạng thái": phần P1 đứng vững và mạnh hơn (P1 tốt lên). Dải \StatefulRate phải tính lại.
- **E7** (cue trong tầng): đổi, vì 11 % khung benchmark có cue khác hẳn (q99 4,5 → 2,8). Kết luận định tính "tầng không cố định trạng thái cue" cần chạy lại mới biết.
- **E1/census:**
  - Đổi các số của LASIESTA: sự kiện, thời lượng, KM, idle.
  - Câu "không khung benchmark nào ≥ 1.800 khung sau sự kiện cuối" **vẫn đúng** (max 1.137).
  - Căn cứ của (b′) (flicker trên VIRAT-P) **không bị ảnh hưởng**, vì VIRAT-P đọc bằng VideoCapture.

## 0.4 Chi phí

- Detector: **không cần dump lại**, vì re-index chính xác (6/6). Nếu vẫn muốn có corpus độc lập (P6, tùy chọn): 18.425 khung LASIESTA ÷ ≈ 2,5 fps (OV CPU FP32, WEIGHTS_EQUIV) ≈ **2 h CPU**. Cả 121 video ≈ 23 h, như ghi trong PROMPT.
- Chạy lại pipeline pre-registered, ước theo log Đợt 3–4:

  | Bước | Thời gian |
  |---|---|
  | cue LASIESTA | ~5 phút |
  | bench_units | ~10 phút |
  | meva_profile | 8 phút |
  | prereg_build (dự đoán, B = 1000) | 4 phút |
  | meva_outcomes | 1 phút |
  | e2_e5 | 2 phút |
  | e6 | 5 phút |
  | e7 | 25 phút |
  | numbers/paper | — |
  | **Tổng** | **≈ 1–1,5 h CPU**, không cần detector, không cần GPU |

## 0.5 Phương án

**(A) Sửa và chạy lại — ĐỀ XUẤT**, vì #09 nhiễm ở cả ba lớp và H4 nằm sát ngưỡng.

1. Thêm `code/frame_order.py`:
   - natural sort cho LASIESTA cộng phép hoán vị re-index cho detections_cpu và frame_scores (đọc file gốc chỉ-đọc, sắp lại trong bộ nhớ, không sửa file của Deep3);
   - assert "thứ tự chữ = thứ tự số" cho mọi dataset khác (checklist B10 của #17).
2. `cue.py` đọc LASIESTA theo thứ tự số. Tạo cache mới `bench_cue_v2.parquet` và giữ bản cũ để truy vết.
3. Chạy lại theo đúng thứ tự, cùng code, cùng ngưỡng, cùng seed; chỉ đổi input: bench_units → operating_points → strata_map/w_B/m_B → ω → prereg_build (dự đoán) → meva_outcomes → E2–E7 → numbers.json.
4. `prereg/AMENDMENT_01.md`:
   - lý do: lỗi thứ tự khung LASIESTA, phát hiện bởi đề tài #17 ngày 27-09, áp vào #09 ngày 03-10, **sau khi đã mở khóa kết cục (01-10) và trước khi nộp**;
   - phạm vi: chỉ input của 48 video, không đổi ngưỡng, code hay quy tắc;
   - cam kết báo cả hai bộ kết quả. Bộ **corrected** là bộ chính trong bài; bộ gốc nằm ở supplement/CSV.
   - Băm SHA-256 file amendment và commit **trước** khi chạy lại.
   - Ghi chú trung thực: dự đoán corrected được sinh **sau** khi đã thấy kết cục MEVA của bộ gốc, nên bộ corrected **không còn là dự đoán mù**. Bài phải nói rõ điều này. Bộ gốc vẫn là bằng chứng pre-registered thật, có hash `209f2e22…`.
5. Paper: E1 LASIESTA, census, toàn bộ số H2–H4/E3–E7 lấy từ numbers.json corrected. Nếu H4 đổi kết luận, phải sửa câu headline.

**(B) Không làm gì:** không áp dụng, vì #09 đã nhiễm.

### Việc cần anh Đạt quyết

1. Chọn A hay B. Em đề xuất A.
2. Với A: H4 sát ngưỡng có thể đổi kết luận. Anh xác nhận trước quy tắc báo cáo: **bộ corrected là kết luận chính, bộ gốc (pre-registered mù) báo song song.** Nếu hai bộ khác kết luận ở một H nào đó thì bài ghi cả hai, không chọn bộ thuận lợi.
3. P6 (dump lại detector) giữ là tùy chọn, vì re-index đã được kiểm chính xác.
