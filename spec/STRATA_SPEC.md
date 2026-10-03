# STRATA_SPEC v2 — đơn vị, tầng, cue, chính sách, metric (Đợt 2, cố định TRƯỚC khi tính bất kỳ hồ sơ MEVA nào)

Phiên bản: **v2** (2026-09-27; anh Đạt duyệt Q-A..Q-D sau báo cáo Đợt 2 tạm). v1 = commit `7c2f426` (giữ làm độ nhạy E6).
File này được băm SHA-256 (`STRATA_SPEC.md.sha256`) và commit **trước** khi chạy `code/meva_profile.py` (chưa có hồ sơ MEVA nào được tính trước v2).
Code: `code/strata.py`, `code/cue.py`, `code/policies.py`, `code/bench_units.py`.

**Thay đổi v1 → v2:** (1) định nghĩa transport chính (b′) = (b) nối khe ≤ 0,2 s (§1); (2) luật gộp tầng thưa "pooled", thứ tự R → D → C (§4); luật v1 giữ làm E6, báo cả hai ω; (3) R-cens/D-cens là tầng thường được vận chuyển, cận ĐL1(b) chỉ làm độ nhạy (§3); (4) false-open báo **theo khung** là chính, /giờ là quy đổi (§7); (5) P1 pre-register K ∈ {2, 5, 10, 30} thay cho K* (§6); (6) D mịn chỉ ở E6.

## 1. Chuỗi và "đoạn" (segment)
- **Đoạn** = đơn vị mà mọi trạng thái được khởi động lại: một clip benchmark (vùng temporalROI của CDnet; toàn bộ clip với LASIESTA/BMC/VIRAT-P), hoặc một **block** MEVA (chuỗi clip 5 phút liên tục theo `MEVA_selection` v1).
- Trong block MEVA, khung được nối theo thứ tự clip. Tại mối nối giữa hai clip: nếu khe thời gian (start clip sau − end clip trước) ≤ 2 s thì coi là liên tục (cue sai khác tính qua mối nối bằng ảnh thu nhỏ khung cuối/khung đầu); nếu > 2 s (các "tiny gap" 300 s đã liệt kê trong MEVA_selection) thì block được **tách thành hai đoạn** tại đó.
- g_t = chỉ báo sự kiện theo khung. **(b)** person (COCO cls 0) conf ≥ 0,7 từ YOLO26s-seg (conf 0,25, imgsz 640).
- **(b′): định nghĩa transport CHÍNH, dùng ở cả hai phía.** Lấy (b) rồi lấp mọi khe idle dài ≤ k khung nằm **giữa** hai run hoạt động trong cùng đoạn, với k = round(0,2 s × fps của nguồn): BMC/CDnet/LASIESTA (25 fps, gồm cả giả định) k = 5; VIRAT-P và MEVA (30 fps) k = 6; CDnet lowFramerate (0,17–1 fps) k = 0 (không nối). Khe chạm biên đoạn không lấp. Hiện thực: hàm `strata.merge_gaps`.
  *Căn cứ (VIRAT-P, 9/10 video có nhãn đối tượng chính thức VIRAT 2.0; `code/viratp_flicker.py`, `results/viratp_flicker.json`):*
  - 557/1.097 = **50,8 %** sự kiện (b) dài 1–3 khung.
  - Trong số sự kiện 1–3 khung, **19,0 %** không trùng GT (không có box GT person nào ở mọi khung của sự kiện); nếu đòi khớp không gian IoU ≥ 0,3 thì 35,4 %. Với sự kiện dài hơn, hai tỉ lệ này là 18,3 % và 32,6 %, tức gần như bằng.
  - → Sự kiện ngắn chủ yếu là **mảnh vỡ của track người thật** (conf dao động quanh 0,7), không phải ảo giác của detector.
  - (b′) giảm số sự kiện từ 1.097 xuống 306.
  - Lưu ý: VIRAT chỉ gán nhãn đối tượng chuyển động. VIRAT_S_000200_05 không có nhãn đối tượng chính thức nên bị loại; VIRAT_S_000200_03 không có nhãn person nào.
- (b) thô giữ trên trục E6. (a) has-box và (c) nhãn hoạt động MEVA chỉ dùng cho census/mô tả.
- **Sự kiện** = run cực đại g = 1 trong một đoạn. Sự kiện **bị kiểm duyệt** nếu chạm khung đầu hoặc khung cuối của đoạn.

## 2. Đơn vị u
- u = cửa sổ **300 khung** liên tiếp trong một đoạn, bắt đầu từ khung 0 của đoạn; cửa sổ cuối có thể ngắn hơn và được giữ nguyên. Mọi trọng số tính theo **khung** (w(h) = tỉ lệ khung thuộc tầng h), nên cửa sổ ngắn tự nhận trọng số nhỏ.
- Giây chỉ để mô tả: fps CDnet lowFramerate lấy từ tên (port 0,17 · tunnelExit 0,35 · turnpike 0,5 · tramCrossroad 1 fps); BMC từ mp4 (25); VIRAT-P từ mp4 (30); CDnet còn lại và LASIESTA giả định 25 fps (độ nhạy 15/30); MEVA đọc fps từ header.

## 3. Tầng h(u) = (S, D, R, C)
| Thành phần | Lớp | Định nghĩa |
|---|---|---|
| S — trạng thái hoạt động | idle / partial / full | tỉ lệ khung g = 1 trong u bằng 0 / trong (0, 1) / bằng 1 |
| D — lớp thời lượng | none / ≤5 / 6–30 / >30 khung | thời lượng (trên toàn đoạn) của sự kiện **dài nhất** giao với u; none nếu S = idle |
| R — khung từ hoạt động gần nhất | <300 / 300–1.800 / 1.800–18.000 / >18.000 | số khung từ khung g = 1 cuối cùng **trước** khung đầu của u (0 nếu sự kiện đang diễn ra) |
| C — cờ kiểm duyệt | 0 / 1 | 1 nếu u giao một sự kiện bị kiểm duyệt **hoặc** lịch sử R bị kiểm duyệt (xem dưới) |

**Lịch sử bị kiểm duyệt.** Nếu từ đầu đoạn tới đầu u chưa có khung g = 1, chỉ biết R ≥ e (e = số khung đã trôi trong đoạn). Nếu lớp của e là >18.000 thì R được xác định; ngược lại u mang nhãn **"R-cens(lớp(e))"** (C = 1) và **không** bị gán bừa vào một lớp R.
**D bị kiểm duyệt.** Sự kiện chạm biên có d_obs ≤ thời lượng thật. Nếu d_obs > 30 thì D = >30 được xác định; ngược lại u mang nhãn "D-cens(lớp(d_obs))", C = 1.
**v2, xử lý chính:** R-cens và D-cens là **tầng thường, quan sát được, được vận chuyển**. Lý do: cơ chế "đầu đoạn, chính sách vừa khởi động lại" giống nhau ở benchmark và MEVA, nên A1 áp dụng được cho chính trạng thái quan sát được này. **Độ nhạy:** coi chúng là tập tương thích và lấy cận ĐL1(b) (inf/sup trên các tầng R/D tương thích).

## 4. Quy tắc gộp tầng thưa (cố định, chỉ dựa trên benchmark, ghi trước khi nhìn MEVA)
1. Tầng h có **hỗ trợ** nếu trên benchmark có ≥ 20 cửa sổ từ ≥ 5 video khác nhau.
2. **v2 ("pooled", luật chính), thứ tự R → D → C** (`strata.build_strata_cells(bw, "v2")`).
   - Tầng có hỗ trợ là một ô riêng.
   - Tầng thô không đủ hỗ trợ nhưng có ≥ 1 cửa sổ benchmark được gán vào ô gồm **mọi** tầng thô benchmark có cùng khóa thô hóa (kể cả anh em có hỗ trợ; các anh em đó vẫn giữ ô riêng của mình). Việc gán diễn ra ở bước đầu tiên mà ô đó đủ hỗ trợ:
     - bước 1 (R): gộp R <300 với 300–1.800;
     - bước 2 (R + D): gộp thêm D ≤5 với 6–30 (và D-cens tương ứng);
     - bước 3 (R + D + C): gộp thêm C = 1 vào C = 0 (chỉ với tầng có R, D đã xác định).
   - *Lý do thứ tự:*
     - Gộp R trước vì bộ nhớ của các chính sách minh họa (≤ 15 + 4 khung, rất nhỏ so với 300) ít nhạy nhất với R, nên gộp R tốn ít tính hợp lệ A1 nhất.
     - D quyết định trực tiếp xác suất bắt sự kiện (P1: min(1, d/K)), nên gộp sau.
     - C (kiểm duyệt) thay đổi thời lượng quan sát được, nên gộp cuối.
3. Tầng vẫn không đủ hỗ trợ, hoặc **không có cửa sổ benchmark nào**, thì thuộc **U**. Dự kiến mọi R ≥ 1.800 rơi vào đây, vì run idle dài nhất trên benchmark ≈ 1.235 khung. **ω = khối khung đích thuộc U.** Không gộp tầng không có cửa sổ benchmark vào tầng có hỗ trợ.
4. **v1 (E6):** luật cũ, chỉ gộp các tầng thưa với nhau theo thứ tự D → C → R (`strata.build_strata_map`). Báo **cả hai ω** (v2 và v1).
5. Bảng ô xuất ở `results/strata_map_benchmark.csv` (cột `rule`, `definition`) và dùng nguyên cho MEVA. Trên dữ liệu tổng hợp, ω trung bình là 0,006 với v2 so với 0,126–0,144 với v1 (`results/synthetic_check.json`).

## 5. Cue d_t (giống hệt trên benchmark và MEVA)
- Khung → xám → resize **giữ tỉ lệ**, cạnh ngắn = 180 (INTER_AREA khi thu, INTER_LINEAR khi phóng).
- mad_t = trung bình |I_t − I_{t−1}|; fpx_t = tỉ lệ pixel có |I_t − I_{t−1}| > 15. Khung đầu đoạn: không có cue.
- Chuẩn hóa **nhân quả**: m_t = trung vị của mad trên tối đa 1.800 khung trước t (chỉ trong đoạn, không gồm t); d̃_t = mad_t / max(m_t, 0,1). Tương tự f̃_t = fpx_t / max(trung vị fpx, 1e-4). Warm-up: khi có < 1.800 khung trước, dùng trung vị của các khung hiện có; t = 1 dùng chính mad_1 (d̃ = 1).
- Benchmark: chỉ số khung phải khớp `frame_scores.parquet` (assert).

## 6. Chính sách (khởi động lại ở đầu mỗi đoạn; khung 0 của đoạn luôn được xử lý)
- **P1** periodic-K: xử lý khung t nếu (t − φ) mod K = 0; kết quả lấy trung bình trên mọi pha φ = 0..K−1. **Pre-register 4 ô K ∈ {2, 5, 10, 30}** cho H2/H3. Không dùng điểm iso-recall, vì với (b) nó cho K* = 1 (tầm thường). Dải K = 2..30 đưa vào phụ lục.
- **P2** ngưỡng: xử lý nếu d̃_t > τ₂ **hoặc** đã bỏ qua 15 khung liên tiếp (backstop Kr = 15).
- **P3** = P2 + giữ: nếu khung xử lý t có g_t = 1 thì xử lý thêm W = 4 khung kế tiếp.
- **P4** π̂: logistic trên đặc trưng nhân quả [log d̃_t, log f̃_t, max d̃ trên 5 khung gần nhất, trung bình d̃ trên 30 khung gần nhất] dự báo g_t(b′); fit MỘT LẦN trên 50 % video benchmark (chọn bằng seed 42, phân tầng theo dataset), đóng băng; xử lý nếu π̂_t > τ₄ hoặc backstop 15.
- Điểm vận hành: τ₂, τ₃, τ₄ chọn trên benchmark (121 video) là τ lớn nhất cho **event recall gộp (b′) ≥ 0,90**. Kết quả ghi vào `results/operating_points.json` và đóng băng; bản tính trên (b) ở Đợt 2 tạm được lưu tại `results/operating_points_v1_b.json`. Trục E6-(b) dùng cùng các τ đã đóng băng.
- P1 là đối chứng dương cho A1 (hành vi chỉ phụ thuộc thời gian sự kiện).

## 7. Metric theo cửa sổ (tử số / mẫu số theo khung, cộng được)
| Metric | Tử | Mẫu |
|---|---|---|
| processed fraction (= 1 − skip) | số khung xử lý P | số khung n |
| frame recall | số khung xử lý có g = 1 (TP) | số khung g = 1 (E) |
| event recall | số sự kiện **bắt đầu** trong u có ≥ 1 khung được xử lý | số sự kiện bắt đầu trong u |
| **false-open / khung** (chính) | số run xử lý cực đại **bắt đầu** trong u không chứa khung g = 1 | n |
| precision | TP | P |
False-open / giờ chỉ là quy đổi: (false-open / khung) × fps đích × 3.600.
Ước lượng transport dạng tỉ số: M̂_T = Σ_h w_T(h)·ν_B(h) / Σ_h w_T(h)·δ_B(h), với ν_B, δ_B là tử/mẫu **trên mỗi khung** trong tầng h của benchmark.
