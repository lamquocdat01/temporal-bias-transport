# AMENDMENT 01 — kết quả: bộ gốc (pre-registered mù) và bộ corrected (thứ tự khung LASIESTA đúng)

Sinh tự động bởi `code/amendment_compare.py` từ `results/orig/` và `results/corrected/`. Pre-registration: gốc `209f2e22…`, corrected `a4fa70a1…`. Theo quy tắc trong `prereg/AMENDMENT_01.md` §6, bộ corrected là kết luận chính; bộ gốc được báo song song.

## 1. H1–H4

| H | ngưỡng | quan sát — gốc | quan sát — corrected | kết luận gốc / corrected | p Holm gốc / corrected |
|---|---|---|---|---|---|
| H1 | TV >= 0.3 in >= 2/3 defs; refuted if duty in CI for >= 2/3 | TV a 0.522 / b' 0.608 / c 0.607; duty in CI 2/3 | TV a 0.563 / b' 0.610 / c 0.608; duty in CI 2/3 | **FAIL / FAIL** | 0,0040 / 0,0040 |
| H2 | >= 1 metric with (A) and (B) in a majority of policy cells | processed_frac: A&B 0/3 (need 2); event_recall: A&B 0/7 (need 4); false_open_per_frame: A&B 0/7 (need 4) | processed_frac: A&B 0/3 (need 2); event_recall: A&B 0/7 (need 4); false_open_per_frame: A&B 0/7 (need 4) | **FAIL / FAIL** | 0,0609 / 0,0425 |
| H3 | pass rate >= 0.80 of 192 cells | 81/192 = 0.422; median error ratio 0.674; PI coverage 0.797 | 76/192 = 0.396; median error ratio 0.688; PI coverage 0.771 | **FAIL / FAIL** | 0,9875 / 0,9985 |
| H4 | AUC(TV/r* -> flip) >= 0.75 | AUC 0.739 [0.635, 0.830]; flips 130/618; margin-score AUC 0.751 | AUC 0.676 [0.566, 0.780]; flips 101/618; margin-score AUC 0.701 | **FAIL / FAIL** | 0,0040 / 0,0040 |

Kết luận khác nhau giữa hai bộ: **không có** (cả bốn H giống nhau ở hai bộ).

## 2. Ô H3 đạt theo chính sách / metric

| nhóm | n | đạt gốc | đạt corrected | tỉ lệ gốc | tỉ lệ corrected | trung vị tỉ số sai số gốc / corrected |
|---|---|---|---|---|---|---|
| H3 denominator | 192 | 81 | 76 | 0,422 | 0,396 | 0,674 / 0,688 |
| metric processed_frac | 24 | 2 | 0 | 0,083 | 0,000 | 0,838 / 0,904 |
| metric frame_recall | 7 | 0 | 0 | 0,000 | 0,000 | 1,302 / 1,103 |
| metric event_recall | 49 | 6 | 5 | 0,122 | 0,102 | 0,745 / 0,877 |
| metric false_open_per_frame | 56 | 34 | 32 | 0,607 | 0,571 | 0,231 / 0,195 |
| metric precision | 56 | 39 | 39 | 0,696 | 0,696 | 0,218 / 0,164 |
| policy P1_K2 | 23 | 16 | 16 | 0,696 | 0,696 | 0,212 / 0,164 |
| policy P1_K5 | 23 | 18 | 17 | 0,783 | 0,739 | 0,214 / 0,161 |
| policy P1_K10 | 23 | 17 | 17 | 0,739 | 0,739 | 0,216 / 0,163 |
| policy P1_K30 | 23 | 18 | 18 | 0,783 | 0,783 | 0,197 / 0,146 |
| policy P2 | 31 | 4 | 3 | 0,129 | 0,097 | 0,826 / 0,864 |
| policy P3 | 31 | 5 | 4 | 0,161 | 0,129 | 0,792 / 0,874 |
| policy P4 | 38 | 3 | 1 | 0,079 | 0,026 | 0,883 / 0,971 |

## 3. Điểm vận hành, ω, cận

| đại lượng | gốc | corrected |
|---|---|---|
| tau2 | 0.9984 | 0.9767 |
| tau3 | 0.9978 | 0.9683 |
| tau4 | 0.2266 | 0.2249 |
| ω (b′) S8 v2 / v1 | 0,514 / 0,615 | 0,514 / 0,615 |
| max \|Δ\| ω_v2, ω_v1, π_pool trên mọi ô camera × phiên | — | 0 |
| block pooling-dependent | G331_2018-03-07_165000 | G331_2018-03-07_165000 |
| E4: vi phạm cận oracle | 0 | 0 |

## 4. E6 (tỉ lệ đạt H3-ô theo trục)

| trục | gốc | corrected | trung vị tỉ số sai số gốc / corrected |
|---|---|---|---|
| main | 0,422 | 0,396 | 0,674 / 0,688 |
| b_raw | 0,458 | 0,385 | 0,620 / 0,759 |
| rule_v1 | 0,438 | 0,417 | 0,602 / 0,641 |
| D_fine | 0,464 | 0,411 | 0,585 / 0,724 |
| window_1800 | 0,494 | 0,438 | 0,518 / 0,772 |
| norm_300 | 0,469 | 0,406 | 0,570 / 0,715 |
| cue_baseline | 0,500 | 0,484 | 0,502 / 0,552 |
| fps_15 | 0,411 | 0,396 | 0,674 / 0,774 |
| fps_30 | 0,458 | 0,438 | 0,591 / 0,624 |
| route_laptop | 0,422 | 0,396 | 0,674 / 0,688 |

## 5. E7 (khám phá, không thuộc pre-registration)

| đại lượng | gốc | corrected |
|---|---|---|
| exceed_tau2_transported | 0,488 | 0,570 |
| exceed_tau2_MEVA | 0,280 | 0,300 |
| within_gap | P2: -0,197 / P3: -0,188 / P4: -0,321 | P2: -0,257 / P3: -0,268 / P4: -0,334 |
| within_gap_cue | P2: -0,208 / P3: -0,210 / P4: -0,347 | P2: -0,270 / P3: -0,297 / P4: -0,363 |
| within_gap_backstop | P2: 0,011 / P3: 0,012 / P4: 0,025 | P2: 0,013 / P3: 0,014 / P4: 0,029 |
| memory_TV | 300: 0,476 / 1800: 0,475 / 18000: 0,446 | 300: 0,476 / 1800: 0,475 / 18000: 0,446 |
