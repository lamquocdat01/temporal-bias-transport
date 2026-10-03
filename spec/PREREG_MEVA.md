# PREREG_MEVA — Đề tài 09, CP1 pre-registration (MEVA S8, definition (b'), STRATA_SPEC v2)

Tạo lúc **2026-10-01T15:36:16+07:00** · HEAD lúc dựng `5c83c51762` · bản máy đọc: `prereg/PREREG_MEVA.json` (file này là bản người đọc, sinh tự động từ cùng dữ liệu). Băm SHA-256 của cả hai file nằm ở `prereg/*.sha256`.

**Luật:** sau khi băm + commit, KHÔNG sửa file này. Mọi thay đổi bắt buộc → `prereg/AMENDMENT_nn.md` (lý do, thời điểm, đã thấy kết cục nào chưa), băm, commit; báo cả kết quả trước và sau.

## 1. Thiết kế (đối tượng của pre-registration)

- **target**: MEVA S8 (plan 1101322c...; 10 blocks, 5 cameras x 3 sessions, 1,062,229 frames), detector route cuda_fp32_kaggle (gate tier event)
- **event_definition**: (b') on BOTH sides: person conf >= 0.7, idle gaps <= round(0.2 s x fps) merged inside a segment (MEVA 30 fps: k = 6)
- **strata**: STRATA_SPEC v2 (pooled R -> D -> C), cells from results/strata_map_benchmark.csv (bprime, v2, main)
- **operating_points**: results/operating_points.json (frozen; NOT re-selected)
- **units**: camera (5) and session (3) of S8; 'all' = summary; the pooling-dependent block is its own unit
- **transportable_part**: frames in cells != U (rule v2) of non-pooling-dependent blocks
- **naive**: pooled ratio over all windows of the 121 main benchmark videos (incl. benchmark U windows)
- **transport**: sum_h w_T(h) nu_B(h) / sum_h w_T(h) delta_B(h) over supported cells (Theorem 1(a))
- **PI**: for the OBSERVED value M_obs: transport +- t(0.975, df=2) = 4.3027 x sqrt(v_boot + v_obs); v_boot = variance of the benchmark-video cluster bootstrap with w_T fixed at the observed profile; v_obs = sum_h nwin_T(h) Var_B(N - M Q | h) / Dhat^2 (Dhat transported) -- the form validated in synthetic_check.py (COND), with 1.96 replaced by t_2 (Q-D/D3.6). B = 1000, seed 42
- **CI_secondary**: percentile CI with the unit's blocks resampled jointly with benchmark videos (block bootstrap, D3.6); also CI_twoway for M_T(w_T) (camera x session two-way for 'all'; for camera / session units the one available dimension = the unit's blocks) and CI_cond (benchmark only)
- **bounds**: Theorem 1(b) closed forms on ALL frames of the unit (incl. U and the pooling-dependent block): supported cells by plug-in, U windows by per-window logical boxes (processed >= floor(n/16)/n for P2-P4, exact phase-averaged count for P1; false-open <= ceil(n/2)/n, <= processed for P1; recall numerator in [0, e]; precision closed form). Also the PREREG_DECISIONS P-2 form [(1-w) M, (1-w) M + w] (width = omega).
- **observed**: M_obs = sum N / sum Q over the unit's transportable windows (computed only after unlock)
- **false_open_per_hour**: false-open / frame x 30 fps x 3600 (MEVA header fps)
- **chính sách**: P1_K2 = periodic K = 2, averaged over phases; P1_K5 = periodic K = 5, averaged over phases; P1_K10 = periodic K = 10, averaged over phases; P1_K30 = periodic K = 30, averaged over phases; P2 = P2; P3 = P3; P4 = P4
- **metric**: processed fraction; frame recall; event recall; false-open / khung; precision
- **block pooling-dependent**: G331_2018-03-07_165000

## 2. H1 — confirmatory-of-profile (observed before pre-registration; not a blind prediction)

TV(w_B, w_T) >= 0.3 in >= 2 of 3 definitions {(a), (b'), (c)}; (c) census only, mapped on the (b') benchmark cells. Refuted (PLAN §3) if the MEVA duty cycle lies inside the benchmark 95 % CI (cluster bootstrap over the 121 videos, frame-weighted) for >= 2 of 3 definitions. H1 = PASS iff the TV criterion holds AND the refutation clause does not fire.

| định nghĩa | TV | CI hai chiều | p(TV* ≤ 0,3) | duty MEVA S8 | duty benchmark [CI 95 %] | trong CI? |
|---|---|---|---|---|---|---|
| a → bench (a) | 0.522 | [0.436; 0.698] | 0.0010 | 0.717 | 0.759 [0.674; 0.843] | có |
| bprime → bench (b') | 0.608 | [0.520; 0.786] | 0.0010 | 0.175 | 0.245 [0.173; 0.317] | có |
| c → bench (b') [cross-definition] | 0.607 | [0.554; 0.749] | 0.0010 | 0.159 | 0.245 [0.173; 0.315] | không |

Thành phần TV: **PASS** (3/3) · điều khoản bác bỏ (duty trong CI): **fires** (2/3) → H1 (đã quan sát được): **FAIL**; p_H1 (thành phần TV) = 0.0010.

Both components are profile facts already seen. The TV part holds (3/3) while the duty-cycle refutation clause fires for (a) and (b'): benchmark and MEVA differ in temporal COMPOSITION (strata: idle-run lengths, censoring) far more than in duty cycle. Flag for Dat: the PLAN wording makes H1 FAIL on duty cycle although TV >= 0.3 in every definition.

## 3. H2

At the 'all' unit (transportable part of S8): for policy x metric with metric in {processed fraction, event recall, false-open/frame}, deterministic-by-construction cells (P1 x processed) excluded: (A) |naive - M_obs| > PI half-width and (B) composition share = (transport - naive) / (M_obs - naive) >= 0.5 (Kitagawa-Oaxaca, ratio form, benchmark reference). H2 PASS if, for >= 1 of the 3 metrics, (A) and (B) hold for a majority of its eligible policy cells (processed: P2-P4, need 2 of 3; event recall and false-open: 7 cells, need 4 of 7).

p: per policy x metric: two-sided t test of bias / (PI_half / t_crit), df = 2; per metric p_m = k-th smallest p (k = required majority); p_H2 = min(1, 3 x min_m p_m) (Bonferroni over the 3 metrics)

## 4. H3

Binding (D3.3): |transport - M_obs| <= 0.5 |naive - M_obs| in >= 80 % of the H3 denominator cells. Coverage of the PI is reported only.

Mẫu số: camera and session units x 7 policies x 5 metrics, transportable part only; EXCLUDED and listed: cells whose transport is undefined from the profile (no transported denominator mass, e.g. recall where every supported window is idle); also EXCLUDED and reported separately: (i) the pooling-dependent block (D3.2) [its frames are removed from camera P-4 and session 03-07]; (ii) expected-FAIL cells P2/P3 x frame recall (D3.5); (iii) deterministic-by-construction cells P1(K) x {processed fraction, frame recall} (phase-averaged P1 processes each frame with probability exactly 1/K; naive, transport and observed all equal 1/K up to segment-start frames, so the error ratio is a clip-boundary artefact ~1e-4) -- added by Claude at CP1, flagged for Dat's review; H3 is also reported with (iii) included as a sensitivity row; (iv) any cell whose observed denominator is 0 (undefined), listed.

**Số ô mẫu số H3 = 192** (ngưỡng tỉ lệ đạt 0,80; tỉ số sai số 0,5). p: exact one-sided binomial P(X >= k | n, 0.5) on the denominator cells (descriptive: cells are not independent).

## 5. H4

AUC(score -> observed flip) >= 0.75 over pair x metric x unit (camera and session units); score = TV(w_B, w_T) / r*, with w_B, w_T renormalised over the supported cells (the ball of flip_radius.py) and w_T from the unit's transportable part; r* = inf -> score 0. Flip observed = sign of Delta on the unit's transportable part differs from sign_B (tie = flip).

r\*: LP flip_radius.py on the benchmark supported cells (NOT the closed form). Số cặp chính sách = 21, cặp × metric = 81, × 8 đơn vị = 648 lần đánh giá. if < 5 flips or < 5 non-flips, H4 = INCONCLUSIVE (reported as not PASS).

Loại khỏi H4: P1 vs P1 pairs for processed, frame recall, event recall, false-open (K-monotone ordering in every window: cannot flip; would only add guaranteed true negatives); kept for precision.

| cặp | metric | Δ_B | r\* | r\* cận dưới | vào H4 |
|---|---|---|---|---|---|
| P1_K2 vs P1_K5 | processed_frac | 0.2998 | inf | inf | không |
| P1_K2 vs P1_K5 | frame_recall | 0.3000 | 0.4497 | 0.4497 | không |
| P1_K2 vs P1_K5 | event_recall | 0.1155 | inf | inf | không |
| P1_K2 vs P1_K5 | false_open_per_frame | 0.2261 | inf | inf | không |
| P1_K2 vs P1_K5 | precision | 0.0004 | 0.1986 | 0.0002 | có |
| P1_K2 vs P1_K10 | processed_frac | 0.3997 | inf | inf | không |
| P1_K2 vs P1_K10 | frame_recall | 0.3999 | 0.4497 | 0.4497 | không |
| P1_K2 vs P1_K10 | event_recall | 0.2050 | inf | inf | không |
| P1_K2 vs P1_K10 | false_open_per_frame | 0.3014 | 0.9515 | 0.9515 | không |
| P1_K2 vs P1_K10 | precision | 0.0011 | 0.1986 | 0.0005 | có |
| P1_K2 vs P1_K30 | processed_frac | 0.4664 | inf | inf | không |
| P1_K2 vs P1_K30 | frame_recall | 0.4666 | inf | inf | không |
| P1_K2 vs P1_K30 | event_recall | 0.3483 | inf | inf | không |
| P1_K2 vs P1_K30 | false_open_per_frame | 0.3517 | inf | inf | không |
| P1_K2 vs P1_K30 | precision | 0.0036 | 0.1986 | 0.0018 | có |
| P1_K2 vs P2 | processed_frac | -0.0246 | 0.1278 | 0.1278 | có |
| P1_K2 vs P2 | frame_recall | -0.1058 | 0.2735 | 0.2735 | có |
| P1_K2 vs P2 | event_recall | -0.0067 | 0.0102 | 0.0102 | có |
| P1_K2 vs P2 | false_open_per_frame | 0.2963 | inf | inf | có |
| P1_K2 vs P2 | precision | -0.0379 | 0.1878 | 0.0187 | có |
| P1_K2 vs P3 | processed_frac | -0.1229 | 0.3118 | 0.3118 | có |
| P1_K2 vs P3 | frame_recall | -0.4819 | 0.4497 | 0.4497 | có |
| P1_K2 vs P3 | event_recall | -0.0048 | 0.0072 | 0.0072 | có |
| P1_K2 vs P3 | false_open_per_frame | 0.2976 | 0.9515 | 0.9515 | có |
| P1_K2 vs P3 | precision | -0.1412 | 0.4497 | 0.0604 | có |
| P1_K2 vs P4 | processed_frac | -0.1536 | inf | inf | có |
| P1_K2 vs P4 | frame_recall | -0.2260 | 0.4497 | 0.4497 | có |
| P1_K2 vs P4 | event_recall | -0.0067 | 0.0170 | 0.0170 | có |
| P1_K2 vs P4 | false_open_per_frame | 0.3362 | inf | inf | có |
| P1_K2 vs P4 | precision | -0.0271 | 0.1114 | 0.0139 | có |
| P1_K5 vs P1_K10 | processed_frac | 0.0999 | inf | inf | không |
| P1_K5 vs P1_K10 | frame_recall | 0.1000 | 0.4497 | 0.4497 | không |
| P1_K5 vs P1_K10 | event_recall | 0.0895 | inf | inf | không |
| P1_K5 vs P1_K10 | false_open_per_frame | 0.0754 | 0.9515 | 0.9515 | không |
| P1_K5 vs P1_K10 | precision | 0.0007 | 0.1986 | 0.0003 | có |
| P1_K5 vs P1_K30 | processed_frac | 0.1666 | inf | inf | không |
| P1_K5 vs P1_K30 | frame_recall | 0.1666 | inf | inf | không |
| P1_K5 vs P1_K30 | event_recall | 0.2328 | inf | inf | không |
| P1_K5 vs P1_K30 | false_open_per_frame | 0.1256 | 0.9515 | 0.9515 | không |
| P1_K5 vs P1_K30 | precision | 0.0033 | 0.1986 | 0.0016 | có |
| P1_K5 vs P2 | processed_frac | -0.3244 | inf | inf | có |
| P1_K5 vs P2 | frame_recall | -0.4058 | 0.4497 | 0.4497 | có |
| P1_K5 vs P2 | event_recall | -0.1222 | 0.4012 | 0.4012 | có |
| P1_K5 vs P2 | false_open_per_frame | 0.0702 | inf | inf | có |
| P1_K5 vs P2 | precision | -0.0383 | 0.1900 | 0.0189 | có |
| P1_K5 vs P3 | processed_frac | -0.4227 | inf | inf | có |
| P1_K5 vs P3 | frame_recall | -0.7819 | 0.4497 | 0.4497 | có |
| P1_K5 vs P3 | event_recall | -0.1203 | 0.4012 | 0.4012 | có |
| P1_K5 vs P3 | false_open_per_frame | 0.0716 | inf | inf | có |
| P1_K5 vs P3 | precision | -0.1416 | 0.4497 | 0.0606 | có |
| P1_K5 vs P4 | processed_frac | -0.4534 | inf | inf | có |
| P1_K5 vs P4 | frame_recall | -0.5259 | 0.4497 | 0.4497 | có |
| P1_K5 vs P4 | event_recall | -0.1222 | inf | inf | có |
| P1_K5 vs P4 | false_open_per_frame | 0.1101 | 0.9515 | 0.9515 | có |
| P1_K5 vs P4 | precision | -0.0275 | 0.1125 | 0.0141 | có |
| P1_K10 vs P1_K30 | processed_frac | 0.0666 | inf | inf | không |
| P1_K10 vs P1_K30 | frame_recall | 0.0667 | inf | inf | không |
| P1_K10 vs P1_K30 | event_recall | 0.1434 | inf | inf | không |
| P1_K10 vs P1_K30 | false_open_per_frame | 0.0502 | inf | inf | không |
| P1_K10 vs P1_K30 | precision | 0.0026 | 0.1986 | 0.0013 | có |
| P1_K10 vs P2 | processed_frac | -0.4244 | inf | inf | có |
| P1_K10 vs P2 | frame_recall | -0.5058 | 0.4497 | 0.4497 | có |
| P1_K10 vs P2 | event_recall | -0.2117 | inf | inf | có |
| P1_K10 vs P2 | false_open_per_frame | -0.0051 | 0.0674 | 0.0674 | có |
| P1_K10 vs P2 | precision | -0.0390 | 0.1936 | 0.0192 | có |
| P1_K10 vs P3 | processed_frac | -0.5226 | inf | inf | có |
| P1_K10 vs P3 | frame_recall | -0.8818 | inf | inf | có |
| P1_K10 vs P3 | event_recall | -0.2098 | inf | inf | có |
| P1_K10 vs P3 | false_open_per_frame | -0.0038 | 0.0498 | 0.0498 | có |
| P1_K10 vs P3 | precision | -0.1423 | 0.4497 | 0.0609 | có |
| P1_K10 vs P4 | processed_frac | -0.5534 | inf | inf | có |
| P1_K10 vs P4 | frame_recall | -0.6259 | 0.4497 | 0.4497 | có |
| P1_K10 vs P4 | event_recall | -0.2117 | inf | inf | có |
| P1_K10 vs P4 | false_open_per_frame | 0.0348 | inf | inf | có |
| P1_K10 vs P4 | precision | -0.0282 | 0.1142 | 0.0144 | có |
| P1_K30 vs P2 | processed_frac | -0.4910 | inf | inf | có |
| P1_K30 vs P2 | frame_recall | -0.5724 | 0.4497 | 0.4497 | có |
| P1_K30 vs P2 | event_recall | -0.3550 | 0.4012 | 0.4012 | có |
| P1_K30 vs P2 | false_open_per_frame | -0.0554 | inf | inf | có |
| P1_K30 vs P2 | precision | -0.0416 | 0.2113 | 0.0205 | có |
| P1_K30 vs P3 | processed_frac | -0.5893 | inf | inf | có |
| P1_K30 vs P3 | frame_recall | -0.9485 | inf | inf | có |
| P1_K30 vs P3 | event_recall | -0.3531 | 0.4012 | 0.4012 | có |
| P1_K30 vs P3 | false_open_per_frame | -0.0540 | inf | inf | có |
| P1_K30 vs P3 | precision | -0.1449 | 0.4497 | 0.0619 | có |
| P1_K30 vs P4 | processed_frac | -0.6200 | inf | inf | có |
| P1_K30 vs P4 | frame_recall | -0.6926 | 0.4497 | 0.4497 | có |
| P1_K30 vs P4 | event_recall | -0.3550 | inf | inf | có |
| P1_K30 vs P4 | false_open_per_frame | -0.0154 | 0.5145 | 0.5145 | có |
| P1_K30 vs P4 | precision | -0.0308 | 0.1213 | 0.0157 | có |
| P2 vs P3 | processed_frac | -0.0983 | inf | inf | có |
| P2 vs P3 | frame_recall | -0.3761 | 0.4079 | 0.4079 | có |
| P2 vs P3 | event_recall | 0.0019 | 0.1758 | 0.1758 | có |
| P2 vs P3 | false_open_per_frame | 0.0013 | 0.3659 | 0.3659 | có |
| P2 vs P3 | precision | -0.1033 | 0.4045 | 0.0411 | có |
| P2 vs P4 | processed_frac | -0.1290 | 0.8544 | 0.8544 | có |
| P2 vs P4 | frame_recall | -0.1201 | inf | inf | có |
| P2 vs P4 | event_recall | 0.0000 | 0.0000 | 0.0000 | có |
| P2 vs P4 | false_open_per_frame | 0.0399 | inf | inf | có |
| P2 vs P4 | precision | 0.0108 | 0.0796 | 0.0052 | có |
| P3 vs P4 | processed_frac | -0.0307 | 0.0493 | 0.0493 | có |
| P3 vs P4 | frame_recall | 0.2559 | 0.3904 | 0.3904 | có |
| P3 vs P4 | event_recall | -0.0019 | 0.0032 | 0.0032 | có |
| P3 vs P4 | false_open_per_frame | 0.0386 | 0.9515 | 0.9515 | có |
| P3 vs P4 | precision | 0.1141 | 0.4497 | 0.0475 | có |

## 6. Bội số

Holm over p_H1..p_H4 at alpha = 0.05. Binding verdicts are the criteria above on point values; Holm-adjusted p is reported alongside (a criterion met without Holm significance is labelled 'PASS (not significant after Holm)').

## 7. Dự đoán theo ô (ước lượng ngây thơ, transport, PI 95 % t₂, CI phụ, cận Manski toàn MEVA)

Cột: ngây thơ · transport · PI · CI block · cận ĐL1(b) toàn đơn vị · cận P-2 · ω toàn đơn vị · vai trò · H3 dự kiến.

### camera — P-1 (khung transportable 92,998; ω toàn đơn vị 0.530)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.2348; 0.7652] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4637; 0.5363] | [0.2348; 0.7652] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.8929 | 0.9035 | [0.8308; 0.9763] | [0.8731; 0.9372] | [0.8137; 0.9131] | [0.4244; 0.9547] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3773 | 0.3689 | [0.3405; 0.3973] | [0.3408; 0.4118] | [0.1733; 0.4384] | [0.1733; 0.7036] | H3_denominator | PASS |
| P1_K2 | precision | 0.2452 | 0.2621 | [0.2053; 0.3190] | [0.1763; 0.3184] | [0.1231; 0.1394] | [0.1231; 0.6535] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0939; 0.6243] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1855; 0.2581] | [0.0939; 0.6243] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7731 | 0.8032 | [0.6883; 0.9182] | [0.7410; 0.8580] | [0.7233; 0.8228] | [0.3772; 0.9076] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1513 | 0.1476 | [0.1362; 0.1590] | [0.1363; 0.1648] | [0.0693; 0.1754] | [0.0693; 0.5997] | H3_denominator | PASS |
| P1_K5 | precision | 0.2448 | 0.2621 | [0.2053; 0.3189] | [0.1763; 0.3184] | [0.1231; 0.1464] | [0.1231; 0.6534] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0470; 0.5773] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0927; 0.1653] | [0.0470; 0.5773] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6805 | 0.7115 | [0.5815; 0.8414] | [0.6306; 0.7830] | [0.6407; 0.7402] | [0.3341; 0.8645] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0760 | 0.0738 | [0.0681; 0.0795] | [0.0682; 0.0824] | [0.0347; 0.0877] | [0.0347; 0.5650] | H3_denominator | PASS |
| P1_K10 | precision | 0.2442 | 0.2621 | [0.2053; 0.3189] | [0.1762; 0.3183] | [0.1231; 0.1474] | [0.1231; 0.6534] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0157; 0.5460] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0335 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0309; 0.1035] | [0.0157; 0.5460] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5326 | 0.5469 | [0.3667; 0.7271] | [0.4421; 0.6485] | [0.4925; 0.5920] | [0.2569; 0.7872] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0246 | [0.0227; 0.0265] | [0.0227; 0.0275] | [0.0116; 0.0292] | [0.0116; 0.5419] | H3_denominator | PASS |
| P1_K30 | precision | 0.2418 | 0.2619 | [0.2051; 0.3187] | [0.1761; 0.3181] | [0.1231; 0.1474] | [0.1230; 0.6533] | H3_denominator | PASS |
| P2 | processed_frac | 0.5276 | 0.4937 | [0.3788; 0.6086] | [0.4501; 0.5466] | [0.2637; 0.7622] | [0.2319; 0.7622] | H3_denominator | PASS |
| P2 | frame_recall | 0.6143 | 0.6201 | [0.3697; 0.8704] | [0.5461; 0.7502] | [0.5750; 0.6476] | [0.2912; 0.8216] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9009 | 0.8887 | [0.7807; 0.9967] | [0.8558; 0.9311] | [0.8003; 0.8998] | [0.4174; 0.9477] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0816 | 0.0829 | [0.0534; 0.1125] | [0.0678; 0.1024] | [0.0389; 0.3041] | [0.0389; 0.5693] | H3_denominator | PASS |
| P2 | precision | 0.2856 | 0.3292 | [0.2065; 0.4520] | [0.2084; 0.4160] | [0.1002; 0.3163] | [0.1546; 0.6850] | H3_denominator | PASS |
| P3 | processed_frac | 0.6240 | 0.5941 | [0.4917; 0.6966] | [0.5269; 0.6440] | [0.3109; 0.8094] | [0.2790; 0.8094] | H3_denominator | PASS |
| P3 | frame_recall | 0.9826 | 0.9838 | [0.9640; 1.0036] | [0.9768; 0.9908] | [0.9124; 0.9850] | [0.4620; 0.9924] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9009 | 0.8855 | [0.7796; 0.9914] | [0.8526; 0.9263] | [0.7974; 0.8969] | [0.4159; 0.9462] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0803 | 0.0819 | [0.0529; 0.1109] | [0.0669; 0.1016] | [0.0384; 0.3036] | [0.0384; 0.5688] | H3_denominator | PASS |
| P3 | precision | 0.3863 | 0.4341 | [0.3428; 0.5253] | [0.3020; 0.5056] | [0.1497; 0.4099] | [0.2039; 0.7342] | H3_denominator | PASS |
| P4 | processed_frac | 0.6581 | 0.5927 | [0.3926; 0.7927] | [0.4968; 0.6826] | [0.3102; 0.8087] | [0.2783; 0.8087] | H3_denominator | PASS |
| P4 | frame_recall | 0.7295 | 0.6908 | [0.4442; 0.9374] | [0.6232; 0.8275] | [0.6407; 0.7133] | [0.3245; 0.8548] | H3_denominator | PASS |
| P4 | event_recall | 0.9009 | 0.8897 | [0.7030; 1.0764] | [0.8279; 0.9659] | [0.8012; 0.9007] | [0.4179; 0.9482] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0407 | 0.0429 | [0.0242; 0.0616] | [0.0357; 0.0535] | [0.0202; 0.2853] | [0.0202; 0.5505] | H3_denominator | PASS |
| P4 | precision | 0.2719 | 0.3056 | [0.1905; 0.4207] | [0.1940; 0.3895] | [0.1052; 0.2975] | [0.1435; 0.6739] | H3_denominator | PASS |

### camera — P-2 (khung transportable 81,909; ω toàn đơn vị 0.621)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.1896; 0.8104] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.3606; 0.6394] | [0.1896; 0.8104] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.8929 | 0.8992 | [0.7759; 1.0224] | [0.8601; 0.9531] | [0.7032; 0.9211] | [0.3409; 0.9618] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3773 | 0.4528 | [0.4316; 0.4740] | [0.4364; 0.4686] | [0.1717; 0.4821] | [0.1717; 0.7925] | H3_denominator | PASS |
| P1_K2 | precision | 0.2452 | 0.0944 | [0.0521; 0.1367] | [0.0627; 0.1272] | [0.0358; 0.0581] | [0.0358; 0.6566] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0758; 0.6967] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1443; 0.4230] | [0.0758; 0.6967] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7731 | 0.8094 | [0.6226; 0.9961] | [0.7537; 0.8912] | [0.6330; 0.8509] | [0.3069; 0.9277] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1513 | 0.1811 | [0.1727; 0.1896] | [0.1746; 0.1875] | [0.0687; 0.1928] | [0.0687; 0.6895] | H3_denominator | PASS |
| P1_K5 | precision | 0.2448 | 0.0944 | [0.0521; 0.1367] | [0.0627; 0.1272] | [0.0358; 0.0669] | [0.0358; 0.6566] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0379; 0.6588] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0721; 0.3508] | [0.0379; 0.6588] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6805 | 0.7296 | [0.5408; 0.9183] | [0.6853; 0.8266] | [0.5706; 0.7885] | [0.2766; 0.8975] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0760 | 0.0906 | [0.0864; 0.0948] | [0.0873; 0.0938] | [0.0343; 0.0964] | [0.0343; 0.6552] | H3_denominator | PASS |
| P1_K10 | precision | 0.2442 | 0.0944 | [0.0521; 0.1367] | [0.0627; 0.1272] | [0.0358; 0.0689] | [0.0358; 0.6566] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0127; 0.6335] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0335 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0240; 0.3028] | [0.0126; 0.6335] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5326 | 0.5707 | [0.2953; 0.8460] | [0.4984; 0.7242] | [0.4463; 0.6642] | [0.2164; 0.8372] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0302 | [0.0288; 0.0316] | [0.0291; 0.0313] | [0.0115; 0.0322] | [0.0115; 0.6323] | H3_denominator | PASS |
| P1_K30 | precision | 0.2418 | 0.0943 | [0.0520; 0.1366] | [0.0626; 0.1271] | [0.0358; 0.0697] | [0.0358; 0.6566] | H3_denominator | PASS |
| P2 | processed_frac | 0.5276 | 0.4768 | [0.3476; 0.6060] | [0.4238; 0.5296] | [0.2180; 0.8016] | [0.1808; 0.8016] | H3_denominator | PASS |
| P2 | frame_recall | 0.6143 | 0.6969 | [0.4719; 0.9218] | [0.6325; 0.7685] | [0.5026; 0.7813] | [0.2642; 0.8851] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9009 | 0.8964 | [0.7680; 1.0249] | [0.8702; 0.9428] | [0.7011; 0.9190] | [0.3399; 0.9607] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0816 | 0.1052 | [0.0650; 0.1453] | [0.0847; 0.1203] | [0.0399; 0.3503] | [0.0399; 0.6607] | H3_denominator | PASS |
| P2 | precision | 0.2856 | 0.1380 | [0.0577; 0.2183] | [0.0885; 0.1919] | [0.0311; 0.1687] | [0.0523; 0.6732] | H3_denominator | PASS |
| P3 | processed_frac | 0.6240 | 0.5071 | [0.3772; 0.6370] | [0.4537; 0.5609] | [0.2295; 0.8131] | [0.1923; 0.8131] | H3_denominator | PASS |
| P3 | frame_recall | 0.9826 | 0.9770 | [0.9418; 1.0122] | [0.9656; 0.9878] | [0.7047; 0.9834] | [0.3704; 0.9913] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9009 | 0.8937 | [0.7675; 1.0198] | [0.8681; 0.9357] | [0.6989; 0.9168] | [0.3388; 0.9597] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0803 | 0.1046 | [0.0648; 0.1444] | [0.0846; 0.1193] | [0.0397; 0.3501] | [0.0397; 0.6605] | H3_denominator | PASS |
| P3 | precision | 0.3863 | 0.1819 | [0.0961; 0.2677] | [0.1184; 0.2455] | [0.0430; 0.2022] | [0.0690; 0.6898] | H3_denominator | PASS |
| P4 | processed_frac | 0.6581 | 0.5837 | [0.3385; 0.8289] | [0.4617; 0.6837] | [0.2586; 0.8422] | [0.2213; 0.8422] | H3_denominator | PASS |
| P4 | frame_recall | 0.7295 | 0.7654 | [0.4977; 1.0331] | [0.6849; 0.8618] | [0.5521; 0.8308] | [0.2902; 0.9110] | H3_denominator | PASS |
| P4 | event_recall | 0.9009 | 0.8964 | [0.6616; 1.1313] | [0.8245; 0.9751] | [0.7011; 0.9190] | [0.3399; 0.9607] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0407 | 0.0538 | [0.0304; 0.0772] | [0.0449; 0.0646] | [0.0204; 0.3308] | [0.0204; 0.6412] | H3_denominator | PASS |
| P4 | precision | 0.2719 | 0.1238 | [0.0569; 0.1907] | [0.0839; 0.1684] | [0.0325; 0.1525] | [0.0469; 0.6678] | H3_denominator | PASS |

### camera — P-3 (khung transportable 52,200; ω toàn đơn vị 0.758)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.1208; 0.8792] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.3157; 0.6843] | [0.1208; 0.8792] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.8929 | 0.9065 | [0.7353; 1.0777] | [0.8748; 0.9526] | [0.4151; 0.9572] | [0.2191; 0.9774] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3773 | 0.4787 | [0.4661; 0.4912] | [0.4378; 0.4936] | [0.1157; 0.4948] | [0.1157; 0.8740] | H3_denominator | PASS |
| P1_K2 | precision | 0.2452 | 0.0427 | [0.0176; 0.0677] | [0.0129; 0.1244] | [0.0103; 0.0201] | [0.0103; 0.7686] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2001] | [0.2000; 0.2001] | [0.2000; 0.2000] | [0.0483; 0.8067] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1263; 0.4949] | [0.0483; 0.8067] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7731 | 0.8090 | [0.5251; 1.0930] | [0.7666; 0.8943] | [0.3705; 0.9125] | [0.1955; 0.9538] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1513 | 0.1915 | [0.1865; 0.1965] | [0.1751; 0.1975] | [0.0463; 0.1979] | [0.0463; 0.8046] | H3_denominator | PASS |
| P1_K5 | precision | 0.2448 | 0.0427 | [0.0176; 0.0677] | [0.0129; 0.1244] | [0.0103; 0.0229] | [0.0103; 0.7686] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1001 | [0.1000; 0.1001] | [0.1000; 0.1001] | [0.1000; 0.1000] | [0.0242; 0.7825] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0631; 0.4317] | [0.0242; 0.7825] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6805 | 0.7189 | [0.4023; 1.0356] | [0.6736; 0.8161] | [0.3292; 0.8713] | [0.1737; 0.9321] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0760 | 0.0958 | [0.0933; 0.0983] | [0.0876; 0.0988] | [0.0231; 0.0990] | [0.0231; 0.7815] | H3_denominator | PASS |
| P1_K10 | precision | 0.2442 | 0.0426 | [0.0176; 0.0677] | [0.0129; 0.1244] | [0.0103; 0.0229] | [0.0103; 0.7686] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0081; 0.7664] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0335 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0210; 0.3896] | [0.0081; 0.7664] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5326 | 0.5525 | [0.1804; 0.9247] | [0.4920; 0.6795] | [0.2530; 0.7951] | [0.1335; 0.8919] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0320 | [0.0311; 0.0328] | [0.0292; 0.0330] | [0.0077; 0.0330] | [0.0077; 0.7661] | H3_denominator | PASS |
| P1_K30 | precision | 0.2418 | 0.0426 | [0.0176; 0.0676] | [0.0129; 0.1242] | [0.0103; 0.0231] | [0.0103; 0.7686] | H3_denominator | PASS |
| P2 | processed_frac | 0.5276 | 0.5015 | [0.3435; 0.6596] | [0.4302; 0.5654] | [0.1667; 0.8795] | [0.1212; 0.8795] | H3_denominator | PASS |
| P2 | frame_recall | 0.6143 | 0.6611 | [0.2815; 1.0407] | [0.5851; 0.7414] | [0.4174; 0.7860] | [0.1598; 0.9181] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9009 | 0.8716 | [0.6170; 1.1263] | [0.8431; 0.9295] | [0.3991; 0.9412] | [0.2106; 0.9690] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0816 | 0.0966 | [0.0607; 0.1326] | [0.0789; 0.1199] | [0.0233; 0.4025] | [0.0233; 0.7817] | H3_denominator | PASS |
| P2 | precision | 0.2856 | 0.0562 | [0.0083; 0.1042] | [0.0150; 0.1815] | [0.0077; 0.0746] | [0.0136; 0.7719] | H3_denominator | PASS |
| P3 | processed_frac | 0.6240 | 0.5185 | [0.3612; 0.6757] | [0.4519; 0.5778] | [0.1708; 0.8836] | [0.1253; 0.8836] | H3_denominator | PASS |
| P3 | frame_recall | 0.9826 | 0.9777 | [0.9266; 1.0288] | [0.9668; 0.9872] | [0.6174; 0.9859] | [0.2363; 0.9946] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9009 | 0.8683 | [0.6169; 1.1198] | [0.8400; 0.9228] | [0.3976; 0.9397] | [0.2098; 0.9682] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0803 | 0.0964 | [0.0605; 0.1322] | [0.0786; 0.1191] | [0.0233; 0.4025] | [0.0233; 0.7816] | H3_denominator | PASS |
| P3 | precision | 0.3863 | 0.0805 | [0.0289; 0.1320] | [0.0229; 0.2392] | [0.0114; 0.0914] | [0.0194; 0.7778] | H3_denominator | PASS |
| P4 | processed_frac | 0.6581 | 0.6168 | [0.3732; 0.8605] | [0.5041; 0.7315] | [0.1946; 0.9074] | [0.1491; 0.9074] | H3_denominator | PASS |
| P4 | frame_recall | 0.7295 | 0.7354 | [0.3206; 1.1503] | [0.6507; 0.8361] | [0.4644; 0.8330] | [0.1777; 0.9361] | H3_denominator | PASS |
| P4 | event_recall | 0.9009 | 0.8880 | [0.5651; 1.2110] | [0.8326; 0.9587] | [0.4066; 0.9487] | [0.2146; 0.9729] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0407 | 0.0511 | [0.0300; 0.0722] | [0.0429; 0.0606] | [0.0123; 0.3915] | [0.0123; 0.7707] | H3_denominator | PASS |
| P4 | precision | 0.2719 | 0.0509 | [0.0089; 0.0928] | [0.0136; 0.1580] | [0.0084; 0.0681] | [0.0123; 0.7706] | H3_denominator | PASS |

### camera — P-4 (khung transportable 36,000; ω toàn đơn vị 0.418)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.2909; 0.7091] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | nan | — | — | [0.4983; 0.5017] | [0.2909; 0.7091] | undefined | n/a |
| P1_K2 | event_recall | 0.8929 | nan | — | — | [0.7804; 0.9498] | [0.5466; 0.9648] | undefined | n/a |
| P1_K2 | false_open_per_frame | 0.3773 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.0840; 0.2931] | [0.0840; 0.5022] | H3_denominator | PASS |
| P1_K2 | precision | 0.2452 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4137; 0.4151] | [0.4137; 0.8319] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1164; 0.5346] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | nan | — | — | [0.1993; 0.2027] | [0.1164; 0.5346] | undefined | n/a |
| P1_K5 | event_recall | 0.7731 | nan | — | — | [0.7134; 0.8827] | [0.4997; 0.9179] | undefined | n/a |
| P1_K5 | false_open_per_frame | 0.1513 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0336; 0.1173] | [0.0336; 0.4518] | H3_denominator | PASS |
| P1_K5 | precision | 0.2448 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4137; 0.4151] | [0.4137; 0.8319] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1001 | [0.1000; 0.1001] | [0.1001; 0.1001] | [0.1000; 0.1000] | [0.0582; 0.4764] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | nan | — | — | [0.0997; 0.1030] | [0.0582; 0.4764] | undefined | n/a |
| P1_K10 | event_recall | 0.6805 | nan | — | — | [0.6584; 0.8277] | [0.4611; 0.8793] | undefined | n/a |
| P1_K10 | false_open_per_frame | 0.0760 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0168; 0.0586] | [0.0168; 0.4350] | H3_denominator | PASS |
| P1_K10 | precision | 0.2442 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4137; 0.4151] | [0.4137; 0.8319] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0194; 0.4376] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0335 | nan | — | — | [0.0332; 0.0366] | [0.0194; 0.4376] | undefined | n/a |
| P1_K30 | event_recall | 0.5326 | nan | — | — | [0.5722; 0.7415] | [0.4008; 0.8190] | undefined | n/a |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0056; 0.0196] | [0.0056; 0.4238] | H3_denominator | PASS |
| P1_K30 | precision | 0.2418 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4136; 0.4151] | [0.4136; 0.8317] | H3_denominator | PASS |
| P2 | processed_frac | 0.5276 | 0.5328 | [0.2712; 0.7944] | [0.4151; 0.6321] | [0.3513; 0.7444] | [0.3262; 0.7444] | H3_denominator | PASS |
| P2 | frame_recall | 0.6143 | nan | — | — | [0.5710; 0.5744] | [0.3333; 0.7515] | undefined | n/a |
| P2 | event_recall | 0.9009 | nan | — | — | [0.7375; 0.9069] | [0.5166; 0.9348] | undefined | n/a |
| P2 | false_open_per_frame | 0.0816 | 0.0819 | [0.0342; 0.1297] | [0.0625; 0.1034] | [0.0138; 0.2229] | [0.0138; 0.4320] | H3_denominator | PASS |
| P2 | precision | 0.2856 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.3184; 0.6762] | [0.4228; 0.8410] | H3_denominator | PASS |
| P3 | processed_frac | 0.6240 | 0.5359 | [0.2756; 0.7962] | [0.4181; 0.6339] | [0.5285; 0.9216] | [0.5034; 0.9216] | H3_denominator | PASS |
| P3 | frame_recall | 0.9826 | nan | — | — | [0.9965; 0.9998] | [0.5817; 0.9999] | undefined | n/a |
| P3 | event_recall | 0.9009 | nan | — | — | [0.7357; 0.9050] | [0.5153; 0.9335] | undefined | n/a |
| P3 | false_open_per_frame | 0.0803 | 0.0820 | [0.0341; 0.1299] | [0.0624; 0.1032] | [0.0138; 0.2229] | [0.0138; 0.4320] | H3_denominator | PASS |
| P3 | precision | 0.3863 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4489; 0.7834] | [0.4781; 0.8963] | H3_denominator | PASS |
| P4 | processed_frac | 0.6581 | 0.6495 | [0.2587; 1.0402] | [0.4810; 0.8255] | [0.3806; 0.7737] | [0.3555; 0.7737] | H3_denominator | PASS |
| P4 | frame_recall | 0.7295 | nan | — | — | [0.5941; 0.5974] | [0.3468; 0.7650] | undefined | n/a |
| P4 | event_recall | 0.9009 | nan | — | — | [0.7375; 0.9069] | [0.5166; 0.9348] | undefined | n/a |
| P4 | false_open_per_frame | 0.0407 | 0.0469 | [0.0179; 0.0758] | [0.0369; 0.0577] | [0.0079; 0.2170] | [0.0079; 0.4261] | H3_denominator | PASS |
| P4 | precision | 0.2719 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.3188; 0.6495] | [0.4036; 0.8218] | H3_denominator | PASS |

### camera — P-5 (khung transportable 163,209; ω toàn đơn vị 0.244)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.3778; 0.6222] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4903; 0.5098] | [0.3778; 0.6222] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.8929 | 0.8972 | [0.8488; 0.9456] | [0.8727; 0.9199] | [0.8485; 0.9028] | [0.6779; 0.9223] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3773 | 0.3387 | [0.3079; 0.3696] | [0.2545; 0.4330] | [0.2559; 0.3782] | [0.2559; 0.5004] | H3_denominator | PASS |
| P1_K2 | precision | 0.2452 | 0.3225 | [0.2607; 0.3843] | [0.1340; 0.4911] | [0.2437; 0.2526] | [0.2437; 0.4881] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1511; 0.3956] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1961; 0.2156] | [0.1511; 0.3956] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7731 | 0.7819 | [0.7006; 0.8632] | [0.7363; 0.8228] | [0.7394; 0.7937] | [0.5907; 0.8352] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1513 | 0.1355 | [0.1232; 0.1479] | [0.1018; 0.1732] | [0.1024; 0.1513] | [0.1024; 0.3468] | H3_denominator | PASS |
| P1_K5 | precision | 0.2448 | 0.3225 | [0.2607; 0.3843] | [0.1339; 0.4910] | [0.2437; 0.2575] | [0.2437; 0.4881] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0756; 0.3200] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0981; 0.1176] | [0.0756; 0.3200] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6805 | 0.6787 | [0.5819; 0.7754] | [0.6247; 0.7277] | [0.6418; 0.6961] | [0.5128; 0.7572] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0760 | 0.0678 | [0.0616; 0.0739] | [0.0509; 0.0866] | [0.0512; 0.0756] | [0.0512; 0.2957] | H3_denominator | PASS |
| P1_K10 | precision | 0.2442 | 0.3225 | [0.2607; 0.3842] | [0.1339; 0.4910] | [0.2437; 0.2586] | [0.2436; 0.4881] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0333; 0.0334] | [0.0333; 0.0333] | [0.0252; 0.2697] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0335 | 0.0333 | [0.0333; 0.0334] | [0.0333; 0.0333] | [0.0327; 0.0522] | [0.0252; 0.2696] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5326 | 0.4973 | [0.3807; 0.6139] | [0.4296; 0.5611] | [0.4703; 0.5246] | [0.3757; 0.6202] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0226 | [0.0205; 0.0247] | [0.0170; 0.0289] | [0.0171; 0.0252] | [0.0171; 0.2615] | H3_denominator | PASS |
| P1_K30 | precision | 0.2418 | 0.3224 | [0.2606; 0.3841] | [0.1338; 0.4909] | [0.2436; 0.2597] | [0.2436; 0.4880] | H3_denominator | PASS |
| P2 | processed_frac | 0.5276 | 0.5095 | [0.4101; 0.6090] | [0.4630; 0.5663] | [0.3996; 0.6294] | [0.3850; 0.6294] | H3_denominator | PASS |
| P2 | frame_recall | 0.6143 | 0.6337 | [0.4346; 0.8327] | [0.5635; 0.7390] | [0.6213; 0.6408] | [0.4788; 0.7232] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9009 | 0.8794 | [0.7803; 0.9785] | [0.8404; 0.9237] | [0.8317; 0.8860] | [0.6644; 0.9089] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0816 | 0.0742 | [0.0470; 0.1014] | [0.0492; 0.1071] | [0.0561; 0.1783] | [0.0561; 0.3005] | H3_denominator | PASS |
| P2 | precision | 0.2856 | 0.4011 | [0.2866; 0.5156] | [0.1752; 0.6004] | [0.2453; 0.3946] | [0.3030; 0.5475] | H3_denominator | PASS |
| P3 | processed_frac | 0.6240 | 0.6280 | [0.5348; 0.7211] | [0.5133; 0.7274] | [0.4891; 0.7189] | [0.4745; 0.7189] | H3_denominator | PASS |
| P3 | frame_recall | 0.9826 | 0.9801 | [0.9603; 0.9998] | [0.9664; 0.9886] | [0.9609; 0.9804] | [0.7405; 0.9849] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9009 | 0.8764 | [0.7788; 0.9740] | [0.8387; 0.9198] | [0.8288; 0.8831] | [0.6622; 0.9066] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0803 | 0.0725 | [0.0463; 0.0987] | [0.0475; 0.1060] | [0.0548; 0.1770] | [0.0548; 0.2992] | H3_denominator | PASS |
| P3 | precision | 0.3863 | 0.5033 | [0.4191; 0.5876] | [0.2354; 0.6817] | [0.3322; 0.4942] | [0.3803; 0.6248] | H3_denominator | PASS |
| P4 | processed_frac | 0.6581 | 0.6123 | [0.4354; 0.7892] | [0.5317; 0.6978] | [0.4773; 0.7071] | [0.4626; 0.7071] | H3_denominator | PASS |
| P4 | frame_recall | 0.7295 | 0.7070 | [0.4919; 0.9220] | [0.6342; 0.8261] | [0.6932; 0.7127] | [0.5341; 0.7786] | H3_denominator | PASS |
| P4 | event_recall | 0.9009 | 0.8896 | [0.7394; 1.0398] | [0.8366; 0.9550] | [0.8413; 0.8956] | [0.6721; 0.9166] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0407 | 0.0370 | [0.0208; 0.0533] | [0.0247; 0.0553] | [0.0280; 0.1502] | [0.0280; 0.2724] | H3_denominator | PASS |
| P4 | precision | 0.2719 | 0.3724 | [0.2616; 0.4832] | [0.1593; 0.5690] | [0.2436; 0.3681] | [0.2814; 0.5258] | H3_denominator | PASS |

### session — 2018-03-07_16:50 (khung transportable 149,718; ω toàn đơn vị 0.335)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.3325; 0.6675] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4925; 0.5075] | [0.3325; 0.6675] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.8929 | 0.9025 | [0.8502; 0.9548] | [0.8727; 0.9411] | [0.8558; 0.9078] | [0.6004; 0.9353] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3773 | 0.3364 | [0.3045; 0.3682] | [0.2658; 0.4480] | [0.1403; 0.3078] | [0.1403; 0.4752] | H3_denominator | PASS |
| P1_K2 | precision | 0.2452 | 0.3273 | [0.2635; 0.3910] | [0.1039; 0.4685] | [0.3844; 0.3938] | [0.3844; 0.7194] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2001] | [0.2000; 0.2000] | [0.1330; 0.4679] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1970; 0.2120] | [0.1330; 0.4679] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7731 | 0.7960 | [0.7081; 0.8839] | [0.7638; 0.8610] | [0.7550; 0.8071] | [0.5297; 0.8646] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1513 | 0.1346 | [0.1218; 0.1473] | [0.1063; 0.1792] | [0.0561; 0.1231] | [0.0561; 0.3911] | H3_denominator | PASS |
| P1_K5 | precision | 0.2448 | 0.3272 | [0.2635; 0.3910] | [0.1039; 0.4685] | [0.3844; 0.3984] | [0.3844; 0.7194] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1001] | [0.1000; 0.1000] | [0.0665; 0.4014] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0985; 0.1135] | [0.0665; 0.4014] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6805 | 0.6983 | [0.5936; 0.8030] | [0.6615; 0.7834] | [0.6626; 0.7147] | [0.4649; 0.7998] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0760 | 0.0673 | [0.0609; 0.0737] | [0.0532; 0.0896] | [0.0281; 0.0616] | [0.0281; 0.3630] | H3_denominator | PASS |
| P1_K10 | precision | 0.2442 | 0.3272 | [0.2635; 0.3909] | [0.1039; 0.4684] | [0.3844; 0.3993] | [0.3844; 0.7193] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0333; 0.0334] | [0.0333; 0.0333] | [0.0222; 0.3571] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0335 | 0.0333 | [0.0333; 0.0334] | [0.0333; 0.0333] | [0.0328; 0.0479] | [0.0222; 0.3571] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5326 | 0.5246 | [0.3887; 0.6605] | [0.4795; 0.6456] | [0.4985; 0.5505] | [0.3497; 0.6847] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0224 | [0.0203; 0.0246] | [0.0177; 0.0299] | [0.0094; 0.0205] | [0.0094; 0.3443] | H3_denominator | PASS |
| P1_K30 | precision | 0.2418 | 0.3271 | [0.2634; 0.3908] | [0.1039; 0.4683] | [0.3843; 0.3999] | [0.3843; 0.7192] | H3_denominator | PASS |
| P2 | processed_frac | 0.5276 | 0.5011 | [0.3946; 0.6077] | [0.4331; 0.5553] | [0.3710; 0.6859] | [0.3509; 0.6859] | H3_denominator | PASS |
| P2 | frame_recall | 0.6143 | 0.6308 | [0.4238; 0.8378] | [0.5686; 0.7445] | [0.5845; 0.5995] | [0.3947; 0.7296] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9009 | 0.8833 | [0.7891; 0.9775] | [0.8506; 0.9251] | [0.8374; 0.8894] | [0.5875; 0.9224] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0816 | 0.0758 | [0.0477; 0.1039] | [0.0533; 0.1182] | [0.0316; 0.1991] | [0.0316; 0.3665] | H3_denominator | PASS |
| P2 | precision | 0.2856 | 0.4119 | [0.2903; 0.5336] | [0.1405; 0.5731] | [0.3326; 0.6224] | [0.4324; 0.7673] | H3_denominator | PASS |
| P3 | processed_frac | 0.6240 | 0.6222 | [0.5236; 0.7209] | [0.4739; 0.7064] | [0.5274; 0.8423] | [0.5073; 0.8423] | H3_denominator | PASS |
| P3 | frame_recall | 0.9826 | 0.9813 | [0.9620; 1.0005] | [0.9708; 0.9886] | [0.9783; 0.9934] | [0.6606; 0.9955] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9009 | 0.8801 | [0.7878; 0.9725] | [0.8485; 0.9200] | [0.8344; 0.8864] | [0.5854; 0.9203] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0803 | 0.0742 | [0.0470; 0.1014] | [0.0515; 0.1173] | [0.0309; 0.1984] | [0.0309; 0.3659] | H3_denominator | PASS |
| P3 | precision | 0.3863 | 0.5161 | [0.4264; 0.6058] | [0.1936; 0.6620] | [0.4534; 0.7283] | [0.5006; 0.8355] | H3_denominator | PASS |
| P4 | processed_frac | 0.6581 | 0.5990 | [0.4077; 0.7902] | [0.5016; 0.6817] | [0.4176; 0.7324] | [0.3975; 0.7324] | H3_denominator | PASS |
| P4 | frame_recall | 0.7295 | 0.7034 | [0.4854; 0.9214] | [0.6347; 0.8263] | [0.6245; 0.6396] | [0.4217; 0.7566] | H3_denominator | PASS |
| P4 | event_recall | 0.9009 | 0.8893 | [0.7320; 1.0465] | [0.8333; 0.9632] | [0.8430; 0.8950] | [0.5914; 0.9263] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0407 | 0.0385 | [0.0213; 0.0558] | [0.0268; 0.0618] | [0.0161; 0.1835] | [0.0161; 0.3510] | H3_denominator | PASS |
| P4 | precision | 0.2719 | 0.3843 | [0.2621; 0.5065] | [0.1329; 0.5528] | [0.3328; 0.5908] | [0.4078; 0.7428] | H3_denominator | PASS |

### session — 2018-03-12_10:00 (khung transportable 245,998; ω toàn đơn vị 0.610)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.1952; 0.8048] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4398; 0.5602] | [0.1952; 0.8048] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.8929 | 0.8967 | [0.8304; 0.9630] | [0.8717; 0.9352] | [0.7405; 0.9147] | [0.3501; 0.9597] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3773 | 0.4342 | [0.4183; 0.4502] | [0.3839; 0.4903] | [0.1695; 0.4743] | [0.1695; 0.7791] | H3_denominator | PASS |
| P1_K2 | precision | 0.2452 | 0.1315 | [0.0996; 0.1634] | [0.0194; 0.2321] | [0.0513; 0.0632] | [0.0513; 0.6610] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0781; 0.6877] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1759; 0.2964] | [0.0781; 0.6877] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7731 | 0.7878 | [0.6854; 0.8903] | [0.7420; 0.8561] | [0.6506; 0.8248] | [0.3076; 0.9172] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1513 | 0.1737 | [0.1673; 0.1801] | [0.1536; 0.1962] | [0.0678; 0.1897] | [0.0678; 0.6774] | H3_denominator | PASS |
| P1_K5 | precision | 0.2448 | 0.1315 | [0.0996; 0.1634] | [0.0193; 0.2321] | [0.0513; 0.0672] | [0.0513; 0.6609] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0391; 0.6487] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0880; 0.2084] | [0.0390; 0.6486] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6805 | 0.6918 | [0.5779; 0.8056] | [0.6338; 0.7749] | [0.5713; 0.7455] | [0.2701; 0.8797] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0760 | 0.0869 | [0.0837; 0.0901] | [0.0768; 0.0981] | [0.0339; 0.0949] | [0.0339; 0.6435] | H3_denominator | PASS |
| P1_K10 | precision | 0.2442 | 0.1315 | [0.0996; 0.1634] | [0.0193; 0.2320] | [0.0513; 0.0679] | [0.0513; 0.6609] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0130; 0.6226] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0335 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0293; 0.1498] | [0.0130; 0.6226] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5326 | 0.5183 | [0.3694; 0.6671] | [0.4404; 0.6437] | [0.4280; 0.6022] | [0.2023; 0.8119] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0290 | [0.0279; 0.0301] | [0.0256; 0.0327] | [0.0113; 0.0316] | [0.0113; 0.6209] | H3_denominator | PASS |
| P1_K30 | precision | 0.2418 | 0.1314 | [0.0995; 0.1632] | [0.0193; 0.2319] | [0.0513; 0.0683] | [0.0513; 0.6609] | H3_denominator | PASS |
| P2 | processed_frac | 0.5276 | 0.5021 | [0.3905; 0.6138] | [0.4549; 0.5608] | [0.2326; 0.8056] | [0.1960; 0.8056] | H3_denominator | PASS |
| P2 | frame_recall | 0.6143 | 0.6476 | [0.4506; 0.8446] | [0.5797; 0.7469] | [0.5696; 0.6901] | [0.2528; 0.8624] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9009 | 0.8819 | [0.7841; 0.9796] | [0.8482; 0.9239] | [0.7282; 0.9024] | [0.3443; 0.9539] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0816 | 0.0903 | [0.0614; 0.1192] | [0.0743; 0.1059] | [0.0353; 0.3401] | [0.0353; 0.6449] | H3_denominator | PASS |
| P2 | precision | 0.2856 | 0.1696 | [0.1016; 0.2377] | [0.0266; 0.2964] | [0.0413; 0.1688] | [0.0662; 0.6758] | H3_denominator | PASS |
| P3 | processed_frac | 0.6240 | 0.5502 | [0.4400; 0.6605] | [0.4899; 0.6073] | [0.2514; 0.8244] | [0.2148; 0.8244] | H3_denominator | PASS |
| P3 | frame_recall | 0.9826 | 0.9797 | [0.9579; 1.0014] | [0.9663; 0.9883] | [0.8617; 0.9821] | [0.3825; 0.9921] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9009 | 0.8789 | [0.7831; 0.9747] | [0.8455; 0.9186] | [0.7258; 0.9000] | [0.3431; 0.9527] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0803 | 0.0896 | [0.0611; 0.1182] | [0.0737; 0.1051] | [0.0350; 0.3398] | [0.0350; 0.6446] | H3_denominator | PASS |
| P3 | precision | 0.3863 | 0.2342 | [0.1683; 0.3001] | [0.0355; 0.3896] | [0.0610; 0.2227] | [0.0914; 0.7010] | H3_denominator | PASS |
| P4 | processed_frac | 0.6581 | 0.6118 | [0.4173; 0.8062] | [0.5261; 0.7063] | [0.2754; 0.8484] | [0.2388; 0.8484] | H3_denominator | PASS |
| P4 | frame_recall | 0.7295 | 0.7196 | [0.5019; 0.9373] | [0.6421; 0.8390] | [0.6329; 0.7534] | [0.2809; 0.8905] | H3_denominator | PASS |
| P4 | event_recall | 0.9009 | 0.8912 | [0.7253; 1.0570] | [0.8322; 0.9604] | [0.7359; 0.9101] | [0.3479; 0.9575] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0407 | 0.0471 | [0.0307; 0.0636] | [0.0397; 0.0561] | [0.0184; 0.3232] | [0.0184; 0.6280] | H3_denominator | PASS |
| P4 | precision | 0.2719 | 0.1547 | [0.0960; 0.2134] | [0.0244; 0.2718] | [0.0435; 0.1563] | [0.0604; 0.6700] | H3_denominator | PASS |

### session — 2018-03-13_17:05 (khung transportable 30,600; ω toàn đơn vị 0.575)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.2125; 0.7875] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4245; 0.5755] | [0.2125; 0.7875] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.8929 | 0.8865 | [0.7808; 0.9923] | [0.8671; 0.9057] | [0.7488; 0.9042] | [0.3768; 0.9518] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3773 | 0.4081 | [0.3787; 0.4375] | [0.4015; 0.4139] | [0.1734; 0.4609] | [0.1734; 0.7485] | H3_denominator | PASS |
| P1_K2 | precision | 0.2452 | 0.1838 | [0.1251; 0.2426] | [0.1721; 0.1969] | [0.0781; 0.0998] | [0.0781; 0.6532] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0850; 0.6600] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1698; 0.3207] | [0.0850; 0.6600] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7731 | 0.7619 | [0.6004; 0.9235] | [0.7309; 0.7910] | [0.6436; 0.7989] | [0.3238; 0.8988] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1513 | 0.1633 | [0.1515; 0.1750] | [0.1606; 0.1656] | [0.0694; 0.1844] | [0.0694; 0.6444] | H3_denominator | PASS |
| P1_K5 | precision | 0.2448 | 0.1838 | [0.1251; 0.2426] | [0.1721; 0.1969] | [0.0781; 0.1104] | [0.0781; 0.6532] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0425; 0.6175] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0849; 0.2358] | [0.0425; 0.6175] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6805 | 0.6547 | [0.4831; 0.8264] | [0.6173; 0.6866] | [0.5530; 0.7084] | [0.2782; 0.8533] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0760 | 0.0816 | [0.0758; 0.0875] | [0.0803; 0.0828] | [0.0347; 0.0922] | [0.0347; 0.6097] | H3_denominator | PASS |
| P1_K10 | precision | 0.2442 | 0.1838 | [0.1250; 0.2425] | [0.1721; 0.1968] | [0.0781; 0.1114] | [0.0781; 0.6531] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0142; 0.5892] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0335 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0283; 0.1792] | [0.0142; 0.5892] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5326 | 0.4657 | [0.2789; 0.6526] | [0.4314; 0.5043] | [0.3934; 0.5487] | [0.1979; 0.7730] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0272 | [0.0253; 0.0292] | [0.0268; 0.0276] | [0.0116; 0.0307] | [0.0116; 0.5866] | H3_denominator | PASS |
| P1_K30 | precision | 0.2418 | 0.1837 | [0.1250; 0.2424] | [0.1720; 0.1967] | [0.0781; 0.1114] | [0.0780; 0.6531] | H3_denominator | PASS |
| P2 | processed_frac | 0.5276 | 0.4880 | [0.3419; 0.6342] | [0.4419; 0.5431] | [0.2419; 0.7824] | [0.2074; 0.7824] | H3_denominator | PASS |
| P2 | frame_recall | 0.6143 | 0.6169 | [0.2828; 0.9509] | [0.5436; 0.7548] | [0.5238; 0.6747] | [0.2622; 0.8372] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9009 | 0.8906 | [0.6919; 1.0892] | [0.8437; 0.9292] | [0.7522; 0.9076] | [0.3785; 0.9535] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0816 | 0.0934 | [0.0528; 0.1340] | [0.0768; 0.1062] | [0.0397; 0.3272] | [0.0397; 0.6147] | H3_denominator | PASS |
| P2 | precision | 0.2856 | 0.2324 | [0.0987; 0.3661] | [0.1984; 0.2757] | [0.0616; 0.2446] | [0.0988; 0.6738] | H3_denominator | PASS |
| P3 | processed_frac | 0.6240 | 0.5597 | [0.4236; 0.6957] | [0.5129; 0.6063] | [0.2723; 0.8129] | [0.2378; 0.8129] | H3_denominator | PASS |
| P3 | frame_recall | 0.9826 | 0.9829 | [0.9529; 1.0129] | [0.9759; 0.9898] | [0.8345; 0.9855] | [0.4177; 0.9927] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9009 | 0.8880 | [0.6902; 1.0858] | [0.8421; 0.9259] | [0.7501; 0.9054] | [0.3774; 0.9524] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0803 | 0.0926 | [0.0526; 0.1326] | [0.0764; 0.1051] | [0.0394; 0.3269] | [0.0394; 0.6144] | H3_denominator | PASS |
| P3 | precision | 0.3863 | 0.3229 | [0.2125; 0.4333] | [0.2903; 0.3554] | [0.0945; 0.3190] | [0.1372; 0.7122] | H3_denominator | PASS |
| P4 | processed_frac | 0.6581 | 0.5970 | [0.3600; 0.8340] | [0.4949; 0.6813] | [0.2882; 0.8288] | [0.2537; 0.8288] | H3_denominator | PASS |
| P4 | frame_recall | 0.7295 | 0.6869 | [0.3551; 1.0188] | [0.6160; 0.8220] | [0.5833; 0.7342] | [0.2919; 0.8670] | H3_denominator | PASS |
| P4 | event_recall | 0.9009 | 0.8941 | [0.6346; 1.1536] | [0.8266; 0.9596] | [0.7552; 0.9105] | [0.3800; 0.9550] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0407 | 0.0469 | [0.0240; 0.0697] | [0.0397; 0.0550] | [0.0199; 0.3074] | [0.0199; 0.5949] | H3_denominator | PASS |
| P4 | precision | 0.2719 | 0.2115 | [0.0893; 0.3338] | [0.1834; 0.2591] | [0.0648; 0.2251] | [0.0899; 0.6649] | H3_denominator | PASS |

### all — ALL (khung transportable 426,316; ω toàn đơn vị 0.514)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.2429; 0.7571] | summary | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4795; 0.5205] | [0.2429; 0.7571] | summary | n/a |
| P1_K2 | event_recall | 0.8929 | 0.8991 | [0.8485; 0.9497] | [0.8759; 0.9275] | [0.7990; 0.9105] | [0.4369; 0.9511] | summary | n/a |
| P1_K2 | false_open_per_frame | 0.3773 | 0.3980 | [0.3782; 0.4178] | [0.3383; 0.4629] | [0.1599; 0.4170] | [0.1599; 0.6740] | summary | n/a |
| P1_K2 | precision | 0.2452 | 0.2040 | [0.1643; 0.2437] | [0.0743; 0.3234] | [0.1661; 0.1777] | [0.1661; 0.6802] | summary | n/a |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0972; 0.6113] | summary | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1918; 0.2328] | [0.0972; 0.6113] | summary | n/a |
| P1_K5 | event_recall | 0.7731 | 0.7905 | [0.7093; 0.8717] | [0.7541; 0.8411] | [0.7026; 0.8141] | [0.3842; 0.8984] | summary | n/a |
| P1_K5 | false_open_per_frame | 0.1513 | 0.1592 | [0.1513; 0.1671] | [0.1353; 0.1852] | [0.0640; 0.1668] | [0.0640; 0.5781] | summary | n/a |
| P1_K5 | precision | 0.2448 | 0.2040 | [0.1643; 0.2437] | [0.0743; 0.3234] | [0.1661; 0.1824] | [0.1661; 0.6802] | summary | n/a |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0486; 0.5627] | summary | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0959; 0.1369] | [0.0486; 0.5627] | summary | n/a |
| P1_K10 | event_recall | 0.6805 | 0.6929 | [0.5975; 0.7883] | [0.6503; 0.7593] | [0.6160; 0.7275] | [0.3368; 0.8510] | summary | n/a |
| P1_K10 | false_open_per_frame | 0.0760 | 0.0796 | [0.0757; 0.0836] | [0.0677; 0.0926] | [0.0320; 0.0834] | [0.0320; 0.5461] | summary | n/a |
| P1_K10 | precision | 0.2442 | 0.2040 | [0.1643; 0.2436] | [0.0742; 0.3234] | [0.1661; 0.1832] | [0.1660; 0.6802] | summary | n/a |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0162; 0.5304] | summary | n/a |
| P1_K30 | frame_recall | 0.0335 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0320; 0.0730] | [0.0162; 0.5303] | summary | n/a |
| P1_K30 | event_recall | 0.5326 | 0.5183 | [0.3911; 0.6454] | [0.4613; 0.6129] | [0.4611; 0.5726] | [0.2521; 0.7663] | summary | n/a |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0266 | [0.0252; 0.0279] | [0.0226; 0.0309] | [0.0107; 0.0278] | [0.0107; 0.5248] | summary | n/a |
| P1_K30 | precision | 0.2418 | 0.2038 | [0.1642; 0.2435] | [0.0742; 0.3232] | [0.1660; 0.1836] | [0.1660; 0.6801] | summary | n/a |
| P2 | processed_frac | 0.5276 | 0.5008 | [0.4031; 0.5984] | [0.4526; 0.5489] | [0.2802; 0.7635] | [0.2493; 0.7635] | summary | n/a |
| P2 | frame_recall | 0.6143 | 0.6362 | [0.4413; 0.8310] | [0.5701; 0.7381] | [0.5793; 0.6204] | [0.2935; 0.8077] | summary | n/a |
| P2 | event_recall | 0.9009 | 0.8832 | [0.7990; 0.9673] | [0.8511; 0.9218] | [0.7847; 0.8962] | [0.4291; 0.9433] | summary | n/a |
| P2 | false_open_per_frame | 0.0816 | 0.0854 | [0.0581; 0.1127] | [0.0676; 0.1050] | [0.0343; 0.2914] | [0.0343; 0.5485] | summary | n/a |
| P2 | precision | 0.2856 | 0.2592 | [0.1730; 0.3453] | [0.1018; 0.4095] | [0.1314; 0.3754] | [0.1955; 0.7097] | summary | n/a |
| P3 | processed_frac | 0.6240 | 0.5762 | [0.4821; 0.6703] | [0.5014; 0.6458] | [0.3464; 0.8297] | [0.3155; 0.8297] | summary | n/a |
| P3 | frame_recall | 0.9826 | 0.9808 | [0.9621; 0.9994] | [0.9720; 0.9886] | [0.9498; 0.9908] | [0.4812; 0.9954] | summary | n/a |
| P3 | event_recall | 0.9009 | 0.8802 | [0.7979; 0.9624] | [0.8499; 0.9182] | [0.7820; 0.8935] | [0.4276; 0.9418] | summary | n/a |
| P3 | false_open_per_frame | 0.0803 | 0.0844 | [0.0577; 0.1112] | [0.0667; 0.1043] | [0.0339; 0.2910] | [0.0339; 0.5481] | summary | n/a |
| P3 | precision | 0.3863 | 0.3473 | [0.2749; 0.4197] | [0.1452; 0.5137] | [0.1983; 0.4869] | [0.2533; 0.7674] | summary | n/a |
| P4 | processed_frac | 0.6581 | 0.6062 | [0.4262; 0.7862] | [0.5263; 0.6861] | [0.3245; 0.8078] | [0.2936; 0.8078] | summary | n/a |
| P4 | frame_recall | 0.7295 | 0.7084 | [0.5001; 0.9166] | [0.6397; 0.8215] | [0.6247; 0.6657] | [0.3165; 0.8307] | summary | n/a |
| P4 | event_recall | 0.9009 | 0.8903 | [0.7385; 1.0421] | [0.8338; 0.9591] | [0.7911; 0.9025] | [0.4326; 0.9467] | summary | n/a |
| P4 | false_open_per_frame | 0.0407 | 0.0441 | [0.0284; 0.0597] | [0.0349; 0.0547] | [0.0177; 0.2748] | [0.0177; 0.5319] | summary | n/a |
| P4 | precision | 0.2719 | 0.2384 | [0.1615; 0.3153] | [0.0937; 0.3799] | [0.1339; 0.3488] | [0.1790; 0.6932] | summary | n/a |

### pooling_dependent_block — G331_2018-03-07_165000 (khung transportable 89,768; ω toàn đơn vị 0.003)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4983; 0.5017] | pooling_dependent | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4983; 0.5017] | [0.4983; 0.5017] | pooling_dependent | n/a |
| P1_K2 | event_recall | 0.8929 | 0.9395 | [0.6452; 1.2339] | [0.8894; 0.9768] | [0.7804; 0.9498] | [0.9364; 0.9397] | pooling_dependent | n/a |
| P1_K2 | false_open_per_frame | 0.3773 | 0.0018 | [-0.0010; 0.0047] | [0.0013; 0.0027] | [0.0018; 0.0035] | [0.0018; 0.0052] | pooling_dependent | n/a |
| P1_K2 | precision | 0.2452 | 0.9963 | [0.9906; 1.0021] | [0.9946; 0.9974] | [0.9930; 0.9963] | [0.9930; 0.9963] | pooling_dependent | n/a |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1993; 0.2027] | pooling_dependent | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1993; 0.2027] | [0.1993; 0.2027] | pooling_dependent | n/a |
| P1_K5 | event_recall | 0.7731 | 0.8588 | [0.3431; 1.3746] | [0.7840; 0.9311] | [0.7134; 0.8827] | [0.8560; 0.8593] | pooling_dependent | n/a |
| P1_K5 | false_open_per_frame | 0.1513 | 0.0007 | [-0.0004; 0.0019] | [0.0005; 0.0011] | [0.0007; 0.0014] | [0.0007; 0.0041] | pooling_dependent | n/a |
| P1_K5 | precision | 0.2448 | 0.9963 | [0.9906; 1.0021] | [0.9946; 0.9974] | [0.9930; 0.9963] | [0.9930; 0.9963] | pooling_dependent | n/a |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0997; 0.1030] | pooling_dependent | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0997; 0.1030] | [0.0997; 0.1030] | pooling_dependent | n/a |
| P1_K10 | event_recall | 0.6805 | 0.7926 | [0.1304; 1.4548] | [0.6956; 0.8889] | [0.6584; 0.8277] | [0.7900; 0.7933] | pooling_dependent | n/a |
| P1_K10 | false_open_per_frame | 0.0760 | 0.0004 | [-0.0002; 0.0009] | [0.0003; 0.0005] | [0.0004; 0.0007] | [0.0004; 0.0037] | pooling_dependent | n/a |
| P1_K10 | precision | 0.2442 | 0.9963 | [0.9906; 1.0021] | [0.9946; 0.9974] | [0.9929; 0.9963] | [0.9930; 0.9963] | pooling_dependent | n/a |
| P1_K30 | processed_frac | 0.0340 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0332; 0.0366] | pooling_dependent | n/a |
| P1_K30 | frame_recall | 0.0335 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0332; 0.0366] | [0.0332; 0.0366] | pooling_dependent | n/a |
| P1_K30 | event_recall | 0.5326 | 0.6888 | [-0.1687; 1.5464] | [0.5737; 0.7985] | [0.5722; 0.7415] | [0.6865; 0.6899] | pooling_dependent | n/a |
| P1_K30 | false_open_per_frame | 0.0258 | 0.0001 | [-0.0001; 0.0003] | [0.0001; 0.0002] | [0.0001; 0.0002] | [0.0001; 0.0035] | pooling_dependent | n/a |
| P1_K30 | precision | 0.2418 | 0.9963 | [0.9906; 1.0021] | [0.9946; 0.9974] | [0.9927; 0.9963] | [0.9930; 0.9963] | pooling_dependent | n/a |
| P2 | processed_frac | 0.5276 | 0.5719 | [0.2754; 0.8684] | [0.4205; 0.6881] | [0.5702; 0.5733] | [0.5700; 0.5733] | pooling_dependent | n/a |
| P2 | frame_recall | 0.6143 | 0.5730 | [0.2758; 0.8701] | [0.4212; 0.6896] | [0.5710; 0.5744] | [0.5710; 0.5744] | pooling_dependent | n/a |
| P2 | event_recall | 0.9009 | 0.8879 | [0.2881; 1.4877] | [0.7861; 0.9597] | [0.7375; 0.9069] | [0.8849; 0.8883] | pooling_dependent | n/a |
| P2 | false_open_per_frame | 0.0816 | 0.0003 | [-0.0004; 0.0010] | [0.0002; 0.0005] | [0.0003; 0.0020] | [0.0003; 0.0037] | pooling_dependent | n/a |
| P2 | precision | 0.2856 | 0.9982 | [0.9946; 1.0017] | [0.9972; 0.9986] | [0.9924; 0.9982] | [0.9948; 0.9982] | pooling_dependent | n/a |
| P3 | processed_frac | 0.6240 | 0.9973 | [0.9924; 1.0022] | [0.9958; 0.9983] | [0.9942; 0.9973] | [0.9940; 0.9973] | pooling_dependent | n/a |
| P3 | frame_recall | 0.9826 | 0.9998 | [0.9992; 1.0005] | [0.9996; 1.0000] | [0.9965; 0.9998] | [0.9965; 0.9998] | pooling_dependent | n/a |
| P3 | event_recall | 0.9009 | 0.8857 | [0.2888; 1.4825] | [0.7843; 0.9567] | [0.7357; 0.9050] | [0.8827; 0.8861] | pooling_dependent | n/a |
| P3 | false_open_per_frame | 0.0803 | 0.0003 | [-0.0004; 0.0010] | [0.0002; 0.0005] | [0.0003; 0.0020] | [0.0003; 0.0036] | pooling_dependent | n/a |
| P3 | precision | 0.3863 | 0.9988 | [0.9968; 1.0009] | [0.9984; 0.9991] | [0.9955; 0.9988] | [0.9955; 0.9988] | pooling_dependent | n/a |
| P4 | processed_frac | 0.6581 | 0.5956 | [0.2598; 0.9314] | [0.4268; 0.7421] | [0.5938; 0.5969] | [0.5936; 0.5969] | pooling_dependent | n/a |
| P4 | frame_recall | 0.7295 | 0.5961 | [0.2590; 0.9331] | [0.4273; 0.7426] | [0.5941; 0.5974] | [0.5941; 0.5974] | pooling_dependent | n/a |
| P4 | event_recall | 0.9009 | 0.8879 | [0.1823; 1.5934] | [0.7578; 0.9703] | [0.7375; 0.9069] | [0.8849; 0.8883] | pooling_dependent | n/a |
| P4 | false_open_per_frame | 0.0407 | 0.0002 | [-0.0003; 0.0007] | [0.0001; 0.0003] | [0.0002; 0.0019] | [0.0002; 0.0035] | pooling_dependent | n/a |
| P4 | precision | 0.2719 | 0.9972 | [0.9896; 1.0047] | [0.9947; 0.9985] | [0.9916; 0.9972] | [0.9938; 0.9972] | pooling_dependent | n/a |

## 8. E6 (độ nhạy, không vào H1–H4)

sensitivity only; does not change H1-H4. Each axis repeats E3 (transport, PI, error ratio) with operating points and P4 coefficients frozen.

- `b_raw`: (b) raw instead of (b') on both sides (benchmark cells (b), v2)
- `rule_v1`: STRATA_SPEC v1 partition merge instead of v2 pooled
- `D_fine`: D classes <=2 / 3-5 / 6-10 / 11-30 / 31-100 / >100 frames (support rule unchanged, v2)
- `cue_baseline`: 4th stratum component: tercile (benchmark cut points) of log10 of the causal cue median m_t at the window's first frame
- `window_1800`: u = 1800-frame windows (R classes unchanged)
- `norm_300`: cue normalisation median over 300 instead of 1800 previous frames (both sides)
- `fps_15`: benchmark videos of unknown fps assumed 15 fps (k_merge 3) instead of 25
- `fps_30`: benchmark videos of unknown fps assumed 30 fps (k_merge 6) instead of 25
- `route_laptop`: 6 P-1 clips with the laptop ov_gpu_fp32x detections instead of Kaggle (g(b') already shown identical on 54,005 frames -> expected identical)

## 9. Cấm

- changing operating points (results/operating_points.json)
- changing the MEVA selection / S8
- changing STRATA_SPEC (v2 main, v1 sensitivity)
- adding or removing cells after unlock
- using (c) or seen (b') numbers to calibrate H2-H4
- extending MEVA based on results (D3.6)

## 10. Những gì đã xem trước pre-registration (chép từ PREREG_DECISIONS)

**Dòng "đã xem" — cập nhật 2026-10-01 (trước pre-registration CP1):**
- ω (b′) theo ô camera × phiên, v2 và v1, và π_pool: `results/omega_bprime_S8.csv` (tổng ω_v2 0,514, ω_v1 0,615, π_pool 0,101; 1 ô pooling-dependent: P-4 G331 × 03-07 16:50, π_pool 0,993);
- tầng chiếm khối chính của ô P-4 × 03-07: `full|gt30|lt300|C1` (98,7 % khung block; 5 cửa sổ / 5 video benchmark; v2 gộp vào `P3:full|gt30|lt1800|C0`, 27 cửa sổ / 13 video, w_B 0,0063);
- (c) sau dựng lại nhãn theo `n_decoded`: S8 10 block 9,17 h duty 0,159, run idle dài nhất 2.102 s; mọi (c) sẵn có 21 block 16,67 h duty 0,135, run idle dài nhất 2.500 s; TV(w_B, w_T) (c) trên ô (b′) benchmark, v2: S8 0,607, sẵn có 0,604; ω (c) v2: 0,489 / 0,540;
- TV(w_B, w_T), v2, arm main, S8: (a) 0,522; (b) 0,572; (b′) 0,608 (thêm cột `TV_v2`, `omega_v2` và dòng `all_S8` vào `results/E1_profile_meva.csv`, 2026-10-01; các dòng cũ không đổi);
- **không** chính sách nào đã chạy trên MEVA; không kết cục nào (m_T, sai số, PI, cận) đã được tính.

## 11. SHA-256 đầu vào

- `results/benchmark_units.parquet` `18ba1039163d8c4a89592c4fd194ede06712424234a7e044b3cbc03a01e9842e`
- `results/operating_points.json` `8fef212f158946f0d82506482a43d81ffce2a1a1ac1b2d1c8e728e13181ab4a8`
- `results/strata_map_benchmark.csv` `d55d5a055b4d3d0e442704bb909dbda37607cb82e988bccd4e71ce4e4ed153ef`
- `results/w_B_strata.csv` `bdd3b192d10ff1fe46bac56a38930c995d8d167824479abfe574593d222ef8b6`
- `results/m_B_strata.csv` `dc351f0b12525d094a498e072421a7dc94d711da6a51a8bdd61ad7bcec4eb72e`
- `results/omega_bprime_S8.csv` `24eb930f56ee606f53df4b84ffd9764869086a59a949212c356bc5777ee9317b`
- `results/meva_wT_strata.csv` `1fbfbd67973310e5c94cf042091a563ad2111a45884e32518fe5f662c88b5213`
- `results/E1_profile_meva.csv` `e56cb9a46abcc4a383ef5665976955ef36586439fa226ac7550aea08dab9e1b8`
- `results/E1_profile_benchmark.csv` `a796d1c08e986a18b726909781d1f6eb40fdbacf0ad31f9f6a6609c7c829a3e5`
- `results/meva_S8_blocks.csv` `b1bc9b0459143e82180bcaab917fcd1cd239508aebf98b11c46969b5101dd1fa`
- `results/meva_manifest.csv` `4be0b8e907000e59d62852d496432f90224c90b51aa52833622eee8f02c0bf27`
- `docs/STRATA_SPEC.md` `cd4eb258115460e09ab98057c0da35e6d3992ec40862a3023402252808b9d688`
- `docs/PREREG_DECISIONS.md` `f6dd7513404f29ea503d197b976024ae4a750f228b83091ea3b4e860150b0748`
- `docs/MEVA_detector_plan.md` `1101322c0d35c8b653c902045b0f94b49a1816d26295c139be42e7850a3bd4be`
- `code/transport.py` `2edddc95dd3db39ca0f50ef2a6f1c9c794e910ff56a52ec131bf1b539683f38c`
- `code/prereg_build.py` `d4354cfcb67a721599fa834eb672cdd5efae326a1cfaa7176c51be62863edb7a`
- `code/meva_frames.py` `94274b3ef92e10d23553edda43d1aa26ef309e0e8b9a59d44546dd0cd12a3b5b`
- `code/policies.py` `23c1bead5578838e58c86a5b1ea2668ae7386ef9119326e7aabda967984d1368`
- `code/strata.py` `4a2a563c9c8547f51ef15469370a116d6a916f522c9346ec523c2df82f23a66b`
- `code/flip_radius.py` `35c0c5e3a7f970bc661c3ef70ac74f743b7141bd85e21ac6e2f075713192445c`
- `code/cue.py` `82ae5857beb8ce5d57c87e0fe8709dd77f81f5cb250eb26a5755a7c4cb51a480`
- `code/meva_outcomes.py` `76ab80938787e4e5679a2deb8bb1eeafec4cd746acf8adf4b35437fe4b5293ae`
- `cache/meva_windows.parquet` `286db24f9c825c8aab0956f379c8ae65f2baba7d9b5071d77e8d0e89f561a4d4`
- `cache/meva_s8_frames.pkl` `e464153f83660899db917026048fe483bea4aa877d4bddec01d6f9722ef3bb46`
