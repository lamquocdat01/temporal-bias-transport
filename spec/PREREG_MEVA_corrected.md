# PREREG_MEVA_corrected — Đề tài 09, AMENDMENT 01 (LASIESTA temporal order; NOT blind: original outcomes seen)

Tạo lúc **2026-10-03T16:07:30+07:00** · HEAD lúc dựng `28b1fc85f1` · bản máy đọc: `prereg/PREREG_MEVA_corrected.json` (file này là bản người đọc, sinh tự động từ cùng dữ liệu). Băm SHA-256 của cả hai file nằm ở `prereg/*.sha256`.

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
| a → bench (a) | 0.563 | [0.477; 0.698] | 0.0010 | 0.717 | 0.759 [0.674; 0.843] | có |
| bprime → bench (b') | 0.610 | [0.515; 0.788] | 0.0010 | 0.175 | 0.240 [0.169; 0.311] | có |
| c → bench (b') [cross-definition] | 0.608 | [0.556; 0.751] | 0.0010 | 0.159 | 0.240 [0.168; 0.310] | không |

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
| P1_K2 vs P1_K5 | frame_recall | 0.3000 | inf | inf | không |
| P1_K2 vs P1_K5 | event_recall | 0.1123 | inf | inf | không |
| P1_K2 vs P1_K5 | false_open_per_frame | 0.2278 | 0.9517 | 0.9517 | không |
| P1_K2 vs P1_K5 | precision | 0.0004 | 0.1979 | 0.0002 | có |
| P1_K2 vs P1_K10 | processed_frac | 0.3997 | inf | inf | không |
| P1_K2 vs P1_K10 | frame_recall | 0.3999 | 0.4519 | 0.4519 | không |
| P1_K2 vs P1_K10 | event_recall | 0.2044 | 0.3941 | 0.3941 | không |
| P1_K2 vs P1_K10 | false_open_per_frame | 0.3037 | inf | inf | không |
| P1_K2 vs P1_K10 | precision | 0.0010 | 0.1979 | 0.0005 | có |
| P1_K2 vs P1_K30 | processed_frac | 0.4664 | inf | inf | không |
| P1_K2 vs P1_K30 | frame_recall | 0.4666 | inf | inf | không |
| P1_K2 vs P1_K30 | event_recall | 0.3499 | inf | inf | không |
| P1_K2 vs P1_K30 | false_open_per_frame | 0.3543 | inf | inf | không |
| P1_K2 vs P1_K30 | precision | 0.0035 | 0.1979 | 0.0018 | có |
| P1_K2 vs P2 | processed_frac | -0.1120 | 0.6976 | 0.6976 | có |
| P1_K2 vs P2 | frame_recall | -0.1886 | inf | inf | có |
| P1_K2 vs P2 | event_recall | 0.0010 | 0.0021 | 0.0021 | có |
| P1_K2 vs P2 | false_open_per_frame | 0.3014 | inf | inf | có |
| P1_K2 vs P2 | precision | -0.0300 | 0.1935 | 0.0147 | có |
| P1_K2 vs P3 | processed_frac | -0.2108 | inf | inf | có |
| P1_K2 vs P3 | frame_recall | -0.4887 | 0.4519 | 0.4519 | có |
| P1_K2 vs P3 | event_recall | 0.0010 | 0.0021 | 0.0021 | có |
| P1_K2 vs P3 | false_open_per_frame | 0.3059 | 0.9517 | 0.9517 | có |
| P1_K2 vs P3 | precision | -0.0936 | 0.4519 | 0.0409 | có |
| P1_K2 vs P4 | processed_frac | -0.0869 | 0.4897 | 0.4897 | có |
| P1_K2 vs P4 | frame_recall | -0.1784 | inf | inf | có |
| P1_K2 vs P4 | event_recall | 0.0031 | 0.0074 | 0.0074 | có |
| P1_K2 vs P4 | false_open_per_frame | 0.3098 | 0.9517 | 0.9517 | có |
| P1_K2 vs P4 | precision | -0.0373 | 0.2671 | 0.0175 | có |
| P1_K5 vs P1_K10 | processed_frac | 0.0999 | inf | inf | không |
| P1_K5 vs P1_K10 | frame_recall | 0.1000 | 0.4519 | 0.4519 | không |
| P1_K5 vs P1_K10 | event_recall | 0.0921 | 0.3941 | 0.3941 | không |
| P1_K5 vs P1_K10 | false_open_per_frame | 0.0759 | 0.9517 | 0.9517 | không |
| P1_K5 vs P1_K10 | precision | 0.0006 | 0.1979 | 0.0003 | có |
| P1_K5 vs P1_K30 | processed_frac | 0.1666 | inf | inf | không |
| P1_K5 vs P1_K30 | frame_recall | 0.1666 | inf | inf | không |
| P1_K5 vs P1_K30 | event_recall | 0.2375 | inf | inf | không |
| P1_K5 vs P1_K30 | false_open_per_frame | 0.1265 | 0.9517 | 0.9517 | không |
| P1_K5 vs P1_K30 | precision | 0.0031 | 0.1979 | 0.0016 | có |
| P1_K5 vs P2 | processed_frac | -0.4118 | inf | inf | có |
| P1_K5 vs P2 | frame_recall | -0.4886 | inf | inf | có |
| P1_K5 vs P2 | event_recall | -0.1113 | 0.3941 | 0.3941 | có |
| P1_K5 vs P2 | false_open_per_frame | 0.0736 | 0.9517 | 0.9517 | có |
| P1_K5 vs P2 | precision | -0.0303 | 0.1951 | 0.0149 | có |
| P1_K5 vs P3 | processed_frac | -0.5106 | inf | inf | có |
| P1_K5 vs P3 | frame_recall | -0.7886 | 0.4519 | 0.4519 | có |
| P1_K5 vs P3 | event_recall | -0.1113 | 0.3941 | 0.3941 | có |
| P1_K5 vs P3 | false_open_per_frame | 0.0781 | 0.9517 | 0.9517 | có |
| P1_K5 vs P3 | precision | -0.0940 | 0.4519 | 0.0411 | có |
| P1_K5 vs P4 | processed_frac | -0.3867 | inf | inf | có |
| P1_K5 vs P4 | frame_recall | -0.4784 | inf | inf | có |
| P1_K5 vs P4 | event_recall | -0.1092 | 0.3941 | 0.3941 | có |
| P1_K5 vs P4 | false_open_per_frame | 0.0820 | inf | inf | có |
| P1_K5 vs P4 | precision | -0.0377 | 0.2690 | 0.0177 | có |
| P1_K10 vs P1_K30 | processed_frac | 0.0666 | inf | inf | không |
| P1_K10 vs P1_K30 | frame_recall | 0.0667 | inf | inf | không |
| P1_K10 vs P1_K30 | event_recall | 0.1455 | 0.3941 | 0.3941 | không |
| P1_K10 vs P1_K30 | false_open_per_frame | 0.0506 | inf | inf | không |
| P1_K10 vs P1_K30 | precision | 0.0025 | 0.1979 | 0.0013 | có |
| P1_K10 vs P2 | processed_frac | -0.5118 | inf | inf | có |
| P1_K10 vs P2 | frame_recall | -0.5885 | inf | inf | có |
| P1_K10 vs P2 | event_recall | -0.2033 | 0.3941 | 0.3941 | có |
| P1_K10 vs P2 | false_open_per_frame | -0.0023 | 0.0287 | 0.0287 | có |
| P1_K10 vs P2 | precision | -0.0310 | 0.1979 | 0.0152 | có |
| P1_K10 vs P3 | processed_frac | -0.6105 | inf | inf | có |
| P1_K10 vs P3 | frame_recall | -0.8886 | 0.4519 | 0.4519 | có |
| P1_K10 vs P3 | event_recall | -0.2033 | inf | inf | có |
| P1_K10 vs P3 | false_open_per_frame | 0.0022 | 0.0208 | 0.0208 | có |
| P1_K10 vs P3 | precision | -0.0946 | 0.4519 | 0.0413 | có |
| P1_K10 vs P4 | processed_frac | -0.4867 | inf | inf | có |
| P1_K10 vs P4 | frame_recall | -0.5784 | 0.4519 | 0.4519 | có |
| P1_K10 vs P4 | event_recall | -0.2013 | inf | inf | có |
| P1_K10 vs P4 | false_open_per_frame | 0.0061 | 0.0866 | 0.0866 | có |
| P1_K10 vs P4 | precision | -0.0384 | 0.2721 | 0.0180 | có |
| P1_K30 vs P2 | processed_frac | -0.5784 | inf | inf | có |
| P1_K30 vs P2 | frame_recall | -0.6552 | 0.4519 | 0.4519 | có |
| P1_K30 vs P2 | event_recall | -0.3488 | 0.3941 | 0.3941 | có |
| P1_K30 vs P2 | false_open_per_frame | -0.0529 | 0.9517 | 0.9517 | có |
| P1_K30 vs P2 | precision | -0.0335 | 0.2086 | 0.0164 | có |
| P1_K30 vs P3 | processed_frac | -0.6771 | inf | inf | có |
| P1_K30 vs P3 | frame_recall | -0.9553 | inf | inf | có |
| P1_K30 vs P3 | event_recall | -0.3488 | 0.3941 | 0.3941 | có |
| P1_K30 vs P3 | false_open_per_frame | -0.0484 | inf | inf | có |
| P1_K30 vs P3 | precision | -0.0972 | 0.4519 | 0.0423 | có |
| P1_K30 vs P4 | processed_frac | -0.5533 | inf | inf | có |
| P1_K30 vs P4 | frame_recall | -0.6451 | inf | inf | có |
| P1_K30 vs P4 | event_recall | -0.3467 | inf | inf | có |
| P1_K30 vs P4 | false_open_per_frame | -0.0445 | 0.9517 | 0.9517 | có |
| P1_K30 vs P4 | precision | -0.0409 | 0.2831 | 0.0192 | có |
| P2 vs P3 | processed_frac | -0.0987 | inf | inf | có |
| P2 vs P3 | frame_recall | -0.3001 | inf | inf | có |
| P2 vs P3 | event_recall | 0.0000 | 0.0000 | 0.0000 | có |
| P2 vs P3 | false_open_per_frame | 0.0045 | 0.5354 | 0.5354 | có |
| P2 vs P3 | precision | -0.0637 | 0.3584 | 0.0273 | có |
| P2 vs P4 | processed_frac | 0.0251 | 0.1021 | 0.1021 | có |
| P2 vs P4 | frame_recall | 0.0101 | 0.0195 | 0.0195 | có |
| P2 vs P4 | event_recall | 0.0021 | 0.0081 | 0.0081 | có |
| P2 vs P4 | false_open_per_frame | 0.0084 | 0.4561 | 0.4561 | có |
| P2 vs P4 | precision | -0.0074 | 0.0720 | 0.0033 | có |
| P3 vs P4 | processed_frac | 0.1238 | inf | inf | có |
| P3 vs P4 | frame_recall | 0.3102 | inf | inf | có |
| P3 vs P4 | event_recall | 0.0021 | 0.0082 | 0.0082 | có |
| P3 vs P4 | false_open_per_frame | 0.0039 | 0.1772 | 0.1772 | có |
| P3 vs P4 | precision | 0.0563 | 0.2653 | 0.0229 | có |

## 6. Bội số

Holm over p_H1..p_H4 at alpha = 0.05. Binding verdicts are the criteria above on point values; Holm-adjusted p is reported alongside (a criterion met without Holm significance is labelled 'PASS (not significant after Holm)').

## 7. Dự đoán theo ô (ước lượng ngây thơ, transport, PI 95 % t₂, CI phụ, cận Manski toàn MEVA)

Cột: ngây thơ · transport · PI · CI block · cận ĐL1(b) toàn đơn vị · cận P-2 · ω toàn đơn vị · vai trò · H3 dự kiến.

### camera — P-1 (khung transportable 92,998; ω toàn đơn vị 0.530)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.2348; 0.7652] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4643; 0.5357] | [0.2348; 0.7652] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.9004 | 0.9027 | [0.8289; 0.9765] | [0.8715; 0.9374] | [0.8105; 0.9126] | [0.4240; 0.9543] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3800 | 0.3667 | [0.3391; 0.3943] | [0.3385; 0.4102] | [0.1722; 0.4374] | [0.1722; 0.7026] | H3_denominator | PASS |
| P1_K2 | precision | 0.2399 | 0.2666 | [0.2114; 0.3218] | [0.1796; 0.3230] | [0.1252; 0.1415] | [0.1252; 0.6555] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0939; 0.6243] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1857; 0.2572] | [0.0939; 0.6243] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7830 | 0.8020 | [0.6870; 0.9171] | [0.7381; 0.8569] | [0.7202; 0.8223] | [0.3767; 0.9070] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1524 | 0.1467 | [0.1357; 0.1577] | [0.1354; 0.1641] | [0.0689; 0.1750] | [0.0689; 0.5992] | H3_denominator | PASS |
| P1_K5 | precision | 0.2395 | 0.2666 | [0.2114; 0.3217] | [0.1796; 0.3230] | [0.1252; 0.1485] | [0.1252; 0.6555] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0470; 0.5773] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0929; 0.1643] | [0.0470; 0.5773] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6878 | 0.7103 | [0.5817; 0.8389] | [0.6266; 0.7817] | [0.6378; 0.7399] | [0.3336; 0.8639] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0765 | 0.0734 | [0.0678; 0.0789] | [0.0677; 0.0821] | [0.0345; 0.0875] | [0.0345; 0.5648] | H3_denominator | PASS |
| P1_K10 | precision | 0.2389 | 0.2665 | [0.2114; 0.3217] | [0.1796; 0.3229] | [0.1252; 0.1495] | [0.1252; 0.6555] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0157; 0.5460] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0336 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0310; 0.1024] | [0.0157; 0.5460] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5390 | 0.5451 | [0.3667; 0.7236] | [0.4371; 0.6475] | [0.4895; 0.5916] | [0.2560; 0.7864] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0245 | [0.0226; 0.0263] | [0.0226; 0.0274] | [0.0115; 0.0292] | [0.0115; 0.5418] | H3_denominator | PASS |
| P1_K30 | precision | 0.2366 | 0.2664 | [0.2112; 0.3215] | [0.1794; 0.3227] | [0.1252; 0.1495] | [0.1251; 0.6554] | H3_denominator | PASS |
| P2 | processed_frac | 0.6156 | 0.5709 | [0.4459; 0.6959] | [0.5245; 0.6240] | [0.3000; 0.7985] | [0.2681; 0.7985] | H3_denominator | PASS |
| P2 | frame_recall | 0.6948 | 0.6672 | [0.4453; 0.8891] | [0.5992; 0.7760] | [0.6196; 0.6910] | [0.3134; 0.8437] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9014 | 0.8846 | [0.7089; 1.0602] | [0.8246; 0.9537] | [0.7942; 0.8963] | [0.4154; 0.9458] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0788 | 0.0827 | [0.0462; 0.1193] | [0.0648; 0.1063] | [0.0389; 0.3040] | [0.0389; 0.5692] | H3_denominator | PASS |
| P2 | precision | 0.2708 | 0.3116 | [0.2094; 0.4138] | [0.1995; 0.3891] | [0.1046; 0.3024] | [0.1463; 0.6767] | H3_denominator | PASS |
| P3 | processed_frac | 0.7130 | 0.6859 | [0.5643; 0.8074] | [0.6215; 0.7373] | [0.3539; 0.8525] | [0.3221; 0.8525] | H3_denominator | PASS |
| P3 | frame_recall | 0.9891 | 0.9891 | [0.9744; 1.0038] | [0.9831; 0.9942] | [0.9184; 0.9899] | [0.4646; 0.9949] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9014 | 0.8877 | [0.7267; 1.0488] | [0.8338; 0.9509] | [0.7971; 0.8992] | [0.4169; 0.9473] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0742 | 0.0790 | [0.0427; 0.1153] | [0.0616; 0.1020] | [0.0371; 0.3023] | [0.0371; 0.5674] | H3_denominator | PASS |
| P3 | precision | 0.3329 | 0.3845 | [0.3029; 0.4660] | [0.2610; 0.4560] | [0.1453; 0.3686] | [0.1806; 0.7109] | H3_denominator | PASS |
| P4 | processed_frac | 0.5897 | 0.5587 | [0.3537; 0.7636] | [0.4604; 0.6452] | [0.2942; 0.7927] | [0.2624; 0.7927] | H3_denominator | PASS |
| P4 | frame_recall | 0.6851 | 0.6857 | [0.4874; 0.8839] | [0.6188; 0.7816] | [0.6367; 0.7081] | [0.3220; 0.8524] | H3_denominator | PASS |
| P4 | event_recall | 0.9014 | 0.8949 | [0.7621; 1.0277] | [0.8459; 0.9473] | [0.8035; 0.9056] | [0.4203; 0.9506] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0705 | 0.0721 | [0.0353; 0.1090] | [0.0542; 0.0938] | [0.0339; 0.2991] | [0.0339; 0.5642] | H3_denominator | PASS |
| P4 | precision | 0.2788 | 0.3272 | [0.2036; 0.4509] | [0.2096; 0.4120] | [0.1083; 0.3158] | [0.1537; 0.6840] | H3_denominator | PASS |

### camera — P-2 (khung transportable 81,909; ω toàn đơn vị 0.621)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.1896; 0.8104] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.3628; 0.6372] | [0.1896; 0.8104] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.9004 | 0.8984 | [0.7730; 1.0238] | [0.8582; 0.9536] | [0.6986; 0.9210] | [0.3406; 0.9615] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3800 | 0.4518 | [0.4310; 0.4726] | [0.4356; 0.4676] | [0.1713; 0.4817] | [0.1713; 0.7921] | H3_denominator | PASS |
| P1_K2 | precision | 0.2399 | 0.0965 | [0.0549; 0.1380] | [0.0648; 0.1288] | [0.0366; 0.0589] | [0.0366; 0.6574] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0758; 0.6967] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1451; 0.4195] | [0.0758; 0.6967] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7830 | 0.8087 | [0.6197; 0.9977] | [0.7521; 0.8922] | [0.6289; 0.8512] | [0.3066; 0.9275] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1524 | 0.1807 | [0.1724; 0.1890] | [0.1743; 0.1871] | [0.0685; 0.1927] | [0.0685; 0.6894] | H3_denominator | PASS |
| P1_K5 | precision | 0.2395 | 0.0965 | [0.0549; 0.1380] | [0.0648; 0.1288] | [0.0366; 0.0677] | [0.0366; 0.6574] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0379; 0.6588] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0726; 0.3469] | [0.0379; 0.6588] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6878 | 0.7293 | [0.5401; 0.9184] | [0.6850; 0.8283] | [0.5671; 0.7895] | [0.2765; 0.8974] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0765 | 0.0904 | [0.0862; 0.0945] | [0.0871; 0.0936] | [0.0343; 0.0964] | [0.0343; 0.6551] | H3_denominator | PASS |
| P1_K10 | precision | 0.2389 | 0.0964 | [0.0549; 0.1380] | [0.0648; 0.1288] | [0.0366; 0.0697] | [0.0366; 0.6574] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0127; 0.6335] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0336 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0242; 0.2986] | [0.0126; 0.6335] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5390 | 0.5701 | [0.2932; 0.8471] | [0.4973; 0.7261] | [0.4433; 0.6657] | [0.2162; 0.8370] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0302 | [0.0288; 0.0315] | [0.0291; 0.0312] | [0.0114; 0.0321] | [0.0114; 0.6323] | H3_denominator | PASS |
| P1_K30 | precision | 0.2366 | 0.0964 | [0.0548; 0.1379] | [0.0647; 0.1287] | [0.0366; 0.0704] | [0.0365; 0.6574] | H3_denominator | PASS |
| P2 | processed_frac | 0.6156 | 0.5624 | [0.4187; 0.7060] | [0.5030; 0.6265] | [0.2505; 0.8341] | [0.2132; 0.8341] | H3_denominator | PASS |
| P2 | frame_recall | 0.6948 | 0.7300 | [0.5188; 0.9412] | [0.6671; 0.7924] | [0.5297; 0.8041] | [0.2768; 0.8976] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9014 | 0.8916 | [0.6648; 1.1184] | [0.8180; 0.9683] | [0.6933; 0.9157] | [0.3380; 0.9589] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0788 | 0.1070 | [0.0554; 0.1586] | [0.0824; 0.1276] | [0.0406; 0.3510] | [0.0406; 0.6614] | H3_denominator | PASS |
| P2 | precision | 0.2708 | 0.1252 | [0.0539; 0.1965] | [0.0815; 0.1736] | [0.0320; 0.1545] | [0.0475; 0.6683] | H3_denominator | PASS |
| P3 | processed_frac | 0.7130 | 0.6224 | [0.4642; 0.7807] | [0.5575; 0.6949] | [0.2733; 0.8568] | [0.2360; 0.8568] | H3_denominator | PASS |
| P3 | frame_recall | 0.9891 | 0.9851 | [0.9599; 1.0103] | [0.9767; 0.9928] | [0.7148; 0.9892] | [0.3735; 0.9944] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9014 | 0.8943 | [0.6811; 1.1075] | [0.8285; 0.9674] | [0.6954; 0.9178] | [0.3391; 0.9599] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0742 | 0.1032 | [0.0509; 0.1554] | [0.0771; 0.1249] | [0.0391; 0.3495] | [0.0391; 0.6600] | H3_denominator | PASS |
| P3 | precision | 0.3329 | 0.1527 | [0.0825; 0.2229] | [0.1012; 0.2079] | [0.0421; 0.1749] | [0.0579; 0.6787] | H3_denominator | PASS |
| P4 | processed_frac | 0.5897 | 0.5376 | [0.2789; 0.7963] | [0.4225; 0.6465] | [0.2411; 0.8247] | [0.2038; 0.8247] | H3_denominator | PASS |
| P4 | frame_recall | 0.6851 | 0.7399 | [0.5149; 0.9648] | [0.6674; 0.8118] | [0.5369; 0.8112] | [0.2805; 0.9014] | H3_denominator | PASS |
| P4 | event_recall | 0.9014 | 0.9051 | [0.7530; 1.0573] | [0.8627; 0.9642] | [0.7039; 0.9262] | [0.3432; 0.9640] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0705 | 0.0930 | [0.0388; 0.1472] | [0.0666; 0.1130] | [0.0353; 0.3457] | [0.0353; 0.6561] | H3_denominator | PASS |
| P4 | precision | 0.2788 | 0.1328 | [0.0588; 0.2067] | [0.0889; 0.1859] | [0.0328; 0.1617] | [0.0503; 0.6712] | H3_denominator | PASS |

### camera — P-3 (khung transportable 52,200; ω toàn đơn vị 0.758)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.1208; 0.8792] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.3184; 0.6816] | [0.1208; 0.8792] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.9004 | 0.9058 | [0.7325; 1.0791] | [0.8730; 0.9530] | [0.4078; 0.9576] | [0.2189; 0.9772] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3800 | 0.4782 | [0.4662; 0.4902] | [0.4369; 0.4933] | [0.1156; 0.4947] | [0.1156; 0.8739] | H3_denominator | PASS |
| P1_K2 | precision | 0.2399 | 0.0436 | [0.0197; 0.0676] | [0.0134; 0.1261] | [0.0105; 0.0203] | [0.0105; 0.7689] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2001] | [0.2000; 0.2001] | [0.2000; 0.2000] | [0.0483; 0.8067] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1273; 0.4906] | [0.0483; 0.8067] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7830 | 0.8082 | [0.5220; 1.0944] | [0.7654; 0.8947] | [0.3639; 0.9136] | [0.1953; 0.9536] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1524 | 0.1913 | [0.1865; 0.1961] | [0.1748; 0.1974] | [0.0462; 0.1979] | [0.0462; 0.8046] | H3_denominator | PASS |
| P1_K5 | precision | 0.2395 | 0.0436 | [0.0197; 0.0676] | [0.0134; 0.1261] | [0.0105; 0.0231] | [0.0105; 0.7689] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1001 | [0.1000; 0.1001] | [0.1000; 0.1001] | [0.1000; 0.1000] | [0.0242; 0.7825] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0637; 0.4270] | [0.0242; 0.7825] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6878 | 0.7183 | [0.4010; 1.0355] | [0.6726; 0.8178] | [0.3234; 0.8732] | [0.1736; 0.9319] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0765 | 0.0957 | [0.0933; 0.0981] | [0.0874; 0.0987] | [0.0231; 0.0990] | [0.0231; 0.7815] | H3_denominator | PASS |
| P1_K10 | precision | 0.2389 | 0.0436 | [0.0197; 0.0676] | [0.0134; 0.1260] | [0.0105; 0.0231] | [0.0105; 0.7689] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0081; 0.7664] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0336 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0212; 0.3845] | [0.0081; 0.7664] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5390 | 0.5513 | [0.1805; 0.9221] | [0.4909; 0.6795] | [0.2482; 0.7980] | [0.1332; 0.8916] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0319 | [0.0311; 0.0327] | [0.0292; 0.0329] | [0.0077; 0.0330] | [0.0077; 0.7661] | H3_denominator | PASS |
| P1_K30 | precision | 0.2366 | 0.0436 | [0.0196; 0.0675] | [0.0134; 0.1259] | [0.0105; 0.0233] | [0.0105; 0.7689] | H3_denominator | PASS |
| P2 | processed_frac | 0.6156 | 0.5912 | [0.4224; 0.7600] | [0.5054; 0.6569] | [0.1884; 0.9012] | [0.1429; 0.9012] | H3_denominator | PASS |
| P2 | frame_recall | 0.6948 | 0.6993 | [0.3501; 1.0486] | [0.6246; 0.7644] | [0.4453; 0.8086] | [0.1690; 0.9273] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9014 | 0.8735 | [0.5570; 1.1900] | [0.8232; 0.9407] | [0.3933; 0.9430] | [0.2111; 0.9694] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0788 | 0.0953 | [0.0530; 0.1376] | [0.0749; 0.1233] | [0.0230; 0.4022] | [0.0230; 0.7814] | H3_denominator | PASS |
| P2 | precision | 0.2708 | 0.0516 | [0.0105; 0.0928] | [0.0141; 0.1664] | [0.0082; 0.0692] | [0.0125; 0.7708] | H3_denominator | PASS |
| P3 | processed_frac | 0.7130 | 0.6415 | [0.4637; 0.8193] | [0.5507; 0.7106] | [0.2005; 0.9134] | [0.1550; 0.9134] | H3_denominator | PASS |
| P3 | frame_recall | 0.9891 | 0.9855 | [0.9487; 1.0222] | [0.9761; 0.9923] | [0.6275; 0.9907] | [0.2382; 0.9965] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9014 | 0.8768 | [0.5779; 1.1756] | [0.8320; 0.9405] | [0.3948; 0.9445] | [0.2119; 0.9702] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0742 | 0.0925 | [0.0489; 0.1361] | [0.0716; 0.1201] | [0.0224; 0.4015] | [0.0224; 0.7807] | H3_denominator | PASS |
| P3 | precision | 0.3329 | 0.0670 | [0.0266; 0.1075] | [0.0195; 0.2031] | [0.0114; 0.0798] | [0.0162; 0.7745] | H3_denominator | PASS |
| P4 | processed_frac | 0.5897 | 0.5896 | [0.3189; 0.8602] | [0.4589; 0.7263] | [0.1880; 0.9008] | [0.1425; 0.9008] | H3_denominator | PASS |
| P4 | frame_recall | 0.6851 | 0.7134 | [0.3557; 1.0711] | [0.6316; 0.7935] | [0.4542; 0.8175] | [0.1724; 0.9307] | H3_denominator | PASS |
| P4 | event_recall | 0.9014 | 0.8897 | [0.6176; 1.1618] | [0.8375; 0.9584] | [0.4006; 0.9503] | [0.2150; 0.9733] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0705 | 0.0835 | [0.0326; 0.1344] | [0.0589; 0.1122] | [0.0202; 0.3993] | [0.0202; 0.7785] | H3_denominator | PASS |
| P4 | precision | 0.2788 | 0.0528 | [0.0076; 0.0980] | [0.0135; 0.1768] | [0.0084; 0.0701] | [0.0128; 0.7711] | H3_denominator | PASS |

### camera — P-4 (khung transportable 36,000; ω toàn đơn vị 0.418)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.2909; 0.7091] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | nan | — | — | [0.4983; 0.5017] | [0.2909; 0.7091] | undefined | n/a |
| P1_K2 | event_recall | 0.9004 | nan | — | — | [0.7377; 0.9638] | [0.5546; 0.9728] | undefined | n/a |
| P1_K2 | false_open_per_frame | 0.3800 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.0837; 0.2928] | [0.0837; 0.5019] | H3_denominator | PASS |
| P1_K2 | precision | 0.2399 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4144; 0.4158] | [0.4144; 0.8326] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1164; 0.5346] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | nan | — | — | [0.1993; 0.2027] | [0.1164; 0.5346] | undefined | n/a |
| P1_K5 | event_recall | 0.7830 | nan | — | — | [0.6912; 0.9172] | [0.5196; 0.9378] | undefined | n/a |
| P1_K5 | false_open_per_frame | 0.1524 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0335; 0.1171] | [0.0335; 0.4517] | H3_denominator | PASS |
| P1_K5 | precision | 0.2395 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4144; 0.4158] | [0.4144; 0.8326] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1001 | [0.1000; 0.1001] | [0.1001; 0.1001] | [0.1000; 0.1000] | [0.0582; 0.4764] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | nan | — | — | [0.0997; 0.1030] | [0.0582; 0.4764] | undefined | n/a |
| P1_K10 | event_recall | 0.6878 | nan | — | — | [0.6433; 0.8693] | [0.4836; 0.9017] | undefined | n/a |
| P1_K10 | false_open_per_frame | 0.0765 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0168; 0.0586] | [0.0168; 0.4349] | H3_denominator | PASS |
| P1_K10 | precision | 0.2389 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4143; 0.4158] | [0.4143; 0.8325] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0194; 0.4376] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0336 | nan | — | — | [0.0332; 0.0366] | [0.0194; 0.4376] | undefined | n/a |
| P1_K30 | event_recall | 0.5390 | nan | — | — | [0.5685; 0.7946] | [0.4274; 0.8456] | undefined | n/a |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0056; 0.0195] | [0.0056; 0.4238] | H3_denominator | PASS |
| P1_K30 | precision | 0.2366 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4142; 0.4157] | [0.4142; 0.8324] | H3_denominator | PASS |
| P2 | processed_frac | 0.6156 | 0.6265 | [0.3543; 0.8987] | [0.4904; 0.7169] | [0.3847; 0.7778] | [0.3596; 0.7778] | H3_denominator | PASS |
| P2 | frame_recall | 0.6948 | nan | — | — | [0.6131; 0.6164] | [0.3579; 0.7761] | undefined | n/a |
| P2 | event_recall | 0.9014 | nan | — | — | [0.7061; 0.9322] | [0.5308; 0.9490] | undefined | n/a |
| P2 | false_open_per_frame | 0.0788 | 0.0769 | [0.0228; 0.1309] | [0.0567; 0.1026] | [0.0129; 0.2220] | [0.0129; 0.4311] | H3_denominator | PASS |
| P2 | precision | 0.2708 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.3277; 0.6640] | [0.4124; 0.8306] | H3_denominator | PASS |
| P3 | processed_frac | 0.7130 | 0.6670 | [0.3831; 0.9508] | [0.5204; 0.7640] | [0.5510; 0.9441] | [0.5259; 0.9441] | H3_denominator | PASS |
| P3 | frame_recall | 0.9891 | nan | — | — | [0.9966; 1.0000] | [0.5818; 1.0000] | undefined | n/a |
| P3 | event_recall | 0.9014 | nan | — | — | [0.7085; 0.9345] | [0.5326; 0.9508] | undefined | n/a |
| P3 | false_open_per_frame | 0.0742 | 0.0756 | [0.0193; 0.1319] | [0.0547; 0.1014] | [0.0126; 0.2217] | [0.0126; 0.4308] | H3_denominator | PASS |
| P3 | precision | 0.3329 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.4389; 0.7528] | [0.4584; 0.8766] | H3_denominator | PASS |
| P4 | processed_frac | 0.5897 | 0.6500 | [0.1949; 1.1050] | [0.4590; 0.8524] | [0.3854; 0.7785] | [0.3603; 0.7785] | H3_denominator | PASS |
| P4 | frame_recall | 0.6851 | nan | — | — | [0.6051; 0.6085] | [0.3533; 0.7714] | undefined | n/a |
| P4 | event_recall | 0.9014 | nan | — | — | [0.7178; 0.9439] | [0.5396; 0.9578] | undefined | n/a |
| P4 | false_open_per_frame | 0.0705 | 0.0681 | [-0.0046; 0.1408] | [0.0426; 0.1061] | [0.0114; 0.2205] | [0.0114; 0.4296] | H3_denominator | PASS |
| P4 | precision | 0.2788 | 0.0000 | [0.0000; 0.0000] | [0.0000; 0.0000] | [0.3232; 0.6542] | [0.4063; 0.8245] | H3_denominator | PASS |

### camera — P-5 (khung transportable 163,209; ω toàn đơn vị 0.244)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.3778; 0.6222] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4904; 0.5096] | [0.3778; 0.6222] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.9004 | 0.8962 | [0.8469; 0.9456] | [0.8711; 0.9193] | [0.8463; 0.9020] | [0.6771; 0.9216] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3800 | 0.3354 | [0.3057; 0.3651] | [0.2496; 0.4312] | [0.2534; 0.3756] | [0.2534; 0.4978] | H3_denominator | PASS |
| P1_K2 | precision | 0.2399 | 0.3292 | [0.2699; 0.3886] | [0.1375; 0.5007] | [0.2488; 0.2577] | [0.2488; 0.4932] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1511; 0.3956] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1962; 0.2153] | [0.1511; 0.3956] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7830 | 0.7801 | [0.6985; 0.8617] | [0.7334; 0.8207] | [0.7367; 0.7923] | [0.5894; 0.8339] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1524 | 0.1342 | [0.1223; 0.1460] | [0.0999; 0.1725] | [0.1014; 0.1503] | [0.1014; 0.3458] | H3_denominator | PASS |
| P1_K5 | precision | 0.2395 | 0.3292 | [0.2698; 0.3886] | [0.1375; 0.5007] | [0.2488; 0.2626] | [0.2487; 0.4932] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0756; 0.3200] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0981; 0.1172] | [0.0756; 0.3200] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6878 | 0.6765 | [0.5803; 0.7727] | [0.6207; 0.7260] | [0.6388; 0.6945] | [0.5111; 0.7556] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0765 | 0.0671 | [0.0612; 0.0730] | [0.0499; 0.0863] | [0.0507; 0.0751] | [0.0507; 0.2952] | H3_denominator | PASS |
| P1_K10 | precision | 0.2389 | 0.3292 | [0.2698; 0.3886] | [0.1375; 0.5007] | [0.2487; 0.2637] | [0.2487; 0.4932] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0333; 0.0334] | [0.0333; 0.0334] | [0.0333; 0.0333] | [0.0252; 0.2697] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0336 | 0.0333 | [0.0333; 0.0334] | [0.0333; 0.0333] | [0.0327; 0.0518] | [0.0252; 0.2696] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5390 | 0.4937 | [0.3790; 0.6084] | [0.4250; 0.5585] | [0.4662; 0.5219] | [0.3730; 0.6175] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0224 | [0.0204; 0.0244] | [0.0167; 0.0288] | [0.0169; 0.0251] | [0.0169; 0.2614] | H3_denominator | PASS |
| P1_K30 | precision | 0.2366 | 0.3291 | [0.2697; 0.3884] | [0.1374; 0.5006] | [0.2487; 0.2648] | [0.2486; 0.4931] | H3_denominator | PASS |
| P2 | processed_frac | 0.6156 | 0.5808 | [0.4748; 0.6867] | [0.5341; 0.6319] | [0.4535; 0.6832] | [0.4388; 0.6832] | H3_denominator | PASS |
| P2 | frame_recall | 0.6948 | 0.6776 | [0.4987; 0.8565] | [0.6120; 0.7647] | [0.6647; 0.6838] | [0.5120; 0.7564] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9014 | 0.8816 | [0.7434; 1.0198] | [0.8299; 0.9397] | [0.8325; 0.8882] | [0.6661; 0.9106] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0788 | 0.0727 | [0.0425; 0.1030] | [0.0458; 0.1102] | [0.0550; 0.1772] | [0.0550; 0.2994] | H3_denominator | PASS |
| P2 | precision | 0.2708 | 0.3842 | [0.2885; 0.4798] | [0.1636; 0.5806] | [0.2467; 0.3792] | [0.2903; 0.5347] | H3_denominator | PASS |
| P3 | processed_frac | 0.7130 | 0.7095 | [0.6054; 0.8136] | [0.6144; 0.7890] | [0.5507; 0.7805] | [0.5361; 0.7805] | H3_denominator | PASS |
| P3 | frame_recall | 0.9891 | 0.9866 | [0.9715; 1.0016] | [0.9763; 0.9924] | [0.9677; 0.9868] | [0.7454; 0.9898] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9014 | 0.8846 | [0.7599; 1.0092] | [0.8379; 0.9388] | [0.8353; 0.8910] | [0.6683; 0.9128] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0742 | 0.0684 | [0.0396; 0.0972] | [0.0425; 0.1047] | [0.0517; 0.1739] | [0.0517; 0.2961] | H3_denominator | PASS |
| P3 | precision | 0.3329 | 0.4578 | [0.3850; 0.5306] | [0.2052; 0.6402] | [0.3144; 0.4512] | [0.3459; 0.5904] | H3_denominator | PASS |
| P4 | processed_frac | 0.5897 | 0.5796 | [0.4018; 0.7575] | [0.4880; 0.6667] | [0.4526; 0.6824] | [0.4379; 0.6824] | H3_denominator | PASS |
| P4 | frame_recall | 0.6851 | 0.6934 | [0.5154; 0.8714] | [0.6264; 0.7839] | [0.6801; 0.6992] | [0.5239; 0.7683] | H3_denominator | PASS |
| P4 | event_recall | 0.9014 | 0.8840 | [0.7686; 0.9993] | [0.8396; 0.9315] | [0.8347; 0.8904] | [0.6679; 0.9123] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0705 | 0.0629 | [0.0343; 0.0914] | [0.0406; 0.0966] | [0.0475; 0.1698] | [0.0475; 0.2920] | H3_denominator | PASS |
| P4 | precision | 0.2788 | 0.3939 | [0.2751; 0.5127] | [0.1681; 0.5928] | [0.2528; 0.3885] | [0.2976; 0.5420] | H3_denominator | PASS |

### session — 2018-03-07_16:50 (khung transportable 149,718; ω toàn đơn vị 0.335)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.3325; 0.6675] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4925; 0.5075] | [0.3325; 0.6675] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.9004 | 0.9017 | [0.8485; 0.9549] | [0.8713; 0.9415] | [0.8536; 0.9072] | [0.5999; 0.9348] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3800 | 0.3331 | [0.3023; 0.3639] | [0.2609; 0.4466] | [0.1388; 0.3062] | [0.1388; 0.4737] | H3_denominator | PASS |
| P1_K2 | precision | 0.2399 | 0.3338 | [0.2722; 0.3955] | [0.1068; 0.4783] | [0.3876; 0.3970] | [0.3876; 0.7225] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2001] | [0.2000; 0.2000] | [0.1330; 0.4679] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1970; 0.2119] | [0.1330; 0.4679] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7830 | 0.7947 | [0.7067; 0.8827] | [0.7611; 0.8607] | [0.7526; 0.8062] | [0.5289; 0.8638] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1524 | 0.1332 | [0.1209; 0.1456] | [0.1044; 0.1787] | [0.0555; 0.1225] | [0.0555; 0.3904] | H3_denominator | PASS |
| P1_K5 | precision | 0.2395 | 0.3338 | [0.2722; 0.3955] | [0.1068; 0.4782] | [0.3876; 0.4015] | [0.3876; 0.7225] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1001] | [0.1000; 0.1000] | [0.0665; 0.4014] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0985; 0.1134] | [0.0665; 0.4014] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6878 | 0.6967 | [0.5930; 0.8005] | [0.6600; 0.7821] | [0.6601; 0.7137] | [0.4639; 0.7988] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0765 | 0.0666 | [0.0605; 0.0728] | [0.0522; 0.0894] | [0.0278; 0.0613] | [0.0278; 0.3627] | H3_denominator | PASS |
| P1_K10 | precision | 0.2389 | 0.3338 | [0.2722; 0.3954] | [0.1068; 0.4782] | [0.3875; 0.4025] | [0.3875; 0.7225] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0333; 0.0334] | [0.0333; 0.0333] | [0.0222; 0.3571] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0336 | 0.0333 | [0.0333; 0.0334] | [0.0333; 0.0333] | [0.0328; 0.0477] | [0.0222; 0.3571] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5390 | 0.5218 | [0.3880; 0.6556] | [0.4768; 0.6452] | [0.4950; 0.5486] | [0.3478; 0.6828] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0222 | [0.0202; 0.0243] | [0.0174; 0.0298] | [0.0093; 0.0204] | [0.0093; 0.3442] | H3_denominator | PASS |
| P1_K30 | precision | 0.2366 | 0.3337 | [0.2721; 0.3953] | [0.1066; 0.4781] | [0.3875; 0.4030] | [0.3874; 0.7224] | H3_denominator | PASS |
| P2 | processed_frac | 0.6156 | 0.5732 | [0.4589; 0.6876] | [0.5134; 0.6261] | [0.4117; 0.7265] | [0.3916; 0.7265] | H3_denominator | PASS |
| P2 | frame_recall | 0.6948 | 0.6756 | [0.4904; 0.8607] | [0.6157; 0.7703] | [0.6273; 0.6422] | [0.4235; 0.7584] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9014 | 0.8820 | [0.7355; 1.0285] | [0.8261; 0.9513] | [0.8349; 0.8885] | [0.5867; 0.9216] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0788 | 0.0751 | [0.0424; 0.1077] | [0.0503; 0.1244] | [0.0312; 0.1987] | [0.0312; 0.3662] | H3_denominator | PASS |
| P2 | precision | 0.2708 | 0.3934 | [0.2920; 0.4948] | [0.1304; 0.5533] | [0.3397; 0.6064] | [0.4192; 0.7541] | H3_denominator | PASS |
| P3 | processed_frac | 0.7130 | 0.7044 | [0.5921; 0.8167] | [0.5748; 0.7748] | [0.5620; 0.8768] | [0.5419; 0.8768] | H3_denominator | PASS |
| P3 | frame_recall | 0.9891 | 0.9874 | [0.9728; 1.0020] | [0.9798; 0.9932] | [0.9806; 0.9955] | [0.6621; 0.9970] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9014 | 0.8851 | [0.7532; 1.0171] | [0.8352; 0.9491] | [0.8378; 0.8915] | [0.5888; 0.9237] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0742 | 0.0708 | [0.0392; 0.1024] | [0.0466; 0.1193] | [0.0295; 0.1969] | [0.0295; 0.3644] | H3_denominator | PASS |
| P3 | precision | 0.3329 | 0.4680 | [0.3887; 0.5473] | [0.1655; 0.6212] | [0.4400; 0.6909] | [0.4735; 0.8085] | H3_denominator | PASS |
| P4 | processed_frac | 0.5897 | 0.5662 | [0.3723; 0.7601] | [0.4561; 0.6525] | [0.4068; 0.7217] | [0.3867; 0.7217] | H3_denominator | PASS |
| P4 | frame_recall | 0.6851 | 0.6921 | [0.5131; 0.8710] | [0.6329; 0.7878] | [0.6281; 0.6430] | [0.4240; 0.7590] | H3_denominator | PASS |
| P4 | event_recall | 0.9014 | 0.8892 | [0.7705; 1.0078] | [0.8430; 0.9462] | [0.8417; 0.8953] | [0.5915; 0.9264] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0705 | 0.0650 | [0.0334; 0.0966] | [0.0424; 0.1108] | [0.0270; 0.1945] | [0.0270; 0.3620] | H3_denominator | PASS |
| P4 | precision | 0.2788 | 0.4081 | [0.2760; 0.5401] | [0.1424; 0.5777] | [0.3424; 0.6143] | [0.4249; 0.7599] | H3_denominator | PASS |

### session — 2018-03-12_10:00 (khung transportable 245,998; ω toàn đơn vị 0.610)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.1952; 0.8048] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4408; 0.5592] | [0.1952; 0.8048] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.9004 | 0.8957 | [0.8283; 0.9631] | [0.8700; 0.9351] | [0.7364; 0.9143] | [0.3497; 0.9593] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3800 | 0.4329 | [0.4173; 0.4485] | [0.3825; 0.4901] | [0.1690; 0.4738] | [0.1690; 0.7786] | H3_denominator | PASS |
| P1_K2 | precision | 0.2399 | 0.1342 | [0.1029; 0.1654] | [0.0197; 0.2349] | [0.0524; 0.0642] | [0.0524; 0.6620] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0781; 0.6877] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1763; 0.2947] | [0.0781; 0.6877] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7830 | 0.7863 | [0.6833; 0.8893] | [0.7393; 0.8562] | [0.6464; 0.8243] | [0.3070; 0.9166] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1524 | 0.1732 | [0.1669; 0.1794] | [0.1530; 0.1961] | [0.0676; 0.1895] | [0.0676; 0.6772] | H3_denominator | PASS |
| P1_K5 | precision | 0.2395 | 0.1342 | [0.1029; 0.1654] | [0.0197; 0.2349] | [0.0524; 0.0683] | [0.0524; 0.6620] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0391; 0.6487] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0882; 0.2065] | [0.0390; 0.6486] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6878 | 0.6901 | [0.5769; 0.8033] | [0.6319; 0.7750] | [0.5673; 0.7453] | [0.2694; 0.8790] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0765 | 0.0866 | [0.0835; 0.0897] | [0.0765; 0.0981] | [0.0338; 0.0948] | [0.0338; 0.6434] | H3_denominator | PASS |
| P1_K10 | precision | 0.2389 | 0.1341 | [0.1029; 0.1654] | [0.0197; 0.2348] | [0.0524; 0.0689] | [0.0524; 0.6620] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0130; 0.6226] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0336 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0294; 0.1477] | [0.0130; 0.6226] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5390 | 0.5157 | [0.3683; 0.6631] | [0.4343; 0.6427] | [0.4239; 0.6019] | [0.2013; 0.8109] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0289 | [0.0279; 0.0299] | [0.0255; 0.0327] | [0.0113; 0.0316] | [0.0113; 0.6209] | H3_denominator | PASS |
| P1_K30 | precision | 0.2366 | 0.1340 | [0.1028; 0.1652] | [0.0197; 0.2347] | [0.0524; 0.0694] | [0.0523; 0.6619] | H3_denominator | PASS |
| P2 | processed_frac | 0.6156 | 0.5859 | [0.4648; 0.7069] | [0.5284; 0.6440] | [0.2653; 0.8383] | [0.2287; 0.8383] | H3_denominator | PASS |
| P2 | frame_recall | 0.6948 | 0.6892 | [0.5096; 0.8688] | [0.6240; 0.7771] | [0.6076; 0.7260] | [0.2691; 0.8787] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9014 | 0.8832 | [0.7290; 1.0374] | [0.8259; 0.9493] | [0.7261; 0.9040] | [0.3448; 0.9544] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0788 | 0.0892 | [0.0549; 0.1235] | [0.0708; 0.1080] | [0.0348; 0.3396] | [0.0348; 0.6444] | H3_denominator | PASS |
| P2 | precision | 0.2708 | 0.1579 | [0.0990; 0.2167] | [0.0238; 0.2790] | [0.0431; 0.1590] | [0.0616; 0.6712] | H3_denominator | PASS |
| P3 | processed_frac | 0.7130 | 0.6601 | [0.5312; 0.7891] | [0.5864; 0.7198] | [0.2943; 0.8673] | [0.2577; 0.8673] | H3_denominator | PASS |
| P3 | frame_recall | 0.9891 | 0.9865 | [0.9702; 1.0028] | [0.9775; 0.9928] | [0.8697; 0.9881] | [0.3851; 0.9947] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9014 | 0.8861 | [0.7455; 1.0267] | [0.8360; 0.9449] | [0.7284; 0.9064] | [0.3459; 0.9555] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0742 | 0.0859 | [0.0513; 0.1206] | [0.0677; 0.1040] | [0.0335; 0.3384] | [0.0335; 0.6432] | H3_denominator | PASS |
| P3 | precision | 0.3329 | 0.2005 | [0.1457; 0.2554] | [0.0300; 0.3422] | [0.0596; 0.1955] | [0.0783; 0.6879] | H3_denominator | PASS |
| P4 | processed_frac | 0.5897 | 0.5815 | [0.3706; 0.7924] | [0.4859; 0.6954] | [0.2636; 0.8366] | [0.2270; 0.8366] | H3_denominator | PASS |
| P4 | frame_recall | 0.6851 | 0.7040 | [0.5227; 0.8853] | [0.6397; 0.7962] | [0.6207; 0.7391] | [0.2748; 0.8845] | H3_denominator | PASS |
| P4 | event_recall | 0.9014 | 0.8893 | [0.7729; 1.0056] | [0.8455; 0.9429] | [0.7311; 0.9090] | [0.3472; 0.9568] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0705 | 0.0779 | [0.0387; 0.1171] | [0.0580; 0.0984] | [0.0304; 0.3352] | [0.0304; 0.6400] | H3_denominator | PASS |
| P4 | precision | 0.2788 | 0.1625 | [0.0946; 0.2304] | [0.0251; 0.2884] | [0.0441; 0.1629] | [0.0634; 0.6730] | H3_denominator | PASS |

### session — 2018-03-13_17:05 (khung transportable 30,600; ω toàn đơn vị 0.575)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.2125; 0.7875] | deterministic_by_construction | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4256; 0.5744] | [0.2125; 0.7875] | deterministic_by_construction | n/a |
| P1_K2 | event_recall | 0.9004 | 0.8853 | [0.7780; 0.9927] | [0.8655; 0.9054] | [0.7456; 0.9034] | [0.3762; 0.9513] | H3_denominator | PASS |
| P1_K2 | false_open_per_frame | 0.3800 | 0.4066 | [0.3787; 0.4345] | [0.4000; 0.4119] | [0.1728; 0.4603] | [0.1728; 0.7478] | H3_denominator | PASS |
| P1_K2 | precision | 0.2399 | 0.1868 | [0.1310; 0.2427] | [0.1762; 0.2001] | [0.0794; 0.1011] | [0.0794; 0.6544] | H3_denominator | PASS |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0850; 0.6600] | deterministic_by_construction | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1702; 0.3191] | [0.0850; 0.6600] | deterministic_by_construction | n/a |
| P1_K5 | event_recall | 0.7830 | 0.7597 | [0.5969; 0.9224] | [0.7283; 0.7889] | [0.6398; 0.7976] | [0.3228; 0.8979] | H3_denominator | PASS |
| P1_K5 | false_open_per_frame | 0.1524 | 0.1626 | [0.1515; 0.1738] | [0.1600; 0.1648] | [0.0691; 0.1841] | [0.0691; 0.6442] | H3_denominator | PASS |
| P1_K5 | precision | 0.2395 | 0.1868 | [0.1310; 0.2426] | [0.1762; 0.2000] | [0.0794; 0.1117] | [0.0794; 0.6544] | H3_denominator | PASS |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0425; 0.6175] | deterministic_by_construction | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0851; 0.2340] | [0.0425; 0.6175] | deterministic_by_construction | n/a |
| P1_K10 | event_recall | 0.6878 | 0.6520 | [0.4806; 0.8234] | [0.6138; 0.6835] | [0.5491; 0.7069] | [0.2771; 0.8521] | H3_denominator | PASS |
| P1_K10 | false_open_per_frame | 0.0765 | 0.0813 | [0.0758; 0.0869] | [0.0800; 0.0824] | [0.0346; 0.0921] | [0.0346; 0.6096] | H3_denominator | PASS |
| P1_K10 | precision | 0.2389 | 0.1868 | [0.1310; 0.2426] | [0.1762; 0.2000] | [0.0794; 0.1127] | [0.0794; 0.6544] | H3_denominator | PASS |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0142; 0.5892] | deterministic_by_construction | n/a |
| P1_K30 | frame_recall | 0.0336 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0284; 0.1772] | [0.0142; 0.5892] | deterministic_by_construction | n/a |
| P1_K30 | event_recall | 0.5390 | 0.4615 | [0.2781; 0.6450] | [0.4266; 0.4998] | [0.3887; 0.5465] | [0.1961; 0.7712] | H3_denominator | PASS |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0271 | [0.0253; 0.0290] | [0.0267; 0.0275] | [0.0115; 0.0307] | [0.0115; 0.5866] | H3_denominator | PASS |
| P1_K30 | precision | 0.2366 | 0.1867 | [0.1309; 0.2424] | [0.1761; 0.1999] | [0.0794; 0.1127] | [0.0793; 0.6544] | H3_denominator | PASS |
| P2 | processed_frac | 0.6156 | 0.5689 | [0.4094; 0.7284] | [0.5173; 0.6306] | [0.2763; 0.8168] | [0.2418; 0.8168] | H3_denominator | PASS |
| P2 | frame_recall | 0.6948 | 0.6648 | [0.3651; 0.9646] | [0.5958; 0.7830] | [0.5658; 0.7147] | [0.2825; 0.8576] | expected_FAIL | FAIL |
| P2 | event_recall | 0.9014 | 0.8921 | [0.6563; 1.1280] | [0.8232; 0.9482] | [0.7513; 0.9092] | [0.3791; 0.9542] | H3_denominator | PASS |
| P2 | false_open_per_frame | 0.0788 | 0.0942 | [0.0446; 0.1439] | [0.0733; 0.1123] | [0.0400; 0.3276] | [0.0400; 0.6151] | H3_denominator | PASS |
| P2 | precision | 0.2708 | 0.2183 | [0.1073; 0.3293] | [0.1916; 0.2528] | [0.0646; 0.2314] | [0.0928; 0.6678] | H3_denominator | PASS |
| P3 | processed_frac | 0.7130 | 0.6607 | [0.5019; 0.8194] | [0.6101; 0.7258] | [0.3153; 0.8558] | [0.2808; 0.8558] | H3_denominator | PASS |
| P3 | frame_recall | 0.9891 | 0.9882 | [0.9659; 1.0105] | [0.9825; 0.9934] | [0.8411; 0.9899] | [0.4199; 0.9950] | expected_FAIL | FAIL |
| P3 | event_recall | 0.9014 | 0.8946 | [0.6710; 1.1182] | [0.8304; 0.9482] | [0.7534; 0.9113] | [0.3802; 0.9552] | H3_denominator | PASS |
| P3 | false_open_per_frame | 0.0742 | 0.0902 | [0.0400; 0.1405] | [0.0684; 0.1075] | [0.0383; 0.3259] | [0.0383; 0.6134] | H3_denominator | PASS |
| P3 | precision | 0.3329 | 0.2795 | [0.1862; 0.3727] | [0.2527; 0.3046] | [0.0917; 0.2823] | [0.1188; 0.6938] | H3_denominator | PASS |
| P4 | processed_frac | 0.5897 | 0.5540 | [0.3169; 0.7911] | [0.4537; 0.6460] | [0.2699; 0.8105] | [0.2354; 0.8105] | H3_denominator | PASS |
| P4 | frame_recall | 0.6851 | 0.6822 | [0.4074; 0.9570] | [0.6189; 0.7766] | [0.5807; 0.7295] | [0.2899; 0.8650] | H3_denominator | PASS |
| P4 | event_recall | 0.9014 | 0.8851 | [0.6907; 1.0796] | [0.8385; 0.9285] | [0.7454; 0.9033] | [0.3761; 0.9512] | H3_denominator | PASS |
| P4 | false_open_per_frame | 0.0705 | 0.0823 | [0.0313; 0.1333] | [0.0604; 0.1005] | [0.0350; 0.3225] | [0.0350; 0.6100] | H3_denominator | PASS |
| P4 | precision | 0.2788 | 0.2301 | [0.0993; 0.3609] | [0.1994; 0.2829] | [0.0668; 0.2415] | [0.0978; 0.6728] | H3_denominator | PASS |

### all — ALL (khung transportable 426,316; ω toàn đơn vị 0.514)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.2429; 0.7571] | summary | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4797; 0.5203] | [0.2429; 0.7571] | summary | n/a |
| P1_K2 | event_recall | 0.9004 | 0.8982 | [0.8466; 0.9499] | [0.8741; 0.9271] | [0.7957; 0.9100] | [0.4365; 0.9506] | summary | n/a |
| P1_K2 | false_open_per_frame | 0.3800 | 0.3960 | [0.3765; 0.4154] | [0.3347; 0.4623] | [0.1590; 0.4161] | [0.1590; 0.6732] | summary | n/a |
| P1_K2 | precision | 0.2399 | 0.2081 | [0.1692; 0.2469] | [0.0754; 0.3307] | [0.1678; 0.1795] | [0.1678; 0.6820] | summary | n/a |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.0972; 0.6113] | summary | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1919; 0.2325] | [0.0972; 0.6113] | summary | n/a |
| P1_K5 | event_recall | 0.7830 | 0.7890 | [0.7075; 0.8706] | [0.7521; 0.8407] | [0.6991; 0.8134] | [0.3835; 0.8977] | summary | n/a |
| P1_K5 | false_open_per_frame | 0.1524 | 0.1584 | [0.1506; 0.1662] | [0.1339; 0.1850] | [0.0636; 0.1664] | [0.0636; 0.5778] | summary | n/a |
| P1_K5 | precision | 0.2395 | 0.2081 | [0.1692; 0.2469] | [0.0754; 0.3307] | [0.1678; 0.1842] | [0.1678; 0.6820] | summary | n/a |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0486; 0.5627] | summary | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0959; 0.1365] | [0.0486; 0.5627] | summary | n/a |
| P1_K10 | event_recall | 0.6878 | 0.6912 | [0.5966; 0.7857] | [0.6489; 0.7583] | [0.6125; 0.7269] | [0.3360; 0.8502] | summary | n/a |
| P1_K10 | false_open_per_frame | 0.0765 | 0.0792 | [0.0753; 0.0831] | [0.0669; 0.0925] | [0.0318; 0.0832] | [0.0318; 0.5460] | summary | n/a |
| P1_K10 | precision | 0.2389 | 0.2080 | [0.1692; 0.2469] | [0.0753; 0.3306] | [0.1678; 0.1850] | [0.1678; 0.6820] | summary | n/a |
| P1_K30 | processed_frac | 0.0340 | 0.0334 | [0.0334; 0.0334] | [0.0334; 0.0334] | [0.0333; 0.0333] | [0.0162; 0.5304] | summary | n/a |
| P1_K30 | frame_recall | 0.0336 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0320; 0.0726] | [0.0162; 0.5303] | summary | n/a |
| P1_K30 | event_recall | 0.5390 | 0.5154 | [0.3902; 0.6407] | [0.4586; 0.6106] | [0.4571; 0.5714] | [0.2507; 0.7649] | summary | n/a |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0264 | [0.0251; 0.0277] | [0.0223; 0.0309] | [0.0106; 0.0278] | [0.0106; 0.5248] | summary | n/a |
| P1_K30 | precision | 0.2366 | 0.2079 | [0.1691; 0.2467] | [0.0753; 0.3305] | [0.1678; 0.1854] | [0.1677; 0.6819] | summary | n/a |
| P2 | processed_frac | 0.6156 | 0.5802 | [0.4738; 0.6866] | [0.5306; 0.6311] | [0.3157; 0.7990] | [0.2848; 0.7990] | summary | n/a |
| P2 | frame_recall | 0.6948 | 0.6799 | [0.5052; 0.8547] | [0.6174; 0.7670] | [0.6211; 0.6617] | [0.3145; 0.8287] | summary | n/a |
| P2 | event_recall | 0.9014 | 0.8832 | [0.7419; 1.0244] | [0.8306; 0.9442] | [0.7823; 0.8966] | [0.4291; 0.9433] | summary | n/a |
| P2 | false_open_per_frame | 0.0788 | 0.0846 | [0.0522; 0.1170] | [0.0642; 0.1068] | [0.0340; 0.2910] | [0.0340; 0.5481] | summary | n/a |
| P2 | precision | 0.2708 | 0.2439 | [0.1712; 0.3165] | [0.0939; 0.3906] | [0.1360; 0.3598] | [0.1854; 0.6995] | summary | n/a |
| P3 | processed_frac | 0.7130 | 0.6757 | [0.5645; 0.7870] | [0.6051; 0.7347] | [0.3865; 0.8698] | [0.3556; 0.8698] | summary | n/a |
| P3 | frame_recall | 0.9891 | 0.9871 | [0.9729; 1.0013] | [0.9805; 0.9930] | [0.9532; 0.9938] | [0.4827; 0.9969] | summary | n/a |
| P3 | event_recall | 0.9014 | 0.8861 | [0.7590; 1.0132] | [0.8389; 0.9430] | [0.7849; 0.8992] | [0.4306; 0.9447] | summary | n/a |
| P3 | false_open_per_frame | 0.0742 | 0.0809 | [0.0488; 0.1131] | [0.0604; 0.1027] | [0.0325; 0.2896] | [0.0325; 0.5466] | summary | n/a |
| P3 | precision | 0.3329 | 0.3040 | [0.2423; 0.3657] | [0.1182; 0.4633] | [0.1917; 0.4430] | [0.2278; 0.7420] | summary | n/a |
| P4 | processed_frac | 0.5897 | 0.5741 | [0.3864; 0.7619] | [0.4838; 0.6631] | [0.3126; 0.7959] | [0.2817; 0.7959] | summary | n/a |
| P4 | frame_recall | 0.6851 | 0.6959 | [0.5251; 0.8666] | [0.6305; 0.7782] | [0.6249; 0.6655] | [0.3164; 0.8306] | summary | n/a |
| P4 | event_recall | 0.9014 | 0.8890 | [0.7811; 0.9968] | [0.8464; 0.9346] | [0.7874; 0.9017] | [0.4320; 0.9461] | summary | n/a |
| P4 | false_open_per_frame | 0.0705 | 0.0737 | [0.0390; 0.1083] | [0.0548; 0.0966] | [0.0296; 0.2867] | [0.0296; 0.5437] | summary | n/a |
| P4 | precision | 0.2788 | 0.2522 | [0.1654; 0.3390] | [0.0949; 0.3933] | [0.1374; 0.3654] | [0.1885; 0.7027] | summary | n/a |

### pooling_dependent_block — G331_2018-03-07_165000 (khung transportable 89,768; ω toàn đơn vị 0.003)

| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |
|---|---|---|---|---|---|---|---|---|---|
| P1_K2 | processed_frac | 0.5004 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4983; 0.5017] | pooling_dependent | n/a |
| P1_K2 | frame_recall | 0.5001 | 0.5000 | [0.5000; 0.5000] | [0.5000; 0.5000] | [0.4983; 0.5017] | [0.4983; 0.5017] | pooling_dependent | n/a |
| P1_K2 | event_recall | 0.9004 | 0.9532 | [0.6159; 1.2904] | [0.9265; 0.9785] | [0.7377; 0.9638] | [0.9500; 0.9533] | pooling_dependent | n/a |
| P1_K2 | false_open_per_frame | 0.3800 | 0.0011 | [-0.0010; 0.0031] | [0.0008; 0.0015] | [0.0011; 0.0027] | [0.0011; 0.0044] | pooling_dependent | n/a |
| P1_K2 | precision | 0.2399 | 0.9979 | [0.9938; 1.0019] | [0.9971; 0.9984] | [0.9945; 0.9979] | [0.9945; 0.9979] | pooling_dependent | n/a |
| P1_K5 | processed_frac | 0.2006 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1993; 0.2027] | pooling_dependent | n/a |
| P1_K5 | frame_recall | 0.2002 | 0.2000 | [0.2000; 0.2000] | [0.2000; 0.2000] | [0.1993; 0.2027] | [0.1993; 0.2027] | pooling_dependent | n/a |
| P1_K5 | event_recall | 0.7830 | 0.8931 | [0.2548; 1.5313] | [0.8377; 0.9522] | [0.6912; 0.9172] | [0.8901; 0.8934] | pooling_dependent | n/a |
| P1_K5 | false_open_per_frame | 0.1524 | 0.0004 | [-0.0004; 0.0012] | [0.0003; 0.0006] | [0.0004; 0.0011] | [0.0004; 0.0038] | pooling_dependent | n/a |
| P1_K5 | precision | 0.2395 | 0.9979 | [0.9938; 1.0019] | [0.9971; 0.9984] | [0.9945; 0.9979] | [0.9945; 0.9979] | pooling_dependent | n/a |
| P1_K10 | processed_frac | 0.1007 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0997; 0.1030] | pooling_dependent | n/a |
| P1_K10 | frame_recall | 0.1002 | 0.1000 | [0.1000; 0.1000] | [0.1000; 0.1000] | [0.0997; 0.1030] | [0.0997; 0.1030] | pooling_dependent | n/a |
| P1_K10 | event_recall | 0.6878 | 0.8311 | [-0.0027; 1.6650] | [0.7460; 0.9173] | [0.6433; 0.8693] | [0.8284; 0.8317] | pooling_dependent | n/a |
| P1_K10 | false_open_per_frame | 0.0765 | 0.0002 | [-0.0002; 0.0006] | [0.0002; 0.0003] | [0.0002; 0.0006] | [0.0002; 0.0035] | pooling_dependent | n/a |
| P1_K10 | precision | 0.2389 | 0.9979 | [0.9938; 1.0019] | [0.9971; 0.9984] | [0.9944; 0.9979] | [0.9945; 0.9979] | pooling_dependent | n/a |
| P1_K30 | processed_frac | 0.0340 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0332; 0.0366] | pooling_dependent | n/a |
| P1_K30 | frame_recall | 0.0336 | 0.0333 | [0.0333; 0.0333] | [0.0333; 0.0333] | [0.0332; 0.0366] | [0.0332; 0.0366] | pooling_dependent | n/a |
| P1_K30 | event_recall | 0.5390 | 0.7346 | [-0.4108; 1.8799] | [0.6027; 0.8632] | [0.5685; 0.7946] | [0.7321; 0.7354] | pooling_dependent | n/a |
| P1_K30 | false_open_per_frame | 0.0260 | 0.0001 | [-0.0001; 0.0002] | [0.0001; 0.0001] | [0.0001; 0.0002] | [0.0001; 0.0034] | pooling_dependent | n/a |
| P1_K30 | precision | 0.2366 | 0.9979 | [0.9938; 1.0019] | [0.9971; 0.9984] | [0.9942; 0.9979] | [0.9945; 0.9979] | pooling_dependent | n/a |
| P2 | processed_frac | 0.6156 | 0.6147 | [0.3083; 0.9211] | [0.4704; 0.7466] | [0.6129; 0.6160] | [0.6127; 0.6160] | pooling_dependent | n/a |
| P2 | frame_recall | 0.6948 | 0.6151 | [0.3086; 0.9217] | [0.4707; 0.7471] | [0.6131; 0.6164] | [0.6131; 0.6164] | pooling_dependent | n/a |
| P2 | event_recall | 0.9014 | 0.9124 | [0.1586; 1.6662] | [0.8621; 0.9626] | [0.7061; 0.9322] | [0.9094; 0.9127] | pooling_dependent | n/a |
| P2 | false_open_per_frame | 0.0788 | 0.0002 | [-0.0003; 0.0006] | [0.0001; 0.0002] | [0.0002; 0.0018] | [0.0002; 0.0035] | pooling_dependent | n/a |
| P2 | precision | 0.2708 | 0.9985 | [0.9947; 1.0024] | [0.9978; 0.9989] | [0.9931; 0.9985] | [0.9952; 0.9985] | pooling_dependent | n/a |
| P3 | processed_frac | 0.7130 | 0.9989 | [0.9957; 1.0020] | [0.9982; 0.9993] | [0.9957; 0.9989] | [0.9955; 0.9989] | pooling_dependent | n/a |
| P3 | frame_recall | 0.9891 | 1.0000 | [0.9997; 1.0002] | [1.0000; 1.0000] | [0.9966; 1.0000] | [0.9966; 1.0000] | pooling_dependent | n/a |
| P3 | event_recall | 0.9014 | 0.9154 | [0.2096; 1.6213] | [0.8675; 0.9612] | [0.7085; 0.9345] | [0.9124; 0.9157] | pooling_dependent | n/a |
| P3 | false_open_per_frame | 0.0742 | 0.0001 | [-0.0003; 0.0006] | [0.0001; 0.0002] | [0.0001; 0.0018] | [0.0001; 0.0035] | pooling_dependent | n/a |
| P3 | precision | 0.3329 | 0.9990 | [0.9964; 1.0015] | [0.9986; 0.9992] | [0.9957; 0.9990] | [0.9957; 0.9990] | pooling_dependent | n/a |
| P4 | processed_frac | 0.5897 | 0.6070 | [0.3003; 0.9137] | [0.3250; 0.6754] | [0.6052; 0.6083] | [0.6050; 0.6083] | pooling_dependent | n/a |
| P4 | frame_recall | 0.6851 | 0.6072 | [0.2996; 0.9147] | [0.3244; 0.6756] | [0.6051; 0.6085] | [0.6051; 0.6085] | pooling_dependent | n/a |
| P4 | event_recall | 0.9014 | 0.9275 | [0.2397; 1.6153] | [0.8637; 0.9807] | [0.7178; 0.9439] | [0.9244; 0.9277] | pooling_dependent | n/a |
| P4 | false_open_per_frame | 0.0705 | 0.0001 | [-0.0003; 0.0006] | [0.0001; 0.0002] | [0.0001; 0.0018] | [0.0001; 0.0035] | pooling_dependent | n/a |
| P4 | precision | 0.2788 | 0.9981 | [0.9925; 1.0038] | [0.9963; 0.9988] | [0.9927; 0.9981] | [0.9948; 0.9981] | pooling_dependent | n/a |

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

## Bổ sung 2026-10-03 — AMENDMENT 01 (lỗi thứ tự khung LASIESTA; anh Đạt chọn P0 = A)

- Chi tiết, cam kết và dự đoán định tính: `prereg/AMENDMENT_01.md` (+ `.sha256`), commit trước khi chạy lại.
- **Quy tắc báo cáo (ràng buộc):** bộ corrected là kết luận chính; bộ gốc (pre-registered mù, `PREREG_MEVA.json` 209f2e22…) báo song song trong supplement (mục "Amendment 01") và CSV; H nào khác kết luận giữa hai bộ thì bài ghi cả hai, không chọn. Bảng "pre-registered vs thực hiện" thêm cột "orig / corrected".
- Dự đoán bộ corrected: `prereg/PREREG_MEVA_corrected.json` (hash riêng). Kết quả: `results/orig/` (bản gốc, không ghi đè) và `results/corrected/`.

AMENDMENT 01: the original-set MEVA outcomes (H1-H4 FAIL, H3 81/192, H4 AUC 0.739) were seen before these corrected predictions were built.

## 11. SHA-256 đầu vào

- `results/corrected/benchmark_units.parquet` `2fc6f8ed2d5a1ec97e33bbed50407a5b152d045d8cc95e7d11cc407aea927691`
- `results/corrected/operating_points.json` `111f259451728e6029fbe15f0b195441ce8e54ab9955be94a3aa8830a5eaf451`
- `results/corrected/strata_map_benchmark.csv` `88176f2470c8d6c644bc5aed16a2f18818af98f9a8cb098c10a536a261af41da`
- `results/corrected/w_B_strata.csv` `751198fc2e29d4a0315cd571e5e0117c90119ee51a44fc2080f50c643885ddb0`
- `results/corrected/m_B_strata.csv` `949aa186f88b9f018dc8097132697425f3878ca2657125898feb37c5d8a53e17`
- `results/corrected/omega_bprime_S8.csv` `24eb930f56ee606f53df4b84ffd9764869086a59a949212c356bc5777ee9317b`
- `results/corrected/meva_wT_strata.csv` `a90bd44af5a59e1dda69aa13795428816bbc53e1ab4371966e348617fd46ea6c`
- `results/corrected/E1_profile_meva.csv` `ea1fc1542d500b7c902d601f7e986ebdebadb647c8af671197f4759100a8bbdc`
- `results/corrected/E1_profile_benchmark.csv` `86af65f9fd17ce64ee1b437b91632f2d4ebc824ca7b910c33bdd08829d9a8e1b`
- `results/corrected/meva_S8_blocks.csv` `b1bc9b0459143e82180bcaab917fcd1cd239508aebf98b11c46969b5101dd1fa`
- `results/corrected/meva_manifest.csv` `4be0b8e907000e59d62852d496432f90224c90b51aa52833622eee8f02c0bf27`
- `docs/STRATA_SPEC.md` `cd4eb258115460e09ab98057c0da35e6d3992ec40862a3023402252808b9d688`
- `docs/PREREG_DECISIONS.md` `2687d613a19e0b578f2571a25b9908a9d06d4bae054fc8eaba8a801cb1c7a216`
- `docs/MEVA_detector_plan.md` `1101322c0d35c8b653c902045b0f94b49a1816d26295c139be42e7850a3bd4be`
- `code/transport.py` `2edddc95dd3db39ca0f50ef2a6f1c9c794e910ff56a52ec131bf1b539683f38c`
- `code/prereg_build.py` `b1fffbab0d5c43c25ceefa333b07de72f7a48459017b607157ebbb3ec7009c77`
- `code/meva_frames.py` `80481c9182084d316615f31b1a81d0aa6362a922afd54761e19be964edcc118f`
- `code/policies.py` `23c1bead5578838e58c86a5b1ea2668ae7386ef9119326e7aabda967984d1368`
- `code/strata.py` `4a2a563c9c8547f51ef15469370a116d6a916f522c9346ec523c2df82f23a66b`
- `code/flip_radius.py` `35c0c5e3a7f970bc661c3ef70ac74f743b7141bd85e21ac6e2f075713192445c`
- `code/cue.py` `2f47a59783fc927c170c41b914d3d70dbe3ca4030766ba0821d6b3fa9f341263`
- `code/meva_outcomes.py` `59266a0d1b9507837f4f4f856be352f1228ae5596e5bb69765163a5e5d599a68`
- `cache/corrected/meva_windows.parquet` `f816df39bbd39b1e9c6c1138c355d3ce855c71c837f89564b23af9307a6012b8`
- `cache/meva_s8_frames.pkl` `e464153f83660899db917026048fe483bea4aa877d4bddec01d6f9722ef3bb46`
