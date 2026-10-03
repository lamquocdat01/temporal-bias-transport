# ENV — môi trường tính toán (đo 2026-09-25)

| Mục | Giá trị |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| CPU | 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz; cores 4; logical 8 |
| RAM | 31.7 GB |
| GPU (WMI) | Intel(R) Iris(R) Xe Graphics |
| nvidia-smi | **không có** (lệnh không tồn tại) → không có GPU NVIDIA/CUDA dùng được |
| Python | 3.11.9 |
| Gói | pandas 2.3.3; pyarrow 22.0.0; opencv 4.10.0; ultralytics 8.4.7; torch 2.2.0+cpu (CUDA available: False); boto3 1.43.102; numpy 1.26.4; matplotlib 3.10.8 |
| git | git version 2.52.0.windows.1 |
| Ổ D: trống | 66G (prompt ghi ~71 GB — thực đo thấp hơn) |

Ghi chú
- boto3 được cài mới trong CP0 (`pip install boto3`). pyarrow đã có sẵn.
- torch là bản CPU → detector chỉ chạy CPU (xem COMPUTE_BUDGET.md).
- Kaplan–Meier tự cài đặt trong `code/e1_profile.py` (không phụ thuộc lifelines).
