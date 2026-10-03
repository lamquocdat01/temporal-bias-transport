# Chạy S8 trên Kaggle (nhánh K, 28-09-2026; gate hai bậc 29-09-2026) — 3 bước cho anh Đạt

Notebook: `kaggle/s8_detect.ipynb` (sinh lại bằng `python kaggle/make_notebook.py`).
Gói input: `<DATA_ROOT>/_derived\09\kaggle_s8_pkg\` (116,3 MB; tạo lại bằng `python kaggle/build_gate_pkg.py`).
Gate: `docs/GATE_B01_KAGGLE_PREREG.md` **v2** (SHA-256 `de514796…eaebcf1`, commit trước lần chạy đầu).
Cell 5 in thêm số box không ghép cặp mỗi bên (chỉ-tham-chiếu / chỉ-mới) và số box không ghép có conf ≥ 0,6.

1. **Upload gói input thành Kaggle Dataset (private).** kaggle.com → Datasets → *New Dataset* → kéo thả TOÀN BỘ nội dung thư mục
   `kaggle_s8_pkg` (giữ thư mục con `gate\`) → đặt tên `meva-s8-pkg` → Private → Create.
   Gói có weights `yolo26s-seg.pt` và khung CDnet/LASIESTA — để **Private**, không public.
2. **Tạo notebook và cài đặt.** Code → *New Notebook* → File → *Import Notebook* → chọn `s8_detect.ipynb`.
   Cột phải (Settings): **Accelerator = GPU T4** (T4 x1 là đủ), **Internet = On**; *Add Input* → chọn dataset `meva-s8-pkg`.
3. **Chạy.** *Save Version* → *Save & Run All (Commit)*. Phiên chạy nền tối đa 12 h; notebook tự dừng sạch sau 11 h.
   - Cell 5 (gate B0.1) chạy trước và **in kết quả cả hai bậc** (từng tiêu chí, `[x]` = đạt):
     - Bậc 1 PASS → S8 chạy, `gate_tier = "bit"`.
     - Bậc 1 FAIL nhưng bậc 2 PASS → S8 vẫn chạy, `gate_tier = "event"` (em sẽ ghi rõ trong báo cáo).
     - **Cả hai FAIL** → notebook báo lỗi và dừng, không chạy S8. Anh tải `gate_b01.json` + `gate_b01_frames.csv` ở tab
       Output gửi em; ta quay nhánh L.
   - Gate đạt → cell 6 chạy 118 clip: tải từng clip từ S3 vào thư mục tạm **ngoài** `/kaggle/working` (`/kaggle/temp` hoặc
     `/tmp`) → kiểm SHA-256 → detect → ghi JSON → **xoá clip ngay** (khối `finally`, xoá cả khi lỗi). Mỗi dòng log in dung lượng
     `/kaggle/working` và số clip tạm còn lại (phải là 0). Chỉ JSON (~vài trăm MB tổng) nằm trong `/kaggle/working`; nếu vượt
     18 GB notebook dừng sạch (giới hạn Kaggle 20 GB).
   - Xong: tab *Output* → tải `s8_detections_shard0of1.zip`, `gate_b01.json`, `env.json` về
     `<DATA_ROOT>/_derived\09\kaggle_s8_out\` (em sẽ giải nén, kiểm SHA256SUMS, rồi đưa JSON vào `MEVA_20h\detections\`).

## Chạy 1 phiên (mặc định) — khi không mở được 2 phiên GPU song song
Mặc định cell 1 là `SHARD, NSHARD = 0, 1` = một phiên làm cả 118 clip. Không cần sửa gì.
1. Version 1: *Save & Run All (Commit)* như bước 3. Nếu hết 11 h chưa đủ 118 clip, log cuối cell 6 in
   `TIME BUDGET reached`; cell 7 in `shard 0/1: k/118`.
2. Version 2: mở lại notebook → *Add Input* → tab *Your Work* / *Notebook Output Files* → chọn **output của Version 1**
   (vẫn giữ input `meva-s8-pkg`) → *Save & Run All (Commit)*. Cell 6 chép các JSON cũ sang `/kaggle/working/detections`
   rồi bỏ qua clip đã xong; gate chạy lại (~vài phút) và phải đạt lại bậc 2.
3. Lặp tới khi cell 7 in `118/118`. Tải output của **version cuối** (zip chứa đủ JSON của mọi version + SHA256SUMS);
   gửi kèm `gate_b01.json` của **mọi** version (mỗi version có gate riêng).
- Không đổi `SHARD, NSHARD` giữa các version khi chạy 1 phiên. Không chạy 2 notebook cùng lúc trên cùng shard.
- Quota GPU Kaggle theo tuần có hạn (thường ~30 h/tuần); S8 ≈ 1,06 triệu khung — tốc độ thực in ở mỗi dòng log (fps), em sẽ
  ước lượng lại sau version 1.

## Chạy 2 phiên song song (nếu tài khoản cho phép)
Tạo 2 notebook (cùng input). Ở cell 1 sửa dòng `SHARD, NSHARD = …` thành `SHARD, NSHARD = 0, 2` (notebook 1) và
`SHARD, NSHARD = 1, 2` (notebook 2). Resume của từng notebook như trên, gắn output version trước **của chính notebook đó**.
Nếu phiên thứ hai không khởi động được (GPU quota/giới hạn phiên), chuyển sang cách 1 phiên: dùng `0, 1` và gắn output của
notebook `0, 2` làm input — clip shard 0 đã xong được bỏ qua.

Không cần API key hay tài khoản mới; em không tự upload hay chạy gì trên Kaggle.
