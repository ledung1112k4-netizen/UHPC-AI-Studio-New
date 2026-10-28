# UHPC AI Studio — DỰ ÁN STREAMLIT MỚI (độc lập)

**Đây là dự án hoàn chỉnh để tạo repository GitHub MỚI.** Không dùng `cap_nhat_repo.py`, không ghi đè dự án cũ và **không phụ thuộc Streamlit app cũ**. Mã nguồn Python, giao diện, tính toán và bộ lọc nằm trong thư mục này. Chỉ cần lấy một lần 2 tài sản AI thật vào `assets/`.

## Có gì trong website?

- **Tổng quan** — tình trạng mô hình AI và dữ liệu, diễn giải cơ sở khoa học.
- **Bài toán thuận** — nhập 9 vật liệu (kg/m³) và tuổi dưỡng hộ, dự đoán bằng mô hình gốc 15 đặc trưng; sàng lọc vật liệu, W/B, thể tích và cảnh báo ngoại suy.
- **Bài toán ngược** — mục tiêu 120–250 MPa, Differential Evolution; cho phép lựa chọn FA/GGBFS, silica fume và bột thạch anh bắt buộc, tùy chọn hoặc 0; nghiệm không đạt không được xuất như cấp phối đã đạt.
- **Phân tích** — feature importance của mô hình (không suy diễn nhân quả), xuất CSV, kiểm tra số mẫu 28 ngày.
- **Hướng dẫn** — phân biệt W/B vật lý và W/tổng bột của notebook.

**Không có các ô min/max từng vật liệu trên giao diện.** Bộ ngưỡng nghiên cứu trong `uhpc_engineering.py` được áp dụng ở cả bài toán thuận và ngược. Tỷ lệ nước/tổng bột của notebook vẫn tính cả bột thạch anh; khác với W/B vật lý. **Ngưỡng không phải TCVN và không bảo đảm cường độ thực nghiệm.**

## Bước 1 — Giải nén

Giải nén ZIP. Mở thư mục `UHPC_AI_Studio_Du_An_Moi` bằng VS Code hoặc File Explorer.

## Bước 2 — Cài Python và tải mô hình AI

Dùng **Python 3.11 hoặc 3.12**. Trên Windows, mở CMD tại thư mục dự án và chạy:

```cmd
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\setup_assets.py
```

Thao tác tải lấy 2 tệp gốc trong repository của bạn, không tạo AI giả. Nếu có sẵn 2 tệp từ repository cũ, bạn chỉ cần chép vào thư mục `assets/` thay cho lệnh tải. *Khi dùng `.pkl` mới từ notebook, phải tương thích 15 đặc trưng.*

## Bước 3 — Chạy website

```cmd
.venv\Scripts\python.exe -m streamlit run app.py
```

Mở http://localhost:8501 trên trình duyệt.

**Đơn giản hơn:** nhấn đúp `CHAY_WEB_WINDOWS.bat` để tự tạo môi trường, cài thư viện, thử tải mô hình (nếu thiếu) và chạy Streamlit.

## Bước 4 — Tạo repository GitHub MỚI

1. Mở https://github.com/new, tạo repository ví dụ `UHPC-AI-Studio-New` (Public). **Để repository trống**, đừng chọn "Add a README" vì đã có README.md trong dự án.
2. Trong VS Code, mở Terminal ở thư mục dự án, chạy:

```bash
git init
git add .
git commit -m "Create new standalone UHPC AI Studio"
git branch -M main
git remote add origin https://github.com/TEN_TAI_KHOAN/UHPC-AI-Studio-New.git
git push -u origin main
```

Thay `TEN_TAI_KHOAN` bằng tên tài khoản GitHub của bạn. **Trước khi `git add .`, đảm bảo đã tải 2 tài sản vào `assets/`, nếu muốn dùng AI trên Streamlit Cloud.** Không commit `.venv` hoặc tệp bí mật.

Nếu chưa dùng Git, GitHub có nút *uploading an existing file* để kéo thả tất cả tệp dự án vào repo mới (kể cả `.streamlit/config.toml` và assets). Đừng tải ZIP vào repo thay vì giải nén.

## Bước 5 — Triển khai trên Streamlit Cloud

1. Truy cập https://share.streamlit.io, đăng nhập bằng GitHub.
2. Chọn **Create app / Deploy an app**.
3. Chọn repository **mới**, nhánh `main` và Main file path = `app.py`.
4. Nhấn Deploy. Streamlit sẽ cài `requirements.txt`, đọc `assets/uhpc_model.pkl` và CSV trong repo mới.
5. Sau khi triển khai, mở website và kiểm tra cảnh báo "Mô hình AI đang hoạt động"; thử bài toán thuận rồi bài toán ngược.

Nếu ứng dụng báo thiếu model, kiểm tra repo mới có `assets/uhpc_model.pkl` và `assets/training_features.csv` chưa. **Không cần mở Colab khi website chạy**, miễn đã có model tương thích.

## Quy tắc an toàn/khoa học

- Các cường độ 120–250 MPa là **mục tiêu nhập**, không phải phạm vi mô hình chắc chắn dự đoán đúng.
- Ngưỡng xi măng nội bộ từ 400 đến 800 kg/m³ theo mục tiêu, cao nhất cho phép 1000 kg/m³; tổng chất kết dính và W/B được lọc. Đây chỉ là chính sách sàng lọc đề tài, phải hiệu chuẩn theo dữ liệu thực nghiệm.
- SF và bột thạch anh có thể 0 nếu trạng thái và dữ liệu huấn luyện hỗ trợ. Không tự nới biên dữ liệu khi không có nghiệm.
- Mô hình đã huấn luyện là do bạn cung cấp; ứng dụng **không tạo cường độ giả** và không thay đổi file `.pkl`.

## Các lệnh kiểm tra

```cmd
.venv\Scripts\python.exe -m pip install pytest
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\kiem_tra_ai.py
```

Bài thử dùng model stub để kiểm tra thuật toán, **không thay thế kiểm chứng trên mô hình và mẫu thí nghiệm thật**.

## Các tệp

```text
UHPC_AI_Studio_Du_An_Moi/
├── app.py
├── uhpc_core.py
├── uhpc_engineering.py
├── uhpc_app_logic.py
├── requirements.txt
├── .streamlit/config.toml
├── .gitignore
├── assets/README.md
├── scripts/setup_assets.py
├── scripts/kiem_tra_ai.py
├── tests/test_standalone.py
├── CHAY_WEB_WINDOWS.bat
├── TAI_MO_HINH_WINDOWS.bat
└── README.md
```
