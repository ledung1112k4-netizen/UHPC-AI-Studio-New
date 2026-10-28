# HƯỚNG DẪN: TẠO WEBSITE UHPC AI MỚI (KHÔNG THAY APP CŨ)

## A. Chuẩn bị trước

Máy tính Windows cần Python 3.11 hoặc Python 3.12, VS Code, internet và tài khoản GitHub. File ZIP đã có đầy đủ mã nguồn. Mô hình AI **không nằm trong ZIP**, vì vậy phải lấy một lần 2 tệp thật từ repository cũ.

## B. Chạy website trên máy

**1.** Giải nén `UHPC_AI_Studio_Du_An_Moi.zip` vào một thư mục mới, chẳng hạn `D:\UHPC_AI_Studio_New`.

**2.** Mở VS Code → File → Open Folder → chọn thư mục vừa giải nén (thư mục phải chứa `app.py`).

**3.** Trên máy Windows, mở thư mục bằng File Explorer và nhấp đúp `TAI_MO_HINH_WINDOWS.bat`. Chương trình sẽ tải `assets/uhpc_model.pkl` và `assets/training_features.csv` từ repository gốc của bạn. Nếu mạng chặn GitHub, tải hai tệp thủ công theo hướng dẫn ở `assets/README.md`.

**4.** Nhấp đúp `CHAY_WEB_WINDOWS.bat`. Chương trình tạo môi trường Python riêng, cài thư viện theo `requirements.txt`, rồi chạy Streamlit.

**5.** Truy cập `http://localhost:8501`. Khi sidebar ghi “Mô hình AI đang hoạt động”, chạy thử bài toán thuận với cấp phối nhập, sau đó bài toán ngược 120 MPa.

> Cảnh báo: cấp phối trong ứng dụng là kết quả mô hình dự đoán và bộ lọc thiết kế. Không phải kết quả thử nén được công nhận.

## C. Tạo GitHub repository mới (không cần xóa kho cũ)

**6.** Mở `https://github.com/new` → đặt tên `UHPC-AI-Studio-New` → chọn Public hoặc Private tùy quyền chia sẻ mô hình → **Create repository**. Không chọn tạo README khi lập repository mới vì dự án đã có sẵn README.

**7.** Trong VS Code, Terminal → New Terminal (đang ở thư mục dự án). Nhập lần lượt:

```powershell
git init
git add .
git commit -m "New Streamlit UHPC AI Studio"
git branch -M main
git remote add origin https://github.com/ledung1112k4-netizen/UHPC-AI-Studio-New.git
git push -u origin main
```

Nếu báo lỗi lệnh `git` thì cài **Git for Windows**. Nếu chưa biết terminal, có thể vào trang repo GitHub trống → **uploading an existing file** rồi chọn các tệp đã giải nén; lưu ý giữ đúng cấu trúc thư mục `assets/`, `.streamlit/`, `scripts/`, `tests/`.

**8.** Trên GitHub repo MỚI, kiểm tra có `app.py`, `uhpc_core.py`, `uhpc_engineering.py`, `uhpc_app_logic.py`, `requirements.txt`, `.streamlit/config.toml` và **hai tệp trong assets**. Nếu thiếu model AI, chức năng dự đoán trên website Cloud sẽ bị khóa.

## D. Đưa website mới lên Streamlit Cloud

**9.** Vào https://share.streamlit.io → chọn **Create app / Deploy an app** → chọn repository `UHPC-AI-Studio-New`, branch `main`, main file path `app.py`, sau đó **Deploy**.

**10.** Đợi Streamlit triển khai; truy cập liên kết mới để kiểm tra. Website cũ vẫn chạy độc lập và không bị thay thế.

## E. Nếu gặp sự cố

- **`No module named streamlit`**: chạy `py -m pip install -r requirements.txt` trong môi trường thích hợp.
- **`Missing assets/uhpc_model.pkl`**: chạy `python scripts/setup_assets.py` hoặc chép hai tệp từ GitHub cũ.
- **AI báo không có nghiệm ở 200–250 MPa**: dữ liệu hoặc mô hình không hỗ trợ điều kiện đó; không nới giới hạn tự động để tạo kết quả giả.
- **GitHub không cho push**: kiểm tra đã đăng nhập đúng tài khoản và có quyền ghi repository mới; có thể dùng GitHub Desktop.
- **Website Cloud load mã cũ**: kiểm tra đã chọn đúng **repository mới**, branch `main` và `app.py`.

**Bảo mật:** Không chia sẻ mật khẩu, token GitHub, mã OTP. Tệp `.pkl` có khả năng thực thi mã khi mở, chỉ dùng từ nguồn bạn kiểm soát. Nếu mô hình là tài sản riêng, cân nhắc repository Private và quyền triển khai phù hợp.
