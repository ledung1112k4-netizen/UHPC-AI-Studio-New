# Thư mục chứa tài sản AI GỐC

Bạn cần hai tệp để chạy đầy đủ ứng dụng:

- `uhpc_model.pkl` — mô hình LightGBM 15 đặc trưng đã huấn luyện, ở dạng bundle `model` và `feature_cols`.
- `training_features.csv` — 15 đặc trưng huấn luyện (dùng làm miền tìm kiếm và kiểm tra vùng dữ liệu).

Tải tự động bằng `python scripts/setup_assets.py`, hoặc chạy `TAI_MO_HINH_WINDOWS.bat`.
Script lấy trực tiếp từ **commit cố định của repository do bạn sở hữu**, kiểm tra Git blob SHA, không tự huấn luyện hoặc sinh dữ liệu giả.

Nếu tải thủ công, vào repository cũ trên GitHub → `assets` → tải 2 tệp này → chép vào thư mục `assets/` ở đây.

Ứng dụng vẫn mở giao diện khi chưa có model, nhưng tính năng dự đoán/tối ưu sẽ khóa. Chỉ mở `.pkl` của chính bạn hoặc nguồn bạn tin cậy vì `pickle` có thể thực thi mã.
