"""Optional smoke check with YOUR real model; never generates fake strengths.

Run from project root: python scripts/kiem_tra_ai.py
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uhpc_app_logic import MODEL_CANDIDATES, TRAIN_CANDIDATES, ensure_bundle, first_file
from uhpc_core import DEFAULT_FIXED, NOTEBOOK_TEST_MIX, predict

model_file = first_file(ROOT, MODEL_CANDIDATES)
if model_file is None:
    raise SystemExit("Không tìm thấy mô hình thật. Cần assets/uhpc_model.pkl.")
print(f"File model: {model_file.relative_to(ROOT)}")
print("Lưu ý: chỉ mở file pickle do bạn sở hữu/tin cậy.")
with open(model_file, "rb") as f:
    bundle = pickle.load(f)
model, features = ensure_bundle(bundle)
print(f"Tên mô hình: {type(model).__name__}; số đặc trưng: {len(features)}")
value = predict(model, NOTEBOOK_TEST_MIX, 28, DEFAULT_FIXED, features)
print(f"Dự đoán mô hình cho cấp phối kiểm thử của notebook: {value:.3f} MPa")
print("Đây không phải cường độ thí nghiệm được xác nhận; chỉ là kiểm tra hệ thống.")
train_file = first_file(ROOT, TRAIN_CANDIDATES)
print("Dữ liệu train:", train_file.relative_to(ROOT) if train_file else "Chưa có")
