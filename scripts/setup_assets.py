"""Tải 2 tài sản AI thực từ commit đã kiểm tra của chính người dùng.

Không tạo dữ liệu huấn luyện giả hoặc dự đoán giả. Chỉ tải và đối chiếu
Git blob SHA-1 với repository đã tham chiếu. Dùng 1 lần để chạy độc lập.

    python scripts/setup_assets.py
"""
from __future__ import annotations
import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
# Ref đã kiểm tra trên repository UHPC-AI-Mix-Design của người dùng
ORIGINAL_REPO = "ledung1112k4-netizen/UHPC-AI-Mix-Design"
SOURCE_COMMIT = "43583a561b0528994bce1836fefa9bf02b8e641e"
SOURCE_URL = f"https://raw.githubusercontent.com/{ORIGINAL_REPO}/{SOURCE_COMMIT}/assets"
# SHA Git blob cho đúng nội dung ở commit này (không phải SHA-256 của file)
EXPECTED_BLOB_SHA = {
    "uhpc_model.pkl": "2787afb6baee3268cdaab60b02ccce67f961e726",
    "training_features.csv": "2f27058b2d16a32f9bfda5ad124e6ec9083dec29",
}


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def verify_bytes(name: str, content: bytes) -> None:
    expected = EXPECTED_BLOB_SHA[name]
    actual = git_blob_sha(content)
    if actual != expected:
        raise ValueError(
            f"{name}: mã SHA Git không khớp bản gốc. "
            "Không dùng file này cho ứng dụng."
        )


def download_asset(name: str, replace: bool = False) -> str:
    if name not in EXPECTED_BLOB_SHA:
        raise ValueError("Tên tài sản không nằm trong danh sách cho phép")
    ASSETS.mkdir(parents=True, exist_ok=True)
    target = ASSETS / name
    if target.exists():
        existing = target.read_bytes()
        if git_blob_sha(existing) == EXPECTED_BLOB_SHA[name]:
            return f"Đã có tệp chính xác: {name}"
        if not replace:
            return (f"Tệp {name} đã tồn tại nhưng khác bản đã xác minh; "
                    "giữ nguyên tệp vì có thể là model riêng. "
                    "Nếu do bạn tự xuất, tự xác nhận nguồn trước khi dùng. "
                    "Dùng --replace chỉ khi thực sự muốn thay thế.")
    request = Request(f"{SOURCE_URL}/{name}", headers={"User-Agent": "UHPC-AI-Studio-asset-setup"})
    try:
        with urlopen(request, timeout=40) as response:
            payload = response.read()
    except (URLError, HTTPError, TimeoutError) as exc:
        raise RuntimeError(f"Không tải được {name}: {exc}") from exc
    verify_bytes(name, payload)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=ASSETS, prefix=".tmp_", delete=False) as fh:
            temp_name = fh.name
            fh.write(payload)
        os.replace(temp_name, target)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink(missing_ok=True)
    return f"Đã tải và kiểm tra: {target} ({len(payload):,} byte)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lấy model và dữ liệu gốc từ GitHub của bạn")
    parser.add_argument("--replace", action="store_true", help="Thay tệp tồn tại nhưng khác bản gốc (cẩn thận)")
    args = parser.parse_args()
    failed = False
    for name in EXPECTED_BLOB_SHA:
        try:
            print(download_asset(name, replace=args.replace))
        except (RuntimeError, ValueError, OSError) as exc:
            failed = True
            print(f"LỖI: {exc}")
    if failed:
        print("Nếu không tải được, tải thủ công từ repo của bạn và chép vào assets/.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
