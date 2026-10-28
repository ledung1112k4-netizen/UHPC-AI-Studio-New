"""UHPC AI Studio — Streamlit full upgrade.

- Uses ONLY the original user-trained 15-feature model from local trusted assets.
- Same notebook-specific W / (cement + FA + GGBFS + SF + quartz) feature.
- All material limits are internal and are NEVER exposed as min/max widgets.
- Strength-dependent constraints are screening assumptions, not strength laws.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from uhpc_core import (
    BINDER_CHOICES, DEFAULT_DENSITIES, DEFAULT_FIXED, FEATURES_15,
    LABELS, MATERIALS, NOTEBOOK_TEST_MIX, POLICY_CHOICES, DesignRules,
    bound_warnings, count_training_matches, feature_frame,
    feature_importance_table, optimize_mix, predict, training_bounds_from_df,
)
from uhpc_engineering import MATERIAL_LIMITS, tier_for_target
from uhpc_app_logic import (
    MODEL_CANDIDATES, TRAIN_CANDIDATES, batch_csv,
    compatibility_report, ensure_bundle, first_file,
    forward_prediction_issues, material_input_issues,
    model_source_status, properties, reported_checks,
    reset_output_on_change, stable_signature,
)

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="UHPC AI Studio | Thiết kế cấp phối bê tông", page_icon="🧪",
    layout="wide", initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
:root{--bg:#081321;--surface:#101f32;--border:#27415d;--fg:#ecf5ff;--muted:#b6c8de;--primary:#2fd0c3;--accent:#83a8ff}
html,body,[data-testid="stApp"], .stApp, [data-testid="stMain"], [data-testid="stMainBlockContainer"]{font-family:'Be Vietnam Pro',system-ui,sans-serif;color:var(--fg)}
.stApp,[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 14% 0%,#103451 0%,#081321 43%,#070f1b 100%)}
[data-testid="stSidebar"]{background:#0c1c30;border-right:1px solid #2c4562}
[data-testid="stSidebar"] *{color:#e4efff}
.block-container{max-width:1500px;padding-top:1.35rem;padding-bottom:3rem}
h1,h2,h3,h4,p,span,label,div{font-family:'Be Vietnam Pro',system-ui,sans-serif}
h1{font-size:clamp(1.9rem,3vw,2.65rem)!important;letter-spacing:-.045em;color:#f4faff!important}
h2{font-size:1.45rem!important;color:#f0f9ff!important}
h3{font-size:1.12rem!important;color:#d8e9ff!important}
small,.stCaption,[data-testid="stCaptionContainer"]{color:#b9cee7!important}
[data-testid="stMarkdownContainer"] p{color:#d1e2f4}
[data-testid="stMarkdownContainer"] strong{color:#f7fbff}
[data-testid="stWidgetLabel"] p{color:#e0ecff!important;font-weight:600}
[data-testid="stMetric"]{border-radius:16px;padding:14px 18px;background:linear-gradient(140deg,#10253c,#132b45);border:1px solid #294660;min-height:111px}
[data-testid="stMetricLabel"]{color:#bad0e5!important}
[data-testid="stMetricValue"]{color:#f1f9ff!important;font-weight:700}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{background:#162a42!important;color:#f7fbff!important;border-color:#355877!important;font-weight:600}
[data-baseweb="select"]>div{background:#172b42!important;color:#fff!important;border-color:#355877!important}
[data-baseweb="select"] span{color:#f4faff!important}
[data-baseweb="popover"],[role="listbox"]{background:#182d45!important;color:#f6f9ff!important}
[data-testid="stForm"], [data-testid="stExpander"]{background:rgba(19,39,61,.73);border:1px solid #2f4a67;border-radius:17px;padding:1.05rem}
button[kind="primary"],button[data-testid="stBaseButton-primaryFormSubmit"]{border:0;background:linear-gradient(95deg,#16a59d,#238dc4)!important;color:white!important;font-weight:700!important;border-radius:9px}
button[kind="primary"]:hover{filter:brightness(1.13)}
button[data-testid="stBaseButton-secondary"]{border:1px solid #426282;background:#15304c;color:#f3f8ff}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:8px;flex-wrap:wrap}
[data-testid="stTabs"] button[role="tab"]{border-radius:9px;background:#11263e;color:#cadaeb;border:1px solid #304c69;font-weight:600;padding:.6rem 1rem}
[data-testid="stTabs"] button[aria-selected="true"]{background:#173f58;color:#58ead8;border-color:#2e9aab}
[data-testid="stDataFrame"]{border:1px solid #2f4b68;border-radius:11px}
[data-testid="stAlert"]{border-radius:10px}
.hero{padding:24px 30px;margin:0 0 15px;border:1px solid #31657d;border-radius:18px;background:linear-gradient(109deg,#173d52 0%,#14324b 57%,#102137 100%);box-shadow:0 10px 25px #040c1780}
.hero .eyebrow{font-size:.76rem;letter-spacing:.17em;color:#7ce6dc;font-weight:700}
.hero h1{margin:.45rem 0 .25rem}
.hero p{color:#c6dcef!important;margin:.5rem 0 0;font-size:.98rem}
.section-note{border-left:3px solid #35c7bc;padding:8px 13px;color:#c5d9ed;background:#112c42;border-radius:7px}
.topbadge{display:inline-block;padding:.35rem .7rem;border:1px solid #337f85;border-radius:24px;color:#75e5db!important;background:#103c49;font-size:.79rem;font-weight:700}
footer{visibility:hidden}
@media(max-width:768px){.hero{padding:18px}.block-container{padding:1rem}.hero h1{font-size:1.7rem!important}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(path: str, mtime: int):
    # WARNING: pickle may execute code; only read self-owned file from trusted repo.
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    model, features = ensure_bundle(bundle)
    return bundle, model, features


@st.cache_data(show_spinner=False)
def load_training(path: str, mtime: int) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


model_path = first_file(ROOT, MODEL_CANDIDATES)
train_path = first_file(ROOT, TRAIN_CANDIDATES)
model, features, bundle, train, bounds = None, FEATURES_15, {}, None, None
load_error = None
train_error = None
if model_path is not None:
    try:
        bundle, model, features = load_model(str(model_path), model_path.stat().st_mtime_ns)
    except Exception as exc:
        load_error = str(exc)
if model is not None:
    try:
        if train_path is not None:
            train = load_training(str(train_path), train_path.stat().st_mtime_ns)
        if isinstance(bundle.get("feature_bounds"), dict):
            bounds = {str(k): tuple(map(float, v)) for k, v in bundle["feature_bounds"].items()}
        elif train is not None:
            bounds = training_bounds_from_df(train, features)
        if bounds is not None and set(bounds) != set(features):
            raise ValueError("Giới hạn dữ liệu train không đủ 15 đặc trưng.")
    except Exception as exc:
        train_error = str(exc)
        bounds = None

st.markdown('''<div class="hero"><span class="eyebrow">UHPC STUDIO • MATERIAL INTELLIGENCE</span>
<h1>Thiết kế cấp phối UHPC bằng AI</h1>
<p>Dự đoán và tối ưu thành phần vật liệu • Tuổi 28 ngày mặc định • Sàng lọc theo miền dữ liệu</p></div>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧪 UHPC AI Studio")
    st.caption("Dự án Streamlit độc lập · Dải mục tiêu 120–250 MPa")
    if model is None:
        st.error("Chưa nạp được mô hình AI")
    else:
        st.success("Mô hình AI đang hoạt động")
    st.caption(model_source_status(model_path, train_path, load_error))
    if load_error:
        st.error(load_error)
    if train_error:
        st.warning(train_error)
    st.divider()
    st.markdown("#### Thuộc tính vật liệu (theo notebook)")
    fixed = {}
    for k, label, value, help_msg in (
        ("FlyAsh_SiO2", "SiO₂ của tro bay (%)", 60.0, "Đặc trưng hóa học của FA trong mô hình"),
        ("FlyAsh_Blaine", "Độ mịn Blaine tro bay", 3500.0, "Đơn vị theo dữ liệu huấn luyện"),
        ("GGBFS_Grade", "Chỉ số GGBFS", 100.0, "Thang chỉ số theo dữ liệu huấn luyện"),
        ("He_so_quy_doi_mau", "Hệ số quy đổi mẫu", 1.0, "Cần thống nhất hình dạng mẫu nén"),
    ):
        fixed[k] = st.number_input(label, value=float(DEFAULT_FIXED.get(k, value)), min_value=0.0, step=0.5 if k == "He_so_quy_doi_mau" else 1.0, help=help_msg, key=f"fixed_{k}")
    st.caption("Giữ đúng định nghĩa đặc trưng như notebook huấn luyện; khi FA hoặc GGBFS = 0, mô hình vẫn nhận thuộc tính vật liệu này.")
    with st.expander("Khối lượng riêng tham khảo", expanded=False):
        st.caption("kg/m³. Chỉnh theo vật liệu thực tế; không phải biên tối ưu.")
        densities = {
            k: st.number_input(LABELS[k], value=float(DEFAULT_DENSITIES[k]), min_value=100.0, max_value=20000.0, step=10.0, key=f"rho_{k}")
            for k in MATERIALS
        }
    st.divider()
    st.caption("AI chỉ đưa ra dự đoán; cần kiểm chứng nén mẫu, tính công tác và khả năng thi công.")

version = stable_signature({"model": str(model_path), "mtime": model_path.stat().st_mtime_ns if model_path else None,
                            "train": str(train_path), "train_mtime": train_path.stat().st_mtime_ns if train_path else None,
                            "fixed": fixed, "densities": densities})


def dataset_warnings(mix: dict[str, float], age: float) -> list[str]:
    if bounds is None:
        return ["Chưa có miền 15 đặc trưng huấn luyện để đánh giá ngoại suy."]
    feats = feature_frame(pd.DataFrame([mix]), age, fixed, features)
    return bound_warnings(feats, bounds)


def show_metrics(mix: dict[str, float], age: float, ai_mpa: float | None = None):
    summary = properties(mix, densities)
    if ai_mpa is not None:
        r = st.columns(4)
        r[0].metric("Cường độ AI dự đoán", f"{ai_mpa:,.1f} MPa")
        r[1].metric("W/B vật lý", f"{summary['water_binder']:.3f}")
        r[2].metric("Tổng chất kết dính", f"{summary['binder']:,.0f} kg/m³")
        r[3].metric("Tuổi dự đoán", f"{age:.0f} ngày")
    else:
        r = st.columns(4)
        r[0].metric("Tổng chất kết dính", f"{summary['binder']:,.0f} kg/m³")
        r[1].metric("W/B vật lý", f"{summary['water_binder']:.3f}")
        r[2].metric("W/Tổng bột của AI", f"{summary['water_powder']:.3f}")
        r[3].metric("Sợi thép", f"{summary['steel_pct']:.2f}% thể tích")
    return summary


def show_calibration_info(ai_mpa: float, target: float | None = None):
    validation = bundle.get("validation", {}) if isinstance(bundle, dict) else {}
    raw = validation.get("calibration_q90_abs_error_mpa") if isinstance(validation, dict) else None
    if isinstance(raw, (int, float)) and np.isfinite(raw) and 0.0 <= raw < 1000.0:
        lower, upper = max(0., ai_mpa - float(raw)), ai_mpa + float(raw)
        st.info(f"Khoảng sai số dự đoán tham khảo (calibration, nếu cùng phân bố): "
                f"**{lower:.1f}–{upper:.1f} MPa**. Đây không phải bảo đảm bao phủ ngoài tập dữ liệu.")
        if target is not None and lower < target:
            st.warning("Cận dưới của khoảng hiệu chuẩn tham khảo thấp hơn mục tiêu. "
                       "Cần thêm mẫu thí nghiệm trước khi kết luận đạt cường độ yêu cầu.")
    else:
        st.caption("Không có dữ liệu calibration độc lập để ước lượng khoảng sai số cường độ.")


def show_similarity_info(result: dict):
    for field, title in (("training_similarity", "Toàn bộ tập train"),
                         ("training_similarity_local", "Nhóm cùng tuổi / tổ hợp vật liệu")):
        info = result.get(field)
        if isinstance(info, dict):
            if info.get("outside_typical_region"):
                st.warning(f"Ứng viên nằm ngoài vùng lân cận thông thường: **{title}**. "
                           "Không coi đây là cấp phối đã kiểm chứng.")
            elif "distance_k" in info and "reference_p95" in info:
                st.caption(f"{title}: khoảng cách chuẩn hóa {info['distance_k']:.3f} "
                           f"(ngưỡng P95 {info['reference_p95']:.3f}); P95 không phải xác suất đạt cường độ.")


def show_mix_table(mix: dict[str, float], prefix: str):
    c1, c2 = st.columns([1, 1])
    with c1:
        batch_mode_label = st.radio("Quy mô mẻ trộn", ["Theo khối lượng (kg)", "Theo thể tích (m³)"], horizontal=True, key=f"batch_mode_{prefix}")
    mode = "kg" if batch_mode_label.startswith("Theo khối") else "m3"
    with c2:
        amount = st.number_input("Khối lượng mẻ (kg)" if mode == "kg" else "Thể tích mẻ (m³)",
                                 min_value=0.001, value=50.0 if mode == "kg" else 0.05,
                                 step=1.0 if mode == "kg" else 0.01, key=f"batch_amount_{prefix}_{mode}")
    from uhpc_core import batch_table
    table, _ = batch_table(mix, mode, float(amount))
    st.dataframe(table, use_container_width=True, hide_index=True,
                 column_config={"Định lượng (kg/m³)": st.column_config.NumberColumn(format="%.2f"),
                                "Khối lượng mẻ (kg)": st.column_config.NumberColumn(format="%.3f"),
                                "Tỷ lệ khối lượng (%)": st.column_config.NumberColumn(format="%.2f%%")})
    st.download_button("⬇️ Tải cấp phối CSV", batch_csv(mix, mode, float(amount)),
                       file_name=f"UHPC_{prefix}.csv", mime="text/csv", key=f"export_{prefix}")


def show_chart(mix: dict[str, float], key: str):
    frame = pd.DataFrame({"Vật liệu": [LABELS[k] for k in MATERIALS],
                          "Khối lượng (kg/m³)": [float(mix[k]) for k in MATERIALS]})
    fig = px.bar(frame, x="Khối lượng (kg/m³)", y="Vật liệu", orientation="h",
                 color="Khối lượng (kg/m³)", color_continuous_scale=[[0, "#236b95"], [1, "#42d9bd"]])
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, margin=dict(t=8, b=15, l=10, r=15), height=365)
    st.plotly_chart(fig, use_container_width=True, key=key)


tab_home, tab_forward, tab_inverse, tab_analysis, tab_notes = st.tabs([
    "⌂ Tổng quan", "🧠 Dự đoán thuận", "⚗️ Tối ưu ngược", "📊 Phân tích", "📖 Hướng dẫn",
])

with tab_home:
    st.subheader("Tổng quan hệ thống")
    a, b, c, d = st.columns(4)
    a.metric("Đặc trưng mô hình", "15")
    b.metric("Nhóm vật liệu", "9")
    c.metric("Dải mục tiêu nhập", "120–250 MPa")
    d.metric("Mẫu train hiện có", "–" if train is None else f"{len(train):,}")
    st.caption("Ứng dụng độc lập, không cần kết nối Colab khi chạy. Mô hình được tải một lần vào assets/.")
    st.markdown("### Trạng thái sẵn sàng")
    if model is not None:
        st.success(f"Mô hình **{type(model).__name__}** đã nạp từ `{model_path.relative_to(ROOT)}`. Cường độ đầu ra do chính mô hình này dự đoán.")
    else:
        st.warning("Chưa có file `assets/uhpc_model.pkl` tương thích. Đưa mô hình `.pkl` của bạn vào thư mục `assets/`; không có chức năng nào tự tạo dự đoán MPa.")
    if train is not None:
        st.info(f"Đã đọc {len(train):,} dòng dữ liệu đặc trưng. Khoảng huấn luyện được dùng để sàng lọc tối ưu và cảnh báo ngoại suy.")
    elif bounds is None:
        st.warning("Chưa có dữ liệu huấn luyện / feature bounds; bài toán ngược được khóa để tránh tìm nghiệm thiếu căn cứ.")
    x, y = st.columns(2)
    with x:
        st.markdown("#### Nghiên cứu cấp phối")
        st.markdown("""- 9 thành phần vật liệu, đơn vị **kg/m³**.
- Tùy chọn **tro bay, GGBFS, silica fume và bột thạch anh**.
- Tự động kiểm tra giới hạn vật liệu và W/B vật lý.
- Chỉ nhận cấp phối ngược khi vượt qua toàn bộ điều kiện.""")
    with y:
        st.markdown("#### Giới hạn khoa học")
        st.markdown("""- Dải mục tiêu 120–250 MPa **không đồng nghĩa** mô hình đã được kiểm chứng tới 250 MPa.
- Bộ ngưỡng khối lượng là **chính sách nghiên cứu**, không phải TCVN.
- Dự đoán cường độ không thay thế thí nghiệm 28 ngày.
- **Bột thạch anh** nằm trong W/tổng bột của AI, nhưng không nằm trong W/B vật lý.""")

with tab_forward:
    st.subheader("Dự đoán cường độ từ cấp phối")
    st.caption("Nhập 9 vật liệu theo kg/m³; hệ thống kiểm tra cấu trúc cấp phối trước/sau khi dự đoán.")
    with st.form("forward_form"):
        mix = {}
        st.markdown("#### Thành phần vật liệu")
        for row in range(3):
            cols = st.columns(3)
            for col, key in zip(cols, MATERIALS[row * 3: row * 3 + 3]):
                with col:
                    mix[key] = st.number_input(LABELS[key] + " (kg/m³)", min_value=0.0, max_value=3000.0,
                                               value=float(NOTEBOOK_TEST_MIX[key]), step=5.0, key=f"forward_{key}")
        age_forward = st.number_input("Tuổi dự đoán (ngày)", 1, 365, 28, 1)
        submit_forward = st.form_submit_button("🧠 Dự đoán bằng mô hình AI", use_container_width=True,
                                                type="primary", disabled=(model is None))
    sig_forward = stable_signature({"mix": mix, "age": age_forward, "env": version})
    reset_output_on_change(st.session_state, "forward_output", "forward_sig", sig_forward)
    if submit_forward:
        st.session_state.pop("forward_output", None)
        try:
            issues = material_input_issues(mix)
            if issues:
                raise ValueError("; ".join(issues))
            yhat = predict(model, mix, float(age_forward), fixed, features)
            checks = forward_prediction_issues(mix, yhat)
            warnings = dataset_warnings(mix, float(age_forward))
            summary = properties(mix, densities)
            if not (0.90 <= summary["absolute_volume"] <= 1.10):
                warnings.append("Tổng thể tích tuyệt đối khác xa 1 m³, cần kiểm tra khối lượng riêng hoặc định lượng vật liệu.")
            st.session_state["forward_output"] = {"mix": mix.copy(), "prediction": yhat,
                                                   "warnings": warnings, "checks": checks,
                                                   "age": int(age_forward)}
            st.session_state["forward_sig"] = sig_forward
        except Exception as exc:
            st.error("Không thể chấp nhận cấp phối đầu vào: " + str(exc))
    payload = st.session_state.get("forward_output")
    if payload:
        st.divider()
        show_metrics(payload["mix"], payload["age"], payload["prediction"])
        if payload["checks"]:
            st.warning("Dự đoán AI được tính nhưng cấp phối **không vượt qua bộ lọc nghiên cứu ở mức cường độ dự đoán**: " + "; ".join(payload["checks"]))
        else:
            st.success("Dự đoán đã được tính từ mô hình AI và cấp phối vượt qua bộ lọc số học nội bộ. Chưa được thí nghiệm xác nhận.")
        show_calibration_info(payload["prediction"])
        for warning in payload["warnings"]:
            st.warning(warning)
        st.markdown("### Bảng thành phần và mẻ trộn")
        show_mix_table(payload["mix"], "thuan")
        with st.expander("Biểu đồ khối lượng cấp phối"):
            show_chart(payload["mix"], "forward_mix_chart")

with tab_inverse:
    st.subheader("Tối ưu cấp phối theo cường độ mục tiêu")
    st.caption("Tối ưu dùng mô hình AI gốc và biên huấn luyện. Nếu không có nghiệm hợp lệ, ứng dụng thông báo rõ và không xuất cấp phối không đạt.")
    with st.form("inverse_form"):
        a, b, c = st.columns(3)
        target = a.number_input("Cường độ yêu cầu (MPa)", min_value=120.0, max_value=250.0, value=120.0, step=5.0)
        age_opt = b.number_input("Tuổi nén (ngày)", min_value=1, max_value=365, value=28, step=1)
        mode = c.selectbox("Mức độ tìm kiếm", ["Nhanh", "Chuyên sâu"], index=0)
        binder_mode = st.selectbox("Tổ hợp chất liên kết", list(BINDER_CHOICES),
                                   format_func=lambda k: BINDER_CHOICES[k], index=0)
        st.markdown("#### Trạng thái vật liệu")
        cols = st.columns(4)
        fa_state = cols[0].selectbox("Tro bay", list(POLICY_CHOICES), format_func=lambda k: POLICY_CHOICES[k], index=0,
                                      disabled=binder_mode != "CUSTOM")
        gg_state = cols[1].selectbox("GGBFS", list(POLICY_CHOICES), format_func=lambda k: POLICY_CHOICES[k], index=0,
                                      disabled=binder_mode != "CUSTOM")
        sf_state = cols[2].selectbox("Silica fume", list(POLICY_CHOICES), format_func=lambda k: POLICY_CHOICES[k], index=1)
        q_state = cols[3].selectbox("Bột thạch anh", list(POLICY_CHOICES), format_func=lambda k: POLICY_CHOICES[k], index=1)
        st.caption("Tùy chọn = AI tìm trong phạm vi huấn luyện. Không sử dụng = bắt buộc 0 kg/m³. Nếu không có dữ liệu hỗ trợ, báo không khả thi.")
        has_similarity = "similarity_policy" in getattr(DesignRules, "__dataclass_fields__", {})
        safety_mode = None
        if has_similarity:
            safety_mode = st.radio("Kiểm soát ngoại suy",
                                   ["Ưu tiên gần dữ liệu huấn luyện", "Khám phá có cảnh báo ngoại suy"],
                                   horizontal=True, index=0)
        submit_opt = st.form_submit_button("⚗️ Tìm cấp phối bằng AI", type="primary", use_container_width=True,
                                           disabled=(model is None or bounds is None))
    sig_inverse = stable_signature({"target": target, "age": age_opt, "mode": mode, "binder": binder_mode,
                                    "fa": fa_state, "gg": gg_state, "sf": sf_state, "qp": q_state,
                                    "safety": safety_mode, "env": version})
    reset_output_on_change(st.session_state, "inverse_output", "inverse_sig", sig_inverse)
    if submit_opt:
        st.session_state.pop("inverse_output", None)
        try:
            rule_settings = dict(wb_min=0.14, wb_max=0.30, air_fraction=0.02,
                                 volume_tolerance=0.02, binder_mode=binder_mode,
                                 custom_fa=fa_state, custom_ggbfs=gg_state,
                                 silica_mode=sf_state, quartz_mode=q_state,
                                 min_flyash=20.0, min_ggbfs=20.0,
                                 min_silica=20.0, min_quartz=20.0,
                                 max_steel_vol_percent=2.0, steel_saving_weight=1.0)
            if has_similarity:
                rule_settings["similarity_policy"] = (
                    "strict" if safety_mode.startswith("Ưu tiên") else "report_only")
            if "prefer_near_target" in getattr(DesignRules, "__dataclass_fields__", {}):
                rule_settings["prefer_near_target"] = True
            rules = DesignRules(**rule_settings)
            rules.validate()
            if train is not None:
                matches = count_training_matches(train, rules, densities["Soi_thep"])
                if matches <= 0:
                    st.warning("Không có dòng train cùng điều kiện vật liệu được chọn. Kết quả sẽ không được khẳng định đáng tin cậy.")
            with st.spinner("AI đang tìm nghiệm và kiểm tra thể tích, W/B, biên huấn luyện..."):
                result = optimize_mix(model, features, bounds, float(age_opt), fixed, densities, rules,
                                      target_mpa=float(target), max_cement=1000.0,
                                      mode=mode, training=train)
            st.session_state["inverse_output"] = result
            st.session_state["inverse_sig"] = sig_inverse
        except Exception as exc:
            st.error("Không tìm được cấp phối hợp lệ: " + str(exc))
    outcome = st.session_state.get("inverse_output")
    if outcome:
        st.divider()
        a, b, c = st.columns(3)
        a.metric("Mục tiêu", f"{outcome['target']:.1f} MPa")
        b.metric("AI dự đoán cho ứng viên", f"{outcome['prediction']:.1f} MPa")
        c.metric("Ứng viên hợp lệ quan sát", f"{outcome.get('feasible_candidates_seen', 0):,}")
        if outcome.get("feasible"):
            st.success("Đã tìm thấy **ứng viên vượt qua ràng buộc toán học và bộ lọc kỹ thuật**. Đây KHÔNG phải cấp phối đã chứng minh đạt cường độ thực nghiệm.")
            show_calibration_info(outcome["prediction"], outcome["target"])
            show_similarity_info(outcome)
            show_metrics(outcome["mix"], outcome["age"])
            st.markdown("### Bảng cấp phối đề xuất")
            show_mix_table(outcome["mix"], "nguoc")
            show_chart(outcome["mix"], "inverse_mix_chart")
        else:
            st.error("Chưa tìm được nghiệm thỏa đồng thời cường độ dự đoán, W/B, thể tích và giới hạn vật liệu. Không xuất ứng viên như cấp phối đạt yêu cầu.")
            show_similarity_info(outcome)
        with st.expander("Chi tiết đánh giá điều kiện", expanded=False):
            st.dataframe(pd.DataFrame([{"Điều kiện": k, "Đạt": "✓" if v else "✗"}
                                       for k, v in outcome.get("checks", {}).items()]),
                         hide_index=True, use_container_width=True)
            if outcome.get("training_support_count") is not None:
                st.caption(f"Số dòng train cùng tổ hợp vật liệu: {outcome['training_support_count']}. Không phải số thí nghiệm xác nhận cấp phối mới.")
            for warning in outcome.get("warnings", []):
                st.warning(warning)

with tab_analysis:
    st.subheader("Phân tích vật liệu và mô hình")
    if model is None:
        st.info("Cần nạp mô hình AI để hiển thị importance từ mô hình thật.")
    else:
        try:
            importance, method = feature_importance_table(model, features)
            subset = importance.loc[importance["Nhóm"].eq("Vật liệu")].copy()
            st.caption("Feature Importance phản ánh mức độ sử dụng đặc trưng trong mô hình cây; KHÔNG có nghĩa là quan hệ nhân quả hay hướng tăng/giảm cường độ.")
            chart = px.bar(subset.sort_values("Điểm gốc"), x="Tỷ trọng trong 9 vật liệu (%)", y="Tên hiển thị",
                           orientation="h", color="Tỷ trọng trong 9 vật liệu (%)",
                           color_continuous_scale=[[0, "#245788"], [1, "#42d9bd"]])
            chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                coloraxis_showscale=False, height=410, margin=dict(t=5, l=5, r=5, b=5))
            st.plotly_chart(chart, use_container_width=True, key="importance_chart")
            st.caption(f"Chỉ số lấy trực tiếp từ mô hình: {method}.")
            st.dataframe(importance[["Tên hiển thị", "Nhóm", "Điểm gốc", "Tỷ trọng toàn mô hình (%)"]],
                         use_container_width=True, hide_index=True)
            st.download_button("⬇️ Xuất thống kê importance", importance.to_csv(index=False).encode("utf-8-sig"),
                               file_name="uhpc_feature_importance.csv", mime="text/csv")
        except Exception as exc:
            st.warning("Không lấy được feature importance của mô hình: " + str(exc))
    st.markdown("### Minh bạch về dữ liệu và giới hạn nghiên cứu")
    if train is not None:
        st.write(f"Đã đọc {len(train):,} dòng dữ liệu train từ `{train_path.relative_to(ROOT)}`.")
        if "Tuoi_duong_ho" in train:
            ages = pd.to_numeric(train["Tuoi_duong_ho"], errors="coerce").dropna()
            if len(ages):
                st.metric("Số mẫu có tuổi 28 ngày trong tệp", f"{int(np.isclose(ages, 28).sum()):,}")
    else:
        st.info("Thêm assets/training_features.csv để kiểm tra khoảng train và phân bố tuổi mẫu.")
    st.caption("Cấp phối có thể hợp lệ theo giả định nhưng không đảm bảo độ chảy, phân tán sợi, cường độ thực nghiệm, hoặc khả năng ngoại suy từ tập dữ liệu.")

with tab_notes:
    st.subheader("Hướng dẫn sử dụng & lưu ý khoa học")
    st.markdown("""**Bài toán thuận:** Nhập định lượng 9 thành phần theo kg/m³ và tuổi nén. Nhấn *Dự đoán bằng mô hình AI* để xem cường độ dự đoán, W/B, bảng định lượng và cảnh báo.

**Bài toán ngược:** Nhập cường độ mục tiêu từ 120 đến 250 MPa, chọn tuổi, hệ chất liên kết và trạng thái silica fume/bột thạch anh. Thuật toán tìm cấp phối thỏa các điều kiện số học, hoặc báo không có nghiệm.

**Các giới hạn min/max của từng vật liệu không hiển thị trên giao diện**. Chúng được cấu hình tại `uhpc_engineering.py`. Các ngưỡng thay đổi theo cường độ là **giả định của đề tài**, không phải giá trị quy định bởi TCVN.

**Khác biệt hai tỷ lệ nước:** `Ty_le_W_B` trong mô hình notebook có tính bột thạch anh trong mẫu số. `W/B vật lý` không tính bột thạch anh. Không thay đổi định nghĩa đặc trưng đầu vào sau huấn luyện.

**Mô hình và kết quả:** Mô hình đã được huấn luyện bởi người dùng, không tạo lại hay tự điền số liệu MPa. Trước khi áp dụng thực tế, cần đối chứng với dữ liệu mới, đánh giá MAE/RMSE và thí nghiệm UHPC tại 28 ngày.""")
    st.caption("Nghiên cứu vật liệu • Dashboard chỉ hỗ trợ phân tích và sàng lọc cấp phối.")
