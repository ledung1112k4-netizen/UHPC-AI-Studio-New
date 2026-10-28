"""Utility functions for the Streamlit application.

This module NEVER fabricates a compressive strength or modifies the fitted model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uhpc_core import DEFAULT_FIXED, FEATURES_15, LABELS, MATERIALS, physical_quantities
from uhpc_engineering import MATERIAL_LIMITS, mix_checks, forward_issues, tier_for_target

MODEL_CANDIDATES = (
    "assets/uhpc_model.pkl", "assets/uhpc_optimal_lightgbm_model.pkl",
    "uhpc_model.pkl", "uhpc_optimal_lightgbm_model.pkl",
)
TRAIN_CANDIDATES = ("assets/training_features.csv", "training_features.csv")


def first_file(root: Path, candidates: tuple[str, ...]) -> Path | None:
    return next((root / item for item in candidates if (root / item).is_file()), None)


def stable_signature(data: Any) -> str:
    """Fingerprint for hiding stale AI outputs after an input/settings change."""
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ensure_bundle(bundle: object) -> tuple[Any, list[str]]:
    """Only accept original 15-feature notebook bundle; no fallback fake model."""
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError("Mô hình .pkl phải chứa 'model' và 'feature_cols' như notebook gốc.")
    from uhpc_core import check_model_columns
    features = list(bundle["feature_cols"])
    check_model_columns(bundle["model"], features)
    return bundle["model"], features


def material_input_issues(mix: dict[str, float]) -> list[str]:
    """Human-friendly diagnostics without running the AI model."""
    return forward_issues(mix)


def forward_prediction_issues(mix: dict[str, float], prediction: float) -> list[str]:
    """Post-prediction design screening, not a scientific proof of strength."""
    return forward_issues(mix, predicted_mpa=float(prediction))


def properties(mix: dict[str, float], density: dict[str, float]) -> dict[str, float]:
    row = physical_quantities(pd.DataFrame([mix]), density).iloc[0]
    binder = sum(float(mix[k]) for k in ("Xi_mang", "Tro_bay", "Xil_lo_cao", "Silica_fume"))
    total_mass = sum(float(mix[k]) for k in MATERIALS)
    return {
        "binder": binder,
        "water_binder": float(row["W_B"]),
        "water_powder": float(row["W_powder"]),
        "absolute_volume": float(row["absolute_volume"]),
        "total_mass": total_mass,
        "steel_pct": float(row["steel_vol_fraction"]) * 100.0,
    }


def batch_csv(mix: dict[str, float], mode: str, amount: float) -> bytes:
    from uhpc_core import batch_table
    table, _ = batch_table(mix, mode, amount)
    return table.to_csv(index=False).encode("utf-8-sig")


def distinct_train_mixes(train: pd.DataFrame | None, features: list[str]) -> int | None:
    if train is None or not all(k in train.columns for k in features):
        return None
    return int(train.loc[:, features].drop_duplicates().shape[0])


def compatibility_report(training: pd.DataFrame | None, features: list[str], target: float) -> dict[str, Any]:
    """Simple transparency indicators; not a calibrated uncertainty interval."""
    tier = tier_for_target(target)
    out: dict[str, Any] = {
        "target": float(target),
        "cement_filter": tier.cement_min,
        "binder_filter": tier.binder_min,
        "physical_wb_max": tier.conventional_wb_max,
        "training_rows": None if training is None else len(training),
        "training_unique": distinct_train_mixes(training, features),
    }
    return out


def reported_checks(mix: dict[str, float], target: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Kiểm tra": label, "Trạng thái": "Đạt" if ok else "Không đạt"}
         for label, ok in mix_checks(mix, target).items()]
    )


def reset_output_on_change(state: Any, output_key: str, previous_sig_key: str, current_sig: str) -> None:
    if state.get(previous_sig_key) != current_sig:
        state.pop(output_key, None)
        state.pop(previous_sig_key, None)


def model_source_status(model_path: Path | None, train_path: Path | None, load_error: str | None) -> str:
    if model_path is None:
        return "Chưa có mô hình"
    if load_error:
        return "Mô hình chưa nạp được"
    if train_path is None:
        return "AI đã sẵn sàng, thiếu dữ liệu train để tối ưu"
    return "AI và dữ liệu đã sẵn sàng"
