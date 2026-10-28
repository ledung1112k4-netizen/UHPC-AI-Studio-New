"""Tests for NEW project, not for old-repository patch workflow."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from uhpc_core import (
    DEFAULT_DENSITIES, DEFAULT_FIXED, FEATURES_15, MATERIALS, NOTEBOOK_TEST_MIX,
    DesignRules, batch_table, feature_frame, optimize_mix,
)
from uhpc_engineering import mix_checks, tier_for_target, tighten_bounds, forward_issues
from uhpc_app_logic import ensure_bundle, material_input_issues, stable_signature, properties
from scripts.setup_assets import EXPECTED_BLOB_SHA, git_blob_sha, verify_bytes

class StubModel:
    feature_names_in_ = np.array(FEATURES_15)
    n_features_in_ = 15
    def predict(self, frame):
        assert list(frame.columns) == FEATURES_15
        return np.full(len(frame), 80.0)

@pytest.mark.parametrize('target',range(120,251))
def test_strength_screening_every_mpa(target):
    t=tier_for_target(target)
    assert 400<=t.cement_min<=800
    assert 650<=t.binder_min<=1100
    assert .18<=t.conventional_wb_max<=.30
    if target>120:
        p=tier_for_target(target-1)
        assert t.cement_min>=p.cement_min
        assert t.binder_min>=p.binder_min
        assert t.conventional_wb_max<=p.conventional_wb_max

@pytest.mark.parametrize('target',[120,140,160,180,200,225,250])
def test_no_60_kg_cement(target):
    invalid={**NOTEBOOK_TEST_MIX,'Xi_mang':60.0}
    assert not all(mix_checks(invalid,target).values())
    bounds=tighten_bounds([(0,1500)]*len(MATERIALS),MATERIALS,target)
    assert bounds[0][0]>=400

def test_no_fabricated_model():
    with pytest.raises(ValueError):
        ensure_bundle({'model': StubModel()})
    model, features = ensure_bundle({'model':StubModel(), 'feature_cols':FEATURES_15})
    assert len(features)==15

def test_model_ratio_keeps_quartz_powder():
    m={**NOTEBOOK_TEST_MIX,'Bot_thach_anh':100.0}
    x=feature_frame(pd.DataFrame([m]),28,DEFAULT_FIXED,FEATURES_15)
    assert x.iloc[0]['Ty_le_W_B']==pytest.approx(m['Nuoc']/sum(m[k] for k in MATERIALS[:5]))
    p=properties(m,DEFAULT_DENSITIES)
    assert p['water_binder']>p['water_powder']

def test_zero_filler_is_allowed():
    m={**NOTEBOOK_TEST_MIX,'Silica_fume':0,'Bot_thach_anh':0}
    assert not any('Silica_fume' in x or 'Bot_thach_anh' in x for x in material_input_issues(m))

def test_no_falsely_feasible_inverse():
    b={k:(0.,2000.) for k in FEATURES_15}
    b['Tuoi_duong_ho']=(1.,90.)
    b['Ty_le_W_B']=(.02,.50)
    for k in DEFAULT_FIXED:
        b[k]=(float(DEFAULT_FIXED[k])-1,float(DEFAULT_FIXED[k])+1)
    r=optimize_mix(StubModel(),FEATURES_15,b,28,DEFAULT_FIXED,DEFAULT_DENSITIES,
                   DesignRules(),180,1000,seed=42)
    assert r['feasible'] is False

def test_no_legacy_patch_tools():
    files={p.name for p in ROOT.glob('*.py')}
    assert 'cap_nhat_repo.py' not in files
    assert not (ROOT/'uhpc_limits.py').exists()
    source=(ROOT/'app.py').read_text(encoding='utf-8')
    ast.parse(source)
    assert 'min_value=120.0, max_value=250.0' in source
    assert 'Biên tối thiểu – tối đa của từng vật liệu' not in source
    assert 'number_input("Xi măng tối thiểu' not in source

def test_integrity_checks_reject_wrong_assets():
    assert len(EXPECTED_BLOB_SHA)==2
    for k in EXPECTED_BLOB_SHA:
        with pytest.raises(ValueError):
            verify_bytes(k,b'not-a-real-model')
    assert len(git_blob_sha(b'test'))==40

def test_signature_changes():
    assert stable_signature({'cement':60})!=stable_signature({'cement':600})

def test_batch_sum():
    out,_=batch_table(NOTEBOOK_TEST_MIX,'kg',50)
    assert out['Khối lượng mẻ (kg)'].sum()==pytest.approx(50.)


def test_asset_downloader_downloads_verified_bytes(monkeypatch, tmp_path):
    import io
    from scripts import setup_assets
    name='training_features.csv'
    example=b'Xi_mang,Tro_bay\n650,100\n'
    expected=git_blob_sha(example)
    monkeypatch.setattr(setup_assets, 'ASSETS', tmp_path)
    monkeypatch.setitem(setup_assets.EXPECTED_BLOB_SHA, name, expected)
    monkeypatch.setattr(setup_assets, 'urlopen', lambda request, timeout: io.BytesIO(example))
    assert 'Đã tải' in setup_assets.download_asset(name)
    assert (tmp_path/name).read_bytes()==example
    assert 'Đã có' in setup_assets.download_asset(name)


def test_asset_downloader_refuses_unexpected_bytes(monkeypatch, tmp_path):
    import io
    from scripts import setup_assets
    name='training_features.csv'
    monkeypatch.setattr(setup_assets, 'ASSETS', tmp_path)
    monkeypatch.setattr(setup_assets, 'urlopen', lambda request, timeout: io.BytesIO(b'wrong-file'))
    with pytest.raises(ValueError,match='không khớp'):
        setup_assets.download_asset(name)
    assert not (tmp_path/name).exists()
