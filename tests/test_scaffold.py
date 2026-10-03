"""Basic scaffolding and imports verification test."""
from pathlib import Path
import yaml
import json

def test_config_files_exist():
    config_dir = Path("config")
    assert (config_dir / "config.yaml").exists()
    assert (config_dir / "tolerances.yaml").exists()
    assert (config_dir / "billet_profiles.yaml").exists()
    assert (config_dir / "calibration.json").exists()

def test_config_content():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["system"]["name"] == "BilletVision"

def test_decision_engine_basic():
    from billetvision.decision.engine import evaluate_tolerances, Verdict
    from billetvision.vision.measure import Measurement
    
    with open("config/tolerances.yaml", "r", encoding="utf-8") as f:
        tols = yaml.safe_load(f)
    
    # Passing billet
    m_pass = Measurement(shape="square", length_mm=1000.0, width_mm=130.0, height_mm=130.0)
    v_pass = evaluate_tolerances(m_pass, tols["square_130"])
    assert v_pass.status == "PASS"
    assert len(v_pass.fail_reasons) == 0

    # Failing billet (width out of range — 131.8 is 1.8 mm over, > 2×tol → FAIL)
    m_fail = Measurement(shape="square", length_mm=1000.0, width_mm=131.8, height_mm=130.0)
    v_fail = evaluate_tolerances(m_fail, tols["square_130"])
    assert v_fail.status in ("FAIL", "REWORK")
    assert len(v_fail.fail_reasons) > 0
    assert "width" in v_fail.fail_reasons[0]
